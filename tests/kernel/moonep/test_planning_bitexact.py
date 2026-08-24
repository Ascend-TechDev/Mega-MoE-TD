# coding=utf-8
"""L1：moonep Triton planning vs vendor oracle 全链 bit-exact 对拍（2 rank）。

同输入（固定 seed 生成，显式喂两侧）跑：
- NPU 侧：MoonepPlanBuffers + launch_moonep_planning（v2 三 kernel：gather
  尾 barrier 的 tpe allgather + B 表冗余自算 + C.1/C.2/src_info/dedup）
- oracle 侧：mega_moe.moonep_ref 的 SimTransport + PlanningKernel 编排（CPU）

逐位比较六张表：dst / cu_seqlens / experts_to_copy / zero_fill_ranges /
remote_stats / src_info；并跑 planning_invariant_errors。
"""

import pytest
import torch
import torch.distributed as dist

import mega_moe.moonep_ref  # noqa: F401  （sys.path bootstrap：worker 内 `import moonep` 可达）
from mega_moe.kernels.moonep_planning import (
    MoonepPlanBuffers,
    launch_moonep_planning,
)


def _make_inputs(S, K, E, R, style, seed=42):
    """确定性生成 topk/tpe（两侧喂同一份；tpe 由 topk 直方图导出）。"""
    g = torch.Generator().manual_seed(seed)
    topks, tpes = [], []
    for r in range(R):
        g.manual_seed(seed + r * 1000)     # rank 各异但可复现
        if style == "rand":
            topk = torch.randint(0, E, (S * K,), generator=g,
                                 dtype=torch.int32)
        elif style == "dup":
            topk = torch.randint(0, E, (S, K), generator=g, dtype=torch.int32)
            topk[: S // 4, 1] = topk[: S // 4, 0]   # 1/4 token 重复选专家
            topk = topk.reshape(-1)
        else:
            raise ValueError(style)
        tpe = torch.bincount(topk.to(torch.int64), minlength=E) \
            .to(torch.int32)
        topks.append(topk)
        tpes.append(tpe)
    return topks, tpes


def _run_case(rank, world_size, style):
    import tests._moe_testkit as kit
    from moonep.planning import allocate_planning_outputs, launch_planning
    from moonep.tests.kernel_test_utils import (
        KernelCase,
        build_world,
        planning_invariant_errors,
        run_ranks,
    )

    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    # R==1 时 oracle api 要求 3*E*R 被 4 整除 → epn=4；其余 epn=2
    S, K, epn, H, B, tp = 64, 2, (4 if world_size == 1 else 2), 16, 1, 2
    E = world_size * epn
    N, NvS = S * K, S * K + (tp - 1) * 2 * epn
    case = KernelCase(f"l1_{style}", S=S, K=K, epn=epn, H=H, num_sms=1,
                      B=B, token_padding=tp, R=world_size, seed=42)
    topks, tpes = _make_inputs(S, K, E, world_size, style)

    # ---- oracle 全链（CPU SimTransport）----
    sim, _arenas, obufs = build_world(case)
    oracle_outs = [None] * world_size

    def _f(r):
        ctx = obufs[r]._require_ctx()
        plan, cu = allocate_planning_outputs(ctx)
        launch_planning(ctx, topks[r].contiguous(), tpes[r], cu, plan)
        oracle_outs[r] = (plan, cu, ctx)

    run_ranks(sim, _f, "oracle")

    # ---- NPU 侧 ----
    with kit.aclshmem_session(rank, world_size, 256 * 1024 * 1024):
        pbufs = MoonepPlanBuffers(world_size, E, B, N, NvS, device)
        outs = {
            "dst": torch.empty(N, dtype=torch.int32, device=device),
            "cu_seqlens": torch.empty(E + B, dtype=torch.int32, device=device),
            "experts_to_copy": torch.empty(world_size, B, dtype=torch.int32,
                                           device=device),
            "zero_fill_ranges": torch.empty(E + B, 2, dtype=torch.int32,
                                            device=device),
            "remote_stats": torch.empty(2, dtype=torch.int32, device=device),
            "src_info": torch.empty(NvS, dtype=torch.int32, device=device),
        }
        try:
            launch_moonep_planning(
                pbufs, outs, topks[rank].to(device),
                rank=rank, world_size=world_size, ep_group=ep_group,
                S=S, K=K, E=E, B=B, NvS=NvS, token_padding=tp,
            )
            plan_o, cu_o, ctx_o = oracle_outs[rank]
            # 显式逐表比对（cu_seqlens 在 oracle 侧独立返回）
            assert torch.equal(outs["dst"].cpu(), plan_o.dst), "dst 不一致"
            assert torch.equal(outs["cu_seqlens"].cpu(), cu_o), \
                "cu_seqlens 不一致"
            assert torch.equal(outs["experts_to_copy"].cpu(),
                               plan_o.experts_to_copy), "etc 不一致"
            assert torch.equal(outs["zero_fill_ranges"].cpu(),
                               plan_o.zero_fill_ranges), "zfr 不一致"
            assert torch.equal(outs["remote_stats"].cpu(),
                               plan_o.remote_stats), "stats 不一致"
            assert torch.equal(outs["src_info"].cpu(), plan_o.src_info), \
                "src_info 不一致"
            errors = planning_invariant_errors(
                case, ctx_o, outs["dst"].cpu(), outs["cu_seqlens"].cpu(),
                outs["experts_to_copy"].cpu())
            assert not errors, f"invariants: {errors[:3]}"
        finally:
            pbufs.finalize()


def _worker(rank, world_size, style):
    _run_case(rank, world_size, style)


@pytest.mark.dist
@pytest.mark.npu
@pytest.mark.parametrize("style", ["rand", "dup"])
def test_planning_bitexact_r2(dist_test, style):
    import functools

    dist_test(functools.partial(_worker, style=style), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
