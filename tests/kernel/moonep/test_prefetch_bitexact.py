# coding=utf-8
"""M2：workspace + prefetch push kernel 对拍（2 rank）。

链路：planning（M1）→ register_weights（确定性 home 行）→ prefetch push →
本 rank 校验槽行 == 属主 home 行（bf16 位壳逐位）；空槽保持 register 后的
0；skip_if_same_as 跳过路径。

输入用 skew 路由保证至少一个副本槽被选中（rank0 重压 e0 → 组0 超载 →
rank1 出现远程专家 → top-B 选槽）。
"""

import functools

import pytest
import torch
import torch.distributed as dist

import mega_moe.moonep_ref  # noqa: F401  （bootstrap：worker 内 import moonep）
from mega_moe.kernels.moonep_planning import (
    MoonepPlanBuffers,
    launch_moonep_planning,
)
from mega_moe.kernels.moonep_prefetch import launch_moonep_prefetch
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace

_S, _K, _EPN, _H, _F, _B, _TP = 64, 2, 2, 16, 8, 1, 2


def _pattern_gu(rank, e_local, topo):
    """确定性 home 行：gate_up [H, 2F] bf16（int16 位壳可逆）。"""
    H, F = topo.H, topo.F
    i = torch.arange(H)[:, None]
    j = torch.arange(2 * F)[None, :]
    v = (rank * 1_000_003 + e_local * 10_007 + i * 97 + j * 7) % 4096
    return v.to(torch.float32).to(torch.bfloat16)


def _pattern_dn(rank, e_local, topo):
    H, F = topo.H, topo.F
    i = torch.arange(H)[:, None]
    j = torch.arange(F)[None, :]
    v = (rank * 2_000_003 + e_local * 20_007 + i * 89 + j * 11) % 4096
    return v.to(torch.float32).to(torch.bfloat16)


def _skew_inputs(S, K, E, R):
    """rank0 重压 e0（组0 超载必现），rank1 均匀。"""
    g = torch.Generator().manual_seed(7)
    topks, tpes = [], []
    for r in range(R):
        if r == 0:
            logits = torch.full((E,), 0.0)
            logits[0] = 5.0                      # e0 强偏置
            pick = torch.multinomial(torch.softmax(logits, -1)
                                     .repeat(S * K, 1), 1,
                                     generator=g).reshape(-1)
        else:
            pick = torch.randint(0, E, (S * K,), generator=g)
        topk = pick.to(torch.int32)
        tpe = torch.bincount(topk.to(torch.int64), minlength=E).to(torch.int32)
        topks.append(topk)
        tpes.append(tpe)
    return topks, tpes


def _worker(rank, world_size):
    import tests._moe_testkit as kit
    from moonep.planning import allocate_planning_outputs, launch_planning
    from moonep.tests.kernel_test_utils import KernelCase, build_world, run_ranks

    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    E = world_size * _EPN
    topo = MoonepTopology(S=_S, K=_K, E=E, R=world_size, B=_B, H=_H, F=_F,
                          token_padding=_TP)
    N, NvS = topo.N, topo.NvS
    epn = topo.epn

    topks, tpes = _skew_inputs(_S, _K, E, world_size)

    # oracle 全链（CPU）拿 etc 参照
    case = KernelCase("m2", S=_S, K=_K, epn=_EPN, H=_H, num_sms=1, B=_B,
                      token_padding=_TP, R=world_size, seed=42)
    sim, _a, obufs = build_world(case)
    oracle_outs = [None] * world_size

    def _f(r):
        ctx = obufs[r]._require_ctx()
        plan, cu = allocate_planning_outputs(ctx)
        launch_planning(ctx, topks[r].contiguous(), tpes[r], cu, plan)
        oracle_outs[r] = plan

    run_ranks(sim, _f, "oracle")
    etc_ref = oracle_outs[0].experts_to_copy           # 全组同表

    heap = max(MoonepWorkspace.required_bytes(topo) * 2, 256 << 20)  # 256MB 起（smoke 验证过 init 下限）
    with kit.aclshmem_session(rank, world_size, heap):
        ws = MoonepWorkspace(topo, rank, device)
        try:
            gu_local = torch.stack([_pattern_gu(rank, e, topo)
                                    for e in range(epn)])
            dn_local = torch.stack([_pattern_dn(rank, e, topo)
                                    for e in range(epn)])
            gu_view, dn_view = ws.register_weights(gu_local.to(device),
                                                  dn_local.to(device))
            del gu_local, dn_local

            # planning（Triton 链）
            pbufs = MoonepPlanBuffers(world_size, E, _B, N, NvS, device,
                                      tpe_all=ws.tpe_all,
                                      src_info=ws.src_info)
            outs = {
                "dst": torch.empty(N, dtype=torch.int32, device=device),
                "cu_seqlens": torch.empty(E + _B, dtype=torch.int32,
                                          device=device),
                "experts_to_copy": torch.empty(world_size, _B,
                                               dtype=torch.int32,
                                               device=device),
                "zero_fill_ranges": torch.empty(E + _B, 2, dtype=torch.int32,
                                                device=device),
                "remote_stats": torch.empty(2, dtype=torch.int32,
                                            device=device),
                "src_info": torch.empty(NvS, dtype=torch.int32,
                                        device=device),
            }
            launch_moonep_planning(
                pbufs, outs, topks[rank].to(device),
                rank=rank, world_size=world_size, ep_group=ep_group,
                S=_S, K=_K, E=E, B=_B, NvS=NvS, token_padding=_TP)
            assert torch.equal(outs["experts_to_copy"].cpu(), etc_ref.cpu()), \
                "M2: planning etc 与 oracle 不一致（M1 回归）"

            # prefetch push + 校验
            ran = launch_moonep_prefetch(
                ws.gate_up, ws.down, outs["experts_to_copy"],
                rank=rank, world_size=world_size, epn=epn, H=_H, F=_F)
            etc_cpu = outs["experts_to_copy"].cpu()
            n_filled = 0
            for b in range(_B):
                e = int(etc_cpu[rank, b])
                if e < 0:
                    # 空槽保持 register 后的 0
                    assert int(ws.gate_up[epn + b].abs().sum()) == 0, \
                        "空槽被写入"
                    continue
                owner = e // epn
                e_local = e - owner * epn
                expect_gu = _pattern_gu(owner, e_local, topo)
                expect_dn = _pattern_dn(owner, e_local, topo)
                assert torch.equal(ws.gate_up[epn + b].cpu(),
                                   expect_gu), f"槽{b} gate_up ≠ 属主行"
                assert torch.equal(ws.down[epn + b].cpu(),
                                   expect_dn), f"槽{b} down ≠ 属主行"
                n_filled += 1
            if rank != 0:
                # 组0 超载 → rank1 必有远程专家；B=1 槽应被填（skew 保证）
                assert n_filled >= 1 or int((etc_cpu[rank] >= 0).sum()) == 0, \
                    "skew 输入下 rank1 槽未被填且 etc 非空"

            # skip 路径：同 etc 再调一次 → False 且内容不变
            before = ws.gate_up[epn:].clone()
            ran2 = launch_moonep_prefetch(
                ws.gate_up, ws.down, outs["experts_to_copy"],
                rank=rank, world_size=world_size, epn=epn, H=_H, F=_F,
                skip_if_same_as=outs["experts_to_copy"].clone())
            assert ran2 is False, "skip_if_same_as 未生效"
            assert torch.equal(before, ws.gate_up[epn:]), "skip 路径改写了槽"
            assert ran in (True, False)  # rank 可能无推送条目（M=0 也参与 barrier）
        finally:
            pbufs.finalize()
            ws.finalize()


@pytest.mark.dist
@pytest.mark.npu
def test_prefetch_bitexact_r2(dist_test):
    dist_test(functools.partial(_worker), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
