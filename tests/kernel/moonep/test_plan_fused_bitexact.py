# coding=utf-8
"""L1：moonep_plan_fused 单 kernel planning vs vendor oracle 逐位对拍。

覆盖 v2 融合 kernel 的关键路径（全 rank 同构：Phase A kernel 内 tpe
allgather → B 七表冗余自算 → C.1/C.2 + src_info symm_at 发布 → D dedup）：

- golden：oracle 标准手算例（R=2,E=4,B=1,tp=2）七表双重锁（硬编码常数 +
  oracle 调用）+ 全链六 outs + 单 kernel 融合版（moonep_plan_fused，
  E4/R2 锚点）对拍；
- ties4：R=4 构造用例——B.1 盈余/亏空双平局（取小）、B.2 配额/专家平局；
- b3tie：R=4,E=16 等量切分——B.3 top-B 平局（取大）+ 空槽 -1；
- odd：奇数 count——B.4 padding（zfr 非零）；
- determinism：同输入两次规划逐位一致；
- raw：同 kernel 内全局 store→load 回读（融合方案的前提原语，单卡）。

R==1 不支持（bisheng 形状 bug，CASE-19；实际使用无此形态）。
"""

import functools

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl


def npow2(x):
    return triton.next_power_of_2(x)

import mega_moe.moonep_ref  # noqa: F401  （sys.path bootstrap：worker 内 `import moonep` 可达）
from mega_moe.kernels.moonep_planning import (
    MoonepPlanBuffers,
    launch_moonep_planning,
)

_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 用例定义与确定性输入构造
# ---------------------------------------------------------------------------
_CASES = {
    # oracle 标准手算例（test_oracle_selftest._STD 同款 tpe_all）
    "golden": dict(S=4, K=2, epn=2, B=1, tp=2,
                   counts=[[2, 2, 2, 2], [6, 0, 2, 0]]),
    # R=4,E=8：col=[16,16,8,8,4,4,4,4] → gtok=[32,16,8,8]、cap=16 →
    # bal=[16,0,-8,-8]：B.1 d 双平局取小（2/3 间取 2）× 两轮 →
    # z[0]=[0,0,8,8] 配额平局 → B.2 d 取小（选 2）；rem=[16,16] 等量 →
    # le 平局取小（选 0）
    "ties4": dict(S=8, K=2, epn=2, B=1, tp=2,
                  cols=[16, 16, 8, 8, 4, 4, 4, 4]),
    # R=4,E=16,epn=4：home0 col=[5,5,5,1]（gtok0=16）、home1=2（亏空
    # -14）、home2/3 各 15 → z[0][1]=12，B.2 home0 rem=[5,5,5,1]、quota=12：
    # 两刀各 5（le 平局取小两次）+ 保留 2 → alloc[1][0]==alloc[1][1]==5 →
    # B.3 平局取大（etc[1][0]=1）；B=3 覆盖第 2、3 轮与空槽 -1
    "b3tie": dict(S=6, K=2, epn=4, B=3, tp=2,
                  cols=[5, 5, 5, 1, 1, 1, 0, 0, 4, 4, 4, 3, 4, 4, 4, 3]),
    # 奇数 count：zfr padding 非零（padded=ceil(cnt/2)*2 > cnt）
    "odd": dict(S=3, K=1, epn=2, B=1, tp=2, cols=[3, 1, 1, 1]),
    # R==1 不支持（bisheng 形状 bug，CASE-19；实际使用无此形态），无对应用例
}

_GOLDEN_TABLES = {
    "tpec": [[2, 2, 2, 2], [8, 2, 4, 2]],
    "alloc_cs": [[6, 8], [2, 2], [0, 4], [0, 2]],
    "eoff": [[0, 6, 0, 0], [6, 0, 0, 4]],
    "cu": [[6, 8, 8, 8, 8], [0, 0, 4, 6, 8]],
    "zfr": [[[0, 0]] * 5 for _ in range(2)],
    "etc": [[-1], [0]],
    "stats": [[0, 1], [1, 0]],
}


def _build_topks(case, world_size):
    """确定性构造各 rank topk：显式每 rank counts 行 / 全组列计数连续切分
    / rand seed 三种模式。"""
    S, K, E = case["S"], case["K"], world_size * case["epn"]
    if "counts" in case:
        counts = case["counts"]
        assert len(counts) == world_size and E == len(counts[0])
        topks = []
        for r in range(world_size):
            c = torch.tensor(counts[r], dtype=torch.int64)
            assert int(c.sum()) == S * K, \
                f"counts 行和 {int(c.sum())} != N={S * K}"
            topk = torch.repeat_interleave(
                torch.arange(E, dtype=torch.int32), c)
            order = torch.argsort(
                torch.arange(S * K, dtype=torch.float32)
                * 0.61803398875 % 1.0)          # 固定打散（可复现）
            topks.append(topk[order].contiguous())
        return topks
    if "cols" in case:
        c = torch.tensor(case["cols"], dtype=torch.int64)
        assert E == c.numel() and int(c.sum()) == world_size * S * K, \
            f"cols 总和 {int(c.sum())} != R*N={world_size * S * K}"
        pool = torch.repeat_interleave(           # 按专家排序的全组条目池
            torch.arange(E, dtype=torch.int32), c)
        topks = []
        for r in range(world_size):
            chunk = pool[r * S * K:(r + 1) * S * K]
            order = torch.argsort(
                torch.arange(S * K, dtype=torch.float32)
                * 0.38196601125 % 1.0)           # 固定打散（可复现）
            topks.append(chunk[order].contiguous())
        return topks
    g = torch.Generator().manual_seed(42)
    topks = []
    for r in range(world_size):
        g.manual_seed(42 + r * 1000)
        topk = torch.randint(0, E, (S * K,), generator=g, dtype=torch.int32)
        topks.append(topk)
    return topks


def _topes(topks, E):
    return [torch.bincount(t.to(torch.int64), minlength=E).to(torch.int32)
            for t in topks]


# ---------------------------------------------------------------------------
# 通用 worker：oracle 全链 + 融合 kernel + 逐位比较
# ---------------------------------------------------------------------------
def _run_case(rank, world_size, case_key, determinism=False):
    import tests._moe_testkit as kit
    from moonep.planning import (  # noqa: F401  （bootstrap 后可达）
        _phase_b_tables,
        allocate_planning_outputs,
        launch_planning,
    )
    from moonep.tests.kernel_test_utils import (
        KernelCase,
        build_world,
        planning_invariant_errors,
        run_ranks,
    )

    case = _CASES[case_key]
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    S, K, epn, B, tp = case["S"], case["K"], case["epn"], case["B"], case["tp"]
    E = world_size * epn
    N, NvS = S * K, S * K + (tp - 1) * 2 * epn
    kcase = KernelCase(f"fused_{case_key}", S=S, K=K, epn=epn, H=16,
                       num_sms=1, B=B, token_padding=tp, R=world_size,
                       seed=42)
    topks = _build_topks(case, world_size)
    tpes = _topes(topks, E)

    # ---- oracle 全链（CPU SimTransport）----
    sim, _arenas, obufs = build_world(kcase)
    oracle_outs = [None] * world_size

    def _f(r):
        ctx = obufs[r]._require_ctx()
        plan, cu = allocate_planning_outputs(ctx)
        launch_planning(ctx, topks[r].contiguous(), tpes[r], cu, plan)
        oracle_outs[r] = (plan, cu, ctx)

    run_ranks(sim, _f, "oracle")

    # ---- oracle Phase B 七表（表级对拍依据）----
    tpe_all = torch.stack([t.to(torch.int64) for t in tpes])
    tbl_o = _phase_b_tables(tpe_all, R=world_size, E=E, B=B, NvS=NvS,
                            NvS_capacity=N, token_padding=tp)

    # ---- NPU 侧：融合 kernel ----
    with kit.aclshmem_session(rank, world_size, 256 * 1024 * 1024):
        pbufs = MoonepPlanBuffers(world_size, E, B, N, NvS, device)
        outs = {
            "dst": torch.empty(N, dtype=torch.int32, device=device),
            "cu_seqlens": torch.empty(E + B, dtype=torch.int32,
                                      device=device),
            "experts_to_copy": torch.empty(world_size, B,
                                           dtype=torch.int32, device=device),
            "zero_fill_ranges": torch.empty(E + B, 2, dtype=torch.int32,
                                            device=device),
            "remote_stats": torch.empty(2, dtype=torch.int32,
                                        device=device),
            "src_info": torch.empty(NvS, dtype=torch.int32, device=device),
        }

        def _launch():
            launch_moonep_planning(
                pbufs, outs, topks[rank].to(device),
                rank=rank, world_size=world_size, ep_group=ep_group,
                S=S, K=K, E=E, B=B, NvS=NvS, token_padding=tp)

        try:
            _launch()
            got = {k: outs[k].cpu().clone() for k in outs
                   if not k.startswith("_")}
            tables = {
                "tpec": pbufs.tpec.cpu().clone(),
                "alloc_cs": pbufs.alloc_cs.cpu().clone(),
                "eoff": pbufs.eoff.cpu().clone(),
                "cu": pbufs.cu.cpu().clone(),
                "zfr": pbufs.zfr.cpu().clone(),
                "etc": pbufs.etc.cpu().clone(),
                "stats": pbufs.stats.cpu().clone(),
            }

            if determinism:
                _launch()                       # 集体：全 rank 一致二跑
                for k, v in got.items():
                    assert torch.equal(outs[k].cpu(), v), \
                        f"determinism: {k} 二跑不一致"
                for k, v in tables.items():
                    got_t = {"tpec": pbufs.tpec, "alloc_cs": pbufs.alloc_cs,
                             "eoff": pbufs.eoff, "cu": pbufs.cu,
                             "zfr": pbufs.zfr, "etc": pbufs.etc,
                             "stats": pbufs.stats}[k].cpu()
                    assert torch.equal(got_t, v), \
                        f"determinism: 表 {k} 二跑不一致"

            # ---- 表级对拍（Phase B）----
            assert torch.equal(tables["tpec"], tbl_o["tpe_cumsum"]), "tpec"
            assert torch.equal(tables["alloc_cs"], tbl_o["alloc_cumsum"]), \
                "alloc_cs"
            assert torch.equal(tables["eoff"], tbl_o["expert_offsets"]), \
                "eoff"
            assert torch.equal(tables["cu"], tbl_o["cu_all"]), "cu"
            assert torch.equal(tables["zfr"], tbl_o["zfr_all"]), "zfr"
            assert torch.equal(tables["etc"], tbl_o["etc_all"]), "etc"
            assert torch.equal(tables["stats"], tbl_o["stats_all"]), "stats"

            # ---- golden 双重锁：硬编码常数（test_oracle_selftest 同款）----
            if case_key == "golden":
                for k, want in _GOLDEN_TABLES.items():
                    assert tables[k].tolist() == want, \
                        f"golden 表 {k}：{tables[k].tolist()} != {want}"

            # ---- 全链六 outs 对拍 ----
            plan_o, cu_o, ctx_o = oracle_outs[rank]
            assert torch.equal(got["dst"], plan_o.dst), "dst 不一致"
            assert torch.equal(got["cu_seqlens"], cu_o), "cu_seqlens 不一致"
            assert torch.equal(got["experts_to_copy"],
                               plan_o.experts_to_copy), "etc 不一致"
            assert torch.equal(got["zero_fill_ranges"],
                               plan_o.zero_fill_ranges), "zfr 不一致"
            assert torch.equal(got["remote_stats"],
                               plan_o.remote_stats), "stats 不一致"
            assert torch.equal(got["src_info"], plan_o.src_info), \
                "src_info 不一致"
            errors = planning_invariant_errors(
                kcase, ctx_o, got["dst"], got["cu_seqlens"],
                got["experts_to_copy"])
            assert not errors, f"invariants: {errors[:3]}"

            # ---- 单 kernel 融合版锚点（golden 形状可编译）：与两段版逐位
            # 一致——编译器 bug 修复后切回单 kernel 的零漂移保证 ----
            if case_key == "golden":
                from mega_moe.kernels.moonep_planning import moonep_plan_fused
                pbufs.eoff.zero_()
                moonep_plan_fused[(1, 1, 1)](
                    topks[rank].to(device), pbufs.tpe_all, pbufs.src_info,
                    outs["dst"], pbufs.order, pbufs.dst_row,
                    pbufs.gtok, pbufs.ecnt, pbufs.z, pbufs.alloc,
                    pbufs.tpec, pbufs.alloc_cs, pbufs.eoff, pbufs.etc,
                    pbufs.stats, pbufs.cu, pbufs.zfr, pbufs.sel,
                    pbufs.expoff,
                    R=world_size, E=E, EPN=epn, B=B, N=N, K=K, NvS=NvS,
                    TP=tp, NP=npow2(N), NV=npow2(NvS), G=npow2(E + B),
                    LOCAL_RANK=rank, BLOCK_HIST=256, BLOCK_E=min(32, E),
                    BLOCK_N=256, BLOCK_D=256)
                fused = {
                    "dst": outs["dst"].cpu(),
                    "tpec": pbufs.tpec.cpu(), "alloc_cs": pbufs.alloc_cs.cpu(),
                    "eoff": pbufs.eoff.cpu(), "cu": pbufs.cu.cpu(),
                    "zfr": pbufs.zfr.cpu(), "etc": pbufs.etc.cpu(),
                    "stats": pbufs.stats.cpu(),
                    "src_info": pbufs.src_info.cpu(),
                }
                assert torch.equal(fused["dst"], got["dst"]), "fused dst"
                assert torch.equal(fused["src_info"], got["src_info"]), \
                    "fused src_info"
                for k in ("tpec", "alloc_cs", "eoff", "cu", "zfr", "etc",
                          "stats"):
                    assert torch.equal(fused[k], tables[k]), f"fused 表 {k}"
        finally:
            pbufs.finalize()


def _worker(rank, world_size, case_key, determinism=False):
    _run_case(rank, world_size, case_key, determinism=determinism)


@pytest.mark.dist
@pytest.mark.npu
def test_fused_golden_r2(dist_test):
    dist_test(functools.partial(_worker, case_key="golden"), world_size=2)


@pytest.mark.dist
@pytest.mark.npu
def test_fused_ties_r4(dist_test):
    dist_test(functools.partial(_worker, case_key="ties4"), world_size=4)


@pytest.mark.dist
@pytest.mark.npu
def test_fused_b3_tie_r4(dist_test):
    dist_test(functools.partial(_worker, case_key="b3tie"), world_size=4)


@pytest.mark.dist
@pytest.mark.npu
def test_fused_odd_zfr_r2(dist_test):
    dist_test(functools.partial(_worker, case_key="odd"), world_size=2)


@pytest.mark.dist
@pytest.mark.npu
def test_fused_determinism_r2(dist_test):
    dist_test(functools.partial(_worker, case_key="golden", determinism=True),
              world_size=2)


# ---------------------------------------------------------------------------
# 同 kernel 全局 store→load 回读（融合方案前提原语，单卡无 dist）
# ---------------------------------------------------------------------------
@triton.jit
def _k_raw_store_load(ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(ptr + offs, offs.to(tl.int64) * 3 + 1)
    v = tl.load(ptr + offs)              # 同 kernel 回读刚写入的全局内存
    tl.store(out_ptr + offs, v + 7)


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="需要 NPU")
@pytest.mark.parametrize("BLOCK", [64, 1024])
def test_same_kernel_global_raw(BLOCK):
    dev = "npu:0"
    buf = torch.empty(BLOCK, dtype=torch.int64, device=dev)
    out = torch.empty(BLOCK, dtype=torch.int64, device=dev)
    _k_raw_store_load[(1,)](buf, out, BLOCK=BLOCK)
    ref = torch.arange(BLOCK, dtype=torch.int64, device=dev) * 3 + 8
    assert torch.equal(out.cpu(), ref.cpu()), "同 kernel RAW 回读失败"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
