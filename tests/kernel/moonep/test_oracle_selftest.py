# coding=utf-8
"""vendor oracle 自测（纯 CPU，无 NPU/分布式依赖）——M0a 验收。

三层自检：
1. 纯函数黄金常数：标准例（R=2,E=4,S=4,K=2,B=1,tp=2）的 B 表 / C1 / C2 / D
   输出与 MoonEP 参考实现实跑验证过的数值逐位一致（常数固化，锁定 vendor
   副本本身正确）；
2. 全链 CPU 模拟：SimTransport 跑 PlanningKernel 编排（R=2 与 R=1），
   不变量检查通过 + 同输入二次规划逐位一致 + 纯函数重算 dst 与链路输出
   交叉一致；
3. 源仓对拍（本机有 MoonEP clone 时）：子进程 import 源仓 @23d71348 跑同
   样的纯函数，与 vendor 副本逐位相等（证明 vendor 无改编失真）。
"""

import os
import subprocess
import sys

import pytest
import torch

from tests._moonep_oracle import (
    _phase_b_tables,
    _phase_c1_order,
    _phase_c2_rank,
    _phase_d_dedup,
)

# 标准例：R=2, E=4(epn=2), S=4, K=2, B=1, tp=2 → N=8, CAP=8, NvS=12
_STD = dict(R=2, E=4, B=1, NvS=12, NvS_capacity=8, token_padding=2)

_ORIG_EXPERT_PARALLEL = (
    "/home/z00905891/MoonEP/MindSpeed-MM_MoonEP/"
    "mindspeed_mm/fsdp/distributed/expert_parallel"
)


def _run_pure_functions():
    """标准例纯函数计算（vendor 副本与源仓对拍共用的载荷）。"""
    tpe_all = torch.tensor([[2, 2, 2, 2], [6, 0, 2, 0]], dtype=torch.int64)
    tbl = _phase_b_tables(tpe_all, **_STD)
    topk = torch.tensor([3, 2, 2, 0, 0, 0, 1, 3], dtype=torch.int32)
    order = _phase_c1_order(topk)
    # C2/D 链：与 tpe 一致的 rank1 输入（e0×6 + e2×2）
    topk1 = torch.tensor([0, 0, 0, 0, 0, 0, 2, 2], dtype=torch.int32)
    tpe1 = torch.tensor([6, 0, 2, 0], dtype=torch.int32)
    dst_row, src_writes = _phase_c2_rank(
        order=_phase_c1_order(topk1), topk_flat=topk1, tpe_local=tpe1,
        tpe_cumsum=tbl["tpe_cumsum"], alloc_cumsum=tbl["alloc_cumsum"],
        expert_offsets=tbl["expert_offsets"], rank=1,
        R=2, E=4, S=4, K=2, NvS=12,
    )
    dst_final = _phase_d_dedup(dst_row, S=4, K=2, R=2, NvS=12)
    return {
        "tpe_cumsum": tbl["tpe_cumsum"],
        "alloc_cumsum": tbl["alloc_cumsum"],
        "expert_offsets": tbl["expert_offsets"],
        "cu_all": tbl["cu_all"],
        "zfr_all": tbl["zfr_all"],
        "etc_all": tbl["etc_all"],
        "stats_all": tbl["stats_all"],
        "order": order,
        "dst_row": dst_row,
        "dst_final": dst_final,
        "src_writes": {k: v for k, v in src_writes.items()},
    }


def test_phase_b_tables_golden():
    out = _run_pure_functions()
    assert out["tpe_cumsum"].tolist() == [[2, 2, 2, 2], [8, 2, 4, 2]]
    assert out["alloc_cumsum"].tolist() == [[6, 8], [2, 2], [0, 4], [0, 2]]
    assert out["expert_offsets"].tolist() == [[0, 6, 0, 0], [6, 0, 0, 4]]
    assert out["cu_all"].tolist() == [[6, 8, 8, 8, 8], [0, 0, 4, 6, 8]]
    assert out["zfr_all"].tolist() == [[[0, 0]] * 5 for _ in range(2)]
    assert out["etc_all"].tolist() == [[-1], [0]]
    assert out["stats_all"].tolist() == [[0, 1], [1, 0]]


def test_c1_c2_dedup_golden():
    out = _run_pure_functions()
    assert out["order"].tolist() == [3, 4, 5, 6, 1, 2, 0, 7]
    assert out["dst_row"].tolist() == [2, 3, 4, 5, 18, 19, 14, 15]
    # token0/1/2/3 的两个条目各落同一卡 → 第 2 份负编码
    assert out["dst_final"].tolist() == [2, -4, 4, -6, 18, -20, 14, -16]


def _run_chain_case(R, **kw):
    """SimTransport 全链跑一次 planning，返回 (case, outs)。"""
    from moonep.tests.kernel_test_utils import (
        KernelCase,
        build_world,
        make_topk,
        run_ranks,
    )

    # R=1 时 3*E*R 须被 4 整除（api.py meta 对齐约束）→ epn 取 4；R=2 用 epn=2
    case_kwargs = dict(S=4, K=2, epn=4 if R == 1 else 2, H=16, num_sms=1,
                       B=1, token_padding=2, R=R, seed=42)
    case_kwargs.update(kw)
    case = KernelCase(f"selftest_r{R}", **case_kwargs)
    sim, _arenas, bufs = build_world(case)
    outs = [None] * case.R

    def _f(r):
        from moonep.planning import allocate_planning_outputs, launch_planning

        ctx = bufs[r]._require_ctx()
        topk, tpe = make_topk(case, r)
        plan, cu = allocate_planning_outputs(ctx)
        launch_planning(ctx, topk.reshape(-1).contiguous(), tpe, cu, plan)
        outs[r] = {
            "ctx": ctx,
            "topk": topk.reshape(-1).clone(),
            "tpe": tpe.clone(),
            "dst": plan.dst.clone(),
            "cu": cu.clone(),
            "etc": plan.experts_to_copy.clone(),
            "zfr": plan.zero_fill_ranges.clone(),
            "stats": plan.remote_stats.clone(),
            "src_info": plan.src_info.clone(),
        }

    run_ranks(sim, _f, f"selftest_r{R}")
    return case, outs


@pytest.mark.parametrize("R", [2, 1])
def test_full_chain_cpu_invariants_and_determinism(R):
    from moonep.tests.kernel_test_utils import planning_invariant_errors

    case, outs = _run_chain_case(R)
    # R=1 时 build_world 的世界即单 rank；不变量 + 二次规划确定性
    case2, outs2 = _run_chain_case(R)
    for r in range(R):
        errors = planning_invariant_errors(
            case, outs[r]["ctx"],
            outs[r]["dst"], outs[r]["cu"], outs[r]["etc"],
        )
        assert not errors, f"rank{r} invariants: {errors[:3]}"
        for field in ("dst", "cu", "etc", "zfr", "stats", "src_info"):
            assert torch.equal(outs[r][field], outs2[r][field]), \
                f"rank{r} {field} 二次规划不一致"


def test_full_chain_matches_pure_functions():
    """链路输出的 dst 与『纯函数重算』逐位一致（编排层交叉验证）。"""
    case, outs = _run_chain_case(2)
    tpe_all = torch.stack([outs[r]["tpe"].to(torch.int64) for r in range(2)])
    E, NvS = case.E(2), case.NvS(2)
    tbl = _phase_b_tables(tpe_all, R=2, E=E, B=case.B, NvS=NvS,
                          NvS_capacity=case.N,
                          token_padding=case.token_padding)
    for r in range(2):
        dst_row, _sw = _phase_c2_rank(
            order=_phase_c1_order(outs[r]["topk"]),
            topk_flat=outs[r]["topk"], tpe_local=outs[r]["tpe"],
            tpe_cumsum=tbl["tpe_cumsum"], alloc_cumsum=tbl["alloc_cumsum"],
            expert_offsets=tbl["expert_offsets"], rank=r,
            R=2, E=E, S=case.S, K=case.K, NvS=NvS,
        )
        expect = _phase_d_dedup(dst_row, S=case.S, K=case.K, R=2,
                                NvS=NvS)
        assert torch.equal(outs[r]["dst"], expect.to(torch.int32)), \
            f"rank{r} 链路 dst ≠ 纯函数重算"


@pytest.mark.skipif(not os.path.isdir(_ORIG_EXPERT_PARALLEL),
                    reason="本机无 MoonEP 源仓 clone")
def test_vendor_matches_source_repo():
    """子进程 import 源仓（@23d71348）跑同一载荷，与 vendor 副本逐位比。"""
    import tempfile

    # 内联与 _run_pure_functions 相同的载荷（源仓侧独立执行）
    script = (
        "import sys, torch\n"
        f"sys.path.insert(0, {_ORIG_EXPERT_PARALLEL!r})\n"
        "from moonep.planning import (_phase_b_tables, _phase_c1_order,"
        " _phase_c2_rank, _phase_d_dedup)\n"
        "t = _phase_b_tables(torch.tensor([[2,2,2,2],[6,0,2,0]],"
        " dtype=torch.int64), R=2, E=4, B=1, NvS=12, NvS_capacity=8,"
        " token_padding=2)\n"
        "o = _phase_c1_order(torch.tensor([3,2,2,0,0,0,1,3],"
        " dtype=torch.int32))\n"
        "tk = torch.tensor([0,0,0,0,0,0,2,2], dtype=torch.int32)\n"
        "tp = torch.tensor([6,0,2,0], dtype=torch.int32)\n"
        "d, _ = _phase_c2_rank(_phase_c1_order(tk), tk, tp,"
        " t['tpe_cumsum'], t['alloc_cumsum'], t['expert_offsets'], 1,"
        " R=2, E=4, S=4, K=2, NvS=12)\n"
        "df = _phase_d_dedup(d, S=4, K=2, R=2, NvS=12)\n"
        "torch.save({'tpe_cumsum': t['tpe_cumsum'],"
        " 'alloc_cumsum': t['alloc_cumsum'], 'expert_offsets':"
        " t['expert_offsets'], 'cu_all': t['cu_all'], 'zfr_all':"
        " t['zfr_all'], 'etc_all': t['etc_all'], 'stats_all':"
        " t['stats_all'], 'order': o, 'dst_row': d, 'dst_final': df},"
        " sys.argv[1])\n"
    )
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "orig.pt")
        proc = subprocess.run([sys.executable, "-c", script, out_path],
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"源仓子进程失败:\n{proc.stderr[-2000:]}"
        orig = torch.load(out_path)
        vend = _run_pure_functions()
        for key in ("tpe_cumsum", "alloc_cumsum", "expert_offsets", "cu_all",
                    "zfr_all", "etc_all", "stats_all", "order", "dst_row",
                    "dst_final"):
            assert torch.equal(orig[key], vend[key]), f"{key} 与源仓不一致"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
