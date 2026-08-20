# coding=utf-8
"""MoonEP 宿主侧元数据构建（消费 planning 输出，喂 dispatch/combine）。

三个构建器（全部纯 torch 宿主计算，planning 输出已 allgather 可用的场景
由调用方提供 dst_all）：

- ``build_moonep_segment_meta``：cu[E+B] → 压缩段表 [Seg=epn+B]。
  **恒等映射**：压缩段 id == 权重表行号（本地专家段 [0,epn) + 槽段
  [epn,Seg)）。成立前提 = 副本槽全覆盖（remote_stats[0] ≤ B，由
  B.3 单源性质保证），launcher 负责断言。
- ``build_moonep_send_meta``：本 rank 的 dst[N] → 发送表（按 dst 稳定
  排序得 (dst, seg) 连续 run；段内 loff 升序 = 段内 source-major 就绪
  协议的前提，MoonEP 的 gidx 构造天然满足）。
- ``build_recv_counts``：全组 dst → 本 rank 的 [R, Seg] 每源每段计数
  （FC1 就绪 wait 的前缀推导输入）。
"""

from __future__ import annotations

import torch


def build_moonep_segment_meta(cu: torch.Tensor, etc_row: torch.Tensor,
                              rank: int, epn: int, E: int, B: int):
    """cu int32 [E+B] → (seg_counts [Seg], seg_offsets [Seg+1],
    seg_expert_global [Seg])，全部 int32/cpu。"""
    cu = cu.to(torch.int64)
    counts = cu - torch.cat([torch.zeros(1, dtype=torch.int64), cu[:-1]])
    seg_counts = torch.cat([counts[rank * epn:(rank + 1) * epn],
                            counts[E:E + B]])
    seg_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(seg_counts, 0)])
    seg_expert = torch.cat([
        torch.arange(rank * epn, (rank + 1) * epn, dtype=torch.int64),
        etc_row.to(torch.int64),
    ])
    return (seg_counts.to(torch.int32), seg_offsets.to(torch.int32),
            seg_expert.to(torch.int32))


def _global_seg_of_loff(cu: torch.Tensor, loff: torch.Tensor) -> torch.Tensor:
    """cu[E+B]（段边界）→ loff 所在全局段号 g（cu 前补 0）。"""
    cu64 = cu.to(torch.int64)
    return torch.searchsorted(torch.cat([torch.zeros(1, dtype=torch.int64),
                                         cu64]), loff.to(torch.int64),
                              right=True) - 1


def _compress_seg(g: torch.Tensor, rank: int, epn: int, E: int) -> torch.Tensor:
    """全局段号 → 压缩段 id（本地段平移 + 槽段接尾）。"""
    return torch.where(g < E, g - rank * epn, epn + (g - E))


def build_moonep_send_meta(dst: torch.Tensor, cu_all: torch.Tensor, K: int,
                           epn: int, E: int, B: int, NvS: int):
    """本 rank dst[N]（int32，含负 dup）→ 发送表 dict（全部 cpu/int32）。

    cu_all：int64 [R, E+B] **全组**段边界（planning 宿主表 cu_all）。
    段查表与压缩都必须按**目的 rank**：cu 每 rank 不同、压缩段号是目的
    rank 的本地段（CASE-11：用本 rank 的 cu/位压缩会把信号槽写错位置，
    消费侧 GEMM 等不到信号 → aicore 超时）。

    send_src_idx[i] = 排序后第 i 个条目的源 token 行（offv // K）
    send_offv[i]    = 原条目位（routing_weights 下标）
    send_loff[i]    = 目的行号
    run_dst/run_seg/run_start/run_count = (dst, seg) 连续 run 描述
    """
    R = cu_all.shape[0]
    d = dst.to(torch.int64)
    raw = torch.where(d < 0, -d - 1, d)
    dr = raw // NvS
    loff = raw - dr * NvS
    offv = torch.arange(d.numel(), dtype=torch.int64)
    g = torch.empty_like(loff)
    for x in range(R):
        m = dr == x
        if bool(m.any()):
            g[m] = _global_seg_of_loff(cu_all[x], loff[m])
    seg = torch.where(g < E, g - dr * epn, epn + (g - E))

    order = torch.argsort(d, stable=True)      # 按 (dst, loff) 稳定序
    send_src_idx = (offv[order] // K).to(torch.int32)
    send_offv = offv[order].to(torch.int32)
    send_loff = loff[order].to(torch.int32)

    sdr = dr[order]
    sseg = seg[order]
    new_run = torch.cat([torch.ones(1, dtype=torch.bool),
                         (sdr[1:] != sdr[:-1]) | (sseg[1:] != sseg[:-1])])
    run_starts = torch.nonzero(new_run, as_tuple=False).reshape(-1)
    run_dst = sdr[run_starts].to(torch.int32)
    run_seg = sseg[run_starts].to(torch.int32)
    run_count = torch.cat([run_starts[1:], torch.tensor([d.numel()])]) \
        - run_starts
    run_count = run_count.to(torch.int32)
    run_start = run_starts.to(torch.int32)
    return {
        "send_src_idx": send_src_idx, "send_offv": send_offv,
        "send_loff": send_loff, "run_dst": run_dst, "run_seg": run_seg,
        "run_start": run_start, "run_count": run_count,
    }


def build_recv_counts(dst_all: torch.Tensor, cu: torch.Tensor, rank: int,
                      epn: int, E: int, B: int, NvS: int) -> torch.Tensor:
    """全组 dst [R,N]（int64/cpu）→ 本 rank 的 [R, Seg] 每源每段计数。"""
    R = dst_all.shape[0]
    Seg = epn + B
    out = torch.zeros(R, Seg, dtype=torch.int32)
    d = dst_all.to(torch.int64)
    raw = torch.where(d < 0, -d - 1, d)
    dr = raw // NvS
    loff = raw - dr * NvS
    g = _global_seg_of_loff(cu, loff)
    seg = _compress_seg(g, rank, epn, E)
    for sr in range(R):
        mine = dr[sr] == rank
        ids = seg[sr][mine]
        out[sr] = torch.bincount(ids, minlength=Seg).to(torch.int32)
    return out
