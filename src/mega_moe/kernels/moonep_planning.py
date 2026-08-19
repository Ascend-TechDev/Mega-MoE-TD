# coding=utf-8
"""MoonEP planning 的 Triton 实现（负载均衡通信规划，v1 bring-up 形态）。

语义基准：mega_moe/moonep_ref/moonep/planning.py（MoonEP torch 参考实现
@23d71348）。输出与 oracle 逐位一致（全 int32 表，无浮点）。

v1 分工（纵向打通 forward 优先；kernel 化优先级 = 计算量）：

- **宿主**（torch，直接调 moonep_ref 参考函数）：B.0-B.4 规划表
  （O(R·E) 微观量表，`_phase_b_tables` 逐位一致由构造保证）、topk/tpe 的
  HCCL allgather、expoff 前缀、src_info 宿主重建、cu/zfr/etc/stats 输出
  （本就是宿主表的切片拷贝）。
- **kernel**：
  - `moonep_c1_counting`：C.1 稳定计数排序（GPU 原版算法：直方图 +
    exclusive 前缀 + 分块 segment 前缀；天然稳定、无寄存器规模上限。
    tl.sort 因「须独占 kernel + 块寄存器上限」被否，见
    tests/kernel/moonep/test_step0_smoke.py 注释）。rank1 额外一次 launch
    代算 rank0 的 order。
  - `moonep_push_r0_order`：rank1 把 rank0 的 order 经 putmem 推给 rank0
    的对称缓冲（planning 唯一的设备侧跨 rank 数据流）。
  - `moonep_c2`：头部 barrier（等 push 完成）→ C.2 逐排序位算 dst——
    O(N) 主力计算，分块向量化；searchsorted 退化为 R 宽比较求和；
    scatter 回原条目位。
  - `moonep_dedup`：Phase D 负编码（逐 token K 扫描、per-token 位掩码，
    纯本地；R ≤ 64）。

rank 分工与参考实现一致（rank0 忙 Phase B 时由 rank1 代算其 C.1；
R==1 时 rank0 自算）。

910B 后端约束（Step 0 smoke 实测，见 test_step0_smoke.py）：tl.sort 须
独占 kernel、tie_break_left=False 不生效、while 内不支持 break——本模块
未用到这三者（计数排序替代 sort；B 表在宿主）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id

__all__ = [
    "MoonepPlanBuffers",
    "launch_moonep_planning",
    "moonep_c1_counting",
    "moonep_c2",
    "moonep_dedup",
    "moonep_push_r0_order",
]

_C1_BLOCK = 256
_C2_BLOCK = 256
_DEDUP_BLOCK = 256


def _device_id(device) -> int:
    s = str(device)
    return int(s.split(":")[-1]) if ":" in s else 0


class MoonepPlanBuffers:
    """planning 的设备缓冲（宿主持有）。

    对称：``order0`` int32 [N]——rank1 putmem 写、rank0 读（putmem 目标
    必须在对称堆；全 rank 同形同序分配，v1 里它是唯一的对称缓冲）。
    本地：``order`` int32 [N]（本 rank C.1 输出）、``dst_row`` int64 [N]
    （C.2 输出、dedup 输入）、``order0_local`` int32 [N]（仅 rank1，
    代算 rank0 的 order 暂存）。
    """

    def __init__(self, N: int, device):
        import shmem as ash

        self.N = N
        self.order0 = ash.aclshmem_create_tensor(
            [N], torch.int32, device_id=_device_id(device))
        self.order = torch.empty(N, dtype=torch.int32, device=device)
        self.dst_row = torch.empty(N, dtype=torch.int64, device=device)
        self.order0_local = torch.empty(N, dtype=torch.int32, device=device)

    def finalize(self):
        import shmem as ash

        ash.aclshmem_free_tensor(self.order0)


# ---------------------------------------------------------------------------
# C.1：稳定计数排序（单程序三阶段）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_c1_counting(
    topk_ptr,               # int32 [N]
    order_ptr,              # int32 [N] 输出稳定序（同专家内按下标升序）
    N: tl.constexpr,
    E: tl.constexpr,        # 2 的幂
    BLOCK: tl.constexpr,    # 2 的幂
):
    offs_e = tl.arange(0, E)

    # 阶段 1：直方图（单程序顺序累加；计数用 int32——N < 2^31）
    cnt = tl.zeros((E,), dtype=tl.int32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)   # 越界哨兵不计数
        cnt += tl.sum(tl.where(offs_e[None, :] == v[:, None], 1, 0), axis=0)

    # 阶段 2：exclusive 前缀 = 各专家在稳定序中的起始位
    start_e = tl.cumsum(cnt, axis=0) - cnt

    # 阶段 3：分块定位——运行计数 + 块内 onehot exclusive 前缀
    running = tl.zeros((E,), dtype=tl.int32)
    for blk in range(0, N, BLOCK):
        offs = blk + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        onehot = ((offs_e[None, :] == v[:, None]) & m[:, None]).to(tl.int32)
        pre = tl.cumsum(onehot, axis=0) - onehot        # 块内 exclusive 前缀
        rank_in_blk = tl.sum(pre * onehot, axis=1)      # [BLOCK]
        e_start = tl.sum(onehot * (start_e + running)[None, :], axis=1)
        tl.store(order_ptr + e_start + rank_in_blk, offs.to(tl.int32), mask=m)
        running += tl.sum(onehot, axis=0)


# ---------------------------------------------------------------------------
# rank1 → rank0：order0 经 putmem 推送（同 offset 惯例）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_push_r0_order(
    order0_local_ptr,       # int32 [N] rank1 本地
    order0_sym_ptr,         # 对称 int32 [N]（rank0 侧同款缓冲）
    N: tl.constexpr,
):
    if sub_vec_id() == 0:
        libshmem_device.putmem(order0_sym_ptr, order0_local_ptr, N * 4, 0)


# ---------------------------------------------------------------------------
# C.2：逐排序位计算 dst（头部 barrier 等 rank1 的推送）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_c2(
    order_ptr,              # int32 [N] 本 rank 稳定序（rank0 用 order0 对称缓冲）
    topk_ptr,               # int32 [N] 本 rank topk
    expoff_ptr,             # int64 [E] 本 rank tpe exclusive 前缀（宿主算）
    tpec_flat_ptr,          # int64 [R*E] tpe_cumsum 行主序 [R,E]
    alloc_flat_ptr,         # int64 [E*R] alloc_cumsum 行主序 [E,R]
    eoff_flat_ptr,          # int64 [R*E] expert_offsets 行主序 [R,E]
    dst_row_ptr,            # int64 [N] 输出（全非负）
    R: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    NvS: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    libshmem_device.barrier_all_vec()   # bar：rank1 的 order0 推送可见
    offs_r = tl.arange(0, R)
    for start in range(0, N, BLOCK):
        pos = start + tl.arange(0, BLOCK)               # 排序位
        m = pos < N
        o = tl.load(order_ptr + pos, mask=m, other=0).to(tl.int64)
        e = tl.load(topk_ptr + o, mask=m, other=0).to(tl.int64)

        prev = tl.zeros((BLOCK,), dtype=tl.int64)
        if LOCAL_RANK > 0:
            prev = tl.load(tpec_flat_ptr + (LOCAL_RANK - 1) * E + e,
                           mask=m, other=0)
        expoff = tl.load(expoff_ptr + e, mask=m, other=0)
        gidx = prev + (pos - expoff)                    # 全组专家序

        rows = tl.load(alloc_flat_ptr + e[:, None] * R + offs_r[None, :],
                       mask=m[:, None], other=0)        # [BLOCK, R]
        lo = tl.sum(tl.where(rows <= gidx[:, None], 1, 0), axis=1)
        pc = tl.sum(tl.where(offs_r[None, :] == (lo - 1)[:, None], rows, 0),
                    axis=1)                             # lo==0 时无匹配得 0
        eoff = tl.load(eoff_flat_ptr + lo * E + e, mask=m, other=0)
        tl.store(dst_row_ptr + o, lo * NvS + eoff + (gidx - pc), mask=m)


# ---------------------------------------------------------------------------
# Phase D：dedup 负编码（逐 token K 扫描，per-token rank 位掩码，纯本地）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_dedup(
    dst_row_ptr,            # int64 [N] 全非负（N = S*K）
    dst_ptr,                # int32 [N] 输出（负 = -raw-1）
    N: tl.constexpr,
    NvS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,    # token 分块（2 的幂）
):
    S: tl.constexpr = N // K
    one = tl.full((BLOCK,), 1, dtype=tl.int64)
    for t0 in range(0, S, BLOCK):
        tok = t0 + tl.arange(0, BLOCK)
        m = tok < S
        seen = tl.zeros((BLOCK,), dtype=tl.int64)
        for k in range(K):
            raw = tl.load(dst_row_ptr + tok * K + k, mask=m, other=0)
            dest = raw // NvS
            dup = (seen >> dest) & one
            enc = tl.where(dup == one, -raw - one, raw)
            tl.store(dst_ptr + tok * K + k, enc.to(tl.int32), mask=m)
            seen = seen | (one << dest)


# ---------------------------------------------------------------------------
# 宿主编排
# ---------------------------------------------------------------------------
def launch_moonep_planning(
    bufs: MoonepPlanBuffers,
    outs: dict,
    topk: torch.Tensor,
    tpe: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    ep_group,
    S: int,
    K: int,
    E: int,
    B: int,
    NvS: int,
    token_padding: int,
) -> None:
    """跑一次 planning，结果原地写入 ``outs`` 的各张量。

    outs 契约（全部预分配、int32 连续、在 topk 同 device）：
        dst [S*K]、cu_seqlens [E+B]、experts_to_copy [R,B]、
        zero_fill_ranges [E+B,2]、remote_stats [2]、src_info [NvS]
    """
    import torch.distributed as dist

    from mega_moe.moonep_ref import _phase_b_tables

    N = S * K
    R = world_size
    dev = topk.device
    assert topk.dtype == torch.int32 and topk.numel() == N
    assert tpe.dtype == torch.int32 and tpe.numel() == E
    assert R <= 64, f"dedup 位掩码要求 R<=64，got {R}"
    assert N % K == 0 and E % R == 0 and (E & (E - 1)) == 0, \
        f"v1 约束：N%K==0 且 E 为 2 的幂，got N={N} K={K} E={E}"

    # ---- 宿主 allgather topk/tpe（HCCL）----
    topk_list = [torch.empty(N, dtype=torch.int32, device=dev)
                 for _ in range(R)]
    tpe_list = [torch.empty(E, dtype=torch.int32, device=dev)
                for _ in range(R)]
    dist.all_gather(topk_list, topk.reshape(-1).contiguous(), group=ep_group)
    dist.all_gather(tpe_list, tpe.reshape(-1).contiguous(), group=ep_group)
    topk_all = torch.stack([t.to(torch.int64).cpu() for t in topk_list])
    tpe_all = torch.stack([t.to(torch.int64).cpu() for t in tpe_list])

    # ---- 宿主 B 表（参考实现原函数——逐位一致由构造保证）----
    tbl = _phase_b_tables(tpe_all, R=R, E=E, B=B, NvS=NvS,
                          NvS_capacity=N, token_padding=token_padding)

    # ---- 上传 C.2 所需表 ----
    tpec_flat = tbl["tpe_cumsum"].reshape(-1).to(dev)               # [R,E] i64
    alloc_flat = tbl["alloc_cumsum"].reshape(-1).to(dev)            # [E,R] i64
    eoff_flat = tbl["expert_offsets"].reshape(-1).to(dev)           # [R,E] i64
    tpe_i64 = tpe_all[rank].to(dev)                                 # [E] i64
    expoff = torch.zeros(E, dtype=torch.int64, device=dev)
    expoff[1:] = torch.cumsum(tpe_i64, 0)[:-1]

    # ---- C.1：计数排序（rank≠0 算自己；rank1 额外代算 rank0；R==1 时 rank0 自算）----
    if rank != 0 or R == 1:
        moonep_c1_counting[(1, 1, 1)](
            topk, bufs.order, N=N, E=E, BLOCK=_C1_BLOCK)
    if rank == 1:
        moonep_c1_counting[(1, 1, 1)](
            topk_list[0], bufs.order0_local, N=N, E=E, BLOCK=_C1_BLOCK)
        moonep_push_r0_order[(1, 1, 1)](
            bufs.order0_local, bufs.order0, N=N)

    # ---- C.2 + dedup ----
    order_arg = bufs.order0 if rank == 0 else bufs.order
    if R == 1:
        order_arg = bufs.order
    moonep_c2[(1, 1, 1)](
        order_arg, topk, expoff, tpec_flat, alloc_flat, eoff_flat,
        bufs.dst_row, R=R, E=E, N=N, NvS=NvS, LOCAL_RANK=rank,
        BLOCK=_C2_BLOCK)
    moonep_dedup[(1, 1, 1)](
        bufs.dst_row, outs["dst"], N=N, NvS=NvS, K=K, BLOCK=_DEDUP_BLOCK)

    # ---- 输出表：宿主切片拷贝 ----
    outs["cu_seqlens"].copy_(tbl["cu_all"][rank].to(torch.int32).to(dev))
    outs["experts_to_copy"].copy_(
        tbl["etc_all"].to(torch.int32).to(dev))
    outs["zero_fill_ranges"].copy_(
        tbl["zfr_all"][rank].to(torch.int32).to(dev))
    outs["remote_stats"].copy_(
        tbl["stats_all"][rank].to(torch.int32).to(dev))

    # ---- src_info 宿主重建：allgather 最终 dst，倒排每个槽的出处 ----
    dst_list = [torch.empty(N, dtype=torch.int32, device=dev)
                for _ in range(R)]
    dist.all_gather(dst_list, outs["dst"], group=ep_group)
    dst_all = torch.stack([d.to(torch.int64).cpu() for d in dst_list])
    src_info = torch.full((NvS,), -1, dtype=torch.int64)
    for sr in range(R):
        v = dst_all[sr]
        raw = torch.where(v < 0, -v - 1, v)
        dr = raw // NvS
        loff = raw - dr * NvS
        sel = (dr == rank).nonzero(as_tuple=False).reshape(-1)   # offv 位
        src_info[loff[sel].long()] = sr * NvS + sel
    outs["src_info"].copy_(src_info.to(torch.int32).to(dev))
