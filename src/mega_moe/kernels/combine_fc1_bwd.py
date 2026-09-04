# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  combine_fc1_bwd.py  —  step 4: fc1 input-grad + reverse-A2A(expert->home)
#  + gate-grad. Two-stream expert-group pipeline mirroring the forward FC2
#  remote-store pipeline (commit f649a63, fc2_combine.py).
# ============================================================================
#
# The Cube fc1-input-grad GEMM and the Vector reverse-A2A push overlap across
# expert groups via two NPU streams + per-group events:
#   - Cube group g: GEMM writes hidden_buf rows for experts [e0, e1) (disjoint
#     across groups) on cube_stream, then records group_events[g].
#   - Vector group g: wait_event(group_events[g]); push those rows into peer_mem
#     on vector_stream. Meanwhile Cube runs group g+1 — depth-1 pipeline.
# One barrier_all_vec (separate AICore-sized launch) fences the cross-rank
# peer_mem writes; then the topk-sum reduce runs on the caller stream.
#
# The step4 Cube->Vector handoff matches the forward FC2 direction. Ascend does
# not reliably publish a Cube-side GM atomic from a mixed kernel, so the
# hidden_buf handoff (LOCAL) is fenced by a stream Event only — no per-tile
# signal/wait (that is the Vector->Cube direction used by step1 dispatch_fc2).
#
# The push iterates src_pos (expert-major) via write_rank_by_src /
# write_off_by_src — a disjoint-row reorder of the same store set the previous
# fused kernel emitted, so peer_mem contents after the push are unchanged.
# The gate (routing-weight) grad stays host-side (_gate_bwd_host): dl.symm_at
# only resolves at heap offset 0, so no second symmetric buffer for the gate.

import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .common import ncore, nvec, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K

# Each peer_mem row packs the H hidden elements plus a trailing gate channel
# (GATE_PAD wide) so the routing-weight grad rides step4's push+reduce instead
# of a separate host HCCL all_to_all. 8 keeps the row stride 16-byte aligned.
GATE_PAD = 8


def _combine_gemm_tile():
    """Combine fc1-input-grad GEMM tile + num_stages, env-tunable for sweeping
    (does NOT touch common.BLOCK_SIZE_*, which dispatch_fc2/wgrad also use).
    Defaults: BM=128 / BN=256 / BK=128 — with the grid schedule (see
    _kernel_combine_fc1_bwd_gemm_grid) this tile is the measured optimum on
    kimi-k3 w8 (31.6ms serial vs 39.6 for BM=256/BN=128 — the persistent
    schedule's optimum flips, so retune here after changing MOE_COMBINE_GEMM_SCHED).
    num_stages>0 turns the K-loop into tl.range(..., num_stages=NS) software
    pipelining — a bare range() ignores the launch-level num_stages kwarg on
    triton-ascend, and the pipeliner only engages once the guarding persistent
    task loop is gone (grid schedule); default 2 (saturates: 2/3/4 within
    0.2ms)."""
    return (
        int(os.environ.get("MOE_COMBINE_GEMM_BM", "128")),
        int(os.environ.get("MOE_COMBINE_GEMM_BN", "256")),
        int(os.environ.get("MOE_COMBINE_GEMM_BK", "128")),
        int(os.environ.get("MOE_COMBINE_GEMM_NS", "2")),
    )


def _gemm_tile_maps(saved, block_m):
    """Per-GEMM-tile expert/row tables for BM>64 combine tiling, cached per BM.

    The forward meta tiling locks tokens into BLOCK_SIZE_M=64 rows, which pins
    the combine GEMM to BM=64 and leaves L0A 1/4-full (a-tile [64,128] bf16 =
    16KB vs 64KB L0A). These tables decouple the GEMM tile height from that
    meta: tiles are derived directly from expert_counts in `block_m` rows,
    expert-major, so the GEMM can use BM=128/256 while the push/reduce phases
    keep their row-range group bounds.

    ``tile_home_bound`` is the first M-tile of the replica segment: tiles are
    expert-major, so the MoonEP home slots [0, epn) own a contiguous tile
    prefix and the dual-table GEMM split is one range cut. Without MoonEP the
    bound covers every tile and the split is never taken."""
    cache_key = f"_combine_gemm_tiles_{block_m}"
    cached = saved.get(cache_key)
    if cached is not None:
        return cached
    device = f"npu:{saved['ep_rank']}"
    counts = saved["expert_counts"].to(device)
    epr = counts.shape[0]
    home_experts = (
        int(saved["home_experts_per_rank"]) if saved.get("use_moonep") else epr
    )
    tiles_per_expert = (counts.to(torch.int64) + block_m - 1) // block_m
    cum_tiles = torch.zeros(epr + 1, dtype=torch.int64, device=device)
    cum_tiles[1:] = tiles_per_expert.cumsum(0)
    tile_expert = torch.repeat_interleave(
        torch.arange(epr, dtype=torch.int64, device=device), tiles_per_expert)
    chunk = torch.arange(int(tiles_per_expert.sum()), dtype=torch.int64, device=device) \
        - cum_tiles[tile_expert]
    row_cum = saved["split_size_cum_per_expert"].to(device).to(torch.int64)
    row0 = row_cum[tile_expert] + chunk * block_m
    rows = torch.clamp(counts.to(torch.int64)[tile_expert] - chunk * block_m,
                       min=0, max=block_m).to(torch.int32)
    cached = dict(
        tile_expert=tile_expert.to(torch.int32).contiguous(),
        tile_row0=row0.to(torch.int32).contiguous(),
        tile_rows=rows.contiguous(),
        num_tiles_m=int(tiles_per_expert.sum().item()),
        tile_home_bound=int(cum_tiles[home_experts].item()),
    )
    saved[cache_key] = cached
    return cached


@triton.jit
def _kernel_combine_fc1_bwd_gemm_group(
    # Phase 1 (Cube): fc1 input-grad GEMM for tile_m in [FIRST_TILE_M, LAST_TILE_M)
    inp_ptr,                  # grad_fc1_output [M, 2*ffn]  (sorted)
    weight_ptr,               # fc1_combined [E, 2*ffn, H]  (K=2*ffn, N=H)
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H] out (LOCAL)
    tile_expert_ptr,          # int32 [T_m] expert of each GEMM tile (expert-major)
    tile_row0_ptr,            # int32 [T_m] first row of each GEMM tile
    tile_rows_ptr,            # int32 [T_m] valid row count of each GEMM tile
    N, K, num_tiles_n,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn,
    FIRST_TILE_M: tl.constexpr, LAST_TILE_M: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """One fc1 input-grad GEMM sweep over the M-tile range
    [FIRST_TILE_M, LAST_TILE_M) against ONE weight table re-based at
    WEIGHT_EXPERT_BASE (0 for the home fc1_combined table, epn for the replica
    gate/up table) — the dual-launch pattern of forward dispatch_fc1."""
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        om = tl.arange(0, BLOCK_M)
        on_ = tl.arange(0, BLOCK_N)
        ok = tl.arange(0, BLOCK_K)
        group_tiles = LAST_TILE_M - FIRST_TILE_M
        total_tasks = group_tiles * num_tiles_n
        # contiguous block partition (per triton_gen/zhoujinggan/output/report.md
        # iter_1): each pid owns [pid*blk, (pid+1)*blk) consecutive (tile_m,tile_n)
        # tasks -> consecutive tiles of one expert reuse its weight rows in L1/L2
        # (better locality than the interleaved stride). blk=ceil(total/ncore) + the
        # task<total guard cover the tail on non-divisible shapes.
        blk = (total_tasks + ncore - 1) // ncore
        base = pid * blk
        for i in range(blk):
            task_id = base + i
            if task_id < total_tasks:
                tile_m = FIRST_TILE_M + (task_id % group_tiles)
                tile_n = task_id // group_tiles
                expert_id = tl.load(tile_expert_ptr + tile_m)
                row_start = tl.load(tile_row0_ptr + tile_m)
                rem = tl.load(tile_rows_ptr + tile_m)
                n_start = tile_n * BLOCK_N
                mm = om < rem
                mn = on_ < (N - n_start)
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                wb = (expert_id.to(tl.int64) - WEIGHT_EXPERT_BASE) * stride_we
                for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
                        mk = ok < (K - ks)
                        ao = (row_start + om[:, None]) * stride_im + (ks + ok[None, :]) * stride_ik
                        a = tl.load(inp_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
                        bo = (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
                        b = tl.load(weight_ptr + wb + bo, mask=mk[:, None] & mn[None, :], other=0.0)
                        acc += tl.dot(a, b)
                co = (row_start + om[:, None]) * N + (n_start + on_[None, :])
                tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty),
                         mask=mm[:, None] & mn[None, :])


@triton.jit
def _kernel_combine_fc1_bwd_gemm_grid(
    # Grid-launch variant of _kernel_combine_fc1_bwd_gemm_group: pid IS the
    # (tile_m fastest) task id — no persistent blk loop, no task<total guard.
    inp_ptr,                  # grad_fc1_output [M, 2*ffn]  (sorted)
    weight_ptr,               # fc1_combined [E, 2*ffn, H]  (K=2*ffn, N=H)
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H] out (LOCAL)
    tile_expert_ptr,          # int32 [T_m] expert of each GEMM tile (expert-major)
    tile_row0_ptr,            # int32 [T_m] first row of each GEMM tile
    tile_rows_ptr,            # int32 [T_m] valid row count of each GEMM tile
    N, K, num_tiles_n,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn,
    FIRST_TILE_M: tl.constexpr, LAST_TILE_M: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, EVEN_K: tl.constexpr,
):
    """The persistent variant keeps a ``blk`` loop wrapped in ``if task_id <
    total_tasks``; the guarded outer loop stops triton-ascend's MTE->cube
    software pipeliner from engaging on the K-loop, leaving every k-iteration's
    loads un-overlapped (~261 tasks/core x 48 iters of MTE round-trip latency —
    the measured 39.5ms is latency, not bandwidth: the ~27GB traffic floor is
    ~11ms). One task per program also keeps the tile_m-fastest order (pid
    order), so expert weight locality is unchanged. Masks are hoisted out of
    the K-loop and EVEN_K (K % BLOCK_K == 0, always true for kimi shapes)
    drops the k-mask entirely — the loop body reduces to two loads + one dot,
    the shape the wgrad kernel (transposed_grouped_gemm.py) proved pipelines
    with num_stages on this backend."""
    pid = tl.program_id(axis=0)
    group_tiles = LAST_TILE_M - FIRST_TILE_M
    tile_m = FIRST_TILE_M + (pid % group_tiles)
    tile_n = pid // group_tiles
    with al.scope(core_mode="cube", disable_auto_sync=True):
        expert_id = tl.load(tile_expert_ptr + tile_m)
        row_start = tl.load(tile_row0_ptr + tile_m).to(tl.int64)
        rem = tl.load(tile_rows_ptr + tile_m)
        n_start = tile_n * BLOCK_N
        om = tl.arange(0, BLOCK_M)
        on_ = tl.arange(0, BLOCK_N)
        ok = tl.arange(0, BLOCK_K)
        mm = om < rem
        mn = on_ < (N - n_start)
        row_base = row_start + om.to(tl.int64)
        wb = (expert_id.to(tl.int64) - WEIGHT_EXPERT_BASE) * stride_we
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            if EVEN_K:
                a = tl.load(inp_ptr + row_base[:, None] * stride_im
                            + (ks + ok[None, :]) * stride_ik,
                            mask=mm[:, None], other=0.0)
                b = tl.load(weight_ptr + wb
                            + (ks + ok[:, None]) * stride_wk
                            + (n_start + on_[None, :]) * stride_wn,
                            mask=mn[None, :], other=0.0)
            else:
                mk = ok < (K - ks)
                a = tl.load(inp_ptr + row_base[:, None] * stride_im
                            + (ks + ok[None, :]) * stride_ik,
                            mask=mm[:, None] & mk[None, :], other=0.0)
                b = tl.load(weight_ptr + wb
                            + (ks + ok[:, None]) * stride_wk
                            + (n_start + on_[None, :]) * stride_wn,
                            mask=mk[:, None] & mn[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = row_base[:, None] * N + (n_start + on_[None, :])
        tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty),
                 mask=mm[:, None] & mn[None, :])


@triton.jit
def _kernel_combine_fc1_bwd_push_group(
    # Phase 2 (Vector): reverse-A2A push (expert->home) for src_pos in
    # [FIRST_SRC_POS, LAST_SRC_POS). Iterates src_pos (expert-major hidden_buf
    # row); write_rank_by_src/write_off_by_src give the per-row peer destination.
    # Each peer_mem row is [hidden (H) | gate (GATE_PAD)] so the per-row routing-
    # weight grad rides the same RMA as the hidden grad (no separate HCCL a2a).
    hidden_buf_ptr,
    write_rank_by_src_ptr, write_off_by_src_ptr,
    peer_mem_ptr,           # symmetric [total_send, H+GATE_PAD] at HEAP OFFSET 0
    grad_gate_ptr,          # [M] bf16 expert-sorted gate channel to pack
    H_push,
    FIRST_SRC_POS: tl.constexpr, LAST_SRC_POS: tl.constexpr,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        # Pure-AIV launch: one Vector core per program. Gate the dl.symm_at store
        # to sub_vec 0 so it is not duplicated, then distribute src_pos directly
        # over pid (forward _kernel_remote_store_transport_group pattern).
        if sub_vec_id() == 0:
            ovp = tl.arange(0, BLOCK_N_PUSH)
            row_stride = H_push + GATE_PAD
            for src_pos in range(FIRST_SRC_POS + pid, LAST_SRC_POS, num_progs):
                sp64 = src_pos.to(tl.int64)
                dst_rank = tl.load(write_rank_by_src_ptr + sp64)
                dst_off = tl.load(write_off_by_src_ptr + sp64).to(tl.int64)
                dst_base = dl.symm_at(peer_mem_ptr, dst_rank) + dst_off * row_stride
                for ns in range(0, H_push, BLOCK_N_PUSH):
                    mask = ovp < (H_push - ns)
                    val = tl.load(hidden_buf_ptr + sp64 * H_push + (ns + ovp),
                                  mask=mask, other=0.0)
                    tl.store(dst_base + ns + ovp, val, mask=mask)
                # pack the gate grad as the trailing channel of this row
                tl.store(dst_base + H_push, tl.load(grad_gate_ptr + sp64))


@triton.jit
def _kernel_combine_fc1_bwd_barrier():
    """Fence grouped Vector RMA with the backend's barrier-sized grid.

    ``barrier_all_vec`` is tied to one participating Vector block per AI Core;
    it must be a separate AICore-sized launch, never embedded in the
    num_vector/num_core transport grid (deadlocks). Mirrors the forward
    _kernel_remote_store_barrier (fc2_combine.py:374)."""
    libshmem_device.barrier_all_vec()


@triton.jit
def _kernel_combine_fc1_bwd_pack(
    src_ptr,        # [*, H] row-major bf16: npu GEMM group output (src_row0=0) or hidden_buf
    gate_ptr,       # [M] bf16 gate grad, src_pos order
    pad_ptr,        # hidden_pad [M, H+GATE_PAD] local staging for the segment puts
    src_row0, pad_row0, rows, H_push,
    BLOCK_N: tl.constexpr, GATE_PAD: tl.constexpr,
):
    """Pack GEMM rows into the padded (H+GATE_PAD) stride the peer_mem contract
    uses, writing the gate grad as each row's trailing channel. The npu grouped
    GEMM cannot emit a strided output, so its [rows, H] result is flattened to
    the peer row stride here — one extra 2*M*(H+8)*2B pass that buys flat
    per-segment putmem transfers (vs the per-row dl.symm_at push)."""
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        off = tl.arange(0, BLOCK_N)
        for i in range(pid, rows, num_progs):
            i64 = i.to(tl.int64)
            sbase = (src_row0 + i64) * H_push
            dbase = (pad_row0 + i64) * (H_push + GATE_PAD)
            for ns in range(0, H_push, BLOCK_N):
                m = off < (H_push - ns)
                tl.store(pad_ptr + dbase + ns + off,
                         tl.load(src_ptr + sbase + ns + off, mask=m, other=0.0),
                         mask=m)
            tl.store(pad_ptr + dbase + H_push, tl.load(gate_ptr + pad_row0 + i64))


@triton.jit
def _kernel_combine_fc1_bwd_put_group(
    pad_ptr,        # hidden_pad [M, H+GATE_PAD] local
    peer_mem_ptr,   # symmetric [total_send, H+GATE_PAD] at heap offset 0
    seg_src0_ptr, seg_dst0_ptr, seg_rows_ptr, seg_rank_ptr,   # int32 [S]
    seg_first, seg_end, H_push,
    GATE_PAD: tl.constexpr,
):
    """Reverse-A2A by per-(expert, dst-rank) SEGMENT putmem — the forward
    _kernel_remote_put_transport_group pattern (fc2_combine.py:274) instead of
    per-row dl.symm_at stores. Rows of one (expert, dst) run are contiguous on
    both sides at the same H+GATE_PAD stride, so one putmem moves the whole run
    (~E*W large transfers instead of M 7KB row RMAs)."""
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        for s in range(seg_first + pid, seg_end, num_progs):
            src0 = tl.load(seg_src0_ptr + s).to(tl.int64)
            dst0 = tl.load(seg_dst0_ptr + s).to(tl.int64)
            rows = tl.load(seg_rows_ptr + s).to(tl.int64)
            rk = tl.load(seg_rank_ptr + s)
            libshmem_device.putmem(
                peer_mem_ptr + dst0 * (H_push + GATE_PAD),
                pad_ptr + src0 * (H_push + GATE_PAD),
                rows * (H_push + GATE_PAD) * 2,
                rk)


@triton.jit
def _kernel_combine_fc1_bwd_reduce(
    # Phase 3 (Vector): topk-sum reduce peer_mem -> grad_hidden, and gather the
    # packed gate channel -> grad_routing_weights [B*topk] (one value per
    # (token,slot), no sum).
    inv_sort_idxs_ptr,       # int64 [total_send]
    peer_mem_ptr,
    output_ptr,              # grad_hidden [B, H]
    grad_routing_ptr,        # grad_routing_weights [B*topk] out
    B, topk, H_push,
    stride_om, stride_on,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_progs = tl.num_programs(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        # Pure-AIV launch: distribute tokens directly over pid (forward
        # _kernel_local_topk_reduce pattern). Per-token topk sum order is fixed.
        ovr = tl.arange(0, BLOCK_N_PUSH)
        row_stride = H_push + GATE_PAD
        for ti in range(pid, B, num_progs):
            ti64 = ti.to(tl.int64)
            # gather the per-(token,slot) gate channel packed at row offset H_push
            for j in range(topk):
                fi = (ti * topk + j).to(tl.int64)
                sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                tl.store(grad_routing_ptr + fi,
                         tl.load(peer_mem_ptr + sp * row_stride + H_push))
            for ns in range(0, H_push, BLOCK_N_PUSH):
                mask = ovr < (H_push - ns)
                acc = tl.zeros((BLOCK_N_PUSH,), dtype=tl.float32)
                for j in range(topk):
                    fi = ti * topk + j
                    sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                    acc += tl.load(peer_mem_ptr + sp * row_stride + (ns + ovr),
                                   mask=mask, other=0.0)
                oo = ti64 * stride_om + (ns + ovr) * stride_on
                tl.store(output_ptr + oo, acc.to(output_ptr.dtype.element_ty), mask=mask)


def _combine_static_maps(saved):
    """Build the dy-independent combine push maps once and cache them on `saved`.
    Vectorized — no per-element Python writes (the old O(M) scalar-assign loop was
    the dominant backward cost: ~265 ms/call).

    A MoonEP ``saved_phys`` reuses the same arrival-buffer algebra: its
    ``splits_recv_list`` / ``inv_local`` are already physical semantics and the
    reverse all-to-all still returns each arrival row to its offset inside the
    source rank's plan send order, so only the group extent (``E``, the physical
    slot count) and the second (replica) weight table change."""
    cache = saved.get("_combine_cache")
    if cache is not None:
        return cache
    device = f"npu:{saved['ep_rank']}"
    pe = saved["ep_rank"]; W = saved["world_size"]; H = saved["hidden_dim"]
    ep_group = saved["ep_group"]; M = saved["M"]

    send_t = torch.tensor(saved["splits_send_list"], dtype=torch.int64, device=device)
    all_send = torch.stack(all_gather_list(send_t, ep_group))     # [W,W]: all_send[r][d] = r sends to d
    send_cum = torch.zeros_like(all_send)
    send_cum[:, 1:] = all_send[:, :-1].cumsum(dim=1)               # send_cum[d, me] = sum_{s<me} all_send[d, s]

    recv_t = torch.tensor(saved["splits_recv_list"], dtype=torch.int64, device=device)
    # write_rank[p] = d for p in dest-d segment  -> repeat_interleave(arange(W), recv_t)
    write_rank = torch.repeat_interleave(torch.arange(W, dtype=torch.int32, device=device), recv_t)
    # within-segment position of each p
    seg_start = torch.zeros(W, dtype=torch.int64, device=device)
    seg_start[1:] = recv_t[:-1].cumsum(0)
    within = torch.arange(M, dtype=torch.int64, device=device) - torch.repeat_interleave(seg_start, recv_t)
    base_per_dest = send_cum[:, pe].to(torch.int64)               # [W]: base offset per dest
    write_off = base_per_dest[write_rank.to(torch.int64)] + within  # [M]

    # inv_local / inv_sort are already in saved (computed in forward) — reuse, don't re-argsort.
    inv_local = saved["inv_local"].to(torch.int64).to(device).contiguous()
    inv_sort = saved["inv_sort"].to(torch.int64).to(device).contiguous()

    # Reindex the push maps to expert-major (src_pos) order so the push kernel
    # can iterate src_pos directly and be range-grouped by expert. inv_local
    # [out_pos] = src_pos, so its argsort local_sort_idxs[src_pos] = out_pos is
    # the inverse permutation; gathering write_rank/write_off through it indexes
    # by src_pos. The (hidden_buf row -> peer_mem slot) store SET is unchanged
    # -> bit-identical; only the iteration order changes, and each pushed
    # peer_mem row is disjoint so order is safe.
    local_sort_idxs = inv_local.argsort()
    write_rank_by_src = write_rank[local_sort_idxs].contiguous()
    write_off_by_src = write_off[local_sort_idxs].contiguous()

    num_tn = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_tm = int(saved["num_tiles_total"].item())
    # Stride view accepted as-is: the GEMM addresses the weight table through
    # we/wk/wn, and the native saved hands over gate_up_weight.transpose(1, 2)
    # (materializing it would copy ~4.6 GiB per step).  The replay's
    # torch.cat output is already contiguous, so that path is unchanged.
    fc1_combined = saved["fc1_combined"]
    if saved.get("use_moonep"):
        # Physical groups: home slots [0, epn) read the legacy home table, the
        # replica slots read the plan's packed replica gate/up table viewed as
        # the same [B, 2*ffn, H] (K, N) layout (a stride-only transpose).
        home_experts = int(saved["home_experts_per_rank"])
        experts_total = int(saved["physical_experts_per_rank"])
        replica_weight = saved["replica_gate_up"].transpose(1, 2)
    else:
        home_experts = saved["experts_per_rank"]
        experts_total = saved["experts_per_rank"]
        replica_weight = fc1_combined  # unused; keeps the launch code uniform
    cache = dict(
        weight=fc1_combined, replica_weight=replica_weight,
        home_experts=home_experts,
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        M=M, N=H, K=fc1_combined.shape[1], E=experts_total, num_tm=num_tm, num_tn=num_tn,
        inv_sort=inv_sort,
        write_rank_by_src=write_rank_by_src, write_off_by_src=write_off_by_src,
        split_size_cum_per_expert=saved["split_size_cum_per_expert"].to(device),
        H=H, B=saved["batch_size"], topk=saved["topk"],
        we=fc1_combined.stride(0), wk=fc1_combined.stride(1), wn=fc1_combined.stride(2),
        rwe=replica_weight.stride(0), rwk=replica_weight.stride(1),
        rwn=replica_weight.stride(2),
        stride_om=H, stride_on=1,
    )
    saved["_combine_cache"] = cache
    return cache


def _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate):
    """Build combine push maps (expert->home). Static maps are cached on `saved`;
    only the grad-dependent strides are added."""
    p = _combine_static_maps(saved)
    p = dict(p)  # shallow copy so we can add grad-dependent fields
    p["inp"] = grad_fc1_output.contiguous()
    p["grad_gate"] = grad_gate.contiguous()
    p["inp_stride_im"] = grad_fc1_output.stride(0)
    p["inp_stride_ik"] = grad_fc1_output.stride(1)
    return p


def _push_block():
    """Vector transport chunk width for the combine push/reduce kernels.
    Default 4096 covers one full H=3584 row in a single RMA store (4x fewer,
    4x larger transactions than the historical 1024 chunk). Env-tunable via
    MOE_COMBINE_PUSH_BN."""
    return int(os.environ.get("MOE_COMBINE_PUSH_BN", "4096"))


def _combine_gemm_backend():
    """MOE_COMBINE_GEMM_BACKEND: "triton" (default, persistent kernel) or "npu"
    (torch_npu.npu_grouped_matmul split-M). Measured on kimi-k3 w8 t8k: the
    triton kernel streams ~2x the HBM bytes the L0-tile cap forces and lands at
    39.5ms / ~145 TFLOPS serial; the CANN grouped GEMM runs the same reduction
    at 18ms / ~320 TFLOPS. Auto-falls back to triton for MoonEP physical saved
    (dual weight table is a stride-view) and non-contiguous weight tables."""
    return os.environ.get("MOE_COMBINE_GEMM_BACKEND", "triton").lower()


def _combine_push_backend():
    """MOE_COMBINE_PUSH: "rows" (default, per-row dl.symm_at) or "putmem"
    (per-(expert,dst) segment putmem, forward fc2_combine pattern). The rowwise
    push is latency-bound small-RMA (38ms measured); segments move the same
    bytes in <= E*W large transfers."""
    return os.environ.get("MOE_COMBINE_PUSH", "rows").lower()


def _combine_put_segments(saved, prep):
    """Per-(expert, dst-rank) contiguous put segments, cached on `saved`.

    Rows are expert-major (src_pos); write_rank_by_src is constant and
    write_off_by_src consecutive within each (expert, dst) run, so diffing
    (rank, off-step, expert) splits [0, M) into <= E*W flat segments that are
    contiguous on BOTH sides at the same H+GATE_PAD row stride. Built entirely
    on device (one cached nonzero sync); the expert-change term keeps segments
    inside one expert so they never straddle an expert-group boundary."""
    cache = saved.get("_combine_put_seg")
    if cache is not None:
        return cache
    device = f"npu:{saved['ep_rank']}"
    rank = prep["write_rank_by_src"]
    off = prep["write_off_by_src"].to(torch.int64)
    M = int(prep["M"])
    epr = int(prep["E"])
    expert_by_src = torch.repeat_interleave(
        torch.arange(epr, dtype=torch.int64, device=device),
        prep["expert_counts"].to(torch.int64))
    brk = torch.zeros(M, dtype=torch.bool, device=device)
    brk[1:] = ((rank[1:] != rank[:-1]) | (off[1:] != off[:-1] + 1)
               | (expert_by_src[1:] != expert_by_src[:-1]))
    idx = brk.nonzero(as_tuple=True)[0]
    src0 = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), idx])
    rows = torch.cat([src0[1:] - src0[:-1],
                      torch.tensor([M], dtype=torch.int64, device=device) - src0[-1:]])
    seg = dict(
        src0=src0.to(torch.int32).contiguous(),
        dst0=off[src0].to(torch.int32).contiguous(),
        rows=rows.to(torch.int32).contiguous(),
        rank=rank[src0].contiguous(),
        src0_host=src0.cpu().tolist(),
        n_seg=int(src0.numel()),
    )
    saved["_combine_put_seg"] = seg
    return seg


def _ensure_combine_put_workspace(saved, prep, device):
    """hidden_pad [M, H+GATE_PAD] staging buffer for the segment puts, cached on
    `saved` (persistent-workspace pattern — one allocation per saved layout)."""
    pad = saved.get("_combine_put_hidden_pad")
    if pad is None or pad.shape[0] < prep["M"] or pad.shape[1] != prep["H"] + GATE_PAD:
        pad = torch.empty(prep["M"], prep["H"] + GATE_PAD,
                          dtype=prep["inp"].dtype, device=device)
        saved["_combine_put_hidden_pad"] = pad
    return pad


def _launch_combine_fc1_bwd_gemm_npu(prep, e0, e1, r0, r1):
    """fc1 input-grad GEMM over experts [e0, e1) via torch_npu.npu_grouped_matmul.

    Flag combo group_type=0 (split-M) / split_item=2 / group_list_type=0 with a
    CUMULATIVE int64 group_list verified bit-exact against a per-expert matmul
    reference. Rows are expert-major, so the expert slice owns the contiguous
    row range [r0, r1); the group_list is re-based by r0. Output is a fresh
    [r1-r0, H] bf16 tensor (no out= on this op — the caller packs it into the
    padded stride, or copies into hidden_buf on the rows-push shim)."""
    ends = prep.get("_npu_row_ends")
    if ends is None:
        cum = prep["split_size_cum_per_expert"].to(torch.int64)
        if cum.numel() == int(prep["E"]) + 1:
            ends = cum[1:]          # inclusive cumsum: entries are the ends
        else:
            ends = cum + prep["expert_counts"].to(torch.int64)
        prep["_npu_row_ends"] = ends
    group_list = ends[e0:e1] - r0
    return torch_npu.npu_grouped_matmul(
        [prep["inp"][r0:r1]], [prep["weight"][e0:e1]],
        group_list=group_list, group_type=0, split_item=2, group_list_type=0)[0]


def _combine_fast_group_plan(saved, prep, group_experts):
    """Per-group (e0, e1, r0, r1, seg_first, seg_end) host ints for the fast
    pipeline, cached on `saved` (one .cpu() sync per saved layout, like
    _combine_bwd_group_bounds). Group size counts EXPERTS, so the segment
    expert-containment from _combine_put_segments keeps every putmem segment
    inside one group."""
    import bisect
    E = int(prep["E"])
    g = max(1, min(int(group_experts), E))
    key = ("v1", g)
    cache = saved.get("_combine_fast_groups")
    if cache is not None and cache[0] == key:
        return cache[1]
    row_starts = prep["split_size_cum_per_expert"].cpu().tolist()
    counts = prep["expert_counts"].cpu().tolist()
    ends = [row_starts[i] + counts[i] for i in range(E)]
    seg = _combine_put_segments(saved, prep)
    src0_host = seg["src0_host"]
    groups = []
    for gi in range((E + g - 1) // g):
        e0 = gi * g
        e1 = min(e0 + g, E)
        r0 = row_starts[e0]
        r1 = ends[e1 - 1]
        s0 = bisect.bisect_left(src0_host, r0)
        s1 = bisect.bisect_left(src0_host, r1)
        groups.append((e0, e1, r0, r1, s0, s1))
    saved["_combine_fast_groups"] = (key, groups)
    return groups


def _combine_bwd_group_bounds(prep, group_experts, block_m):
    """Per-group Cube M-tile range and push src_pos range. Expert e occupies
    M-tiles [cum_tiles[e], cum_tiles[e+1]) — cum_tiles derived from
    expert_counts (ceil(count/block_m), using the GEMM tile height, NOT the
    forward 64-row meta); and rows
    [split_size_cum_per_expert[e], split_size_cum_per_expert[e+1]). The Cube
    group writes the same hidden_buf rows its push group reads, so a per-group
    stream event fences the handoff. Returns (num_groups, first_tile_m[],
    last_tile_m[], first_src[], last_src[]) as Python int lists."""
    expert_counts = prep["expert_counts"].cpu().tolist()
    tiles_per_expert = [(c + block_m - 1) // block_m for c in expert_counts]
    cum_tiles = [0]
    for t in tiles_per_expert:
        cum_tiles.append(cum_tiles[-1] + t)            # cum_tiles[e] = first tile of expert e
    row_cum = prep["split_size_cum_per_expert"].cpu().tolist()
    E = prep["E"]
    g = max(1, min(int(group_experts), E))
    num_groups = (E + g - 1) // g
    ftm, ltm, fs, ls = [], [], [], []
    for gi in range(num_groups):
        e0 = gi * g
        e1 = min(e0 + g, E)
        ftm.append(cum_tiles[e0]); ltm.append(cum_tiles[e1])
        fs.append(row_cum[e0]); ls.append(row_cum[e1])
    return num_groups, ftm, ltm, fs, ls


def _ensure_combine_bwd_pipeline_runtime(saved, num_groups, device):
    """Create (once) and reuse the two NPU streams + events for the step4
    expert-group pipeline, cached on `saved` (the backward is function-based,
    no module instance). Mirrors forward _ensure_group_pipeline_runtime
    (ops/forward.py:234). Single-in-flight reuse is guaranteed by autograd."""
    if saved.get("_combine_bwd_cube_stream") is None:
        saved["_combine_bwd_cube_stream"] = torch.npu.Stream(device=device)
        saved["_combine_bwd_vector_stream"] = torch.npu.Stream(device=device)
        saved["_combine_bwd_start_event"] = torch.npu.Event()
        saved["_combine_bwd_done_event"] = torch.npu.Event()
        saved["_combine_bwd_group_events"] = []
    events = saved["_combine_bwd_group_events"]
    while len(events) < num_groups:
        events.append(torch.npu.Event())
    return (
        saved["_combine_bwd_cube_stream"],
        saved["_combine_bwd_vector_stream"],
        events,
        saved["_combine_bwd_start_event"],
        saved["_combine_bwd_done_event"],
    )


def _launch_combine_fc1_bwd_gemm_range(prep, tiles, hidden_buf, first_tile_m,
                                       last_tile_m, num_tiles_n, gemm_kwargs):
    """Issue the fc1 input-grad GEMM over M-tiles [first_tile_m, last_tile_m).

    Tiles are expert-major, so the MoonEP home/replica weight-table split is one
    range cut at ``tiles["tile_home_bound"]``: the home segment launches against
    ``prep["weight"]`` (legacy home fc1_combined) and the replica segment against
    ``prep["replica_weight"]`` re-based at ``prep["home_experts"]`` — the forward
    dispatch_fc1 dual-launch pattern. Without MoonEP the bound covers every tile
    and exactly one (home) launch runs, unchanged."""
    home_bound = tiles["tile_home_bound"]
    home_last = min(last_tile_m, home_bound)
    # MOE_COMBINE_GEMM_SCHED=grid (one task per program, guard-free — lets the
    # MTE->cube pipeliner engage; measured 31.6ms vs 39.5ms persistent on
    # kimi-k3 w8 t8k) or persistent (blk loop per core, historical schedule).
    grid_sched = os.environ.get("MOE_COMBINE_GEMM_SCHED", "grid") == "grid"
    even_k = prep["K"] % gemm_kwargs["BLOCK_K"] == 0
    if grid_sched:
        gkw = dict(gemm_kwargs, EVEN_K=even_k)
        if first_tile_m < home_last:
            _kernel_combine_fc1_bwd_gemm_grid[
                ((home_last - first_tile_m) * num_tiles_n, 1, 1)](
                prep["inp"], prep["weight"], hidden_buf,
                tiles["tile_expert"], tiles["tile_row0"], tiles["tile_rows"],
                prep["N"], prep["K"], num_tiles_n,
                prep["inp_stride_im"], prep["inp_stride_ik"],
                prep["we"], prep["wk"], prep["wn"],
                FIRST_TILE_M=first_tile_m, LAST_TILE_M=home_last,
                WEIGHT_EXPERT_BASE=0,
                **gkw)
        replica_first = max(first_tile_m, home_bound)
        if replica_first < last_tile_m:
            _kernel_combine_fc1_bwd_gemm_grid[
                ((last_tile_m - replica_first) * num_tiles_n, 1, 1)](
                prep["inp"], prep["replica_weight"], hidden_buf,
                tiles["tile_expert"], tiles["tile_row0"], tiles["tile_rows"],
                prep["N"], prep["K"], num_tiles_n,
                prep["inp_stride_im"], prep["inp_stride_ik"],
                prep["rwe"], prep["rwk"], prep["rwn"],
                FIRST_TILE_M=replica_first, LAST_TILE_M=last_tile_m,
                WEIGHT_EXPERT_BASE=prep["home_experts"],
                **gkw)
        return
    if first_tile_m < home_last:
        _kernel_combine_fc1_bwd_gemm_group[(ncore(), 1, 1)](
            prep["inp"], prep["weight"], hidden_buf,
            tiles["tile_expert"], tiles["tile_row0"], tiles["tile_rows"],
            prep["N"], prep["K"], num_tiles_n,
            prep["inp_stride_im"], prep["inp_stride_ik"],
            prep["we"], prep["wk"], prep["wn"],
            FIRST_TILE_M=first_tile_m, LAST_TILE_M=home_last,
            WEIGHT_EXPERT_BASE=0,
            **gemm_kwargs)
    replica_first = max(first_tile_m, home_bound)
    if replica_first < last_tile_m:
        _kernel_combine_fc1_bwd_gemm_group[(ncore(), 1, 1)](
            prep["inp"], prep["replica_weight"], hidden_buf,
            tiles["tile_expert"], tiles["tile_row0"], tiles["tile_rows"],
            prep["N"], prep["K"], num_tiles_n,
            prep["inp_stride_im"], prep["inp_stride_ik"],
            prep["rwe"], prep["rwk"], prep["rwn"],
            FIRST_TILE_M=replica_first, LAST_TILE_M=last_tile_m,
            WEIGHT_EXPERT_BASE=prep["home_experts"],
            **gemm_kwargs)


def _launch_combine_fc1_bwd_pipeline(prep, peer_mem, hidden_buf, output, grad_routing, saved,
                                     cube_tail=None):
    """Two-stream expert-group pipeline mirroring the forward FC2 remote-store
    pipeline (_launch_fc2_remote_store_pipeline, fc2_combine.py:570). Cube group
    g's fc1-input-grad GEMM overlaps Vector group (g-1)'s reverse-A2A push via
    per-group NPU events. The Cube->push handoff is intra-rank (hidden_buf is
    local) so a stream Event is enough; one barrier_all_vec after all groups
    fences the cross-rank peer_mem writes before the reduce runs on the caller
    stream. MOE_COMBINE_BWD_GROUP_EXPERTS tunes the group size (default 16).

    ``cube_tail`` (optional zero-arg callable) is enqueued on the cube stream
    right after the LAST GEMM group, so a pure-Cube op (the caller's fc1 wgrad)
    overlaps the pure-Vector push/barrier/reduce drain instead of serializing
    before the whole combine. Its return value is passed through; the caller
    stream waits a tail event before consuming it."""
    device = peer_mem.device
    _gbm, _gbn, _gbk, _gns = _combine_gemm_tile()
    _g_num_tn = (prep["N"] + _gbn - 1) // _gbn
    _gkw = dict(BLOCK_M=_gbm, BLOCK_N=_gbn, BLOCK_K=_gbk,
                NUM_STAGES=max(_gns, 1),
                num_warps=int(os.environ.get("MOE_COMBINE_GEMM_WARPS", "8")))
    group_env = int(os.environ.get("MOE_COMBINE_BWD_GROUP_EXPERTS", "16"))
    # The bounds derive only from the static expert_counts / row cumulative
    # offsets already cached in prep, but _combine_bwd_group_bounds pays a
    # .cpu().tolist() host sync — caching removes that per-iteration stall.
    # Keyed on (group_env, BM) because the M-tile ranges are in GEMM-tile units.
    cached_bounds = saved.get("_combine_bwd_bounds")
    if cached_bounds is None or cached_bounds[:2] != (group_env, _gbm):
        cached_bounds = (group_env, _gbm, _combine_bwd_group_bounds(prep, group_env, _gbm))
        saved["_combine_bwd_bounds"] = cached_bounds
    num_groups, ftm, ltm, fs, ls = cached_bounds[2]
    cube_stream, vector_stream, group_events, start_ev, done_ev = \
        _ensure_combine_bwd_pipeline_runtime(saved, num_groups, device)

    current_stream = torch.npu.current_stream(device)
    start_ev.record(current_stream)
    cube_stream.wait_event(start_ev)
    vector_stream.wait_event(start_ev)

    # group g's GEMM writes hidden_buf rows for experts [e0,e1) (disjoint across
    # groups); group g's push reads the same rows, gated by group_events[g]. So
    # while Vector pushes group g, Cube can run group g+1 — depth-1 pipeline at
    # expert-group granularity, exactly like the forward FC2 pipeline.
    tail_result = None
    tail_event = None
    tiles = _gemm_tile_maps(saved, _gbm)
    for g in range(num_groups):
        with torch.npu.stream(cube_stream):
            _launch_combine_fc1_bwd_gemm_range(
                prep, tiles, hidden_buf, ftm[g], ltm[g], _g_num_tn, _gkw)
            group_events[g].record(cube_stream)
            if cube_tail is not None and g == num_groups - 1:
                # After the last GEMM group the cube engine would idle while the
                # vector stream drains the remaining pushes + barrier + reduce;
                # fill that window with the caller's pure-Cube tail op. The last
                # group's push is already released by group_events[g], which was
                # recorded BEFORE the tail was enqueued.
                tail_result = cube_tail()
                tail_event = torch.npu.Event()
                tail_event.record(cube_stream)

        vector_stream.wait_event(group_events[g])
        with torch.npu.stream(vector_stream):
            _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
                hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
                peer_mem, prep["grad_gate"], prep["H"],
                FIRST_SRC_POS=fs[g], LAST_SRC_POS=ls[g],
                BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
                        num_warps=8)

    # All group pushes are ordered on vector_stream; one barrier_all_vec (AICore
    # grid, separate launch) fences their cross-rank RMA before the caller's
    # current stream is released and before the local reduce reads peer_mem.
    with torch.npu.stream(vector_stream):
        _kernel_combine_fc1_bwd_barrier[(ncore(), 1, 1)]()

    done_ev.record(vector_stream)
    current_stream.wait_event(done_ev)
    if tail_event is not None:
        current_stream.wait_event(tail_event)

    _kernel_combine_fc1_bwd_reduce[(nvec(), 1, 1)](
        prep["inv_sort"], peer_mem, output, grad_routing,
        prep["B"], prep["topk"], prep["H"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
        num_warps=8)
    # tail_result is None when no cube_tail was given.
    return tail_result


def _launch_combine_fc1_bwd_pipeline_fast(prep, peer_mem, hidden_buf, output,
                                          grad_routing, saved, fast_npu, fast_put,
                                          cube_tail=None):
    """Fast combine pipeline (MOE_COMBINE_GEMM_BACKEND=npu and/or
    MOE_COMBINE_PUSH=putmem) on the same two-stream expert-group choreography as
    _launch_combine_fc1_bwd_pipeline: cube group g produces rows [r0, r1)
    (npu grouped GEMM -> fresh [rows, H] tensor, or the legacy triton kernel ->
    hidden_buf); group_events[g] then releases the vector side, which packs the
    GEMM output + gate grad into hidden_pad's H+GATE_PAD stride and issues the
    group's putmem segments. With fast_npu but not fast_put, the npu output is
    copied into hidden_buf and the legacy rowwise push runs instead (A/B shim);
    with fast_put but not fast_npu, the triton GEMM's hidden_buf feeds the pack.
    The barrier / reduce / cube_tail choreography is identical to the legacy
    pipeline."""
    device = peer_mem.device
    group_env = int(os.environ.get("MOE_COMBINE_BWD_GROUP_EXPERTS", "16"))
    groups = _combine_fast_group_plan(saved, prep, group_env)
    num_groups = len(groups)
    seg = _combine_put_segments(saved, prep)
    pad = _ensure_combine_put_workspace(saved, prep, device) if fast_put else None
    # triton-GEMM branch (fast_put only, or fallback) shares the legacy tiles
    _gbm, _gbn, _gbk, _gns = _combine_gemm_tile()
    _g_num_tn = (prep["N"] + _gbn - 1) // _gbn
    _gkw = dict(BLOCK_M=_gbm, BLOCK_N=_gbn, BLOCK_K=_gbk, NUM_STAGES=max(_gns, 1),
                num_warps=int(os.environ.get("MOE_COMBINE_GEMM_WARPS", "8")))
    tiles = _gemm_tile_maps(saved, _gbm)
    cached_bounds = saved.get("_combine_bwd_bounds")
    if cached_bounds is None or cached_bounds[:2] != (group_env, _gbm):
        cached_bounds = (group_env, _gbm, _combine_bwd_group_bounds(prep, group_env, _gbm))
        saved["_combine_bwd_bounds"] = cached_bounds
    _, ftm, ltm, _fs, _ls = cached_bounds[2]

    cube_stream, vector_stream, group_events, start_ev, done_ev = \
        _ensure_combine_bwd_pipeline_runtime(saved, num_groups, device)
    current_stream = torch.npu.current_stream(device)
    start_ev.record(current_stream)
    cube_stream.wait_event(start_ev)
    vector_stream.wait_event(start_ev)

    last = num_groups - 1
    outs = []          # npu outputs: alive past enqueue via record_stream
    tail_result = None
    tail_event = None
    for gidx, (e0, e1, r0, r1, s0, s1) in enumerate(groups):
        with torch.npu.stream(cube_stream):
            out_g = None
            if fast_npu and r1 > r0:
                out_g = _launch_combine_fc1_bwd_gemm_npu(prep, e0, e1, r0, r1)
            else:
                _launch_combine_fc1_bwd_gemm_range(
                    prep, tiles, hidden_buf, ftm[gidx], ltm[gidx], _g_num_tn, _gkw)
            group_events[gidx].record(cube_stream)
            if cube_tail is not None and gidx == last:
                tail_result = cube_tail()
                tail_event = torch.npu.Event()
                tail_event.record(cube_stream)

        vector_stream.wait_event(group_events[gidx])
        with torch.npu.stream(vector_stream):
            if out_g is not None:
                out_g.record_stream(vector_stream)
                outs.append(out_g)
            if fast_put:
                src = out_g if out_g is not None else hidden_buf
                src_row0 = 0 if out_g is not None else r0
                _kernel_combine_fc1_bwd_pack[(nvec(), 1, 1)](
                    src, prep["grad_gate"], pad,
                    src_row0, r0, r1 - r0, prep["H"],
                    BLOCK_N=_push_block(), GATE_PAD=GATE_PAD, num_warps=8)
                _kernel_combine_fc1_bwd_put_group[(nvec(), 1, 1)](
                    pad, peer_mem,
                    seg["src0"], seg["dst0"], seg["rows"], seg["rank"],
                    s0, s1, prep["H"],
                    GATE_PAD=GATE_PAD, num_warps=8)
            elif out_g is not None:
                # npu GEMM + legacy rowwise push: land the rows in hidden_buf
                hidden_buf[r0:r1].copy_(out_g)
                _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
                    hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
                    peer_mem, prep["grad_gate"], prep["H"],
                    FIRST_SRC_POS=r0, LAST_SRC_POS=r1,
                    BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD, num_warps=8)
            else:
                # triton GEMM without putmem never takes the fast dispatch, but
                # keep the matrix total: legacy rowwise push over the group.
                _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
                    hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
                    peer_mem, prep["grad_gate"], prep["H"],
                    FIRST_SRC_POS=r0, LAST_SRC_POS=r1,
                    BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD, num_warps=8)

    with torch.npu.stream(vector_stream):
        _kernel_combine_fc1_bwd_barrier[(ncore(), 1, 1)]()

    done_ev.record(vector_stream)
    current_stream.wait_event(done_ev)
    if tail_event is not None:
        current_stream.wait_event(tail_event)

    _kernel_combine_fc1_bwd_reduce[(nvec(), 1, 1)](
        prep["inv_sort"], peer_mem, output, grad_routing,
        prep["B"], prep["topk"], prep["H"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
        num_warps=8)
    return tail_result


def _launch_combine_fc1_bwd_serial(prep, peer_mem, hidden_buf, output, grad_routing, saved,
                                   fast_npu=False, fast_put=False):
    """Diagnostic 3-phase SERIAL combine (no two-stream group overlap): GEMM,
    push+barrier, reduce run back-to-back on the caller stream. With
    MOE_COMBINE_PHASE_TIMING=1, record NPU events around each phase
    (combine_gemm / combine_push_barrier / combine_reduce), MAX-reduce across
    ranks, append to saved['_combine_phase_samples']. Used to split combine's
    wall time into its three components (the default pipeline overlaps phase 1
    and 2 across expert groups).

    fast_npu / fast_put run the MOE_COMBINE_GEMM_BACKEND=npu /
    MOE_COMBINE_PUSH=putmem variants over the whole expert range, so the
    breakdown splits the fast path exactly where the pipeline overlaps it
    (phase 2 includes the pack when putmem is on)."""
    _pt = os.environ.get("MOE_COMBINE_PHASE_TIMING") == "1"
    # MOE_COMBINE_PHASE_SPLIT=1 additionally fences push and barrier separately,
    # emitting 4 intervals (gemm/push/barrier/reduce) into
    # saved["_combine_phase4_samples"] to attribute the push+barrier slice,
    # while the 3-phase protocol samples stay unchanged.
    _ps = _pt and os.environ.get("MOE_COMBINE_PHASE_SPLIT") == "1"
    if _pt:
        _pev = [torch.npu.Event(enable_timing=True) for _ in range(5 if _ps else 4)]
        _pev[0].record()
    # phase 1: fc1 input-grad GEMM -> hidden_buf (Cube, all experts)
    out_all = None
    if fast_npu and prep["M"] > 0:
        out_all = _launch_combine_fc1_bwd_gemm_npu(
            prep, 0, int(prep["E"]), 0, int(prep["M"]))
    else:
        _gbm, _gbn, _gbk, _gns = _combine_gemm_tile()
        _g_num_tn = (prep["N"] + _gbn - 1) // _gbn
        _gkw = dict(BLOCK_M=_gbm, BLOCK_N=_gbn, BLOCK_K=_gbk,
                    NUM_STAGES=max(_gns, 1),
                    num_warps=int(os.environ.get("MOE_COMBINE_GEMM_WARPS", "8")))
        tiles = _gemm_tile_maps(saved, _gbm)
        _launch_combine_fc1_bwd_gemm_range(
            prep, tiles, hidden_buf, 0, tiles["num_tiles_m"], _g_num_tn, _gkw)
    if _pt:
        _pev[1].record()
    # phase 2: reverse-A2A push hidden_buf -> peer_mem (Vector, all rows) + cross-rank fence
    if fast_put:
        pad = _ensure_combine_put_workspace(saved, prep, peer_mem.device)
        seg = _combine_put_segments(saved, prep)
        src = out_all if out_all is not None else hidden_buf
        _kernel_combine_fc1_bwd_pack[(nvec(), 1, 1)](
            src, prep["grad_gate"], pad,
            0, 0, prep["M"], prep["H"],
            BLOCK_N=_push_block(), GATE_PAD=GATE_PAD, num_warps=8)
        _kernel_combine_fc1_bwd_put_group[(nvec(), 1, 1)](
            pad, peer_mem,
            seg["src0"], seg["dst0"], seg["rows"], seg["rank"],
            0, seg["n_seg"], prep["H"],
            GATE_PAD=GATE_PAD, num_warps=8)
    else:
        if out_all is not None:
            hidden_buf.copy_(out_all)
        _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
            hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
            peer_mem, prep["grad_gate"], prep["H"],
            FIRST_SRC_POS=0, LAST_SRC_POS=prep["M"],
            BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
            num_warps=8)
    _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
        hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
        peer_mem, prep["grad_gate"], prep["H"],
        FIRST_SRC_POS=0, LAST_SRC_POS=prep["M"],
        BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
        num_warps=8)
    # MOE_COMBINE_PHASE_SPLIT: fence push before the barrier (event 2).
    if _ps:
        _pev[2].record()
    _kernel_combine_fc1_bwd_barrier[(ncore(), 1, 1)]()
    if _pt:
        _pev[3 if _ps else 2].record()
    # phase 3: topk-sum reduce peer_mem -> grad_hidden (Vector)
    _kernel_combine_fc1_bwd_reduce[(nvec(), 1, 1)](
        prep["inv_sort"], peer_mem, output, grad_routing,
        prep["B"], prep["topk"], prep["H"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_N_PUSH=_push_block(), GATE_PAD=GATE_PAD,
        num_warps=8)
    if _pt:
        _last = 4 if _ps else 3
        _pev[_last].record()
        _pev[_last].synchronize()
        if _ps:
            _iv4 = [_pev[i].elapsed_time(_pev[i + 1]) for i in range(4)]
            _tmax4 = torch.tensor(_iv4, dtype=torch.float32, device=peer_mem.device)
            dist.all_reduce(_tmax4, op=dist.ReduceOp.MAX, group=saved["ep_group"])
            saved.setdefault("_combine_phase4_samples", []).append(
                [float(x) for x in _tmax4.cpu().tolist()])
        _iv = [
            _pev[0].elapsed_time(_pev[1]),          # gemm
            _pev[1].elapsed_time(_pev[_last - 1]),  # push+barrier
            _pev[_last - 1].elapsed_time(_pev[_last]),  # reduce
        ]
        _tmax = torch.tensor(_iv, dtype=torch.float32, device=peer_mem.device)
        dist.all_reduce(_tmax, op=dist.ReduceOp.MAX, group=saved["ep_group"])
        saved.setdefault("_combine_phase_samples", []).append(
            [float(x) for x in _tmax.cpu().tolist()])
    return output


def combine_fc1_bwd_triton(saved, grad_fc1_output, grad_gate, peer_mem, return_hidden=False,
                           cube_tail=None):
    """Step 4: returns (grad_hidden [B,H], grad_routing_weights [B,topk]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused from step 1,
    which has finished by now). The gate (routing-weight) grad rides step4's
    push+reduce as a packed peer_mem channel (row = [hidden | gate]), so there is
    no host HCCL all_to_all for the gate. If return_hidden, also returns
    hidden_buf (=grad_recv_hidden_sorted). If cube_tail is given (pipeline path
    only), it is enqueued on the cube stream after the last GEMM group and its
    return value is appended to the result for the caller to consume after the
    tail event."""
    prep = _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate)
    # GEMM writes every hidden_buf tile (meta covers all tokens); reduce writes
    # every output row -> empty, no zero-fill needed. hidden_buf is a plain
    # (non-symmetric) local workspace; peer_mem is the only symmetric buffer and
    # is fully overwritten by the push, so no host zero/barrier is needed.
    hidden_buf = torch.empty(prep["M"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    output = torch.empty(prep["B"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    grad_routing_flat = torch.empty(prep["B"] * prep["topk"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    tail_result = None
    # Fast-path gates: npu grouped GEMM needs a contiguous weight table (the
    # MoonEP replica view is stride-only); putmem transport is layout-identical
    # to the rowwise push, but the MoonEP serial-forced schedule keeps the
    # legacy kernels until Stage N3 wires the physical branch. return_hidden
    # callers want hidden_buf itself — the fast path stages rows elsewhere.
    fast_npu = (_combine_gemm_backend() == "npu"
                and not saved.get("use_moonep")
                and prep["weight"].is_contiguous())
    fast_put = (_combine_push_backend() == "putmem"
                and not saved.get("use_moonep"))
    if (fast_npu or fast_put) and not return_hidden:
        if os.environ.get("MOE_BWD_COMBINE_SERIAL") == "1":
            _launch_combine_fc1_bwd_serial(
                prep, peer_mem, hidden_buf, output, grad_routing_flat, saved,
                fast_npu=fast_npu, fast_put=fast_put)
        else:
            tail_result = _launch_combine_fc1_bwd_pipeline_fast(
                prep, peer_mem, hidden_buf, output, grad_routing_flat, saved,
                fast_npu, fast_put, cube_tail=cube_tail)
    elif os.environ.get("MOE_BWD_COMBINE_SERIAL") == "1":
        # diagnostic: 3-phase serial combine (no group overlap) + per-phase timing
        _launch_combine_fc1_bwd_serial(prep, peer_mem, hidden_buf, output, grad_routing_flat, saved)
    else:
        tail_result = _launch_combine_fc1_bwd_pipeline(
            prep, peer_mem, hidden_buf, output, grad_routing_flat, saved,
            cube_tail=cube_tail)
    grad_routing_weights = grad_routing_flat.view(prep["B"], prep["topk"])
    if cube_tail is not None:
        return output, grad_routing_weights, tail_result
    if return_hidden:
        return output, grad_routing_weights, hidden_buf
    return output, grad_routing_weights
