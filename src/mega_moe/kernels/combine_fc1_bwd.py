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


@triton.jit
def _kernel_combine_fc1_bwd_gemm_group(
    # Phase 1 (Cube): fc1 input-grad GEMM for tile_m in [FIRST_TILE_M, LAST_TILE_M)
    inp_ptr,                  # grad_fc1_output [M, 2*ffn]  (sorted)
    weight_ptr,               # fc1_combined [E, 2*ffn, H]  (K=2*ffn, N=H)
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H] out (LOCAL)
    meta_expert_ids_ptr, meta_split_cum_ptr, meta_tile_num_ptr, expert_counts_ptr,
    N, K, num_tiles_n,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn,
    FIRST_TILE_M: tl.constexpr, LAST_TILE_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        om = tl.arange(0, BLOCK_M)
        on_ = tl.arange(0, BLOCK_N)
        ok = tl.arange(0, BLOCK_K)
        group_tiles = LAST_TILE_M - FIRST_TILE_M
        total_tasks = group_tiles * num_tiles_n
        for task_id in range(pid, total_tasks, ncore):
            tile_m = FIRST_TILE_M + (task_id % group_tiles)
            tile_n = task_id // group_tiles
            expert_id = tl.load(meta_expert_ids_ptr + tile_m)
            cum_before = tl.load(meta_split_cum_ptr + tile_m)
            tile_in_exp = tl.load(meta_tile_num_ptr + tile_m)
            row_start = cum_before + tile_in_exp * BLOCK_M
            n_start = tile_n * BLOCK_N
            cnt = tl.load(expert_counts_ptr + expert_id)
            rem = cnt - tile_in_exp * BLOCK_M
            mm = om < rem
            mn = on_ < (N - n_start)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            wb = expert_id.to(tl.int64) * stride_we
            for ks in range(0, K, BLOCK_K):
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
    the dominant backward cost: ~265 ms/call)."""
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
    fc1_combined = saved["fc1_combined"].contiguous()
    cache = dict(
        weight=fc1_combined,
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        M=M, N=H, K=fc1_combined.shape[1], E=saved["experts_per_rank"], num_tn=num_tn,
        inv_sort=inv_sort,
        write_rank_by_src=write_rank_by_src, write_off_by_src=write_off_by_src,
        split_size_cum_per_expert=saved["split_size_cum_per_expert"].to(device),
        H=H, B=saved["batch_size"], topk=saved["topk"],
        we=fc1_combined.stride(0), wk=fc1_combined.stride(1), wn=fc1_combined.stride(2),
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


def _combine_bwd_group_bounds(prep, group_experts):
    """Per-group Cube M-tile range and push src_pos range. Expert e occupies
    M-tiles [cum_tiles[e], cum_tiles[e+1]) — cum_tiles derived from
    expert_counts (ceil(count/BLOCK_M)); saved meta_tile_num_cum is indexed
    per-TILE, not per-expert, so it cannot be used here — and rows
    [split_size_cum_per_expert[e], split_size_cum_per_expert[e+1]). The Cube
    group writes the same hidden_buf rows its push group reads, so a per-group
    stream event fences the handoff. Returns (num_groups, first_tile_m[],
    last_tile_m[], first_src[], last_src[]) as Python int lists."""
    expert_counts = prep["expert_counts"].cpu().tolist()
    tiles_per_expert = [(c + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M for c in expert_counts]
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


def _launch_combine_fc1_bwd_pipeline(prep, peer_mem, hidden_buf, output, grad_routing, saved):
    """Two-stream expert-group pipeline mirroring the forward FC2 remote-store
    pipeline (_launch_fc2_remote_store_pipeline, fc2_combine.py:570). Cube group
    g's fc1-input-grad GEMM overlaps Vector group (g-1)'s reverse-A2A push via
    per-group NPU events. The Cube->push handoff is intra-rank (hidden_buf is
    local) so a stream Event is enough; one barrier_all_vec after all groups
    fences the cross-rank peer_mem writes before the reduce runs on the caller
    stream. MOE_COMBINE_BWD_GROUP_EXPERTS tunes the group size (default 16)."""
    device = peer_mem.device
    group_env = int(os.environ.get("MOE_COMBINE_BWD_GROUP_EXPERTS", "16"))
    num_groups, ftm, ltm, fs, ls = _combine_bwd_group_bounds(prep, group_env)
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
    for g in range(num_groups):
        with torch.npu.stream(cube_stream):
            _kernel_combine_fc1_bwd_gemm_group[(ncore(), 1, 1)](
                prep["inp"], prep["weight"], hidden_buf,
                prep["meta_expert_ids"], prep["meta_split_cum"],
                prep["meta_tile_num"], prep["expert_counts"],
                prep["N"], prep["K"], prep["num_tn"],
                prep["inp_stride_im"], prep["inp_stride_ik"],
                prep["we"], prep["wk"], prep["wn"],
                FIRST_TILE_M=ftm[g], LAST_TILE_M=ltm[g],
                BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K,
                num_warps=8)
            group_events[g].record(cube_stream)

        vector_stream.wait_event(group_events[g])
        with torch.npu.stream(vector_stream):
            _kernel_combine_fc1_bwd_push_group[(nvec(), 1, 1)](
                hidden_buf, prep["write_rank_by_src"], prep["write_off_by_src"],
                peer_mem, prep["grad_gate"], prep["H"],
                FIRST_SRC_POS=fs[g], LAST_SRC_POS=ls[g],
                BLOCK_N_PUSH=1024, GATE_PAD=GATE_PAD,
                num_warps=8)

    # All group pushes are ordered on vector_stream; one barrier_all_vec (AICore
    # grid, separate launch) fences their cross-rank RMA before the caller's
    # current stream is released and before the local reduce reads peer_mem.
    with torch.npu.stream(vector_stream):
        _kernel_combine_fc1_bwd_barrier[(ncore(), 1, 1)]()

    done_ev.record(vector_stream)
    current_stream.wait_event(done_ev)

    _kernel_combine_fc1_bwd_reduce[(nvec(), 1, 1)](
        prep["inv_sort"], peer_mem, output, grad_routing,
        prep["B"], prep["topk"], prep["H"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_N_PUSH=1024, GATE_PAD=GATE_PAD,
        num_warps=8)
    return output


def combine_fc1_bwd_triton(saved, grad_fc1_output, grad_gate, peer_mem, return_hidden=False):
    """Step 4: returns (grad_hidden [B,H], grad_routing_weights [B,topk]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused from step 1,
    which has finished by now). The gate (routing-weight) grad rides step4's
    push+reduce as a packed peer_mem channel (row = [hidden | gate]), so there is
    no host HCCL all_to_all for the gate. If return_hidden, also returns
    hidden_buf (=grad_recv_hidden_sorted)."""
    prep = _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate)
    # GEMM writes every hidden_buf tile (meta covers all tokens); reduce writes
    # every output row -> empty, no zero-fill needed. hidden_buf is a plain
    # (non-symmetric) local workspace; peer_mem is the only symmetric buffer and
    # is fully overwritten by the push, so no host zero/barrier is needed.
    hidden_buf = torch.empty(prep["M"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    output = torch.empty(prep["B"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    grad_routing_flat = torch.empty(prep["B"] * prep["topk"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    _launch_combine_fc1_bwd_pipeline(prep, peer_mem, hidden_buf, output, grad_routing_flat, saved)
    grad_routing_weights = grad_routing_flat.view(prep["B"], prep["topk"])
    if return_hidden:
        return output, grad_routing_weights, hidden_buf
    return output, grad_routing_weights
