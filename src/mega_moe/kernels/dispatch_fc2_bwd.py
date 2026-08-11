# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  dispatch_fc2_bwd.py  —  step 1: dispatch-A2A(home->expert) + fc2 input-grad
#  ONE FUSED KERNEL: Phase 1 push (Vector) -> barrier_all -> Phase 2 fc2 input-grad
#  GEMM (Cube). The dual of combine_fc1_bwd's GEMM -> push -> reduce.
# ============================================================================

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

from .common import ncore, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K


# ONE FUSED KERNEL: Phase 1 push (Vector) -> barrier_all -> Phase 2 fc2 input-grad
# GEMM (Cube). This is the Ascend "AllGather+GEMM" fused pattern (see the
# AscendKernelWiki kernel-allgather-gemm page): a mixed Cube/Vector kernel that the
# compiler auto-splits into a vector_func (the dl.symm_at/tl.store push) and a
# cube_func (the tl.dot GEMM), with the single barrier_all() living in BOTH funcs
# so it fences the cross-rank push before any rank's GEMM gathers it.
#
# Why auto-split (no explicit al.scope) and not the explicit-scope form used by
# combine_fc1_bwd: every verified fused kernel on this backend either runs Cube
# FIRST then barrier then Vector (combine_fc1_bwd, forward _kernel_fc2_combine),
# or runs Vector+Cube CONCURRENTLY with signal/wait (forward _kernel_dispatch_fc1).
# There is NO verified instance of explicit `al.scope("vector") -> barrier_all ->
# al.scope("cube")` — and in fact the explicit-scope form of THIS kernel was tried
# and deadlocked (aicore execution timeout). The auto-split form (no al.scope)
# matches the wiki's verified AllGather+GEMM, where cube GEMM legitimately runs
# AFTER the barrier.
#
# Two fixes vs the original deadlocking fused kernel (commit 10347fe^):
#   1. sub_vec_id() gating on the push — a mixed kernel has 2 vector sub-cores;
#      without gating BOTH duplicate the dl.symm_at remote stores (the hazard
#      dispatch_fc1.py:187-191 documents), racing the comm engine. Only
#      sub_vec_id < PUSH_VECTOR_WORKERS may push.
#   2. auto-split instead of explicit al.scope — see above.
#
# Phase 1 dispatch push (home->expert): for each home row h (sort_idxs order),
#   write grad_combined_out_flat[h] -> dl.symm_at(peer_mem, h_dst_rank[h]) +
#   h_dst_off[h]*H  (vector, sub_vec_id gated)
# libshmem_device.barrier_all()
# Phase 2 fc2 input-grad GEMM (Cube): a = peer_mem[local_sort_idxs[row]]
#   (gather back to sorted order), b = fc2[e][H,ffn]; acc[M,ffn] = a @ b
@triton.jit
def kernel_dispatch_fc2_bwd(
    # ---- Phase 1: dispatch push (home->expert, Vector) ----
    gco_ptr,                  # grad_combined_out_flat [total_send, H] (home, sort_idxs order)
    h_dst_rank_ptr,           # int32 [total_send]
    h_dst_off_ptr,            # int64 [total_send]
    peer_mem_ptr,             # symmetric [total_recv, H] at HEAP OFFSET 0 (reused with step 4)
    total_send, H,
    stride_gm, stride_gk,     # gco strides (H, 1)
    # ---- Phase 2: fc2 input-grad GEMM (Cube) ----
    fc2_ptr,                  # [E, H, ffn]  (K=H, N=ffn, weight_reduce_last_dim=True)
    local_sort_idxs_ptr,      # int64 [total_recv]  (arrival -> sorted)
    meta_expert_ids_ptr, meta_split_cum_ptr, meta_tile_num_ptr, expert_counts_ptr,
    M, N, K, E, num_tiles_m, num_tiles_n,
    stride_we, stride_wk, stride_wn,   # fc2 [E,H,ffn]: (H*ffn, ffn, 1)
    # ---- output ----
    out_ptr,                  # grad_swiglu [M, ffn]
    # ---- constexpr ----
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
    PUSH_VECTOR_WORKERS: tl.constexpr,
):
    """Fused dispatch push (home->expert) + fc2 input-grad GEMM in one launch.
    Phase 1 push + device barrier_all make every rank's pushes globally visible
    before Phase 2 GEMM gathers them. No explicit al.scope: the compiler auto-
    splits this mixed kernel (Vector push | Cube GEMM) and places the single
    barrier_all in both halves — the verified AllGather+GEMM codegen path."""
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)

    # ===== Phase 1: dispatch push (home -> expert) -> peer_mem (VECTOR) =====
    # sub_vec_id gating is mandatory in a mixed kernel: only one vector sub-core
    # owns the dl.symm_at remote stores so the cube side cannot duplicate them.
    # PUSH_VECTOR_WORKERS=1 -> push_worker_id=pid, num_push_workers=ncore, i.e.
    # identical work distribution to the proven split push kernel.
    push_sub_id = sub_vec_id().to(tl.int32)
    if push_sub_id < PUSH_VECTOR_WORKERS:
        push_worker_id = pid * PUSH_VECTOR_WORKERS + push_sub_id
        num_push_workers = ncore * PUSH_VECTOR_WORKERS
        ovh = tl.arange(0, BLOCK_H_PUSH)
        for h in range(push_worker_id, total_send, num_push_workers):
            h64 = h.to(tl.int64)
            dst_rank = tl.load(h_dst_rank_ptr + h)
            dst_off = tl.load(h_dst_off_ptr + h64).to(tl.int64)
            rp = dl.symm_at(peer_mem_ptr, dst_rank)
            for ns in range(0, H, BLOCK_H_PUSH):
                mask = ovh < (H - ns)
                ro = h64 * stride_gm + (ns + ovh) * stride_gk
                val = tl.load(gco_ptr + ro, mask=mask, other=0.0)
                ro2 = dst_off * H + (ns + ovh)
                tl.store(rp + ro2, val, mask=mask)

    libshmem_device.barrier_all()

    # ===== Phase 2: fc2 input-grad GEMM (gather via local_sort_idxs) (CUBE) =====
    # acc[M, ffn] = peer_mem[local_sort_idxs[row]] @ fc2[e]. Pure Cube (tl.dot) —
    # the compiler routes it to the cube_func; reads peer_mem, now fully populated
    # + globally visible after the barrier.
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    total_tasks = num_tiles_m * num_tiles_n
    for task_id in range(pid, total_tasks, ncore):
        tile_m = task_id % num_tiles_m
        tile_n = task_id // num_tiles_m
        expert_id = tl.load(meta_expert_ids_ptr + tile_m)
        cum_before = tl.load(meta_split_cum_ptr + tile_m)
        tile_in_exp = tl.load(meta_tile_num_ptr + tile_m)
        row_start = cum_before + tile_in_exp * BLOCK_M
        n_start = tile_n * BLOCK_N
        cnt = tl.load(expert_counts_ptr + expert_id)
        rem = cnt - tile_in_exp * BLOCK_M
        mm = om < rem
        mn = on_ < (N - n_start)
        # gather: sorted row (row_start+om) -> arrival row local_sort_idxs[row_start+om]
        rows = tl.load(local_sort_idxs_ptr + (row_start + om).to(tl.int64), mask=om < BLOCK_M, other=0).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        wb = expert_id.to(tl.int64) * stride_we
        for ks in range(0, K, BLOCK_K):
            mk = ok < (K - ks)
            ao = rows[:, None] * H + (ks + ok[None, :])            # peer_mem[arrival_row, H-col]
            a = tl.load(peer_mem_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
            bo = (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
            b = tl.load(fc2_ptr + wb + bo, mask=mk[:, None] & mn[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = (row_start + om[:, None]) * N + (n_start + on_[None, :])
        tl.store(out_ptr + co, acc.to(out_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])


def _dispatch_static_maps(saved):
    """Build the dy-independent dispatch push maps once and cache them on `saved`."""
    cache = saved.get("_dispatch_cache")
    if cache is not None:
        return cache
    device = f"npu:{saved['ep_rank']}"
    pe = saved["ep_rank"]; W = saved["world_size"]; H = saved["hidden_dim"]
    ep_group = saved["ep_group"]; total_send = saved["total_send"]

    send_t = torch.tensor(saved["splits_send_list"], dtype=torch.int64, device=device)  # [W] own per-dest
    all_send = torch.stack(all_gather_list(send_t, ep_group))                           # [W,W]: all_send[s,d]
    cum_src = all_send.cumsum(dim=0)                       # cum_src[pe,d] = sum_{s<=pe} all_send[s,d]
    offset = cum_src - all_send                            # offset[pe,d] = sum_{s<pe} all_send[s,d]
    pe_offset_for_d = offset[pe]                           # [W]

    own_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                           send_t.cumsum(0)[:-1]])         # own_start[d] = sum_{d'<d} send_t[d']
    h_dst_rank = torch.repeat_interleave(torch.arange(W, dtype=torch.int32, device=device), send_t)
    h_pos = torch.arange(total_send, dtype=torch.int64, device=device) - own_start[h_dst_rank.to(torch.int64)]
    h_dst_off = pe_offset_for_d[h_dst_rank.to(torch.int64)] + h_pos

    num_tiles_m = int(saved["num_tiles_total"].item())
    num_tiles_n = (saved["ffn_dim"] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    # ---- expert-major metadata for signal/wait variant (mirror forward dispatch_fc1) ----
    EPR = saved["experts_per_rank"]
    flat = saved["selected_experts"].to(torch.int64).to(device).reshape(-1)   # [total_send] global expert
    send_counts_re = torch.bincount(flat, minlength=W * EPR).to(torch.int32).reshape(W, EPR)  # [W,EPR]
    all_send_e = torch.stack(all_gather_list(send_counts_re.reshape(-1), ep_group)).reshape(W, W, EPR)
    recv_counts_re = all_send_e[:, pe, :]                                       # [W, EPR] (source, le)
    recv_per_expert = recv_counts_re.sum(0).to(torch.int32)                     # [EPR]
    recv_expert_offs = torch.zeros(EPR + 1, dtype=torch.int32, device=device)
    recv_expert_offs[1:] = recv_per_expert.cumsum(0).to(torch.int32)            # [EPR+1]
    bwd_expert_sort = torch.argsort(flat.to(torch.float32), stable=True).to(torch.int32)  # [total_send]
    send_counts_flat = send_counts_re.reshape(-1)
    send_bucket_starts = torch.zeros(W * EPR + 1, dtype=torch.int32, device=device)
    send_bucket_starts[1:] = send_counts_flat.cumsum(0).to(torch.int32)
    dst_total = all_send_e.sum(0)                                               # [W, EPR]
    expert_base = torch.zeros_like(dst_total)
    expert_base[:, 1:] = dst_total.cumsum(1)[:, :-1]
    source_prefix = all_send_e[:pe].sum(0).to(torch.int32) if pe > 0 else \
        torch.zeros(W, EPR, dtype=torch.int32, device=device)
    send_bucket_dst_starts = (expert_base.to(torch.int32) + source_prefix).reshape(-1).contiguous()

    # sanity: expert-major metadata must match the rank-major counts already in saved
    assert int(send_counts_re.sum().item()) == total_send, \
        f"send_counts_re sum {send_counts_re.sum().item()} != total_send {total_send}"
    assert int(recv_per_expert.sum().item()) == saved["total_recv"], \
        f"recv_per_expert sum {recv_per_expert.sum().item()} != total_recv {saved['total_recv']}"

    cache = dict(
        h_dst_rank=h_dst_rank, h_dst_off=h_dst_off,
        M=saved["M"], N=saved["ffn_dim"], K=H, E=EPR,
        num_tm=num_tiles_m, num_tn=num_tiles_n,
        fc2=saved["fc2"].contiguous(), local_sort_idxs=saved["local_sort_idxs"].to(torch.int64).to(device),
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        total_send=total_send, H=H, total_recv=saved["total_recv"], sort_idxs=saved["sort_idxs"].to(device),
        send_start_per_dst=own_start, send_count_per_dst=send_t, dst_off_per_dst=pe_offset_for_d,
        # expert-major signal/wait metadata
        send_counts_re=send_counts_flat.contiguous(), send_bucket_starts=send_bucket_starts,
        send_bucket_dst_starts=send_bucket_dst_starts,
        recv_per_expert=recv_per_expert, recv_expert_offs=recv_expert_offs,
        bwd_expert_sort=bwd_expert_sort,
    )
    saved["_dispatch_cache"] = cache
    return cache


def _prepare_dispatch_fc2_bwd(saved, dy):
    """Build dispatch push maps (home->expert), the dual of 06's combine maps.
    Static maps are cached on `saved`; only the dy-dependent gco is built per call."""
    p = _dispatch_static_maps(saved)
    topk = saved["topk"]
    # grad_combined_out_flat = dy.repeat_interleave(topk)[sort_idxs]  (home, sort_idxs order)
    grad_combined_out_flat = dy.repeat_interleave(topk, dim=0)[p["bwd_expert_sort"].to(torch.int64)].contiguous()
    p = dict(p)
    p["gco"] = grad_combined_out_flat
    return p


def _launch_dispatch_fc2_bwd(prep, peer_mem, out):
    # Single fused launch: dispatch push (vector) + fc2 input-grad GEMM (cube),
    # fenced by the in-kernel barrier_all — it makes every rank's pushes globally
    # visible before the GEMM gathers them. No host sync (same rationale as
    # combine_fc1_bwd's single fused launch). The GEMM writes every (m,n) tile
    # (meta covers all tokens), so `out` needs no zero-fill.
    kernel_dispatch_fc2_bwd[(ncore(), 1, 1)](
        prep["gco"], prep["h_dst_rank"], prep["h_dst_off"], peer_mem,
        prep["total_send"], prep["H"],
        prep["gco"].stride(0), prep["gco"].stride(1),
        prep["fc2"], prep["local_sort_idxs"],
        prep["meta_expert_ids"], prep["meta_split_cum"], prep["meta_tile_num"], prep["expert_counts"],
        prep["M"], prep["N"], prep["K"], prep["E"], prep["num_tm"], prep["num_tn"],
        prep["fc2"].stride(0), prep["fc2"].stride(1), prep["fc2"].stride(2),
        out,
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K,
        BLOCK_H_PUSH=1024, PUSH_VECTOR_WORKERS=2, num_warps=8)
    return out


# ============================================================================
# signal/wait variant (expert-major, mirror forward dispatch_fc1): putmem push
# by (dst,expert) bucket -> peer_mem expert-major CONTIGUOUS -> consumer reads
# CONTIGUOUS (NO local_sort gather) + dl.wait/consume_token per expert. This is
# the only form where consume_token (required to pair fence/signal_op, else
# undefined _mlir_ciface_*.vector) is compatible with the GEMM load: contiguous
# load dependency tracking works; gather did not. Enable with MOE_BWD_SIGNAL=1.
# ============================================================================
@triton.jit
def _dispatch_direct_grad_buckets(
    pid, num_cores,
    gco_staging_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    H: tl.constexpr, stride_gm,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
):
    """Mirror forward _dispatch_direct_expert_buckets (dispatch_fc1.py:659-707).
    Expert-major work sharing so every dst receives early expert buckets
    concurrently. No routing-weight putmem (reverse step1 doesn't carry it)."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    for work_id in range(pid, num_tasks, num_cores):
        dst_rank = work_id % WORLD_SIZE
        expert_id = work_id // WORLD_SIZE
        task_id = dst_rank * EXPERTS_PER_RANK + expert_id
        task_start = tl.load(send_bucket_starts_ptr + task_id)
        task_count = tl.load(send_counts_re_ptr + task_id)
        task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
        if task_count > 0:
            libshmem_device.putmem(
                peer_mem_ptr + task_dst_start * H,
                gco_staging_ptr + task_start * stride_gm,
                task_count * H * 2, dst_rank)
            libshmem_device.fence()
        libshmem_device.signal_op(
            signal_mem_ptr + expert_id * 16, 1,
            libshmem_device.ACLSHMEM_SIGNAL_ADD, dst_rank)


@triton.jit
def _fc2_bwd_gemm_one_mn_tile(
    input_ptr, weight_ptr, output_ptr,
    expert_id, m_off, m_size, n_tile, N, K,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    dtype: tl.constexpr,
):
    """fc2 input-grad GEMM one tile. A = peer_mem[m_off, H] CONTIGUOUS (no gather),
    B = fc2[expert][H, ffn], out = grad_swiglu[m_off, ffn]."""
    if m_size > 0:
        om = tl.arange(0, BLOCK_M); on_ = tl.arange(0, BLOCK_N); ok = tl.arange(0, BLOCK_K)
        m_offs = m_off + om; m_mask = om < m_size
        n_offs = n_tile * BLOCK_N + on_; n_mask = n_offs < N
        wb = expert_id.to(tl.int64) * stride_we
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ks in range(0, K, BLOCK_K):
            k_offs = ks + ok; k_mask = k_offs < K
            ao = m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
            a = tl.load(input_ptr + ao, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            bo = k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            b = tl.load(weight_ptr + wb + bo, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
        tl.store(output_ptr + co, acc.to(output_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _triton_grouped_gemm_fc2_bwd_expert_wait(
    pid, ncore,
    peer_mem_ptr, signal_mem_ptr, fc2_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K, stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr, dtype: tl.constexpr,
):
    """Mirror _triton_grouped_gemm_expert_tiles_wait (dispatch_fc1.py:364-409).
    Per-expert dl.wait + consume_token (pairs fence/signal_op -> no .vector
    undefined) + CONTIGUOUS GEMM (no local_sort_idxs gather)."""
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_m_tiles = tl.cdiv(expert_size, BLOCK_M)
        num_expert_tasks = num_m_tiles * num_n_tiles
        if num_expert_tasks > 0:
            token = dl.wait(
                signal_mem_ptr + expert_id * 16, 1, "gpu", "acquire",
                waitValue=signal_epoch * WORLD_SIZE)
            ready_input_ptr = dl.consume_token(peer_mem_ptr, token)
            for t in range(pid, num_expert_tasks, ncore):
                m_tile = t // num_n_tiles; n_tile = t % num_n_tiles
                m_off = expert_off + m_tile * BLOCK_M
                m_size = tl.minimum(expert_size - m_tile * BLOCK_M, BLOCK_M)
                _fc2_bwd_gemm_one_mn_tile(
                    ready_input_ptr, fc2_ptr, output_ptr,
                    expert_id, m_off, m_size, n_tile, N, K,
                    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
                    BLOCK_M, BLOCK_N, BLOCK_K, dtype)


@triton.jit(do_not_specialize=["signal_epoch"])
def kernel_dispatch_fc2_bwd_signal(
    gco_staging_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    total_send, H, stride_gm,
    fc2_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr,
    N, K, stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    signal_epoch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
    PUSH_VECTOR_WORKERS: tl.constexpr,
):
    """#3 barrier probe: auto-split (NO al.scope) + putmem push (expert-major) +
    barrier_all + CONTIGUOUS GEMM. Isolates metadata/putmem from signal/wait."""
    pid = tl.program_id(axis=0)
    ncore_v = tl.num_programs(axis=0)
    push_sub_id = sub_vec_id().to(tl.int32)
    if push_sub_id < PUSH_VECTOR_WORKERS:
        push_worker_id = pid * PUSH_VECTOR_WORKERS + push_sub_id
        num_push_workers = ncore_v * PUSH_VECTOR_WORKERS
        num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
        for work_id in range(push_worker_id, num_tasks, num_push_workers):
            dst_rank = work_id % WORLD_SIZE
            expert_id = work_id // WORLD_SIZE
            task_id = dst_rank * EXPERTS_PER_RANK + expert_id
            task_count = tl.load(send_counts_re_ptr + task_id)
            if task_count > 0:
                task_start = tl.load(send_bucket_starts_ptr + task_id)
                task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
                libshmem_device.putmem(
                    peer_mem_ptr + task_dst_start * H,
                    gco_staging_ptr + task_start * stride_gm,
                    task_count * H * 2, dst_rank)
    libshmem_device.barrier_all()
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_m_tiles = tl.cdiv(expert_size, BLOCK_M)
        num_expert_tasks = num_m_tiles * num_n_tiles
        for t in range(pid, num_expert_tasks, ncore_v):
            m_tile = t // num_n_tiles
            n_tile = t % num_n_tiles
            m_off = expert_off + m_tile * BLOCK_M
            m_size = tl.minimum(expert_size - m_tile * BLOCK_M, BLOCK_M)
            if m_size > 0:
                mm = om < m_size
                n_start = n_tile * BLOCK_N
                mn = on_ < (N - n_start)
                wb = expert_id.to(tl.int64) * stride_we
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for ks in range(0, K, BLOCK_K):
                    mk = ok < (K - ks)
                    ao = (m_off + om)[:, None] * H + (ks + ok[None, :])
                    a = tl.load(peer_mem_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
                    bo = (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
                    b = tl.load(fc2_ptr + wb + bo, mask=mk[:, None] & mn[None, :], other=0.0)
                    acc += tl.dot(a, b)
                co = (m_off + om)[:, None] * N + (n_start + on_[None, :])
                tl.store(output_ptr + co, acc.to(output_ptr.dtype.element_ty),
                         mask=mm[:, None] & mn[None, :])


def _launch_dispatch_fc2_bwd_signal(prep, peer_mem, out, saved):
    W = saved["world_size"]
    fc2 = prep["fc2"]; H = prep["H"]; N = prep["N"]; K = prep["K"]
    kernel_dispatch_fc2_bwd_signal[(ncore(), 1, 1)](
        prep["gco"], peer_mem, None,
        prep["send_bucket_starts"], prep["send_counts_re"], prep["send_bucket_dst_starts"],
        prep["total_send"], H, prep["gco"].stride(0),
        fc2, out,
        prep["recv_per_expert"], prep["recv_expert_offs"],
        N, K, H, 1, fc2.stride(0), fc2.stride(1), fc2.stride(2), N, 1, 0,
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K,
        WORLD_SIZE=W, EXPERTS_PER_RANK=prep["E"], PUSH_VECTOR_WORKERS=2, num_warps=8)
    return out


def dispatch_fc2_bwd_triton(saved, dy, peer_mem, use_signal=None):
    """Step 1: returns (grad_swiglu [M,ffn], grad_fc2_out_sorted [M,H]).
    MOE_BWD_SIGNAL=1 selects the expert-major signal/wait variant (putmem +
    dl.wait/consume_token, no barrier_all, no gather); default is the auto-split
    + barrier_all fused kernel."""
    prep = _prepare_dispatch_fc2_bwd(saved, dy)
    out = torch.empty(prep["M"], prep["N"], dtype=dy.dtype, device=dy.device)
    if use_signal is None:
        use_signal = os.environ.get("MOE_BWD_SIGNAL") == "1"
    if use_signal:
        _launch_dispatch_fc2_bwd_signal(prep, peer_mem, out, saved)
        # expert-major peer_mem IS the sorted layout -> identity view (no gather)
        grad_fc2_out_sorted = peer_mem.view(-1)[:prep["total_recv"] * prep["H"]].view(
            prep["total_recv"], prep["H"]).contiguous()
    else:
        _launch_dispatch_fc2_bwd(prep, peer_mem, out)
        peer_view = peer_mem.view(-1)[:prep["total_recv"] * prep["H"]].view(prep["total_recv"], prep["H"])
        grad_fc2_out_sorted = peer_view[prep["local_sort_idxs"]].contiguous()
    return out, grad_fc2_out_sorted
