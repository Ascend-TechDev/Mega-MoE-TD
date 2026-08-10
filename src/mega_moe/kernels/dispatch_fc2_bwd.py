# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  dispatch_fc2_bwd.py  —  step 1: dispatch-A2A(home->expert) + fc2 input-grad
#  ONE FUSED KERNEL: Phase 1 push (Vector) -> barrier_all -> Phase 2 fc2 input-grad
#  GEMM (Cube). The dual of combine_fc1_bwd's GEMM -> push -> reduce.
# ============================================================================

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
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
    cache = dict(
        h_dst_rank=h_dst_rank, h_dst_off=h_dst_off,
        M=saved["M"], N=saved["ffn_dim"], K=H, E=saved["experts_per_rank"],
        num_tm=num_tiles_m, num_tn=num_tiles_n,
        fc2=saved["fc2"].contiguous(), local_sort_idxs=saved["local_sort_idxs"].to(torch.int64).to(device),
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        total_send=total_send, H=H, total_recv=saved["total_recv"], sort_idxs=saved["sort_idxs"].to(device),
    )
    saved["_dispatch_cache"] = cache
    return cache


def _prepare_dispatch_fc2_bwd(saved, dy):
    """Build dispatch push maps (home->expert), the dual of 06's combine maps.
    Static maps are cached on `saved`; only the dy-dependent gco is built per call."""
    p = _dispatch_static_maps(saved)
    topk = saved["topk"]
    # grad_combined_out_flat = dy.repeat_interleave(topk)[sort_idxs]  (home, sort_idxs order)
    grad_combined_out_flat = dy.repeat_interleave(topk, dim=0)[p["sort_idxs"]].contiguous()
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


def dispatch_fc2_bwd_triton(saved, dy, peer_mem):
    """Step 1: returns (grad_swiglu [M,ffn], grad_fc2_out_sorted [M,H]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused with step 4)."""
    prep = _prepare_dispatch_fc2_bwd(saved, dy)
    # GEMM writes every (m,n) tile (meta covers all tokens) -> empty, no zero-fill
    out = torch.empty(prep["M"], prep["N"], dtype=dy.dtype, device=dy.device)
    # peer_mem is fully overwritten by the push phase (every arrival row read by
    # the GEMM is written by some sender's push); the fused kernel's barrier_all
    # syncs push->GEMM, so no host zero/barrier is needed here.
    _launch_dispatch_fc2_bwd(prep, peer_mem, out)
    # grad_fc2_out_sorted = peer_mem (arrival) gathered by local_sort_idxs
    peer_view = peer_mem.view(-1)[:prep["total_recv"] * prep["H"]].view(prep["total_recv"], prep["H"])
    grad_fc2_out_sorted = peer_view[prep["local_sort_idxs"]].contiguous()
    return out, grad_fc2_out_sorted
