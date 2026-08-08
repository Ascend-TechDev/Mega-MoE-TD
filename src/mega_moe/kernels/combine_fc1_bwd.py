# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  combine_fc1_bwd.py  —  step 4: fc1 input-grad + reverse-A2A(expert->home)
#  + gate-grad  (GEMM -> push -> reduce, isomorphic to 06 + gate channel)
# ============================================================================

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


# ONE FUSED KERNEL with three phases mirroring the forward _kernel_fc2_combine
# (fc2_combine.py:246-461), which fuses the IDENTICAL structure (Cube GEMM ->
# barrier_all -> Vector push -> barrier_all -> Vector reduce -> barrier_all) in a
# single verified-working launch.
#
# step 4 is Cube-write(hidden_buf) -> barrier -> Vector-read, the direction that
# hung in the old fused attempt. The forward proves the fix is the scope
# discipline the old split kernels lacked: each phase wrapped in
# `al.scope(core_mode=..., disable_auto_sync=True)`, and the SHMEM comm gated to
# a single vector sub-core via `sub_vec_id() < WORKERS` so the Cube side cannot
# duplicate the dl.symm_at stores (dispatch_fc1.py:187-191 documents exactly
# this hazard). disable_auto_sync=True + the in-kernel barrier_all() is the
# complete fence — no membar, no host sync.
#
# Phase 1 fc1 input-grad GEMM: acc[M,H] = grad_fc1_output @ fc1_combined[e]
#   -> local hidden_buf (cube scope)
# libshmem_device.barrier_all()
# Phase 2 reverse-A2A push (expert->home): hidden_buf[inv_local[out]] -> peer_mem
#   (vector scope, sub_vec_id gated)
# libshmem_device.barrier_all()
# Phase 3 reduce: grad_hidden[b] = sum_j peer_mem[inv_sort[b*topk+j]]  (06 Phase3)
#   (vector scope, sub_vec_id gated)
# libshmem_device.barrier_all()
# (gate channel stays host-side: _gate_bwd_host — dl.symm_at only resolves at
#  heap offset 0, so no second symmetric buffer for the scalar gate.)
@triton.jit
def kernel_combine_fc1_bwd(
    # ---- Phase 1: fc1 input-grad GEMM (Cube) ----
    inp_ptr,                  # grad_fc1_output [M, 2*ffn]  (sorted)
    weight_ptr,               # fc1_combined [E, 2*ffn, H]  (K=2*ffn, N=H)
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H] out (LOCAL workspace)
    meta_expert_ids_ptr, meta_split_cum_ptr, meta_tile_num_ptr, expert_counts_ptr,
    M, N, K, E, num_tiles_m, num_tiles_n,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn,
    # ---- Phase 2: reverse-A2A push (Vector, expert->home) ----
    inv_local_sort_idxs_ptr, write_rank_ptr, write_off_ptr,
    peer_mem_ptr,             # symmetric [total_send, H] at HEAP OFFSET 0
    total_recv, H_push,
    # ---- Phase 3: topk-sum reduce (Vector) ----
    inv_sort_idxs_ptr,        # int64 [total_send]
    output_ptr,               # grad_hidden [B, H]
    B, topk, total_send,
    stride_om, stride_on,
    # ---- constexpr ----
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_N_PUSH: tl.constexpr,
    REVERSE_VECTOR_WORKERS: tl.constexpr,
    REDUCE_VECTOR_WORKERS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)

    # ===== Phase 1: fc1 input-grad GEMM -> hidden_buf (PURE CUBE) =====
    with al.scope(core_mode="cube", disable_auto_sync=True):
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
            tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])

    libshmem_device.barrier_all()

    # ===== Phase 2: reverse-A2A push (expert->home): hidden_buf -> peer_mem (VECTOR) =====
    # REVERSE_VECTOR_WORKERS=1 -> push_worker_id=pid, num_push_workers=ncore: identical
    # work distribution to the proven split push kernel, now scope-wrapped + single
    # sub-vec-owned so the cube side cannot duplicate the SHMEM stores.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        push_sub_id = sub_vec_id().to(tl.int32)
        if push_sub_id < REVERSE_VECTOR_WORKERS:
            push_worker_id = pid * REVERSE_VECTOR_WORKERS + push_sub_id
            num_push_workers = ncore * REVERSE_VECTOR_WORKERS
            ovp = tl.arange(0, BLOCK_N_PUSH)
            for out_pos in range(push_worker_id, total_recv, num_push_workers):
                op64 = out_pos.to(tl.int64)
                src_pos = tl.load(inv_local_sort_idxs_ptr + op64).to(tl.int64)
                dst_rank = tl.load(write_rank_ptr + out_pos)
                dst_off = tl.load(write_off_ptr + op64).to(tl.int64)
                rp = dl.symm_at(peer_mem_ptr, dst_rank)
                for ns in range(0, H_push, BLOCK_N_PUSH):
                    mask = ovp < (H_push - ns)
                    val = tl.load(hidden_buf_ptr + src_pos * H_push + (ns + ovp), mask=mask, other=0.0)
                    tl.store(rp + dst_off * H_push + (ns + ovp), val, mask=mask)

    libshmem_device.barrier_all()

    # ===== Phase 3: topk-sum reduce: peer_mem -> grad_hidden (VECTOR) =====
    with al.scope(core_mode="vector", disable_auto_sync=True):
        reduce_sub_id = sub_vec_id().to(tl.int32)
        if reduce_sub_id < REDUCE_VECTOR_WORKERS:
            reduce_worker_id = pid * REDUCE_VECTOR_WORKERS + reduce_sub_id
            num_reduce_workers = ncore * REDUCE_VECTOR_WORKERS
            ovr = tl.arange(0, BLOCK_N_PUSH)
            for ti in range(reduce_worker_id, B, num_reduce_workers):
                ti64 = ti.to(tl.int64)
                for ns in range(0, H_push, BLOCK_N_PUSH):
                    mask = ovr < (H_push - ns)
                    acc = tl.zeros((BLOCK_N_PUSH,), dtype=tl.float32)
                    for j in range(topk):
                        fi = ti * topk + j
                        sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                        acc += tl.load(peer_mem_ptr + sp * H_push + (ns + ovr), mask=mask, other=0.0)
                    oo = ti64 * stride_om + (ns + ovr) * stride_on
                    tl.store(output_ptr + oo, acc.to(output_ptr.dtype.element_ty), mask=mask)

    libshmem_device.barrier_all()


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

    num_tm = int(saved["num_tiles_total"].item())
    num_tn = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    fc1_combined = saved["fc1_combined"].contiguous()
    cache = dict(
        weight=fc1_combined,
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        M=M, N=H, K=fc1_combined.shape[1], E=saved["experts_per_rank"], num_tm=num_tm, num_tn=num_tn,
        inv_local=inv_local, write_rank=write_rank, write_off=write_off, inv_sort=inv_sort,
        total_recv=saved["total_recv"], H=H, B=saved["batch_size"], topk=saved["topk"],
        total_send=saved["total_send"],
        we=fc1_combined.stride(0), wk=fc1_combined.stride(1), wn=fc1_combined.stride(2),
        stride_om=H, stride_on=1,
    )
    saved["_combine_cache"] = cache
    return cache


def _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate):
    """Build combine push maps (expert->home) — same as 06 _prepare_fc2_combine.
    Static maps are cached on `saved`; only the grad-dependent strides are added."""
    p = _combine_static_maps(saved)
    p = dict(p)  # shallow copy so we can add grad-dependent fields
    p["inp"] = grad_fc1_output.contiguous()
    p["grad_gate"] = grad_gate.contiguous()
    p["inp_stride_im"] = grad_fc1_output.stride(0)
    p["inp_stride_ik"] = grad_fc1_output.stride(1)
    return p


def _gate_bwd_host(saved, grad_gate):
    """Host-side gate (routing-weight) grad: grad_gate [M] (expert, sorted) ->
    grad_routing_weights [B, topk] (home) via hccl all_to_all. Kept on the host
    because dl.symm_at only resolves correctly at heap offset 0 with a varying
    rank, so a second symmetric buffer for the scalar gate can't be used. The
    gate is a tiny [M] vector, so a host all_to_all is cheap."""
    B = saved["batch_size"]; topk = saved["topk"]; H = saved["hidden_dim"]
    dtype = grad_gate.dtype; device = grad_gate.device
    ep_group = saved["ep_group"]
    grad_gate_unsorted = grad_gate[saved["inv_local"]]                       # arrival order
    grad_sorted_weights = torch.empty((saved["total_send"],), dtype=dtype, device=device)
    dist.all_to_all_single(grad_sorted_weights, grad_gate_unsorted,
                           output_split_sizes=saved["splits_send_list"],
                           input_split_sizes=saved["splits_recv_list"], group=ep_group)
    grad_routing_flat = grad_sorted_weights[saved["inv_sort"]]               # [B*topk]
    return grad_routing_flat.view(B, topk)


def _launch_combine_fc1_bwd(prep, peer_mem, hidden_buf, output):
    # Single fused launch: fc1 input-grad GEMM (cube) + reverse-A2A push (vector)
    # + topk reduce (vector), fenced by the three in-kernel barrier_all() — the
    # first fences the local Cube->Vector handoff (hidden_buf), the second the
    # cross-rank push->reduce handoff (peer_mem), the trailing one quiesces
    # peer_mem before the next forward reuses it. No host sync.
    kernel_combine_fc1_bwd[(ncore(), 1, 1)](
        prep["inp"], prep["weight"], hidden_buf,
        prep["meta_expert_ids"], prep["meta_split_cum"], prep["meta_tile_num"], prep["expert_counts"],
        prep["M"], prep["N"], prep["K"], prep["E"], prep["num_tm"], prep["num_tn"],
        prep["inp_stride_im"], prep["inp_stride_ik"], prep["we"], prep["wk"], prep["wn"],
        prep["inv_local"], prep["write_rank"], prep["write_off"], peer_mem,
        prep["total_recv"], prep["H"],
        prep["inv_sort"], output,
        prep["B"], prep["topk"], prep["total_send"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K,
        BLOCK_N_PUSH=512,
        REVERSE_VECTOR_WORKERS=1, REDUCE_VECTOR_WORKERS=1,
        num_warps=8)
    return output


def combine_fc1_bwd_triton(saved, grad_fc1_output, grad_gate, peer_mem, return_hidden=False):
    """Step 4: returns (grad_hidden [B,H], grad_routing_weights [B,topk]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused from step 1,
    which has finished by now). The gate grad is computed on the host (all_to_all).
    If return_hidden, also returns hidden_buf (=grad_recv_hidden_sorted)."""
    prep = _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate)
    # GEMM writes every hidden_buf tile (meta covers all tokens); reduce
    # writes every output row -> empty, no zero-fill needed. hidden_buf is a plain
    # (non-symmetric) local workspace; peer_mem is the only symmetric buffer.
    hidden_buf = torch.empty(prep["M"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    output = torch.empty(prep["B"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    # peer_mem is fully overwritten by the push phase (every row read by the
    # reduce was written by some rank's push); the fused kernel's barrier_all
    # handles cross-rank sync, so no host zero/barrier is needed.
    _launch_combine_fc1_bwd(prep, peer_mem, hidden_buf, output)
    grad_routing_weights = _gate_bwd_host(saved, grad_gate)
    if return_hidden:
        return output, grad_routing_weights, hidden_buf
    return output, grad_routing_weights
