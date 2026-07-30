# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  dispatch_fc2_bwd.py  —  step 1: dispatch-A2A(home->expert) + fc2 input-grad
#  (push -> barrier -> GEMM; the dual of 06's GEMM -> push -> reduce)
# ============================================================================

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device

from .common import ncore, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K


# Phase 1 push (home -> expert): for each home row h (in sort_idxs order), write
#   grad_combined_out_flat[h] -> dl.symm_at(peer_mem, h_dst_rank[h]) + h_dst_off[h]*H
# barrier_all()
# Phase 2 fc2 input-grad GEMM (Cube): a = peer_mem[local_sort_idxs[row]] (gather
#   back to sorted order), b = fc2[e][H,ffn]; acc[M,ffn] = a @ b
@triton.jit
def kernel_dispatch_fc2_bwd(
    # ---- push (home->expert) ----
    gco_ptr,                  # grad_combined_out_flat [total_send, H] (home, sort_idxs order)
    h_dst_rank_ptr,           # int32 [total_send]
    h_dst_off_ptr,            # int64 [total_send]
    peer_mem_ptr,             # symmetric [total_recv, H] at HEAP OFFSET 0 (reused with step 4)
    total_send, H,
    # ---- GEMM (fc2 input-grad) ----
    fc2_ptr,                  # [E, H, ffn]  (K=H, N=ffn, weight_reduce_last_dim=True)
    local_sort_idxs_ptr,      # int64 [total_recv]  (arrival -> sorted)
    meta_expert_ids_ptr, meta_split_cum_ptr, meta_tile_num_ptr, expert_counts_ptr,
    M, N, K, E, num_tiles_m, num_tiles_n,
    stride_gm, stride_gk,     # gco strides (H, 1)
    stride_we, stride_wk, stride_wn,   # fc2 [E,H,ffn]: (H*ffn, ffn, 1)
    # ---- output ----
    out_ptr,                  # grad_swiglu [M, ffn]
    # ---- constexpr ----
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    ovh = tl.arange(0, BLOCK_H_PUSH)

    # ===== Phase 1: dispatch push (home -> expert) =====
    ts64 = total_send
    for h in range(pid, ts64, ncore):
        h64 = h.to(tl.int64)
        dst_rank = tl.load(h_dst_rank_ptr + h)
        dst_off = tl.load(h_dst_off_ptr + h64).to(tl.int64)
        rp = dl.symm_at(peer_mem_ptr, dst_rank)
        for ns in range(0, H, BLOCK_H_PUSH):
            mask = ovh < (H - ns)
            ro = h64 * H + (ns + ovh)
            val = tl.load(gco_ptr + ro, mask=mask, other=0.0)
            ro2 = dst_off * H + (ns + ovh)
            tl.store(rp + ro2, val, mask=mask)

    libshmem_device.barrier_all()

    # ===== Phase 2: fc2 input-grad GEMM (gather via local_sort_idxs) =====
    # acc[M, ffn] = peer_mem[local_sort_idxs[row]] @ fc2[e]
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


def _prepare_dispatch_fc2_bwd(saved, dy):
    """Build dispatch push maps (home->expert), the dual of 06's combine maps."""
    device = dy.device
    pe = saved["ep_rank"]
    W = saved["world_size"]
    H = saved["hidden_dim"]
    B = saved["batch_size"]; topk = saved["topk"]
    ep_group = saved["ep_group"]
    sort_idxs = saved["sort_idxs"].to(device)
    total_send = saved["total_send"]

    # grad_combined_out_flat = dy.repeat_interleave(topk)[sort_idxs]  (home, sort_idxs order)
    grad_combined_out_flat = dy.repeat_interleave(topk, dim=0)[sort_idxs].contiguous()

    send_t = torch.tensor(saved["splits_send_list"], dtype=torch.int64, device=device)  # [W] own per-dest
    all_send = torch.stack(all_gather_list(send_t, ep_group))                           # [W,W]: all_send[s,d]
    # offset(pe, d) = sum_{s<pe} all_send[s, d]
    cum_src = all_send.cumsum(dim=0)                       # cum_src[pe,d] = sum_{s<=pe} all_send[s,d]
    offset = cum_src - all_send                            # offset[pe,d] = sum_{s<pe} all_send[s,d]
    pe_offset_for_d = offset[pe]                           # [W]

    own_start = torch.cat([torch.zeros(1, dtype=torch.int64, device=device),
                           send_t.cumsum(0)[:-1]])         # own_start[d] = sum_{d'<d} send_t[d']
    h_dst_rank = torch.repeat_interleave(torch.arange(W, dtype=torch.int32, device=device), send_t)
    h_pos = torch.arange(total_send, dtype=torch.int64, device=device) - own_start[h_dst_rank.to(torch.int64)]
    h_dst_off = pe_offset_for_d[h_dst_rank.to(torch.int64)] + h_pos

    # GEMM metadata
    num_tiles_m = int(saved["num_tiles_total"].item())
    num_tiles_n = (saved["ffn_dim"] + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    return dict(
        gco=grad_combined_out_flat, h_dst_rank=h_dst_rank, h_dst_off=h_dst_off,
        M=saved["M"], N=saved["ffn_dim"], K=H, E=saved["experts_per_rank"],
        num_tm=num_tiles_m, num_tn=num_tiles_n,
        fc2=saved["fc2"].contiguous(), local_sort_idxs=saved["local_sort_idxs"].to(torch.int64).to(device),
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        total_send=total_send, H=H, total_recv=saved["total_recv"],
    )


def _launch_dispatch_fc2_bwd(prep, peer_mem, out):
    kernel_dispatch_fc2_bwd[(ncore(), 1, 1)](
        prep["gco"], prep["h_dst_rank"], prep["h_dst_off"], peer_mem,
        prep["total_send"], prep["H"],
        prep["fc2"], prep["local_sort_idxs"],
        prep["meta_expert_ids"], prep["meta_split_cum"], prep["meta_tile_num"], prep["expert_counts"],
        prep["M"], prep["N"], prep["K"], prep["E"], prep["num_tm"], prep["num_tn"],
        prep["gco"].stride(0), prep["gco"].stride(1),
        prep["fc2"].stride(0), prep["fc2"].stride(1), prep["fc2"].stride(2),
        out,
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K,
        BLOCK_H_PUSH=512, num_warps=8)
    return out


def dispatch_fc2_bwd_triton(saved, dy, peer_mem):
    """Step 1: returns (grad_swiglu [M,ffn], grad_fc2_out_sorted [M,H]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused with step 4)."""
    prep = _prepare_dispatch_fc2_bwd(saved, dy)
    out = torch.zeros(prep["M"], prep["N"], dtype=dy.dtype, device=dy.device)
    peer_mem.zero_(); dist.barrier(saved["ep_group"])
    _launch_dispatch_fc2_bwd(prep, peer_mem, out)
    # grad_fc2_out_sorted = peer_mem (arrival) gathered by local_sort_idxs
    peer_view = peer_mem.view(-1)[:prep["total_recv"] * prep["H"]].view(prep["total_recv"], prep["H"])
    grad_fc2_out_sorted = peer_view[prep["local_sort_idxs"]].contiguous()
    return out, grad_fc2_out_sorted
