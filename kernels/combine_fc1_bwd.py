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

from .common import ncore, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K


# Phase 1 fc1 input-grad GEMM: acc[M,H] = grad_fc1_output @ fc1_combined[e]
# barrier_all()
# Phase 2 reverse-A2A push (expert->home): hidden_buf[inv_local[out]] -> peer_mem
#   AND grad_gate[inv_local[out]] -> peer_mem_gate   (06 Phase2 + gate channel)
# barrier_all()
# Phase 3 reduce: grad_hidden[b] = sum_j peer_mem[inv_sort[b*topk+j]]  (06 Phase3)
#   AND grad_routing[b*topk+j] = peer_mem_gate[inv_sort[b*topk+j]]  (gather, no sum)
@triton.jit
def kernel_fc1_input_grad_gemm(
    # ---- GEMM (fc1 input-grad) ----
    inp_ptr,                  # grad_fc1_output [M, 2*ffn]  (sorted)
    weight_ptr,               # fc1_combined [E, 2*ffn, H]  (K=2*ffn, N=H)
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H] out
    meta_expert_ids_ptr, meta_split_cum_ptr, meta_tile_num_ptr, expert_counts_ptr,
    M, N, K, E, num_tiles_m, num_tiles_n,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
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


# push + reduce (hidden) + gate channel. Split from the GEMM into its own launch:
# in a single fused kernel the GEMM phase corrupted the push's read of hidden_buf
# (barrier was fine, cause unclear — likely a codegen interaction). The GEMM-only
# and push+reduce-only kernels are each verified correct, so we launch them
# separately with a host sync between.
@triton.jit
def kernel_combine_push_reduce(
    hidden_buf_ptr,           # grad_recv_hidden_sorted [M, H]  (GEMM output, in)
    inv_local_sort_idxs_ptr, write_rank_ptr, write_off_ptr,
    peer_mem_ptr,             # symmetric [total_send, H] at HEAP OFFSET 0 (symm_at+loaded rank needs offset 0)
    total_recv, H_push,
    inv_sort_idxs_ptr,        # int64 [total_send]
    output_ptr,               # grad_hidden [B, H]
    B, topk, total_send,
    stride_om, stride_on,
    BLOCK_N_PUSH: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    ovp = tl.arange(0, BLOCK_N_PUSH)

    # ===== Phase 2: reverse-A2A push (expert->home) for hidden — 06 Phase2 =====
    for out_pos in range(pid, total_recv, ncore):
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

    # ===== Phase 3: topk sum (hidden) — 06 Phase3 =====
    for ti in range(pid, B, ncore):
        ti64 = ti.to(tl.int64)
        for ns in range(0, H_push, BLOCK_N_PUSH):
            mask = ovp < (H_push - ns)
            acc = tl.zeros((BLOCK_N_PUSH,), dtype=tl.float32)
            for j in range(topk):
                fi = ti * topk + j
                sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                acc += tl.load(peer_mem_ptr + sp * H_push + (ns + ovp), mask=mask, other=0.0)
            oo = ti64 * stride_om + (ns + ovp) * stride_on
            tl.store(output_ptr + oo, acc.to(output_ptr.dtype.element_ty), mask=mask)


def _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate):
    """Build combine push maps (expert->home) — same as 06 _prepare_fc2_combine."""
    device = grad_fc1_output.device
    pe = saved["ep_rank"]
    W = saved["world_size"]
    H = saved["hidden_dim"]
    ep_group = saved["ep_group"]
    M = saved["M"]
    total_send = saved["total_send"]
    total_recv = saved["total_recv"]

    send_t = torch.tensor(saved["splits_send_list"], dtype=torch.int64, device=device)
    all_send = torch.stack(all_gather_list(send_t, ep_group))     # [W,W]: all_send[r][d] = r sends to d
    send_cum = torch.zeros_like(all_send)
    send_cum[:, 1:] = all_send[:, :-1].cumsum(dim=1)               # send_cum[d, me] = sum_{s<me} all_send[d, s]

    recv_t = torch.tensor(saved["splits_recv_list"], dtype=torch.int64, device=device)
    write_rank = torch.zeros(M, dtype=torch.int32, device=device)
    write_off = torch.zeros(M, dtype=torch.int64, device=device)
    seg = 0
    for d in range(W):
        n = int(recv_t[d].item())
        base = int(send_cum[d, pe].item())
        for p in range(seg, seg + n):
            write_rank[p] = d
            write_off[p] = base + (p - seg)
        seg += n

    inv_local = torch.argsort(saved["local_sort_idxs"]).to(torch.int64).to(device).contiguous()
    inv_sort = torch.argsort(saved["sort_idxs"]).to(torch.int64).to(device).contiguous()

    num_tm = int(saved["num_tiles_total"].item())
    num_tn = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    fc1_combined = saved["fc1_combined"].contiguous()
    return dict(
        inp=grad_fc1_output.contiguous(), weight=fc1_combined, grad_gate=grad_gate.contiguous(),
        meta_expert_ids=saved["meta_expert_ids"].to(device), meta_split_cum=saved["meta_split_cum"].to(device),
        meta_tile_num=saved["meta_tile_num"].to(device), expert_counts=saved["expert_counts"].to(device),
        M=M, N=H, K=fc1_combined.shape[1], E=saved["experts_per_rank"], num_tm=num_tm, num_tn=num_tn,
        inv_local=inv_local, write_rank=write_rank, write_off=write_off, inv_sort=inv_sort,
        total_recv=total_recv, H=H, B=saved["batch_size"], topk=saved["topk"], total_send=total_send,
        inp_stride_im=grad_fc1_output.stride(0), inp_stride_ik=grad_fc1_output.stride(1),
        we=fc1_combined.stride(0), wk=fc1_combined.stride(1), wn=fc1_combined.stride(2),
        stride_om=H, stride_on=1,
    )


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
    # Launch 1: fc1 input-grad GEMM -> hidden_buf
    kernel_fc1_input_grad_gemm[(ncore(), 1, 1)](
        prep["inp"], prep["weight"], hidden_buf,
        prep["meta_expert_ids"], prep["meta_split_cum"], prep["meta_tile_num"], prep["expert_counts"],
        prep["M"], prep["N"], prep["K"], prep["E"], prep["num_tm"], prep["num_tn"],
        prep["inp_stride_im"], prep["inp_stride_ik"], prep["we"], prep["wk"], prep["wn"],
        BLOCK_M=BLOCK_SIZE_M, BLOCK_N=BLOCK_SIZE_N, BLOCK_K=BLOCK_SIZE_K, num_warps=8)
    # host sync: GEMM must finish writing hidden_buf before the push reads it.
    torch.npu.synchronize()
    dist.barrier()
    # Launch 2: reverse-A2A push + topk reduce (hidden)
    kernel_combine_push_reduce[(ncore(), 1, 1)](
        hidden_buf,
        prep["inv_local"], prep["write_rank"], prep["write_off"], peer_mem,
        prep["total_recv"], prep["H"],
        prep["inv_sort"], output,
        prep["B"], prep["topk"], prep["total_send"],
        prep["stride_om"], prep["stride_on"],
        BLOCK_N_PUSH=512, num_warps=8)
    return output


def combine_fc1_bwd_triton(saved, grad_fc1_output, grad_gate, peer_mem, return_hidden=False):
    """Step 4: returns (grad_hidden [B,H], grad_routing_weights [B,topk]).
    peer_mem is the shared symmetric buffer at heap offset 0 (reused from step 1,
    which has finished by now). The gate grad is computed on the host (all_to_all).
    If return_hidden, also returns hidden_buf (=grad_recv_hidden_sorted)."""
    prep = _prepare_combine_fc1_bwd(saved, grad_fc1_output, grad_gate)
    hidden_buf = torch.zeros(prep["M"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    output = torch.zeros(prep["B"], prep["N"], dtype=grad_fc1_output.dtype, device=grad_fc1_output.device)
    peer_mem.zero_(); dist.barrier(saved["ep_group"])
    _launch_combine_fc1_bwd(prep, peer_mem, hidden_buf, output)
    grad_routing_weights = _gate_bwd_host(saved, grad_gate)
    if return_hidden:
        return output, grad_routing_weights, hidden_buf
    return output, grad_routing_weights
