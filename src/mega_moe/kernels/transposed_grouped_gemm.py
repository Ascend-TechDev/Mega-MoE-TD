# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  transposed_grouped_gemm.py  —  step 3/5: weight-grad grouped GEMM (fc2 / fc1)
#  (ported from GPU transposed_moe_grouped_gemm, Ascend strided schedule)
# ============================================================================

import torch
import triton
import triton.language as tl

from .common import ncore, WGRAD_BLOCK_M, WGRAD_BLOCK_N, WGRAD_BLOCK_K


# grad_weight[e][n][k] = sum_m grad_out[m,n] * orig_in[m,k] = grad_out^T @ orig_in
# tl.dot has no trans_a on this build. Loading grad_out as a transposed
# [BLOCK_N, BLOCK_M] tile directly from [M,N] is a stride-N gather per vector
# (non-contiguous), which triggers a UB bus error on Ascend. Instead we
# pre-transpose grad_out -> grad_out_T [N,M] contiguous in Python, so each
# tile-row (fixed n) reads contiguous m (stride 1). Then a=[BLOCK_N,BLOCK_M]
# (contig) @ b=[BLOCK_M,BLOCK_K] (contig) -> acc [BLOCK_N, BLOCK_K]. Tiles
# (e, n, k) are unique and non-overlapping -> no atomics.
@triton.jit
def kernel_transposed_grouped_gemm(
    grad_out_T_ptr,           # [N, M] contiguous (grad_out transposed)
    orig_in_ptr,              # [M, K] contiguous
    grad_w_ptr,               # [E, N, K]
    split_size_cum_per_expert_ptr, expert_counts_ptr,
    N, K, E, num_tiles_n, num_tiles_k,
    stride_tn, stride_tm,     # grad_out_T: (M, 1)
    stride_om, stride_ok,     # orig_in:   (K, 1)
    stride_we, stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    total = E * num_tiles_n * num_tiles_k
    for task in range(pid, total, ncore):
        e = task // (num_tiles_n * num_tiles_k)
        rem = task % (num_tiles_n * num_tiles_k)
        tn = rem // num_tiles_k
        tk = rem % num_tiles_k
        split_begin = tl.load(split_size_cum_per_expert_ptr + e)
        split_size = tl.load(expert_counts_ptr + e)
        n_start = tn * BLOCK_N
        k_start = tk * BLOCK_K
        nmask = (n_start + offs_n) < N
        kmask = (k_start + offs_k) < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m in range(0, split_size, BLOCK_M):
            mm = m + offs_m
            mmask = mm < split_size
            # a[n, m] = grad_out_T[n, split_begin+m]  (contiguous in m)
            a_off = (n_start + offs_n[:, None]) * stride_tn + (split_begin + mm[None, :]) * stride_tm
            a = tl.load(grad_out_T_ptr + a_off, mask=nmask[:, None] & mmask[None, :], other=0.0)
            # b[m, k] = orig_in[split_begin+m, k_start+k]  (contiguous in k)
            b_off = (split_begin + mm[:, None]) * stride_om + (k_start + offs_k[None, :]) * stride_ok
            b = tl.load(orig_in_ptr + b_off, mask=mmask[:, None] & kmask[None, :], other=0.0)
            acc += tl.dot(a, b)
        c_off = e * stride_we + (n_start + offs_n[:, None]) * stride_wn + (k_start + offs_k[None, :]) * stride_wk
        tl.store(grad_w_ptr + c_off, acc.to(grad_w_ptr.dtype.element_ty),
                 mask=nmask[:, None] & kmask[None, :])


def transposed_grouped_gemm_triton(grad_out, orig_in, expert_counts, split_size_cum_per_expert):
    """Weight-grad grouped gemm: grad_w [E,N,K] = grad_out^T @ orig_in.
    grad_out [M,N], orig_in [M,K] (rows sorted by expert)."""
    M, N = grad_out.shape
    K = orig_in.shape[1]
    E = int(expert_counts.shape[0])
    grad_out_T = grad_out.T.contiguous()       # [N, M], makes the a-tile contiguous
    orig_in_c = orig_in.contiguous()
    dev = grad_out.device
    split_size_cum_per_expert = split_size_cum_per_expert.to(dev)
    expert_counts = expert_counts.to(dev)
    # kernel writes every (e,n,k) tile (0-token experts store a zero acc) -> empty
    grad_w = torch.empty(E, N, K, dtype=grad_out.dtype, device=dev)
    num_tn = (N + WGRAD_BLOCK_N - 1) // WGRAD_BLOCK_N
    num_tk = (K + WGRAD_BLOCK_K - 1) // WGRAD_BLOCK_K
    kernel_transposed_grouped_gemm[(ncore(), 1, 1)](
        grad_out_T, orig_in_c, grad_w,
        split_size_cum_per_expert, expert_counts,
        N, K, E, num_tn, num_tk,
        grad_out_T.stride(0), grad_out_T.stride(1), orig_in_c.stride(0), orig_in_c.stride(1),
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        BLOCK_M=WGRAD_BLOCK_M, BLOCK_N=WGRAD_BLOCK_N, BLOCK_K=WGRAD_BLOCK_K, num_warps=8,
        use_bytecode=True)
    return grad_w
