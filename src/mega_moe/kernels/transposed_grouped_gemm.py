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
    N: tl.constexpr, K: tl.constexpr, E, num_tiles_n: tl.constexpr, num_tiles_k: tl.constexpr,
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


def transposed_grouped_gemm_triton(grad_out, orig_in, expert_counts, split_size_cum_per_expert,
                                   block_m=None, block_n=None, block_k=None, grid=None):
    """Weight-grad grouped gemm: grad_w [E,N,K] = grad_out^T @ orig_in.
    grad_out [M,N], orig_in [M,K] (rows sorted by expert).

    `block_m/n/k` and `grid` default to the module constants and `ncore()`, so an
    unchanged call site behaves exactly as before — this is a parameterisation, not a
    retune. They exist because the shipped constants were fitted to one part: their own
    comments justify BM=256 against "L0A 1/4-full" and BK=256 against a 192KB UB, and
    `ncore()` asserted the part had <=24 AICore.

    Measured 2026-08-06, Ascend950DT_9582 (cube=32), Qwen3-30B-A3B, 64 experts/rank,
    bf16, isolated single-card, stock triton (no `use_bytecode`):

        tokens  rows/expert   shipped(24,256,128,256)   (32,512,256,256)
          4096      256            0.412 ms                0.385 ms   1.07x
          8192      512            0.782 ms                0.434 ms   1.80x
         16384     1024            6.894 ms                3.166 ms   2.18x

    The gain appears exactly where per-expert rows exceed BLOCK_M, i.e. where the
    reduction starts needing multiple passes — the same shape range where the repo's
    README reports the backward ratio collapsing (0.51x at 8k, 0.37x at 16k).

    Output was bit-identical between the two configurations at 16384. That check is
    only meaningful because the comparator was first shown to be live: perturbing one
    input element moved it by 3.5 absolute. ⚠️ Still one seed and one shape — measured,
    not proven. ⚠️ Also NOT established: whether every output region is written. That
    test used even routing, which has no empty experts, so it is structurally unable to
    speak to the empty-expert case.
    """
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
    bm = WGRAD_BLOCK_M if block_m is None else block_m
    bn = WGRAD_BLOCK_N if block_n is None else block_n
    bk = WGRAD_BLOCK_K if block_k is None else block_k
    num_tn = (N + bn - 1) // bn
    num_tk = (K + bk - 1) // bk
    kernel_transposed_grouped_gemm[(ncore() if grid is None else grid, 1, 1)](
        grad_out_T, orig_in_c, grad_w,
        split_size_cum_per_expert, expert_counts,
        N, K, E, num_tn, num_tk,
        grad_out_T.stride(0), grad_out_T.stride(1), orig_in_c.stride(0), orig_in_c.stride(1),
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=8,
        use_bytecode=True)
    return grad_w
