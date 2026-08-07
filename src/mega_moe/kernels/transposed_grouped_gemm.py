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
#
# grad_out is read in its NATURAL [M, N] layout and transposed inside the tile.
#
# The previous version materialised grad_out^T as an [N, M] tensor in Python. Its
# comment justified that by a real constraint -- loading a [BLOCK_N, BLOCK_M] tile
# *directly* out of [M, N] is a stride-N gather per vector and faults on Ascend --
# and that constraint still holds. What is done here is a different operation and
# does not hit it: the tile is loaded along its natural axes as [BLOCK_M, BLOCK_N]
# (contiguous in n) and then transposed in-register with tl.trans, so no strided
# gather is ever issued.
#
# WHY IT MATTERS, measured (Ascend950DT_9582, single card, non-bytecode path, grid
# pinned 24, Qwen3-30B-A3B shapes, 64 experts/rank, bf16):
#
#   the [N, M] layout made the a-tile's BLOCK_N rows `stride = M` apart, so the
#   tile's address span GREW WITH THE TOKEN COUNT: 64 KB at M=32768, 128 KB at
#   M=65536. Time per unit work was flat up to M=32768 and then collapsed.
#
#     tokens 16384 (M=65536)   step 3  6.878 ms -> 1.502 ms   4.58x
#                              step 5 13.813 ms -> 2.900 ms   4.76x
#     tokens 4096 / 8192       1.00-1.01x  (no regression, and no gain -- these
#                              shapes were never collapsing)
#     outputs bit-identical at every shape (max |diff| = 0.000e+00)
#
#   Controls behind those numbers: an A/A negative control in the same runs read
#   1.0008-1.0023, so the effects are far above what the harness can confuse; the
#   cause was isolated to M rather than to the m-loop trip count (M fixed + trips
#   doubled = 0.97x; trips fixed + M doubled = 4.53x) and to footprint rather than
#   stride aliasing (padding the physical row stride changed nothing).
#
#   Reading grad_out[m, n] directly fixes the a-tile row stride at N, independent
#   of M, which is what removes the growth term. Tile sizes are untouched -- this
#   is not a retune, and tile tuning cannot substitute for it.
#
# ⚠️ Not established: the effect on end-to-end backward (steps 1 and 4 are
# comm-fused and were not measured), and the numbers above are this part's; A3 has
# different cache geometry so the threshold and magnitude will differ there, though
# the mechanism (stride grows with M) is layout-determined and structural.
#
# Tiles (e, n, k) are unique and non-overlapping -> no atomics.
@triton.jit
def kernel_transposed_grouped_gemm(
    grad_out_ptr,             # [M, N] contiguous (natural layout, NOT transposed)
    orig_in_ptr,              # [M, K] contiguous
    grad_w_ptr,               # [E, N, K]
    split_size_cum_per_expert_ptr, expert_counts_ptr,
    N: tl.constexpr, K: tl.constexpr, E, num_tiles_n: tl.constexpr, num_tiles_k: tl.constexpr,
    stride_gm, stride_gn,     # grad_out:  (N, 1) -- row stride independent of M
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
            # a_raw[m, n] = grad_out[split_begin+m, n_start+n]  (contiguous in n; rows
            # stride_gm = N apart, i.e. independent of how many tokens there are)
            a_off = (split_begin + mm[:, None]) * stride_gm + (n_start + offs_n[None, :]) * stride_gn
            a_raw = tl.load(grad_out_ptr + a_off, mask=mmask[:, None] & nmask[None, :], other=0.0)
            a = tl.trans(a_raw)          # -> [BLOCK_N, BLOCK_M], no strided gather issued
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
    # No .T.contiguous() here: materialising [N, M] is what made the a-tile's row
    # stride grow with the token count. See the kernel header for the measurements.
    grad_out_c = grad_out.contiguous()
    orig_in_c = orig_in.contiguous()
    dev = grad_out.device
    split_size_cum_per_expert = split_size_cum_per_expert.to(dev)
    expert_counts = expert_counts.to(dev)
    # kernel writes every (e,n,k) tile (0-token experts store a zero acc) -> empty
    grad_w = torch.empty(E, N, K, dtype=grad_out.dtype, device=dev)
    num_tn = (N + WGRAD_BLOCK_N - 1) // WGRAD_BLOCK_N
    num_tk = (K + WGRAD_BLOCK_K - 1) // WGRAD_BLOCK_K
    kernel_transposed_grouped_gemm[(ncore(), 1, 1)](
        grad_out_c, orig_in_c, grad_w,
        split_size_cum_per_expert, expert_counts,
        N, K, E, num_tn, num_tk,
        grad_out_c.stride(0), grad_out_c.stride(1), orig_in_c.stride(0), orig_in_c.stride(1),
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        BLOCK_M=WGRAD_BLOCK_M, BLOCK_N=WGRAD_BLOCK_N, BLOCK_K=WGRAD_BLOCK_K, num_warps=8,
        use_bytecode=True)
    return grad_w
