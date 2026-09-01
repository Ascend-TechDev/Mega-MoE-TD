# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  fused_swiglu_bwd_fc2_wgrad.py
#
#  Fuses step2 (SwiGLU backward, Vector) + step3 (fc2 weight-grad, Cube) into
#  ONE launch so the Ascend AICore runs the vector and cube instruction streams
#  concurrently (wall ~= max(cube, vector) ~= cube). Both scopes are
#  data-independent, wrapped in al.scope(..., disable_auto_sync=True) with no
#  barrier between them; no sub_vec_id gate (pure local GM, no SHMEM comm).
#
#  Ported from triton_gen/zhoujinggan/output/opt_iter_1/optimized_code.py with
#  two integration fixes:
#    1. cube tile is the LOCAL FUSED_W{M,N,K} (BM=64) — NOT the repo
#       WGRAD_BLOCK_M (=256). BM=64 is both the tuned optimum AND keeps the
#       no-host-transpose direct [M,N] load below the Ascend UB-bus-error
#       threshold (BM=256 would force the .T.contiguous() copy the standalone
#       transposed_grouped_gemm pays).
#    2. the contiguous cube partition has a remainder guard so non-divisible
#       (total, ncore) shapes do not silently drop tasks (correctness on the
#       functional suite's smaller shapes).
# ============================================================================

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al  # noqa: F401  (al.scope only)

from .common import ncore

# Cube (fc2 wgrad) tile. BM=64 is the tuned optimum and keeps the direct
# transposed [M,N] load UB-bus-error-safe; do NOT swap for WGRAD_BLOCK_M (256).
# MOE_FUSED_WGRAD_BLOCK_M overrides BM for experimentation.
FUSED_WBM = 64
FUSED_WBN = 128
FUSED_WBK = 256


@triton.jit
def kernel_fused_swiglu_bwd_fc2_wgrad(
    # ---- vector: SwiGLU backward ----
    dC_ptr, dC_stride,        # grad_swiglu [M, ffn]
    AB_ptr, AB_stride,        # fc1_output  [M, 2*ffn]  (gate first half | up second half)
    ffn,
    scale_ptr,                # recv_weights_sorted [M]
    dAB_ptr,                  # grad_fc1_output out [M, 2*ffn]
    dscale_ptr,               # grad_gate out [M]
    n_rows,
    situ_beta, situ_linear_beta,   # SiTU params (ignored when ACTIVATION == 0)
    # ---- cube: fc2 weight-grad grouped GEMM ----
    grad_out_ptr,             # [M, N] contiguous (grad_fc2_out_sorted, NO host transpose)
    orig_in_ptr,              # [M, K] contiguous (swiglu_out_weighted.contiguous())
    grad_w_ptr,               # [E, N, K] out
    split_size_cum_per_expert_ptr, expert_counts_ptr,
    N: tl.constexpr, K: tl.constexpr, E,
    num_tiles_n: tl.constexpr, num_tiles_k: tl.constexpr,
    stride_outm, stride_outn, # grad_out [M,N]  (row stride = N, col stride = 1)
    stride_om, stride_ok,     # orig_in     (K, 1)
    stride_we, stride_wn, stride_wk,
    # ---- constexpr tile sizes ----
    BLOCK_SIZE: tl.constexpr,           # swiglu row tile = next_power_of_2(ffn)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,           # 0 = swiglu, 1 = situglu (vector scope)
    HAS_LINEAR_BETA: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)

    # ====================== Vector: SwiGLU/SiTU backward =================
    # Same activation pair as kernels/swiglu_bwd.py (host reference:
    # mindspeed_mm/fsdp/ops/glu/situ_triton.py backward kernel).
    with al.scope(core_mode="vector", disable_auto_sync=True):
        offs = tl.arange(0, BLOCK_SIZE)
        for row in range(pid, n_rows, ncore):
            r64 = row.to(tl.int64)
            a_ptr = AB_ptr + r64 * AB_stride          # gate half
            b_ptr = a_ptr + ffn                        # up half
            dc_ptr = dC_ptr + r64 * dC_stride
            mask = offs < ffn
            dc = tl.load(dc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            if ACTIVATION == 0:
                sigmoid_a = tl.sigmoid(a)
                act_a = a * sigmoid_a
                dact_a = act_a * (1 - sigmoid_a) + sigmoid_a
                v = b
                dv = 1.0
            else:
                t = 2.0 * tl.sigmoid(2.0 * a / situ_beta) - 1.0
                s = tl.sigmoid(a)
                act_a = situ_beta * t * s
                dact_a = (1.0 - t * t) * s + situ_beta * t * s * (1.0 - s)
                if HAS_LINEAR_BETA:
                    tu = 2.0 * tl.sigmoid(2.0 * b / situ_linear_beta) - 1.0
                    v = situ_linear_beta * tu
                    dv = 1.0 - tu * tu
                else:
                    v = b
                    dv = 1.0
            sc = tl.load(scale_ptr + r64)
            da = dc * dact_a * v * sc
            db = dc * act_a * dv * sc
            tl.store(dAB_ptr + r64 * AB_stride + offs, da.to(AB_ptr.dtype.element_ty), mask=mask)
            tl.store(dAB_ptr + r64 * AB_stride + ffn + offs, db.to(AB_ptr.dtype.element_ty), mask=mask)
            tl.store(dscale_ptr + r64, tl.sum(act_a * v * dc))

    # ====================== Cube: fc2 weight-grad GEMM ===================
    # CONTIGUOUS block partition: pid owns [pid*blk, (pid+1)*blk) with a
    # remainder guard (blk = ceil(total/ncore)) so non-divisible shapes do not
    # drop tasks. Each program walks consecutive (e,tn,tk) tiles of one expert
    # -> 336 tiles reuse the same split_size rows back-to-back (L1/L2 locality).
    # a[n,m] = grad_out[split_begin+m, n_start+n] read DIRECTLY from [M,N] (n
    # contiguous, m strided by N) — no host .T.contiguous() copy.
    with al.scope(core_mode="cube", disable_auto_sync=True):
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        total = E * num_tiles_n * num_tiles_k
        blk = (total + ncore - 1) // ncore              # ceil; covers all tasks
        base = pid * blk
        for i in range(blk):
            task = base + i
            if task < total:
                e = task // (num_tiles_n * num_tiles_k)
                rem = task - e * (num_tiles_n * num_tiles_k)   # % w/o modulo op
                tn = rem // num_tiles_k
                tk = rem - tn * num_tiles_k                   # % w/o modulo op
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
                    a_off = (split_begin + mm[None, :]) * stride_outm + (n_start + offs_n[:, None]) * stride_outn
                    a = tl.load(grad_out_ptr + a_off, mask=nmask[:, None] & mmask[None, :], other=0.0)
                    b_off = (split_begin + mm[:, None]) * stride_om + (k_start + offs_k[None, :]) * stride_ok
                    b = tl.load(orig_in_ptr + b_off, mask=mmask[:, None] & kmask[None, :], other=0.0)
                    acc += tl.dot(a, b)
                c_off = e * stride_we + (n_start + offs_n[:, None]) * stride_wn + (k_start + offs_k[None, :]) * stride_wk
                tl.store(grad_w_ptr + c_off, acc.to(grad_w_ptr.dtype.element_ty),
                         mask=nmask[:, None] & kmask[None, :])


def fused_swiglu_bwd_fc2_wgrad(grad_swiglu, fc1_output, recv_weights_sorted,
                               grad_fc2_out_sorted, swiglu_out_weighted,
                               expert_counts, split_size_cum_per_expert, *,
                               activation="swiglu", situ_beta=None,
                               situ_linear_beta=None):
    """Fused step2 (SwiGLU/SiTU bwd) + step3 (fc2 wgrad): returns
    (grad_fc1_output [M,2ffn], grad_gate [M], grad_fc2 [E,N,K]).

    Vector SwiGLU and Cube fc2-wgrad run concurrently in one launch
    (cube ~= wall, vector hidden). Reads grad_fc2_out_sorted directly as [M,N]
    (no host transpose). MOE_FUSED_WGRAD_BLOCK_M overrides the cube BM (default 64).
    ``activation`` selects the vector-scope derivative exactly like
    :func:`swiglu_bwd_triton`."""
    M, two_ffn = fc1_output.shape
    ffn = two_ffn // 2
    N = grad_fc2_out_sorted.shape[1]
    K = swiglu_out_weighted.shape[1]
    E = int(expert_counts.shape[0])
    dev = fc1_output.device

    if activation in (None, "swiglu"):
        act, beta, lbeta, has_lb = 0, 1.0, 1.0, False
    elif activation == "situglu":
        act = 1
        beta = 1.0 if situ_beta is None else float(situ_beta)
        lbeta = 1.0 if situ_linear_beta is None else float(situ_linear_beta)
        has_lb = situ_linear_beta is not None
    else:
        raise ValueError(f"unknown activation for the backward: {activation!r}")

    dAB = torch.empty_like(fc1_output)                           # [M, 2*ffn]
    dscale = torch.empty(M, dtype=fc1_output.dtype, device=dev)  # [M]

    orig_in_c = swiglu_out_weighted.contiguous()                 # [M, K]
    split_size_cum_per_expert = split_size_cum_per_expert.to(dev)
    expert_counts = expert_counts.to(dev)
    grad_w = torch.empty(E, N, K, dtype=grad_fc2_out_sorted.dtype, device=dev)

    BLOCK_SIZE = triton.next_power_of_2(ffn)
    block_m = int(os.environ.get("MOE_FUSED_WGRAD_BLOCK_M", str(FUSED_WBM)))
    num_tn = (N + FUSED_WBN - 1) // FUSED_WBN
    num_tk = (K + FUSED_WBK - 1) // FUSED_WBK

    kernel_fused_swiglu_bwd_fc2_wgrad[(ncore(), 1, 1)](
        grad_swiglu, grad_swiglu.stride(0),
        fc1_output, fc1_output.stride(0),
        ffn, recv_weights_sorted, dAB, dscale, M, beta, lbeta,
        grad_fc2_out_sorted, orig_in_c, grad_w,
        split_size_cum_per_expert, expert_counts,
        N, K, E, num_tn, num_tk,
        grad_fc2_out_sorted.stride(0), grad_fc2_out_sorted.stride(1),
        orig_in_c.stride(0), orig_in_c.stride(1),
        grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_M=block_m, BLOCK_N=FUSED_WBN, BLOCK_K=FUSED_WBK,
        ACTIVATION=act, HAS_LINEAR_BETA=has_lb,
        num_warps=8)
    return dAB, dscale, grad_w
