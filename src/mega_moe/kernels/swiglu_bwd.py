# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  swiglu_bwd.py  —  step 2: SwiGLU backward kernel (ported from kernels/nvidia/swiglu.py)
# ============================================================================

import torch
import triton
import triton.language as tl

from .common import ncore


@triton.jit
def kernel_swiglu_bwd(
    dC_ptr, dC_stride,        # grad_swiglu [M, ffn]
    AB_ptr, AB_stride,        # fc1_output [M, 2*ffn]  (A=gate first half, B=up second half)
    ffn,
    scale_ptr,                # recv_weights_sorted [M]
    dAB_ptr,                  # grad_fc1_output [M, 2*ffn] out
    dscale_ptr,               # grad_gate [M] out
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    for row in range(pid, n_rows, ncore):
        r64 = row.to(tl.int64)
        a_ptr = AB_ptr + r64 * AB_stride          # gate half
        b_ptr = a_ptr + ffn                        # up half
        dc_ptr = dC_ptr + r64 * dC_stride
        mask = offs < ffn
        dc = tl.load(dc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)
        sigmoid_a = tl.sigmoid(a)
        silu_a = a * sigmoid_a
        silu_prime = silu_a * (1 - sigmoid_a) + sigmoid_a
        sc = tl.load(scale_ptr + r64)
        da = dc * silu_prime * b * sc
        db = dc * silu_a * sc
        tl.store(dAB_ptr + r64 * AB_stride + offs, da.to(AB_ptr.dtype.element_ty), mask=mask)
        tl.store(dAB_ptr + r64 * AB_stride + ffn + offs, db.to(AB_ptr.dtype.element_ty), mask=mask)
        tl.store(dscale_ptr + r64, tl.sum(silu_a * b * dc))


# Row-fused variant of the above. Opt-in only: see swiglu_bwd_triton's `rows_per_iter`.
#
# The shipped kernel walks ONE row per loop iteration, so at 16384 tokens each program
# issues 65536/ncore separate 1-row passes of BLOCK_SIZE(=1024, padded from ffn=768)
# elements. a5_ops KB TR-OL-12 names that shape: one row per program with M >> core count
# is dispatch-bound below roughly 2048 elements per program, and the fix is to fuse rows.
# Its evidence op is SwiGLU itself, measured 1 -> 8 rows = 6.4x on A3 (24 AIVs).
#
# Measured here (Ascend950DT_9582, 64 vector cores, non-bytecode, M=65536, ffn=768):
#
#   rows/iter      ms     vs shipped   elements/program
#           1   2.011          --                 1024
#           2   1.170        1.72x                2048
#           4   0.786        2.56x                4096
#           8   0.581        3.47x                8192
#          16   MLIRCompilationError -- 16*1024 hits the tile-element cap (KB TR-EC-4)
#
# Direction matches the KB, magnitude does not: 3.47x here against 6.4x there, consistent
# with this part having 64 vector cores rather than 24 so each core carries less dispatch
# pressure to begin with. The KB entry is usable as INPUT; its number is not transferable.
#
# ⚠️ NUMERICS -- this is a change, not an equivalence, and that is why it is opt-in:
#   dAB    (the [M, 2*ffn] gradient, the bulk of the output)  BIT-IDENTICAL, max|d| = 0
#   dscale (per-row gate gradient, produced by a REDUCTION)   max|d| = 6.25e-2, which is
#          8.22e-4 relative to max|dscale|. Summing a row of a 2-D tile does not add in
#          the same order as summing a 1-D block, and the result is stored in bf16.
# That sits inside the repo's own bf16 gradient tolerance (2e-2), but a caller that needs
# the shipped reduction order must not enable this. Default stays 1 = shipped behaviour.
@triton.jit
def kernel_swiglu_bwd_rows(
    dC_ptr, dC_stride, AB_ptr, AB_stride, ffn, scale_ptr, dAB_ptr, dscale_ptr, n_rows,
    BLOCK_SIZE: tl.constexpr, ROWS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ncore = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    rows = tl.arange(0, ROWS)
    cmask = offs < ffn
    for base in range(pid * ROWS, n_rows, ncore * ROWS):
        r = base + rows
        rmask = r < n_rows
        r64 = r.to(tl.int64)
        m2 = rmask[:, None] & cmask[None, :]
        a_off = r64[:, None] * AB_stride + offs[None, :]
        dc = tl.load(dC_ptr + r64[:, None] * dC_stride + offs[None, :], mask=m2, other=0.0).to(tl.float32)
        a = tl.load(AB_ptr + a_off, mask=m2, other=0.0).to(tl.float32)
        b = tl.load(AB_ptr + a_off + ffn, mask=m2, other=0.0)
        sigmoid_a = tl.sigmoid(a)
        silu_a = a * sigmoid_a
        silu_prime = silu_a * (1 - sigmoid_a) + sigmoid_a
        sc = tl.load(scale_ptr + r64, mask=rmask, other=0.0)[:, None]
        tl.store(dAB_ptr + a_off, (dc * silu_prime * b * sc).to(AB_ptr.dtype.element_ty), mask=m2)
        tl.store(dAB_ptr + a_off + ffn, (dc * silu_a * sc).to(AB_ptr.dtype.element_ty), mask=m2)
        tl.store(dscale_ptr + r64, tl.sum(silu_a * b * dc, axis=1), mask=rmask)


def swiglu_bwd_triton(grad_swiglu, fc1_output, recv_weights_sorted, rows_per_iter=1):
    """Step 2: returns (grad_fc1_output [M,2ffn], grad_gate [M]).

    rows_per_iter > 1 selects the row-fused kernel. It is 3.47x faster at 8 rows on
    Ascend950DT, and it changes dscale by up to 8.22e-4 relative (reduction order); see
    the note above kernel_swiglu_bwd_rows. Default 1 = shipped kernel, shipped numerics.
    """
    M, two_ffn = fc1_output.shape
    ffn = two_ffn // 2
    dAB = torch.empty_like(fc1_output)
    dscale = torch.empty(M, dtype=fc1_output.dtype, device=fc1_output.device)
    BLOCK_SIZE = triton.next_power_of_2(ffn)
    if rows_per_iter > 1:
        # BLOCK_SIZE * ROWS must stay under the tile-element cap; 16*1024 fails to compile.
        if BLOCK_SIZE * rows_per_iter > 8192:
            raise ValueError(
                f"rows_per_iter={rows_per_iter} with BLOCK_SIZE={BLOCK_SIZE} exceeds the "
                f"tile-element budget (measured: 16*1024 raises MLIRCompilationError)")
        kernel_swiglu_bwd_rows[(ncore(), 1, 1)](
            grad_swiglu, grad_swiglu.stride(0), fc1_output, fc1_output.stride(0),
            ffn, recv_weights_sorted, dAB, dscale, M,
            BLOCK_SIZE=BLOCK_SIZE, ROWS=rows_per_iter, num_warps=8, use_bytecode=True)
        return dAB, dscale
    kernel_swiglu_bwd[(ncore(), 1, 1)](
        grad_swiglu, grad_swiglu.stride(0), fc1_output, fc1_output.stride(0),
        ffn, recv_weights_sorted, dAB, dscale, M, BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        use_bytecode=True)
    return dAB, dscale
