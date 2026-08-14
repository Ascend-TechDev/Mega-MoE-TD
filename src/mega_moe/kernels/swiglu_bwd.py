# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  swiglu_bwd.py  —  step 2: SwiGLU backward kernel (ported from kernels/nvidia/swiglu.py)
# ============================================================================

import torch
import triton
import triton.language as tl

from .common import nvec


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
    nprogs = tl.num_programs(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    for row in range(pid, n_rows, nprogs):
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


def swiglu_bwd_triton(grad_swiglu, fc1_output, recv_weights_sorted):
    """Step 2: returns (grad_fc1_output [M,2ffn], grad_gate [M])."""
    M, two_ffn = fc1_output.shape
    ffn = two_ffn // 2
    dAB = torch.empty_like(fc1_output)
    dscale = torch.empty(M, dtype=fc1_output.dtype, device=fc1_output.device)
    BLOCK_SIZE = triton.next_power_of_2(ffn)
    # Pure-vector elementwise kernel: launch on nvec() (48) not ncore() (24) to
    # use both vector lanes per AI core — 2x the vector-core parallelism.
    kernel_swiglu_bwd[(nvec(), 1, 1)](
        grad_swiglu, grad_swiglu.stride(0), fc1_output, fc1_output.stride(0),
        ffn, recv_weights_sorted, dAB, dscale, M, BLOCK_SIZE=BLOCK_SIZE, num_warps=8,
        use_bytecode=True)
    return dAB, dscale
