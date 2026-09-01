# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  swiglu_bwd.py  —  step 2: SwiGLU/SiTU-GLU backward kernel
#  (ported from kernels/nvidia/swiglu.py; SiTU branch mirrors the host repo
#   mindspeed_mm/fsdp/ops/glu/situ_triton.py backward kernel)
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
    situ_beta, situ_linear_beta,   # SiTU params (ignored when ACTIVATION == 0)
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,      # 0 = swiglu, 1 = situglu
    HAS_LINEAR_BETA: tl.constexpr,
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
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # activation value/derivative pair: out = act_a * v * scale
        #   swiglu:  act_a = silu(gate)            dact_a = silu'(gate)      v = up    dv = 1
        #   situglu: act_a = beta*tanh(g/beta)*s   dact_a = (1-t^2)*s
        #                                            + beta*t*s*(1-s)        v = lb*tanh(up/lb)
        #                                                                    dv = 1 - tanh^2(up/lb)
        # (tanh via the 2*sigmoid(2x)-1 rewrite, same as the host reference)
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


def swiglu_bwd_triton(grad_swiglu, fc1_output, recv_weights_sorted, *,
                      activation="swiglu", situ_beta=None, situ_linear_beta=None):
    """Step 2: returns (grad_fc1_output [M,2ffn], grad_gate [M]).

    ``activation`` selects the derivative: "swiglu" (default, silu'(gate)) or
    "situglu" (Kimi SiTU-GLU with ``situ_beta``/``situ_linear_beta``; the
    latter may be None for the untransformed-up variant)."""
    M, two_ffn = fc1_output.shape
    ffn = two_ffn // 2
    dAB = torch.empty_like(fc1_output)
    dscale = torch.empty(M, dtype=fc1_output.dtype, device=fc1_output.device)
    BLOCK_SIZE = triton.next_power_of_2(ffn)
    if activation in (None, "swiglu"):
        act, beta, lbeta, has_lb = 0, 1.0, 1.0, False
    elif activation == "situglu":
        act = 1
        beta = 1.0 if situ_beta is None else float(situ_beta)
        lbeta = 1.0 if situ_linear_beta is None else float(situ_linear_beta)
        has_lb = situ_linear_beta is not None
    else:
        raise ValueError(f"unknown activation for the backward: {activation!r}")
    # Pure-vector elementwise kernel: launch on nvec() (48) not ncore() (24) to
    # use both vector lanes per AI core — 2x the vector-core parallelism.
    kernel_swiglu_bwd[(nvec(), 1, 1)](
        grad_swiglu, grad_swiglu.stride(0), fc1_output, fc1_output.stride(0),
        ffn, recv_weights_sorted, dAB, dscale, M, beta, lbeta,
        BLOCK_SIZE=BLOCK_SIZE, ACTIVATION=act, HAS_LINEAR_BETA=has_lb,
        num_warps=8)
    return dAB, dscale
