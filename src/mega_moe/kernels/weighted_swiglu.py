# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Ascend Triton kernel for routing-weighted gated activation (SwiGLU / SiTU-GLU).

The FC1 output is packed as ``[M, 2 * F]`` with the gate projection in the
first half and the up projection in the second half.  The activation is applied
in FP32 and the per-row FP32 routing weight is folded in before the BF16 store.

Two gated activations are selectable at compile time via ``ACTIVATION``:

* ``swiglu``  (0):  ``silu(gate) * up``               = ``gate * sigmoid(gate) * up``
* ``situglu`` (1):  ``beta * tanh(gate / beta) * sigmoid(gate) * up``
                    with an optional ``linear_beta * tanh(up / linear_beta)``
                    transform on ``up`` when ``HAS_LINEAR_BETA`` is set.

``ACTIVATION`` and ``HAS_LINEAR_BETA`` are ``tl.constexpr`` so the unused
branch is dead-code eliminated; ``situ_beta`` / ``situ_linear_beta`` are runtime
scalar arguments only read by the SiTU-GLU branch.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


_BLOCK_M = 8
_BLOCK_N = 128

# Compile-time activation ids (kept in sync with ``weighted_swiglu_forward``).
# Use literals inside the @jit body — Triton cannot reference module globals.
_SWIGLU = 0
_SITUGLU = 1


@triton.jit
def _weighted_activation_kernel(
    fc1_output_ptr,
    routing_weight_ptr,
    output_ptr,
    num_rows,
    ffn_dim,
    situ_beta,                                   # runtime float, SiTU-GLU only
    situ_linear_beta,                            # runtime float, SiTU-GLU + HAS_LINEAR_BETA only
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """Apply a gated activation and one routing scale to each dispatched token row."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_n_tiles = tl.cdiv(ffn_dim, BLOCK_N)
    num_m_tiles = tl.cdiv(num_rows, BLOCK_M)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in range(pid, num_tiles, num_programs):
        tile_m = tile_id // num_n_tiles
        tile_n = tile_id % num_n_tiles

        offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < num_rows
        mask_n = offs_n < ffn_dim
        mask = mask_m[:, None] & mask_n[None, :]

        # The host contract rejects non-contiguous inputs, and ``output`` is
        # allocated contiguous here.  Encode those layouts directly instead
        # of specializing a second copy of their fixed strides.
        gate_offsets = offs_m[:, None] * (2 * ffn_dim) + offs_n[None, :]
        up_offsets = gate_offsets + ffn_dim
        output_offsets = offs_m[:, None] * ffn_dim + offs_n[None, :]

        gate = tl.load(fc1_output_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(fc1_output_ptr + up_offsets, mask=mask, other=0.0).to(tl.float32)
        # FC1 is BF16.  Promote the inputs so the activation and transported
        # route scaling are evaluated in FP32; the output store is the only cast
        # back to BF16.
        if ACTIVATION == 0:
            # SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
            activated = gate * tl.sigmoid(gate) * up
        else:
            # SiTU-GLU: beta * tanh(gate / beta) * sigmoid(gate) * up
            situ_a = situ_beta * tl.math.tanh(gate / situ_beta) * tl.sigmoid(gate)
            if HAS_LINEAR_BETA:
                up = situ_linear_beta * tl.math.tanh(up / situ_linear_beta)
            activated = situ_a * up
        routing_weight = tl.load(
            routing_weight_ptr + offs_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        activated *= routing_weight[:, None]
        tl.store(output_ptr + output_offsets, activated, mask=mask)


def weighted_swiglu_forward(
    fc1_output: torch.Tensor,
    routing_weight_recv: torch.Tensor,
    num_cores: int,
    *,
    activation: str = "swiglu",
    situ_beta: float = 1.0,
    situ_linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Compute a routing-weighted gated activation for an Ascend FC1 result.

    Args:
        fc1_output: Contiguous BF16 tensor shaped ``[M, 2 * F]``.  Its
            first half is the gate projection and its second half is the up
            projection.
        routing_weight_recv: Contiguous FP32 tensor shaped ``[M]`` and laid
            out in the same dispatched-row order as ``fc1_output``.  The
            public routing-weight input remains FP32 throughout dispatch and
            this function consumes the transported FP32 payload directly.
        num_cores: Number of physical Vector-core programs to launch.  The
            caller obtains this from the device-owned ``NPUUtils`` query.
        activation: ``"swiglu"`` (default, ``silu(gate) * up``) or ``"situglu"``
            (``beta * tanh(gate / beta) * sigmoid(gate) * up``).
        situ_beta: Gate tanh width for SiTU-GLU.  Ignored for SwiGLU.
        situ_linear_beta: When not None, apply
            ``linear_beta * tanh(up / linear_beta)`` to the up projection
            (SiTU-GLU only).  Ignored for SwiGLU.
    Returns:
        Contiguous BF16 tensor shaped ``[M, F]`` containing the routing-weighted
        activation.  The activation and routing-weight multiplication are
        evaluated in FP32 and converted to BF16 on store.
    """
    if fc1_output.ndim != 2:
        raise ValueError(f"fc1_output must be 2D [M, 2F], got shape {tuple(fc1_output.shape)}")
    if fc1_output.dtype != torch.bfloat16:
        raise TypeError(f"fc1_output must have dtype torch.bfloat16, got {fc1_output.dtype}")
    if not fc1_output.is_contiguous():
        raise ValueError("fc1_output must be contiguous")

    num_rows, packed_dim = fc1_output.shape
    if packed_dim <= 0 or packed_dim % 2 != 0:
        raise ValueError(f"fc1_output last dimension must be a positive even value, got {packed_dim}")
    ffn_dim = packed_dim // 2

    if routing_weight_recv.ndim != 1 or routing_weight_recv.shape[0] != num_rows:
        raise ValueError(
            "routing_weight_recv must have shape [M] matching fc1_output; "
            f"got {tuple(routing_weight_recv.shape)} for M={num_rows}"
        )
    if routing_weight_recv.dtype != torch.float32:
        raise TypeError(
            "routing_weight_recv must have dtype torch.float32, "
            f"got {routing_weight_recv.dtype}"
        )
    if not routing_weight_recv.is_contiguous():
        raise ValueError("routing_weight_recv must be contiguous")
    if routing_weight_recv.device != fc1_output.device:
        raise ValueError(
            "routing_weight_recv and fc1_output must be on the same device; "
            f"got {routing_weight_recv.device} and {fc1_output.device}"
        )
    if not isinstance(num_cores, int) or isinstance(num_cores, bool) or num_cores <= 0:
        raise ValueError(f"num_cores must be a positive integer, got {num_cores!r}")

    if activation == "swiglu":
        act_id = _SWIGLU
    elif activation == "situglu":
        act_id = _SITUGLU
    else:
        raise ValueError(
            f"activation must be 'swiglu' or 'situglu', got {activation!r}"
        )
    if float(situ_beta) <= 0.0:
        raise ValueError("situ_beta must be positive")
    has_linear_beta = situ_linear_beta is not None
    if has_linear_beta and float(situ_linear_beta) <= 0.0:
        raise ValueError("situ_linear_beta must be positive when set")
    # The linear-beta value is only read inside the HAS_LINEAR_BETA branch; pass
    # a harmless 0.0 placeholder otherwise so the kernel signature stays uniform.
    linear_beta_val = float(situ_linear_beta) if has_linear_beta else 0.0

    output = torch.empty(
        (num_rows, ffn_dim),
        dtype=fc1_output.dtype,
        device=fc1_output.device,
    )
    if num_rows == 0:
        return output

    num_tiles = triton.cdiv(num_rows, _BLOCK_M) * triton.cdiv(ffn_dim, _BLOCK_N)
    # The launch count is supplied by the device-owned physical Vector-core
    # query.  Do not derive it from a user-controlled environment variable or
    # from the Cube count: 950DT reports 32 Cube and 64 Vector cores.
    num_programs = min(num_cores, num_tiles)
    _weighted_activation_kernel[(num_programs, )](
        fc1_output,
        routing_weight_recv,
        output,
        num_rows,
        ffn_dim,
        float(situ_beta),
        linear_beta_val,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        ACTIVATION=act_id,
        HAS_LINEAR_BETA=has_linear_beta,
    )
    return output


__all__ = ["weighted_swiglu_forward"]
