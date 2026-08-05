# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Ascend Triton kernel for routing-weighted SwiGLU activation."""

import torch
import triton
import triton.language as tl


_BLOCK_M = 8
_BLOCK_N = 128


@triton.jit
def _weighted_swiglu_kernel(
    fc1_output_ptr,
    routing_weight_ptr,
    output_ptr,
    num_rows,
    ffn_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply SwiGLU and one routing scale to each dispatched token row."""
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
        # FC1 is BF16.  Promote the inputs so SwiGLU and transported route
        # scaling are evaluated in FP32; the output store is the only cast
        # back to BF16.
        activated = gate * tl.sigmoid(gate) * up
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
) -> torch.Tensor:
    """Compute routing-weighted SwiGLU for an Ascend FC1 result.

    Args:
        fc1_output: Contiguous BF16 tensor shaped ``[M, 2 * F]``.  Its
            first half is the gate projection and its second half is the up
            projection.
        routing_weight_recv: Contiguous FP32 tensor shaped ``[M]`` and laid
            out in the same dispatched-row order as ``fc1_output``.  The
            public routing-weight input remains FP32 throughout dispatch and
            this function consumes the transported FP32 payload directly.
        num_cores: Maximum number of persistent Triton programs to launch.
    Returns:
        Contiguous BF16 tensor shaped ``[M, F]`` containing
        ``SiLU(gate) * up * routing_weight``.  SwiGLU and routing-weight
        multiplication are evaluated in FP32 and converted to BF16 on store.
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
    output = torch.empty(
        (num_rows, ffn_dim),
        dtype=fc1_output.dtype,
        device=fc1_output.device,
    )
    if num_rows == 0:
        return output

    num_tiles = triton.cdiv(num_rows, _BLOCK_M) * triton.cdiv(ffn_dim, _BLOCK_N)
    num_programs = min(num_cores, num_tiles)
    _weighted_swiglu_kernel[(num_programs, )](
        fc1_output,
        routing_weight_recv,
        output,
        num_rows,
        ffn_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
    )
    return output


__all__ = ["weighted_swiglu_forward"]
