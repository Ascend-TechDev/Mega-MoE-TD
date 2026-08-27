# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Expert-group weighted gated activation for the FC2 shadow pipeline.

FC1 output is packed as ``[M, 2 * F]`` with gate followed by up. Activation
and FP32 route scaling are evaluated in FP32 before the BF16 output store.
``ACTIVATION`` and ``HAS_LINEAR_BETA`` are compile-time selectors so only the
configured SwiGLU or SiTU-GLU branch is emitted.
"""

import triton
import triton.language as tl


_BLOCK_M = 8
_BLOCK_N = 128


@triton.jit
def _weighted_activation_rows(
    fc1_output_ptr,
    routing_weight_ptr,
    output_ptr,
    row_start,
    num_rows,
    ffn_dim,
    situ_beta,
    situ_linear_beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """Apply one contiguous row range into its matching output range."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_n_tiles = tl.cdiv(ffn_dim, BLOCK_N)
    num_m_tiles = tl.cdiv(num_rows, BLOCK_M)
    num_tiles = num_m_tiles * num_n_tiles

    for tile_id in range(pid, num_tiles, num_programs):
        tile_m = tile_id // num_n_tiles
        tile_n = tile_id % num_n_tiles

        local_offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = local_offs_m < num_rows
        mask_n = offs_n < ffn_dim
        mask = mask_m[:, None] & mask_n[None, :]
        global_offs_m = local_offs_m.to(tl.int64) + row_start

        gate_offsets = global_offs_m[:, None] * (2 * ffn_dim) + offs_n[None, :]
        up_offsets = gate_offsets + ffn_dim
        output_offsets = global_offs_m[:, None] * ffn_dim + offs_n[None, :]

        gate = tl.load(fc1_output_ptr + gate_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(fc1_output_ptr + up_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if ACTIVATION == 0:
            activated = gate * tl.sigmoid(gate) * up
        else:
            situ_a = situ_beta * tl.math.tanh(gate / situ_beta) * tl.sigmoid(gate)
            if HAS_LINEAR_BETA:
                up = situ_linear_beta * tl.math.tanh(up / situ_linear_beta)
            activated = situ_a * up
        routing_weight = tl.load(
            routing_weight_ptr + global_offs_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        activated *= routing_weight[:, None]
        tl.store(output_ptr + output_offsets, activated, mask=mask)


@triton.jit(do_not_specialize=["group_id"])
def _weighted_activation_expert_group_kernel(
    fc1_output_ptr,
    routing_weight_ptr,
    output_ptr,
    received_expert_offsets_ptr,
    group_id,
    ffn_dim,
    situ_beta,
    situ_linear_beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    GROUP_EXPERTS: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Apply activation to one expert-major row group without a host D2H."""
    first_expert = group_id * GROUP_EXPERTS
    last_expert = tl.minimum(first_expert + GROUP_EXPERTS, EXPERTS_PER_RANK)
    row_start = tl.load(received_expert_offsets_ptr + first_expert)
    row_end = tl.load(received_expert_offsets_ptr + last_expert)
    _weighted_activation_rows(
        fc1_output_ptr,
        routing_weight_ptr,
        output_ptr,
        row_start,
        row_end - row_start,
        ffn_dim,
        situ_beta,
        situ_linear_beta,
        BLOCK_M,
        BLOCK_N,
        ACTIVATION,
        HAS_LINEAR_BETA,
    )
