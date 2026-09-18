# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BF16 Ascend Triton kernel for MoE FC2 and distributed combine (v0).

Step-1 fusion: the weighted SwiGLU activation, the FC2 GEMM, and the
device-put transport from ``old_fc2_combine.py`` are folded into one mixed
kernel (AIV0 -> AIC -> AIV1) while keeping the original numerical layout and
computation byte-for-byte identical.

The stage handoffs use coarse ``sync_block_all`` phase barriers plus a final
rank-wide ``barrier_all`` that fences the remote puts before the caller's
reduction kernel is released.  This preserves the old multi-kernel,
three-stream event chain's end-to-end data contract:
    weighted_activation -> peer_mem -> fc2_buf -> local top-k reduction.
"""

import os

import torch
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id
import triton.language.extra.cann.extension as al

from .weighted_swiglu import (
    _BLOCK_M as _WEIGHTED_BLOCK_M,
    _BLOCK_N as _WEIGHTED_BLOCK_N,
)


_ROUTE_BLOCK = 256
_ACLSHMEM_PUTMEM_MAX_BYTES = (1 << 32) - 1

# Event ids for the two cross-core handoffs.  Phase-level ``sync_block_all``
# barriers proved too weak here: the counting barrier outside the cube scope
# does not fence AIV0's GM stores against the Cube's phase-2 loads, so the
# FC2 read stale activation rows and the failed ``-full`` cases were
# nondeterministic across runs.  We use instead the semaphore
# ``sync_block_set/wait`` pairs on a per-item basis -- all three stages walk
# the same striped item sequence, so each program's handoffs pair 1:1:
#   event 0: AIV0 -> AIC  "activation window ready in GM"
#             (PIPE_MTE3 store -> PIPE_MTE2 load);
#   event 2: AIC -> AIV1  "FC2 window ready in GM peer_mem"
#             (PIPE_FIX store -> PIPE_MTE2 load).
_ACT_TO_FC2_EVENT: tl.constexpr = 0
_FC2_TO_PUT_EVENT: tl.constexpr = 2


def _fc2_reduce_block_n(num_rows: int) -> int:
    """Return the validated reduction width for the current route count."""
    return 1024 if num_rows >= 1024 else 256


def _validate_putmem_descriptor_capacity(max_rows: int, row_width: int) -> None:
    """Reject a possible BF16 descriptor that exceeds putmem's uint32 ABI."""
    if max_rows < 0 or row_width <= 0:
        raise ValueError("max_rows must be non-negative and row_width positive")
    if max_rows * row_width * 2 > _ACLSHMEM_PUTMEM_MAX_BYTES:
        raise ValueError(
            "device-put FC2 capacity exceeds the ACLSHMEM uint32 "
            "byte-count ABI; reduce max_tokens_per_rank/top_k/hidden_size"
        )


@triton.jit
def _fill_route_to_send_kernel(route_to_send_ptr, num_routes, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(route_to_send_ptr + offs, -1, mask=offs < num_routes)


@triton.jit
def _scatter_route_to_send_kernel(send_route_idx_ptr, route_to_send_ptr, num_send, BLOCK: tl.constexpr):
    sorted_offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = sorted_offs < num_send
    route_ids = tl.load(send_route_idx_ptr + sorted_offs, mask=mask, other=0)
    tl.store(route_to_send_ptr + route_ids, sorted_offs, mask=mask)


@triton.jit
def _prepare_fc2_device_put_metadata_kernel(
    counts_mem_ptr,
    recv_expert_offs_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    """Describe each local expert/source segment for device-put transport.

    For a local expert, received rows are source-major.  The source rank's
    stable-send rows are global-expert major, so both offsets can be derived
    from the replicated count cube without another host-side collective.
    """
    source_rank = tl.program_id(axis=0)
    remote_send_cursor = 0
    for bucket in range(0, WORLD_SIZE * EXPERTS_PER_RANK):
        route_count = tl.load(
            counts_mem_ptr + source_rank * NUM_BINS_PAD + bucket
        )
        destination_rank = bucket // EXPERTS_PER_RANK
        expert_id = bucket % EXPERTS_PER_RANK
        if destination_rank == LOCAL_RANK:
            source_local_start = tl.load(recv_expert_offs_ptr + expert_id)
            for prior_source in range(0, WORLD_SIZE):
                source_local_start += tl.load(
                    counts_mem_ptr
                    + prior_source * NUM_BINS_PAD
                    + bucket,
                    mask=prior_source < source_rank,
                    other=0,
                )
            segment_id = expert_id * WORLD_SIZE + source_rank
            tl.store(
                pull_tile_rank_ptr + segment_id,
                tl.where(route_count > 0, source_rank, -1),
            )
            tl.store(
                pull_tile_src_start_ptr + segment_id,
                source_local_start,
            )
            tl.store(
                pull_tile_dst_start_ptr + segment_id,
                remote_send_cursor,
            )
            tl.store(
                pull_tile_row_count_ptr + segment_id,
                route_count,
            )
        remote_send_cursor += route_count


@triton.jit
def _fc2_gemm_one_mn_tile(
    input_ptr,
    weight_ptr,
    fc2_buf_ptr,
    expert_id,
    row_start,
    row_count,
    n_tile,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_fc2_m,
    stride_fc2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    WEIGHT_NK_LOAD: tl.constexpr = False,
):
    """Compute one FC2 M/N tile for the coarse expert-group schedule."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    row_start64 = row_start.to(tl.int64)
    rows = row_start64 + offs_m.to(tl.int64)
    cols = n_tile * BLOCK_N + offs_n
    mask_m = offs_m < row_count
    mask_n = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    weight_expert = expert_id.to(tl.int64) - WEIGHT_EXPERT_BASE
    weight_base = weight_ptr + weight_expert * stride_weight_e
    for k_start in range(0, K, BLOCK_K):
        red = k_start + offs_k
        mask_k = red < K
        a_ptrs = (
            input_ptr
            + rows[:, None] * stride_input_m
            + red[None, :] * stride_input_k
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        if WEIGHT_NK_LOAD:
            # Preserve the contiguous K dimension after consume_token. The
            # acquired pointer otherwise takes an implicit-transpose lowering
            # that miscompiles the replica weight load on this Ascend backend.
            b_nk = tl.load(
                weight_base + cols[:, None] * stride_weight_n
                + red[None, :] * stride_weight_k,
                mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            b = tl.trans(b_nk)
        else:
            b = tl.load(
                weight_base + cols[None, :] * stride_weight_n
                + red[:, None] * stride_weight_k,
                mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
    out_ptrs = (
        fc2_buf_ptr
        + rows[:, None] * stride_fc2_m
        + cols[None, :] * stride_fc2_n
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _kernel_fc2_combine_v0_mix(
    fc1_output_ptr,
    routing_weight_ptr,
    down_weight_ptr,
    replica_down_weight_ptr,
    peer_mem_ptr,
    fc2_buf_ptr,
    weighted_activation_ptr,
    item_expert_ptr,
    item_row_base_ptr,
    item_row_count_ptr,
    item_count_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    stride_activation_m,
    stride_activation_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_reverse_m,
    N: tl.constexpr,
    K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    ACTIVE_EXPERTS: tl.constexpr,
    HOME_EXPERTS: tl.constexpr,
    ACT_BLOCK_M: tl.constexpr,
    ACT_BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
):
    """Single-launch per-item pipelined CV mix (AIV0 -> AIC -> AIV1).

    ``peer_mem`` is the flat BF16 FC2 mailbox (expert-major received rows).
    ``fc2_buf`` is the reverse buffer in stable send-row order on the
    destination ranks.  Every stage walks the same striped item sequence
    ``range(pid, num_items, ncore)`` over an item table of (expert,
    row_base, row_count) M-windows (``row_count <= BLOCK_M``), so a given
    program produces, computes, and ships the *same* items and the per-item
    semaphore pairs are 1:1 on that program (no cross-program id stealing):

      event 0: AIV0 -> AIC  "activation window ready in GM"
               (PIPE_MTE3 store -> PIPE_MTE2 load);
      event 2: AIC -> AIV1  "FC2 window ready in peer_mem"
               (PIPE_FIX store -> PIPE_MTE2 load).

    The tail ``barrier_all_vec`` fences every remote put before the caller's
    final local-top-k reduction reads ``fc2_buf``.
    """
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    sub = sub_vec_id()
    num_items = tl.load(item_count_ptr)

    # AIV0 (sub-block 0): weighted gated activation for each owned item window.
    # Each item holds up to ``BLOCK_M`` FC2 rows (item table granularity), so
    # the rows are produced in ``ACT_BLOCK_M``-row sub-tiles; the whole item's
    # activation is visible in GM before a single set releases the Cube.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        for item in range(pid, num_items, ncore):
            if sub == 0:
                rb = tl.load(item_row_base_ptr + item).to(tl.int64)
                rc = tl.load(item_row_count_ptr + item)
                for m0 in range(0, rc, ACT_BLOCK_M):
                    rcb = tl.minimum(ACT_BLOCK_M, rc - m0)
                    offs_m = tl.arange(0, ACT_BLOCK_M)
                    row64 = rb + m0 + offs_m.to(tl.int64)
                    mask_m = offs_m < rcb
                    weight = tl.load(
                        routing_weight_ptr + row64, mask=mask_m, other=0.0
                    ).to(tl.float32)
                    for k_base in range(0, K, ACT_BLOCK_N):
                        offs_n = k_base + tl.arange(0, ACT_BLOCK_N)
                        k_mask = offs_n < K
                        m = mask_m[:, None] & k_mask[None, :]
                        gate = tl.load(
                            fc1_output_ptr + row64[:, None] * (2 * K)
                            + offs_n[None, :],
                            mask=m, other=0.0,
                        ).to(tl.float32)
                        up = tl.load(
                            fc1_output_ptr + row64[:, None] * (2 * K)
                            + K + offs_n[None, :],
                            mask=m, other=0.0,
                        ).to(tl.float32)
                        if ACTIVATION == 0:
                            activated = gate * tl.sigmoid(gate) * up
                        else:
                            situ_a = (
                                SITU_BETA * tl.math.tanh(gate / SITU_BETA)
                                * tl.sigmoid(gate)
                            )
                            if HAS_LINEAR_BETA:
                                up = SITU_LINEAR_BETA * tl.math.tanh(
                                    up / SITU_LINEAR_BETA
                                )
                            activated = situ_a * up
                        out = (activated * weight[:, None]).to(tl.bfloat16)
                        tl.store(
                            weighted_activation_ptr
                            + row64[:, None] * stride_activation_m
                            + offs_n[None, :] * stride_activation_k,
                            out,
                            mask=m,
                        )
            # Activation rows for this item are now visible in GM.
            al.sync_block_set(
                "vector", "cube", _ACT_TO_FC2_EVENT,
                al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2,
            )

    # AIC (cube): FC2 for each owned item once its activation is ready.
    with al.scope(core_mode="cube", disable_auto_sync=True):
        for item in range(pid, num_items, ncore):
            al.sync_block_wait(
                "vector", "cube", _ACT_TO_FC2_EVENT,
                al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE2,
            )
            expert_id = tl.load(item_expert_ptr + item)
            rb = tl.load(item_row_base_ptr + item).to(tl.int64)
            rc = tl.load(item_row_count_ptr + item)
            for n_tile in tl.static_range(0, N // BLOCK_N):
                if expert_id < HOME_EXPERTS:
                    _fc2_gemm_one_mn_tile(
                        weighted_activation_ptr,
                        down_weight_ptr,
                        peer_mem_ptr,
                        expert_id,
                        rb,
                        rc,
                        n_tile,
                        N,
                        K,
                        stride_activation_m,
                        stride_activation_k,
                        stride_weight_e,
                        stride_weight_n,
                        stride_weight_k,
                        N,
                        1,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        WEIGHT_EXPERT_BASE=0,
                    )
                else:
                    _fc2_gemm_one_mn_tile(
                        weighted_activation_ptr,
                        replica_down_weight_ptr,
                        peer_mem_ptr,
                        expert_id,
                        rb,
                        rc,
                        n_tile,
                        N,
                        K,
                        stride_activation_m,
                        stride_activation_k,
                        stride_weight_e,
                        stride_weight_n,
                        stride_weight_k,
                        N,
                        1,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        WEIGHT_EXPERT_BASE=HOME_EXPERTS,
                    )
            # FC2 rows for this item are now visible in peer_mem.
            al.sync_block_set(
                "cube", "vector", _FC2_TO_PUT_EVENT,
                al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2,
            )

    # AIV1 (sub-block 1): descriptor-based RMA transport once FC2 is ready.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub == 1:
            for item in range(pid, num_items, ncore):
                al.sync_block_wait(
                    "cube", "vector", _FC2_TO_PUT_EVENT,
                    al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2,
                )
                expert = tl.load(item_expert_ptr + item)
                rb = tl.load(item_row_base_ptr + item).to(tl.int64)
                rc = tl.load(item_row_count_ptr + item)
                seg_base = expert * WORLD_SIZE
                for source in tl.static_range(0, WORLD_SIZE):
                    seg_id = seg_base + source
                    rank = tl.load(pull_tile_rank_ptr + seg_id)
                    seg_len = tl.load(
                        pull_tile_row_count_ptr + seg_id
                    ).to(tl.int64)
                    if seg_len > 0:
                        seg_lo = tl.load(
                            pull_tile_src_start_ptr + seg_id
                        ).to(tl.int64)
                        ov_lo = tl.maximum(rb, seg_lo)
                        ov_hi = tl.minimum(rb + rc, seg_lo + seg_len)
                        ov_len = ov_hi - ov_lo
                        if ov_len > 0:
                            local_off = ov_lo - seg_lo
                            dst_start = tl.load(
                                pull_tile_dst_start_ptr + seg_id
                            ).to(tl.int64)
                            libshmem_device.putmem(
                                fc2_buf_ptr
                                + (dst_start + local_off) * stride_reverse_m,
                                peer_mem_ptr + ov_lo * N,
                                ov_len * N * 2,
                                rank % WORLD_SIZE,
                            )

    # Fence every remote put before the caller's reduction kernel can run.
    # ``barrier_all_vec`` is tied to one Vector block per AI-Core and is
    # intentionally kept as a rank-wide tail (mirrors the proven transport
    # barrier pattern); the per-item semaphores above already serialize the
    # three stages on each program.
    libshmem_device.barrier_all_vec()


@triton.jit
def _kernel_local_topk_reduce(
    fc2_buf_ptr,
    route_to_send_ptr,
    output_ptr,
    batch_size,
    num_send,
    stride_fc2_m,
    stride_fc2_n,
    stride_output_m,
    stride_output_n,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N_REDUCE: tl.constexpr,
):
    """Reduce locally staged route rows after remote puts are visible."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)

    with al.scope(core_mode="vector", disable_auto_sync=True):
        # A pure AIV launch has one Vector core per program.  Unlike a mixed
        # Cube/Vector kernel, there is no second sub-vector worker to address
        # with sub_vec_id(); distribute tokens directly over the launch grid.
        reduce_cols = tl.arange(0, BLOCK_N_REDUCE)
        for token_id in range(pid, batch_size, ncore):
            token_id64 = token_id.to(tl.int64)
            route_base = token_id * TOPK
            for col_start in range(0, N, BLOCK_N_REDUCE):
                cols = col_start + reduce_cols
                mask_n = cols < N
                acc = tl.zeros((BLOCK_N_REDUCE,), dtype=tl.float32)
                for topk_slot in tl.static_range(0, TOPK):
                    # Scalar route loads keep the send-row indices in
                    # single-result ops: the multi-result tl.split
                    # de-interleave they replace trips 910B1's
                    # TritonToLinalg UseAnalysis, while caching the scalars
                    # in a Python tuple carried across tl.static_range
                    # iterations fails the 950DT frontend with NameError.
                    # Load straight into the consumer; the re-loads hit L1.
                    send_row = tl.load(
                        route_to_send_ptr + route_base + topk_slot
                    )
                    valid = (send_row >= 0) & (send_row < num_send)
                    safe_row = tl.where(valid, send_row, 0).to(tl.int64)
                    values = tl.load(
                        fc2_buf_ptr
                        + safe_row * stride_fc2_m
                        + cols * stride_fc2_n,
                        mask=mask_n & valid,
                        other=0.0,
                    ).to(tl.float32)
                    acc += values
                tl.store(
                    output_ptr
                    + token_id64 * stride_output_m
                    + cols * stride_output_n,
                    acc.to(tl.bfloat16),
                    mask=mask_n,
                )


def build_route_to_send(send_route_idx: torch.Tensor, route_to_send: torch.Tensor) -> torch.Tensor:
    """Build flattened-route -> stable send-row mapping, using -1 for drops."""
    if send_route_idx.dtype != torch.int32 or route_to_send.dtype != torch.int32:
        raise TypeError("route indices and route_to_send must use torch.int32")
    if send_route_idx.ndim != 1 or route_to_send.ndim != 1:
        raise ValueError("route indices and route_to_send must both be 1D")
    if send_route_idx.device != route_to_send.device:
        raise ValueError("route indices and route_to_send must be on the same device")
    if not send_route_idx.is_contiguous() or not route_to_send.is_contiguous():
        raise ValueError("route indices and route_to_send must be contiguous")
    num_routes = route_to_send.numel()
    if send_route_idx.numel() > num_routes:
        raise ValueError("the valid send count cannot exceed the flattened route count")
    if num_routes:
        _fill_route_to_send_kernel[(triton.cdiv(num_routes, _ROUTE_BLOCK), )](
            route_to_send, num_routes, BLOCK=_ROUTE_BLOCK)
    num_send = send_route_idx.numel()
    if num_send:
        _scatter_route_to_send_kernel[(triton.cdiv(num_send, _ROUTE_BLOCK), )](
            send_route_idx, route_to_send, num_send, BLOCK=_ROUTE_BLOCK)
    return route_to_send


def prepare_fc2_device_put_metadata(
    counts_mem: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    *,
    local_rank: int,
    world_size: int,
    experts_per_rank: int,
    num_bins_pad: int,
) -> None:
    """Build one device-put descriptor per local expert and source rank."""
    named_tensors = {
        "counts_mem": counts_mem,
        "received_expert_offsets": received_expert_offsets,
        "pull_tile_rank": pull_tile_rank,
        "pull_tile_src_start": pull_tile_src_start,
        "pull_tile_dst_start": pull_tile_dst_start,
        "pull_tile_row_count": pull_tile_row_count,
    }
    device = counts_mem.device
    for name, tensor in named_tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if world_size <= 0 or experts_per_rank <= 0:
        raise ValueError("world_size and experts_per_rank must be positive")
    if not 0 <= local_rank < world_size:
        raise ValueError("local_rank must be in [0, world_size)")
    if num_bins_pad < world_size * experts_per_rank:
        raise ValueError("num_bins_pad is too small for the global expert buckets")
    if counts_mem.numel() < world_size * num_bins_pad:
        raise ValueError("counts_mem is smaller than the padded count cube")
    if received_expert_offsets.numel() < experts_per_rank + 1:
        raise ValueError(
            "received_expert_offsets must contain experts_per_rank + 1 entries"
        )
    required_slots = world_size * experts_per_rank
    for name, tensor in (
        ("pull_tile_rank", pull_tile_rank),
        ("pull_tile_src_start", pull_tile_src_start),
        ("pull_tile_dst_start", pull_tile_dst_start),
        ("pull_tile_row_count", pull_tile_row_count),
    ):
        if tensor.numel() < required_slots:
            raise ValueError(
                f"{name} must provide one slot per expert/source pair"
            )

    _prepare_fc2_device_put_metadata_kernel[(world_size, )](
        counts_mem,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BINS_PAD=num_bins_pad,
    )


def _build_fc2_item_table(
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    active_experts_per_rank: int,
    block_m: int,
    num_received_routes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (expert, M-window) work items from the received-row layout.

    One entry is emitted for every non-empty ``block_m``-row window of each
    active expert, in expert-major / M-window-minor order.  AIV0, AIC, and AIV1
    all consume this exact item sequence (striped across AI cores), which is
    what lets one per-core semaphore pair hand the activation / FC2 results
    between the stages without cross-pid signal interference.

    The table is built by a device kernel by default (see
    ``_build_fc2_item_table_device``): the hot path must contain zero
    ``.item()`` calls -- every ``.item()`` drains the whole stream queue, and
    the previous host-side build turned the gap between
    ``prepare_fc2_device_put_metadata`` and the mixed-kernel launch into a
    multi-millisecond void where the device sat idle.  Set
    ``FC2_V1_HOST_ITEM_TABLE=1`` to restore the synchronous host build for
    one-variable A/B.
    """
    if os.environ.get("FC2_V1_HOST_ITEM_TABLE"):
        return _build_fc2_item_table_host(
            received_routes_per_expert,
            received_expert_offsets,
            active_experts_per_rank,
            block_m,
        )
    return _build_fc2_item_table_device(
        received_routes_per_expert,
        received_expert_offsets,
        active_experts_per_rank,
        block_m,
        num_received_routes,
    )


def _build_fc2_item_table_host(
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    active_experts_per_rank: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synchronous host-side item-table build (diagnostic fallback)."""
    device = received_routes_per_expert.device
    expert, rb, rc = [], [], []
    for e in range(active_experts_per_rank):
        cnt = int(received_routes_per_expert[e].item())
        base = int(received_expert_offsets[e].item())
        for m in range(0, (cnt + block_m - 1) // block_m):
            expert.append(e)
            rb.append(base + m * block_m)
            rc.append(min(block_m, cnt - m * block_m))
    n = len(expert)
    return (
        torch.as_tensor(expert, dtype=torch.int32, device=device),
        torch.as_tensor(rb, dtype=torch.int32, device=device),
        torch.as_tensor(rc, dtype=torch.int32, device=device),
        torch.as_tensor([n], dtype=torch.int32, device=device),
    )


def _build_fc2_item_table_device(
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    active_experts_per_rank: int,
    block_m: int,
    num_received_routes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the item table entirely on device -- no host sync on the hot path.

    One program per active expert recomputes the expert-major item prefix
    from the received counts (a scalar O(E) loop over at most a few dozen
    experts), stores its own windows, and the last program stores the total.
    Semantics match ``_build_fc2_item_table_host`` entry for entry: same
    expert-major / M-window-minor order, same tail-window row-count shrink,
    zero items for an empty expert.

    Workspace sizing note: the table's true length is only known on device
    (it depends on the per-expert received counts), so the buffers are
    allocated at the static worst-case bound held by the caller --
    ``num_received_routes`` entries (each M-window covers at least one
    received row and windows never overlap, so the emitted item count can
    never exceed the received-row count), floored at the expert count to
    keep the E-sized program grid of the degenerate empty case valid.
    Note ``received_routes_per_expert.numel()`` is the expert count, not a
    row bound -- the first version sized the table to it and the window
    stores ran off the end of an E-sized buffer, corrupting adjacent GM
    allocations that later surfaced as faults in ``_kernel_local_topk_reduce``.
    """
    device = received_routes_per_expert.device
    # TensorFactory warning noted elsewhere applies: base format is fine.
    table_rows = max(num_received_routes, active_experts_per_rank)
    expert = torch.empty(table_rows, dtype=torch.int32, device=device)
    row_base = torch.empty_like(expert)
    row_count = torch.empty_like(expert)
    item_count = torch.zeros(1, dtype=torch.int32, device=device)
    _build_fc2_item_table_kernel[(active_experts_per_rank, 1, 1)](
        received_routes_per_expert,
        received_expert_offsets,
        expert,
        row_base,
        row_count,
        item_count,
        ACTIVE_EXPERTS=active_experts_per_rank,
        BLOCK_M=block_m,
    )
    return expert, row_base, row_count, item_count


@triton.jit
def _build_fc2_item_table_kernel(
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    item_expert_ptr,
    item_row_base_ptr,
    item_row_count_ptr,
    item_count_ptr,
    ACTIVE_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Emit one M-window item per ``BLOCK_M`` rows of each active expert.

    Program ``e`` owns expert ``e``'s windows.  Every program first sums
    ``cdiv(count, BLOCK_M)`` over the *preceding* experts to find its
    expert-major prefix -- a scalar O(E) loop that costs nothing at
    E <= 32 -- so the flat item sequence is identical to the host build
    without any host synchronization.
    """
    expert_id = tl.program_id(0)
    prefix = tl.zeros((), dtype=tl.int32)
    for prior in range(0, ACTIVE_EXPERTS):
        prior_count = tl.load(recv_per_expert_ptr + prior)
        prior_windows = tl.cdiv(prior_count, BLOCK_M)
        if prior < expert_id:
            prefix += prior_windows
    count = tl.load(recv_per_expert_ptr + expert_id)
    base = tl.load(recv_expert_offs_ptr + expert_id)
    for window in range(0, tl.cdiv(count, BLOCK_M)):
        item = prefix + window
        tl.store(item_expert_ptr + item, expert_id)
        tl.store(item_row_base_ptr + item, base + window * BLOCK_M)
        tl.store(
            item_row_count_ptr + item,
            tl.minimum(BLOCK_M, count - window * BLOCK_M),
        )
    if expert_id == ACTIVE_EXPERTS - 1:
        total = tl.load(recv_per_expert_ptr + ACTIVE_EXPERTS - 1)
        windows = tl.cdiv(total, BLOCK_M)
        for prior in range(0, ACTIVE_EXPERTS - 1):
            prior_count = tl.load(recv_per_expert_ptr + prior)
            windows += tl.cdiv(prior_count, BLOCK_M)
        tl.store(item_count_ptr, windows)


def _launch_fc2_combine_v0_kernel(
    activation_fc1_output: torch.Tensor,
    activation_routing_weights: torch.Tensor,
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
    replica_down_weight: torch.Tensor,
    fc2_buf: torch.Tensor,
    peer_mem: torch.Tensor,
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    *,
    num_program_cores: int,
    block_m: int,
    block_n: int,
    block_k: int,
    world_size: int,
    home_experts_per_rank: int,
    active_experts_per_rank: int,
    activation_id: int,
    activation_situ_beta: float,
    activation_situ_linear_beta: float,
    activation_has_linear_beta: bool,
) -> None:
    """Launch the single CV activation/FC2/transport kernel."""
    K = weighted_activation.shape[1]
    N = down_weight.shape[1]
    launch_options = (
        {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"}
        if block_m * block_n > 128 * 256
        else {}
    )
    (
        item_expert,
        item_row_base,
        item_row_count,
        item_count,
    ) = _build_fc2_item_table(
        received_routes_per_expert,
        received_expert_offsets,
        active_experts_per_rank,
        block_m,
        num_received_routes=weighted_activation.shape[0],
    )
    _kernel_fc2_combine_v0_mix[(num_program_cores, 1, 1)](
        activation_fc1_output,
        activation_routing_weights,
        down_weight,
        replica_down_weight,
        peer_mem,
        fc2_buf,
        weighted_activation,
        item_expert,
        item_row_base,
        item_row_count,
        item_count,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        weighted_activation.stride(0),
        weighted_activation.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        down_weight.stride(2),
        fc2_buf.stride(0),
        N=N,
        K=K,
        WORLD_SIZE=world_size,
        ACTIVE_EXPERTS=active_experts_per_rank,
        HOME_EXPERTS=home_experts_per_rank,
        ACT_BLOCK_M=_WEIGHTED_BLOCK_M,
        ACT_BLOCK_N=_WEIGHTED_BLOCK_N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        ACTIVATION=activation_id,
        HAS_LINEAR_BETA=activation_has_linear_beta,
        SITU_BETA=activation_situ_beta,
        SITU_LINEAR_BETA=activation_situ_linear_beta,
        disable_auto_inject_block_sync=True,
        limit_auto_multi_buffer_buffer="no-limit",
        **launch_options,
    )


def _launch_fc2_combine(
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
    replica_down_weight: torch.Tensor,
    fc2_buf: torch.Tensor,
    peer_mem: torch.Tensor,
    route_to_send: torch.Tensor,
    output: torch.Tensor,
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    num_send: int,
    *,
    topk: int,
    num_program_cores: int,
    num_vector_programs: int,
    reduce_block_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    world_size: int,
    home_experts_per_rank: int,
    physical_experts_per_rank: int,
    active_experts_per_rank: int,
    pipeline_group_experts: int,
    pipeline_cube_stream,
    pipeline_vector_stream,
    pipeline_transfer_stream,
    pipeline_group_events,
    pipeline_start_event,
    pipeline_done_event,
    prefetch_done_event,
    pipeline_activation_events,
    activation_fc1_output,
    activation_routing_weights,
    activation_id: int = 0,
    activation_situ_beta: float = 1.0,
    activation_situ_linear_beta: float = 0.0,
    activation_has_linear_beta: bool = False,
    pipeline_group_ids: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Launch the fused FC2/combine kernel, then the local top-k reduction."""
    if weighted_activation.ndim != 2 or down_weight.ndim != 3:
        raise ValueError("weighted_activation must be [M, K] and down_weight must be [E, N, K]")
    if weighted_activation.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise TypeError("weighted_activation and down_weight must use torch.bfloat16")
    if weighted_activation.shape[1] != down_weight.shape[2]:
        raise ValueError("weighted_activation K must match down_weight K")
    if replica_down_weight.shape != down_weight.shape:
        raise ValueError("replica_down_weight must match down_weight shape")
    if replica_down_weight.dtype != down_weight.dtype:
        raise TypeError("replica_down_weight must match down_weight dtype")
    if down_weight.shape[0] != home_experts_per_rank:
        raise ValueError("down_weight must contain home_experts_per_rank rows")
    if physical_experts_per_rank not in (
        home_experts_per_rank,
        2 * home_experts_per_rank,
    ):
        raise ValueError(
            "physical_experts_per_rank must describe home-only or fixed-B slots"
        )
    if not home_experts_per_rank <= active_experts_per_rank <= physical_experts_per_rank:
        raise ValueError(
            "active_experts_per_rank must be between the home and physical counts"
        )
    M, K = weighted_activation.shape
    N = down_weight.shape[1]
    if (
        fc2_buf.ndim != 2
        or fc2_buf.shape[0] < num_send
        or fc2_buf.shape[1] != N
    ):
        raise ValueError("fc2_buf must provide one row per sent route")
    if fc2_buf.dtype != torch.bfloat16:
        raise TypeError("fc2_buf must use torch.bfloat16")
    if peer_mem.ndim != 1 or peer_mem.dtype != torch.bfloat16:
        raise ValueError("peer_mem must be a flat BF16 symmetric tensor")
    if output.ndim != 2 or output.shape[1] != N or output.dtype != torch.bfloat16:
        raise ValueError("output must be a BF16 tensor shaped [tokens, N]")
    if route_to_send.ndim != 1 or route_to_send.dtype != torch.int32:
        raise ValueError("route_to_send must be a flat torch.int32 tensor")
    if route_to_send.numel() != output.shape[0] * topk:
        raise ValueError("route_to_send length must equal output tokens * topk")
    if not 0 <= num_send <= route_to_send.numel():
        raise ValueError("num_send must be in [0, route_to_send.numel()]")
    if peer_mem.numel() < M * N:
        raise ValueError("peer_mem is too small for the symmetric FC2 rows")
    if fc2_buf.data_ptr() == peer_mem.data_ptr():
        raise ValueError(
            "symmetric combine workspace must not alias local FC2 rows"
        )
    experts_per_rank = physical_experts_per_rank
    if experts_per_rank <= 0:
        raise ValueError("down_weight must contain at least one local expert")
    if (
        received_routes_per_expert.ndim != 1
        or received_routes_per_expert.numel() < experts_per_rank
    ):
        raise ValueError(
            "received_routes_per_expert must contain one entry per local expert"
        )
    if (
        received_expert_offsets.ndim != 1
        or received_expert_offsets.numel() < experts_per_rank + 1
    ):
        raise ValueError(
            "received_expert_offsets must contain experts_per_rank + 1 entries"
        )
    tensors = (
        weighted_activation,
        down_weight,
        replica_down_weight,
        fc2_buf,
        peer_mem,
        route_to_send,
        output,
        received_routes_per_expert,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
    )
    if any(tensor.device != weighted_activation.device for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be contiguous")
    activation_args = (
        activation_fc1_output,
        activation_routing_weights,
    )
    if any(value is None for value in activation_args):
        raise ValueError("fused FC2 requires activation inputs")
    if activation_fc1_output.shape[1] != 2 * K:
        raise ValueError("shadow activation FC1 output must have shape [M, 2 * K]")
    if activation_fc1_output.ndim != 2:
        raise ValueError("shadow activation FC1 output must be a 2D tensor")
    if activation_fc1_output.dtype != torch.bfloat16:
        raise TypeError("shadow activation FC1 output must use torch.bfloat16")
    if activation_routing_weights.shape != (M,):
        raise ValueError("shadow activation routing weights must have shape [M]")
    if activation_routing_weights.dtype != torch.float32:
        raise TypeError("shadow activation routing weights must use torch.float32")
    if (
        activation_fc1_output.device != weighted_activation.device
        or activation_routing_weights.device != weighted_activation.device
    ):
        raise ValueError("shadow activation tensors must be on the FC2 device")
    if (
        not activation_fc1_output.is_contiguous()
        or not activation_routing_weights.is_contiguous()
    ):
        raise ValueError("shadow activation tensors must be contiguous")
    if activation_id not in (0, 1):
        raise ValueError("activation_id must select SwiGLU (0) or SiTU-GLU (1)")
    if activation_situ_beta <= 0.0:
        raise ValueError("activation_situ_beta must be positive")
    if activation_has_linear_beta and activation_situ_linear_beta <= 0.0:
        raise ValueError(
            "activation_situ_linear_beta must be positive when enabled"
        )
    metadata_tensors = (
        route_to_send,
        received_routes_per_expert,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
    )
    for tensor in metadata_tensors:
        if tensor.dtype != torch.int32:
            raise TypeError("all FC2/combine metadata tensors must use torch.int32")
    if topk <= 0 or num_program_cores <= 0:
        raise ValueError("topk and num_program_cores must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    required_slots = world_size * experts_per_rank
    if any(
        tensor.numel() < required_slots
        for tensor in (
            pull_tile_rank,
            pull_tile_src_start,
            pull_tile_dst_start,
            pull_tile_row_count,
        )
    ):
        raise ValueError(
            "pull metadata must provide one slot per expert/source pair"
        )
    if pipeline_group_experts <= 0:
        raise ValueError("pipeline_group_experts must be positive")
    if pipeline_group_experts > experts_per_rank:
        raise ValueError("pipeline_group_experts cannot exceed experts_per_rank")
    for name, block in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
    ):
        if block < 16 or block & (block - 1):
            raise ValueError(f"{name} must be a power of two no smaller than 16")
    if N % block_n or K % block_k:
        raise ValueError(
            "FC2 block_n and block_k must divide N and K on the current Ascend backend"
        )

    if reduce_block_n not in (256, 1024):
        raise ValueError("reduce_block_n must be the production value 256 or 1024")
    # The fused CV kernel replaces the three per-group launches.  It consumes
    # activation input directly, feeds FC2 through the GM working buffer, and
    # performs all descriptor puts before returning to the caller's stream.
    _launch_fc2_combine_v0_kernel(
        activation_fc1_output,
        activation_routing_weights,
        weighted_activation,
        down_weight,
        replica_down_weight,
        fc2_buf,
        peer_mem,
        received_routes_per_expert,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        num_program_cores=num_program_cores,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        world_size=world_size,
        home_experts_per_rank=home_experts_per_rank,
        active_experts_per_rank=active_experts_per_rank,
        activation_id=activation_id,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
        activation_has_linear_beta=activation_has_linear_beta,
    )

    # The fused kernel has already fenced remote writes.  Keep the validated
    # deterministic final route reduction on the caller stream.
    _kernel_local_topk_reduce[(num_vector_programs, 1, 1)](
        fc2_buf,
        route_to_send,
        output,
        output.shape[0],
        num_send,
        fc2_buf.stride(0),
        fc2_buf.stride(1),
        output.stride(0),
        output.stride(1),
        N=down_weight.shape[1],
        TOPK=topk,
        BLOCK_N_REDUCE=reduce_block_n,
    )
    return output


__all__ = [
    "build_route_to_send",
    "prepare_fc2_device_put_metadata",
]
