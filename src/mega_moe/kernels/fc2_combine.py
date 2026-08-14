# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BF16 Ascend Triton kernel for MoE FC2 and distributed combine.

The input rows are already grouped as local-expert major, then source-rank
major.  The production pipeline computes weighted SwiGLU on the Vector stream,
feeds FC2 on the Cube stream, and then performs route-output transport and the
final top-k reduction.  Vector programs resolve each source rank's symmetric
combine workspace and store contiguous source/expert FC2 tiles there before
reducing locally.
"""

import torch
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .weighted_swiglu import (
    _BLOCK_M as _WEIGHTED_BLOCK_M,
    _BLOCK_N as _WEIGHTED_BLOCK_N,
    _weighted_activation_expert_group_kernel,
)


_META_BLOCK = 256
_ROUTE_BLOCK = 256
_FC2_REMOTE_STORE_BLOCK = 4096
_FC2_TRANSPORT_BLOCK_M = 256


def _fc2_reduce_block_n(num_rows: int) -> int:
    """Return the validated reduction width for the current route count."""
    return 1024 if num_rows >= 1024 else 256


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
def _prepare_fc2_remote_store_metadata_kernel(
    counts_mem_ptr,
    recv_expert_offs_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    group_segment_start_ptr,
    group_segment_count_ptr,
    num_pull_slots,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    META_BLOCK: tl.constexpr,
    GROUP_EXPERTS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_INDEX_STRIDE: tl.constexpr,
):
    """Describe local FC2 tiles that are stored into each source rank.

    For a local expert, received rows are source-major.  The source rank's
    stable-send rows are global-expert major, so both offsets can be derived
    from the replicated count cube without another host-side collective.
    """
    # Grouped remote-store uses a tiny segment table instead of making every
    # Vector group scan the complete descriptor workspace.  The table stores
    # one ``(start, count)`` pair per ``(expert-group, source-rank)``.  This
    # metadata kernel has a single program, so resetting and updating the
    # counters is race-free and does not require a host synchronization or a
    # device atomic.
    for group_id in range(0, NUM_GROUPS):
        for source_rank in range(0, WORLD_SIZE):
            segment_id = group_id * GROUP_INDEX_STRIDE + source_rank
            tl.store(group_segment_start_ptr + segment_id, 0)
            tl.store(group_segment_count_ptr + segment_id, 0)

    meta_offs = tl.arange(0, META_BLOCK)
    for start in range(0, num_pull_slots, META_BLOCK):
        offs = start + meta_offs
        tl.store(pull_tile_rank_ptr + offs, -1, mask=offs < num_pull_slots)

    transport_cursor = 0
    # Each source gets its own stable-send cursor.  Scan global buckets in the
    # same order used by build_routing_plan and emit only this rank's buckets.
    for source_rank in range(0, WORLD_SIZE):
        remote_send_cursor = 0
        for bucket in range(0, WORLD_SIZE * EXPERTS_PER_RANK):
            route_count = tl.load(
                counts_mem_ptr + source_rank * NUM_BINS_PAD + bucket
            )
            destination_rank = bucket // EXPERTS_PER_RANK
            expert_id = bucket % EXPERTS_PER_RANK
            if destination_rank == LOCAL_RANK:
                source_local_start = tl.load(recv_expert_offs_ptr + expert_id)
                for prior_source in range(0, source_rank):
                    source_local_start += tl.load(
                        counts_mem_ptr
                        + prior_source * NUM_BINS_PAD
                        + bucket
                    )
                num_tiles = tl.cdiv(route_count, BLOCK_M)
                for tile_id in range(0, num_tiles):
                    tile_delta = tile_id * BLOCK_M
                    row_count = tl.minimum(
                        BLOCK_M, route_count - tile_delta
                    )
                    encoded_rank = source_rank + expert_id * WORLD_SIZE
                    tl.store(
                        pull_tile_rank_ptr + transport_cursor, encoded_rank
                    )
                    tl.store(
                        pull_tile_src_start_ptr + transport_cursor,
                        source_local_start + tile_delta,
                    )
                    tl.store(
                        pull_tile_dst_start_ptr + transport_cursor,
                        remote_send_cursor + tile_delta,
                    )
                    tl.store(
                        pull_tile_row_count_ptr + transport_cursor, row_count
                    )
                    # Descriptors are emitted source-major and expert-major,
                    # so one group's entries for a source are contiguous.
                    segment_id = (
                        (expert_id // GROUP_EXPERTS) * GROUP_INDEX_STRIDE
                        + source_rank
                    )
                    segment_count = tl.load(
                        group_segment_count_ptr + segment_id
                    )
                    if segment_count == 0:
                        tl.store(
                            group_segment_start_ptr + segment_id,
                            transport_cursor,
                        )
                    tl.store(
                        group_segment_count_ptr + segment_id,
                        segment_count + 1,
                    )
                    transport_cursor += 1
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
):
    """Compute one FC2 M/N tile for the coarse expert-group schedule."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rows = row_start + offs_m
    cols = n_tile * BLOCK_N + offs_n
    mask_m = offs_m < row_count
    mask_n = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    weight_base = weight_ptr + expert_id.to(tl.int64) * stride_weight_e
    for k_start in range(0, K, BLOCK_K):
        red = k_start + offs_k
        mask_k = red < K
        a_ptrs = (
            input_ptr
            + rows[:, None] * stride_input_m
            + red[None, :] * stride_input_k
        )
        b_ptrs = (
            weight_base
            + cols[None, :] * stride_weight_n
            + red[:, None] * stride_weight_k
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
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
def _kernel_fc2_expert_group(
    input_ptr,
    weight_ptr,
    peer_mem_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_peer_m,
    stride_peer_n,
    GROUP_FIRST_EXPERT: tl.constexpr,
    GROUP_LAST_EXPERT: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Cube-only FC2 for one contiguous expert group."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        num_n_tiles: tl.constexpr = N // BLOCK_N
        first_task = GROUP_FIRST_EXPERT * num_n_tiles
        last_task = GROUP_LAST_EXPERT * num_n_tiles
        for task_id in range(pid + first_task, last_task, ncore):
            expert_id = task_id // num_n_tiles
            n_tile = task_id % num_n_tiles
            expert_size = tl.load(recv_per_expert_ptr + expert_id)
            expert_off = tl.load(recv_expert_offs_ptr + expert_id)
            if expert_size > 0:
                num_m_windows = tl.cdiv(expert_size, BLOCK_M)
                for m_window in range(0, num_m_windows):
                    row_start = expert_off + m_window * BLOCK_M
                    row_count = tl.minimum(
                        BLOCK_M,
                        expert_size - m_window * BLOCK_M,
                    )
                    _fc2_gemm_one_mn_tile(
                        input_ptr,
                        weight_ptr,
                        peer_mem_ptr,
                        expert_id,
                        row_start,
                        row_count,
                        n_tile,
                        N,
                        K,
                        stride_input_m,
                        stride_input_k,
                        stride_weight_e,
                        stride_weight_n,
                        stride_weight_k,
                        stride_peer_m,
                        stride_peer_n,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                    )


@triton.jit
def _remote_store_one_tile(
    reverse_buf_ptr,
    peer_mem_ptr,
    src_start,
    dst_start,
    row_count,
    stride_reverse_m,
    peer_rank,
    N: tl.constexpr,
    REMOTE_STORE_BLOCK: tl.constexpr,
):
    """Store one contiguous FC2 descriptor into its source rank."""
    remote_reverse = libshmem_device.remote_ptr(reverse_buf_ptr, peer_rank)
    flat_offsets = tl.arange(0, REMOTE_STORE_BLOCK)
    num_elements = row_count * N
    source_base = peer_mem_ptr + src_start * N
    destination_base = remote_reverse + dst_start * stride_reverse_m
    for flat_start in range(0, num_elements, REMOTE_STORE_BLOCK):
        offsets = flat_start + flat_offsets
        mask = offsets < num_elements
        values = tl.load(
            source_base + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(destination_base + offsets, values, mask=mask)


@triton.jit
def _kernel_remote_store_transport_group(
    reverse_buf_ptr,
    peer_mem_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    group_segment_start_ptr,
    group_segment_count_ptr,
    stride_reverse_m,
    GROUP_ID: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    GROUP_INDEX_STRIDE: tl.constexpr,
    REMOTE_STORE_BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    """Vector-only remote store for one encoded expert group.

    Each group/source pair is contiguous in the descriptor stream emitted by
    ``_prepare_fc2_remote_store_metadata_kernel``.
    """
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            segment_base = GROUP_ID * GROUP_INDEX_STRIDE
            for source_phase in range(0, WORLD_SIZE):
                # Rotate peer order by pid to avoid all Vector programs
                # targeting the same remote rank in one phase.
                source_rank = (source_phase + pid + GROUP_ID) % WORLD_SIZE
                segment_id = segment_base + source_rank
                segment_start = tl.load(group_segment_start_ptr + segment_id)
                segment_count = tl.load(group_segment_count_ptr + segment_id)
                for segment_off in range(pid, segment_count, ncore):
                    tile_id = segment_start + segment_off
                    encoded_rank = tl.load(pull_tile_rank_ptr + tile_id)
                    if encoded_rank >= 0:
                        peer_rank = encoded_rank % WORLD_SIZE
                        src_start = tl.load(
                            pull_tile_src_start_ptr + tile_id
                        ).to(tl.int64)
                        dst_start = tl.load(
                            pull_tile_dst_start_ptr + tile_id
                        ).to(tl.int64)
                        row_count = tl.load(
                            pull_tile_row_count_ptr + tile_id
                        )
                        _remote_store_one_tile(
                            reverse_buf_ptr,
                            peer_mem_ptr,
                            src_start,
                            dst_start,
                            row_count,
                            stride_reverse_m,
                            peer_rank,
                            N,
                            REMOTE_STORE_BLOCK,
                        )


@triton.jit
def _kernel_remote_store_barrier():
    """Fence grouped Vector RMA with the backend's barrier-sized grid.

    On Ascend950DT, ``barrier_all_vec`` is tied to one participating Vector
    block per AI Core.  The transport itself can use all physical Vector
    programs, but putting the barrier inside a 64-block transport launch can
    leave the collective waiting forever.  Keep the data movement launch and
    the rank-wide barrier as separate, stream-ordered kernels; the latter is
    launched with the Cube/AICore-sized grid by the helper below.
    """
    libshmem_device.barrier_all_vec()


@triton.jit
def _split_route_rows_16(rows):
    """Split a 16-element route lookup into scalar values."""
    rows_even, rows_odd = tl.split(rows.reshape((8, 2)))
    rows_0mod4, rows_2mod4 = tl.split(rows_even.reshape((4, 2)))
    rows_1mod4, rows_3mod4 = tl.split(rows_odd.reshape((4, 2)))

    rows_0mod8, rows_4mod8 = tl.split(rows_0mod4.reshape((2, 2)))
    rows_2mod8, rows_6mod8 = tl.split(rows_2mod4.reshape((2, 2)))
    rows_1mod8, rows_5mod8 = tl.split(rows_1mod4.reshape((2, 2)))
    rows_3mod8, rows_7mod8 = tl.split(rows_3mod4.reshape((2, 2)))

    row_0, row_8 = tl.split(rows_0mod8)
    row_4, row_12 = tl.split(rows_4mod8)
    row_2, row_10 = tl.split(rows_2mod8)
    row_6, row_14 = tl.split(rows_6mod8)
    row_1, row_9 = tl.split(rows_1mod8)
    row_5, row_13 = tl.split(rows_5mod8)
    row_3, row_11 = tl.split(rows_3mod8)
    row_7, row_15 = tl.split(rows_7mod8)
    return (
        row_0,
        row_1,
        row_2,
        row_3,
        row_4,
        row_5,
        row_6,
        row_7,
        row_8,
        row_9,
        row_10,
        row_11,
        row_12,
        row_13,
        row_14,
        row_15,
    )


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
    """Reduce locally staged route rows after remote stores are visible."""
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
            if TOPK <= 16:
                route_offsets = tl.arange(0, 16)
                safe_route_offsets = tl.where(route_offsets < TOPK, route_offsets, 0)
                (
                    row_0,
                    row_1,
                    row_2,
                    row_3,
                    row_4,
                    row_5,
                    row_6,
                    row_7,
                    row_8,
                    row_9,
                    row_10,
                    row_11,
                    row_12,
                    row_13,
                    row_14,
                    row_15,
                ) = _split_route_rows_16(
                    tl.load(route_to_send_ptr + route_base + safe_route_offsets)
                )
            for col_start in range(0, N, BLOCK_N_REDUCE):
                cols = col_start + reduce_cols
                mask_n = cols < N
                acc = tl.zeros((BLOCK_N_REDUCE,), dtype=tl.float32)
                for topk_slot in tl.static_range(0, TOPK):
                    if TOPK <= 16:
                        send_row = (
                            row_0,
                            row_1,
                            row_2,
                            row_3,
                            row_4,
                            row_5,
                            row_6,
                            row_7,
                            row_8,
                            row_9,
                            row_10,
                            row_11,
                            row_12,
                            row_13,
                            row_14,
                            row_15,
                        )[topk_slot]
                    else:
                        send_row = tl.load(route_to_send_ptr + route_base + topk_slot)
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


def prepare_fc2_remote_store_metadata(
    counts_mem: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    num_pull_slots: int,
    *,
    local_rank: int,
    world_size: int,
    experts_per_rank: int,
    num_bins_pad: int,
    block_m: int,
    group_segment_starts: torch.Tensor,
    group_segment_counts: torch.Tensor,
    group_experts: int,
) -> None:
    """Build grouped descriptors for remote stores to source ranks."""
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
    if num_pull_slots <= 0:
        raise ValueError("num_pull_slots must be positive")
    for name, tensor in (
        ("pull_tile_rank", pull_tile_rank),
        ("pull_tile_src_start", pull_tile_src_start),
        ("pull_tile_dst_start", pull_tile_dst_start),
        ("pull_tile_row_count", pull_tile_row_count),
    ):
        if tensor.numel() < num_pull_slots:
            raise ValueError(f"{name} has fewer than num_pull_slots entries")
    if block_m < 16 or block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two no smaller than 16")

    if group_experts <= 0:
        raise ValueError("group_experts must be positive")
    if group_experts > experts_per_rank:
        raise ValueError("group_experts cannot exceed experts_per_rank")
    num_groups = (experts_per_rank + group_experts - 1) // group_experts
    metadata_device = counts_mem.device
    for name, tensor in (
        ("group_segment_starts", group_segment_starts),
        ("group_segment_counts", group_segment_counts),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32")
        if tensor.device != metadata_device:
            raise ValueError(f"{name} must be on {metadata_device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    required_segments = num_groups * world_size
    if (
        group_segment_starts.numel() < required_segments
        or group_segment_counts.numel() < required_segments
    ):
        raise ValueError(
            "group segment metadata is smaller than the configured group/rank table"
        )

    _prepare_fc2_remote_store_metadata_kernel[(1, )](
        counts_mem,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        group_segment_starts,
        group_segment_counts,
        num_pull_slots,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BINS_PAD=num_bins_pad,
        BLOCK_M=block_m,
        META_BLOCK=_META_BLOCK,
        GROUP_EXPERTS=group_experts,
        NUM_GROUPS=num_groups,
        GROUP_INDEX_STRIDE=world_size,
    )


def _launch_fc2_remote_store_pipeline(
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
    fc2_buf: torch.Tensor,
    peer_mem: torch.Tensor,
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    *,
    group_segment_starts: torch.Tensor,
    group_segment_counts: torch.Tensor,
    num_program_cores: int,
    num_vector_programs: int,
    block_m: int,
    block_n: int,
    block_k: int,
    world_size: int,
    pipeline_group_experts: int,
    cube_stream,
    vector_stream,
    group_events,
    start_event,
    done_event,
    activation_events,
    activation_fc1_output,
    activation_routing_weights,
    activation_id: int = 0,
    activation_situ_beta: float = 1.0,
    activation_situ_linear_beta: float = 0.0,
    activation_has_linear_beta: bool = False,
) -> None:
    """Overlap coarse Cube groups and Vector remote-store groups.

    The two streams are deliberately separate: Ascend950DT does not reliably
    publish a Cube-side GM atomic from a mixed kernel, while a normal stream
    event is ordered by the runtime and is cheap at the 7--14 group cadence.
    ``done_event`` is waited by the caller's current stream before reduction.
    """
    experts_per_rank = down_weight.shape[0]
    num_groups = (
        experts_per_rank + pipeline_group_experts - 1
    ) // pipeline_group_experts
    if num_vector_programs <= 0:
        raise ValueError("num_vector_programs must be positive")
    if cube_stream is None or vector_stream is None:
        raise ValueError("FC2 pipeline streams must be provided")
    if group_events is None or len(group_events) < num_groups:
        raise ValueError("FC2 pipeline requires one event per expert group")
    if start_event is None or done_event is None:
        raise ValueError("FC2 pipeline start and completion events must be provided")
    if activation_events is None or len(activation_events) < num_groups:
        raise ValueError("FC2 production pipeline requires one activation event per expert group")
    if activation_fc1_output is None or activation_routing_weights is None:
        raise ValueError("FC2 production pipeline requires shadow activation inputs")

    current_stream = torch.npu.current_stream(weighted_activation.device)
    start_event.record(current_stream)
    cube_stream.wait_event(start_event)
    vector_stream.wait_event(start_event)

    N = down_weight.shape[1]
    K = down_weight.shape[2]
    # Queue every activation group before transport. Cube consumes each group
    # as soon as its event fires while Vector produces later groups; transport
    # follows after the activation producer tail.
    with torch.npu.stream(vector_stream):
        for group_id in range(num_groups):
            _weighted_activation_expert_group_kernel[(num_vector_programs,)](
                activation_fc1_output,
                activation_routing_weights,
                weighted_activation,
                received_expert_offsets,
                group_id,
                K,
                activation_situ_beta,
                activation_situ_linear_beta,
                BLOCK_M=_WEIGHTED_BLOCK_M,
                BLOCK_N=_WEIGHTED_BLOCK_N,
                ACTIVATION=activation_id,
                HAS_LINEAR_BETA=activation_has_linear_beta,
                GROUP_EXPERTS=pipeline_group_experts,
                EXPERTS_PER_RANK=experts_per_rank,
            )
            activation_events[group_id].record(vector_stream)

    for group_id in range(num_groups):
        first_expert = group_id * pipeline_group_experts
        last_expert = min(first_expert + pipeline_group_experts, experts_per_rank)
        cube_stream.wait_event(activation_events[group_id])
        with torch.npu.stream(cube_stream):
            _kernel_fc2_expert_group[(num_program_cores, 1, 1)](
                weighted_activation,
                down_weight,
                peer_mem,
                received_routes_per_expert,
                received_expert_offsets,
                weighted_activation.stride(0),
                weighted_activation.stride(1),
                down_weight.stride(0),
                down_weight.stride(1),
                down_weight.stride(2),
                N,
                1,
                GROUP_FIRST_EXPERT=first_expert,
                GROUP_LAST_EXPERT=last_expert,
                N=N,
                K=K,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
            )
            group_events[group_id].record(cube_stream)

    with torch.npu.stream(vector_stream):
        for group_id in range(num_groups):
            vector_stream.wait_event(group_events[group_id])
            _kernel_remote_store_transport_group[(num_vector_programs, 1, 1)](
                fc2_buf,
                peer_mem,
                pull_tile_rank,
                pull_tile_src_start,
                pull_tile_dst_start,
                pull_tile_row_count,
                group_segment_starts,
                group_segment_counts,
                fc2_buf.stride(0),
                GROUP_ID=group_id,
                WORLD_SIZE=world_size,
                GROUP_INDEX_STRIDE=world_size,
                REMOTE_STORE_BLOCK=_FC2_REMOTE_STORE_BLOCK,
                N=N,
            )

    # All group transports are ordered on vector_stream.  A dedicated
    # barrier-sized launch now fences their RMA writes before the caller's
    # current stream is released and before local reduction reads them.
    with torch.npu.stream(vector_stream):
        _kernel_remote_store_barrier[(num_program_cores, 1, 1)]()

    done_event.record(vector_stream)
    current_stream.wait_event(done_event)


def _launch_fc2_combine(
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
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
    num_pull_slots: int,
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
    pipeline_group_experts: int,
    group_segment_starts: torch.Tensor,
    group_segment_counts: torch.Tensor,
    pipeline_cube_stream,
    pipeline_vector_stream,
    pipeline_group_events,
    pipeline_start_event,
    pipeline_done_event,
    pipeline_activation_events,
    activation_fc1_output,
    activation_routing_weights,
    activation_id: int = 0,
    activation_situ_beta: float = 1.0,
    activation_situ_linear_beta: float = 0.0,
    activation_has_linear_beta: bool = False,
) -> torch.Tensor:
    """Launch pipelined FC2, remote-store transport, and local reduction."""
    if weighted_activation.ndim != 2 or down_weight.ndim != 3:
        raise ValueError("weighted_activation must be [M, K] and down_weight must be [E, N, K]")
    if weighted_activation.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise TypeError("weighted_activation and down_weight must use torch.bfloat16")
    if weighted_activation.shape[1] != down_weight.shape[2]:
        raise ValueError("weighted_activation K must match down_weight K")
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
    experts_per_rank = down_weight.shape[0]
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
        pipeline_activation_events,
    )
    if any(value is None for value in activation_args):
        raise ValueError("production FC2 requires all shadow activation arguments")
    if activation_fc1_output.shape != (M, 2 * K):
        raise ValueError(
            "shadow activation FC1 output must have shape [M, 2 * K]"
        )
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
    if any(
        tensor.numel() < num_pull_slots
        for tensor in (
            pull_tile_rank,
            pull_tile_src_start,
            pull_tile_dst_start,
            pull_tile_row_count,
        )
    ):
        raise ValueError("pull metadata workspace is smaller than num_pull_slots")
    if num_pull_slots <= 0:
        raise ValueError("num_pull_slots must be positive")
    if topk <= 0 or num_program_cores <= 0:
        raise ValueError("topk and num_program_cores must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if pipeline_group_experts <= 0:
        raise ValueError("pipeline_group_experts must be positive")
    if pipeline_group_experts > experts_per_rank:
        raise ValueError("pipeline_group_experts cannot exceed experts_per_rank")
    num_groups = (
        experts_per_rank + pipeline_group_experts - 1
    ) // pipeline_group_experts
    required_segments = num_groups * world_size
    for name, tensor in (
        ("group_segment_starts", group_segment_starts),
        ("group_segment_counts", group_segment_counts),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32")
        if tensor.device != weighted_activation.device:
            raise ValueError(f"{name} must be on {weighted_activation.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.numel() < required_segments:
            raise ValueError(
                f"{name} is smaller than the configured group/rank table"
            )
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
    _launch_fc2_remote_store_pipeline(
        weighted_activation,
        down_weight,
        fc2_buf,
        peer_mem,
        received_routes_per_expert,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        num_program_cores=num_program_cores,
        num_vector_programs=num_vector_programs,
        group_segment_starts=group_segment_starts,
        group_segment_counts=group_segment_counts,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        world_size=world_size,
        pipeline_group_experts=pipeline_group_experts,
        cube_stream=pipeline_cube_stream,
        vector_stream=pipeline_vector_stream,
        group_events=pipeline_group_events,
        start_event=pipeline_start_event,
        done_event=pipeline_done_event,
        activation_events=pipeline_activation_events,
        activation_fc1_output=activation_fc1_output,
        activation_routing_weights=activation_routing_weights,
        activation_id=activation_id,
        activation_situ_beta=activation_situ_beta,
        activation_situ_linear_beta=activation_situ_linear_beta,
        activation_has_linear_beta=activation_has_linear_beta,
    )
    # The remote-store barrier fences every route row before local reduction.
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
    "prepare_fc2_remote_store_metadata",
]
