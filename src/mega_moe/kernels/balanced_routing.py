# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoonEP balanced-routing metadata kernels and launcher."""

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


_ROUTE_BLOCK_SIZE = 256
_MAX_ROUTE_PROGRAMS = 64


@triton.jit
def _kernel_build_balanced_count_cube(
    tpe_all_ptr,
    alloc_cumsum_ptr,
    inverse_ptr,
    counts_ptr,
    local_expert_starts_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    EPN: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    TPE_ROW_STRIDE: tl.constexpr,
    COUNT_ROW_STRIDE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
    STORE_LOCAL_STARTS: tl.constexpr,
):
    """Write one destination's disjoint count-cube slice per program."""
    destination = tl.program_id(axis=0)
    physical_slots: tl.constexpr = 2 * EPN

    with al.scope(core_mode="vector", disable_auto_sync=True):
        # A destination program owns these slots for every source, so clearing
        # and populating them in the same program needs no cross-core barrier.
        slot_offsets = tl.arange(0, BLOCK_SLOTS)
        valid_slot = slot_offsets < physical_slots
        destination_base = destination * physical_slots
        for source in tl.static_range(0, R):
            tl.store(
                counts_ptr
                + source * COUNT_ROW_STRIDE
                + destination_base
                + slot_offsets,
                tl.zeros((BLOCK_SLOTS,), dtype=tl.int32),
                mask=valid_slot,
            )

        for expert_base in tl.static_range(0, E, BLOCK_E):
            expert = expert_base + tl.arange(0, BLOCK_E)
            valid_expert = expert < E
            allocation_hi = tl.load(
                alloc_cumsum_ptr + expert * R + destination,
                mask=valid_expert,
                other=0,
            )
            previous_destination = tl.maximum(destination - 1, 0)
            allocation_lo = tl.load(
                alloc_cumsum_ptr + expert * R + previous_destination,
                mask=valid_expert & (destination > 0),
                other=0,
            )

            owner = expert // EPN
            replica_slot = tl.load(
                inverse_ptr + destination * E + expert,
                mask=valid_expert,
                other=-1,
            )
            is_home = owner == destination
            mapped = valid_expert & (is_home | (replica_slot >= 0))
            physical_slot = tl.where(
                is_home,
                expert - owner * EPN,
                EPN + replica_slot,
            )

            source_lo = tl.zeros((BLOCK_E,), dtype=tl.int32)
            for source in tl.static_range(0, R):
                source_count = tl.load(
                    tpe_all_ptr + source * TPE_ROW_STRIDE + expert,
                    mask=valid_expert,
                    other=0,
                )
                source_hi = source_lo + source_count
                overlap = tl.maximum(
                    tl.minimum(source_hi, allocation_hi)
                    - tl.maximum(source_lo, allocation_lo),
                    0,
                )
                # A home slot names exactly one home expert.  B.3 also assigns
                # every replica slot to exactly one remote expert, so these
                # stores are unique and atomics would only add serialization.
                tl.store(
                    counts_ptr
                    + source * COUNT_ROW_STRIDE
                    + destination_base
                    + physical_slot,
                    overlap,
                    mask=mapped,
                )
                source_lo = source_hi

        if STORE_LOCAL_STARTS:
            running = 0
            for expert_base in tl.static_range(0, E, BLOCK_E):
                expert = expert_base + tl.arange(0, BLOCK_E)
                valid_expert = expert < E
                local_count = tl.load(
                    tpe_all_ptr + LOCAL_RANK * TPE_ROW_STRIDE + expert,
                    mask=valid_expert,
                    other=0,
                )
                local_start = (
                    running
                    + tl.cumsum(local_count, axis=0)
                    - local_count
                )
                tl.store(
                    local_expert_starts_ptr + expert,
                    local_start,
                    mask=valid_expert & (destination == 0),
                )
                running += tl.sum(local_count, axis=0)


@triton.jit
def _kernel_finalize_balanced_metadata(
    counts_ptr,
    send_bucket_starts_ptr,
    send_bucket_receive_offsets_ptr,
    receive_counts_ptr,
    received_per_slot_ptr,
    received_slot_offsets_ptr,
    R: tl.constexpr,
    EPN: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    COUNT_ROW_STRIDE: tl.constexpr,
    BLOCK_SLOTS: tl.constexpr,
):
    """Derive all small dispatch metadata directly from the int32 cube."""
    destination = tl.program_id(axis=0)
    physical_slots: tl.constexpr = 2 * EPN

    with al.scope(core_mode="vector", disable_auto_sync=True):
        source = tl.arange(0, R)
        slot = tl.arange(0, BLOCK_SLOTS)
        valid_slot = slot < physical_slots
        destination_base = destination * physical_slots
        counts = tl.load(
            counts_ptr
            + source[:, None] * COUNT_ROW_STRIDE
            + destination_base
            + slot[None, :],
            mask=valid_slot[None, :],
            other=0,
        )

        local_counts = tl.sum(
            tl.where(source[:, None] == LOCAL_RANK, counts, 0), axis=0
        )
        send_destination_base = 0
        for previous_destination in tl.static_range(0, R):
            previous_counts = tl.load(
                counts_ptr
                + LOCAL_RANK * COUNT_ROW_STRIDE
                + previous_destination * physical_slots
                + slot,
                mask=valid_slot,
                other=0,
            )
            send_destination_base += tl.where(
                previous_destination < destination,
                tl.sum(previous_counts, axis=0),
                0,
            )
        send_starts = (
            send_destination_base
            + tl.cumsum(local_counts, axis=0)
            - local_counts
        )
        tl.store(
            send_bucket_starts_ptr + destination_base + slot,
            send_starts,
            mask=valid_slot,
        )

        total_by_slot = tl.sum(counts, axis=0)
        slot_start = tl.cumsum(total_by_slot, axis=0) - total_by_slot
        source_prefix = tl.sum(
            tl.where(source[:, None] < LOCAL_RANK, counts, 0), axis=0
        )
        tl.store(
            send_bucket_receive_offsets_ptr + destination_base + slot,
            slot_start + source_prefix,
            mask=valid_slot,
        )

        is_local_destination = destination == LOCAL_RANK
        tl.store(
            receive_counts_ptr
            + source[:, None] * physical_slots
            + slot[None, :],
            counts,
            mask=is_local_destination & valid_slot[None, :],
        )
        tl.store(
            received_per_slot_ptr + slot,
            total_by_slot,
            mask=is_local_destination & valid_slot,
        )
        tl.store(
            received_slot_offsets_ptr + slot,
            slot_start,
            mask=is_local_destination & valid_slot,
        )
        tl.store(
            received_slot_offsets_ptr + physical_slots,
            tl.sum(total_by_slot, axis=0),
            mask=is_local_destination,
        )


@triton.jit
def _kernel_map_balanced_routes(
    sorted_experts_ptr,
    expert_order_ptr,
    tpe_all_ptr,
    alloc_cumsum_ptr,
    inverse_ptr,
    local_expert_starts_ptr,
    send_bucket_starts_ptr,
    send_token_indices_ptr,
    send_route_indices_ptr,
    num_routes,
    R: tl.constexpr,
    E: tl.constexpr,
    EPN: tl.constexpr,
    TOP_K: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    TPE_ROW_STRIDE: tl.constexpr,
    BLOCK_ROUTES: tl.constexpr,
):
    """Map stable expert order directly into the final dispatch workspaces."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_routes, BLOCK_ROUTES)
    destination_lanes = tl.arange(0, R)
    physical_slots: tl.constexpr = 2 * EPN

    with al.scope(core_mode="vector", disable_auto_sync=True):
        for tile_id in range(pid, num_tiles, num_programs):
            sorted_position = (
                tile_id * BLOCK_ROUTES + tl.arange(0, BLOCK_ROUTES)
            )
            valid_route = sorted_position < num_routes
            safe_sorted_position = tl.minimum(
                sorted_position, num_routes - 1
            )
            expert = tl.load(
                sorted_experts_ptr + safe_sorted_position,
                mask=valid_route,
                other=0,
            ).to(tl.int32)
            route_id = tl.load(
                expert_order_ptr + safe_sorted_position,
                mask=valid_route,
                other=0,
            ).to(tl.int32)
            local_start = tl.load(
                local_expert_starts_ptr + expert,
                mask=valid_route,
                other=0,
            )
            local_ordinal = sorted_position - local_start

            previous_rank_count = tl.zeros(
                (BLOCK_ROUTES,), dtype=tl.int32
            )
            for source in tl.static_range(0, LOCAL_RANK):
                previous_rank_count += tl.load(
                    tpe_all_ptr + source * TPE_ROW_STRIDE + expert,
                    mask=valid_route,
                    other=0,
                )
            global_ordinal = previous_rank_count + local_ordinal
            allocation_prefix = tl.load(
                alloc_cumsum_ptr
                + expert[:, None] * R
                + destination_lanes[None, :],
                mask=valid_route[:, None],
                other=0,
            )
            destination = tl.sum(
                (allocation_prefix <= global_ordinal[:, None]).to(tl.int32),
                axis=1,
            )
            # Ascend masked-load lowering may still form the physical address;
            # keep the tail lanes inside every table before issuing gathers.
            destination = tl.where(valid_route, destination, 0)
            previous_destination = tl.maximum(destination - 1, 0)
            allocation_lo = tl.load(
                alloc_cumsum_ptr + expert * R + previous_destination,
                mask=valid_route & (destination > 0),
                other=0,
            )

            owner = expert // EPN
            replica_slot = tl.load(
                inverse_ptr + destination * E + expert,
                mask=valid_route & (destination != owner),
                other=0,
            )
            physical_slot = tl.where(
                destination == owner,
                expert - owner * EPN,
                EPN + replica_slot,
            )
            bucket = destination * physical_slots + physical_slot
            within_bucket = global_ordinal - tl.maximum(
                previous_rank_count, allocation_lo
            )
            scatter_position = (
                tl.load(
                    send_bucket_starts_ptr + bucket,
                    mask=valid_route,
                    other=0,
                )
                + within_bucket
            )
            tl.store(
                send_token_indices_ptr + scatter_position,
                route_id // TOP_K,
                mask=valid_route,
            )
            tl.store(
                send_route_indices_ptr + scatter_position,
                route_id,
                mask=valid_route,
            )


def launch_balanced_routing_metadata(
    sorted_experts: torch.Tensor,
    expert_order: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    *,
    rank: int,
    top_k: int,
    count_rows: torch.Tensor,
    local_expert_starts: torch.Tensor,
    send_token_indices: torch.Tensor,
    send_route_indices: torch.Tensor,
    send_bucket_starts: torch.Tensor,
    send_bucket_receive_offsets: torch.Tensor,
    receive_counts_by_source_slot: torch.Tensor,
    received_routes_per_slot: torch.Tensor,
    received_slot_offsets: torch.Tensor,
) -> None:
    """Queue the MoonEP count, metadata, and route-map kernels."""
    if sorted_experts.device.type != "npu":
        raise ValueError("balanced routing is only available on NPU")
    world_size, num_experts = (int(value) for value in tpe_all.shape)
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    experts_per_rank = num_experts // world_size
    physical_slots = 2 * experts_per_rank
    num_buckets = world_size * physical_slots
    num_routes = int(sorted_experts.numel())
    if not 0 <= rank < world_size or top_k <= 0:
        raise ValueError("rank and top_k must describe the active EP input")

    expected = {
        "expert_order": ((num_routes,), None),
        "alloc_cumsum": ((num_experts, world_size), torch.int32),
        "inverse_experts_to_copy": (
            (world_size, num_experts),
            torch.int32,
        ),
        "count_rows": ((world_size, count_rows.shape[1]), torch.int32),
        "local_expert_starts": ((num_experts,), torch.int32),
        "send_bucket_starts": ((num_buckets,), torch.int32),
        "send_bucket_receive_offsets": ((num_buckets,), torch.int32),
        "receive_counts_by_source_slot": (
            (world_size, physical_slots),
            torch.int32,
        ),
        "received_routes_per_slot": ((physical_slots,), torch.int32),
        "received_slot_offsets": ((physical_slots + 1,), torch.int32),
    }
    tensors = {
        "expert_order": expert_order,
        "alloc_cumsum": alloc_cumsum,
        "inverse_experts_to_copy": inverse_experts_to_copy,
        "count_rows": count_rows,
        "local_expert_starts": local_expert_starts,
        "send_bucket_starts": send_bucket_starts,
        "send_bucket_receive_offsets": send_bucket_receive_offsets,
        "receive_counts_by_source_slot": receive_counts_by_source_slot,
        "received_routes_per_slot": received_routes_per_slot,
        "received_slot_offsets": received_slot_offsets,
    }
    device = sorted_experts.device
    for name, tensor in tensors.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape or (
            dtype is not None and tensor.dtype != dtype
        ):
            raise ValueError(f"{name} has an incompatible shape or dtype")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
    if count_rows.shape[1] < num_buckets:
        raise ValueError("count row stride is smaller than the bucket count")
    for name, tensor in (
        ("send_token_indices", send_token_indices),
        ("send_route_indices", send_route_indices),
    ):
        if (
            tensor.ndim != 1
            or tensor.numel() < num_routes
            or tensor.dtype != torch.int32
            or tensor.device != device
        ):
            raise ValueError(f"{name} cannot hold every balanced route")

    block_e = min(32, triton.next_power_of_2(num_experts))
    while num_experts % block_e:
        block_e //= 2
    block_slots = triton.next_power_of_2(physical_slots)
    _kernel_build_balanced_count_cube[(world_size, 1, 1)](
        tpe_all,
        alloc_cumsum,
        inverse_experts_to_copy,
        count_rows,
        local_expert_starts,
        R=world_size,
        E=num_experts,
        EPN=experts_per_rank,
        LOCAL_RANK=rank,
        TPE_ROW_STRIDE=tpe_all.stride(0),
        COUNT_ROW_STRIDE=count_rows.stride(0),
        BLOCK_E=block_e,
        BLOCK_SLOTS=block_slots,
        STORE_LOCAL_STARTS=True,
    )
    _kernel_finalize_balanced_metadata[(world_size, 1, 1)](
        count_rows,
        send_bucket_starts,
        send_bucket_receive_offsets,
        receive_counts_by_source_slot,
        received_routes_per_slot,
        received_slot_offsets,
        R=world_size,
        EPN=experts_per_rank,
        LOCAL_RANK=rank,
        COUNT_ROW_STRIDE=count_rows.stride(0),
        BLOCK_SLOTS=block_slots,
    )
    if num_routes:
        num_programs = min(
            _MAX_ROUTE_PROGRAMS,
            triton.cdiv(num_routes, _ROUTE_BLOCK_SIZE),
        )
        _kernel_map_balanced_routes[(num_programs, 1, 1)](
            sorted_experts,
            expert_order,
            tpe_all,
            alloc_cumsum,
            inverse_experts_to_copy,
            local_expert_starts,
            send_bucket_starts,
            send_token_indices,
            send_route_indices,
            num_routes,
            R=world_size,
            E=num_experts,
            EPN=experts_per_rank,
            TOP_K=top_k,
            LOCAL_RANK=rank,
            TPE_ROW_STRIDE=tpe_all.stride(0),
            BLOCK_ROUTES=_ROUTE_BLOCK_SIZE,
        )


__all__ = [
    "launch_balanced_routing_metadata",
]
