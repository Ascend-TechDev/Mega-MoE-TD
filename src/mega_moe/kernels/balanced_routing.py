# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Small-grid device kernels for MoonEP balanced-routing metadata."""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def _kernel_build_balanced_count_cube(
    tpe_all_ptr,
    alloc_cumsum_ptr,
    experts_to_copy_ptr,
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
        # Masked lanes still form their physical address on this backend;
        # clamp every padded lane into its table before it reaches a pointer
        # (_kernel_map_balanced_routes keeps its gather lanes in range for
        # the same reason).
        slot_offsets = tl.arange(0, BLOCK_SLOTS)
        valid_slot = slot_offsets < physical_slots
        safe_slot_offsets = tl.minimum(slot_offsets, physical_slots - 1)
        destination_base = destination * physical_slots
        for source in tl.static_range(0, R):
            tl.store(
                counts_ptr
                + source * COUNT_ROW_STRIDE
                + destination_base
                + safe_slot_offsets,
                tl.zeros((BLOCK_SLOTS,), dtype=tl.int32),
                mask=valid_slot,
            )

        # Contiguous populate.  The 910B1 vector core drops masked and
        # data-dependent store offsets (the original expert-blocked scatter
        # faults or races at small-EPN shapes), so every store below is a
        # plain unmasked write to "own slice + lane", with the home/replica
        # choice folded into the VALUE via arithmetic selects.  Lanes past
        # 2*EPN fold back onto the slice tail as duplicate writers of a
        # value they already hold, because BLOCK_SLOTS can exceed
        # 2*EPN and spilling into the next destination's head slots races
        # with that program's stores.
        lane = tl.arange(0, BLOCK_SLOTS)
        off = lane - ((lane >= 2 * EPN).to(tl.int32)) * (
            BLOCK_SLOTS - 2 * EPN
        )
        home_sel = (off < EPN).to(tl.int32)
        rep_sel = ((off >= EPN) & (off < 2 * EPN)).to(tl.int32)
        home_expert = destination * EPN + tl.minimum(off, EPN - 1)
        copied_idx = tl.maximum(tl.minimum(off - EPN, EPN - 1), 0)
        copied = tl.load(
            experts_to_copy_ptr + destination * EPN + copied_idx
        )
        safe_copied = tl.maximum(copied, 0)
        sel_expert = home_sel * home_expert + rep_sel * safe_copied
        keep = home_sel + rep_sel * (copied >= 0).to(tl.int32)
        previous_destination = tl.maximum(destination - 1, 0)
        allocation_hi = tl.load(
            alloc_cumsum_ptr + sel_expert * R + destination
        )
        allocation_lo = tl.load(
            alloc_cumsum_ptr + sel_expert * R + previous_destination,
            mask=(destination > 0) & (lane >= 0),
            other=0,
        )
        source_lo = tl.zeros((BLOCK_SLOTS,), dtype=tl.int32)
        for source in tl.static_range(0, R):
            source_count = tl.load(
                tpe_all_ptr + source * TPE_ROW_STRIDE + sel_expert
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
                + off,
                overlap * keep,
            )
            source_lo = source_hi

        if STORE_LOCAL_STARTS:
            running = 0
            for expert_base in tl.static_range(0, E, BLOCK_E):
                expert = expert_base + tl.arange(0, BLOCK_E)
                valid_expert = expert < E
                safe_expert = tl.minimum(expert, E - 1)
                local_count = tl.load(
                    tpe_all_ptr + LOCAL_RANK * TPE_ROW_STRIDE + safe_expert,
                    mask=valid_expert,
                    other=0,
                )
                local_start = (
                    running
                    + tl.cumsum(local_count, axis=0)
                    - local_count
                )
                tl.store(
                    local_expert_starts_ptr + safe_expert,
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
        # Masked tail lanes still form their physical address on this
        # backend; clamp them into the cube so the padded lanes of the last
        # row cannot read past the buffer (_kernel_map_balanced_routes keeps
        # its gather lanes in range for the same reason).
        safe_slot = tl.minimum(slot, physical_slots - 1)
        destination_base = destination * physical_slots
        counts = tl.load(
            counts_ptr
            + source[:, None] * COUNT_ROW_STRIDE
            + destination_base
            + safe_slot[None, :],
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
                + safe_slot,
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
            send_bucket_starts_ptr + destination_base + safe_slot,
            send_starts,
            mask=valid_slot,
        )

        total_by_slot = tl.sum(counts, axis=0)
        slot_start = tl.cumsum(total_by_slot, axis=0) - total_by_slot
        source_prefix = tl.sum(
            tl.where(source[:, None] < LOCAL_RANK, counts, 0), axis=0
        )
        tl.store(
            send_bucket_receive_offsets_ptr + destination_base + safe_slot,
            slot_start + source_prefix,
            mask=valid_slot,
        )

        is_local_destination = destination == LOCAL_RANK
        tl.store(
            receive_counts_ptr
            + source[:, None] * physical_slots
            + safe_slot[None, :],
            counts,
            mask=is_local_destination & valid_slot[None, :],
        )
        tl.store(
            received_per_slot_ptr + safe_slot,
            total_by_slot,
            mask=is_local_destination & valid_slot,
        )
        tl.store(
            received_slot_offsets_ptr + safe_slot,
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


@triton.jit
def _kernel_scatter_balanced_routes(
    scatter_positions_ptr,
    valid_route_ids_ptr,
    expert_order_ptr,
    buckets_in_expert_order_ptr,
    final_order_ptr,
    balanced_buckets_ptr,
    num_routes,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter the two aligned route arrays with a deliberately small grid."""
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(num_routes, BLOCK_SIZE)

    with al.scope(core_mode="vector", disable_auto_sync=True):
        for tile_id in range(pid, num_tiles, num_programs):
            route_offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = route_offset < num_routes
            destination_offset = tl.load(
                scatter_positions_ptr + route_offset,
                mask=mask,
                other=0,
            )
            expert_order_offset = tl.load(
                expert_order_ptr + route_offset,
                mask=mask,
                other=0,
            )
            route_id = tl.load(
                valid_route_ids_ptr + expert_order_offset,
                mask=mask,
                other=0,
            )
            bucket = tl.load(
                buckets_in_expert_order_ptr + route_offset,
                mask=mask,
                other=0,
            )
            # Direct-scatter validation guarantees that destination offsets are
            # unique, so the two stores need no atomics.
            tl.store(final_order_ptr + destination_offset, route_id, mask=mask)
            tl.store(
                balanced_buckets_ptr + destination_offset,
                bucket,
                mask=mask,
            )


__all__ = [
    "_kernel_build_balanced_count_cube",
    "_kernel_finalize_balanced_metadata",
    "_kernel_map_balanced_routes",
    "_kernel_scatter_balanced_routes",
]
