# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pure-Torch reference for MoonEP-balanced Mega-MoE routing metadata.

This module is intentionally independent from the production Triton routing
path.  It defines the exact stable bucket and source-major offset contract
that a fused device implementation must preserve.  The reference stable-sort
route grouping remains the default; an opt-in direct scatter implements the
same order in O(number of valid routes). Production MoonEP routing always
requests direct scatter; the default here preserves an independent oracle.
"""

from dataclasses import dataclass

import torch
import triton


_INT32_MAX = torch.iinfo(torch.int32).max
_DIRECT_SCATTER_BLOCK_SIZE = 256
_DIRECT_SCATTER_MAX_PROGRAMS = 8
_FUSED_ROUTE_MAP_MAX_PROGRAMS = 64


@dataclass(frozen=True, slots=True)
class BalancedRoutingMetadata:
    """Balanced route order and the corresponding physical-slot metadata.

    ``final_order`` contains original flattened ``(token, top-k slot)`` route
    ids.  ``balanced_buckets`` is aligned with that final send order and uses
    ``destination * physical_slots_per_rank + physical_slot``.

    ``balanced_counts[source, bucket]`` describes the complete EP group, not
    just the local source rank.  Receive storage is physical-slot-major, with
    source ranks laid out in increasing rank order inside each slot.
    """

    rank: int
    world_size: int
    num_experts: int
    experts_per_rank: int
    replica_slots_per_rank: int
    physical_slots_per_rank: int
    num_input_tokens: int
    num_valid_routes: int
    valid_route_mask: torch.Tensor
    final_order: torch.Tensor
    send_token_indices: torch.Tensor
    send_route_indices: torch.Tensor
    balanced_buckets: torch.Tensor
    balanced_counts: torch.Tensor
    send_bucket_starts: torch.Tensor
    send_bucket_receive_offsets: torch.Tensor
    receive_counts_by_source_slot: torch.Tensor
    received_routes_per_slot: torch.Tensor
    received_slot_offsets: torch.Tensor


def _require_integral_tensor(name: str, tensor: torch.Tensor, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have dtype torch.int32 or torch.int64")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")


def _all(tensor: torch.Tensor) -> bool:
    return bool(torch.all(tensor).item())


def _validate_inputs(
    selected_experts: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    rank: int,
) -> tuple[int, int, int]:
    _require_integral_tensor("selected_experts", selected_experts, 2)
    _require_integral_tensor("tpe_all", tpe_all, 2)
    _require_integral_tensor("alloc_cumsum", alloc_cumsum, 2)
    _require_integral_tensor(
        "inverse_experts_to_copy", inverse_experts_to_copy, 2
    )

    device = selected_experts.device
    for name, tensor in (
        ("tpe_all", tpe_all),
        ("alloc_cumsum", alloc_cumsum),
        ("inverse_experts_to_copy", inverse_experts_to_copy),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")

    world_size, num_experts = (int(value) for value in tpe_all.shape)
    if world_size <= 0 or num_experts <= 0:
        raise ValueError("world_size and num_experts must be positive")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must identify a source rank in the EP group")
    if selected_experts.shape[1] <= 0:
        raise ValueError("selected_experts top-k dimension must be positive")
    if alloc_cumsum.shape != (num_experts, world_size):
        raise ValueError("alloc_cumsum must have shape [E, R]")
    if inverse_experts_to_copy.shape != (world_size, num_experts):
        raise ValueError("inverse_experts_to_copy must have shape [R, E]")

    tpe_i64 = tpe_all.to(torch.int64)
    alloc_i64 = alloc_cumsum.to(torch.int64)
    inverse_i64 = inverse_experts_to_copy.to(torch.int64)
    if not _all(tpe_i64 >= 0):
        raise ValueError("tpe_all counts must be non-negative")
    if not _all(alloc_i64 >= 0):
        raise ValueError("alloc_cumsum must be non-negative")
    if world_size > 1 and not _all(alloc_i64[:, 1:] >= alloc_i64[:, :-1]):
        raise ValueError("alloc_cumsum must be monotone along destination rank")
    if not torch.equal(alloc_i64[:, -1], tpe_i64.sum(dim=0)):
        raise ValueError("alloc_cumsum must conserve every logical expert")

    experts_per_rank = num_experts // world_size
    if not _all((inverse_i64 >= -1) & (inverse_i64 < experts_per_rank)):
        raise ValueError("inverse replica slots must be -1 or in [0, E/R)")

    allocation = torch.diff(
        torch.cat(
            (
                torch.zeros(
                    (num_experts, 1), dtype=torch.int64, device=device
                ),
                alloc_i64,
            ),
            dim=1,
        ),
        dim=1,
    )
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    owner_ranks = torch.div(
        expert_ids, experts_per_rank, rounding_mode="floor"
    )
    for destination in range(world_size):
        home_mask = owner_ranks == destination
        if not _all(inverse_i64[destination, home_mask] == -1):
            raise ValueError("inverse table must not assign slots to home experts")

        assigned = inverse_i64[destination]
        assigned = assigned[assigned >= 0]
        if int(torch.unique(assigned).numel()) != int(assigned.numel()):
            raise ValueError("inverse table contains duplicate replica slots")

        required = (~home_mask) & (allocation[:, destination] > 0)
        if not _all(inverse_i64[destination, required] >= 0):
            raise ValueError("a remote allocation is missing its replica slot")

    return world_size, num_experts, experts_per_rank


def _derive_experts_to_copy(
    inverse_experts_to_copy: torch.Tensor,
    *,
    world_size: int,
    num_experts: int,
    experts_per_rank: int,
) -> torch.Tensor:
    """Recover the dest-major replica expert list from the inverse table."""
    device = inverse_experts_to_copy.device
    slots = torch.arange(experts_per_rank, device=device)
    expert_ids = torch.arange(num_experts, device=device)
    match = inverse_experts_to_copy[:, None, :] == slots[None, :, None]
    return (
        torch.where(match, expert_ids[None, None, :], -1)
        .max(dim=2)
        .values.to(torch.int32)
    )


def _repair_count_cube_head(
    counts: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    *,
    world_size: int,
    experts_per_rank: int,
) -> None:
    """Recompute the home slot-0 column the vector-core kernel cannot land.

    The 910B1 vector store engine drops the first four bytes of every
    destination's slice window, so the cell at ``destination * 2 * EPN``
    reads back as zero no matter which lane carries it or in which order.
    Regular torch writes go through the ACL path and stick, and this column
    is two small gathers on planning tables that are already resident.
    """
    device = counts.device
    dest = torch.arange(world_size, device=device)
    head_expert = dest * experts_per_rank
    alloc_hi = alloc_cumsum[head_expert, dest]
    alloc_lo = torch.zeros_like(alloc_hi)
    alloc_lo[1:] = alloc_cumsum[head_expert[1:], dest[:-1]]
    source_count = tpe_all[:, head_expert]
    source_lo = source_count.cumsum(dim=0) - source_count
    overlap = (
        torch.minimum(source_lo + source_count, alloc_hi)
        - torch.maximum(source_lo, alloc_lo)
    ).clamp_(min=0)
    physical_slots = 2 * experts_per_rank
    cube = counts[:, : world_size * physical_slots].view(
        world_size, world_size, physical_slots
    )
    cube[:, :, 0] = overlap


def _build_balanced_count_cube(
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    *,
    world_size: int,
    num_experts: int,
    experts_per_rank: int,
    validate: bool = True,
    use_fused_npu: bool = True,
    experts_to_copy: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derive ``[source, destination, physical slot]`` counts by intervals."""
    device = tpe_all.device
    physical_slots = 2 * experts_per_rank
    if device.type == "npu" and use_fused_npu:
        from mega_moe.kernels.balanced_routing import (
            _kernel_build_balanced_count_cube,
        )

        if experts_to_copy is None:
            experts_to_copy = _derive_experts_to_copy(
                inverse_experts_to_copy,
                world_size=world_size,
                num_experts=num_experts,
                experts_per_rank=experts_per_rank,
            )
        counts = torch.empty(
            (world_size, world_size * physical_slots),
            dtype=torch.int32,
            device=device,
        )
        block_e = min(32, triton.next_power_of_2(num_experts))
        while num_experts % block_e:
            block_e //= 2
        _kernel_build_balanced_count_cube[(world_size, 1, 1)](
            tpe_all,
            alloc_cumsum,
            experts_to_copy,
            counts,
            counts,
            R=world_size,
            E=num_experts,
            EPN=experts_per_rank,
            LOCAL_RANK=0,
            TPE_ROW_STRIDE=tpe_all.stride(0),
            COUNT_ROW_STRIDE=world_size * physical_slots,
            BLOCK_E=block_e,
            BLOCK_SLOTS=triton.next_power_of_2(physical_slots),
            STORE_LOCAL_STARTS=False,
        )
        _repair_count_cube_head(
            counts,
            tpe_all,
            alloc_cumsum,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
        )
        if validate and not torch.equal(
            counts.sum(dim=1), tpe_all.to(torch.int64).sum(dim=1)
        ):
            raise RuntimeError("balanced count construction lost source routes")
        return counts

    tpe_i64 = tpe_all.to(torch.int64)
    alloc_hi = alloc_cumsum.to(torch.int64)
    alloc_lo = torch.cat(
        (
            torch.zeros((num_experts, 1), dtype=torch.int64, device=device),
            alloc_hi[:, :-1],
        ),
        dim=1,
    )
    source_hi = tpe_i64.cumsum(dim=0)
    source_lo = source_hi - tpe_i64
    inverse_i64 = inverse_experts_to_copy.to(torch.int64)

    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    owners = torch.div(expert_ids, experts_per_rank, rounding_mode="floor")
    home_slots = expert_ids.remainder(experts_per_rank)
    counts = torch.zeros(
        (world_size, world_size * physical_slots),
        dtype=torch.int64,
        device=device,
    )
    for destination in range(world_size):
        overlap = torch.clamp(
            torch.minimum(
                source_hi, alloc_hi[:, destination].unsqueeze(0)
            )
            - torch.maximum(
                source_lo, alloc_lo[:, destination].unsqueeze(0)
            ),
            min=0,
        )
        physical_slot = torch.where(
            owners == destination,
            home_slots,
            experts_per_rank + inverse_i64[destination],
        )
        bucket = destination * physical_slots + physical_slot
        counts.scatter_add_(
            1, bucket.unsqueeze(0).expand(world_size, -1), overlap
        )

    if validate and not torch.equal(counts.sum(dim=1), tpe_i64.sum(dim=1)):
        raise RuntimeError("balanced count construction lost source routes")
    return counts


def _to_metadata_int32(
    name: str, tensor: torch.Tensor, *, validate: bool = True
) -> torch.Tensor:
    if validate and tensor.numel() and int(tensor.max().item()) > _INT32_MAX:
        raise ValueError(f"{name} exceeds the int32 metadata range")
    return tensor.to(torch.int32).contiguous()


def _scatter_balanced_routes(
    scatter_positions: torch.Tensor,
    valid_route_ids: torch.Tensor,
    expert_order: torch.Tensor,
    buckets_in_expert_order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write both direct-scatter outputs without reserving the full AIV grid."""
    num_routes = int(scatter_positions.numel())
    final_order = torch.empty_like(valid_route_ids)
    balanced_buckets = torch.empty_like(buckets_in_expert_order)
    if num_routes == 0:
        return final_order, balanced_buckets

    if scatter_positions.device.type == "npu":
        from mega_moe.kernels.balanced_routing import (
            _kernel_scatter_balanced_routes,
        )

        num_programs = min(
            _DIRECT_SCATTER_MAX_PROGRAMS,
            (num_routes + _DIRECT_SCATTER_BLOCK_SIZE - 1)
            // _DIRECT_SCATTER_BLOCK_SIZE,
        )
        _kernel_scatter_balanced_routes[(num_programs, 1, 1)](
            scatter_positions,
            valid_route_ids,
            expert_order,
            buckets_in_expert_order,
            final_order,
            balanced_buckets,
            num_routes,
            BLOCK_SIZE=_DIRECT_SCATTER_BLOCK_SIZE,
        )
    else:
        # Preserve the pure-Torch reference path for CPU tests and callers.
        final_order.scatter_(
            0, scatter_positions, valid_route_ids[expert_order]
        )
        balanced_buckets.scatter_(
            0, scatter_positions, buckets_in_expert_order
        )
    return final_order, balanced_buckets


def build_balanced_routing_metadata_inplace(
    sorted_experts: torch.Tensor,
    expert_order: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    experts_to_copy: torch.Tensor,
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
    """Queue the production MoonEP count, metadata, and route-map kernels."""
    if sorted_experts.device.type != "npu":
        raise ValueError("in-place balanced routing is only available on NPU")
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
        "experts_to_copy": ((world_size, experts_per_rank), torch.int32),
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
        "experts_to_copy": experts_to_copy,
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

    from mega_moe.kernels.balanced_routing import (
        _kernel_build_balanced_count_cube,
        _kernel_finalize_balanced_metadata,
        _kernel_map_balanced_routes,
    )

    block_e = min(32, triton.next_power_of_2(num_experts))
    while num_experts % block_e:
        block_e //= 2
    block_slots = triton.next_power_of_2(physical_slots)
    _kernel_build_balanced_count_cube[(world_size, 1, 1)](
        tpe_all,
        alloc_cumsum,
        experts_to_copy,
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
    # The vector core cannot land the first cell of each destination slice
    # (see _repair_count_cube_head); patch that column before finalize reads
    # the cube on device.
    _repair_count_cube_head(
        count_rows,
        tpe_all,
        alloc_cumsum,
        world_size=world_size,
        experts_per_rank=experts_per_rank,
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
            _FUSED_ROUTE_MAP_MAX_PROGRAMS,
            triton.cdiv(num_routes, _DIRECT_SCATTER_BLOCK_SIZE),
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
            BLOCK_ROUTES=_DIRECT_SCATTER_BLOCK_SIZE,
        )


def build_balanced_routing_metadata(
    selected_experts: torch.Tensor,
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    rank: int,
    *,
    validate: bool = True,
    expert_order: torch.Tensor | None = None,
    sorted_experts: torch.Tensor | None = None,
    use_direct_bucket_scatter: bool = False,
    use_fused_count_cube: bool = True,
) -> BalancedRoutingMetadata:
    """Build the correctness-reference balanced dispatch metadata.

    Invalid expert ids are omitted from communication but remain represented by
    ``valid_route_mask`` and by gaps in the original route ids.  The local
    valid-expert histogram must equal ``tpe_all[rank]``.

    The replica budget is fixed to ``B = E/R`` and physical slots are laid out
    as ``[home slots, replica slots]``.  A trusted dropless production caller
    may pass Mega's existing stable ``expert_order`` and ``sorted_experts`` to
    avoid sorting the logical experts a second time.  By default the physical
    bucket order is produced by a stable sort.  ``use_direct_bucket_scatter``
    enables an equivalent O(M) placement using the source/allocation interval
    intersection.
    """
    if type(validate) is not bool:
        raise TypeError("validate must be a bool")
    if type(use_direct_bucket_scatter) is not bool:
        raise TypeError("use_direct_bucket_scatter must be a bool")
    if type(use_fused_count_cube) is not bool:
        raise TypeError("use_fused_count_cube must be a bool")
    if validate:
        world_size, num_experts, experts_per_rank = _validate_inputs(
            selected_experts,
            tpe_all,
            alloc_cumsum,
            inverse_experts_to_copy,
            rank,
        )
    else:
        world_size, num_experts = (int(value) for value in tpe_all.shape)
        experts_per_rank = num_experts // world_size
    device = selected_experts.device
    top_k = int(selected_experts.shape[1])
    num_flat_routes = int(selected_experts.numel())
    if num_flat_routes > _INT32_MAX:
        raise ValueError("flattened route ids exceed the int32 metadata range")

    if (expert_order is None) != (sorted_experts is None):
        raise ValueError(
            "expert_order and sorted_experts must either both be provided or "
            "both be omitted"
        )
    reuse_expert_order = expert_order is not None

    flat_experts = selected_experts.reshape(-1).to(torch.int64)
    valid_route_mask = (flat_experts >= 0) & (flat_experts < num_experts)
    if reuse_expert_order:
        for name, tensor in (
            ("expert_order", expert_order),
            ("sorted_experts", sorted_experts),
        ):
            _require_integral_tensor(name, tensor, 1)
            if tensor.device != device:
                raise ValueError(f"{name} must be on {device}")
            if tensor.numel() != num_flat_routes:
                raise ValueError(
                    f"{name} must contain every route in the dropless input"
                )
        if validate and not _all(valid_route_mask):
            raise ValueError(
                "precomputed expert order is only valid for dropless routes"
            )
        valid_route_ids = torch.arange(
            num_flat_routes, dtype=torch.int64, device=device
        )
        kept_experts = flat_experts
        expert_order = expert_order.to(torch.int64)
        sorted_experts = sorted_experts.to(torch.int64)
        if validate:
            expected_order = torch.argsort(
                kept_experts.to(torch.float32), stable=True
            )
            if not torch.equal(expert_order, expected_order):
                raise ValueError("expert_order is not Mega's stable expert order")
            if not torch.equal(sorted_experts, kept_experts[expert_order]):
                raise ValueError(
                    "sorted_experts does not match selected_experts[expert_order]"
                )
        local_counts = tpe_all[rank].to(torch.int64)
    else:
        valid_route_ids = torch.arange(
            num_flat_routes, dtype=torch.int64, device=device
        )[valid_route_mask]
        kept_experts = flat_experts[valid_route_mask]
        local_counts = torch.bincount(kept_experts, minlength=num_experts)
        # Expert ids are far below float32's exact-integer bound.  Sorting this
        # view avoids Ascend's integer ArgSort AICPU fallback while preserving
        # the stable expert order required by global ordinals.
        expert_order = torch.argsort(
            kept_experts.to(torch.float32), stable=True
        )
        sorted_experts = kept_experts[expert_order]
    if validate and not torch.equal(local_counts, tpe_all[rank].to(torch.int64)):
        raise ValueError(
            "the valid selected_experts histogram must equal tpe_all[rank]"
        )
    num_valid = int(sorted_experts.numel())
    local_exclusive = local_counts.cumsum(dim=0) - local_counts
    local_ordinal = torch.arange(
        num_valid, dtype=torch.int64, device=device
    ) - local_exclusive[sorted_experts]
    previous_rank_count = (
        tpe_all[:rank].to(torch.int64).sum(dim=0)[sorted_experts]
        if rank
        else torch.zeros(num_valid, dtype=torch.int64, device=device)
    )
    global_ordinal = previous_rank_count + local_ordinal

    alloc_i64 = alloc_cumsum.to(torch.int64)
    allocation_prefix = alloc_i64[sorted_experts]
    destination = (allocation_prefix <= global_ordinal.unsqueeze(1)).sum(dim=1)
    if validate and destination.numel() and not _all(destination < world_size):
        raise RuntimeError("a route has no destination allocation")

    owners = torch.div(
        sorted_experts, experts_per_rank, rounding_mode="floor"
    )
    home_slots = sorted_experts.remainder(experts_per_rank)
    replica_slots = inverse_experts_to_copy.to(torch.int64)[
        destination, sorted_experts
    ]
    remote_routes = destination != owners
    if (
        validate
        and replica_slots.numel()
        and not _all(replica_slots[remote_routes] >= 0)
    ):
        raise RuntimeError("a remote route has no replica slot")

    physical_slots = 2 * experts_per_rank
    execution_slots = torch.where(
        remote_routes, experts_per_rank + replica_slots, home_slots
    )
    buckets_in_expert_order = destination * physical_slots + execution_slots

    if use_direct_bucket_scatter:
        balanced_counts_i64 = _build_balanced_count_cube(
            tpe_all,
            alloc_cumsum,
            inverse_experts_to_copy,
            world_size=world_size,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            validate=validate,
            use_fused_npu=use_fused_count_cube,
        )
        local_bucket_counts = balanced_counts_i64[rank]
        if validate and not torch.equal(
            torch.bincount(
                buckets_in_expert_order,
                minlength=world_size * physical_slots,
            ),
            local_bucket_counts,
        ):
            raise RuntimeError("local route buckets disagree with balanced counts")
        send_bucket_starts_i64 = (
            local_bucket_counts.cumsum(dim=0) - local_bucket_counts
        )

        # The interval for source rank ``rank`` and expert ``e`` is
        # ``[previous_rank_count, previous_rank_count + tpe_all[rank, e])``.
        # MoonEP's destination interval starts at ``alloc_lo[e, destination]``.
        # Their intersection is already in the stable order of the first expert
        # sort, so its offset inside the destination bucket is an O(1) expression.
        alloc_lo_all = torch.cat(
            (
                torch.zeros((num_experts, 1), dtype=torch.int64, device=device),
                alloc_i64[:, :-1],
            ),
            dim=1,
        )
        allocation_lo = alloc_lo_all[sorted_experts, destination]
        intersection_lo = torch.maximum(previous_rank_count, allocation_lo)
        within_bucket = global_ordinal - intersection_lo
        scatter_positions = (
            send_bucket_starts_i64[buckets_in_expert_order] + within_bucket
        )
        if validate and num_valid:
            positions_are_in_range = _all(
                (scatter_positions >= 0) & (scatter_positions < num_valid)
            )
            positions_are_unique = positions_are_in_range and _all(
                torch.bincount(scatter_positions, minlength=num_valid) == 1
            )
            if not positions_are_unique:
                raise RuntimeError(
                    "direct balanced bucket scatter produced duplicate or "
                    "out-of-range positions"
                )
        final_order, balanced_buckets = _scatter_balanced_routes(
            scatter_positions,
            valid_route_ids,
            expert_order,
            buckets_in_expert_order,
        )
    else:
        bucket_order = torch.argsort(
            buckets_in_expert_order.to(torch.float32), stable=True
        )
        final_order = valid_route_ids[expert_order[bucket_order]].contiguous()
        balanced_buckets = buckets_in_expert_order[bucket_order].contiguous()
        balanced_counts_i64 = _build_balanced_count_cube(
            tpe_all,
            alloc_cumsum,
            inverse_experts_to_copy,
            world_size=world_size,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            validate=validate,
            use_fused_npu=use_fused_count_cube,
        )
        local_bucket_counts = torch.bincount(
            balanced_buckets, minlength=world_size * physical_slots
        )
        if validate and not torch.equal(
            local_bucket_counts, balanced_counts_i64[rank]
        ):
            raise RuntimeError("local route buckets disagree with balanced counts")
        send_bucket_starts_i64 = (
            local_bucket_counts.cumsum(dim=0) - local_bucket_counts
        )

    counts_by_source_destination_slot = balanced_counts_i64.view(
        world_size, world_size, physical_slots
    )
    totals_by_destination_slot = counts_by_source_destination_slot.sum(dim=0)
    destination_slot_starts = (
        totals_by_destination_slot.cumsum(dim=1)
        - totals_by_destination_slot
    )
    source_prefix = balanced_counts_i64[:rank].sum(dim=0)
    send_bucket_receive_offsets_i64 = (
        destination_slot_starts.reshape(-1) + source_prefix
    )

    receive_counts_i64 = counts_by_source_destination_slot[:, rank, :]
    received_per_slot_i64 = receive_counts_i64.sum(dim=0)
    received_slot_offsets_i64 = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            received_per_slot_i64.cumsum(dim=0),
        )
    )

    return BalancedRoutingMetadata(
        rank=rank,
        world_size=world_size,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        replica_slots_per_rank=experts_per_rank,
        physical_slots_per_rank=physical_slots,
        num_input_tokens=int(selected_experts.shape[0]),
        num_valid_routes=num_valid,
        valid_route_mask=valid_route_mask.contiguous(),
        final_order=final_order,
        send_token_indices=(final_order // top_k).to(torch.int32).contiguous(),
        send_route_indices=final_order.to(torch.int32).contiguous(),
        balanced_buckets=balanced_buckets.to(torch.int32).contiguous(),
        balanced_counts=_to_metadata_int32(
            "balanced_counts", balanced_counts_i64, validate=validate
        ),
        send_bucket_starts=_to_metadata_int32(
            "send_bucket_starts", send_bucket_starts_i64, validate=validate
        ),
        send_bucket_receive_offsets=_to_metadata_int32(
            "send_bucket_receive_offsets",
            send_bucket_receive_offsets_i64,
            validate=validate,
        ),
        receive_counts_by_source_slot=_to_metadata_int32(
            "receive_counts_by_source_slot", receive_counts_i64, validate=validate
        ),
        received_routes_per_slot=_to_metadata_int32(
            "received_routes_per_slot", received_per_slot_i64, validate=validate
        ),
        received_slot_offsets=_to_metadata_int32(
            "received_slot_offsets", received_slot_offsets_i64, validate=validate
        ),
    )


__all__ = [
    "BalancedRoutingMetadata",
    "build_balanced_routing_metadata",
    "build_balanced_routing_metadata_inplace",
]
