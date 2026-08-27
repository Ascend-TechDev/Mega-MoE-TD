# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host reference for the MoonEP B.0-B.3 load-balancing tables.

This module deliberately stops before MoonEP's route layout phases.  Mega-MoE
only needs the inclusive allocation prefix and the remote-expert slot table to
map its existing stable route order onto physical execution buckets.

The first integration targets dropless routing: every source rank contributes
the same number of routes.  Rejecting unequal row totals is intentional because
MoonEP B.1 uses that common row total as every destination rank's capacity.
"""

from dataclasses import dataclass

import torch


_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True, slots=True)
class MoonEPPlanningResult:
    """The two MoonEP planning tables consumed by Mega-MoE routing.

    ``alloc_cumsum[e, d]`` is the inclusive number of routes for logical
    expert ``e`` assigned to destination ranks ``0..d``.

    ``experts_to_copy[d, b]`` is the logical expert placed in replica slot
    ``b`` on destination rank ``d``.  Unused slots contain ``-1``.  The replica
    budget is fixed to ``experts_per_rank``.
    """

    alloc_cumsum: torch.Tensor
    experts_to_copy: torch.Tensor

    def __post_init__(self) -> None:
        alloc_cumsum = self.alloc_cumsum
        experts_to_copy = self.experts_to_copy
        for name, tensor in (
            ("alloc_cumsum", alloc_cumsum),
            ("experts_to_copy", experts_to_copy),
        ):
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a CPU tensor")
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must have dtype torch.int32")
            if tensor.ndim != 2 or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous rank-2 tensor")

        num_experts, world_size = alloc_cumsum.shape
        etc_world_size, replica_budget = experts_to_copy.shape
        if world_size <= 0 or num_experts <= 0:
            raise ValueError("planning table dimensions must be positive")
        if etc_world_size != world_size:
            raise ValueError("planning tables disagree on world_size")
        if num_experts % world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if replica_budget != num_experts // world_size:
            raise ValueError("replica budget must equal experts_per_rank")

    @property
    def replica_slot_high_water(self) -> int:
        """Return one past the highest occupied replica slot in any rank."""
        occupied = torch.nonzero(self.experts_to_copy >= 0, as_tuple=False)
        return (
            int(occupied[:, 1].max().item()) + 1
            if occupied.numel()
            else 0
        )


def _validate_tpe_all(
    tpe_all: torch.Tensor,
) -> tuple[int, int, int, int, list[list[int]]]:
    if not isinstance(tpe_all, torch.Tensor):
        raise TypeError("tpe_all must be a torch.Tensor")
    if tpe_all.device.type != "cpu":
        raise ValueError("tpe_all must be a CPU tensor")
    if tpe_all.dtype not in (torch.int32, torch.int64):
        raise TypeError("tpe_all must have dtype torch.int32 or torch.int64")
    if tpe_all.ndim != 2:
        raise ValueError("tpe_all must have shape [world_size, num_experts]")
    if not tpe_all.is_contiguous():
        raise ValueError("tpe_all must be contiguous")

    world_size, num_experts = (int(v) for v in tpe_all.shape)
    if world_size <= 0 or num_experts <= 0:
        raise ValueError("world_size and num_experts must be positive")
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")

    counts = [[int(value) for value in row] for row in tpe_all.tolist()]
    if any(value < 0 for row in counts for value in row):
        raise ValueError("tpe_all counts must be non-negative")
    if any(value > _INT32_MAX for row in counts for value in row):
        raise ValueError("each tpe_all count must fit in int32")

    row_totals = [sum(row) for row in counts]
    capacity = row_totals[0]
    if any(total != capacity for total in row_totals[1:]):
        raise ValueError(
            "MoonEP B.0-B.3 currently requires equal route counts per source rank"
        )
    if capacity > _INT32_MAX:
        raise ValueError("routes per source rank must fit in int32")

    experts_per_rank = num_experts // world_size
    return world_size, num_experts, experts_per_rank, capacity, counts


def _build_allocation(
    counts: list[list[int]],
    *,
    world_size: int,
    num_experts: int,
    experts_per_rank: int,
    capacity: int,
) -> tuple[list[list[int]], list[int]]:
    # B.0: global logical-expert counts and home-rank loads.
    expert_count = [
        sum(counts[source][expert] for source in range(world_size))
        for expert in range(num_experts)
    ]
    if any(value > _INT32_MAX for value in expert_count):
        raise ValueError("global route count for each expert must fit in int32")
    group_tokens = [
        sum(
            expert_count[
                home * experts_per_rank : (home + 1) * experts_per_rank
            ]
        )
        for home in range(world_size)
    ]

    # B.1: single-source fill.  Ties use the lowest rank index.
    balance = [value - capacity for value in group_tokens]
    transfers = [[0] * world_size for _ in range(world_size)]
    while True:
        surplus = max(balance)
        deficit = min(balance)
        if surplus <= 0 or deficit >= 0:
            break
        source = balance.index(surplus)
        destination = balance.index(deficit)
        moved = -deficit
        transfers[source][destination] = moved
        balance[source] -= moved
        balance[destination] = 0
    if any(balance):
        raise RuntimeError("MoonEP B.1 failed to balance destination capacities")

    # B.2: greedily cut each home group across destination ranks.  Destination
    # and local-expert argmax ties both use the lowest index.
    allocation = [[0] * num_experts for _ in range(world_size)]
    for home in range(world_size):
        first_expert = home * experts_per_rank
        remaining = expert_count[
            first_expert : first_expert + experts_per_rank
        ].copy()
        quotas = transfers[home].copy()
        per_destination = [
            [0] * experts_per_rank for _ in range(world_size)
        ]
        per_destination[home] = remaining.copy()

        while max(quotas) > 0:
            destination = quotas.index(max(quotas))
            largest_remaining = max(remaining)
            if largest_remaining <= 0:
                raise RuntimeError("MoonEP B.2 could not satisfy a transfer quota")
            local_expert = remaining.index(largest_remaining)
            moved = min(quotas[destination], largest_remaining)
            quotas[destination] -= moved
            remaining[local_expert] -= moved
            per_destination[destination][local_expert] += moved
            per_destination[home][local_expert] = remaining[local_expert]

        for destination in range(world_size):
            allocation[destination][
                first_expert : first_expert + experts_per_rank
            ] = per_destination[destination]

    for expert in range(num_experts):
        if sum(allocation[d][expert] for d in range(world_size)) != expert_count[expert]:
            raise RuntimeError("MoonEP B.2 violated per-expert route conservation")
    if any(sum(row) != capacity for row in allocation):
        raise RuntimeError("MoonEP B.2 violated destination rank capacity")
    return allocation, expert_count


def _build_experts_to_copy(
    allocation: list[list[int]],
    *,
    world_size: int,
    num_experts: int,
    experts_per_rank: int,
) -> list[list[int]]:
    # B.3: select remote non-empty experts by allocation count.  Unlike B.1
    # and B.2, equal counts select the largest logical expert id.
    experts_to_copy = [
        [-1] * experts_per_rank for _ in range(world_size)
    ]
    for destination in range(world_size):
        first_local = destination * experts_per_rank
        last_local = first_local + experts_per_rank
        remote_counts = [
            0 if first_local <= expert < last_local else allocation[destination][expert]
            for expert in range(num_experts)
        ]
        remote_nonzero = sum(value > 0 for value in remote_counts)
        if remote_nonzero > experts_per_rank:
            raise RuntimeError(
                "fixed replica budget cannot cover all remote allocated experts"
            )

        for slot in range(experts_per_rank):
            largest = max(remote_counts)
            if largest <= 0:
                break
            expert = max(
                index for index, value in enumerate(remote_counts) if value == largest
            )
            experts_to_copy[destination][slot] = expert
            remote_counts[expert] = 0

        copied = {expert for expert in experts_to_copy[destination] if expert >= 0}
        required = {
            expert
            for expert in range(num_experts)
            if not first_local <= expert < last_local
            and allocation[destination][expert] > 0
        }
        if copied != required:
            raise RuntimeError("a remote allocation is missing its replica slot")
    return experts_to_copy


def plan_moonep_b0_b3(tpe_all: torch.Tensor) -> MoonEPPlanningResult:
    """Build MoonEP B.0-B.3 tables for dropless Mega-MoE routing.

    Args:
        tpe_all: Contiguous CPU int32/int64 counts shaped ``[R, E]``.  Every
            row must have the same sum, ``E`` must be divisible by ``R``, and
            every count/prefix must fit in int32.

    Returns:
        A result containing contiguous CPU int32 ``alloc_cumsum [E, R]`` and
        ``experts_to_copy [R, E/R]``.
    """

    world_size, num_experts, experts_per_rank, capacity, counts = (
        _validate_tpe_all(tpe_all)
    )
    allocation, expert_count = _build_allocation(
        counts,
        world_size=world_size,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        capacity=capacity,
    )
    experts_to_copy_list = _build_experts_to_copy(
        allocation,
        world_size=world_size,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
    )

    alloc_cumsum_list = [[0] * world_size for _ in range(num_experts)]
    for expert in range(num_experts):
        prefix = 0
        for destination in range(world_size):
            prefix += allocation[destination][expert]
            alloc_cumsum_list[expert][destination] = prefix
        if prefix != expert_count[expert]:
            raise RuntimeError("allocation prefix does not end at the expert count")

    return MoonEPPlanningResult(
        alloc_cumsum=torch.tensor(alloc_cumsum_list, dtype=torch.int32),
        experts_to_copy=torch.tensor(experts_to_copy_list, dtype=torch.int32),
    )


def build_inverse_experts_to_copy(
    experts_to_copy: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Invert ``[R, E/R]`` replica slots into an ``[R, E]`` lookup table."""

    if not isinstance(experts_to_copy, torch.Tensor):
        raise TypeError("experts_to_copy must be a torch.Tensor")
    if experts_to_copy.device.type != "cpu":
        raise ValueError("experts_to_copy must be a CPU tensor")
    if experts_to_copy.dtype != torch.int32:
        raise TypeError("experts_to_copy must have dtype torch.int32")
    if experts_to_copy.ndim != 2 or not experts_to_copy.is_contiguous():
        raise ValueError("experts_to_copy must be a contiguous rank-2 tensor")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")

    world_size, replica_budget = (int(v) for v in experts_to_copy.shape)
    if world_size <= 0 or num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by world_size")
    experts_per_rank = num_experts // world_size
    if replica_budget != experts_per_rank:
        raise ValueError("replica budget must equal experts_per_rank")

    inverse = torch.full(
        (world_size, num_experts), -1, dtype=torch.int32
    )
    for destination, row in enumerate(experts_to_copy.tolist()):
        seen: set[int] = set()
        for slot, expert in enumerate(row):
            if expert == -1:
                continue
            if expert < 0 or expert >= num_experts:
                raise ValueError("experts_to_copy contains an invalid expert id")
            if expert // experts_per_rank == destination:
                raise ValueError("experts_to_copy must contain only remote experts")
            if expert in seen:
                raise ValueError("experts_to_copy contains duplicate experts in a row")
            seen.add(expert)
            inverse[destination, expert] = slot
    return inverse


__all__ = [
    "MoonEPPlanningResult",
    "build_inverse_experts_to_copy",
    "plan_moonep_b0_b3",
]
