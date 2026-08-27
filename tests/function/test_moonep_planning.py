# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only tests for the MoonEP B.0-B.3 planning reference."""

import pytest
import torch

from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)


def _allocation_from_prefix(alloc_cumsum: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(
        (alloc_cumsum.shape[0], 1), dtype=alloc_cumsum.dtype
    )
    return torch.diff(torch.cat((zeros, alloc_cumsum), dim=1), dim=1).t()


def _assert_planning_invariants(
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    experts_to_copy: torch.Tensor,
) -> None:
    world_size, num_experts = tpe_all.shape
    experts_per_rank = num_experts // world_size
    allocation = _allocation_from_prefix(alloc_cumsum)

    assert alloc_cumsum.dtype == torch.int32
    assert alloc_cumsum.is_contiguous()
    assert alloc_cumsum.shape == (num_experts, world_size)
    assert experts_to_copy.dtype == torch.int32
    assert experts_to_copy.is_contiguous()
    assert experts_to_copy.shape == (world_size, experts_per_rank)
    assert bool((allocation >= 0).all())
    assert torch.equal(allocation.sum(dim=0), tpe_all.sum(dim=0).to(torch.int32))
    assert torch.equal(
        allocation.sum(dim=1), tpe_all.sum(dim=1).to(torch.int32)
    )

    for destination in range(world_size):
        required = {
            expert
            for expert in range(num_experts)
            if expert // experts_per_rank != destination
            and int(allocation[destination, expert]) > 0
        }
        copied = {
            int(expert)
            for expert in experts_to_copy[destination]
            if int(expert) >= 0
        }
        assert copied == required
        assert len(copied) == int((experts_to_copy[destination] >= 0).sum())


def test_balanced_input_keeps_all_experts_on_their_owner() -> None:
    tpe_all = torch.tensor(
        [[2, 2, 0, 0], [0, 0, 2, 2]], dtype=torch.int32
    )

    result = plan_moonep_b0_b3(tpe_all)

    assert torch.equal(
        result.alloc_cumsum,
        torch.tensor(
            [[2, 2], [2, 2], [0, 2], [0, 2]], dtype=torch.int32
        ),
    )
    assert torch.equal(
        result.experts_to_copy,
        torch.full((2, 2), -1, dtype=torch.int32),
    )
    assert result.replica_slot_high_water == 0
    _assert_planning_invariants(
        tpe_all, result.alloc_cumsum, result.experts_to_copy
    )


def test_hot_expert_is_split_and_added_to_remote_slot() -> None:
    tpe_all = torch.tensor(
        [[4, 0, 0, 0], [4, 0, 0, 0]], dtype=torch.int64
    )

    result = plan_moonep_b0_b3(tpe_all)

    assert torch.equal(
        result.alloc_cumsum,
        torch.tensor(
            [[4, 8], [0, 0], [0, 0], [0, 0]], dtype=torch.int32
        ),
    )
    assert torch.equal(
        result.experts_to_copy,
        torch.tensor([[-1, -1], [0, -1]], dtype=torch.int32),
    )
    assert result.replica_slot_high_water == 1
    inverse = build_inverse_experts_to_copy(result.experts_to_copy, 4)
    assert int(inverse[1, 0]) == 0
    assert int((inverse >= 0).sum()) == 1
    _assert_planning_invariants(
        tpe_all, result.alloc_cumsum, result.experts_to_copy
    )


def test_replica_selection_tie_uses_larger_expert_id() -> None:
    # Every source has eight routes.  Home group 0 owns sixteen routes while
    # home group 3 owns none, so rank 3 receives four routes from both e0/e1.
    row = [1, 1, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
    tpe_all = torch.tensor([row] * 4, dtype=torch.int32)

    result = plan_moonep_b0_b3(tpe_all)

    assert torch.equal(
        result.experts_to_copy[3],
        torch.tensor([1, 0, -1, -1], dtype=torch.int32),
    )
    assert result.replica_slot_high_water == 2
    inverse = build_inverse_experts_to_copy(result.experts_to_copy, 16)
    assert int(inverse[3, 1]) == 0
    assert int(inverse[3, 0]) == 1
    _assert_planning_invariants(
        tpe_all, result.alloc_cumsum, result.experts_to_copy
    )


def test_random_dropless_counts_preserve_routes_and_capacity() -> None:
    generator = torch.Generator().manual_seed(20260821)
    world_size, num_experts, capacity = 4, 16, 97
    for _ in range(20):
        routes = torch.randint(
            0,
            num_experts,
            (world_size, capacity),
            generator=generator,
        )
        tpe_all = torch.stack(
            [
                torch.bincount(row, minlength=num_experts).to(torch.int32)
                for row in routes
            ]
        ).contiguous()
        result = plan_moonep_b0_b3(tpe_all)
        _assert_planning_invariants(
            tpe_all, result.alloc_cumsum, result.experts_to_copy
        )


@pytest.mark.parametrize("tokens_per_rank", [4096, 8192, 16384])
def test_kimi_w8_moderate_wide_has_exact_load_and_copy_counts(
    tokens_per_rank,
) -> None:
    """Lock the nonzero W8 imbalance and its 52-expert prefetch layout."""
    world_size, num_experts, experts_per_rank, top_k = 8, 896, 112, 16
    owner_counts = (19, 27, 11, 11, 15, 15, 15, 15)
    owner_period = torch.tensor(
        [
            owner
            for owner, count in enumerate(owner_counts)
            for _ in range(count)
        ],
        dtype=torch.int64,
    )
    rows = []
    for source_rank in range(world_size):
        route_ids = torch.arange(
            tokens_per_rank * top_k,
            dtype=torch.int64,
        )
        owners = owner_period[route_ids.remainder(len(owner_period))]
        global_route_ids = route_ids + source_rank * tokens_per_rank * top_k
        selected = (
            owners * experts_per_rank
            + global_route_ids.remainder(experts_per_rank)
        )
        rows.append(
            torch.bincount(selected.reshape(-1), minlength=num_experts)
        )

    tpe_all = torch.stack(rows).to(torch.int64).contiguous()
    result = plan_moonep_b0_b3(tpe_all)
    owner_routes = (
        tpe_all.sum(dim=0)
        .view(world_size, experts_per_rank)
        .sum(dim=1)
    )
    route_unit = tokens_per_rank * top_k

    assert owner_routes.tolist() == [
        value * route_unit * world_size // len(owner_period)
        for value in owner_counts
    ]
    expected_copy_counts = [0, 0, 18, 18, 4, 4, 4, 4]
    assert (result.experts_to_copy >= 0).sum(dim=1).tolist() == expected_copy_counts
    assert int((result.experts_to_copy >= 0).sum()) == 52
    expected_copy_owners = [set(), set(), {1}, {1}, {0}, {0}, {0}, {1}]
    assert [
        {
            int(expert) // experts_per_rank
            for expert in row
            if int(expert) >= 0
        }
        for row in result.experts_to_copy
    ] == expected_copy_owners
    _assert_planning_invariants(
        tpe_all, result.alloc_cumsum, result.experts_to_copy
    )


def test_kimi_w8_rank3_only_functional_oracle_layout() -> None:
    """Keep the 8x single-owner route only as a planner correctness oracle."""
    world_size, num_experts, experts_per_rank = 8, 896, 112
    row = torch.zeros(num_experts, dtype=torch.int32)
    row[3 * experts_per_rank : 4 * experts_per_rank] = 1
    tpe_all = row.repeat(world_size, 1).contiguous()

    result = plan_moonep_b0_b3(tpe_all)

    owner_routes = tpe_all.sum(0).view(world_size, experts_per_rank).sum(1)
    assert owner_routes.tolist() == [0, 0, 0, 8 * experts_per_rank, 0, 0, 0, 0]
    assert (result.experts_to_copy >= 0).sum(dim=1).tolist() == [
        14, 14, 14, 0, 14, 14, 14, 14
    ]
    _assert_planning_invariants(
        tpe_all, result.alloc_cumsum, result.experts_to_copy
    )


@pytest.mark.parametrize(
    ("tpe_all", "error", "message"),
    [
        (
            torch.zeros((2, 4), dtype=torch.float32),
            TypeError,
            "dtype",
        ),
        (
            torch.zeros((4, 2), dtype=torch.int32).t(),
            ValueError,
            "contiguous",
        ),
        (
            torch.zeros((2, 3), dtype=torch.int32),
            ValueError,
            "divisible",
        ),
        (
            torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
            ValueError,
            "equal route counts",
        ),
        (
            torch.tensor([[0, -1], [0, -1]], dtype=torch.int32),
            ValueError,
            "non-negative",
        ),
        (
            torch.tensor(
                [[(1 << 31) - 1, 0], [(1 << 31) - 1, 0]],
                dtype=torch.int64,
            ),
            ValueError,
            "global route count",
        ),
    ],
)
def test_invalid_planner_inputs_are_rejected(tpe_all, error, message) -> None:
    with pytest.raises(error, match=message):
        plan_moonep_b0_b3(tpe_all)


def test_inverse_rejects_home_and_duplicate_experts() -> None:
    with pytest.raises(ValueError, match="remote"):
        build_inverse_experts_to_copy(
            torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32), 4
        )
    with pytest.raises(ValueError, match="duplicate"):
        build_inverse_experts_to_copy(
            torch.tensor([[2, 2], [-1, -1]], dtype=torch.int32), 4
        )
