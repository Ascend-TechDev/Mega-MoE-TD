# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU tests for the pure-Torch balanced routing metadata reference."""

import pytest
import torch

from mega_moe.runtime.balanced_routing import (
    build_balanced_routing_metadata,
)
from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)


_METADATA_TENSOR_FIELDS = (
    "final_order",
    "send_token_indices",
    "send_route_indices",
    "balanced_buckets",
    "balanced_counts",
    "send_bucket_starts",
    "send_bucket_receive_offsets",
    "receive_counts_by_source_slot",
    "received_routes_per_slot",
    "received_slot_offsets",
)


def _assert_metadata_equal(actual, expected) -> None:
    for field in _METADATA_TENSOR_FIELDS:
        torch.testing.assert_close(
            getattr(actual, field), getattr(expected, field)
        )


def _plans_for_all_sources(selected_by_source: list[torch.Tensor]):
    world_size = len(selected_by_source)
    num_experts = 4
    tpe_all = torch.stack(
        [
            torch.bincount(
                selected[(selected >= 0) & (selected < num_experts)].to(
                    torch.int64
                ),
                minlength=num_experts,
            ).to(torch.int32)
            for selected in selected_by_source
        ]
    ).contiguous()
    planning = plan_moonep_b0_b3(tpe_all)
    inverse = build_inverse_experts_to_copy(
        planning.experts_to_copy, num_experts
    )
    plans = [
        build_balanced_routing_metadata(
            selected_by_source[source],
            tpe_all,
            planning.alloc_cumsum,
            inverse,
            source,
        )
        for source in range(world_size)
    ]
    return tpe_all, planning, inverse, plans


def test_hot_expert_routes_match_allocation_and_source_major_offsets() -> None:
    selected_by_source = [
        torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
        torch.tensor([[0, 0], [0, 0]], dtype=torch.int32),
    ]
    _, planning, inverse, plans = _plans_for_all_sources(selected_by_source)

    assert int(inverse[1, 0]) == 0
    expected_counts = torch.tensor(
        [
            [4, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 4, 0],
        ],
        dtype=torch.int32,
    )
    for plan in plans:
        torch.testing.assert_close(plan.balanced_counts, expected_counts)
        torch.testing.assert_close(
            plan.received_routes_per_slot,
            torch.tensor(
                [4, 0, 0, 0] if plan.rank == 0 else [0, 0, 4, 0],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            plan.received_slot_offsets,
            torch.tensor(
                [0, 4, 4, 4, 4] if plan.rank == 0 else [0, 0, 0, 4, 4],
                dtype=torch.int32,
            ),
        )

    torch.testing.assert_close(
        plans[0].balanced_buckets,
        torch.zeros(4, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plans[1].balanced_buckets,
        torch.full((4,), 6, dtype=torch.int32),
    )
    torch.testing.assert_close(
        plans[0].send_bucket_receive_offsets,
        torch.tensor([0, 4, 4, 4, 0, 0, 0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        plans[1].send_bucket_receive_offsets,
        torch.tensor([4, 4, 4, 4, 0, 0, 0, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        planning.alloc_cumsum[:, -1],
        torch.tensor([8, 0, 0, 0], dtype=torch.int32),
    )


def test_invalid_routes_are_omitted_without_renumbering_original_routes() -> None:
    selected_by_source = [
        torch.tensor([[0, -1], [4, 0]], dtype=torch.int32),
        torch.tensor([[0, 0]], dtype=torch.int32),
    ]
    tpe_all, planning, inverse, plans = _plans_for_all_sources(
        selected_by_source
    )
    rank_zero = plans[0]
    direct = build_balanced_routing_metadata(
        selected_by_source[0],
        tpe_all,
        planning.alloc_cumsum,
        inverse,
        rank=0,
        use_direct_bucket_scatter=True,
    )

    torch.testing.assert_close(
        rank_zero.valid_route_mask,
        torch.tensor([True, False, False, True]),
    )
    torch.testing.assert_close(
        rank_zero.final_order, torch.tensor([0, 3], dtype=torch.int64)
    )
    torch.testing.assert_close(
        rank_zero.send_route_indices, torch.tensor([0, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        rank_zero.send_token_indices, torch.tensor([0, 1], dtype=torch.int32)
    )
    assert rank_zero.num_valid_routes == 2
    _assert_metadata_equal(direct, rank_zero)


def test_direct_bucket_scatter_handles_empty_routes() -> None:
    selected_by_source = [
        torch.empty((0, 2), dtype=torch.int32),
        torch.empty((0, 2), dtype=torch.int32),
    ]
    tpe_all, planning, inverse, stable_plans = _plans_for_all_sources(
        selected_by_source
    )

    for source, stable in enumerate(stable_plans):
        direct = build_balanced_routing_metadata(
            selected_by_source[source],
            tpe_all,
            planning.alloc_cumsum,
            inverse,
            source,
            use_direct_bucket_scatter=True,
        )
        _assert_metadata_equal(direct, stable)
        assert direct.num_valid_routes == 0


def test_final_order_and_count_cube_agree_for_mixed_experts() -> None:
    selected_by_source = [
        torch.tensor([[3, 0], [1, 0], [3, 0], [1, 0]], dtype=torch.int32),
        torch.tensor([[3, 0], [3, 0], [3, 0], [1, 0]], dtype=torch.int32),
    ]
    _, _, _, plans = _plans_for_all_sources(selected_by_source)
    expected_cube = plans[0].balanced_counts

    for plan in plans:
        torch.testing.assert_close(plan.balanced_counts, expected_cube)
        assert bool((plan.balanced_buckets[1:] >= plan.balanced_buckets[:-1]).all())
        torch.testing.assert_close(
            torch.bincount(
                plan.balanced_buckets.to(torch.int64),
                minlength=plan.world_size * plan.physical_slots_per_rank,
            ).to(torch.int32),
            expected_cube[plan.rank],
        )
        torch.testing.assert_close(
            torch.sort(plan.final_order).values,
            torch.arange(plan.num_valid_routes, dtype=torch.int64),
        )

        counts = plan.balanced_counts.view(
            plan.world_size, plan.world_size, plan.physical_slots_per_rank
        )
        destination_totals = counts.sum(dim=0)
        destination_starts = (
            destination_totals.cumsum(dim=1) - destination_totals
        )
        source_prefix = counts[: plan.rank].sum(dim=0)
        torch.testing.assert_close(
            plan.send_bucket_receive_offsets,
            (destination_starts + source_prefix).reshape(-1).to(torch.int32),
        )


def test_trusted_production_path_matches_validated_reference() -> None:
    selected_by_source = [
        torch.tensor([[3, 0], [1, 0], [3, 0], [1, 0]], dtype=torch.int32),
        torch.tensor([[3, 0], [3, 0], [3, 0], [1, 0]], dtype=torch.int32),
    ]
    tpe_all, planning, inverse, validated = _plans_for_all_sources(
        selected_by_source
    )

    for source, expected in enumerate(validated):
        trusted = build_balanced_routing_metadata(
            selected_by_source[source],
            tpe_all,
            planning.alloc_cumsum,
            inverse,
            source,
            validate=False,
        )
        flat_experts = selected_by_source[source].reshape(-1)
        expert_order = torch.argsort(
            flat_experts.to(torch.float32), stable=True
        )
        reused = build_balanced_routing_metadata(
            selected_by_source[source],
            tpe_all,
            planning.alloc_cumsum,
            inverse,
            source,
            validate=False,
            expert_order=expert_order,
            sorted_experts=flat_experts[expert_order].contiguous(),
        )
        direct = build_balanced_routing_metadata(
            selected_by_source[source],
            tpe_all,
            planning.alloc_cumsum,
            inverse,
            source,
            validate=False,
            expert_order=expert_order,
            sorted_experts=flat_experts[expert_order].contiguous(),
            use_direct_bucket_scatter=True,
        )
        _assert_metadata_equal(trusted, expected)
        _assert_metadata_equal(reused, expected)
        _assert_metadata_equal(direct, expected)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_direct_bucket_scatter_randomly_matches_stable_sort(
    world_size: int,
) -> None:
    generator = torch.Generator().manual_seed(7300 + world_size)
    num_experts = world_size * 4
    routes_per_source = 64
    accepted_cases = 0

    for _ in range(24):
        selected_by_source = [
            torch.randint(
                num_experts,
                (routes_per_source // 2, 2),
                dtype=torch.int32,
                generator=generator,
            )
            for _ in range(world_size)
        ]
        tpe_all = torch.stack(
            [
                torch.bincount(
                    selected.reshape(-1).to(torch.int64),
                    minlength=num_experts,
                ).to(torch.int32)
                for selected in selected_by_source
            ]
        ).contiguous()
        try:
            planning = plan_moonep_b0_b3(tpe_all)
        except RuntimeError as error:
            if "fixed replica budget" not in str(error):
                raise
            continue
        inverse = build_inverse_experts_to_copy(
            planning.experts_to_copy, num_experts
        )

        for source, selected in enumerate(selected_by_source):
            stable = build_balanced_routing_metadata(
                selected,
                tpe_all,
                planning.alloc_cumsum,
                inverse,
                source,
            )
            direct = build_balanced_routing_metadata(
                selected,
                tpe_all,
                planning.alloc_cumsum,
                inverse,
                source,
                use_direct_bucket_scatter=True,
            )
            _assert_metadata_equal(direct, stable)
        accepted_cases += 1

    assert accepted_cases >= 12


def test_direct_bucket_scatter_flag_must_be_bool() -> None:
    selected = torch.tensor([[0, 1]], dtype=torch.int32)
    tpe_all = torch.tensor([[1, 1]], dtype=torch.int32)
    alloc_cumsum = torch.tensor([[1], [1]], dtype=torch.int32)
    inverse = torch.full((1, 2), -1, dtype=torch.int32)

    with pytest.raises(TypeError, match="use_direct_bucket_scatter"):
        build_balanced_routing_metadata(
            selected,
            tpe_all,
            alloc_cumsum,
            inverse,
            rank=0,
            use_direct_bucket_scatter=1,
        )


def test_local_histogram_must_match_published_counts() -> None:
    selected = torch.tensor([[0, 0]], dtype=torch.int32)
    tpe_all = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
    alloc_cumsum = torch.tensor([[1, 1], [0, 1]], dtype=torch.int32)
    inverse = torch.full((2, 2), -1, dtype=torch.int32)

    with pytest.raises(ValueError, match="histogram"):
        build_balanced_routing_metadata(
            selected, tpe_all, alloc_cumsum, inverse, rank=0
        )
