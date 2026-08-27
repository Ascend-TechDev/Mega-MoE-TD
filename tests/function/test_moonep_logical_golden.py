# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only semantic tests for the MoonEP routing contract.

The test deliberately does not use the Triton routing or communication path.
It models all EP ranks in one process, maps every route through MoonEP's
``alloc_cumsum``/``experts_to_copy`` tables, and evaluates it with the
physical home/replica weights.  The result must equal a direct logical-expert
Torch MoE evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)


def _route_mlp(
    hidden: torch.Tensor,
    route_weight: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one routed token with one logical or physical expert."""
    activation = F.silu(hidden @ gate.T) * (hidden @ up.T)
    return (activation * route_weight) @ down.T


def _logical_expert_golden(
    hidden_by_source: list[torch.Tensor],
    weights_by_source: list[torch.Tensor],
    experts_by_source: list[torch.Tensor],
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> list[torch.Tensor]:
    """Reference output that ignores ownership, routing, and replica slots."""
    outputs = []
    for hidden, weights, experts in zip(
        hidden_by_source, weights_by_source, experts_by_source
    ):
        tokens, topk = experts.shape
        out = torch.zeros_like(hidden)
        for token in range(tokens):
            for route_slot in range(topk):
                expert = int(experts[token, route_slot])
                out[token] += _route_mlp(
                    hidden[token],
                    weights[token, route_slot],
                    gate[expert],
                    up[expert],
                    down[expert],
                )
        outputs.append(out)
    return outputs


def _physical_weight_tables(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    experts_to_copy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the intended [home slots | replica slots] layout on CPU."""
    world_size, replica_slots = experts_to_copy.shape
    num_experts, ffn_dim, hidden_size = gate.shape
    experts_per_rank = num_experts // world_size
    physical_slots = experts_per_rank + replica_slots

    physical_gate = torch.empty(
        world_size, physical_slots, ffn_dim, hidden_size, dtype=gate.dtype
    )
    physical_up = torch.empty_like(physical_gate)
    physical_down = torch.empty(
        world_size, physical_slots, hidden_size, ffn_dim, dtype=down.dtype
    )
    for destination in range(world_size):
        home_begin = destination * experts_per_rank
        for home_slot in range(experts_per_rank):
            expert = home_begin + home_slot
            physical_gate[destination, home_slot] = gate[expert]
            physical_up[destination, home_slot] = up[expert]
            physical_down[destination, home_slot] = down[expert]
        for replica_slot, expert in enumerate(experts_to_copy[destination].tolist()):
            if expert < 0:
                continue
            physical_slot = experts_per_rank + replica_slot
            physical_gate[destination, physical_slot] = gate[expert]
            physical_up[destination, physical_slot] = up[expert]
            physical_down[destination, physical_slot] = down[expert]
    return physical_gate, physical_up, physical_down


def _balanced_physical_execution(
    hidden_by_source: list[torch.Tensor],
    weights_by_source: list[torch.Tensor],
    experts_by_source: list[torch.Tensor],
    tpe_all: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    physical_gate: torch.Tensor,
    physical_up: torch.Tensor,
    physical_down: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    """Apply the proposed stable two-sort balanced routing on CPU.

    Returns source outputs, the aggregate [destination, logical-expert] route
    count, and per-source physical slots in final send order.  Keeping this
    independent from production routing makes it a semantic golden for the
    future Triton implementation.
    """
    world_size, num_experts = tpe_all.shape
    experts_per_rank = num_experts // world_size
    physical_slots = physical_gate.shape[1]
    received_by_destination_expert = torch.zeros(
        world_size, num_experts, dtype=torch.int64
    )
    outputs = []
    final_physical_slots = []

    for source, (hidden, weights, experts) in enumerate(
        zip(hidden_by_source, weights_by_source, experts_by_source)
    ):
        tokens, topk = experts.shape
        route_experts = experts.reshape(-1).to(torch.int64)
        route_tokens = torch.arange(tokens, dtype=torch.int64).repeat_interleave(topk)
        route_weights = weights.reshape(-1)
        num_routes = route_experts.numel()

        # First stable sort is the existing Mega expert order.
        expert_order = torch.argsort(route_experts, stable=True)
        sorted_experts = route_experts[expert_order]
        local_counts = torch.bincount(route_experts, minlength=num_experts)
        local_exclusive = torch.zeros(num_experts, dtype=torch.int64)
        local_exclusive[1:] = local_counts.cumsum(0)[:-1]
        local_ordinal = (
            torch.arange(num_routes, dtype=torch.int64)
            - local_exclusive[sorted_experts]
        )
        previous_rank_count = torch.zeros(num_routes, dtype=torch.int64)
        if source:
            previous_rank_count = tpe_all[:source].sum(0)[sorted_experts]
        global_ordinal = previous_rank_count + local_ordinal

        # First destination prefix strictly greater than the global ordinal.
        destination = torch.searchsorted(
            alloc_cumsum[sorted_experts], global_ordinal[:, None], right=True
        ).squeeze(1)
        assert bool((destination < world_size).all())

        owner = torch.div(sorted_experts, experts_per_rank, rounding_mode="floor")
        home_slot = sorted_experts.remainder(experts_per_rank)
        replica_slot = inverse_experts_to_copy[destination, sorted_experts]
        physical_slot = torch.where(
            destination == owner,
            home_slot,
            experts_per_rank + replica_slot,
        )
        assert bool((physical_slot >= 0).all())
        assert bool((physical_slot < physical_slots).all())

        balanced_bucket = destination * physical_slots + physical_slot
        bucket_order = torch.argsort(balanced_bucket, stable=True)
        final_order = expert_order[bucket_order]
        final_destination = destination[bucket_order]
        final_slot = physical_slot[bucket_order]
        final_experts = sorted_experts[bucket_order]

        # The final order must remain a permutation of original flattened IDs.
        torch.testing.assert_close(
            torch.sort(final_order).values,
            torch.arange(num_routes, dtype=torch.int64),
        )
        sorted_buckets = balanced_bucket[bucket_order]
        assert bool((sorted_buckets[1:] >= sorted_buckets[:-1]).all())

        route_outputs = torch.empty_like(hidden).repeat_interleave(topk, dim=0)
        for ordered_row in range(num_routes):
            original_route = int(final_order[ordered_row])
            destination_rank = int(final_destination[ordered_row])
            slot = int(final_slot[ordered_row])
            token = int(route_tokens[original_route])
            route_outputs[original_route] = _route_mlp(
                hidden[token],
                route_weights[original_route],
                physical_gate[destination_rank, slot],
                physical_up[destination_rank, slot],
                physical_down[destination_rank, slot],
            )
            received_by_destination_expert[destination_rank, final_experts[ordered_row]] += 1

        outputs.append(route_outputs.view(tokens, topk, -1).sum(1))
        final_physical_slots.append(final_slot)

    return outputs, received_by_destination_expert, final_physical_slots


def test_moonep_balanced_physical_execution_matches_logical_expert_golden_cpu():
    """A remote expert route must execute through its replica without changing output."""
    # Every source produces eight valid routes, satisfying the initial dropless
    # MoonEP planning contract.  Expert 0 is hot enough to migrate to rank 1.
    tpe_all = torch.tensor(
        [[5, 1, 0, 2], [3, 1, 0, 4]], dtype=torch.int32
    )
    result = plan_moonep_b0_b3(tpe_all)
    alloc_cumsum = result.alloc_cumsum
    experts_to_copy = result.experts_to_copy
    inverse = build_inverse_experts_to_copy(experts_to_copy, num_experts=4)

    assert alloc_cumsum.device.type == "cpu"
    assert alloc_cumsum.dtype == torch.int32
    assert alloc_cumsum.shape == (4, 2)
    assert experts_to_copy.shape == (2, 2)
    torch.testing.assert_close(alloc_cumsum[:, -1].to(torch.int64), tpe_all.sum(0).to(torch.int64))

    alloc = torch.diff(
        torch.cat((torch.zeros(4, 1, dtype=torch.int32), alloc_cumsum), dim=1),
        dim=1,
    ).T
    # B=epn means every remote expert receiving work has a valid replica slot.
    for destination in range(2):
        for expert in range(4):
            if alloc[destination, expert] and expert // 2 != destination:
                assert int(inverse[destination, expert]) >= 0

    generator = torch.Generator(device="cpu").manual_seed(1234)
    hidden_by_source = [
        torch.randn(4, 3, generator=generator),
        torch.randn(4, 3, generator=generator),
    ]
    weights_by_source = [
        torch.softmax(torch.randn(4, 2, generator=generator), dim=-1),
        torch.softmax(torch.randn(4, 2, generator=generator), dim=-1),
    ]
    experts_by_source = [
        torch.tensor([[0, 0], [0, 0], [0, 1], [3, 3]], dtype=torch.int32),
        torch.tensor([[0, 0], [0, 1], [3, 3], [3, 3]], dtype=torch.int32),
    ]
    gate = torch.randn(4, 5, 3, generator=generator)
    up = torch.randn(4, 5, 3, generator=generator)
    down = torch.randn(4, 3, 5, generator=generator)

    expected = _logical_expert_golden(
        hidden_by_source, weights_by_source, experts_by_source, gate, up, down
    )
    physical_gate, physical_up, physical_down = _physical_weight_tables(
        gate, up, down, experts_to_copy
    )
    actual, received, final_slots = _balanced_physical_execution(
        hidden_by_source,
        weights_by_source,
        experts_by_source,
        tpe_all.to(torch.int64),
        alloc_cumsum.to(torch.int64),
        inverse.to(torch.int64),
        physical_gate,
        physical_up,
        physical_down,
    )

    for actual_rank, expected_rank in zip(actual, expected):
        torch.testing.assert_close(actual_rank, expected_rank, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(received, alloc.to(torch.int64))
    # This case must exercise at least one remote physical replica slot.
    assert any(bool((slots >= 2).any()) for slots in final_slots)
