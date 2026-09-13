# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The trimmed benchmark must retain imbalance and distinct top-k choices."""

import pytest
import torch

from benchmark.layer._kimi_routes import KIMI_OWNER_COUNTS, kimi_skewed_routes
from config import resolve_case
from mega_moe.runtime.moonep_planning import plan_moonep_b0_b3


@pytest.mark.parametrize("experts", (32, 896))
@pytest.mark.parametrize("topk", (8, 16))
def test_kimi_owner_loads_and_unique_choices(experts, topk):
    tokens = 4096
    rows = []
    for rank in range(8):
        routes = kimi_skewed_routes(tokens, experts, topk, rank)
        ordered = routes.sort(dim=1).values
        assert torch.all(ordered[:, 1:] != ordered[:, :-1])
        assert routes.min() >= 0 and routes.max() < experts
        counts = torch.bincount(routes.flatten().long(), minlength=experts)
        assert counts.view(8, experts // 8).sum(1).tolist() == [
            count * tokens * topk // 128 for count in KIMI_OWNER_COUNTS]
        rows.append(counts)
    plan = plan_moonep_b0_b3(torch.stack(rows))
    allocation = torch.diff(plan.alloc_cumsum, dim=1, prepend=torch.zeros(experts, 1)).t()
    assert allocation.sum(1).tolist() == [tokens * topk] * 8
    if experts == 32:
        assert (plan.experts_to_copy >= 0).sum(1).tolist() == [0, 0, 1, 1, 1, 1, 1, 1]


def test_skewed_full_and_trimmed_shapes_only_change_experts():
    full = resolve_case("performance-fwd-kimi-k3-skewed-w8-t4k")
    trimmed = resolve_case("performance-fwd-kimi-k3-trimmed-skewed-w8-t4k")
    for field in ("tokens", "world_size", "hidden", "ffn", "topk", "capacity_factor"):
        assert getattr(full, field) == getattr(trimmed, field)
    assert (full.num_experts, trimmed.num_experts) == (896, 32)
