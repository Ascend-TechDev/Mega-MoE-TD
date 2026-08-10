"""Current-main unfused grouped-GEMM + HCCL forward provider."""

from __future__ import annotations

import importlib
from typing import Any, Mapping


REPOSITORY_URL = "https://gitcode.com/jzhoujg/Mega-MoE-TD.git"
REPOSITORY_COMMIT = "2887022970f6ae5c7ab5be096680809591100c30"
REPOSITORY_TREE = "2b5c447da8ff9ff788125aae5eec644579df105d"
ARM_ID = "unfused_grouped_hccl_forward"


def describe() -> dict[str, Any]:
    return {
        "provider_id": "current-main-grouped-hccl",
        "test_only": False,
        "arms": [ARM_ID],
        "source": {
            "repository_url": REPOSITORY_URL,
            "commit": REPOSITORY_COMMIT,
            "tree": REPOSITORY_TREE,
            "paths": ["benchmark/layer/bench_full_forward.py"],
        },
        "entrypoint": "providers.grouped_hccl.execute_forward",
    }


def execute_forward(fixture: Mapping[str, Any]):
    """Call the exact current-main unfused provider, never its old runner."""
    module = importlib.import_module("benchmark.layer.bench_full_forward")
    function = module.torch_npu_grouped_hccl_full_post_routing
    return function(
        fixture["hidden_states"],
        fixture["selected_experts"],
        fixture["routing_weights"],
        fixture["packed_w1"],
        fixture["torch_w2_kn"],
        fixture["experts_per_rank"],
        fixture["ep_group"],
    )
