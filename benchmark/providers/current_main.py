"""Adapters for the exact current-main fused forward and backward arms."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from typing import Any, Mapping


REPOSITORY_URL = "https://gitcode.com/jzhoujg/Mega-MoE-TD.git"
REPOSITORY_COMMIT = "2887022970f6ae5c7ab5be096680809591100c30"
REPOSITORY_TREE = "2b5c447da8ff9ff788125aae5eec644579df105d"
FORWARD_ARM = "fused_current_main_forward"
BACKWARD_DEFAULT_ARM = "backward_default_torch_wgrad"
BACKWARD_TRITON_ARM = "backward_optin_triton_wgrad"


def describe() -> dict[str, Any]:
    return {
        "provider_id": "current-main-production",
        "test_only": False,
        "arms": [FORWARD_ARM, BACKWARD_DEFAULT_ARM, BACKWARD_TRITON_ARM],
        "source": {
            "repository_url": REPOSITORY_URL,
            "commit": REPOSITORY_COMMIT,
            "tree": REPOSITORY_TREE,
            "paths": [
                "src/mega_moe/ops/forward.py",
                "src/mega_moe/ops/backward.py",
                "src/mega_moe/kernels/dispatch_fc1.py",
                "src/mega_moe/kernels/weighted_swiglu.py",
                "src/mega_moe/kernels/fc2_combine.py",
            ],
        },
        "entrypoints": {
            FORWARD_ARM: "providers.current_main.execute_forward",
            BACKWARD_DEFAULT_ARM: "providers.current_main.execute_backward",
            BACKWARD_TRITON_ARM: "providers.current_main.execute_backward",
        },
    }


@contextmanager
def _wgrad_mode(use_triton: bool):
    name = "MOE_WGRAD_TRITON"
    previous = os.environ.get(name)
    if use_triton:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def execute_forward(fixture: Mapping[str, Any]):
    module = importlib.import_module("benchmark.layer.bench_full_forward")
    function = module.ascend_full_post_routing
    return function(
        fixture["op"],
        fixture["hidden_states"],
        fixture["selected_experts"],
        fixture["packed_w1"],
        fixture["down_weight"],
        fixture["routing_weights"],
    )


def execute_backward(fixture: Mapping[str, Any], *, use_triton_wgrad: bool):
    mega_moe = importlib.import_module("mega_moe")
    with _wgrad_mode(use_triton_wgrad):
        return mega_moe.moe_backward_triton(
            fixture["saved"], fixture["dy"], fixture["peer_mem"]
        )
