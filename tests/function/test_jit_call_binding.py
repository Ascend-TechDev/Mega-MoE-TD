# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the fused-forward jit call bindings.

The FC1 group helper takes the wave geometry (``wave_expert`` /
``wave_row_start`` / ``wave_rows`` / ``replica_ready_ptr`` /
``WEIGHT_EXPERT_BASE`` / ``WAIT_REPLICA``) as a positional run, followed
by experiment knobs (save format, the MOE_FWD_TIMING dead args).
Inserting a new knob into the positional run shifts the geometry one slot
and zeroes most expert groups — exactly the w8/w4 correctness failure of
2026-09-16, which compiled fine and only failed numerically.

These checks freeze the convention that guards it: the knob tail of the
touched helpers must be bound by keyword at every call site, keyword names
must exist in the callee signature, and the positional prefix must end
exactly where the keyword tail begins (no gaps, no double binding).
"""

import ast
from pathlib import Path

import pytest

_FUSED_FORWARD = (
    Path(__file__).resolve().parents[2]
    / "src" / "mega_moe" / "kernels" / "fused_forward.py"
)

# Helpers whose trailing knobs must be keyword-bound, with the tail params
# each call site must cover (only those present in the signature count —
# branches add/remove knobs).
_HELPERS = {
    "_partition_pipeline_fc1_activation_group_ub": (
        "FC1_FP8", "FC1_SAVE_FP16", "ring_base", "ring_slot",
        "TIMING", "RING_SLOTS",
    ),
    "_run_dynamic_wave_pipeline": (
        "acc_ptr", "ring_ptr", "fc2w_ptr", "TIMING", "ACC_SLOTS",
        "RING_SLOTS",
    ),
    "_return_dynamic_wave": (
        "fc2_wave_start", "TIMING", "dummy",
    ),
}


def _parse():
    return ast.parse(_FUSED_FORWARD.read_text(encoding="utf-8"))


def _signature_params(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [
                arg.arg
                for arg in node.args.args + node.args.kwonlyargs
            ]
    raise AssertionError(f"helper {name!r} not found in fused_forward.py")


def _call_sites(tree, name):
    sites = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    assert sites, f"no call sites found for {name!r}"
    return sites


@pytest.mark.parametrize("helper", sorted(_HELPERS))
def test_knob_tail_is_keyword_bound(helper):
    tree = _parse()
    params = _signature_params(tree, helper)
    required = [name for name in _HELPERS[helper] if name in params]
    assert required, f"{helper} lost every knob param — update the test"
    for call in _call_sites(tree, helper):
        keyword_names = [keyword.arg for keyword in call.keywords]
        # No unknown/duplicated keywords, nothing double-bound.
        assert len(keyword_names) == len(set(keyword_names))
        for keyword in keyword_names:
            assert keyword in params, f"{helper}: unknown keyword {keyword}"
        # The positional run must stop before the keyword tail: every
        # keyword param must sit after the last positional slot, and the
        # prefix plus keywords must cover the whole signature.
        prefix = len(call.args)
        tail = params[prefix:]
        assert sorted(keyword_names) == sorted(tail), (
            f"{helper}: keywords {sorted(keyword_names)} do not exactly "
            f"cover the params after the positional prefix {sorted(tail)}"
        )
        for name in required:
            assert name in keyword_names, (
                f"{helper}: knob {name} must be keyword-bound, not "
                "positional (inserting a positional arg here shifts the "
                "wave geometry)"
            )
