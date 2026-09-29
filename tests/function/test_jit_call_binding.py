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
import builtins
import inspect
from pathlib import Path
import symtable

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


def test_local_helper_calls_bind_complete_signatures():
    tree = _parse()
    signatures = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        parameters = []
        arguments = node.args.posonlyargs + node.args.args
        required = len(arguments) - len(node.args.defaults)
        for index, argument in enumerate(arguments):
            kind = (inspect.Parameter.POSITIONAL_ONLY
                    if index < len(node.args.posonlyargs)
                    else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            parameters.append(inspect.Parameter(
                argument.arg, kind,
                default=inspect.Parameter.empty if index < required else None,
            ))
        if node.args.vararg:
            parameters.append(inspect.Parameter(
                node.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            parameters.append(inspect.Parameter(
                argument.arg, inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None,
            ))
        if node.args.kwarg:
            parameters.append(inspect.Parameter(
                node.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
        signatures[node.name] = inspect.Signature(parameters)
    for call in ast.walk(tree):
        if (not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Name)
                or call.func.id not in signatures):
            continue
        assert not any(isinstance(arg, ast.Starred) for arg in call.args)
        names = [keyword.arg for keyword in call.keywords]
        assert None not in names and len(names) == len(set(names))
        try:
            signatures[call.func.id].bind(
                *([None] * len(call.args)), **dict.fromkeys(names))
        except TypeError as exc:
            pytest.fail(f"{call.func.id} at line {call.lineno}: {exc}")


def test_fused_forward_helpers_have_no_unbound_global_names():
    table = symtable.symtable(
        _FUSED_FORWARD.read_text(encoding="utf-8"), str(_FUSED_FORWARD), "exec")
    known = set(dir(builtins)) | {
        symbol.get_name() for symbol in table.get_symbols()
        if symbol.is_assigned() or symbol.is_imported()
    }
    for function in table.get_children():
        missing = {
            symbol.get_name() for symbol in function.get_symbols()
            if symbol.is_global() and symbol.is_referenced()
            and symbol.get_name() not in known
        }
        assert not missing, f"{function.get_name()}: unbound names {sorted(missing)}"


def test_host_launch_binds_wave_workspaces_in_order():
    root = _FUSED_FORWARD.parents[3]
    host = ast.parse((root / "src/mega_moe/ops/forward.py").read_text())
    params = _signature_params(_parse(), "_kernel_fused_forward")
    launches = [node for node in ast.walk(host)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_kernel_fused_forward"]
    assert len(launches) == 1
    call = launches[0]
    names = [keyword.arg for keyword in call.keywords if keyword.arg is not None]
    options = [keyword.value for keyword in call.keywords if keyword.arg is None]
    assert len(options) == 1
    assert isinstance(options[0], ast.Name) and options[0].id == "launch_options"
    assert len(names) == len(set(names))
    assert set(params[len(call.args):]) == set(names)
    for parameter, attribute in (
        ("wave_expert_offsets_ptr", "_single_wave_expert_offsets"),
        ("wave_task_offsets_ptr", "_single_wave_task_offsets"),
        ("wave_tasks_ptr", "_single_wave_tasks"),
        ("raw_counts_ptr", None),
    ):
        argument = call.args[params.index(parameter)]
        if attribute is not None:
            assert isinstance(argument, ast.Attribute) and argument.attr == attribute
        else:
            assert isinstance(argument, ast.IfExp)
            assert isinstance(argument.body, ast.Attribute)
            assert argument.body.attr == "planning_counts_mem"
