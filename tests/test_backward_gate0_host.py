# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only controls for backward Gate 0 correctness and harness status."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch, name, relative_path, stubs):
    for module_name, module in stubs.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _npu_stubs():
    torch_npu = types.ModuleType("torch_npu")
    kernels = types.ModuleType("mega_moe.kernels")
    for name in (
        "dispatch_fc2_bwd_triton",
        "swiglu_bwd_triton",
        "transposed_grouped_gemm_triton",
        "combine_fc1_bwd_triton",
    ):
        setattr(kernels, name, lambda *args, **kwargs: None)
    golden = types.ModuleType("mega_moe.ops._legacy_backward_golden")
    golden.moe_forward = lambda *args, **kwargs: None
    return {
        "torch_npu": torch_npu,
        "mega_moe": types.ModuleType("mega_moe"),
        "mega_moe.ops": types.ModuleType("mega_moe.ops"),
        "mega_moe.kernels": kernels,
        "mega_moe.ops._legacy_backward_golden": golden,
    }


def _load_backward(monkeypatch):
    return _load_module(
        monkeypatch,
        "mega_moe.ops.backward",
        "src/mega_moe/ops/backward.py",
        _npu_stubs(),
    )


def _load_harness(monkeypatch, kind):
    mega_moe = types.ModuleType("mega_moe")
    mega_moe.moe_backward_triton = lambda *args, **kwargs: None
    mega_moe.MegaMoEBackwardFunction = object
    golden = types.ModuleType("mega_moe.ops._legacy_backward_golden")
    golden.moe_forward = lambda *args, **kwargs: None
    golden.moe_backward_torch = lambda *args, **kwargs: None
    shmem = types.ModuleType("shmem")
    path = {
        "layer": "tests/layer/test_moe_backward.py",
        "function": "tests/function/test_moe_backward_function.py",
    }[kind]
    return _load_module(
        monkeypatch,
        f"gate0_{kind}_harness",
        path,
        {
            "torch_npu": types.ModuleType("torch_npu"),
            "shmem": shmem,
            "mega_moe": mega_moe,
            "mega_moe.ops": types.ModuleType("mega_moe.ops"),
            "mega_moe.ops._legacy_backward_golden": golden,
        },
    )


def _stub_distributed_runtime(monkeypatch, harness):
    class InitAttr:
        def __init__(self):
            self.option_attr = types.SimpleNamespace()

    monkeypatch.setattr(
        harness,
        "ash",
        types.SimpleNamespace(
            InitAttr=InitAttr,
            OpEngineType=types.SimpleNamespace(MTE="mte"),
            set_conf_store_tls=lambda *args: None,
            aclshmem_init=lambda attr: 0,
            aclshmem_finalize=lambda: 0,
        ),
    )
    monkeypatch.setattr(
        harness,
        "dist",
        types.SimpleNamespace(
            group=types.SimpleNamespace(WORLD="world"),
            get_rank=lambda group=None: 0,
            get_world_size=lambda group=None: 1,
            barrier=lambda *args: None,
        ),
    )
    monkeypatch.setattr(harness, "_all_ranks_pass", lambda passed, *args: passed)


def _stub_main_runtime(monkeypatch, harness):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(
        harness.torch,
        "npu",
        types.SimpleNamespace(set_device=lambda rank: None),
        raising=False,
    )
    monkeypatch.setattr(
        harness,
        "dist",
        types.SimpleNamespace(
            init_process_group=lambda **kwargs: None,
            get_world_size=lambda: 1,
            barrier=lambda: None,
        ),
    )


def test_grouped_wgrad_empty_expert_matches_independent_golden(monkeypatch):
    backward = _load_backward(monkeypatch)
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        return real_empty(*args, **kwargs).fill_(37)

    monkeypatch.setattr(backward.torch, "empty", poisoned_empty)
    counts = torch.tensor([2, 0, 1], dtype=torch.int64)
    grad_out = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    orig_in = torch.tensor([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])

    actual = backward._grouped_wgrad_torch(grad_out, orig_in, counts)
    golden = torch.zeros(3, 2, 2)
    golden[0] = grad_out[:2].T @ orig_in[:2]
    golden[2] = grad_out[2:].T @ orig_in[2:]

    assert torch.equal(actual, golden)
    assert torch.count_nonzero(actual[1]).item() == 0


@pytest.mark.parametrize("kind", ["layer", "function"])
@pytest.mark.parametrize("passed", [True, False])
def test_backward_harness_distributed_verdict(monkeypatch, kind, passed):
    harness = _load_harness(monkeypatch, kind)
    _stub_distributed_runtime(monkeypatch, harness)
    if kind == "layer":
        result = ("controlled", 1.0, 1.0, 1.0, passed)
    else:
        result = passed
    monkeypatch.setattr(harness, "run_one", lambda *args: result)

    assert harness.run_test_distributed() is passed


@pytest.mark.parametrize("kind", ["layer", "function"])
@pytest.mark.parametrize("passed, expected", [(True, 0), (False, 1)])
def test_backward_harness_main_exit_code(monkeypatch, kind, passed, expected):
    harness = _load_harness(monkeypatch, kind)
    _stub_main_runtime(monkeypatch, harness)
    monkeypatch.setattr(harness, "run_test_distributed", lambda: passed)

    assert harness.main() == expected


def test_layer_harness_skip_is_enumerated_and_machine_readable(monkeypatch):
    harness = _load_harness(monkeypatch, "layer")
    skip = harness.HarnessSkip(
        harness.SkipReason.SYMMETRIC_HEAP_CAPACITY,
        required_bytes=2048,
        configured_bytes=1024,
    )
    rendered = harness._format_skip(("small", 8, 16, 32, 2, 4), skip)
    payload = json.loads(rendered)

    assert payload["event"] == "skip"
    assert payload["reason"] == "symmetric_heap_capacity"
    assert payload["required_bytes"] == 2048
    assert payload["configured_bytes"] == 1024


def test_layer_harness_all_explicit_skips_fail_without_evidence(monkeypatch, capsys):
    harness = _load_harness(monkeypatch, "layer")
    _stub_distributed_runtime(monkeypatch, harness)

    def skip_capacity(*args):
        raise harness.HarnessSkip(
            harness.SkipReason.SYMMETRIC_HEAP_CAPACITY,
            required_bytes=2048,
            configured_bytes=1024,
        )

    monkeypatch.setattr(harness, "run_one", skip_capacity)

    assert harness.run_test_distributed() is False
    skip_lines = [line for line in capsys.readouterr().out.splitlines() if "[skip]" in line]
    assert skip_lines
    assert all(json.loads(line.split("[skip] ", 1)[1])["reason"] == "symmetric_heap_capacity" for line in skip_lines)


def test_layer_harness_unexpected_exception_is_not_converted_to_skip(monkeypatch):
    harness = _load_harness(monkeypatch, "layer")
    _stub_distributed_runtime(monkeypatch, harness)
    monkeypatch.setattr(harness, "run_one", lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        harness.run_test_distributed()


def test_layer_harness_main_propagates_unexpected_exception(monkeypatch):
    harness = _load_harness(monkeypatch, "layer")
    _stub_main_runtime(monkeypatch, harness)
    monkeypatch.setattr(
        harness,
        "run_test_distributed",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        harness.main()
