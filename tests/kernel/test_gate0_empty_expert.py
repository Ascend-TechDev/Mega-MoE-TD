# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only Gate-0 controls for empty-expert weight gradients.

The Triton wrapper tests replace the device launcher with a CPU recorder. They bind
the allocation, expert metadata, and non-empty writes in the exact wrapper, but they
do not replace the remaining ``use_bytecode=True`` Ascend device semantic gate.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "src" / "mega_moe" / "kernels"


def _load_backward(monkeypatch):
    kernels = types.ModuleType("mega_moe.kernels")
    kernels.__path__ = []
    for name in (
        "dispatch_fc2_bwd_triton",
        "swiglu_bwd_triton",
        "transposed_grouped_gemm_triton",
        "combine_fc1_bwd_triton",
    ):
        setattr(kernels, name, lambda *args, **kwargs: None)

    torch_forward = types.ModuleType("mega_moe.ops._torch_forward")
    torch_forward.moe_forward = lambda *args, **kwargs: None
    common = types.ModuleType("mega_moe.kernels.common")
    common.ncore = lambda: 32
    common.validate_wgrad_launch_params = lambda **kwargs: None
    mega_moe = types.ModuleType("mega_moe")
    mega_moe.__path__ = []
    ops = types.ModuleType("mega_moe.ops")
    ops.__path__ = []
    for name, module in {
        "torch_npu": types.ModuleType("torch_npu"),
        "mega_moe": mega_moe,
        "mega_moe.ops": ops,
        "mega_moe.kernels": kernels,
        "mega_moe.kernels.common": common,
        "mega_moe.ops._torch_forward": torch_forward,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "mega_moe.ops.backward", ROOT / "src" / "mega_moe" / "ops" / "backward.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _load_transposed_gemm(monkeypatch, tmp_path):
    common = types.ModuleType("_gate0_common")
    common.WGRAD_BLOCK_M = 256
    common.WGRAD_BLOCK_N = 128
    common.WGRAD_BLOCK_K = 256
    common.ncore = lambda: 32
    common.validate_wgrad_launch_params = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, common.__name__, common)

    source = (KERNELS / "transposed_grouped_gemm.py").read_text().replace(
        "from .common import", f"from {common.__name__} import"
    )
    target = tmp_path / "_gate0_tgg.py"
    target.write_text(source)
    spec = importlib.util.spec_from_file_location("_gate0_tgg", target)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _golden(grad_out, orig_in, counts):
    result = torch.zeros(len(counts), grad_out.shape[1], orig_in.shape[1])
    start = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            result[expert] = grad_out[start : start + count].T @ orig_in[start : start + count]
            start += count
    return result


def _seeded_inputs(seed, counts, n=5, k=7):
    generator = torch.Generator().manual_seed(seed)
    rows = int(counts.sum().item())
    return (
        torch.randn(rows, n, generator=generator),
        torch.randn(rows, k, generator=generator),
    )


@pytest.mark.parametrize("seed", [17, 83])
def test_torch_wgrad_zeroes_poisoned_empty_experts_and_preserves_non_empty(
    monkeypatch, seed
):
    backward = _load_backward(monkeypatch)
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        return real_empty(*args, **kwargs).fill_(37)

    monkeypatch.setattr(backward.torch, "empty", poisoned_empty)
    counts = torch.tensor([2, 0, 1, 0, 3], dtype=torch.int64)
    grad_out, orig_in = _seeded_inputs(seed, counts)

    actual = backward._grouped_wgrad_torch(grad_out, orig_in, counts)
    expected = _golden(grad_out, orig_in, counts)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[counts == 0]).item() == 0


def test_torch_wgrad_all_non_empty_positive_control(monkeypatch):
    backward = _load_backward(monkeypatch)
    counts = torch.tensor([1, 2, 3], dtype=torch.int64)
    grad_out, orig_in = _seeded_inputs(101, counts)

    assert torch.equal(
        backward._grouped_wgrad_torch(grad_out, orig_in, counts),
        _golden(grad_out, orig_in, counts),
    )


class _NonEmptyKernelRecorder:
    """Emulate only valid non-empty stores; empty slices retain allocation state."""

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            grad_out_t, orig_in, grad_w = args[:3]
            counts = args[4]
            start = 0
            for expert, count in enumerate(counts.tolist()):
                if count:
                    grad_w[expert].copy_(
                        grad_out_t[:, start : start + count]
                        @ orig_in[start : start + count]
                    )
                    start += count

        return call


@pytest.mark.parametrize("seed", [17, 83])
def test_triton_wrapper_zeroes_poisoned_empty_experts_and_preserves_non_empty(
    monkeypatch, tmp_path, seed
):
    module = _load_transposed_gemm(monkeypatch, tmp_path)
    real_empty = torch.empty

    def poisoned_empty(*args, **kwargs):
        return real_empty(*args, **kwargs).fill_(37)

    monkeypatch.setattr(module.torch, "empty", poisoned_empty)
    module.kernel_transposed_grouped_gemm = _NonEmptyKernelRecorder()
    counts = torch.tensor([2, 0, 1, 0, 3], dtype=torch.int32)
    grad_out, orig_in = _seeded_inputs(seed, counts)
    cumulative = torch.zeros(len(counts) + 1, dtype=torch.int32)
    cumulative[1:] = counts.cumsum(0)

    actual = module.transposed_grouped_gemm_triton(
        grad_out, orig_in, counts, cumulative
    )
    expected = _golden(grad_out, orig_in, counts)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[counts == 0]).item() == 0


def test_triton_wrapper_all_non_empty_positive_control(monkeypatch, tmp_path):
    module = _load_transposed_gemm(monkeypatch, tmp_path)
    module.kernel_transposed_grouped_gemm = _NonEmptyKernelRecorder()
    counts = torch.tensor([1, 2, 3], dtype=torch.int32)
    grad_out, orig_in = _seeded_inputs(101, counts)
    cumulative = torch.zeros(len(counts) + 1, dtype=torch.int32)
    cumulative[1:] = counts.cumsum(0)

    assert torch.equal(
        module.transposed_grouped_gemm_triton(
            grad_out, orig_in, counts, cumulative
        ),
        _golden(grad_out, orig_in, counts),
    )


def test_triton_store_is_guarded_against_empty_experts():
    tree = ast.parse((KERNELS / "transposed_grouped_gemm.py").read_text())
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "kernel_transposed_grouped_gemm"
    )
    stores = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    ]

    assert len(stores) == 1
    mask = next(keyword.value for keyword in stores[0].keywords if keyword.arg == "mask")
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "split_size"
        and isinstance(node.ops[0], ast.Gt)
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value == 0
        for node in ast.walk(mask)
    ), "the device store must not overwrite zero-initialized empty-expert slices"
