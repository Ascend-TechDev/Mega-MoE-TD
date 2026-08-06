# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only contracts for validated wgrad tile and grid controls."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "src" / "mega_moe" / "kernels"


def _load_common(monkeypatch, core_count, name="_wgrad_common"):
    class NPUUtils:
        def get_aicore_num(self):
            return core_count

    driver = types.ModuleType("triton.backends.ascend.driver")
    driver.NPUUtils = NPUUtils
    ascend = types.ModuleType("triton.backends.ascend")
    ascend.__path__ = []
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "triton.backends.ascend", ascend)
    monkeypatch.setitem(sys.modules, "triton.backends.ascend.driver", driver)

    spec = importlib.util.spec_from_file_location(name, KERNELS / "common.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_kernel(monkeypatch, tmp_path, core_count=48):
    common = _load_common(monkeypatch, core_count)
    source = (KERNELS / "transposed_grouped_gemm.py").read_text().replace(
        "from .common import", f"from {common.__name__} import"
    )
    target = tmp_path / "_tgg_under_test.py"
    target.write_text(source)
    spec = importlib.util.spec_from_file_location("_tgg_under_test", target)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, common


class _Recorder:
    def __init__(self):
        self.grid = None
        self.kwargs = None
        self.args = None

    def __getitem__(self, grid):
        self.grid = grid

        def call(*args, **kwargs):
            self.kwargs = kwargs
            self.args = args

        return call


def _invoke(module, **overrides):
    recorder = _Recorder()
    module.kernel_transposed_grouped_gemm = recorder
    experts, rows, output_dim, reduction_dim = 4, 64, 512, 512
    grad_out = torch.zeros(rows, output_dim)
    orig_in = torch.zeros(rows, reduction_dim)
    counts = torch.full((experts,), rows // experts, dtype=torch.int32)
    cumulative = torch.zeros(experts + 1, dtype=torch.int32)
    cumulative[1:] = counts.cumsum(0)
    module.transposed_grouped_gemm_triton(
        grad_out, orig_in, counts, cumulative, **overrides
    )
    return recorder


@pytest.mark.parametrize("core_count", [16, 48])
def test_ncore_returns_the_device_count_without_a_fixed_24_limit(monkeypatch, core_count):
    common = _load_common(monkeypatch, core_count, f"_common_{core_count}")
    assert common.ncore() == core_count


@pytest.mark.parametrize("core_count", [16, 48])
def test_defaults_preserve_shipped_tiles_and_device_grid(
    monkeypatch, tmp_path, core_count
):
    module, common = _load_kernel(monkeypatch, tmp_path, core_count)
    recorder = _invoke(module)

    assert recorder.kwargs["BLOCK_M"] == common.WGRAD_BLOCK_M
    assert recorder.kwargs["BLOCK_N"] == common.WGRAD_BLOCK_N
    assert recorder.kwargs["BLOCK_K"] == common.WGRAD_BLOCK_K
    assert recorder.grid == (core_count, 1, 1)


def test_overrides_reach_the_low_level_kernel(monkeypatch, tmp_path):
    module, _ = _load_kernel(monkeypatch, tmp_path, core_count=64)
    recorder = _invoke(
        module, block_m=512, block_n=256, block_k=128, grid=48
    )

    assert recorder.kwargs["BLOCK_M"] == 512
    assert recorder.kwargs["BLOCK_N"] == 256
    assert recorder.kwargs["BLOCK_K"] == 128
    assert recorder.grid == (48, 1, 1)


def test_block_n_override_independently_derives_num_tiles_n(monkeypatch, tmp_path):
    module, _ = _load_kernel(monkeypatch, tmp_path)
    recorder = _invoke(module, block_n=256)

    assert recorder.args[8] == 2
    assert recorder.args[9] == 2


def test_block_k_override_independently_derives_num_tiles_k(monkeypatch, tmp_path):
    module, _ = _load_kernel(monkeypatch, tmp_path)
    recorder = _invoke(module, block_k=128)

    assert recorder.args[8] == 4
    assert recorder.args[9] == 4


@pytest.mark.parametrize("name", ["block_m", "block_n", "block_k"])
@pytest.mark.parametrize("value", [True, 1.5, "32"])
def test_non_integer_block_sizes_fail_closed(monkeypatch, tmp_path, name, value):
    module, _ = _load_kernel(monkeypatch, tmp_path)
    with pytest.raises(TypeError, match=rf"{name} must be an integer"):
        _invoke(module, **{name: value})


@pytest.mark.parametrize("name", ["block_m", "block_n", "block_k"])
@pytest.mark.parametrize(
    "value, message",
    [(0, "must be positive"), (-16, "must be positive"), (8, "at least 16"), (48, "power of two")],
)
def test_invalid_integer_block_sizes_fail_closed(
    monkeypatch, tmp_path, name, value, message
):
    module, _ = _load_kernel(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message):
        _invoke(module, **{name: value})


@pytest.mark.parametrize("value", [True, 1.5, "32"])
def test_non_integer_grid_fails_closed(monkeypatch, tmp_path, value):
    module, _ = _load_kernel(monkeypatch, tmp_path, core_count=64)
    with pytest.raises(TypeError, match="grid must be an integer"):
        _invoke(module, grid=value)


@pytest.mark.parametrize(
    "value, message",
    [(0, "grid must be positive"), (-1, "grid must be positive"), (65, "physical AICore count 64")],
)
def test_invalid_integer_grid_fails_closed(monkeypatch, tmp_path, value, message):
    module, _ = _load_kernel(monkeypatch, tmp_path, core_count=64)
    with pytest.raises(ValueError, match=message):
        _invoke(module, grid=value)


def _load_backward(monkeypatch, core_count=64):
    common = _load_common(
        monkeypatch, core_count, "mega_moe.kernels.common"
    )
    calls = []

    def transposed(grad_out, orig_in, expert_counts, cumulative, **kwargs):
        calls.append((grad_out, orig_in, kwargs))
        return torch.zeros(
            len(expert_counts), grad_out.shape[1], orig_in.shape[1]
        )

    kernels = types.ModuleType("mega_moe.kernels")
    kernels.__path__ = []
    kernels.dispatch_fc2_bwd_triton = (
        lambda saved, dy, peer_mem: (saved["grad_swiglu"], saved["fc2_grad_out"])
    )
    kernels.swiglu_bwd_triton = (
        lambda grad_swiglu, fc1_output, weights: (
            torch.zeros_like(fc1_output),
            torch.zeros_like(weights),
        )
    )
    kernels.transposed_grouped_gemm_triton = transposed
    kernels.combine_fc1_bwd_triton = (
        lambda saved, grad_fc1_output, grad_gate, peer_mem: (
            saved["grad_hidden"],
            saved["grad_routing_weights"],
        )
    )

    torch_forward = types.ModuleType("mega_moe.ops._torch_forward")
    torch_forward.moe_forward = lambda *args, **kwargs: None
    mega_moe = types.ModuleType("mega_moe")
    mega_moe.__path__ = []
    ops = types.ModuleType("mega_moe.ops")
    ops.__path__ = []
    for name, module in {
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
    return module, calls


def _saved_for_backward():
    experts, rows, hidden, ffn = 3, 6, 4, 8
    return {
        "fc1_1": torch.zeros(experts, ffn, hidden),
        "fc1_output": torch.zeros(rows, ffn * 2),
        "recv_weights_sorted": torch.zeros(rows),
        "swiglu_out_weighted": torch.zeros(rows, ffn),
        "recv_hidden_sorted": torch.zeros(rows, hidden),
        "expert_counts": torch.tensor([1, 2, 3], dtype=torch.int32),
        "split_size_cum_per_expert": torch.tensor([0, 1, 3, 6], dtype=torch.int32),
        "grad_swiglu": torch.zeros(rows, ffn),
        "fc2_grad_out": torch.zeros(rows, hidden),
        "grad_hidden": torch.zeros(rows, hidden),
        "grad_routing_weights": torch.zeros(rows),
    }


def test_public_backward_defaults_keep_both_wgrad_calls_positional(monkeypatch):
    backward, calls = _load_backward(monkeypatch)
    backward.moe_backward_triton(
        _saved_for_backward(), torch.zeros(6, 4), object()
    )

    assert [kwargs for _, _, kwargs in calls] == [{}, {}]


def test_public_backward_wires_independent_fc2_and_fc1_controls(monkeypatch):
    backward, calls = _load_backward(monkeypatch)
    backward.moe_backward_triton(
        _saved_for_backward(),
        torch.zeros(6, 4),
        object(),
        fc2_wgrad_block_m=512,
        fc2_wgrad_block_n=256,
        fc2_wgrad_block_k=128,
        fc2_wgrad_grid=48,
        fc1_wgrad_block_m=128,
        fc1_wgrad_block_n=64,
        fc1_wgrad_block_k=256,
        fc1_wgrad_grid=32,
    )

    assert calls[0][2] == {
        "block_m": 512,
        "block_n": 256,
        "block_k": 128,
        "grid": 48,
    }
    assert calls[1][2] == {
        "block_m": 128,
        "block_n": 64,
        "block_k": 256,
        "grid": 32,
    }


def test_public_backward_rejects_invalid_controls_before_any_kernel(monkeypatch):
    backward, calls = _load_backward(monkeypatch)
    with pytest.raises(ValueError, match="fc2_wgrad_block_n.*power of two"):
        backward.moe_backward_triton(
            _saved_for_backward(),
            torch.zeros(6, 4),
            object(),
            fc2_wgrad_block_n=48,
        )
    assert calls == []


@pytest.mark.parametrize("stage", ["fc1", "fc2"])
def test_public_backward_rejects_overrides_hidden_by_torch_fallback(
    monkeypatch, stage
):
    backward, calls = _load_backward(monkeypatch)
    monkeypatch.setenv(f"MOE_{stage.upper()}_WGRAD_TORCH", "1")
    with pytest.raises(ValueError, match=rf"{stage} wgrad overrides.*Torch fallback"):
        backward.moe_backward_triton(
            _saved_for_backward(),
            torch.zeros(6, 4),
            object(),
            **{f"{stage}_wgrad_block_m": 512},
        )
    assert calls == []
