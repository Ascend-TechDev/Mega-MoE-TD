# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only contract test for the wgrad tile parameterisation.

Two properties, and BOTH are needed. Testing only the first would be satisfied by an
implementation that accepts the new arguments and silently ignores them — the defaults
would still be right, and the override would be dead. That failure mode is invisible
from the default path alone, so the override arm is what makes this test mean anything.

  P1  an unchanged call site passes exactly the shipped constants and `ncore()`
      => the parameterisation cannot alter existing behaviour
  P2  an override actually reaches the kernel, AND changes the derived tile counts
      => the argument is live, not decorative

No NPU is required: the launcher is replaced with a recorder, so this exercises the
plumbing that was changed and nothing else.
"""
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _load_module(monkeypatch):
    """Import the kernel module without the package __init__, which pulls in the
    distributed kernels (triton_dist) — unavailable on a host-only runner and unrelated
    to this contract."""
    import importlib.util
    import pathlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "mega_moe" / "kernels"
    common_src = (root / "common.py").read_text()
    common = types.ModuleType("_c")
    # only the four names the target module imports; executing common.py wholesale would
    # drag in torch.distributed helpers this test has no business touching
    for line in common_src.splitlines():
        if line.startswith(("WGRAD_BLOCK_", "BLOCK_SIZE_")):
            exec(line, common.__dict__)
    common.ncore = lambda: 24
    sys.modules["_c"] = common

    text = (root / "transposed_grouped_gemm.py").read_text().replace(
        "from .common import", "from _c import")
    # @triton.jit requires the decorated function to live in a real file on disk, so the
    # rewritten source is materialised rather than exec'd from a string.
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp()) / "_tgg_under_test.py"
    tmp.write_text(text)
    spec = importlib.util.spec_from_file_location("_tgg_under_test", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tgg_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod, common


class _Recorder:
    """Stands in for the JIT kernel: records the grid and the constexpr kwargs."""

    def __init__(self):
        self.grid = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid

        def _call(*args, **kwargs):
            self.kwargs = kwargs
            self.args = args
        return _call


def _invoke(mod, rec, **over):
    mod.kernel_transposed_grouped_gemm = rec
    e, m, n, k = 4, 64, 512, 512   # large enough that doubling BN/BK halves the tile counts
    grad_out = torch.zeros(m, n)
    orig_in = torch.zeros(m, k)
    counts = torch.full((e,), m // e, dtype=torch.int32)
    cum = torch.zeros(e + 1, dtype=torch.int32)
    cum[1:] = torch.cumsum(counts, 0)
    mod.transposed_grouped_gemm_triton(grad_out, orig_in, counts, cum, **over)


def test_defaults_are_the_shipped_constants(monkeypatch):
    """P1 — an unchanged call site is byte-for-byte the old behaviour."""
    mod, common = _load_module(monkeypatch)
    rec = _Recorder()
    _invoke(mod, rec)
    assert rec.kwargs["BLOCK_M"] == common.WGRAD_BLOCK_M
    assert rec.kwargs["BLOCK_N"] == common.WGRAD_BLOCK_N
    assert rec.kwargs["BLOCK_K"] == common.WGRAD_BLOCK_K
    assert rec.grid == (common.ncore(), 1, 1)


def test_overrides_reach_the_kernel(monkeypatch):
    """P2 — the arguments are live. Without this, P1 passes on a no-op implementation."""
    mod, _ = _load_module(monkeypatch)
    rec = _Recorder()
    _invoke(mod, rec, block_m=512, block_n=256, block_k=128, grid=32)
    assert rec.kwargs["BLOCK_M"] == 512
    assert rec.kwargs["BLOCK_N"] == 256
    assert rec.kwargs["BLOCK_K"] == 128
    assert rec.grid == (32, 1, 1)


def test_override_also_changes_the_derived_tile_counts(monkeypatch):
    """The block sizes are not only forwarded — `num_tiles_n/k` are computed from them.
    A version that forwarded the constexprs while still deriving tile counts from the
    module constants would launch a grid that does not cover the output, and both
    assertions above would still pass."""
    mod, common = _load_module(monkeypatch)
    rec_default = _Recorder()
    _invoke(mod, rec_default, )
    n_default = rec_default.args[8], rec_default.args[9]   # num_tiles_n, num_tiles_k

    rec_wide = _Recorder()
    _invoke(mod, rec_wide, block_n=common.WGRAD_BLOCK_N * 2,
            block_k=common.WGRAD_BLOCK_K * 2)
    n_wide = rec_wide.args[8], rec_wide.args[9]

    assert n_wide != n_default, (
        "doubling the block sizes left the tile counts unchanged — the override is "
        "forwarded to the kernel but not used to size the launch")
