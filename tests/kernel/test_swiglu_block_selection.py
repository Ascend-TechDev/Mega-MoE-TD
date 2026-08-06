# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only contract for `_select_block_m` — the adaptive M-tile rule.

The rule exists because `_BLOCK_M = 8` makes the number of M-tiles grow with the row
count while the work per tile stays constant, so per-tile overhead dominates at long
sequences. It must, however, LEAVE SHORT SEQUENCES ALONE: a first version of the rule
grew the tile too early and measured 0.85x (a regression) at 1024 rows. These assertions
encode that boundary so it cannot silently move back.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "mega_moe" / "kernels" / "weighted_swiglu.py"


def _load():
    """Load without the package __init__ (it pulls in the distributed kernels)."""
    text = _SRC.read_text().replace("from .common import", "from _c import")
    tmp = Path(tempfile.mkdtemp()) / "_ws.py"
    tmp.write_text(text)
    spec = importlib.util.spec_from_file_location("_ws_sel", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ws_sel"] = mod
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("triton")
ws = _load()


@pytest.mark.parametrize("rows,cores", [(0, 24), (128, 24), (1024, 24), (4096, 24),
                                        (1024, 64), (4096, 64)])
def test_short_sequences_keep_the_original_tile(rows, cores):
    """The regression guard. Growing the tile early COSTS time — measured 0.85x at
    1024 rows with the first version of this rule. Anything at or below the crossover
    must come out exactly as before."""
    assert ws._select_block_m(rows, cores) == ws._BLOCK_M


@pytest.mark.parametrize("rows,cores", [(16384, 24), (65536, 24), (65536, 64)])
def test_long_sequences_grow_the_tile(rows, cores):
    """The other arm. Without it, a rule that never fires would pass the test above and
    deliver nothing — the two arms together are what pin the behaviour."""
    assert ws._select_block_m(rows, cores) > ws._BLOCK_M


def test_never_exceeds_the_measured_ceiling():
    for rows in (2 ** 20, 2 ** 24):
        assert ws._select_block_m(rows, 24) <= ws._MAX_BLOCK_M


def test_degenerate_inputs_do_not_crash():
    assert ws._select_block_m(0, 0) == ws._BLOCK_M
    assert ws._select_block_m(-1, 24) == ws._BLOCK_M


def test_result_is_a_power_of_two():
    """The kernel indexes with tl.arange(0, BLOCK_M); a non-power-of-two would be a
    silent correctness hazard rather than a slow path."""
    for rows in (1024, 16384, 65536, 2 ** 20):
        for cores in (1, 24, 32, 64):
            b = ws._select_block_m(rows, cores)
            assert b & (b - 1) == 0, f"{b} is not a power of two"
