# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`ncore()` must not cap the device's AICore count at 24.

The removed line was `assert n <= 24`. No Ascend950 part in service can satisfy it —
CANN 9.1.0's platform table lists 28, 32 and 36 core counts, and the only <= 24 entry
(950PR_950z = 4) is not deployed. Measured on Ascend950DT_9582: get_aicore_num() = 32.

Why it needed a test rather than just a deletion: the failure was SILENT. `ncore()` is
called only from the backward kernels, so every backward shape raised, `run_benchmark`
caught it per shape and printed `[skip]`, and the benchmark still exited 0 with
`"configs": []`. A green run that measured nothing looks exactly like a green run that
measured everything, so a regression here would not announce itself.

These are host-only: `NPUUtils` is stubbed, so nothing here needs a device.
"""
import importlib.util
import contextlib
import sys
import types
from pathlib import Path

import pytest


@contextlib.contextmanager
def _load_common(aicore_num):
    """Load kernels/common.py with NPUUtils().get_aicore_num() returning `aicore_num`.

    The module is loaded from its own path rather than imported through the package,
    because `mega_moe.kernels.__init__` pulls in the distributed kernels (triton_dist),
    which a host-only runner does not have and which are irrelevant to this contract.

    `torch_npu` and the ascend triton backend are stubbed, so no device is required.
    Every stub is removed again in `finally`, so the modules a later test imports are
    the real ones -- a leaked stub would make an unrelated test pass for the wrong
    reason.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "mega_moe" / "kernels"

    class _NPUUtils:
        def get_aicore_num(self):
            return aicore_num

    names = ("torch_npu", "triton.backends.ascend", "triton.backends.ascend.driver")
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
    ascend = types.ModuleType("triton.backends.ascend")
    driver = types.ModuleType("triton.backends.ascend.driver")
    driver.NPUUtils = _NPUUtils
    ascend.driver = driver
    sys.modules["triton.backends.ascend"] = ascend
    sys.modules["triton.backends.ascend.driver"] = driver
    try:
        spec = importlib.util.spec_from_file_location("_common_uut", root / "common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_common_uut"] = mod
        spec.loader.exec_module(mod)
        # ``ncore`` now resolves the driver only at the authorized device
        # boundary, so retain the test-only driver through the actual call.
        yield mod
    finally:
        sys.modules.pop("_common_uut", None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.parametrize("n", [28, 32, 36])
def test_real_ascend950_core_counts_are_accepted(n):
    """The three counts that actually ship. 32 is what this fleet reports.

    Before the change every one of these raised, and the benchmark reported success
    while measuring nothing.
    """
    with _load_common(n) as common:
        assert common.ncore() == n


def test_the_old_24_bound_is_gone():
    """Pins the defect itself: 25 is the first value the old assert rejected.

    Asserting only on 32 would pass again if someone reintroduced a cap at, say, 64 --
    this asserts on the boundary that was actually wrong.
    """
    with _load_common(25) as common:
        assert common.ncore() == 25


@pytest.mark.parametrize("bad", [0, -1, True, 24.0, "24", None])
def test_non_positive_or_non_int_still_rejected(bad):
    """Removing the cap must not remove the type/range check.

    `True` is in this list deliberately: `isinstance(True, int)` is True in Python, so a
    bool would slip through an isinstance-based check and reach a launch grid as 1.
    """
    with _load_common(1) as common:
        with pytest.raises(RuntimeError, match="positive integer"):
            common.validate_physical_aicore_count(bad)
