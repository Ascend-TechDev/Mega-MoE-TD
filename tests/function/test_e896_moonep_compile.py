# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Opt-in regression for the full E896 MoonEP native compiler hang.

Run in the validated UDMA environment with MOE_RUN_COMPILE_TESTS=1.
These tests compile real fused kernels in fresh caches; they do not launch
the operator and do not replace the distributed numerical correctness gate.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
@pytest.mark.functional
@pytest.mark.skipif(
    os.environ.get("MOE_RUN_COMPILE_TESTS") != "1",
    reason="requires explicit UDMA compiler regression run",
)
@pytest.mark.parametrize(
    "case,save_fc1",
    [
        ("performance-fwd-kimi-k3-skewed-w8-t4k", False),
        ("performance-fwd-kimi-k3-skewed-w8-t4k", True),
        ("performance-fwd-kimi-k3-w8-t4k", False),
        ("performance-fwd-kimi-k3-skewed-w8-t8k", False),
    ],
)
def test_e896_moonep_cold_compile_finishes(tmp_path, case, save_fc1):
    result_dir = tmp_path / "result"
    command = [
        sys.executable,
        str(_ROOT / "benchmark/layer/compile_single_kernel_forward.py"),
        "--case", case, "--moonep",
        "--fc1-block", "128", "512", "128",
        "--fc2-block", "128", "512", "128",
        "--dispatch-block", "256", "--wave-windows", "32",
        "--output-dir", str(result_dir),
    ]
    if save_fc1:
        command.append("--save-fc1")
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "cache"))
    # Generous hang guard, not a compiler performance benchmark. Kill the
    # entire owned process group on failure, including bishengir children.
    timeout_s = 300 if save_fc1 else 180
    with (tmp_path / "compile.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=_ROOT, env=env, stdout=log, stderr=log,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            pytest.fail(f"E896 MoonEP compilation exceeded {timeout_s}s; see {log.name}")
    assert returncode == 0, (tmp_path / "compile.log").read_text()[-8000:]
    result = json.loads((result_dir / "compile_result.json").read_text())
    constants = result["constants"]
    assert result["binary_bytes"] > 0
    assert constants["NUM_EXPERTS"] == 896
    assert constants["WORLD_SIZE"] == 8
    assert constants["TOPK"] == 16
    assert constants["HIDDEN"] == 3584 and constants["FFN"] == 3072
    assert constants["MOONEP"] is True
    assert constants["SAVE_FC1"] is save_fc1
