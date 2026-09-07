# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the benchmark's device-ownership gate."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from benchmark.layer import _npu_occupancy as occupancy


HEADER = "| NPU ID | Process id | Process name | Process memory(MB) | Process id in container |\n"


def test_idle_devices_are_reported():
    output = HEADER + "\n".join(
        f"| No running processes found in NPU {device} |" for device in range(8))
    assert occupancy.parse_npu_processes(output, range(8)) == []


def test_multiple_processes_and_unselected_devices():
    output = HEADER + (
        "| 0 | 3451052 | python | 793 | 291279 |\n"
        "| 0 | 3407187 | python3.11 | 38644 | 4147875 |\n"
        "| 7 | 999 | python | 12 | 999 |\n")
    processes = occupancy.parse_npu_processes(output, [0])
    assert [process["container_pid"] for process in processes] == [291279, 4147875]


@pytest.mark.parametrize("output", [
    "",
    HEADER + "| 0 | unavailable | python | 793 | 291279 |\n",
])
def test_unknown_or_missing_ownership_fails_closed(output):
    with pytest.raises(RuntimeError):
        occupancy.parse_npu_processes(output, [0])


@pytest.mark.parametrize("container_header", [True, False])
def test_gate_accepts_only_own_worker(tmp_path, monkeypatch, container_header):
    header = HEADER if container_header else HEADER.replace(" Process id in container |", "")
    row = "| 0 | 3451052 | python | 793 |" + (" 291279 |" if container_header else "")
    monkeypatch.setattr(occupancy.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=header + row))
    occupancy.check_npu_occupancy(
        tmp_path, [0], [291279 if container_header else 3451052], phase="running")
    assert json.loads((tmp_path / "device_occupancy.jsonl").read_text())["status"] == "passed"


def test_gate_rejects_foreign_worker_even_when_memory_is_small(tmp_path, monkeypatch):
    output = HEADER + "| 0 | 3451052 | python | 1 | 291279 |\n"
    monkeypatch.setattr(occupancy.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(RuntimeError, match="unrelated processes"):
        occupancy.check_npu_occupancy(tmp_path, [0], [3451052], phase="running")
    report = json.loads((tmp_path / "device_occupancy.jsonl").read_text())
    assert report["status"] == "rejected"
    assert report["foreign_processes"][0]["container_pid"] == 291279


def test_running_query_timeout_defers_observation(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("npu-smi", 30)

    monkeypatch.setattr(occupancy.subprocess, "run", timeout)
    assert occupancy.check_npu_occupancy(tmp_path, [0], [123], phase="running") is False
    report = json.loads((tmp_path / "device_occupancy.jsonl").read_text())
    assert report["status"] == "deferred"
    with pytest.raises(subprocess.TimeoutExpired):
        occupancy.check_npu_occupancy(tmp_path, [0], phase="before_spawn")


def test_teardown_retains_verified_host_mapping(tmp_path, monkeypatch):
    outputs = iter([
        HEADER + "| 0 | 3451052 | python | 793 | 291279 |\n",
        HEADER + "| 0 | 3451052 | python | 793 | 0 |\n",
        HEADER + "| 0 | 3451052 | python | 793 | 999999 |\n",
        HEADER + "| 0 | 3451053 | python | 793 | 0 |\n",
    ])
    monkeypatch.setattr(occupancy.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=next(outputs)))
    known_hosts = set()
    for _ in range(2):
        assert occupancy.check_npu_occupancy(
            tmp_path, [0], [291279], phase="running", known_host_pids=known_hosts)
    assert known_hosts == {3451052}
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unrelated processes"):
            occupancy.check_npu_occupancy(
                tmp_path, [0], [291279], phase="running", known_host_pids=known_hosts)
