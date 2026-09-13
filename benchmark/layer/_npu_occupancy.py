# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Reject benchmark runs that share their devices with unrelated processes."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def parse_npu_processes(output: str, device_ids) -> list[dict]:
    selected = set(device_ids)
    reported = set()
    processes = []
    in_process_table = False
    has_container_pid = False
    for line in output.splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if "NPU ID" in fields and "Process id" in fields:
            in_process_table = True
            has_container_pid = "Process id in container" in fields
            continue
        if not in_process_table:
            continue
        idle = re.search(r"No running processes found in NPU (\d+)", line)
        if idle:
            reported.add(int(idle.group(1)))
            continue
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        device = int(fields[0])
        if device not in selected:
            continue
        if len(fields) < 4 or not fields[1].isdigit():
            raise RuntimeError(f"Unrecognized NPU process row: {line}")
        reported.add(device)
        container_pid = None
        if has_container_pid and len(fields) >= 5 and fields[4].isdigit():
            container_pid = int(fields[4])
        processes.append({
            "device": device,
            "host_pid": int(fields[1]),
            "container_pid": container_pid,
            "name": fields[2],
        })
    missing = selected - reported
    if not in_process_table or missing:
        raise RuntimeError(f"Cannot verify NPU process ownership for devices {sorted(missing)}")
    return processes


def check_npu_occupancy(output_dir: Path, device_ids, owned_pids=(), *, phase: str,
                        known_host_pids: set[int] | None = None):
    record = {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase}
    try:
        result = subprocess.run(
            ["npu-smi", "info"], text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=30, check=True,
        )
        record["npu_smi"] = result.stdout
        processes = parse_npu_processes(result.stdout, device_ids)
        owned = set(owned_pids)
        known_hosts = known_host_pids if known_host_pids is not None else set()
        foreign = []
        for process in processes:
            container_pid, host_pid = process["container_pid"], process["host_pid"]
            if container_pid:
                is_owned = container_pid in owned
                if is_owned:
                    known_hosts.add(host_pid)
            else:
                # During teardown npu-smi can lose the container PID before
                # removing its already-verified host process entry.
                is_owned = host_pid in owned or host_pid in known_hosts
            if not is_owned:
                foreign.append(process)
        record.update(processes=processes, foreign_processes=foreign,
                      known_host_pids=sorted(known_hosts))
        if foreign:
            raise RuntimeError(f"NPU devices occupied by unrelated processes: {foreign}")
        record["status"] = "passed"
        return True
    except subprocess.TimeoutExpired as error:
        record.update(status="deferred", error=str(error))
        if phase == "running":
            # A missing observation is not evidence that a worker has stopped.
            return False
        raise
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        record.update(status="rejected", error=str(error))
        raise
    finally:
        with (Path(output_dir) / "device_occupancy.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
