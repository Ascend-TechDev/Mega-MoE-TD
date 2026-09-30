# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Run the W8 Kimi token/expert/MoonEP matrix with read-only device telemetry."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import resolve_case
from benchmark.layer._npu_occupancy import parse_npu_processes


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_jobs(profile, trimmed_topk):
    if profile == "skewed" and trimmed_topk != 16:
        raise ValueError("The registered matched skewed matrix uses top-k=16")
    jobs = []
    for tokens in (4096, 8192, 16384):
        for trimmed in (True, False):
            prefix = "performance-fwd-kimi-k3"
            if trimmed:
                prefix += "-trimmed"
                if profile == "uniform" and trimmed_topk == 16:
                    prefix += "-top16"
            if profile == "skewed":
                prefix += "-skewed"
            case = resolve_case(f"{prefix}-w8-t{tokens // 1024}k").validate()
            for moonep in (False, True):
                jobs.append({
                    "point": f"{'trimmed' if trimmed else 'full'}_t{tokens // 1024}k_moonep{int(moonep)}",
                    "case": case.as_dict(), "moonep": moonep,
                    "fc1_block": [256, 256, 128] if trimmed else [128, 512, 128],
                    "fc2_block": [256, 256, 128] if trimmed else [128, 512, 128],
                    "dispatch_block": 256, "wave_windows": 32 if moonep else 16,
                })
    return jobs


def wait_for_idle(output):
    with (output / "idle_checks.jsonl").open("a") as log:
        while True:
            result = subprocess.run(["npu-smi", "info"], text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    timeout=30, check=True)
            processes = parse_npu_processes(result.stdout, range(8))
            row = {"time_utc": datetime.now(timezone.utc).isoformat(),
                   "processes": processes, "npu_smi": result.stdout}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if not processes:
                return
            print(json.dumps({"waiting_for_idle": processes}), flush=True)
            time.sleep(10)


class DeviceSampler:
    def __init__(self):
        self.lib = ctypes.CDLL("/usr/local/Ascend/driver/lib64/driver/libdcmi.so")
        self.lib.dcmiv2_init.argtypes = []
        self.lib.dcmiv2_init.restype = ctypes.c_int
        code = self.lib.dcmiv2_init()
        if code != 0:
            raise RuntimeError(f"dcmiv2_init returned {code}")
        self.lib.dcmiv2_get_device_frequency.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
        self.lib.dcmiv2_get_device_frequency.restype = ctypes.c_int
        self.lib.dcmiv2_get_device_power_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.dcmiv2_get_device_power_info.restype = ctypes.c_int

    def sample(self):
        for rank in range(8):
            frequency, power = ctypes.c_uint(), ctypes.c_int()
            begin = time.monotonic_ns()
            rf = self.lib.dcmiv2_get_device_frequency(rank, 7, ctypes.byref(frequency))
            rp = self.lib.dcmiv2_get_device_power_info(rank, ctypes.byref(power))
            yield {"rank": rank, "device": rank, "start_monotonic_ns": begin,
                   "end_monotonic_ns": time.monotonic_ns(),
                   "frequency_status": rf, "power_status": rp,
                   "frequency_mhz": frequency.value if rf == 0 else None,
                   "power_w": power.value * 0.1 if rp == 0 else None}


def run_job(job, output, sampler, interval, timeout):
    output.mkdir()
    wait_for_idle(output)
    command = [sys.executable, str(ROOT / "benchmark/layer/profile_single_kernel_forward.py"),
               "--case", job["case"]["case_id"], "--benchmark-only", "--record-host-intervals",
               "--fc1-block", *map(str, job["fc1_block"]),
               "--fc2-block", *map(str, job["fc2_block"]),
               "--dispatch-block", str(job["dispatch_block"]),
               "--wave-windows", str(job["wave_windows"]),
               "--output-dir", str(output / "benchmark")]
    if job["moonep"]:
        command.append("--moonep")
    write_json(output / "job.json", {**job, "command": command})
    started = time.monotonic()
    status = {"point": job["point"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "artifact": str(output)}
    with (output / "benchmark.log").open("w") as log, (output / "telemetry.jsonl").open("w") as telemetry:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        status["pid"] = process.pid
        write_json(output / "running.json", status)
        try:
            while process.poll() is None:
                tick = time.monotonic()
                for row in sampler.sample():
                    telemetry.write(json.dumps(row) + "\n")
                telemetry.flush()
                if tick - started > timeout:
                    status["stop_reason"] = f"Explicit per-case execution budget of {timeout} seconds exceeded"
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                time.sleep(max(0, interval - (time.monotonic() - tick)))
            status["returncode"] = process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    status.update(ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic() - started)
    write_json(output / "status.json", status)
    result_path = output / "benchmark/benchmark_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        status["metrics"] = result["metrics"]
        status["speedup"] = result["torch_over_single_kernel_median"]
        status["occupancy"] = result.get("occupancy_gate")
    print(json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routing-profile", choices=("skewed", "uniform"), default="skewed")
    parser.add_argument("--trimmed-topk", type=int, choices=(8, 16), default=16)
    parser.add_argument("--points", nargs="+")
    parser.add_argument("--sample-interval", type=float, default=0.01)
    parser.add_argument("--case-timeout", type=float, default=900)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = make_jobs(args.routing_profile, args.trimmed_topk)
    if args.points:
        missing = set(args.points) - {j["point"] for j in jobs}
        if missing:
            raise ValueError(f"Unknown points: {sorted(missing)}")
        jobs = [j for j in jobs if j["point"] in args.points]
    kernel = ROOT / "src/mega_moe/kernels/fused_forward.py"
    frozen_hash = hashlib.sha256(kernel.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "routing_profile": args.routing_profile, "trimmed_topk": args.trimmed_topk,
                "frozen_kernel_sha256": frozen_hash, "telemetry_interval_s": args.sample_interval,
                "telemetry_method": "read-only DCMI v2, frequency type 7; power unit 0.1 W",
                "planned_jobs": jobs, "completed": []}
    write_json(output / "matrix.json", manifest)
    sampler = DeviceSampler()
    for job in jobs:
        if hashlib.sha256(kernel.read_bytes()).hexdigest() != frozen_hash:
            raise RuntimeError("Fused kernel source changed during the matrix benchmark")
        print(json.dumps({"starting": job["point"], "case": job["case"]["case_id"]}), flush=True)
        status = run_job(job, output / job["point"], sampler, args.sample_interval, args.case_timeout)
        manifest["completed"].append(status)
        write_json(output / "matrix.json", manifest)
    if any(item["returncode"] != 0 for item in manifest["completed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
