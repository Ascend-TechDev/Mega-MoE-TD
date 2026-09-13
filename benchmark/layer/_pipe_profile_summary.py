# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Summarize native MAC counters without substituting Cube task occupancy."""

import csv
import json
import math
import statistics
from pathlib import Path


_COLUMNS = {
    "duration_us": "Duration(us)",
    "aic_time_us": "aicore_time(us)",
    "aiv_time_us": "aiv_time(us)",
    "mac_time_us": "aic_mac_time(us)",
    "mac_ratio": "aic_mac_ratio",
    "cube_utilization_percent": "cube_utilization(%)",
    "mte2_time_us": "aic_mte2_time(us)",
    "scalar_time_us": "aic_scalar_time(us)",
}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def summarize_pipe_profiles(traces, world_size, active_iterations):
    """Keep every fused-kernel sample; missing or invalid MAC data cannot pass."""
    rank_samples = {}
    for trace in traces:
        trace = Path(trace)
        rank = int(trace.name.split("_", 1)[0].removeprefix("rank"))
        if rank in rank_samples or not 0 <= rank < world_size:
            raise ValueError(f"Unexpected or duplicate profile rank: {rank}")
        details = trace / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"
        samples = []
        with details.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("Name", "").startswith("_kernel_fused_forward"):
                    continue
                sample = {key: _number(row.get(column))
                          for key, column in _COLUMNS.items()}
                ratio = sample["mac_ratio"]
                if ratio is not None and ratio > 1:
                    sample["mac_ratio"] = None
                sample["mac_percent"] = (
                    100 * sample["mac_ratio"]
                    if sample["mac_ratio"] is not None else None)
                duration, mac_time = sample["duration_us"], sample["mac_time_us"]
                sample["mac_time_over_kernel_duration_percent"] = (
                    100 * mac_time / duration
                    if duration and mac_time is not None else None)
                samples.append(sample)
        rank_samples[rank] = samples

    values = [sample["mac_percent"] for samples in rank_samples.values()
              for sample in samples if sample["mac_percent"] is not None]
    complete = (
        set(rank_samples) == set(range(world_size))
        and all(len(samples) == active_iterations for samples in rank_samples.values())
        and len(values) == world_size * active_iterations)
    return {
        "metric": "aic_mac_ratio",
        "threshold_percent": 90.0,
        "status": ("passed" if min(values) >= 90 else "below_threshold")
                  if complete else "incomplete",
        "expected_ranks": world_size,
        "expected_samples_per_rank": active_iterations,
        "valid_mac_samples": len(values),
        "min_mac_percent": min(values) if values else None,
        "median_mac_percent": statistics.median(values) if values else None,
        "max_mac_percent": max(values) if values else None,
        "interpretation": (
            "Native MAC ratio uses AIC task cycles, not the full mixed-kernel "
            "wall time. cube_utilization is a separate task-cycle occupancy "
            "metric. Neither aggregate metric identifies the cause of idle "
            "cycles or proves the absence of communication waits."),
        "ranks": [{"rank": rank, "samples": samples}
                  for rank, samples in sorted(rank_samples.items())],
    }


def write_pipe_profile_summary(output_dir, traces, world_size, active_iterations):
    summary = summarize_pipe_profiles(traces, world_size, active_iterations)
    path = Path(output_dir) / "mac_profile_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print(f"[mac-profile] {summary['status']}: "
          f"min={summary['min_mac_percent']}% "
          f"median={summary['median_mac_percent']}%; {path}", flush=True)
    return summary
