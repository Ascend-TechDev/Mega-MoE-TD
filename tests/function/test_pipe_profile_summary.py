# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for native MAC profile reporting."""

import csv

import pytest

from benchmark.layer._pipe_profile_summary import summarize_pipe_profiles


def _trace(tmp_path, rank, ratios, cube=99.0):
    trace = tmp_path / f"rank{rank}_123_ascend_pt"
    output = trace / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    with (output / "kernel_details.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "Name", "Duration(us)", "aicore_time(us)", "aiv_time(us)",
            "aic_mac_time(us)", "aic_mac_ratio", "cube_utilization(%)"])
        writer.writeheader()
        writer.writerow({"Name": "hccl_barrier", "aic_mac_ratio": 0.0})
        for ratio in ratios:
            writer.writerow({"Name": "_kernel_fused_forward_hash",
                             "Duration(us)": 120, "aicore_time(us)": 100,
                             "aiv_time(us)": 120, "aic_mac_time(us)": 91,
                             "aic_mac_ratio": ratio, "cube_utilization(%)": cube})
    return trace


def test_cube_occupancy_does_not_replace_mac_ratio(tmp_path):
    trace = _trace(tmp_path, 0, [0.68, 0.69])
    summary = summarize_pipe_profiles([trace], 1, 2)
    assert summary["status"] == "below_threshold"
    assert summary["median_mac_percent"] == pytest.approx(68.5)
    assert summary["ranks"][0]["samples"][0]["cube_utilization_percent"] == 99


def test_all_ranks_and_samples_must_reach_threshold(tmp_path):
    traces = [_trace(tmp_path, 0, [0.99, 0.99]),
              _trace(tmp_path, 1, [0.899, 0.99])]
    summary = summarize_pipe_profiles(traces, 2, 2)
    assert summary["status"] == "below_threshold"
    assert summary["min_mac_percent"] == pytest.approx(89.9)
    assert summary["valid_mac_samples"] == 4


def test_native_ratio_and_full_duration_remain_distinct(tmp_path):
    summary = summarize_pipe_profiles([_trace(tmp_path, 0, [0.91, 0.90])], 1, 2)
    assert summary["status"] == "passed"
    sample = summary["ranks"][0]["samples"][0]
    assert sample["mac_percent"] == 91
    assert sample["mac_time_over_kernel_duration_percent"] == pytest.approx(91 / 120 * 100)


@pytest.mark.parametrize("ratio", [None, "NaN", 90])
def test_invalid_mac_samples_cannot_pass(tmp_path, ratio):
    summary = summarize_pipe_profiles([_trace(tmp_path, 0, [ratio, 0.95])], 1, 2)
    assert summary["status"] == "incomplete"
    assert summary["valid_mac_samples"] == 1


@pytest.mark.parametrize("world_size,iterations,ratios", [
    (2, 2, [0.95, 0.95]), (1, 2, [0.95]),
    (1, 2, [0.95, 0.95, 0.95]),
])
def test_incomplete_or_extra_samples_cannot_pass(tmp_path, world_size, iterations, ratios):
    summary = summarize_pipe_profiles([_trace(tmp_path, 0, ratios)], world_size, iterations)
    assert summary["status"] == "incomplete"


def test_duplicate_rank_is_rejected(tmp_path):
    trace = _trace(tmp_path, 0, [0.95, 0.95])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_pipe_profiles([trace, trace], 1, 2)
