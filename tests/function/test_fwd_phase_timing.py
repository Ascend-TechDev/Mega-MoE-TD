# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the single-kernel forward phase-timing reduction."""

import json

import pytest

from benchmark.layer._fwd_phase_timing import (
    FWD_ACC_SLOTS,
    FWD_TS_SLOTS,
    apply_ticks_per_us,
    calibrate_ticks_per_us,
    fwd_ring_slots,
    summarize_acc,
    summarize_fc2_waves,
    summarize_ring,
    summarize_ts_stamps,
)


def _ts_matrix(spans):
    """One [2, FWD_TS_SLOTS] sample whose consecutive-stamp diffs are spans."""
    matrix = [[0] * FWD_TS_SLOTS for _ in range(2)]
    for row in matrix:
        clock = 0
        for slot in range(FWD_TS_SLOTS):
            row[slot] = clock
            if slot < len(spans):
                clock += spans[slot]
    return matrix


def test_ring_slots_are_powers_of_two_with_a_floor():
    assert fwd_ring_slots(1, 1) == 64
    assert fwd_ring_slots(5, 16) == 256
    assert fwd_ring_slots(4, 8) == 64
    slots = fwd_ring_slots(3, 5)
    assert slots >= 2 * 3 * 5
    assert slots & (slots - 1) == 0


def test_ts_segments_use_cross_core_wall_spans():
    # Barrier-aligned checkpoints are identical across cores mid-kernel;
    # only entry (staggered launch) and exit differ.  Segments must take
    # max(end over cores) - min(start over cores), so the entry stagger
    # widens zero_histogram and the exit stagger widens the tail.
    # Segment names follow the WORK each span contains — the original
    # names were off by one slot (stamps sit after their barriers).
    shifted = [[10, 100, 150, 150, 150, 250, 300, 320, 830, 840],
               [0, 100, 150, 150, 150, 250, 300, 320, 830, 850]]
    summary = summarize_ts_stamps([shifted])
    segments = summary["segments"]
    assert segments["kernel_total"]["p50"] == 850
    assert segments["zero_histogram"]["p50"] == 100
    assert segments["counts_publish"]["p50"] == 50
    assert segments["moonep_plan"]["p50"] == 0  # MOONEP-off pass-through
    assert segments["destination_metadata"]["p50"] == 0
    assert segments["stable_cursors"]["p50"] == 100
    assert segments["route_scatter"]["p50"] == 50
    assert segments["wave_pipeline"]["p50"] == 510
    assert segments["moonep_quiet_tail"]["p50"] == 20
    assert segments["routing_metadata_total"]["p50"] == 320


def test_ts_stats_aggregate_across_samples():
    samples = [_ts_matrix([10, 20, 0, 5, 5, 5, 10, 500, 5]),
               _ts_matrix([12, 22, 0, 6, 6, 6, 12, 520, 6])]
    segments = summarize_ts_stamps(samples)["segments"]
    assert segments["zero_histogram"]["min"] == 10
    assert segments["zero_histogram"]["max"] == 12
    assert segments["zero_histogram"]["p50"] == 11
    assert segments["zero_histogram"]["avg"] == pytest.approx(11)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError, match="ts samples"):
        summarize_ts_stamps([[[0, 1]]])
    with pytest.raises(ValueError, match="acc must hold 3 rows per core"):
        summarize_acc([[[0] * FWD_ACC_SLOTS, [0] * FWD_ACC_SLOTS]])
    with pytest.raises(ValueError, match="acc samples"):
        summarize_acc([[[0] * (FWD_ACC_SLOTS - 1) for _ in range(3)]])
    with pytest.raises(ValueError, match="3 rows per core"):
        summarize_ring([[[0, 0, 0]]])


def test_acc_takes_column_max_over_part_rows():
    # Two cores x three part rows.  The owning part carries the value; the
    # other two parts store zeros in that column, so the per-column max
    # over the three rows recovers it.  Columns 0/1 are the dispatch RAW
    # pair (lane 1's row) — the host subtracts per row; column 7 is the
    # FC2 wall.
    def core_rows(d0, d1, ret0, ret1, wait, red0, red1, fc2_wall):
        return [
            [0, 0, ret0, 0, wait, red0, 0, fc2_wall],  # vector lane 0
            [d0, d1, 0, ret1, 0, 0, red1, 0],          # vector lane 1
            [0, 0, 0, 0, 0, 0, 0, 0],                  # cube (never measures)
        ]

    acc = core_rows(1000, 1600, 7, 8, 9, 10, 11, 400) + core_rows(
        2000, 2600, 17, 18, 19, 20, 21, 500)
    summary = summarize_acc([acc])
    assert set(summary) == {
        "dispatch_raw_start", "dispatch_raw_end", "return_issue_v0",
        "return_issue_v1", "wait_returns", "reduce_issue_v0",
        "reduce_issue_v1", "fc2_wall", "dispatch_issue",
    }
    assert summary["dispatch_issue"]["min"] == 600
    assert summary["dispatch_issue"]["max"] == 600  # both rows delta 600
    assert summary["dispatch_raw_start"]["max"] == 2000  # raws stay visible
    assert summary["dispatch_raw_end"]["max"] == 2600
    assert summary["return_issue_v1"]["min"] == 8
    assert summary["return_issue_v1"]["max"] == 18
    assert summary["reduce_issue_v1"]["max"] == 21
    assert summary["fc2_wall"]["min"] == 400
    assert summary["fc2_wall"]["max"] == 500


def test_acc_dispatch_raw_pair_skips_invalid_rows():
    # Rows without a valid pair (start 0 — lane 0 / cube rows, or a
    # collapsed in-kernel read pair) never enter the delta stats.
    acc = [
        [0, 0, 0, 0, 0, 0, 0, 0],     # lane 0: no dispatch
        [5000, 5000, 0, 0, 0, 0, 0, 0],  # lane 1: collapsed pair (delta 0)
        [0, 0, 0, 0, 0, 0, 0, 0],     # cube
    ]
    summary = summarize_acc([acc])
    # The collapsed pair still counts (delta 0) — the raws reveal it.
    assert summary["dispatch_issue"]["p50"] == 0
    assert summary["dispatch_raw_start"]["p50"] == 5000
    assert summary["dispatch_raw_end"]["p50"] == 5000


def test_ring_sums_per_part_and_save_wall_takes_lane_max():
    # Two cores, three rows each; every call column is
    # [cube, vact, save, vact_raw].
    ring = []
    for core in range(2):
        rows = [[] for _ in range(3)]
        for _ in range(2):  # two FC1 group calls
            rows[0].append([0, 30, 10, 900])   # vec lane 0
            rows[1].append([0, 40, 25, 950])   # vec lane 1 (slower saver)
            rows[2].append([50, 0, 0, 0])      # cube
        ring.extend(rows)
    summary = summarize_ring([ring])
    assert summary["fc1_cube_wall"]["p50"] == 100   # 50 per call, two calls
    assert summary["fc1_vact_v0"]["p50"] == 60
    assert summary["fc1_vact_v1"]["p50"] == 80
    assert summary["fc1_save_v0"]["p50"] == 20
    assert summary["fc1_save_v1"]["p50"] == 50
    assert summary["fc1_save_wall"]["p50"] == 50  # max(20, 50) per core
    assert summary["fc1_vact_raw_v0"]["p50"] == 1800  # 900 per call
    assert summary["fc1_vact_raw_v1"]["p50"] == 1900


def test_fc2_waves_reports_single_value_and_per_wave_detail():
    # Two cores x four wave slots; wave 2 never ran (all zeros), waves
    # carry per-core walls [core0, core1].
    fc2w = [
        [4000, 3000, 0, 5000],   # core 0
        [5000, 2000, 0, 6000],   # core 1
    ]
    summary = summarize_fc2_waves([fc2w])
    assert summary["waves"] == 3
    # wall = max over cores, residual = min over cores.
    assert summary["per_wave_wall"] == [5000, 3000, 6000]
    assert summary["per_wave_residual"] == [4000, 2000, 5000]
    assert summary["wall"]["p50"] == 5000
    assert summary["residual"]["p50"] == 4000


def test_fc2_waves_with_no_waves_degrades_gracefully():
    summary = summarize_fc2_waves([[[0, 0], [0, 0]]])
    assert summary["waves"] == 0
    assert summary["wall"]["p50"] == 0
    assert summary["per_wave_wall"] == []


def test_apply_ticks_per_us_annotates_every_block():
    summary = summarize_ts_stamps([_ts_matrix([10] * 9)])
    apply_ticks_per_us(summary, 2.0)
    first = summary["segments"]["zero_histogram"]
    assert first["us"]["p50"] == pytest.approx(5.0)
    assert summary["per_core_total"]["us"]["p50"] == pytest.approx(45.0)
    assert summary["ticks_per_us"] == 2.0


def test_calibration_takes_the_median_ratio():
    ratio = calibrate_ticks_per_us(
        [(2000, 2.0), (3000, 3.0), (4800, 6.0)]
    )
    assert ratio == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        calibrate_ticks_per_us([])


def test_summaries_stay_json_serializable():
    # The benchmark writes phase_timing.json straight from these summaries;
    # numpy scalars (from .cpu().numpy() products) must not leak through.
    ts = [[[index * 100 + slot for slot in range(FWD_TS_SLOTS)]
           for index in range(4)]]
    acc = [[[10 * index + column for column in range(FWD_ACC_SLOTS)]
            for index in range(2 * 3)]]
    ring = [[[index * 3 + column for column in range(4)]
             for _ in range(8)]
            for index in range(4 * 3)]
    summary = summarize_ts_stamps(ts)
    payload = {
        "calibration": calibrate_ticks_per_us(
            [(row[-1] - row[0], 1.0) for row in ts[0]]),
        "segments": summary["segments"],
        "busy": summarize_acc(acc),
        "ring": summarize_ring([ring]),
    }
    assert isinstance(json.loads(json.dumps(payload)), dict)
