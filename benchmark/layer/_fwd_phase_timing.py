# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Summarize MOE_FWD_TIMING phase stamps of the single-kernel forward.

The single-kernel forward records three SYS_CNT products when
``MOE_FWD_TIMING=1`` (see ``mega_moe/kernels/fused_forward.py``):

* a per-program stamp table ``ts[cores, FWD_TS_SLOTS]`` — one stamp per
  program at every pipeline-wide checkpoint.  Checkpoints sit right after a
  full ``_mixed_forward_barrier`` (or at kernel entry/exit), so a stamp is
  the issuing stream's clock once every engine has drained; the per-phase
  wall time is ``max(end over cores) - min(start over cores)``.
* a per-part busy-tick table ``acc[cores * 3, FWD_ACC_SLOTS]`` — issue-stream
  ticks the wave pipeline's overlap participants spend inside their regions
  (dispatch / FC2 / return / return-wait / reduce), one row per part like
  the ring.  These are *busy* measures: the pipeline overlaps the parts
  across cores, so they compare per part, not additively to a wall time —
  except the FC2 column, a vector-clock wall (the Cube engine cannot
  execute tick arithmetic, so FC2 and the FC1 cube traversal are
  bracketed from the Vector lanes).
* a per-call ring ``ring[cores * 3, FWD_RING_SLOTS, 3]`` — one
  ``[cube wall, vector activation, save]`` tick triple per FC1 group
  call, rows ``core * 3 + {0: vec lane 0, 1: vec lane 1, 2: cube}``.  The
  cube-wall column is vector-bracketed (UB-release ack -> fixpipe
  publication, lane 0's clock); the save column is the SAVE_FC1
  quantize/store block (``0`` when the forward runs without
  ``return_saved``).

This module keeps the slot tables, the ring sizing formula, and the
host-side reduction.  It is stdlib-only on purpose: the WSL host tests
exercise it without torch/triton (``tests/function/test_fwd_phase_timing.py``).
The NPU benchmark asserts the constants against the kernel module so the
two definitions cannot drift.

SYS_CNT frequency varies by part; every summary is reported in raw ticks
plus a ``us`` view calibrated by the caller (``apply_ticks_per_us``) — the
benchmark derives ``ticks_per_us`` from an NPU-event-timed run of the same
forward.
"""

import math


# ---- ts[core, slot] checkpoints (kernel entry -> exit) -----------------------
FWD_TS_SLOTS = 10
FWD_TS_NAMES = (
    "entry",
    "routing_zero_count_done",    # post barrier: workspace zero + histogram
    "counts_published",           # post barrier: count row exchange
    "moonep_plan_done",           # post MoonEP planning barriers (0 when off)
    "destination_metadata_done",  # post barrier: send/recv + wave offsets
    "cursors_done",               # post barrier: stable cursors + pull starts
    "routing_metadata_done",      # post barrier: scatter
    "pipeline_entry",
    "pipeline_done",              # dynamic waves + top-k reduce complete
    "exit",                       # post MoonEP UDMA quiet tail
)

# Wall-clock segments between stamps, named for the WORK each one contains.
# Stamps sit AFTER their barrier, so a segment holds everything from its
# begin stamp to its end stamp — the original names were off by one slot,
# misattributing the publish to "zero_count" and the scatter to
# "stable_cursors" (found 2026-09-18; JSONs from earlier runs map
# old routing_zero_count / stable_cursors to counts_publish / route_scatter).
# ``routing_metadata`` is the direct counterpart of the staged path's
# ``build_routing_plan`` stage (planner work folded into the launch);
# ``wave_pipeline`` spans dispatch / FC1 / activation / FC2 / return /
# reduce and is compared against the staged ``dispatch_fc1`` +
# ``fc2_combine`` stages together.
FWD_TS_SEGMENTS = (
    ("zero_histogram", 0, 1),        # workspace zero + per-core histogram
    ("counts_publish", 1, 2),        # pid-0 count-row reduction + putmem fanout
    ("moonep_plan", 2, 3),           # MoonEP planning chain (empty when off)
    ("destination_metadata", 3, 4),  # send/recv tables + wave offsets
    ("stable_cursors", 4, 5),        # cursor conversion + pull starts + stats
    ("route_scatter", 5, 6),         # stable route scatter
    ("pipeline_entry", 6, 7),        # capacity check fall-through
    ("routing_metadata_total", 0, 7),
    ("wave_pipeline", 7, 8),
    ("moonep_quiet_tail", 8, 9),
    ("kernel_total", 0, 9),
)

# ---- acc[core * 3 + part, column] busy-tick columns --------------------------
# One row per part (0 = vector lane 0, 1 = vector lane 1, 2 = cube) like the
# ring: each part stores its whole (8,) accumulator — zeros in the columns it
# does not own — and the reduction takes the per-column max over the three
# rows.  The Cube engine has no vector/scalar ALU: data-path arithmetic in a
# cube scope is dropped whole (2026-09-18), so every Cube-side quantity is
# bracketed from the Vector lanes instead.
#   * columns 0/1 are the LAST dispatch site's RAW entry/exit ticks (the
#     in-kernel delta pair collapsed to 0 via read merging) — lane 1's row
#     is the writer; the host subtracts per row, never across rows;
#   * column 7 is the FC2-wave wall sum (activation signal -> the return
#     worker's completion wait, both on the vector clock);
#   * return/reduce each vector lane its own column, wait_returns the
#     vector-0 lane (both lanes wait).
FWD_ACC_SLOTS = 8
FWD_ACC_NAMES = (
    "dispatch_raw_start",  # 0: last dispatch site entry tick (raw)
    "dispatch_raw_end",    # 1: last dispatch site exit tick (raw)
    "return_issue_v0",     # 2: vector lane 0: reverse-transport calls
    "return_issue_v1",     # 3: vector lane 1: reverse-transport calls
    "wait_returns",        # 4: vector: dl.wait on every wave's returns (exact)
    "reduce_issue_v0",     # 5: vector lane 0: top-k reduce rows
    "reduce_issue_v1",     # 6: vector lane 1: top-k reduce rows
    "fc2_wall",            # 7: FC2 wave walls, vector clock (signal->wait)
)

# ---- ring[core * 3 + part, call, {cube, vact, save, vact_raw}] ---------------
# vact_raw keeps the bracket's raw start tick: a collapsed read pair
# (delta 0 with valid raws) is distinguishable from a real measurement
# only with the raw value kept.
FWD_RING_COLS = 4
FWD_RING_CUBE = 0
FWD_RING_VACT = 1
FWD_RING_SAVE = 2
FWD_RING_VACT_RAW = 3

RING_PART_VEC0 = 0
RING_PART_VEC1 = 1
RING_PART_CUBE = 2


def fwd_ring_slots(max_pipeline_groups, physical_experts_per_rank):
    """Per-program FC1 group-call ring capacity.

    A program makes at most one FC1 group call per expert per wave per
    weight table (home/replica under MoonEP): ``groups * physical_experts *
    2``.  Rounded up to a power of two with a floor of 64; the kernel clamps
    the ring store, so an undersized ring only drops per-call resolution.
    """
    bound = 2 * max_pipeline_groups * physical_experts_per_rank
    return max(64, 1 << math.ceil(math.log2(bound)))


def _stats(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    median = ordered[mid] if count % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "min": ordered[0],
        "p50": median,
        "avg": sum(ordered) / count,
        "max": ordered[-1],
    }


def _check_shape(name, samples, rows, cols):
    if not samples:
        raise ValueError(f"{name} samples must be non-empty")
    for sample in samples:
        if len(sample) != rows or any(len(row) != cols for row in sample):
            raise ValueError(
                f"{name} samples must be [{rows}, {cols}], got "
                f"[{len(sample)}, {sorted({len(row) for row in sample})}]"
            )


def summarize_ts_stamps(samples):
    """Reduce per-iteration ``ts[cores, FWD_TS_SLOTS]`` ticks to segments.

    A segment's wall span per iteration is ``max(end over cores) -
    min(start over cores)``; the returned stats aggregate those spans across
    the collected iterations.
    """
    cores = len(samples[0])
    _check_shape("ts", samples, cores, FWD_TS_SLOTS)
    segments = {}
    for name, begin, end in FWD_TS_SEGMENTS:
        spans = [
            max(row[end] for row in ts) - min(row[begin] for row in ts)
            for ts in samples
        ]
        segments[name] = _stats(spans)
    per_core_total = [
        row[FWD_TS_SLOTS - 1] - row[0] for ts in samples for row in ts
    ]
    return {"cores": cores, "segments": segments,
            "per_core_total": _stats(per_core_total)}


def summarize_acc(samples):
    """Reduce per-iteration ``acc[cores*3, FWD_ACC_SLOTS]`` busy ticks.

    A regular column's value on one core is the max over its three part
    rows (the owning part's accumulator; the other two store zeros); the
    returned stats aggregate those per-core values across cores and
    iterations.

    The dispatch raw pair (columns 0/1) is special: the two ticks belong
    to the same site on the same row, so the delta is computed per row
    FIRST and only valid pairs (nonzero start, lane-1 rows) enter the
    stats — a cross-row max would subtract one site's start from
    another's exit.  ``dispatch_issue`` reports those per-row deltas; the
    two raw columns stay in the output for diagnosing read merging (equal
    raws = the pair collapsed in-kernel again).
    """
    rows = len(samples[0])
    if rows % 3:
        raise ValueError(f"acc must hold 3 rows per core, got {rows}")
    _check_shape("acc", samples, rows, FWD_ACC_SLOTS)

    def column(column_index):
        return [
            max(sample[core * 3][column_index],
                sample[core * 3 + 1][column_index],
                sample[core * 3 + 2][column_index])
            for sample in samples for core in range(rows // 3)
        ]

    def raw_pair_delta(column_start, column_end):
        deltas = [
            row[column_end] - row[column_start]
            for sample in samples
            for row in sample
            if row[column_start] > 0
        ]
        return _stats(deltas if deltas else [0.0])

    result = {
        name: _stats(column(column_index))
        for column_index, name in enumerate(FWD_ACC_NAMES)
    }
    result["dispatch_issue"] = raw_pair_delta(0, 1)
    return result


def summarize_ring(samples):
    """Reduce per-iteration ``ring[cores * 3, FWD_RING_SLOTS, 3]`` ticks.

    The cube / vec-lane-0 / vec-lane-1 rows of one core each hold that
    part's per-call ticks.  A part's busy share on a core is the sum over
    its calls; the returned stats aggregate those sums across cores and
    iterations.  ``save_wall`` takes the per-core max of the two vector
    lanes (they save in parallel — the slower lane is the wall-clock cost).
    """
    rows = len(samples[0])
    if rows % 3:
        raise ValueError(f"ring must hold 3 rows per core, got {rows}")
    for sample in samples:
        if any(len(row) != FWD_RING_COLS for call_rows in sample
               for row in call_rows):
            raise ValueError(
                f"ring calls must have {FWD_RING_COLS} columns")

    def part_sums(sample, part, column):
        return [
            sum(call[column] for call in sample[core * 3 + part])
            for core in range(rows // 3)
        ]

    def flat(part, column):
        return [value for sample in samples
                for value in part_sums(sample, part, column)]

    totals = {
        "fc1_cube_wall": _stats(flat(RING_PART_CUBE, FWD_RING_CUBE)),
        "fc1_vact_v0": _stats(flat(RING_PART_VEC0, FWD_RING_VACT)),
        "fc1_vact_v1": _stats(flat(RING_PART_VEC1, FWD_RING_VACT)),
        "fc1_save_v0": _stats(flat(RING_PART_VEC0, FWD_RING_SAVE)),
        "fc1_save_v1": _stats(flat(RING_PART_VEC1, FWD_RING_SAVE)),
        # Raw bracket starts (huge values): diagnostic only — a collapsed
        # read pair shows as delta 0 with valid raws, and the raw spread
        # shows whether the calls ran back-to-back or spread out.
        "fc1_vact_raw_v0": _stats(flat(RING_PART_VEC0, FWD_RING_VACT_RAW)),
        "fc1_vact_raw_v1": _stats(flat(RING_PART_VEC1, FWD_RING_VACT_RAW)),
    }
    save_wall = [
        max(sums0, sums1)
        for sample in samples
        for sums0, sums1 in zip(
            part_sums(sample, RING_PART_VEC0, FWD_RING_SAVE),
            part_sums(sample, RING_PART_VEC1, FWD_RING_SAVE),
        )
    ]
    totals["fc1_save_wall"] = _stats(save_wall)
    return totals


def summarize_fc2_waves(samples):
    """Reduce per-iteration ``fc2w[cores, max_waves]`` per-wave walls.

    One wall per (core, wave): that core's activation signal for the wave
    -> the completion wait shared by every core's FC2.  ``wall`` takes the
    max over cores (earliest signaler -> all done — the wave's full FC2
    span including cube-queue contention with the next wave's FC1);
    ``residual`` takes the min (latest signaler -> all done — what
    remained after the last signal).  The single headline value is
    ``wall_p50`` over waves; ``residual_p50`` and the per-wave list carry
    the rest.
    """
    cores = len(samples[0])
    wave_count = len(samples[0][0])
    walls = []
    residuals = []
    for wave in range(wave_count):
        values = [
            sample[core][wave]
            for sample in samples for core in range(cores)
            if sample[core][wave] > 0
        ]
        if values:
            walls.append(max(values))
            residuals.append(min(values))
    return {
        "waves": len(walls),
        "wall": _stats(walls if walls else [0.0]),
        "residual": _stats(residuals if residuals else [0.0]),
        "per_wave_wall": walls,
        "per_wave_residual": residuals,
    }


def annotate_us(section, ticks_per_us):
    """Add a ``us`` view to every stats block of a ``{name: stats}`` dict."""
    for block in section.values():
        block["us"] = {
            key: value / ticks_per_us
            for key, value in block.items()
            if key != "us"
        }
    return section


def apply_ticks_per_us(summary, ticks_per_us):
    """Add a ``us`` view next to every raw-tick stats block, in place."""
    annotate_us(summary["segments"], ticks_per_us)
    annotate_us({"per_core_total": summary["per_core_total"]}, ticks_per_us)
    summary["ticks_per_us"] = ticks_per_us
    return summary


def calibrate_ticks_per_us(tick_spans_us_pairs):
    """Derive SYS_CNT ticks per microsecond from (ticks, us) pairs.

    Each pair is one forward call's total tick span (kernel entry -> exit)
    against its NPU-event-measured duration; the median ratio rejects
    event-timing outliers on a busy device.
    """
    if not tick_spans_us_pairs:
        raise ValueError("calibration needs at least one (ticks, us) pair")
    ratios = [float(ticks) / float(us)
              for ticks, us in tick_spans_us_pairs if us > 0]
    if not ratios:
        raise ValueError("calibration pairs must have positive durations")
    ordered = sorted(ratios)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2)
