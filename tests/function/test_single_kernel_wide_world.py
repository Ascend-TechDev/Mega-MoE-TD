# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pure-Python models of the single-kernel forward's world-size math.

These tests pin the lane/destination assignment and readiness-count formulas
used by ``kernels/fused_forward.py`` for EP worlds larger than the physical
AICore count.  The device formulas are mirrored here as plain functions so
the invariants can be checked on any host, without torch/triton/NPU:

* ``dispatch_units`` / ``return_units`` mirror the unified unit-strided
  assignment (``for unit in range(pid, TOTAL, NUM_CORES)`` with
  ``TOTAL = max(cores, world)`` / ``max(2 * cores, world)``).
* ``legacy_dispatch`` / ``legacy_return`` / ``legacy_expected`` mirror the
  original ``pid % WORLD_SIZE`` formulas that assumed ``world <= cores``.
* ``alloc_destination`` mirrors the scatter's linear destination lookup and
  ``destination_binary_search`` its planned binary-search replacement.
* ``overlapping_sources`` mirrors the planned recv-segment prefix lookup for
  ``_wait_dispatch_row_range``.

The key equivalence property: for every ``world <= cores`` (dispatch) and
``world <= 2 * cores`` (return) the unified assignment reproduces the legacy
formulas exactly, so the W=2/8 dist suites keep validating the same mapping
they always did.
"""

import itertools
import random

import pytest


def _cdiv(a, b):
    return -(-a // b)


# ---------------------------------------------------------------------------
# Unified unit-strided assignment (the new formulas).
# ---------------------------------------------------------------------------

def dispatch_total(cores, world):
    """TOTAL_D constexpr in ``_dispatch_dynamic_wave``."""
    return max(cores, world)


def dispatch_units(pid, cores, world):
    """(destination, lane) pairs handled by dispatch program ``pid``."""
    total = dispatch_total(cores, world)
    units = range(pid, total, cores)
    return [(unit % world, unit // world) for unit in units]


def dispatch_lanes(destination, cores, world):
    """Lane count for one destination (``lanes`` arg of the tile striping)."""
    return _cdiv(dispatch_total(cores, world) - destination, world)


def return_total(cores, world):
    """TOTAL_R constexpr in ``_return_dynamic_wave``."""
    return max(2 * cores, world)


def return_units(worker, cores, world):
    """(source, lane) pairs handled by return worker ``worker``."""
    total = return_total(cores, world)
    units = range(worker, total, 2 * cores)
    return [(unit % world, unit // world) for unit in units]


def return_lanes(source, cores, world):
    return _cdiv(return_total(cores, world) - source, world)


def expected_return_signals(local_rank, cores, world):
    """``expected`` in ``_wait_dynamic_wave_returns``."""
    return _cdiv(return_total(cores, world) - local_rank, world)


# ---------------------------------------------------------------------------
# Legacy formulas (valid only for world <= cores / 2 * cores).
# ---------------------------------------------------------------------------

def legacy_dispatch(pid, world):
    return (pid % world, pid // world)


def legacy_dispatch_lanes(destination, cores, world):
    return _cdiv(cores - destination, world)


def legacy_return(worker, world):
    return (worker % world, worker // world)


def legacy_expected(local_rank, cores, world):
    return _cdiv(2 * cores - local_rank, world)


# ---------------------------------------------------------------------------
# Scatter destination lookup: linear reference vs binary search.
# ---------------------------------------------------------------------------

def alloc_destination_linear(ordinal, alloc_cumsum_row, count_ranks=None):
    """Current scatter: count of ``r`` with ``ordinal >= row[r]``."""
    if count_ranks is None:
        count_ranks = len(alloc_cumsum_row)
    destination = 0
    for rank in range(count_ranks):
        if ordinal >= alloc_cumsum_row[rank]:
            destination += 1
    return destination


def destination_binary_search(ordinal, alloc_cumsum_row):
    """First index with ``row[idx] > ordinal`` on a non-decreasing row.

    Equals the linear count over all ranks whenever
    ``ordinal < row[-1]`` (the planner's invariant: the last entry is the
    expert's total route count).
    """
    lo, hi = 0, len(alloc_cumsum_row)
    while lo < hi:
        mid = (lo + hi) // 2
        if alloc_cumsum_row[mid] <= ordinal:
            lo = mid + 1
        else:
            hi = mid
    return lo


def fixed_step_search(row, key, steps, hi_bound):
    """Mirror of the device kernels' guarded fixed-step binary search.

    Searches the closed interval [0, hi_bound] for the first index whose
    ``row`` entry fails ``<= key``; converged lanes stop updating.  This
    model must stay in lockstep with ``_single_moonep_scatter`` and
    ``_wait_dispatch_row_range`` — the step count is a launch constexpr, and
    an under-stepped search silently returns an unconverged ``lo``.
    """
    lo, hi = 0, hi_bound
    for _ in range(steps):
        active = lo < hi
        mid = (lo + hi) // 2 if active else 0
        bound = row[min(mid, len(row) - 1)]
        take = active and bound <= key
        if take:
            lo = mid + 1
        elif active:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Recv-segment prefix lookup for _wait_dispatch_row_range.
# ---------------------------------------------------------------------------

def overlapping_sources(row_start, row_end, seg_start):
    """Sources whose [seg_start[s], seg_start[s+1]) rows overlap the range.

    ``seg_start`` has world+1 entries (a prefix table).  The binary-search
    formulation returns the half-open source index range; the brute-force
    formulation scans every source.
    """
    world = len(seg_start) - 1
    # First source whose segment end exceeds row_start.
    lo = 0
    while lo < world and seg_start[lo + 1] <= row_start:
        lo += 1
    # First source whose segment start reaches row_end (exclusive bound).
    hi = lo
    while hi < world and seg_start[hi] < row_end:
        hi += 1
    return list(range(lo, hi))


def overlapping_sources_bruteforce(row_start, row_end, seg_start):
    world = len(seg_start) - 1
    return [
        source
        for source in range(world)
        if seg_start[source] < row_end and seg_start[source + 1] > row_start
    ]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

_CORE_WORLD_PAIRS = [
    (32, 2), (32, 8), (28, 8), (36, 8), (32, 16), (32, 32),
    (32, 64), (32, 128), (28, 128), (36, 128), (32, 100), (32, 40),
]


@pytest.mark.parametrize("cores,world", _CORE_WORLD_PAIRS)
def test_dispatch_assignment_covers_every_destination(cores, world):
    lanes_by_destination = {destination: set() for destination in range(world)}
    for pid in range(cores):
        for destination, lane in dispatch_units(pid, cores, world):
            assert 0 <= destination < world
            lanes_by_destination[destination].add(lane)
    for destination, lanes in lanes_by_destination.items():
        expected = dispatch_lanes(destination, cores, world)
        assert len(lanes) == expected
        assert lanes == set(range(expected)), (
            f"cores={cores} world={world} destination={destination}: "
            f"lanes {sorted(lanes)} != range({expected})"
        )


@pytest.mark.parametrize("cores,world", _CORE_WORLD_PAIRS)
def test_dispatch_total_units_match(cores, world):
    total = sum(
        len(dispatch_units(pid, cores, world)) for pid in range(cores)
    )
    assert total == dispatch_total(cores, world)


@pytest.mark.parametrize("cores,world", _CORE_WORLD_PAIRS)
def test_return_assignment_covers_every_source(cores, world):
    lanes_by_source = {source: set() for source in range(world)}
    for worker in range(2 * cores):
        for source, lane in return_units(worker, cores, world):
            assert 0 <= source < world
            lanes_by_source[source].add(lane)
    for source, lanes in lanes_by_source.items():
        expected = return_lanes(source, cores, world)
        assert lanes == set(range(expected)), (
            f"cores={cores} world={world} source={source}: "
            f"lanes {sorted(lanes)} != range({expected})"
        )
        if world > 2 * cores:
            assert expected == 1


@pytest.mark.parametrize("cores,world", _CORE_WORLD_PAIRS)
def test_expected_return_signal_sum_matches_worker_count(cores, world):
    # Every return unit fires exactly one ADD on its source-rank slot, so the
    # waiter's expected value must equal the number of units assigned to the
    # local rank across all workers.
    units_for_local = 0
    for worker in range(2 * cores):
        units_for_local += sum(
            1 for source, _ in return_units(worker, cores, world)
            if source == world - 1  # any fixed local rank works; use max
        )
    assert expected_return_signals(world - 1, cores, world) == units_for_local


# Worlds at or below the physical core count must reproduce the legacy
# mapping bit-for-bit; this is what makes the W=2/8 dist suites an
# equivalence proof for the unified formulas.
@pytest.mark.parametrize(
    "cores,world",
    [pair for pair in _CORE_WORLD_PAIRS if pair[1] <= pair[0]],
)
def test_dispatch_matches_legacy_when_world_fits_cores(cores, world):
    for pid in range(cores):
        assert dispatch_units(pid, cores, world) == [
            legacy_dispatch(pid, world)
        ]
    for destination in range(world):
        assert (
            dispatch_lanes(destination, cores, world)
            == legacy_dispatch_lanes(destination, cores, world)
        )


@pytest.mark.parametrize(
    "cores,world",
    [pair for pair in _CORE_WORLD_PAIRS if pair[1] <= 2 * pair[0]],
)
def test_return_matches_legacy_when_world_fits_workers(cores, world):
    for worker in range(2 * cores):
        assert return_units(worker, cores, world) == [
            legacy_return(worker, world)
        ]
    for local_rank in range(world):
        assert (
            expected_return_signals(local_rank, cores, world)
            == legacy_expected(local_rank, cores, world)
        )


# Tile striping: each destination's source tiles are split across its lanes
# exactly once (mirrors ``for source_tile in range(begin + lane, count,
# task_cores)`` inside _dispatch_one_source_tile_task).
@pytest.mark.parametrize("cores,world", _CORE_WORLD_PAIRS)
def test_dispatch_tile_striping_is_exact_partition(cores, world):
    rng = random.Random(cores * 1000 + world)
    for _ in range(8):
        tile_count = rng.randrange(0, 5 * cores + 1)
        for destination in range(world):
            lanes = dispatch_lanes(destination, cores, world)
            covered = []
            for lane in range(lanes):
                covered.extend(range(lane, tile_count, lanes))
            assert sorted(covered) == list(range(tile_count)), (
                f"cores={cores} world={world} destination={destination} "
                f"tiles={tile_count}"
            )


@pytest.mark.parametrize("world", [8, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_destination_binary_search_matches_linear(world, seed):
    rng = random.Random(seed * 1000 + world)
    for _ in range(64):
        row = [0]
        for _ in range(world - 1):
            row.append(row[-1] + rng.randrange(0, 5))
        assert row[-1] > 0 or world == 1
        ordinal = rng.randrange(0, row[-1] + 1)
        over_all = alloc_destination_linear(ordinal, row)
        over_prefix = alloc_destination_linear(ordinal, row, world - 1)
        if ordinal < row[-1]:
            # The scatter only reaches the last rank when the ordinal equals
            # the expert total, which allocation bounds exclude.
            assert over_all == over_prefix
        assert destination_binary_search(ordinal, row) == over_all


@pytest.mark.parametrize("world", [2, 4, 8])
def test_fixed_step_search_exhaustive_small_worlds(world):
    """Exhaustive check of the guarded search and its step-count formula.

    The destination search answers live in the closed interval [0, W], so
    convergence needs W.bit_length() steps; (W-1).bit_length() leaves the
    worst case (head-zero rows with ordinal 0, e.g. row=[0,1,...]) one
    halving short and silently returns an unconverged lane.
    """
    rows = []
    for increments in itertools.product(range(3), repeat=world - 1):
        row = [0]
        for step in increments:
            row.append(row[-1] + step)
        rows.append(row)
    for row in rows:
        for ordinal in range(row[-1] + 1):
            expected = destination_binary_search(ordinal, row)
            assert (
                fixed_step_search(row, ordinal, world.bit_length(), world)
                == expected
            )


@pytest.mark.parametrize("world", [16, 100, 128, 256])
@pytest.mark.parametrize("seed", [0, 1])
def test_fixed_step_search_random_wide_worlds(world, seed):
    rng = random.Random(seed * 1000 + world)
    for _ in range(128):
        row = [0]
        for _ in range(world - 1):
            row.append(row[-1] + rng.randrange(0, 5))
        for _ in range(16):
            ordinal = rng.randrange(0, row[-1] + 1)
            expected = destination_binary_search(ordinal, row)
            assert (
                fixed_step_search(row, ordinal, world.bit_length(), world)
                == expected
            )


@pytest.mark.parametrize("world", [8, 32, 128])
@pytest.mark.parametrize("seed", [0, 1])
def test_overlapping_sources_matches_bruteforce(world, seed):
    rng = random.Random(seed * 1000 + world)
    for _ in range(64):
        seg_start = [0]
        for _ in range(world):
            seg_start.append(seg_start[-1] + rng.randrange(0, 4))
        total_rows = seg_start[-1]
        if total_rows == 0:
            continue
        for _ in range(8):
            row_start = rng.randrange(0, total_rows)
            row_end = min(total_rows, row_start + rng.randrange(1, 300))
            assert (
                overlapping_sources(row_start, row_end, seg_start)
                == overlapping_sources_bruteforce(
                    row_start, row_end, seg_start
                )
            )
