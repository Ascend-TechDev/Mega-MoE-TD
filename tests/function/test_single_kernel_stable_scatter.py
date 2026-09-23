# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""End-to-end host execution of production histogram, cursors and scatter."""

import numpy as np
import pytest

from tests.function.test_single_kernel_routing_metadata import (
    Pointer, buffer, power2, production_helpers,
)


@pytest.mark.parametrize("experts", [1, 8, 32, 33, 64, 127, 128, 129, 896])
@pytest.mark.parametrize("routes,cores,distribution", [
    (0, 32, "random"), (3, 32, "random"), (513, 3, "random"),
    (2051, 32, "random"), (4099, 5, "hot"), (513, 3, "invalid"),
])
def test_stable_scatter_matches_sort(experts, routes, cores, distribution):
    h = production_helpers()
    bins = power2(experts)
    rng = np.random.default_rng(experts + routes)
    selected = rng.integers(-2, experts + 2, size=routes)
    if distribution == "hot":
        selected[:] = experts - 1
        selected[::11] = 0
        selected[::37] = -1
    elif distribution == "invalid":
        selected[:] = experts
    cursor = buffer(cores * bins, 99)
    inverse = buffer(routes, 99)
    selected_ptr = Pointer(selected)
    for core in range(cores):
        h._count_routes_by_core(core, selected_ptr, cursor, inverse,
                               routes, cores, experts, bins, 256)
    expected_histogram = np.bincount(selected[(selected >= 0) & (selected < experts)], minlength=bins)
    np.testing.assert_array_equal(cursor.values.reshape(cores, bins).sum(0), expected_histogram)
    np.testing.assert_array_equal(inverse.values, -np.ones(routes))
    counts = cursor.values.reshape(cores, bins).copy()
    starts = Pointer(np.cumsum(expected_histogram[:experts]) - expected_histogram[:experts])
    for core in range(cores):
        h._convert_counts_to_stable_cursors(core, cursor, starts, cores, experts, bins)
    initial_cursors = cursor.values.copy()
    inverse.writes.clear()
    tokens, send = buffer(routes, -9), buffer(routes, -9)
    workers = [(core, lane) for core in range(cores) for lane in range(2)]
    rng.shuffle(workers)
    for core, lane in workers:
        h.tl.lane = lane
        h._scatter_stable_routes(core, selected_ptr, cursor, tokens, send, inverse,
                                 routes, cores, experts, bins, 2, 128)
    expected = sorted((r for r in range(routes) if 0 <= selected[r] < experts),
                      key=lambda r: selected[r])
    size = len(expected)
    np.testing.assert_array_equal(send.values[:size], expected)
    np.testing.assert_array_equal(tokens.values[:size], np.asarray(expected, dtype=int) // 2)
    np.testing.assert_array_equal(send.values[size:], -9)
    reference_inverse = np.full(routes, -1)
    reference_inverse[expected] = np.arange(size)
    np.testing.assert_array_equal(inverse.values, reference_inverse)
    np.testing.assert_array_equal(cursor.values, initial_cursors + counts.reshape(-1))
    for pointer in (send, tokens, inverse):
        offsets = [offset for _, offset in pointer.writes]
        assert len(offsets) == size
        assert len(set(offsets)) == size, "two workers wrote the same output slot"
        lane0 = {offset for lane, offset in pointer.writes if lane == 0}
        lane1 = {offset for lane, offset in pointer.writes if lane == 1}
        assert lane0.isdisjoint(lane1)


@pytest.mark.parametrize("world,epr", [(3, 5), (8, 16), (8, 112)])
def test_destination_cursor_scatter_contract(world, epr):
    h = production_helpers()
    experts, cores, routes = world * epr, 5, 517
    bins = power2(experts)
    rng = np.random.default_rng(experts)
    selected = rng.integers(-1, experts + 1, size=(world, routes))
    counts = np.zeros((world, bins), dtype=np.int64)
    cursors = []
    inverses = []
    for rank in range(world):
        cursor, inverse = buffer(cores * bins, 99), buffer(routes, 99)
        for core in range(cores):
            h._count_routes_by_core(core, Pointer(selected[rank]), cursor, inverse,
                                   routes, cores, experts, bins, 256)
        counts[rank] = cursor.values.reshape(cores, bins).sum(0)
        cursors.append(cursor)
        inverses.append(inverse)
    count_ptr = Pointer(counts.reshape(-1))
    for rank in range(world):
        starts, dst_starts = buffer(experts, -9), buffer(experts, -9)
        recv, totals = buffer(world * epr), buffer(epr)
        offs, stats, waves = buffer(epr + 1), buffer(world + 2), buffer(world * (epr + 1) * 2)
        for dst in range(world):
            h._build_destination_metadata(dst, count_ptr, starts, dst_starts,
                                          recv, totals, offs, stats, waves,
                                          rank, world, epr, bins, 256)
        for core in range(cores):
            h._convert_counts_to_stable_cursors(core, cursors[rank], starts,
                                               cores, experts, bins)
        send, tokens = buffer(routes, -1), buffer(routes, -1)
        for lane in (1, 0):
            h.tl.lane = lane
            for core in range(cores):
                h._scatter_stable_routes(core, Pointer(selected[rank]), cursors[rank],
                                         tokens, send, inverses[rank], routes,
                                         cores, experts, bins, 2, 128)
        expected = sorted((r for r in range(routes) if 0 <= selected[rank, r] < experts),
                          key=lambda r: selected[rank, r])
        np.testing.assert_array_equal(send.values[:len(expected)], expected)
        for expert in range(experts):
            start, count = starts.values[expert], counts[rank, expert]
            np.testing.assert_array_equal(selected[rank, send.values[start:start + count]], expert)
            dst = expert // epr
            receive_base = counts[:, dst * epr:expert].sum() + counts[:rank, expert].sum()
            assert dst_starts.values[expert] == receive_base


def test_full_kimi_shape_stable_order():
    h = production_helpers()
    experts, routes, cores, bins, topk = 896, 65536, 32, 1024, 16
    rng = np.random.default_rng(20260922)
    selected = Pointer(rng.integers(0, experts, routes))
    cursor, inverse = buffer(cores * bins), buffer(routes)
    for core in range(cores):
        h._count_routes_by_core(core, selected, cursor, inverse, routes, cores, experts, bins, 256)
    totals = cursor.values.reshape(cores, bins).sum(0)[:experts]
    starts = Pointer(np.cumsum(totals) - totals)
    for core in range(cores):
        h._convert_counts_to_stable_cursors(core, cursor, starts, cores, experts, bins)
    send, tokens = buffer(routes, -1), buffer(routes, -1)
    for lane in (1, 0):
        h.tl.lane = lane
        for core in reversed(range(cores)):
            h._scatter_stable_routes(core, selected, cursor, tokens, send, inverse,
                                     routes, cores, experts, bins, topk, 128)
    expected = np.argsort(selected.values, kind="stable")
    np.testing.assert_array_equal(send.values, expected)
    np.testing.assert_array_equal(tokens.values, expected // topk)
    np.testing.assert_array_equal(inverse.values[expected], np.arange(routes))
