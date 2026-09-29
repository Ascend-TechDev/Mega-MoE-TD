# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoonEP route-chunk scatter: the single-pass form against the windowed oracle.

``_single_moonep_scatter_ordinal`` (E >= 128) and ``_single_moonep_scatter_dense``
(the historical window sweep, frozen as the oracle) must place the same route
in the same send row on the same tables.  The tables here are synthesized to
the planner's contracts:

* ``alloc_cumsum[e]`` is the inclusive destination prefix of a partition of
  expert ``e``'s global ordinal space, with ``row[R-1]`` equal to the expert's
  total;
* every split expert hands its remainder to exactly one neighbor rank, so no
  destination's remote-expert set exceeds the EPR replica budget;
* ``inverse[d*E+e]`` numbers destination ``d``'s remote experts with distinct
  slots in ``[0, EPN)``;
* each source's cursors are the exclusive prefix over cores of its own
  per-core expert counts, and its ``source_prefix`` adds the counts of every
  lower-numbered source.

The shapes keep ``routes / cores`` a multiple of the dense path's 128-lane
block: the shim bounds-checks the addresses of masked lanes too, and only a
whole number of blocks lets that reference form stay inside the route chunk.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.function.test_single_kernel_routing_metadata import (
    Language, Pointer, power2, tensor,
)

SOURCE = Path(__file__).resolve().parents[2] / "src/mega_moe/kernels/fused_moonep.py"
HELPERS = (
    "_single_moonep_scatter", "_single_moonep_scatter_dense",
    "_single_moonep_scatter_ordinal",
)
DENSE_BLOCK = 128


class MoonEPLanguage(Language):
    """Base shim plus the ``tl.full`` the destination search needs."""

    @staticmethod
    def full(shape, value, dtype):
        return tensor(np.full(shape, value, dtype=dtype))


class ReadingLanguage(MoonEPLanguage):
    """Counts the lanes whose addresses are formed against one pointer."""

    def __init__(self):
        super().__init__()
        self.watched = None
        self.lanes = 0

    def load(self, pointer, mask=True, other=0):
        if self.watched is not None and pointer.values is self.watched:
            self.lanes += int(np.size(pointer.offsets))
        return MoonEPLanguage.load(pointer, mask=mask, other=other)


def moonep_helpers(language=None):
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    assert {node.name for node in functions} == set(HELPERS)
    for node in functions:
        node.decorator_list = []
    tl = language if language is not None else MoonEPLanguage()
    scope = {
        "tl": tl,
        "triton": SimpleNamespace(next_power_of_2=power2,
                                  cdiv=lambda x, y: (x + y - 1) // y),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"),
         scope)
    return SimpleNamespace(**scope)


def make_routes(rng, routes, experts, distribution):
    if distribution == "uniform":
        selected = rng.integers(0, experts, routes)
    elif distribution == "hot":
        selected = np.full(routes, experts - 1)
        selected[::5] = 0
    else:
        selected = np.minimum(
            (rng.integers(0, experts, routes) ** 2) // max(experts, 1),
            experts - 1)
    selected[::7] = -1                     # dropped slot
    selected[::11] = experts               # out-of-range slot
    return selected.astype(np.int64)


def build_case(rng, world, experts, epr, cores, routes, distribution,
               split_experts, drop_all=False):
    """Synthesize the planner tables plus one route chunk per source."""
    selected = [make_routes(rng, routes, experts, distribution)
                for _ in range(world)]
    if drop_all:
        selected[-1][:] = -1

    counts = np.zeros((world, experts), dtype=np.int64)
    for source in range(world):
        chunk = selected[source]
        ok = (chunk >= 0) & (chunk < experts)
        counts[source] = np.bincount(chunk[ok], minlength=experts)[:experts]
    totals = counts.sum(0)

    alloc = np.zeros((world, experts), dtype=np.int64)
    for expert in range(experts):
        total = int(totals[expert])
        if total == 0:
            continue
        home = expert // epr
        if split_experts and total > 1 and rng.random() < 0.75:
            split = int(rng.integers(1, total))
            alloc[home, expert] += split
            alloc[(home + 1) % world, expert] += total - split
        else:
            alloc[home, expert] += total
    # The device table is expert-major [E, R] (`ptr + expert * R + destination`),
    # matching _kernel_moonep_alloc_cumsum's transpose of the destination prefix.
    alloc_cumsum = np.cumsum(alloc, axis=0).T
    assert np.array_equal(alloc_cumsum[:, world - 1], totals)

    inverse = np.full((world, experts), -1, dtype=np.int64)
    for destination in range(world):
        slot = 0
        for expert in range(experts):
            if alloc[destination, expert] and expert // epr != destination:
                inverse[destination, expert] = slot
                slot += 1
        assert slot <= epr, f"destination {destination} needs {slot} replicas"

    cursor_stride = max(2 * power2(experts), power2(experts))
    cursors = np.zeros((world, cores, cursor_stride), dtype=np.int64)
    source_prefix = np.zeros((world, experts), dtype=np.int64)
    send_starts = np.zeros((world, 2 * world * epr), dtype=np.int64)
    expected_rows = np.full((world, routes), -1, dtype=np.int64)
    offsets_by_bucket = {}
    per_core = (routes + cores - 1) // cores

    for source in range(world):
        if source:
            source_prefix[source] = counts[:source].sum(0)
        chunk = selected[source]
        bucket_count = np.zeros(2 * world * epr, dtype=np.int64)
        # Independent expectation pass: the ordinal definition the dense
        # window sweep computes, evaluated route by route.
        per_expert_pos = np.zeros(experts, dtype=np.int64)
        for route in range(routes):
            expert = int(chunk[route])
            if not 0 <= expert < experts:
                continue
            ordinal = int(per_expert_pos[expert])
            per_expert_pos[expert] += 1
            global_ordinal = int(source_prefix[source, expert]) + ordinal
            destination = min(
                int(np.searchsorted(alloc_cumsum[expert], global_ordinal,
                                    side="right")), world - 1)
            allocation_lo = (0 if destination == 0
                             else int(alloc_cumsum[expert, destination - 1]))
            slot = (expert % epr if destination == expert // epr
                    else epr + int(inverse[destination, expert]))
            bucket = destination * (2 * epr) + slot
            offset = global_ordinal - max(int(source_prefix[source, expert]),
                                          allocation_lo)
            offsets_by_bucket.setdefault((source, bucket), []).append(offset)
            bucket_count[bucket] += 1
        running = 0
        for bucket in range(2 * world * epr):
            send_starts[source, bucket] = running
            running += bucket_count[bucket]
        for (src, bucket), offsets in offsets_by_bucket.items():
            if src == source:
                assert sorted(offsets) == list(range(len(offsets)))
        # Cursors are the exclusive prefix along the CORE axis, per expert
        # (_convert_counts_to_stable_cursors); MoonEP adds no send base.
        per_core_counts = np.zeros((cores, experts), dtype=np.int64)
        for core in range(cores):
            begin = core * per_core
            end = min(begin + per_core, routes)
            part = chunk[begin:end]
            part = part[(part >= 0) & (part < experts)]
            per_core_counts[core] = np.bincount(part, minlength=experts)[:experts]
        cursors[source, :, :experts] = (
            np.cumsum(per_core_counts, axis=0) - per_core_counts)
        per_expert_pos[:] = 0
        for route in range(routes):
            expert = int(chunk[route])
            if not 0 <= expert < experts:
                continue
            ordinal = int(per_expert_pos[expert])
            per_expert_pos[expert] += 1
            global_ordinal = int(source_prefix[source, expert]) + ordinal
            destination = min(
                int(np.searchsorted(alloc_cumsum[expert], global_ordinal,
                                    side="right")), world - 1)
            allocation_lo = (0 if destination == 0
                             else int(alloc_cumsum[expert, destination - 1]))
            slot = (expert % epr if destination == expert // epr
                    else epr + int(inverse[destination, expert]))
            bucket = destination * (2 * epr) + slot
            expected_rows[source, route] = (
                int(send_starts[source, bucket]) + global_ordinal
                - max(int(source_prefix[source, expert]), allocation_lo))

    return {
        "selected": selected,
        "counts": counts,
        "alloc": alloc,
        "alloc_cumsum": alloc_cumsum,
        "inverse": inverse,
        "cursors": cursors,
        "source_prefix": source_prefix,
        "send_starts": send_starts,
        "expected_rows": expected_rows,
        "offsets_by_bucket": offsets_by_bucket,
    }


def run_form(helpers, form, source, core, case, world, experts, epr, cores,
             topk):
    routes = len(case["selected"][source])
    counters = {
        "tokens": Pointer(np.full(routes, -7, dtype=np.int64)),
        "routes": Pointer(np.full(routes, -7, dtype=np.int64)),
        "inverse": Pointer(np.full(routes, -7, dtype=np.int64)),
    }
    cursors = Pointer(case["cursors"][source].reshape(-1).copy())
    target = getattr(helpers, form)
    kwargs = dict(
        pid=core,
        selected_ptr=Pointer(case["selected"][source]),
        cursors_ptr=cursors,
        source_prefix_ptr=Pointer(case["source_prefix"][source]),
        alloc_cumsum_ptr=Pointer(case["alloc_cumsum"].reshape(-1)),
        inverse_ptr=Pointer(case["inverse"].reshape(-1)),
        send_starts_ptr=Pointer(case["send_starts"][source]),
        send_tokens_ptr=counters["tokens"],
        send_routes_ptr=counters["routes"],
        route_to_send_ptr=counters["inverse"],
        num_routes=routes,
        NUM_CORES=cores,
        R=world,
        E=experts,
        EPN=epr,
        CURSOR_STRIDE=case["cursors"].shape[2],
        TOPK=topk,
        SEARCH_STEPS=max(1, world.bit_length()),
        BLOCK=DENSE_BLOCK,
    )
    # The single-pass form carries no route-block argument of its own.
    target(**{name: value for name, value in kwargs.items()
              if name in inspect.signature(target).parameters})
    return cursors, counters


@pytest.mark.parametrize("world,experts,epr,cores,routes", [
    (2, 128, 64, 2, 256),
    (4, 256, 64, 4, 512),
    (8, 896, 112, 4, 1024),
    (8, 32, 4, 2, 256),
])
@pytest.mark.parametrize("distribution", ["uniform", "hot", "skewed"])
@pytest.mark.parametrize("split_experts", [True, False])
def test_ordinal_matches_dense(world, experts, epr, cores, routes,
                               distribution, split_experts):
    rng = np.random.default_rng(
        abs(hash((world, experts, epr, cores, routes, distribution,
                  split_experts))) % (2 ** 31))
    case = build_case(rng, world, experts, epr, cores, routes, distribution,
                      split_experts)
    helpers = moonep_helpers()
    per_core = (routes + cores - 1) // cores
    for source in range(world):
        for core in range(cores):
            if core * per_core >= routes:
                continue
            dense = run_form(helpers, "_single_moonep_scatter_dense", source,
                             core, case, world, experts, epr, cores, 2)
            ordinal = run_form(helpers, "_single_moonep_scatter_ordinal",
                               source, core, case, world, experts, epr, cores,
                               2)
            for name in ("tokens", "routes", "inverse"):
                np.testing.assert_array_equal(
                    ordinal[1][name].values, dense[1][name].values,
                    err_msg=f"{name} differs at source {source} core {core}")
            np.testing.assert_array_equal(
                ordinal[0].values, dense[0].values,
                err_msg=f"cursors differ at source {source} core {core}")


@pytest.mark.parametrize("world,experts,epr,cores,routes", [
    (4, 256, 64, 4, 512),
    (8, 896, 112, 4, 1024),
])
@pytest.mark.parametrize("drop_all", [False, True])
def test_rows_tile_the_send_space(world, experts, epr, cores, routes, drop_all):
    rng = np.random.default_rng(4242 + world)
    case = build_case(rng, world, experts, epr, cores, routes, "uniform", True,
                      drop_all=drop_all)
    helpers = moonep_helpers()
    topk = 4
    per_core = (routes + cores - 1) // cores
    for source in range(world):
        chunk = case["selected"][source]
        expected = case["expected_rows"][source]
        for core in range(cores):
            if core * per_core >= routes:
                continue
            _, counters = run_form(helpers, "_single_moonep_scatter_ordinal",
                                   source, core, case, world, experts, epr,
                                   cores, topk)
            begin = core * per_core
            end = min(begin + per_core, routes)
            for route in range(begin, end):
                if not 0 <= int(chunk[route]) < experts:
                    continue
                row = int(expected[route])
                assert int(counters["inverse"].values[route]) == row
                assert int(counters["routes"].values[row]) == route
                assert int(counters["tokens"].values[row]) == route // topk
        rows = sorted(int(row) for row in expected if row >= 0)
        assert rows == list(range(len(rows))), (
            f"source {source}: rows are not the exact prefix of the send space")


@pytest.mark.parametrize("world,experts,epr,cores,routes", [
    (4, 128, 32, 5, 517),
    (8, 896, 112, 5, 2051),
])
def test_ragged_chunks_keep_masked_addresses_in_range(world, experts, epr,
                                                      cores, routes):
    """A per-core chunk that is not a whole number of blocks.

    The masked tail lanes overshoot the chunk, so the load and the three stores
    must clamp their addresses; the shim rejects anything outside the tables.
    """
    rng = np.random.default_rng(routes)
    case = build_case(rng, world, experts, epr, cores, routes, "uniform", False)
    helpers = moonep_helpers()
    per_core = (routes + cores - 1) // cores
    assert per_core % 32, "the case must produce a partial tail block"
    for source in range(world):
        chunk = case["selected"][source]
        expected = case["expected_rows"][source]
        for core in range(cores):
            if core * per_core >= routes:
                continue
            _, counters = run_form(helpers, "_single_moonep_scatter_ordinal",
                                   source, core, case, world, experts, epr,
                                   cores, 2)
            begin = core * per_core
            end = min(begin + per_core, routes)
            for route in range(begin, end):
                if not 0 <= int(chunk[route]) < experts:
                    continue
                assert int(counters["inverse"].values[route]) == int(
                    expected[route])


def test_cursor_row_is_read_only():
    """Both forms read the per-core prefix and keep the advance in registers."""
    world, experts, epr, cores, routes = 8, 896, 112, 4, 1024
    rng = np.random.default_rng(99)
    case = build_case(rng, world, experts, epr, cores, routes, "skewed", True)
    helpers = moonep_helpers()
    per_core = (routes + cores - 1) // cores
    for form in ("_single_moonep_scatter_dense",
                 "_single_moonep_scatter_ordinal"):
        for source in (0, world - 1):
            before = case["cursors"][source].copy()
            for core in range(cores):
                if core * per_core >= routes:
                    continue
                cursor, _ = run_form(helpers, form, source, core, case, world,
                                     experts, epr, cores, 2)
                np.testing.assert_array_equal(
                    cursor.values.reshape(before.shape), before,
                    err_msg=f"{form} rewrote the cursor row")


def test_single_pass_reads_the_route_chunk_once():
    world, experts, epr, cores, routes = 8, 896, 112, 4, 1024
    rng = np.random.default_rng(7)
    case = build_case(rng, world, experts, epr, cores, routes, "uniform", True)
    per_core = (routes + cores - 1) // cores
    lanes = {}
    for form in ("_single_moonep_scatter_dense",
                 "_single_moonep_scatter_ordinal"):
        language = ReadingLanguage()
        helpers = moonep_helpers(language)
        language.watched = case["selected"][0]
        run_form(helpers, form, 0, 0, case, world, experts, epr, cores, 2)
        lanes[form] = language.lanes
    windows = (experts + 31) // 32
    blocks = (per_core + DENSE_BLOCK - 1) // DENSE_BLOCK
    assert lanes["_single_moonep_scatter_dense"] == windows * blocks * DENSE_BLOCK
    assert lanes["_single_moonep_scatter_ordinal"] == per_core
    assert (lanes["_single_moonep_scatter_dense"]
            >= lanes["_single_moonep_scatter_ordinal"] * windows)


def test_dispatcher_splits_on_the_expert_count():
    tree = ast.parse(SOURCE.read_text())
    dispatcher = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_single_moonep_scatter")
    branch = next(node for node in ast.walk(dispatcher)
                  if isinstance(node, ast.If))
    assert ast.unparse(branch.test) == "E >= 128"
    called = lambda body: next(  # noqa: E731 - local name for the branch calls
        call.func.id for statement in body for call in ast.walk(statement)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id.startswith("_single_moonep_scatter_"))
    assert called(branch.body) == "_single_moonep_scatter_ordinal"
    assert called(branch.orelse) == "_single_moonep_scatter_dense"
