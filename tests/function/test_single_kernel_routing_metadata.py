# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Execute production routing helpers with host tensor semantics, without Triton."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = (Path(__file__).resolve().parents[2]
          / "src/mega_moe/kernels/fused_forward.py")


class Tensor(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype).view(Tensor)


def tensor(value):
    return np.asarray(value).view(Tensor)


class Pointer:
    def __init__(self, values, offsets=0):
        self.values = values
        self.offsets = np.asarray(offsets)
        self.writes = []

    def __add__(self, offsets):
        result = Pointer(self.values, self.offsets + offsets)
        result.writes = self.writes
        return result


def buffer(size, fill=0):
    return Pointer(np.full(size, fill, dtype=np.int64))


class Language:
    constexpr = int
    int32 = np.int32
    int64 = np.int64

    def __init__(self):
        self.lane = 0

    @staticmethod
    def arange(start, end):
        return tensor(np.arange(start, end))

    @staticmethod
    def zeros(shape, dtype):
        return tensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def cdiv(left, right):
        return tensor((left + right - 1) // right)

    @staticmethod
    def minimum(left, right):
        return tensor(np.minimum(left, right))

    @staticmethod
    def maximum(left, right):
        return tensor(np.maximum(left, right))

    @staticmethod
    def sum(value, axis):
        return tensor(np.sum(value, axis=axis))

    @staticmethod
    def max(value, axis):
        return tensor(np.max(value, axis=axis))

    @staticmethod
    def cumsum(value, axis):
        return tensor(np.cumsum(value, axis=axis))

    @staticmethod
    def where(condition, left, right):
        return tensor(np.where(condition, left, right))

    @staticmethod
    def histogram(values, bins):
        assert np.all((values >= 0) & (values < bins))
        return tensor(np.bincount(values, minlength=bins).astype(np.int32))

    @staticmethod
    def gather(values, indices, axis):
        assert values.ndim == indices.ndim
        assert all(a == b for i, (a, b) in enumerate(zip(values.shape, indices.shape))
                   if i != axis)
        assert np.all((indices >= 0) & (indices < values.shape[axis]))
        return tensor(np.take_along_axis(values, indices, axis=axis))

    @staticmethod
    def load(pointer, mask=True, other=0):
        assert np.all((pointer.offsets >= 0) & (pointer.offsets < pointer.values.size))
        return tensor(np.where(mask, pointer.values[pointer.offsets], other))

    def store(self, pointer, values, mask=True):
        assert np.all((pointer.offsets >= 0) & (pointer.offsets < pointer.values.size))
        offsets, values, mask = np.broadcast_arrays(pointer.offsets, values, mask)
        active = offsets[mask.astype(bool)].reshape(-1)
        assert np.unique(active).size == active.size, "conflicting stores in one instruction"
        pointer.values[active] = values[mask.astype(bool)].reshape(-1)
        pointer.writes.extend((self.lane, int(offset)) for offset in active)


def power2(value):
    return 1 << (int(value) - 1).bit_length()


def production_helpers(extra_names=(), extra_globals=None):
    tree = ast.parse(SOURCE.read_text())
    names = {
        "_zero_pipeline_counters", "_count_routes_by_core",
        "_build_destination_metadata", "_convert_counts_to_stable_cursors",
        "_build_pull_destination_starts", "_build_recv_segment_starts",
        "_scatter_routes_by_ordinal", "_scatter_stable_routes",
    } | set(extra_names)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    for node in functions:
        node.decorator_list = []
    tl = Language()
    scope = {"tl": tl, "sub_vec_id": lambda: tl.lane,
             "triton": SimpleNamespace(next_power_of_2=power2,
                                       cdiv=lambda x, y: (x + y - 1) // y)}
    scope.update(extra_globals or {})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(**scope)


@pytest.mark.parametrize("world,epr", [(1, 1), (3, 5), (8, 1), (8, 4), (8, 112), (128, 7), (128, 14)])
@pytest.mark.parametrize("local_choice", ["first", "middle", "last"])
def test_destination_tables(world, epr, local_choice):
    h = production_helpers()
    local = {"first": 0, "middle": world // 2, "last": world - 1}[local_choice]
    bins = power2(world * epr)
    rng = np.random.default_rng(912 + world + epr)
    counts = rng.integers(0, 600, size=(world, bins))
    counts[:, world * epr:] = 9999
    counts[:, ::5] = 0
    count_ptr = Pointer(counts.reshape(-1))
    starts, dst_starts = buffer(world * epr, -9), buffer(world * epr, -9)
    recv, totals = buffer(world * epr, -9), buffer(epr, -9)
    offs, stats = buffer(epr + 1), buffer(world + 2, -9)
    waves = buffer(world * (epr + 1) * 2)
    for dst in range(world):
        h._build_destination_metadata(dst, count_ptr, starts, dst_starts, recv,
                                      totals, offs, stats, waves, local, world, epr, bins, 256)
        block = counts[:, dst * epr:(dst + 1) * epr]
        sums = block.sum(0)
        expert_starts = np.cumsum(sums) - sums
        np.testing.assert_array_equal(dst_starts.values[dst * epr:(dst + 1) * epr],
                                      expert_starts + block[:local].sum(0))
        expected_wave = np.stack((np.r_[0, np.cumsum((sums + 255) // 256)],
                                  np.r_[0, np.cumsum(sums)]), axis=1)
        np.testing.assert_array_equal(waves.values.reshape(world, epr + 1, 2)[dst], expected_wave)
        assert stats.values[2 + dst] == sums.sum()
    local_counts = counts[local, :world * epr]
    np.testing.assert_array_equal(starts.values, np.cumsum(local_counts) - local_counts)
    local_recv = counts[:, local * epr:(local + 1) * epr]
    np.testing.assert_array_equal(recv.values.reshape(world, epr), local_recv)
    np.testing.assert_array_equal(totals.values, local_recv.sum(0))
    # The EPR sentinel is dynamic (this rank's total received rows) and must
    # land for every EPR, including EPR==1 — the saved contract reads it as
    # offsets[-1] == total_recv (upstream 595b5b9).  Only the invariant zero
    # base at index 0 stays workspace-initialized at EPR==1.
    np.testing.assert_array_equal(offs.values, np.r_[0, np.cumsum(local_recv.sum(0))])
    assert stats.values[0] == local_recv.sum()
    pull = buffer(world * epr, -9)
    for source in range(world):
        h._build_pull_destination_starts(source, count_ptr, pull, local, world,
                                        epr, world * epr, bins)
        expected = np.cumsum(counts[source]) - counts[source]
        np.testing.assert_array_equal(pull.values.reshape(epr, world)[:, source],
                                      expected[local * epr:(local + 1) * epr])
    segments = buffer(epr * (world + 1), -9)
    for core in range(32):
        h._build_recv_segment_starts(core, recv, segments, 32, epr, world, power2(world))
    np.testing.assert_array_equal(segments.values.reshape(epr, world + 1),
                                  np.column_stack((np.zeros(epr), np.cumsum(local_recv.T, axis=1))))
    if epr == 1:
        # Index 0 is the invariant zero base and index 1 the dynamic sentinel:
        # the base must never be rewritten from device (native putmem scratch
        # can alias a constant zero kept live in UB), the sentinel must be.
        assert not any(offset == 0 for _, offset in offs.writes)
        assert any(offset == 1 for _, offset in offs.writes)
        assert not any(offset % 4 < 2 for _, offset in waves.writes)


@pytest.mark.parametrize("experts,cores", [(1, 1), (8, 3), (32, 32), (33, 5), (128, 32), (896, 32)])
@pytest.mark.parametrize("moonep", [False, True])
def test_core_cursor_prefix(experts, cores, moonep):
    h = production_helpers()
    bins = power2(experts * (2 if moonep else 1))
    rng = np.random.default_rng(experts)
    raw = rng.integers(0, 80, size=(cores, bins))
    raw[:, experts:] = 0
    raw[-1] = 0
    cursors = Pointer(raw.copy().reshape(-1))
    totals = raw[:, :experts].sum(0)
    starts = Pointer(np.cumsum(totals) - totals)
    for core in range(cores):
        h._convert_counts_to_stable_cursors(core, cursors, starts, cores, experts, bins, moonep)
    expected = np.cumsum(raw, axis=0) - raw
    if not moonep:
        expected[:, :experts] += starts.values
    np.testing.assert_array_equal(cursors.values.reshape(cores, bins), expected)
    assert len(cursors.writes) == cores * experts
    assert len({offset for _, offset in cursors.writes}) == cores * experts


@pytest.mark.parametrize("count,cores", [(1, 32), (257, 3), (1944, 32), (5000, 24)])
def test_counter_reset_keeps_padding(count, cores):
    h = production_helpers()
    storage = buffer(count * 16, 37)
    for core in range(cores):
        h._zero_pipeline_counters(core, storage, cores, count)
    expected = np.full(count * 16, 37)
    expected[::16] = 0
    np.testing.assert_array_equal(storage.values, expected)
    assert len(storage.writes) == count


def reset_route_inverse(inverse, routes):
    """Execute the production host reset statement on shared CPU storage."""
    class Fillable:
        def __init__(self, values, selection=slice(None)):
            self.values = values
            self.selection = selection

        def __getitem__(self, selection):
            return Fillable(self.values, selection)

        def fill_(self, value):
            self.values[self.selection].fill(value)

    path = SOURCE.parents[3] / "src/mega_moe/ops/forward.py"
    tree = ast.parse(path.read_text())
    fills = [node for node in ast.walk(tree)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Attribute)
             and node.value.func.attr == "fill_"
             and "_route_to_send" in ast.unparse(node.value.func.value)]
    assert len(fills) == 1
    scope = {"self": SimpleNamespace(_route_to_send=Fillable(inverse.values)),
             "num_routes": routes}
    exec(compile(ast.Module(body=fills, type_ignores=[]), str(path), "exec"), scope)


@pytest.mark.parametrize("routes", [0, 1, 3, 255, 256, 257, 2051, 65536])
def test_route_reset_covers_active_prefix_only(routes):
    inverse = buffer(routes + 19, 99)
    reset_route_inverse(inverse, routes)
    np.testing.assert_array_equal(inverse.values[:routes], -1)
    np.testing.assert_array_equal(inverse.values[routes:], 99)


def test_route_reset_is_separate_from_histogram():
    tree = ast.parse(SOURCE.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    count = functions["_count_routes_by_core"]
    assert "route_to_send_ptr" not in {arg.arg for arg in count.args.args}
    stores = [node for node in ast.walk(count)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "store"]
    assert len(stores) == 1
    assert "core_bucket_cursor_ptr" in ast.unparse(stores[0].args[0])

    kernel = functions["_kernel_fused_forward"]
    reset_calls = [node for node in ast.walk(kernel)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "_reset_route_to_send"]
    host = ast.parse((SOURCE.parents[3] / "src/mega_moe/ops/forward.py").read_text())
    launch = next(node for node in ast.walk(host)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "_kernel_fused_forward")
    fills = [node for node in ast.walk(host)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "fill_"
             and "_route_to_send" in ast.unparse(node.func.value)
             and ast.unparse(node.args[0]) == "-1"]
    # The device-verified reset has one owner: the launch stream on host.
    assert "_reset_route_to_send" not in functions
    assert not reset_calls
    assert len(fills) == 1
    assert fills[0].lineno < launch.lineno


def test_metadata_does_not_repeat_wave_reductions():
    tree = ast.parse(SOURCE.read_text())
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "_build_dynamic_wave_offsets" not in names
    assert "_zero_route_workspaces" not in names
