# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Production task-builder and return-acquire relay checks, without an NPU."""

import ast
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from tests.function.test_single_kernel_routing_metadata import (
    Pointer, SOURCE, buffer, production_helpers, tensor,
)


TASK_HELPERS = (
    "_build_wave_tasks", "_wave_task_range", "_load_wave_task",
    "_dynamic_wave_expert_range", "_advance_fc1_tile",
)


def make_tables(counts, block, windows, max_waves):
    counts = np.asarray(counts, dtype=np.int64)
    world, epr = counts.shape
    table = np.zeros((world, epr + 1, 2), dtype=np.int64)
    table[:, 1:, 0] = np.cumsum((counts + block - 1) // block, axis=1)
    table[:, 1:, 1] = np.cumsum(counts, axis=1)
    return (Pointer(table.reshape(-1)), buffer(world * (max_waves + 1), -9),
            buffer(world * (max_waves + epr) * 5, -9))


@pytest.mark.parametrize("epr", [1, 4, 7, 14, 33, 112])
@pytest.mark.parametrize("windows", [1, 4, 16, 64])
@pytest.mark.parametrize("kind", ["random", "empty", "hot", "holes"])
def test_compact_tasks_equal_dense_wave_scan(epr, windows, kind):
    h = production_helpers(TASK_HELPERS)
    rng = np.random.default_rng(2026 + epr + windows)
    counts = rng.integers(0, 1900, (3, epr))
    if kind == "empty":
        counts[:] = 0
    elif kind == "hot":
        counts[:] = 0
        counts[1, epr // 2] = 33001
    elif kind == "holes":
        counts[:, ::2] = 0
        counts[:, -1] = 0
    capacity = max(1, int(counts.sum(1).max()))
    max_waves = (capacity + epr * 255 + windows * 256 - 1) // (windows * 256)
    table, offsets, tasks = make_tables(counts, 256, windows, max_waves)
    for rank in range(3):
        h._build_wave_tasks(rank, table, offsets, tasks, epr, max_waves, 256, windows, capacity)
        waves = (int(((counts[rank] + 255) // 256).sum()) + windows - 1) // windows
        seen = []
        for wave in range(max_waves):
            start, end = h._wave_task_range(offsets, rank, wave, max_waves)
            actual = [tuple(map(int, h._load_wave_task(tasks, rank, task, epr, max_waves)))
                      for task in range(start, end)]
            expected = []
            for expert in range(epr):
                begin, end_row, row_base, block_base = h._dynamic_wave_expert_range(
                    table, rank, expert, wave, epr, 256, windows)
                if begin < end_row:
                    expected.append(tuple(map(int, (expert, begin, end_row, row_base, block_base))))
            assert actual == expected
            seen.extend(actual)
        assert len(seen) <= np.count_nonzero(counts[rank]) + max(0, waves - 1)
        assert len(seen) <= max_waves + epr
    for pointer in (offsets, tasks):
        writes = [index for _, index in pointer.writes]
        assert len(writes) == len(set(writes))


def test_task_builder_reuse_and_capacity_guard():
    h = production_helpers(TASK_HELPERS)
    epr, max_waves, capacity = 7, 10, 900
    table, offsets, tasks = make_tables([[300, 0, 1, 255, 0, 100, 2]], 256, 4, max_waves)
    h._build_wave_tasks(0, table, offsets, tasks, epr, max_waves, 256, 4, capacity)
    table.values[:] = 0
    h._build_wave_tasks(0, table, offsets, tasks, epr, max_waves, 256, 4, capacity)
    np.testing.assert_array_equal(offsets.values, 0)
    table.values[-1] = capacity + 1
    offsets.values[:] = -11
    before = tasks.values.copy()
    h._build_wave_tasks(0, table, offsets, tasks, epr, max_waves, 256, 4, capacity)
    np.testing.assert_array_equal(offsets.values, -11)
    np.testing.assert_array_equal(tasks.values, before)


@pytest.mark.parametrize("cores", [3, 24, 32])
@pytest.mark.parametrize("parts", [1, 2, 3, 7, 16, 33, 64])
def test_fc1_incremental_tiles_keep_original_assignment(cores, parts):
    h = production_helpers(TASK_HELPERS)
    for lane in range(cores):
        row, n = tensor(lane % parts), tensor(lane // parts)
        for step in range(80):
            assert int(row) == (lane + step * cores) % parts
            assert int(n) == (lane + step * cores) // parts
            row, n = h._advance_fc1_tile(row, n, parts, cores % parts, cores // parts)


class ReturnRelay:
    ACLSHMEM_SIGNAL_SET = 0

    def __init__(self, signals, checked_base, checkers, epoch):
        self.signals = signals
        self.checked_base = checked_base
        self.checkers = checkers
        self.epoch = epoch
        self.events = []
        self.worker = -1
        self.final_checks = []

    def wait(self, pointer, count, scope, order, waitValue):
        assert scope == "gpu" and order == "acquire"
        slots = [int(pointer.offsets) // 16 + i for i in range(int(count))]
        assert int(pointer.offsets) % 16 == 0
        if slots[0] == self.checked_base:
            assert slots == list(range(self.checked_base, self.checked_base + self.checkers))
            assert waitValue == self.epoch
            self.final_checks.append(self.worker)
        else:
            assert count == 1
            assert self.signals.values[int(pointer.offsets)] == waitValue
            self.events.append((self.worker, "remote-acquire", slots[0]))
        return tensor(count)

    def consume_token(self, pointer, token):
        assert pointer.values is self.signals.values
        self.events.append((self.worker, "consume", int(token)))
        return pointer

    def fence(self):
        self.events.append((self.worker, "fence", None))

    def signal_op(self, pointer, value, operation, rank):
        assert operation == self.ACLSHMEM_SIGNAL_SET
        assert self.events[-1][1] == "fence"
        assert self.events[-2][1] == "consume"
        slot = int(pointer.offsets) // 16
        assert slot == self.checked_base + self.worker
        assert value == self.epoch
        self.signals.values[int(pointer.offsets)] = value
        self.events.append((self.worker, "publish", slot))


@pytest.mark.parametrize("world,cores", [(1, 32), (3, 24), (8, 32), (64, 32), (128, 32)])
@pytest.mark.parametrize("epoch", [1, 7])
def test_return_relay_acquires_every_counter_once(world, cores, epoch):
    epr, max_waves, windows = 7, 23, 16
    wave_counts = [0 if rank % 5 == 0 else 1 + rank % 21 for rank in range(world)]
    table = np.zeros((world, epr + 1, 2), dtype=np.int64)
    table[:, -1, 0] = np.asarray(wave_counts) * windows
    checked_base = max_waves * (2 * cores + world)
    checkers = min(2 * cores, world)
    return_base = 2 * max_waves * cores
    for local in sorted({0, world // 2, world - 1}):
        expected = (max(2 * cores, world) - local + world - 1) // world
        signals = buffer((checked_base + checkers) * 16)
        for destination, waves in enumerate(wave_counts):
            for wave in range(waves):
                signals.values[(return_base + wave * world + destination) * 16] = expected
        relay = ReturnRelay(signals, checked_base, checkers, epoch)
        h = production_helpers(("_wait_dynamic_wave_returns",),
                               {"dl": relay, "libshmem_device": relay})
        for worker in reversed(range(2 * cores)):
            relay.worker = worker
            h._wait_dynamic_wave_returns(worker, Pointer(table.reshape(-1)), signals,
                                         epoch, cores, local, world, epr, max_waves, windows)
        actual = Counter(slot for _, event, slot in relay.events if event == "remote-acquire")
        desired = Counter(return_base + wave * world + destination
                          for destination, waves in enumerate(wave_counts) for wave in range(waves))
        assert actual == desired
        assert sorted(relay.final_checks) == list(range(2 * cores))
        np.testing.assert_array_equal(signals.values[checked_base * 16::16], epoch)


@pytest.mark.parametrize("missing_destination", [0, 3, 7])
def test_missing_early_return_cannot_publish_or_release_reduce(missing_destination):
    world, cores, epr, max_waves, windows, epoch = 8, 32, 4, 3, 16, 9
    checked_base = max_waves * (2 * cores + world)
    return_base = 2 * max_waves * cores
    signals = buffer((checked_base + world) * 16)
    for wave in range(3):
        for destination in range(world):
            signals.values[(return_base + wave * world + destination) * 16] = 8
    signals.values[(return_base + missing_destination) * 16] = 7
    signals.values[checked_base * 16::16] = epoch - 1
    table = np.zeros((world, epr + 1, 2), dtype=np.int64)
    table[:, -1, 0] = 3 * windows

    class Blocked(Exception):
        pass

    class StrictRelay(ReturnRelay):
        def wait(self, pointer, count, scope, order, waitValue):
            offsets = int(pointer.offsets) + np.arange(int(count)) * 16
            if not np.all(self.signals.values[offsets] == waitValue):
                raise Blocked
            return super().wait(pointer, count, scope, order, waitValue)

    relay = StrictRelay(signals, checked_base, world, epoch)
    h = production_helpers(("_wait_dynamic_wave_returns",),
                           {"dl": relay, "libshmem_device": relay})
    for worker in range(2 * cores):
        relay.worker = worker
        with pytest.raises(Blocked):
            h._wait_dynamic_wave_returns(worker, Pointer(table.reshape(-1)), signals,
                                         epoch, cores, 0, world, epr, max_waves, windows)
    published = {worker for worker, event, _ in relay.events if event == "publish"}
    assert published == set(range(world)) - {missing_destination}
    signals.values[(return_base + missing_destination) * 16] = 8
    relay.worker = missing_destination
    h._wait_dynamic_wave_returns(missing_destination, Pointer(table.reshape(-1)), signals,
                                 epoch, cores, 0, world, epr, max_waves, windows)
    for worker in range(2 * cores):
        relay.worker = worker
        h._wait_dynamic_wave_returns(worker, Pointer(table.reshape(-1)), signals,
                                     epoch, cores, 0, world, epr, max_waves, windows)


@pytest.mark.parametrize("cores,epr,windows", [(3, 7, 4), (32, 112, 16), (32, 4, 16)])
def test_fc2_tasks_preserve_per_core_tile_order(cores, epr, windows):
    rng = np.random.default_rng(epr)
    counts = rng.integers(0, 1600, (1, epr))
    counts[:, ::3] = 0
    capacity = int(counts.sum())
    max_waves = (capacity + epr * 255 + windows * 256 - 1) // (windows * 256)
    table, offsets, tasks = make_tables(counts, 256, windows, max_waves)
    issued = []
    dl = SimpleNamespace(wait=lambda *args, **kwargs: 1, consume_token=lambda ptr, token: ptr)
    shmem = SimpleNamespace(fence=lambda: None, signal_op=lambda *args: None,
                            ACLSHMEM_SIGNAL_SET=0)
    h = production_helpers(TASK_HELPERS + ("_fc2_dynamic_wave",), {
        "dl": dl, "libshmem_device": shmem,
        "_fc2_gemm_one_mn_tile": lambda *args: issued.append(tuple(int(args[i]) for i in (3, 4, 5, 6))),
    })
    h._build_wave_tasks(0, table, offsets, tasks, epr, max_waves, 256, windows, capacity)
    signals = buffer(max_waves * (2 * cores + 1) * 16)
    waves = (int(((counts[0] + 255) // 256).sum()) + windows - 1) // windows
    for wave in range(waves):
        for pid in range(cores):
            issued.clear()
            h._fc2_dynamic_wave(pid, wave, None, None, None, signals, offsets, tasks,
                                1, 1, 1, 1, cores, 0, epr, max_waves, 3584, 3072,
                                256, 256, 128, windows, None, None, epr, False)
            expected = []
            for expert in range(epr):
                begin, end, base, first = h._dynamic_wave_expert_range(
                    table, 0, expert, wave, epr, 256, windows)
                if begin < end:
                    tile_base = (first + begin // 256) * 14
                    for tile in range((pid + cores - tile_base % cores) % cores,
                                      int((end - begin + 255) // 256) * 14, cores):
                        row = begin + tile // 14 * 256
                        expected.append((expert, int(base + row), min(256, int(end - row)), tile % 14))
            assert issued == expected


@pytest.mark.parametrize("world,cores,max_waves", [(1, 3, 1), (3, 24, 2), (8, 32, 28), (128, 32, 22)])
def test_signal_allocation_and_reset_cover_checker_slots(world, cores, max_waves):
    kernel = ast.parse(SOURCE.read_text())
    host = ast.parse((SOURCE.parents[3] / "src/mega_moe/ops/forward.py").read_text())
    reset = next(node.value for node in ast.walk(kernel)
                 if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                 and node.target.id == "pipeline_counter_count")
    allocated = next(node.value for node in ast.walk(host)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "pipeline_slots"
                             for target in node.targets))
    reset_count = eval(compile(ast.Expression(reset), str(SOURCE), "eval"), {
        "MAX_PIPELINE_GROUPS": max_waves, "NUM_PROGRAM_CORES": cores, "WORLD_SIZE": world,
    })
    allocated_count = eval(compile(ast.Expression(allocated), "forward.py", "eval"), {
        "self": SimpleNamespace(_single_pipeline_max_groups=max_waves,
                                num_aicore_programs=cores, world_size=world),
    })
    assert reset_count == allocated_count == max_waves * (2 * cores + world) + min(2 * cores, world)
    storage = buffer(allocated_count * 16, 19)
    h = production_helpers()
    for pid in range(cores):
        h._zero_pipeline_counters(pid, storage, cores, reset_count)
    np.testing.assert_array_equal(storage.values[::16], 0)
    np.testing.assert_array_equal(storage.values.reshape(-1, 16)[:, 1:], 19)


def test_pipeline_consumers_use_compact_tasks():
    tree = ast.parse(SOURCE.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_run_dynamic_wave_pipeline", "_dispatch_dynamic_wave",
                 "_fc2_dynamic_wave", "_return_dynamic_wave"):
        calls = [node.func.id for node in ast.walk(functions[name])
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        assert "_load_wave_task" in calls
        assert "_dynamic_wave_expert_range" not in calls
    fc1 = functions["_partition_pipeline_fc1_activation_group_ub"]
    loops = [node for node in ast.walk(fc1) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "pipeline_step"]
    assert len(loops) == 1
    for node in ast.walk(loops[0]):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            assert not (isinstance(node.right, ast.Name)
                        and node.right.id in ("cube_group_row_parts", "vector_group_row_parts"))
