# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only checks for the full-active dispatch task mapping.

The production helper is a Triton JIT function, so importing its package would
pull in device-only dependencies.  Extracting and executing that helper's
Python AST tests the actual arithmetic body without substituting a copied
implementation or requiring an NPU.
"""

import ast
from pathlib import Path

import pytest


def _load_full_active_task_id():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mega_moe"
        / "kernels"
        / "dispatch_fc1.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_full_active_task_id"
    )
    function.decorator_list = []
    function.returns = None
    for argument in function.args.args:
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, source_path, "exec"), namespace)
    return namespace["_full_active_task_id"]


@pytest.mark.parametrize(
    ("world_size", "experts_per_rank"),
    [(2, 64), (4, 224), (8, 112)],
)
def test_full_active_map_matches_expert_major_schedule(
    world_size, experts_per_rank
):
    map_task = _load_full_active_task_id()
    expected = [
        dst_rank * experts_per_rank + expert_id
        for expert_id in range(experts_per_rank)
        for dst_rank in range(world_size)
    ]
    actual = [
        map_task(active_id, world_size, experts_per_rank)
        for active_id in range(world_size * experts_per_rank)
    ]

    assert actual == expected
    assert sorted(actual) == list(range(world_size * experts_per_rank))


@pytest.mark.parametrize("num_cores", [7, 28, 32, 64])
def test_large_task_space_direct_striping_visits_every_bucket_once(num_cores):
    """The production path partitions raw task ordinals without compaction."""
    world_size = 8
    experts_per_rank = 112
    num_tasks = world_size * experts_per_rank
    map_task = _load_full_active_task_id()

    visited = [
        map_task(schedule_task, world_size, experts_per_rank)
        for pid in range(num_cores)
        for schedule_task in range(pid, num_tasks, num_cores)
    ]

    assert len(visited) == num_tasks
    assert sorted(visited) == list(range(num_tasks))


def test_large_task_space_does_not_call_sparse_compaction_scan():
    """Pin the load-bearing split: the large-task arm cannot rescan all tasks."""
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mega_moe"
        / "kernels"
        / "dispatch_fc1.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dispatch_count_derived_source_tiles"
    )
    large_task_arm = next(
        node
        for node in function.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.ops[0], ast.Gt)
    )

    large_arm_calls = {
        node.func.id
        for statement in large_task_arm.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    fallback_calls = {
        node.func.id
        for statement in large_task_arm.orelse
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "_dispatch_one_source_tile_task" in large_arm_calls
    assert "_find_nth_nonempty_task" not in large_arm_calls
    assert "_find_nth_nonempty_task" in fallback_calls
