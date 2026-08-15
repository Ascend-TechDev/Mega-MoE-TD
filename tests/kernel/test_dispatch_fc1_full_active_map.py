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
