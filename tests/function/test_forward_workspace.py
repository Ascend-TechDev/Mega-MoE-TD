# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Exercise real workspace sizing using CPU tensors in place of ACLSHMEM."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("grouped_tile,fused_tile", [(128, None), (128, 256), (256, 128), (128, 16)])
@pytest.mark.parametrize("moonep", [False, True])
def test_dispatch_readiness_covers_both_launches(monkeypatch, grouped_tile, fused_tile, moonep):
    path = Path(__file__).resolve().parents[2] / "src/mega_moe/runtime/workspace.py"
    spec = importlib.util.spec_from_file_location("workspace_sizing_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "shmem", SimpleNamespace(
        my_pe=lambda: 0,
        pe_count=lambda: 2,
        aclshmem_create_tensor=lambda shape, dtype, device_id: torch.empty(shape, dtype=dtype),
    ))
    context = module.create_moe_forward_context(
        max_tokens_per_rank=257, hidden_size=16, top_k=3, num_experts=14,
        rank=0, world_size=2, receive_capacity_factor=2,
        dispatch_fc1_block_size_m=grouped_tile,
        single_kernel_dispatch_block_size_m=fused_tile,
        enable_moonep=moonep,
    )
    experts = context.physical_experts_per_rank
    dispatch_slots = 2 * experts * context.max_source_tiles
    for tile in (grouped_tile, fused_tile or grouped_tile):
        # Worst-case routing sends every route from one source to one expert.
        tiles_needed = (257 * 3 + tile - 1) // tile
        assert tiles_needed <= context.max_source_tiles
        last_slot = (2 * experts - 1) * context.max_source_tiles + tiles_needed - 1
        assert last_slot < dispatch_slots
    assert context.signal_mem.numel() == (dispatch_slots + 3 * context.replica_budget) * 16
    if moonep:
        assert context.replica_gate_ready.storage_offset() == dispatch_slots * 16
        assert context.replica_down_ready.storage_offset() == (dispatch_slots + 7) * 16
