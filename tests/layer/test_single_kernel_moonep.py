# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""UDMA single-launch correctness, routing churn, and weight refresh."""

import pytest
import torch
import torch.distributed as dist

from benchmark.layer._kimi_routes import kimi_skewed_routes
from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
from mega_moe.runtime.moonep_planning import plan_moonep_b0_b3
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    make_gate_up_weights, make_down_weights, make_routing_weights, run_full_one,
)


def run_single_moonep(rank, world, tokens=64, chunk_bytes=64 * 1024 * 1024):
    device = f"npu:{rank}"
    experts, hidden, ffn, topk = 32, 256, 512, 16
    bootstrap = torch.zeros(world, device=device)
    dist.all_reduce(bootstrap)
    dist.all_to_all_single(torch.empty_like(bootstrap), bootstrap)
    torch.npu.synchronize()
    with kit.aclshmem_session(rank, world, kit.get_ash_size_bytes(2), enable_udma=True):
        op = FusedMoEForward(
            dist.group.WORLD, max_tokens_per_rank=tokens, hidden_size=hidden,
            top_k=topk, num_experts=experts,
            config=MoEForwardConfig(
                enable_single_kernel_forward=True, enable_moonep=True,
                receive_capacity_factor=float(world), moonep_udma_chunk_bytes=chunk_bytes),
        )
        try:
            gate, up = make_gate_up_weights(experts, hidden, ffn, world, rank,
                                            torch.bfloat16, device)
            packed = pack_gate_up_weights(gate, up)
            down = make_down_weights(experts, hidden, ffn, world, rank,
                                     torch.bfloat16, device)
            torch.manual_seed(991 + rank)
            hs = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).mul_(0.5)
            weights = make_routing_weights(tokens, topk, device, seed=819 + rank)
            base = kimi_skewed_routes(tokens, experts, topk, rank)
            for iteration, shift in enumerate((0, 4, 0)):
                routes = ((base + shift) % experts).to(device)
                assert run_full_one(op, hs, routes, weights, gate, up, packed, down,
                                    experts, f"UDMA-skewed-{iteration}", rank, device)
                counts = torch.stack([
                    torch.bincount(((kimi_skewed_routes(tokens, experts, topk, r) + shift)
                                    % experts).flatten().long(), minlength=experts)
                    for r in range(world)])
                oracle = plan_moonep_b0_b3(counts)
                assert torch.equal(op.context.planning_experts_to_copy.cpu(), oracle.experts_to_copy)
                assert torch.equal(op.context.planning_alloc_cumsum.cpu(), oracle.alloc_cumsum)
                assert op.context.metadata_recv_per_expert.sum().item() == tokens * topk
                # Neither unchanged pointers nor repeated slots may reuse stale weights.
                packed.mul_(0.75)
                gate.mul_(0.75)
                up.mul_(0.75)
                down.mul_(0.5)
            for dropped in (False, True):
                routes = base.to(device)
                routes[:, : (topk if dropped else 3)] = -1
                assert run_full_one(op, hs, routes, weights, gate, up, packed, down,
                                    experts, f"UDMA-dropped-{dropped}", rank, device)
        finally:
            torch.npu.synchronize()
            dist.barrier()
            op.finalize()


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("chunk_bytes", (64 * 1024 * 1024, 128 * 1024))
@pytest.mark.parametrize("world", (2, 8))
def test_single_kernel_moonep_udma(dist_test, chunk_bytes, world):
    dist_test(run_single_moonep, world_size=world, args=(64, chunk_bytes))
