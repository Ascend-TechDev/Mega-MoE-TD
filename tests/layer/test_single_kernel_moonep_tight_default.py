# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Single-kernel MoonEP under the tight default receive window (cf=2.0).

Exercises the default factor resolution end-to-end: no explicit
receive_capacity_factor, so the op-library default applies (2.0 on the
MoonEP path — the B.0-B.3 planner pins every destination's receive at
S*topk and the backward wgrad pad adds at most one more S*topk).  The
window assertion pins the sizing; run_full_one green over skewed routes
proves fwd+bwd through the 4x-smaller peer_mem.
"""

import pytest
import torch
import torch.distributed as dist

from benchmark.layer._kimi_routes import kimi_skewed_routes
from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    make_gate_up_weights, make_down_weights, make_routing_weights, run_full_one,
)
from mega_moe.runtime.device import device_str, resolve_local_device


def run_tight_default(rank, world, tokens=64):
    device = device_str(resolve_local_device(rank))
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
                enable_single_kernel_forward=True, enable_moonep=True),
        )
        try:
            # No explicit factor: the MoonEP default must resolve to the
            # proven tight bound 2.0 (NOT world_size).
            window_rows = op.context.peer_mem.numel() // hidden
            assert window_rows == 2 * tokens * topk, (
                f"tight-default window is {window_rows} rows, expected "
                f"{2 * tokens * topk} (cf=2.0)")
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
                                    experts, f"tight-default-{iteration}", rank, device)
                assert op.context.metadata_recv_per_expert.sum().item() == tokens * topk
                # Weight refresh between iterations, like the UDMA suite.
                packed.mul_(0.75)
                gate.mul_(0.75)
                up.mul_(0.75)
                down.mul_(0.5)
        finally:
            torch.npu.synchronize()
            dist.barrier()
            op.finalize()


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world", (2, 8))
def test_single_kernel_moonep_tight_default(dist_test, world):
    dist_test(run_tight_default, world_size=world, args=(64,))
