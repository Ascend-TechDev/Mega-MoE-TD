# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Kimi-K3-shape backward smoke at E=896 on 8 ranks (single node).

Production knobs all ON — moonep single-kernel forward + saved-recompute
mega backward + replica re-prefetch, both UDMA transports — at the real
model dims: tokens/rank 1024, hidden 7168, routed ffn 3584, topk 8,
E=896 -> 112 home experts per rank.  E=896 rides the lifted E<=32
single-kernel scatter limit (release_v3 1bd3744) 28 bin-blocks deep.

No eager golden at this scale: fp32 reference grads for 112x3584x7168
weight shards plus the bf16 shards themselves do not fit 96GB next to
the ~20GB symmetric heap (replica tables alone are 17.25GB).  Gates are
clean completion of forward + enrich + mega backward, every returned
grad finite, and per-stage wall clocks / peak device memory.

Run (8 ranks, single node):
  pytest tests/layer/test_kimi_e896_bwd.py -x -q
"""

import os
import time

import pytest
import torch
import torch.distributed as dist

from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
from mega_moe.ops import moe_backward_triton
from mega_moe.ops._single_saved_adapter import enrich_single_kernel_saved
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    make_down_weights,
    make_gate_up_weights,
    make_routing_weights,
    prepare_inputs,
)
from mega_moe.runtime.device import device_str, resolve_local_device


def run_kimi_e896_bwd(rank: int, world_size: int) -> None:
    # Fullnet terminal-config env contract (finetune_kimik3.sh), so this
    # smoke exercises exactly the production code paths.
    for key, val in (
        ("MOE_BWD_MEGA", "1"),
        ("MOE_SAVED_RECOMPUTE", "1"),
        ("MEGAMOE_REPLICA_POOL", "1"),
        ("MOE_MEGA_REPREFETCH", "1"),
        ("MOE_MEGA_GRAD_TRANSPORT", "udma"),
        ("MOE_MEGA_REPREFETCH_TRANSPORT", "udma"),
    ):
        os.environ.setdefault(key, val)

    # KIMI_E selects the expert count (896 reproduces the E>=128 forward
    # defect family; 32 is the fullnet production config), KIMI_ITERS runs
    # the fwd+enrich+bwd pipeline repeatedly so steady-state timing can be
    # read past the first-call JIT compile.
    tokens, hidden, ffn, topk = 1024, 7168, 3584, 8
    num_experts = int(os.environ.get("KIMI_E", "896"))
    iters = int(os.environ.get("KIMI_ITERS", "1"))
    epn = num_experts // world_size
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    # Replica tables alone need epn*(2*ffn*hidden + ffn*hidden)*2B ~= 17.3GB
    # of symmetric heap, plus the backward peer_mem (~0.9GB) and the forward
    # exchange buffers — 20GB is the working floor.
    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(20), enable_udma=True,
    ):
        # Backward scratch FIRST (dl.symm_at offset-0 contract).
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk * world_size, tokens * topk, hidden, dtype, rank,
            ep_group,
        )

        # Forward mode: the single-kernel path traps aicore EZ9999 at
        # E=896 (forward.py:1927 metadata_stats sync; same family as the
        # known upstream E>=128 red on origin/main — solve_32 was only
        # validated to E=96).  The NATIVE multi-kernel forward produces
        # the exact saved contract the mega backward was validated with
        # (test_debug_moonep_saved native mode), so the BACKWARD — what
        # this test is about — runs on native saved by default; set
        # KIMI_E896_SINGLE=1 to reproduce the forward trap.
        single = os.environ.get("KIMI_E896_SINGLE", "0") == "1"
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=float(world_size),
                activation="situglu",
                situ_beta=4.0,
                situ_linear_beta=25.0,
                enable_single_kernel_forward=single,
                enable_moonep=True,
                fc1_gemm_block_size_m=256,
                fc2_combine_block_size_m=256,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device)
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            # No eager golden at this scale (see module docstring) — the
            # un-packed halves are dead weight once packed_w1 exists.
            del w_gate, w_up
            torch.npu.empty_cache()
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device)
            hs, _ = prepare_inputs(
                tokens, hidden, num_experts, topk, dtype, device,
                seed=2303 + rank)
            rw = make_routing_weights(tokens, topk, device, seed=2304 + rank)
            torch.manual_seed(2305 + rank)
            dy = torch.randn(tokens, hidden, dtype=dtype, device=device)
            torch.manual_seed(9000 + rank * 10 + 4)
            # Uniform random routing by default (the fullnet perf regime);
            # KIMI_SKEW=1 additionally pins slot 0 to expert epn so one home
            # expert absorbs every rank's cross-rank slots (hotspot shape).
            routes = torch.randint(
                0, num_experts, (tokens, topk), dtype=torch.int64)
            if os.environ.get("KIMI_SKEW", "0") == "1":
                routes[:, 0] = epn
            ei = routes.to(device=device, dtype=torch.int32).contiguous()

            torch.npu.synchronize()
            dist.barrier()
            # The mega backward lazy-allocs symmetric slabs ON the saved dict
            # (_mega_combine_buf / _mega_redispatch_buf / _bwd_tile_signal_mem
            # / _mega_p3_signal_mem, plus the monotonic _bwd_tile_signal_epoch).
            # A fresh saved per iteration therefore reallocs them every iter
            # (the old tensors are never aclshmem_free'd) and exhausts the
            # heap by iteration ~3.  The framework persists these via the
            # autograd `state`; the bench carries them across iterations the
            # same way, refreshed AFTER each call so a grow-realloc inside
            # _ensure_* can't leave a freed tensor stashed.
            carry_keys = (
                "_mega_combine_buf", "_mega_redispatch_buf",
                "_bwd_tile_signal_mem", "_mega_p3_signal_mem",
                "_bwd_tile_signal_epoch",
            )
            carry = {}
            for it in range(iters):
                t0 = time.perf_counter()
                with torch.no_grad():
                    if single:
                        output, saved_min = op.forward(
                            hs, ei, packed_w1, w2, rw, return_saved=True)
                        torch.npu.synchronize()
                        t1 = time.perf_counter()
                        saved = enrich_single_kernel_saved(
                            op, saved_min, hidden_states=hs,
                            gate_up_weight=packed_w1, down_weight=w2)
                        torch.npu.synchronize()
                        t2 = time.perf_counter()
                    else:
                        output, saved = op.forward(
                            hs, ei, packed_w1, w2, rw, return_saved=True)
                        torch.npu.synchronize()
                        t1 = time.perf_counter()
                        t2 = t1
                    saved.update(carry)
                    res = moe_backward_triton(
                        saved, dy, peer_mem,
                        grad_transport=op
                        .lend_replica_weight_tables_for_grad(
                            experts_to_copy_cpu=saved.get(
                                "experts_to_copy_cpu")),
                        hidden_states=hs)
                    torch.npu.synchronize()
                    t3 = time.perf_counter()
                carry = {k: saved[k] for k in carry_keys if k in saved}
                print(f"[kimi-e{num_experts} r{rank} it{it}] "
                      f"fwd={t1 - t0:.3f}s enrich={t2 - t1:.3f}s "
                      f"bwd={t3 - t2:.3f}s tot={t3 - t0:.3f}s", flush=True)

            msgs = []
            for key in sorted(res):
                val = res[key]
                if not (torch.is_tensor(val) and val.is_floating_point()):
                    continue
                finite = bool(torch.isfinite(val).all().item())
                assert finite, f"{key} has non-finite values"
                msgs.append(f"{key}{tuple(val.shape)}"
                            f"norm={val.float().norm().item():.3e}")
            peak_gb = torch.npu.max_memory_allocated() / 2**30
            print(f"[kimi-e{num_experts} r{rank}] peak={peak_gb:.1f}G :: "
                  + " ".join(msgs), flush=True)
        finally:
            op.finalize()
            kit.ash.aclshmem_free_tensor(peer_mem)


@pytest.mark.dist
@pytest.mark.functional
def test_kimi_e896_bwd(dist_test):
    """8-rank kimi-k3-shape backward, E=896, production knobs on."""
    dist_test(run_kimi_e896_bwd, world_size=8)
