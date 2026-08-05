# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared helpers for the multi-rank Mega-MoE tests and benchmarks.

Consolidates the ACLSHMEM lifecycle, per-rank input generation, peer-memory
allocation, benchmark timing, and the bf16 grad-comparison metric that were
duplicated across the forward/backward layer and function tests.

These helpers are test/benchmark-only: they live under ``tests/`` and are not
shipped by the ``mega_moe`` package.
"""

import os
import time

import torch
import torch_npu  # noqa: F401
import shmem as ash
import torch.distributed as dist

from mega_moe.ops._torch_forward import moe_forward

# ----------------------------------------------------------------------------
# ANSI colors (disabled under NO_COLOR for CI logs)
# ----------------------------------------------------------------------------

if os.environ.get("NO_COLOR"):
    GREEN = RED = RESET = BOLD = ""
else:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# ----------------------------------------------------------------------------
# ACLSHMEM bootstrap / lifecycle
# ----------------------------------------------------------------------------

def get_ash_size_bytes(default_gb=2):
    """Symmetric-heap size in bytes, driven by ``MOE_FUSED_ASH_SIZE_GB``."""
    gb = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", str(default_gb)))
    if gb <= 0:
        raise ValueError("MOE_FUSED_ASH_SIZE_GB must be a positive integer")
    return gb * 1024 * 1024 * 1024


def get_ash_ip_port():
    """ACLSHMEM bootstrap endpoint, overridable via ``ASH_MASTER_ADDR/PORT``."""
    addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("ASH_MASTER_PORT", "8666")
    return f"tcp://{addr}:{port}"


def init_aclshmem(rank, world_size, size_bytes, ip_port=None):
    """Initialize the ACLSHMEM symmetric heap for this rank.

    Caller is responsible for ``aclshmem_finalize()`` in a ``finally`` block.
    """
    ash.set_conf_store_tls(False, "")
    attr = ash.InitAttr()
    attr.my_rank = rank
    attr.n_ranks = world_size
    attr.local_mem_size = size_bytes
    attr.ip_port = ip_port if ip_port is not None else get_ash_ip_port()
    attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    if ash.aclshmem_init(attr) != 0:
        raise RuntimeError("aclshmem_init failed")


# ----------------------------------------------------------------------------
# Per-rank input generation (bf16, on NPU)
# ----------------------------------------------------------------------------

def make_backward_inputs(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42):
    """Build rank-distinct bf16 MoE backward inputs.

    Returns ``(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device)``.
    The shared expert routing table (``gw``) is broadcast from rank 0 so every
    rank derives the same per-token expert assignment; weights and activations
    stay rank-distinct for realistic asymmetric load.
    """
    pe = dist.get_rank(ep_group)
    world_size = dist.get_world_size(ep_group)
    epr = num_experts // world_size
    dtype = torch.bfloat16
    device = f"npu:{pe}"
    torch.manual_seed(seed + pe * 1000)
    hs = torch.randn(ntokens, hidden_dim, dtype=dtype, device=device)
    gw = torch.randn(num_experts, hidden_dim, dtype=dtype, device=device)
    fc1_1 = torch.randn(epr, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc1_2 = torch.randn(epr, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc2 = torch.randn(epr, hidden_dim, ffn_dim, dtype=dtype, device=device)
    dist.broadcast(gw, src=0, group=ep_group)
    logits = hs.float() @ gw.float().T
    rw = torch.softmax(logits, dim=-1).to(dtype)
    topk_w, topk_idx = torch.topk(rw, topk, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    dy = torch.randn_like(hs)
    return hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device


def build_backward_saved(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42):
    """Like :func:`make_backward_inputs` but also runs the torch forward to
    produce the ``saved`` intermediates consumed by the backward.

    Returns ``(saved, dy, dtype, device)``.
    """
    hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device = make_backward_inputs(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed)
    with torch.no_grad():
        _, saved = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
    return saved, dy, dtype, device


def make_peer_mem(saved, dtype, rank):
    """Allocate the shared symmetric buffer sized to the larger of send/recv.

    ``dl.symm_at`` only resolves at heap offset 0 (see mega_moe.kernels), so one
    peer_mem is allocated per config and reused by backward step 1 and step 4.
    """
    peer_elems = max(saved["total_recv"], saved["total_send"]) * saved["hidden_dim"]
    return ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=rank)


# ----------------------------------------------------------------------------
# Benchmark timing
# ----------------------------------------------------------------------------

def bench(fn, warmup, iters, ep_group):
    """Warm up ``fn`` then time it over ``iters`` runs.

    Returns the slowest rank's average per-call latency in ms (collective → the
    slowest rank is the truth).
    """
    for _ in range(warmup):
        fn()
    dist.barrier(ep_group)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    local_ms = (t1 - t0) / iters * 1000.0
    t = torch.tensor([local_ms], device=f"npu:{dist.get_rank(ep_group)}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ep_group)
    return float(t.item())
