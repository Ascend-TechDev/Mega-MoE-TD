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


def make_peer_mem(saved, dtype, rank):
    """Allocate the shared symmetric buffer sized to the GLOBAL max of send/recv.

    ``dl.symm_at`` only resolves at heap offset 0 (see mega_moe.kernels), so one
    peer_mem is allocated per config and reused by backward step 1 and step 4.
    Using the GLOBAL max (all_reduce MAX) — not per-rank max — keeps peer_mem the
    SAME size on every rank, so any subsequent symmetric allocation (e.g. the
    backward signal_mem) lands at the same heap offset on every rank. Without this,
    signal_mem's offset differs per rank and signal_op RMA writes land at the wrong
    dst offset -> dl.wait never resolves -> deadlock.
    """
    local_elems = max(saved["total_recv"], saved["total_send"]) * saved["hidden_dim"]
    t = torch.tensor([local_elems], dtype=torch.int64, device=f"npu:{rank}")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=saved["ep_group"])
    peer_elems = int(t.item())
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
