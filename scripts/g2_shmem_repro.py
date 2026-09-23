#!/usr/bin/env python
"""Minimal cross-'node' aclshmem bootstrap repro: dist init + aclshmem_init only.

Plays the G2a rendezvous without pytest/triton so one iteration is ~40s.
Both instances run on this machine; election path is identical to the real
G2a because ASH_MASTER_ADDR points at this host's business IP.

Works with the per-machine 8-rank /etc/hccl_rootinfo.json in place (no /etc
swap needed) — single-machine fallback topology is self-consistent, so the
full GA1..GA8 exchange including step-8 descriptors completes. This is the
calibration reference for what THIS build's 176-byte descriptor looks like.

Usage: REPRO_RANK=0|1 python scripts/g2_shmem_repro.py (see repro_drive.sh)
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

rank = int(os.environ["REPRO_RANK"])
dist.init_process_group(
    backend="hccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    world_size=2,
    rank=rank,
)
torch.npu.set_device(0)
print(f"[r{rank}] dist init OK", flush=True)

from tests import _moe_testkit as kit

size = 2 * 1024 * 1024 * 1024
kit.init_aclshmem(rank, 2, size)  # engine + ip_port from env, same as the test
print(f"[r{rank}] aclshmem_init OK", flush=True)

kit.ash.aclshmem_finalize()
dist.destroy_process_group()
print(f"[r{rank}] finalize OK", flush=True)
