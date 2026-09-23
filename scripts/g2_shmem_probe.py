#!/usr/bin/env python
"""G2 cross-node aclshmem bootstrap probe (dist init + aclshmem_init only).

Cuts one G2a debug iteration from ~8.5 min (pytest) to ~1.5 min and prints
its own forensics (newest aclshmem plog tail) on failure, so a single log
paste from each node is enough to diagnose.

Usage (both nodes, same env block as G2a plus SHMEM_LOG_LEVEL=INFO):
  node0: MMT_NODE_RANK=0 ... python scripts/g2_shmem_probe.py
  node1: MMT_NODE_RANK=1 ... python scripts/g2_shmem_probe.py

Exit 0 + "G2_PROBE_RESULT=PASS" line = bootstrap (store exchange + Hcomm
connect) is healthy on this rank; run the real pytest gate next.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SHMEM_LOG_DIR = "/root/shmem/log"


def tail_newest_plog(own_pid, lines=40):
    """Print the tail of this process's aclshmem plog (for the relay paste)."""
    try:
        logs = sorted(
            (f for f in os.listdir(SHMEM_LOG_DIR) if f.startswith(f"aclshmem_{own_pid}_")),
            reverse=True,
        )
    except OSError:
        print(f"[probe] no plog dir {SHMEM_LOG_DIR}")
        return
    if not logs:
        print(f"[probe] no plog for pid {own_pid}")
        return
    path = os.path.join(SHMEM_LOG_DIR, logs[0])
    print(f"[probe] plog tail: {path}")
    try:
        out = subprocess.run(["tail", "-n", str(lines), path],
                             capture_output=True, text=True, timeout=10)
        print(out.stdout)
    except Exception as exc:  # noqa: BLE001 - forensics must never mask the result
        print(f"[probe] tail failed: {exc}")


def main() -> int:
    rank = int(os.environ.get("MMT_NODE_RANK", "0"))
    world = 2
    master = os.environ.get("MASTER_ADDR", "141.61.95.70")
    port = os.environ.get("MASTER_PORT", "29531")
    print(f"[probe r{rank}] dist init tcp://{master}:{port} world={world}", flush=True)

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    dist.init_process_group(
        backend="hccl",
        init_method=f"tcp://{master}:{port}",
        world_size=world,
        rank=rank,
    )
    torch.npu.set_device(0)
    print(f"[probe r{rank}] dist init OK", flush=True)

    from tests import _moe_testkit as kit

    size = 2 * 1024 * 1024 * 1024
    kit.init_aclshmem(rank, world, size)
    print(f"[probe r{rank}] aclshmem_init OK", flush=True)

    kit.ash.aclshmem_finalize()
    dist.destroy_process_group()
    print(f"[probe r{rank}] finalize OK", flush=True)
    print("G2_PROBE_RESULT=PASS")
    return 0


if __name__ == "__main__":
    own_pid = os.getpid()
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001 - report, then attach forensics
        print(f"[probe r{os.environ.get('MMT_NODE_RANK', '?')}] FAILED: {exc!r}", flush=True)
        tail_newest_plog(own_pid)
        print("G2_PROBE_RESULT=FAIL")
        code = 1
    sys.exit(code)
