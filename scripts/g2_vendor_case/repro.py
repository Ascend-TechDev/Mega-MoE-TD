#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standalone dual-node ACLSHMEM transport reproducer (Atlas 950 SuperPoD).

No repository dependencies — python torch + the cann_shmem wheel only.  Three
cases map to the three headline cells of the behavioral matrix established in
rounds r27d-r32 (see docs/g2-vendor-escalation.md):

  push   — one put_signal per rank, then read back the local tensor.
            Expected (BUG): node1's bulk lands on node0, node0's bulk to
            node1 is SILENTLY DROPPED (local fill value stays; signal word
            lands on BOTH sides — the control plane is healthy).
  pull   — ONLY rank 1 issues a cross-node getmem from rank 0's tensor.
            Expected (BUG): rank 1 worker SIGABRTs (ERR02005 DIST internal
            error) and the device reports aicore error 271 "The address for
            scalar to access the internal buffer is out of bounds" in the
            getmem copy task.  Rank 0 stays passive and hangs in the HCCL
            barrier until torchrun reaps it.
  tunnel — 256 payload words carried one SIGNAL_SET per slot, both
            directions.  Expected (CONTROL, WORKS): every word lands on both
            sides — proves the machines/link/bootstrap are fine and the fault
            is specific to the bulk/pull engine paths.

Run on BOTH nodes (examples with node0=141.61.95.70, node1=141.61.95.30):

  node0$ LD_LIBRARY_PATH=<site-packages>/shmem/backends/950:$LD_LIBRARY_PATH \
         ASH_MASTER_ADDR=141.61.95.70 ASH_MASTER_PORT=41921 MOE_ASH_ENGINE=udma \
         torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
                  --master_addr=141.61.95.70 --master_port=29561 repro.py push
  node1$ (same env, --node_rank=1)

Each case takes ~15 s (init dominates).  The engine mirror (mte) reproduces
the same silent drop with the direction flipped.
"""

import os
import sys
import time

import torch
import torch.distributed as dist

import shmem as ash
from shmem.core.direct import SignalOp
from shmem.core.rma import put_signal
from shmem.core.utils import Buffer

MAGIC = 0x5A5A


def setup():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")

    ash.set_conf_store_tls(False, "")
    attr = ash.InitAttr()
    attr.my_rank = rank
    attr.n_ranks = world
    attr.local_mem_size = 1 << 30  # 1 GB symmetric heap
    attr.ip_port = f"tcp://{os.environ['ASH_MASTER_ADDR']}:{os.environ['ASH_MASTER_PORT']}"
    engine = os.environ.get("MOE_ASH_ENGINE", "udma")
    if engine == "udma":
        attr.option_attr.data_op_engine_type = ash.OpEngineType.UDMA
    elif engine == "mte":
        attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    else:  # combo
        attr.option_attr.data_op_engine_type = ash.OpEngineType(
            ash.OpEngineType.MTE.value | ash.OpEngineType.UDMA.value)
    if ash.aclshmem_init(attr) != 0:
        raise RuntimeError("aclshmem_init failed")
    return rank, world, local_rank


def heap_tensor(n, dev_id):
    return ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)


def gathered_flags(my_flag, dev_id):
    t = torch.tensor([1 if my_flag else 0], dtype=torch.int64, device=f"npu:{dev_id}")
    parts = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, t)
    return [int(p.item()) for p in parts]


def case_push(rank, dev_id):
    """r27d: put_signal bulk leg — asymmetric silent drop under udma."""
    peer = 1 - rank
    data = heap_tensor(64, dev_id)
    sig = heap_tensor(1, dev_id)
    data.fill_(0)
    sig.fill_(0)
    torch.npu.synchronize()
    dist.barrier()

    data.fill_(MAGIC + rank)
    torch.npu.synchronize()
    put_signal(Buffer(data.data_ptr(), data.numel() * data.element_size()),
               Buffer(data.data_ptr(), data.numel() * data.element_size()),
               Buffer(sig.data_ptr(), sig.numel() * sig.element_size()),
               1, SignalOp.SIGNAL_ADD, remote_pe=peer)
    dist.barrier()
    torch.npu.synchronize()

    got = data.tolist()
    landed = all(v == MAGIC + peer for v in got)
    print(f"[push node{rank}] data[0]={got[0]:#x} (peer magic {MAGIC + peer:#x}) "
          f"-> peer bulk {'LANDED' if landed else 'SILENTLY DROPPED'}; "
          f"sig={int(sig.item())} (control plane {'OK' if int(sig.item()) == 1 else 'BROKEN'})",
          flush=True)

    flags = gathered_flags(landed, dev_id)
    # rank0 observes the n1->n0 direction; rank1 observes n0->n1.
    dirs = {0: "n1->n0", 1: "n0->n1"}
    state = " ".join(f"{dirs[i]}={'LANDED' if f else 'DROPPED'}" for i, f in enumerate(flags))
    if rank == 0:
        if flags[0] and not flags[1]:
            print(f"[push] VERDICT: {state} — BUG REPRODUCED "
                  f"(asymmetric silent bulk drop; signals land both ways)", flush=True)
        elif flags[0] and flags[1]:
            print(f"[push] VERDICT: {state} — BUG NOT REPRODUCED (both directions landed)",
                  flush=True)
        else:
            print(f"[push] VERDICT: {state} — different failure shape than observed; "
                  f"see docs/g2-vendor-escalation.md matrix", flush=True)


def case_pull(rank, dev_id):
    """r31: node1-initiated cross-node getmem — SIGABRT + aicore error 271."""
    peer = 1 - rank
    data = heap_tensor(64, dev_id)
    recv = heap_tensor(64, dev_id)
    data.fill_(0)
    recv.fill_(0)
    torch.npu.synchronize()
    dist.barrier()

    data.fill_(MAGIC + rank)
    torch.npu.synchronize()
    dist.barrier()

    if rank == 1:
        print("[pull node1] issuing cross-node getmem from node0 ...", flush=True)
        ash.aclshmem_getmem(recv.data_ptr(), data.data_ptr(),
                            data.numel() * data.element_size(), peer)
        torch.npu.synchronize()
        got = recv.tolist()
        print(f"[pull node1] recv[0]={got[0]:#x} want={MAGIC:#x} — UNEXPECTED SURVIVAL",
              flush=True)
    else:
        # Passive: with the bug, node1 SIGABRTs and this barrier never
        # completes — torchrun's elastic agent reaps us.  That hang IS part
        # of the reproduction; no local timeout is imposed on purpose.
        print("[pull node0] passive barrier (expected to hang until agent reaps)",
              flush=True)
    dist.barrier()


def case_tunnel(rank, dev_id):
    """r32 control case: signal leg carries payload BOTH directions."""
    peer = 1 - rank
    n_words = 256
    data = heap_tensor(64, dev_id)
    sigarr = heap_tensor(n_words, dev_id)
    data.fill_(0)
    sigarr.fill_(0)
    torch.npu.synchronize()
    dist.barrier()
    data_buf = Buffer(data.data_ptr(), data.numel() * data.element_size())

    def send(base):
        t0 = time.perf_counter()
        for i in range(n_words):
            put_signal(data_buf, data_buf,
                       Buffer(sigarr.data_ptr() + i * 8, 8), base + i,
                       SignalOp.SIGNAL_SET, remote_pe=peer)
        print(f"[tunnel node{rank}] sent {n_words} signal words in "
              f"{(time.perf_counter() - t0) * 1000:.1f}ms "
              f"({(time.perf_counter() - t0) / n_words * 1e6:.0f}us/op)", flush=True)

    def check(base, tag):
        torch.npu.synchronize()
        got = sigarr.tolist()
        bad = sum(1 for i, v in enumerate(got) if v != base + i)
        print(f"[tunnel node{rank}] {tag}: bad_slots={bad}/{n_words}", flush=True)
        return bad == 0

    ok = True
    if rank == 1:
        send(0x5A5A0000)
        torch.npu.synchronize()
    dist.barrier()
    if rank == 0:
        ok = check(0x5A5A0000, "r1->r0")
    sigarr.fill_(0)
    torch.npu.synchronize()
    dist.barrier()
    if rank == 0:
        send(0x5B5B0000)
        torch.npu.synchronize()
    dist.barrier()
    if rank == 1:
        ok = check(0x5B5B0000, "r0->r1")

    flags = gathered_flags(ok, dev_id)
    if rank == 0:
        print(f"[tunnel] VERDICT: signal leg payload {'WORKS both directions (control OK)' if all(flags) else 'FAILED — see above'}",
              flush=True)


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CASE", "push")
    rank, world, dev_id = setup()
    print(f"[node{rank}] case={case} engine={os.environ.get('MOE_ASH_ENGINE', 'udma')} "
          f"world={world}", flush=True)
    try:
        if case == "push":
            case_push(rank, dev_id)
        elif case == "pull":
            case_pull(rank, dev_id)
        elif case == "tunnel":
            case_tunnel(rank, dev_id)
        else:
            raise SystemExit(f"unknown case: {case} (push|pull|tunnel)")
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
