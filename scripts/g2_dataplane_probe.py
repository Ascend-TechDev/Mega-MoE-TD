#!/usr/bin/env python
"""G2 cross-node DATA-PLANE probe v2 (r20): push (putmem) vs pull (getmem).

r18/r19 proved the kernel's cross-node RMA writes never become visible
(both nodes materialize only their own routing rows, offsets verified
identical) while HCCL and the bootstrap stay green.  This probe separates
the two surviving hypotheses:

  A1  kernel race — counts exchange has no arrival gate, the read beats
      the peer's write (cross-node latency > exchange->read window).
  A2  cross-node putmem transport is broken outright (the r13 bootstrap
      only validated the GW metadata plane, never the data QPs).

Construction: every send is followed by a GLOO barrier before the receiver
polls, so a delivery failure here CANNOT be a latency race — PASS =>
transport fine (fix = kernel arrival gate), FAIL => transport broken.

  1. gloo dist init + kit.init_aclshmem (same engine selection as the gate)
  2. first symmetric alloc 512-elem int64; print heap base/offset both
     sides; all_reduce audit of offset symmetry
  3. PUSH phase, both directions: fill local buf (pattern+MAGIC),
     aclshmem_putmem_nbi to the peer, quiet + npu sync, gloo barrier,
     peer polls locally (30s) and verifies byte-exact
  4. PULL phase, both directions: getmem from the peer's buf, verify
  5. prints one G2_RMA_<dir> verdict per direction + final G2_RMA=PASS/FAIL

Usage identical to g2_shmem_probe.py (MMT_NODE_RANK, MASTER_ADDR/PORT,
ASH_MASTER_ADDR/PORT, MEGAMOE_MULTI_NODE=1, atomic /etc rootinfo swap).
"""
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MAGIC0, MAGIC1 = 0xC0FFEE0B, 0xBA0BAB1E
PAT0, PAT1 = 0x11110000, 0x22220000
WATCHDOG_S = 300
POLL_S = 30


def main() -> int:
    rank = int(os.environ.get("MMT_NODE_RANK", "0"))
    world = 2
    master = os.environ.get("MASTER_ADDR", "141.61.95.70")
    port = os.environ.get("MASTER_PORT", "29541")

    def bomb():
        print(f"G2_RMA=TIMEOUT r{rank} (watchdog)", flush=True)
        os._exit(3)
    watchdog = threading.Timer(WATCHDOG_S, bomb)
    watchdog.daemon = True
    watchdog.start()

    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    dist.init_process_group(
        backend="gloo", init_method=f"tcp://{master}:{port}",
        world_size=world, rank=rank,
    )
    torch.npu.set_device(0)
    print(f"[rma r{rank}] gloo init OK", flush=True)

    from tests import _moe_testkit as kit

    kit.init_aclshmem(rank, world, 2 * 1024 * 1024 * 1024)
    ash = kit.ash
    print(f"[rma r{rank}] aclshmem_init OK pe={ash.my_pe()}", flush=True)

    heap_base = ash.aclshmemx_get_heap_base()
    n = 512
    buf = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=0)
    off = buf.data_ptr() - heap_base
    t = torch.tensor([off], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    off_max = int(t.item())
    t = torch.tensor([off], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    print(f"[rma r{rank}] base=0x{heap_base:x} buf_off=0x{off:x} "
          f"(peer max=0x{off_max:x} min=0x{int(t.item()):x} "
          f"{'SYMMETRIC' if off_max == int(t.item()) else 'ASYMMETRIC!!'})",
          flush=True)

    pe = 1 - rank
    my_magic = MAGIC0 if rank == 0 else MAGIC1
    my_pat = PAT0 if rank == 0 else PAT1
    peer_magic = MAGIC1 if rank == 0 else MAGIC0
    peer_pat = PAT1 if rank == 0 else PAT0
    quiet = None
    for cand in ("aclshmem_quiet", "aclshmem_fence", "aclshmem_barrier_all"):
        quiet = getattr(ash, cand, None)
        if quiet:
            print(f"[rma r{rank}] flush primitive: {cand}", flush=True)
            break
    if quiet is None:
        print("[rma r{rank}] WARNING: no flush primitive found; relying on "
              "nbi issue + poll window", flush=True)

    def fill_and_sync():
        buf.fill_(my_pat)
        buf[n - 1].fill_(my_magic)
        torch.npu.synchronize()

    def verify(tag):
        ok_magic = int(buf[n - 1].item()) == peer_magic
        body = buf[: n - 1].cpu()
        ok_pat = bool((body == peer_pat).all().item())
        print(f"G2_RMA_{tag}_r{rank}={'PASS' if ok_magic and ok_pat else 'FAIL'} "
              f"magic={int(buf[n - 1].item()):#x} pat_ok={ok_pat}", flush=True)
        return ok_magic and ok_pat

    results = []

    # ---- PUSH: even rank first (odd rank's poll also covers r1->r0 by
    # swapping roles each half).  Direction A: r0 sends -> r1 polls.
    if rank == 0:
        fill_and_sync()
    dist.barrier()
    t0 = time.time()
    if rank == 0:
        ash.aclshmem_putmem_nbi(buf.data_ptr(), buf.data_ptr(), n * 8, pe)
        if quiet:
            quiet()
        torch.npu.synchronize()
        print(f"[rma r0] push issued ({time.time()-t0:.1f}s)", flush=True)
    dist.barrier()  # <-- kills the race by construction
    if rank == 1:
        deadline = time.time() + POLL_S
        while time.time() < deadline and int(buf[n - 1].item()) != peer_magic:
            time.sleep(0.2)
        results.append(("push_r0_to_r1", verify("push_0to1")))
    dist.barrier()

    # Direction B: r1 sends -> r0 polls.
    if rank == 1:
        fill_and_sync()
    dist.barrier()
    if rank == 1:
        ash.aclshmem_putmem_nbi(buf.data_ptr(), buf.data_ptr(), n * 8, pe)
        if quiet:
            quiet()
        torch.npu.synchronize()
        print(f"[rma r1] push issued ({time.time()-t0:.1f}s)", flush=True)
    dist.barrier()
    if rank == 0:
        deadline = time.time() + POLL_S
        while time.time() < deadline and int(buf[n - 1].item()) != peer_magic:
            time.sleep(0.2)
        results.append(("push_r1_to_r0", verify("push_1to0")))
    dist.barrier()

    # ---- PULL: getmem the peer's buffer into local scratch.  Both pushes
    # above left each buf holding the PEER's pattern — re-fill with our own
    # so the pull verifies fresh data end to end.
    fill_and_sync()
    dist.barrier()
    scratch = torch.zeros(n, dtype=torch.int64, device=buf.device)
    for src in (0, 1):
        if rank != src:
            ash.aclshmem_getmem(
                scratch.data_ptr(), heap_base + off, n * 8, src)
            torch.npu.synchronize()
        dist.barrier()
        if rank != src:
            want_magic = MAGIC0 if src == 0 else MAGIC1
            want_pat = PAT0 if src == 0 else PAT1
            ok = int(scratch[n - 1].item()) == want_magic and bool(
                (scratch[: n - 1].cpu() == want_pat).all().item())
            print(f"G2_RMA_pull_{src}to{1-src}_r{rank}="
                  f"{'PASS' if ok else 'FAIL'} "
                  f"magic={int(scratch[n-1].item()):#x}", flush=True)
            results.append((f"pull_{src}_to_{1-src}", ok))
        dist.barrier()

    ash.aclshmem_finalize()
    dist.destroy_process_group()
    watchdog.cancel()
    bad = [name for name, ok in results if not ok]
    print(f"G2_RMA=PASS r{rank}" if not bad else
          f"G2_RMA=FAIL r{rank} failed={bad}", flush=True)
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
