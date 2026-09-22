# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Repository-wide pytest setup for the standalone Mega-MoE tutorial."""

import gc
import os
import traceback
import queue
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Workers must finish within this budget; a full error_queue pipe must never
# be able to wedge the join (drained continuously below).
_DIST_TEST_TIMEOUT_S = int(os.environ.get("DIST_TEST_TIMEOUT_S", 900))

# Multi-node spawn layout (dual-node adaptation, 2026-09-22): each node runs
# its own pytest session; MMT_NNODES/MMT_NODE_RANK split the global world
# into per-node spawn groups.  Defaults keep the historical single-node path
# bit-identical (rank == spawn index == device).
_MMT_NNODES = max(1, int(os.environ.get("MMT_NNODES", "1")))
_MMT_NODE_RANK = int(os.environ.get("MMT_NODE_RANK", "0"))


def _worker_wrapper(
    local_i, global_world, backend, fn, args, error_queue,
    nproc_per_node, node_rank,
):
    # Global ACLSHMEM PE = node offset + local spawn index; the NPU device is
    # ALWAYS the local index (node 1's PEs 8..15 are its devices 0..7 — see
    # mega_moe.runtime.device).
    rank = node_rank * nproc_per_node + local_i
    if _MMT_NNODES == 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
    # Multi-node: MASTER_ADDR/MASTER_PORT were validated in the parent pytest
    # process (both nodes must rendezvous at the SAME node0 store) and are
    # inherited verbatim — never defaulted per-process here.
    try:
        torch.npu.set_device(local_i)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=global_world,
        )
        # Align HCCL workers before each test initializes ACLSHMEM.
        dist.barrier()
        fn(rank, global_world, *args)
    except Exception as error:
        # Print at raise time: the parent only drains the queue after every
        # worker exits, and a peer of a failed rank usually hangs inside a
        # collective — without this the traceback stays invisible until the
        # join deadline (or the watchdog) kills the session.
        print(f"[rank {rank}] worker exception:", flush=True)
        traceback.print_exc()
        error_queue.put((rank, error))
    finally:
        # Finalize deferred HCCL Work objects while the NPU runtime is still
        # alive.  Their destructor inserts allocator events (SetDevice on the
        # GC thread, whose current device defaults to 0); after
        # destroy_process_group/aclshmem_finalize that reopen fails with
        # error 507033 and the c10::Error thrown inside the destructor
        # aborts the worker (flaky SIGABRT seen on 4-card runs).
        gc.collect()
        torch.npu.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dist_test(fn, world_size=2, backend="hccl", args=()):
    """Run a worker function in an isolated multi-process HCCL world."""
    # Every pytest session spawns a fresh HCCL world; with the default port
    # those worlds collide on the NPU socket port 16666 (EI0020 "already
    # bound" -> HcclAllreduce failure or an outright hang) whenever a
    # previous/concurrent world's sockets linger on this shared node.  Pick
    # a per-session port range instead (inherited by the spawned workers).
    base = 20000 + (os.getpid() % 200) * 64
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", f"{base}-{base + 63}")
    # The TCPStore rendezvous port has the same shared-node collision
    # problem: torch's default 29500 is claimed by other containers, and a
    # failed bind surfaces as EADDRINUSE on rank 0 plus an HCCL
    # RootInfoDetect hang or failure on the remaining ranks.  Pick a
    # per-session port outside the HCCL range above (also inherited).
    # Multi-node is the exception: both nodes' pytest sessions must agree on
    # ONE port, so the runbook exports it and defaulting is refused.
    if _MMT_NNODES > 1:
        if os.environ.get("MASTER_ADDR", "localhost") in ("localhost", "127.0.0.1"):
            raise RuntimeError(
                "MMT_NNODES>1 requires MASTER_ADDR=<node0-IP> on both nodes"
            )
        if "MASTER_PORT" not in os.environ:
            raise RuntimeError(
                "MMT_NNODES>1 requires an explicit, identical MASTER_PORT on "
                "both nodes"
            )
    else:
        os.environ.setdefault(
            "MASTER_PORT", str(40000 + (os.getpid() % 20000))
        )
    context = mp.get_context("spawn")
    if world_size % _MMT_NNODES:
        raise ValueError(
            f"world_size={world_size} is not divisible by MMT_NNODES={_MMT_NNODES}"
        )
    nprocs = world_size // _MMT_NNODES
    error_queue = context.Queue()
    spawn_context = mp.spawn(
        _worker_wrapper,
        args=(world_size, backend, fn, args, error_queue, nprocs, _MMT_NODE_RANK),
        nprocs=nprocs,
        join=False,
    )

    # Drain while joining: a rank that puts a large error message can block
    # its queue feeder on a full pipe and then never exit, which deadlocks
    # a plain join-before-drain wait.
    errors = []

    def drain() -> None:
        while True:
            try:
                errors.append(error_queue.get_nowait())
            except queue.Empty:
                return

    deadline = time.monotonic() + _DIST_TEST_TIMEOUT_S
    try:
        while not spawn_context.join(timeout=5):
            drain()
            if time.monotonic() > deadline:
                pytest.fail(
                    f"dist workers did not finish within {_DIST_TEST_TIMEOUT_S}s"
                )
    finally:
        drain()
        # Abandoned live workers poison every later node in the session: they
        # keep their NPU contexts and the MASTER_PORT TCPStore, so the next
        # test's ranks fail to open the device (SetDevice 507033 / TsdOpen)
        # or bind the store (EADDRINUSE) and one flaky node cascades into a
        # whole failed batch.  This fires only when workers are still alive
        # (deadline raise, or peers spinning in a collective after a failed
        # rank reported through the error queue); on normal exits every
        # process is gone and the loop is a no-op.
        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
        for process in spawn_context.processes:
            process.join(15)
            if process.is_alive():
                process.kill()
            process.join()
    if errors:
        messages = "\n".join(f"[rank {rank}] {error}" for rank, error in errors)
        pytest.fail(f"distributed worker failure:\n{messages}")


@pytest.fixture
def dist_test():
    return run_dist_test
