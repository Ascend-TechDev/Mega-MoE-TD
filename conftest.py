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


def _worker_wrapper(rank, world_size, backend, fn, args, error_queue):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    try:
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        # Align HCCL workers before each test initializes ACLSHMEM.
        dist.barrier()
        fn(rank, world_size, *args)
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
    os.environ.setdefault(
        "MASTER_PORT", str(40000 + (os.getpid() % 20000))
    )
    context = mp.get_context("spawn")
    error_queue = context.Queue()
    spawn_context = mp.spawn(
        _worker_wrapper,
        args=(world_size, backend, fn, args, error_queue),
        nprocs=world_size,
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
