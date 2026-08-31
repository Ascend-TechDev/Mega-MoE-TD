# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Repository-wide pytest setup for the standalone Mega-MoE tutorial."""

import gc
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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
    mp.spawn(
        _worker_wrapper,
        args=(world_size, backend, fn, args, error_queue),
        nprocs=world_size,
        join=True,
    )

    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    if errors:
        messages = "\n".join(f"[rank {rank}] {error}" for rank, error in errors)
        pytest.fail(f"distributed worker failure:\n{messages}")


@pytest.fixture
def dist_test():
    return run_dist_test
