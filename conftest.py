# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Repository-wide pytest setup for the standalone Mega-MoE tutorial."""

import os
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
        error_queue.put((rank, error))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_dist_test(fn, world_size=2, backend="hccl", args=()):
    """Run a worker function in an isolated multi-process HCCL world."""
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
    if errors:
        messages = "\n".join(f"[rank {rank}] {error}" for rank, error in errors)
        pytest.fail(f"distributed worker failure:\n{messages}")


@pytest.fixture
def dist_test():
    return run_dist_test
