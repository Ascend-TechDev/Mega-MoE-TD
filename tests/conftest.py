# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Pytest setup for the standalone Mega-MoE tutorial."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "dist: requires a multi-rank Ascend HCCL/ACLSHMEM run"
    )


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
