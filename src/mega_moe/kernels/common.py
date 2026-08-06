# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  common.py
#
#  Shared configuration + helpers for the Ascend MoE-backward triton kernels.
#  GEMM dot-tiles match 06 / the GPU backward tiling.
# ============================================================================

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from triton.backends.ascend.driver import NPUUtils

# GEMM dot-tiles for the input-grad GEMMs (step1 dispatch_fc2, step4 combine_fc1).
# BLOCK_SIZE_M is LOCKED to 64: the forward meta tiling (prepare_moe_metadata in
# _torch_forward.py) tiles tokens in BLOCK_M chunks and BOTH step1 and
# step4 consume that meta, so they must share M. K=256 (reduction tile) fills L0B
# (b-tile [256,128] bf16 = 64KB) per the Ascend 'small reduction tile -> MTE-bound'
# wiki pattern (same fix as wgrad BM=256). M=128 was tried but overflows UB with
# K=256 ([128,256]+[256,128] double-buffered = 256KB > 192KB UB) and loses the K=256
# gain at K=128 — net worse (see debug bench history).
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 256
# weight-grad tiling. BM is the reduction dim (tokens/expert, the GEMM-K of the
# wgrad). BM=256 fills L0A (a-tile [128,256] bf16 = 64KB) and L0C; the GPU's BM=64
# left L0A 1/4-full and stalled the cube (MTE-bound) at long sequences — see
# debug/bench_wgrad3.py: 2.25-2.39x on 16384 tokens, no regression at 4096.
WGRAD_BLOCK_M = 256
WGRAD_BLOCK_N = 128
WGRAD_BLOCK_K = 256


def ncore():
    """Return the device-reported physical AICore count."""
    return NPUUtils().get_aicore_num()


def validate_wgrad_launch_params(
    *,
    block_m=None,
    block_n=None,
    block_k=None,
    grid=None,
    max_grid=None,
    name_prefix="",
):
    """Validate universal wgrad tile constraints and an optional device grid bound."""
    for name, value in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
    ):
        if value is None:
            continue
        qualified_name = f"{name_prefix}{name}"
        if type(value) is not int:
            raise TypeError(f"{qualified_name} must be an integer, got {value!r}")
        if value <= 0:
            raise ValueError(f"{qualified_name} must be positive, got {value}")
        if value < 16:
            raise ValueError(
                f"{qualified_name} must be at least 16, got {value}"
            )
        if value & (value - 1):
            raise ValueError(
                f"{qualified_name} must be a power of two, got {value}"
            )

    if grid is not None:
        qualified_name = f"{name_prefix}grid"
        if type(grid) is not int:
            raise TypeError(f"{qualified_name} must be an integer, got {grid!r}")
        if grid <= 0:
            raise ValueError(f"{qualified_name} must be positive, got {grid}")

    if max_grid is not None:
        if type(max_grid) is not int or max_grid <= 0:
            raise RuntimeError(
                f"physical AICore count must be a positive integer, got {max_grid!r}"
            )
        if grid is not None and grid > max_grid:
            raise ValueError(
                f"{name_prefix}grid={grid} must not exceed physical AICore count "
                f"{max_grid}"
            )


def all_gather_list(t, group):
    """all_gather a 1-D tensor into a list of per-rank copies."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return out
