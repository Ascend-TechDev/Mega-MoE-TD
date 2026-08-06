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
    """Physical AICore count — launch grids must not exceed it.

    The count is READ from the device; a part with more cores than the one this was
    developed on is not an error condition. The previous `assert n <= 24` encoded the
    development part's core count as a correctness invariant, so on a larger part every
    kernel routed through here raised AssertionError at launch — a crash, not a slowdown.
    Measured 2026-08-06 on Ascend950DT_9582 (cube=32, vector=64): the assert fires and
    nothing runs.

    The docstring's actual invariant — "launch grids must not exceed it" — is satisfied
    by returning the physical count, which is what callers use as their grid. Grids of
    32 and 64 were exercised on that part with bit-identical results to grid 24.
    """
    return NPUUtils().get_aicore_num()


def all_gather_list(t, group):
    """all_gather a 1-D tensor into a list of per-rank copies."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return out
