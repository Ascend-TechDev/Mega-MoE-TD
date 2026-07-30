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

# GEMM dot-tiles (match 06 / GPU backward tiling)
BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 128
# weight-grad tiling (match GPU transposed_moe_grouped_gemm)
WGRAD_BLOCK_M = 64
WGRAD_BLOCK_N = 128
WGRAD_BLOCK_K = 256


def ncore():
    """Physical AICore count — launch grids must not exceed it."""
    n = NPUUtils().get_aicore_num()
    assert n <= 24, "launch grid must not exceed physical aicore num"
    return n


def all_gather_list(t, group):
    """all_gather a 1-D tensor into a list of per-rank copies."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return out
