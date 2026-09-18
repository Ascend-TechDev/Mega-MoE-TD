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
import triton
import triton.language as tl
from triton.backends.ascend.driver import NPUUtils

# GEMM dot-tiles for the input-grad GEMMs (step1 dispatch_fc2, step4 combine_fc1).
# BLOCK_SIZE_M is LOCKED to 64: the forward meta tiling (prepare_moe_metadata in
# ops/_torch_forward.py) tiles tokens in BLOCK_M chunks and BOTH step1 and
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
    return validate_physical_aicore_count(NPUUtils().get_aicore_num())


def nvec():
    """Return the device-reported physical AIVector core count.

    Used as the launch grid for the pure-Vector pipeline kernels (push/reduce),
    mirroring the forward FC2-combine pipeline. Falls back to ``ncore()`` if the
    driver reports a non-positive value (older CANN on 910B1).
    """
    value = NPUUtils().get_aivector_core_num()
    if type(value) is not int or value <= 0:
        return ncore()
    return value


def validate_physical_aicore_count(value):
    """Require an exact positive device-reported physical AICore count.

    This replaces `assert n <= 24`. That bound could not be satisfied by any
    Ascend950 part actually in service: CANN 9.1.0's platform table lists
    950PR 957b/9579/957c/957d = 28, 950PR 9589/958b = 32, 950DT 958x = 32,
    950DT 959x/95Ax = 36; the only <= 24 entry is 950PR_950z = 4, which is not
    deployed. Measured on Ascend950DT_9582: get_aicore_num() = 32.

    The failure was silent rather than loud. `ncore()` is called only from the
    backward kernels, so every backward shape raised, `run_benchmark` caught the
    exception per shape and printed `[skip]`, and the benchmark still exited 0
    with `"configs": []` -- a green run that measured nothing. The forward path
    is unaffected because it uses `config.num_aicore_programs` instead.

    What is still checked: the value must be an exact positive int. A bool, a
    float, 0 or a negative would previously have flowed into a launch grid.
    """
    if type(value) is not int or value <= 0:
        raise RuntimeError(
            f"physical AICore count must be a positive integer, got {value!r}"
        )
    return value


def all_gather_list(t, group):
    """all_gather a 1-D tensor into a list of per-rank copies."""
    out = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return out


@triton.jit
def _sys_cnt_tick(dummy):
    """Read the NPU system clock (SYS_CNT, ~1 GHz on this part — measured
    977.97 ticks/us on 2026-09-18, not the ~20 MHz this comment long
    claimed — calibrate host-side against a known-duration launch).  The
    inline-asm form is the sb_rw_benchmark.py pattern, proven to lower on
    this toolchain; ``is_pure`` must be False or the compiler hoists/CSEs
    the reads away."""
    return tl.inline_asm_elementwise(
        asm="MOV $0, SYS_CNT;",
        constraints="=l,l",
        args=[dummy],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _phase_stamp(ts_ptr, slot: tl.constexpr, pid, TS_SLOTS: tl.constexpr):
    """Store one SYS_CNT reading into ``ts[pid, slot]``.

    Shared by the mega backward (MOE_MEGA_TIMING) and the single-kernel
    forward (MOE_FWD_TIMING): both stamp per-program checkpoints right after
    full barriers, so a stamp is the issuing stream's clock once every
    engine has drained; the host reduces each segment to
    ``max(end over cores) - min(start over cores)``.  TIMING=0 call sites
    compile every stamp out (dead-arg pattern)."""
    dummy = tl.arange(0, 1)
    t = _sys_cnt_tick(dummy)
    tl.store(ts_ptr + pid * TS_SLOTS + slot + dummy, t)


@triton.jit
def _phase_stamp_row(ts_row_ptr, slot: tl.constexpr, dummy):
    """Row-based _phase_stamp for UB-constrained kernels.

    The caller precomputes ``ts_ptr + pid * TS_SLOTS`` once and shares one
    ``dummy = tl.arange(0, 1)`` across every stamp site (the dummy doubles
    as the store's (1,) offset).  One arange and one scalar multiply per
    kernel instead of one per stamp — the scattered form's per-site
    auto-buffer allocations overflowed UB on the single-kernel forward
    (2026-09-17: requires 2003712 bits vs 1769472 available)."""
    t = _sys_cnt_tick(dummy)
    tl.store(ts_row_ptr + slot + dummy, t)
