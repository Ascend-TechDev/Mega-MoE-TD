# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  mega_bwd.py  —  MOE_BWD_MEGA=1: the WHOLE non-MoonEP backward (steps 1-5)
#  in ONE triton kernel launch, phases serialized by in-kernel
#  libshmem_device.barrier_all().
#
#  Proven-by-M0 structure (tests/fstage/test_mega_bwd_probes.py, w8, 950DT):
#    * mixed vector/cube scopes + barrier_all() + disable_auto_sync=True
#      publishes BOTH cube->vector GM stores and remote putmem->local loads
#      (probe 1a);
#    * FIVE literal barrier_all() phases with mixed-scope work between them
#      neither wedge nor lose increments (probe 2a);
#    * a barrier inside a DEVICE-side for-loop with an SSA value carried
#      across barriers WEDGES all ranks (probe 2b) -> every phase here is a
#      literal source block and nothing crosses a barrier as an SSA value;
#    * tl.sum() of a tl.dot INSIDE a cube scope trips a bishengir sync-solver
#      assertion (CUBE_OR_VECTOR) -> cube scopes only load/dot/store; every
#      reduction lives in a vector phase.
#
#  Phase map (mirrors ops.backward.moe_backward_triton's 5-op order):
#
#    P1   dispatch A2A (vec, putmem+signal) + fc2 dgrad (cube, dl.wait)   ]
#    B1   barrier_all  — publish P1's remote puts + grad_swiglu           ] step 1
#    P2   swiglu/situ backward (vec, ungated)                             ]
#    P3   fc2 wgrad (cube, BM=64 direct transposed read of peer_mem)      ] step2+3
#    B2   barrier_all  — all ranks done READING peer_mem (P3) before any
#                        rank's P4b may overwrite it; publishes P2 outputs
#    P4a  fc1 dgrad GEMM (cube) -> hidden_buf                             ]
#    B3   barrier_all  — local cube->vector hidden_buf handoff            ] step 4
#    P4b  reverse A2A push (vec, sub_vec0-gated dl.symm_at remote stores) ]
#      ∥ P5a  fc1 wgrad first half (cube) — the P2∥P3 adjacent-scope      ] step 5
#             concurrency recipe; P5 needs only B2's dAB, and msprof
#             showed the vector engine ~97% idle across the kernel
#    B4   barrier_all  — cross-rank: all pushes landed before any reduce  ]
#    P4c  topk reduce (vec) -> grad_hidden + grad_routing_weights         ]
#      ∥ P5b  fc1 wgrad second half (cube) — no trailing barrier: after B4
#             no rank writes another rank's memory, so programs exit when
#             their own reduce + wgrad remainder finish
#
#  MoonEP (use_moonep) rides the same launch: P1 runs a second fc2-dgrad
#  sweep against the replica down table and P4a's GEMM reads the replica
#  gate/up table past tile_home_bound (both dual-table cuts are the standalone
#  kernels' pattern, inlined).  When the caller lends a grad_transport, the
#  M3 ReplicaGradTransport chain rides as tail phases:
#
#    B5   barrier_all  — publish the physical grad_fc1 (P5) / grad_fc2 (P3)
#    P6a  seed+sink (vec): fp32 seed from the home segments + this rank's
#          replica segments sunk into its own symmetric slots (transposed)
#    B6   barrier_all  — = transport barrier #1 (sunk slots published)
#    P6b  owner-pull (vec): by-home getmem + fp32 accumulate, (peer,slot)
#          order preserved -> bit-identical to the fused transport kernel
#
#  (the seed/sink are IN-kernel because the host copies would read tensors
#  this same launch only produces at B5; post_sink_hook cannot fire and is
#  bypassed — sunk/reduced are still set.  The transport's THIRD stage —
#  zeroing the consumed slots — runs HOST-side, stream-ordered behind the
#  launch: it is purely local (all remote getmem readers retire at the
#  kernel exit), so an in-kernel P6c tail phase bought nothing and its
#  zero stores proved unreliable on this backend — w8 moderate-wide,
#  910B1 2026-09-09: residual 256..57k elements per slot, run-varying,
#  not the sunk values — while the same zeroing as a host op never missed.)
#
#  MOE_MEGA_FUSE_P4=1 replaces the whole P4a/B3/P4b run above with ONE
#  fused tile loop: per program, per strided m-tile — [cube: all n-tiles of
#  tile_m + fence + signal own slot] [vec: wait own slot + push tile_m's
#  rows].  Self-produce-self-push makes B3 vanish (the handoff is
#  intra-program) and iteration i+1's GEMM overlaps iteration i's push —
#  the structural borrow of MoonEP's AIC/AIV wave pipeline; on A5's
#  independent engine arrays this is true concurrent cube-GEMM/transport
#  overlap.  Probe 4 (test_mega_bwd_probes.py) gates the scope-alternation-
#  in-loop + self-signal chain; the barriered form above is the fallback.
#
#  Every program reaches every barrier unconditionally (phase work is behind
#  constexpr PHASE flags / strided task-range bounds, never an early
#  return) — the replica_grad_reduce contract.
#
#  Disabled-knob contract: under MOE_BWD_MEGA=1 the MOE_WGRAD_TRITON /
#  MOE_WGRAD_TORCH / MOE_FUSED_SWIGLU_WGRAD / MOE_BWD_DUAL_STREAM /
#  MOE_BWD_WGRAD_TAIL / MOE_BWD_COMBINE_SERIAL / MOE_BWD_STAGE_TIMING
#  orchestrator knobs are inert (the early return in ops/backward.py skips
#  them). MOE_DISPATCH_GEMM_* / MOE_COMBINE_GEMM_* / MOE_COMBINE_PUSH_BN /
#  MOE_FUSED_WGRAD_BLOCK_M tiles still apply (the kernel reuses those
#  getters), plus the mega-local MOE_MEGA_WGRAD_BN / MOE_MEGA_WGRAD_BK /
#  MOE_MEGA_WGRAD_NS wgrad tile knobs, MOE_MEGA_TILE_B3=1 (replace the
#  B3 barrier with per-(tile,n) readiness signals; the local cube->vector
#  fence/signal/dl.wait chain it rests on is proven by probes 3/4,
#  w8 910B1) and MOE_MEGA_FUSE_P4=1 (the P4a+P4b self-produce-self-push
#  fusion — takes precedence over MOE_MEGA_TILE_B3).
#  MoonEP: use_moonep saved dicts are supported with the dual weight tables
#  always on; the P6 grad_reduce tail turns on iff the caller lends a
#  grad_transport (ops.backward passes it whenever the forward lent the
#  replica tables — the M3 path).  The transport's own knobs
#  (transport_staging_chunk_bytes / acc_block) carry over as GU_CHUNK /
#  DN_CHUNK / ACC_BLK / BLK6; MOONEP_GRAD_TRANSPORT=legacy is inert here
#  (the in-kernel chain replaces both transport modes).
# ============================================================================

import os

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .common import ncore, NPUUtils
from .dispatch_fc2_bwd import (
    _prepare_dispatch_fc2_bwd,
    _dispatch_grad_source_tiles,
    _fc2_bwd_gemm_merged_tiles_wait,
    _dispatch_gemm_tile,
    _ensure_bwd_signal_mem,
)
from .combine_fc1_bwd import (
    _combine_static_maps,
    _gemm_tile_maps,
    _combine_gemm_tile,
    _push_block,
    GATE_PAD,
)
from .fused_swiglu_bwd_fc2_wgrad import FUSED_WBM, FUSED_WBN, FUSED_WBK
from .replica_grad_reduce import (
    build_owner_pull_descriptors_by_home,
    zero_consumed_replica_slots,
)


# ============================================================================
# jit phase helpers (scopes live in the mega kernel, the dispatch_fc2_bwd
# pattern — helpers are scope-free bodies the caller wraps)
# ============================================================================
@triton.jit
def _mega_swiglu_bwd_row(
    row,
    dC_ready_ptr, dC_stride,
    AB_ptr, AB_stride,
    ffn,
    scale_ptr,
    dAB_ptr,
    dscale_ptr,
    situ_beta, situ_linear_beta,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """One row of the step-2 swiglu/situ backward (kernels/swiglu_bwd.py:32-70
    verbatim); dC_ready_ptr is grad_swiglu either behind a consume_token (the
    TILE_B1 windowed consumer) or the plain pointer."""
    offs = tl.arange(0, BLOCK_SIZE)
    r64 = row.to(tl.int64)
    a_ptr = AB_ptr + r64 * AB_stride          # gate half
    b_ptr = a_ptr + ffn                        # up half
    dc_ptr = dC_ready_ptr + r64 * dC_stride
    mask = offs < ffn
    dc = tl.load(dc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == 0:
        sigmoid_a = tl.sigmoid(a)
        act_a = a * sigmoid_a
        dact_a = act_a * (1 - sigmoid_a) + sigmoid_a
        v = b
        dv = 1.0
    else:
        t = 2.0 * tl.sigmoid(2.0 * a / situ_beta) - 1.0
        s = tl.sigmoid(a)
        act_a = situ_beta * t * s
        dact_a = (1.0 - t * t) * s + situ_beta * t * s * (1.0 - s)
        if HAS_LINEAR_BETA:
            tu = 2.0 * tl.sigmoid(2.0 * b / situ_linear_beta) - 1.0
            v = situ_linear_beta * tu
            dv = 1.0 - tu * tu
        else:
            v = b
            dv = 1.0
    sc = tl.load(scale_ptr + r64)
    da = dc * dact_a * v * sc
    db = dc * act_a * dv * sc
    tl.store(dAB_ptr + r64 * AB_stride + offs, da.to(AB_ptr.dtype.element_ty), mask=mask)
    tl.store(dAB_ptr + r64 * AB_stride + ffn + offs, db.to(AB_ptr.dtype.element_ty), mask=mask)
    tl.store(dscale_ptr + r64, tl.sum(act_a * v * dc))


@triton.jit
def _mega_swiglu_bwd(
    pid, nprogs,
    dC_ptr, dC_stride,
    AB_ptr, AB_stride,
    ffn,
    scale_ptr,
    dAB_ptr,
    dscale_ptr,
    n_rows,
    situ_beta, situ_linear_beta,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """Step-2 body (kernels/swiglu_bwd.py:32-70 verbatim): ungated vector
    work, rows partitioned by pid over the mega grid (= ncore())."""
    for row in range(pid, n_rows, nprogs):
        _mega_swiglu_bwd_row(
            row, dC_ptr, dC_stride, AB_ptr, AB_stride, ffn,
            scale_ptr, dAB_ptr, dscale_ptr,
            situ_beta, situ_linear_beta,
            BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA)


@triton.jit
def _mega_swiglu_bwd_windowed(
    pid, nprogs,
    dC_ptr, dC_stride,
    AB_ptr, AB_stride,
    ffn,
    scale_ptr,
    dAB_ptr,
    dscale_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr,
    b1_signal_ptr, b1_epoch,
    num_n_tiles,
    situ_beta, situ_linear_beta,
    EPR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    """MOE_MEGA_TILE_B1=1 consumer for P2: WINDOW-strided ownership.  The P1
    cube GEMM signals every (expert, n_tile, m_window) tile of grad_swiglu it
    stores; this walk takes every nprogs-strided GLOBAL window, merged-waits
    its num_n_tiles b1 slots (the P1 production idiom), consume_tokens, then
    processes the window's rows.  max_win is re-derived from recv_per_expert
    exactly as the producer does — same formula, no host table.  Must run
    inside a vector scope behind the sub_vec_id()==0 gate (dl.wait /
    consume_token must not run twice per program)."""
    max_win = 1
    for e0 in range(EPR):
        sz0 = tl.load(recv_per_expert_ptr + e0)
        max_win = tl.maximum(max_win, tl.cdiv(sz0, BLOCK_M))
    gw = 0  # global window index over the ragged (expert, window) space
    for e in range(EPR):
        size = tl.load(recv_per_expert_ptr + e)
        off = tl.load(recv_expert_offs_ptr + e)
        nwin = tl.cdiv(size, BLOCK_M)
        for w in range(nwin):
            if gw % nprogs == pid:
                token = 0
                for nt in range(num_n_tiles):
                    token += dl.wait(
                        b1_signal_ptr + ((e * num_n_tiles + nt) * max_win + w) * 16,
                        1, "gpu", "acquire", waitValue=b1_epoch)
                dC_ready = dl.consume_token(dC_ptr, token)
                row0 = off + w * BLOCK_M
                rows = tl.minimum(BLOCK_M, size - w * BLOCK_M)
                for r in range(rows):
                    _mega_swiglu_bwd_row(
                        row0 + r, dC_ready, dC_stride, AB_ptr, AB_stride, ffn,
                        scale_ptr, dAB_ptr, dscale_ptr,
                        situ_beta, situ_linear_beta,
                        BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA)
            gw += 1


@triton.jit
def _mega_wgrad_sweep(
    pid, ncores,
    grad_out_ptr, stride_outm, stride_outn,   # [M, N] expert-major (NO host transpose)
    orig_in_ptr, stride_om, stride_ok,        # [M, K]
    grad_w_ptr, stride_we, stride_wn, stride_wk,   # [E, N, K] out
    split_size_cum_per_expert_ptr, expert_counts_ptr,
    N, K, num_tiles_n, num_tiles_k, task_begin, task_end, max_rows,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    WAIT_DISP: tl.constexpr,
    signal_mem_ptr, signal_epoch_val, recv_counts_re_ptr,
    WORLD_SIZE_C: tl.constexpr, EPR_C: tl.constexpr,
    MAX_BWD_TILES_C: tl.constexpr, TILE_M_C: tl.constexpr,
):
    """Grouped weight-grad GEMM ``grad_w[E,N,K] = grad_out^T @ orig_in``.

    Ported from the proven cube scope of fused_swiglu_bwd_fc2_wgrad (BM=64
    direct [M,N] transposed load) with N/K/tile counts downgraded from
    constexpr to RUNTIME — the constexpr-N/K form is the documented Kimi
    5-minute-compile pathology and would also recompile whenever routing
    changes the row counts.

    STRIDED task ownership (task = task_begin + pid + i*ncores): consecutive
    task ids belong to the same expert, so the original contiguous per-core
    blocks left the low-pid cores far busier than the high-pid ones under
    skewed routing (msprof PipeUtilization on kimi t4k: cube busy 96.7ms on
    block0 vs 52.5ms on block2, staggered kernel exits). Interleaving
    spreads every expert's tiles over all cores.

    task_begin/task_end carve the sweep's task range so the fc1 wgrad can
    ride the P4b/P4c vector windows in two halves (see the kernel body).

    WAIT_DISP=1 (MOE_MEGA_TILE_B1=1, the P3 fc2-wgrad call only): grad_out IS
    peer_mem, and with the B1 barrier gone nothing else publishes the remote
    dispatch — so each task merged-waits EVERY (source, tile) slot feeding
    its expert's rows (the P1 acquisition restricted to the whole-expert
    range; every task reads all of the expert's rows) and consume_tokens
    before the m-loop.  The dispatch slots are plain SET values — P1's GEMM
    already consumed them once; a second wait/consume on the same epoch is
    the re-read (validated on the TILE_B1 functional gate).  Must run inside
    a cube scope; no reductions here (cube-scope dot reductions trip the
    bishengir CUBE_OR_VECTOR assertion)."""
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)
    for task in range(task_begin + pid, task_end, ncores):
        e = task // (num_tiles_n * num_tiles_k)
        rem = task - e * (num_tiles_n * num_tiles_k)   # % w/o modulo op
        tn = rem // num_tiles_k
        tk = rem - tn * num_tiles_k                    # % w/o modulo op
        split_begin = tl.load(split_size_cum_per_expert_ptr + e)
        split_size = tl.load(expert_counts_ptr + e)
        grad_out_base = grad_out_ptr
        if WAIT_DISP:
            token = 0
            for source_id in range(WORLD_SIZE_C):
                source_size = tl.load(
                    recv_counts_re_ptr + source_id * EPR_C + e)
                if source_size > 0:
                    last_source_tile = (source_size - 1) // TILE_M_C
                    for source_tile in range(last_source_tile + 1):
                        token += dl.wait(
                            signal_mem_ptr
                            + ((source_id * EPR_C + e) * MAX_BWD_TILES_C
                               + source_tile) * 16,
                            1, "gpu", "acquire", waitValue=signal_epoch_val)
            grad_out_base = dl.consume_token(grad_out_ptr, token)
        n_start = tn * BLOCK_N
        k_start = tk * BLOCK_K
        nmask = (n_start + offs_n) < N
        kmask = (k_start + offs_k) < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        # m bounded by the HOST-passed max_rows (>=1), never by the loaded
        # split_size: a load-derived trip count inside this kernel's task
        # loop miscompiles on this backend (w2, 910B1 2026-09-09 — zero-count
        # experts accumulated garbage rows and the count loads themselves
        # returned 0 where the table held 128, shifting with unrelated
        # constexpr changes).  Runtime-arg trip counts (this bound, same
        # shape as _mega_combine_gemm's K-loop) and loaded values as MASKS
        # only (mmask below) are both proven-safe idioms here.
        for m in tl.range(0, max_rows, BLOCK_M, num_stages=NUM_STAGES):
            mm = m + offs_m
            mmask = mm < split_size
            # DO NOT touch this address chain: for late/small experts
            # split_begin + max_rows runs past the buffers and this backend's
            # masked loads trap aicore on an illegal base address (507015, w2
            # hot-expert probe, 910B1 2026-09-09) — but BOTH clamp forms tried
            # on row64 (tl.where on the load-derived mmask, and an arg-derived
            # tl.minimum row bound) DETERMINISTICALLY faulted the whole
            # mixed-scope kernel (fftsplus aicore + aivector, identical pc
            # across runs, w8 moderate-wide 2026-09-09).  This loop's codegen
            # is bistable and the bare add is its only proven-good shape; the
            # OOB is fixed HOST-side instead — the wrapper pads the read
            # buffers (grad_out/orig_in/hidden_buf) with max_rows extra rows
            # so masked-lane addresses stay mapped (peer_mem relies on its
            # recv-budget slack, total_recv + max_rows <= budget rows).
            row64 = (split_begin + mm).to(tl.int64)
            a_off = row64[None, :] * stride_outm + (n_start + offs_n[:, None]) * stride_outn
            a = tl.load(grad_out_base + a_off, mask=nmask[:, None] & mmask[None, :], other=0.0)
            b_off = row64[:, None] * stride_om + (k_start + offs_k[None, :]) * stride_ok
            b = tl.load(orig_in_ptr + b_off, mask=mmask[:, None] & kmask[None, :], other=0.0)
            acc += tl.dot(a, b)
        c_off = (e.to(tl.int64) * stride_we
                 + (n_start + offs_n[:, None]) * stride_wn
                 + (k_start + offs_k[None, :]) * stride_wk)
        tl.store(grad_w_ptr + c_off, acc.to(grad_w_ptr.dtype.element_ty),
                 mask=nmask[:, None] & kmask[None, :])


@triton.jit
def _mega_combine_gemm(
    pid, ncores,
    inp_ptr, stride_im, stride_ik,          # grad_fc1_output [M, 2*ffn]
    weight_ptr, stride_we, stride_wk, stride_wn,   # ONE table [E, 2*ffn, H] (K, N)
    expert_base,                            # table re-base (0 / MoonEP home_base)
    hidden_buf_ptr,                         # [M, H] out (row stride = N)
    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
    N, K, num_tiles_n, num_tiles_m,
    task_begin, task_end,
    signal_mem_ptr, signal_epoch,           # B3 readiness slots (SIGNAL_ON)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SIGNAL_ON: tl.constexpr, LOCAL_RANK: tl.constexpr,
):
    """fc1 input-grad GEMM ``hidden_buf[m, :] = grad_fc1_output[m, :] @
    weight[e]`` over the global task range [task_begin, task_end) against ONE
    weight table re-based at expert_base (bounds RUNTIME — the constexpr-bound
    production variant would recompile whenever routing moves a tile).
    Persistent STRIDED task partition (same imbalance fix as
    _mega_wgrad_sweep), from _kernel_combine_fc1_bwd_gemm_group
    (combine_fc1_bwd.py:114-171).  The MoonEP dual weight table is TWO calls
    of this helper (home tasks then replica tasks — tiles are expert-major,
    so the split is a contiguous task-range cut), NOT a runtime select on the
    weight pointer: an arith.select on pointers/strides inside the GEMM loop
    does not lower (CANN TritonToUnstructure fails, w2 910B1). Must run
    inside a cube scope.

    SIGNAL_ON=1 (MOE_MEGA_TILE_B3=1) fires a local readiness signal after
    each (tile_m, tile_n) task's store lands: fence() orders the FixPipe
    store before signal_op SET (probe 3, w8 910B1 2026-09-08), so the P4b
    push can dl.wait per M-tile instead of crossing the B3 barrier. Slot
    layout tile_m*num_tiles_n + tile_n matches the push-side merged wait."""
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    group_tiles = num_tiles_m
    for task_id in range(task_begin + pid, task_end, ncores):
        tile_m = task_id % group_tiles
        tile_n = task_id // group_tiles
        expert_id = tl.load(tile_expert_ptr + tile_m)
        row_start = tl.load(tile_row0_ptr + tile_m)
        rem = tl.load(tile_rows_ptr + tile_m)
        n_start = tile_n * BLOCK_N
        mm = om < rem
        mn = on_ < (N - n_start)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        wb = (expert_id.to(tl.int64) - expert_base) * stride_we
        row_base = row_start.to(tl.int64) + om.to(tl.int64)
        for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            mk = ok < (K - ks)
            ao = row_base[:, None] * stride_im + (ks + ok[None, :]) * stride_ik
            a = tl.load(inp_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
            bo = wb + (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
            b = tl.load(weight_ptr + bo, mask=mk[:, None] & mn[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = row_base[:, None] * N + (n_start + on_[None, :])
        tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty),
                 mask=mm[:, None] & mn[None, :])
        if SIGNAL_ON:
            libshmem_device.fence()
            libshmem_device.signal_op(
                signal_mem_ptr + (tile_m * num_tiles_n + tile_n) * 16,
                signal_epoch, libshmem_device.ACLSHMEM_SIGNAL_SET, LOCAL_RANK)


@triton.jit
def _mega_gemm_mtile(
    tile_m,
    inp_ptr, stride_im, stride_ik,          # grad_fc1_output [M, 2*ffn]
    weight_ptr, stride_we, stride_wk, stride_wn,   # ONE table [E, 2*ffn, H] (K, N)
    expert_base,                            # table re-base (0 / MoonEP home_base)
    hidden_buf_ptr,                         # [M, H] out (row stride = N)
    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
    N, K, num_tiles_n,
    signal_mem_ptr, signal_epoch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, LOCAL_RANK: tl.constexpr,
):
    """FUSE_P4 cube body: ALL n-tiles of ONE m-tile (the same math as
    _mega_combine_gemm restricted to a single tile_m, so every n-tile of the
    row-block lands from THIS program), then one fence + ONE readiness signal
    on the tile's own slot ``tile_m``.  Self-produce-self-push: the adjacent
    vector scope (_mega_push_mtile) waits exactly this slot, so no cross-
    program B3 handoff exists.  Slot layout: one slot per m-tile (each slot
    written once per launch — SET epoch semantics hold).  ONE weight table
    re-based at expert_base: the MoonEP dual table is TWO m-tile loops at the
    call site (home range then replica range — same no-runtime-select rule as
    _mega_combine_gemm).  Must run inside a cube scope; probe 4 gates the
    scope-alternation-in-loop form."""
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    expert_id = tl.load(tile_expert_ptr + tile_m)
    row_start = tl.load(tile_row0_ptr + tile_m)
    rem = tl.load(tile_rows_ptr + tile_m)
    mm = om < rem
    wb = (expert_id.to(tl.int64) - expert_base) * stride_we
    row_base = row_start.to(tl.int64) + om.to(tl.int64)
    for tile_n in range(num_tiles_n):
        n_start = tile_n * BLOCK_N
        mn = on_ < (N - n_start)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            mk = ok < (K - ks)
            ao = row_base[:, None] * stride_im + (ks + ok[None, :]) * stride_ik
            a = tl.load(inp_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
            bo = wb + (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
            b = tl.load(weight_ptr + bo, mask=mk[:, None] & mn[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = row_base[:, None] * N + (n_start + on_[None, :])
        tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty),
                 mask=mm[:, None] & mn[None, :])
    libshmem_device.fence()
    libshmem_device.signal_op(
        signal_mem_ptr + tile_m * 16, signal_epoch,
        libshmem_device.ACLSHMEM_SIGNAL_SET, LOCAL_RANK)


@triton.jit
def _mega_push_mtile(
    tile_m,
    hidden_buf_ptr,
    write_rank_by_src_ptr, write_off_by_src_ptr,
    peer_mem_ptr,
    grad_gate_ptr,
    tile_row0_ptr, tile_rows_ptr,
    signal_mem_ptr, signal_epoch,
    H_push,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    """FUSE_P4 vector body (behind sub_vec_id()==0): wait THIS program's own
    signal for tile_m (already satisfied when the adjacent cube scope
    finished — the wait is the visibility fence, not a stall), consume_token,
    push the tile's rows.  Row work is _mega_push_rows restricted to one
    tile; dl.wait/consume_token per loop iteration is the P1 production
    idiom (dispatch_fc2_bwd.py:327-333, one consume per window, many per
    program).  Must run inside a vector scope behind the sub_vec_id()==0
    gate."""
    ovp = tl.arange(0, BLOCK_N_PUSH)
    row_stride = H_push + GATE_PAD
    row_start = tl.load(tile_row0_ptr + tile_m)
    rows = tl.load(tile_rows_ptr + tile_m)
    token = dl.wait(signal_mem_ptr + tile_m * 16, 1, "gpu", "acquire",
                    waitValue=signal_epoch)
    ready_ptr = dl.consume_token(hidden_buf_ptr, token)
    for r in range(rows):
        sp = row_start + r
        sp64 = sp.to(tl.int64)
        dst_rank = tl.load(write_rank_by_src_ptr + sp64)
        dst_off = tl.load(write_off_by_src_ptr + sp64).to(tl.int64)
        dst_base = dl.symm_at(peer_mem_ptr, dst_rank) + dst_off * row_stride
        for ns in range(0, H_push, BLOCK_N_PUSH):
            mask = ovp < (H_push - ns)
            val = tl.load(ready_ptr + sp64 * H_push + (ns + ovp),
                          mask=mask, other=0.0)
            tl.store(dst_base + ns + ovp, val, mask=mask)
        # pack the gate grad as the trailing channel of this row
        tl.store(dst_base + H_push, tl.load(grad_gate_ptr + sp))


@triton.jit
def _mega_push_rows(
    pid, num_progs,
    hidden_buf_ptr,
    write_rank_by_src_ptr, write_off_by_src_ptr,
    peer_mem_ptr,
    grad_gate_ptr,
    H_push, M,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    """Reverse-A2A push (expert->home) over src_pos in [0, M) — RUNTIME
    bounds port of _kernel_combine_fc1_bwd_push_group
    (combine_fc1_bwd.py:243-278). Each peer_mem row packs [hidden (H) | gate
    (GATE_PAD)]. Must run inside a vector scope behind the sub_vec_id()==0
    gate."""
    ovp = tl.arange(0, BLOCK_N_PUSH)
    row_stride = H_push + GATE_PAD
    for src_pos in range(pid, M, num_progs):
        sp64 = src_pos.to(tl.int64)
        dst_rank = tl.load(write_rank_by_src_ptr + sp64)
        dst_off = tl.load(write_off_by_src_ptr + sp64).to(tl.int64)
        dst_base = dl.symm_at(peer_mem_ptr, dst_rank) + dst_off * row_stride
        for ns in range(0, H_push, BLOCK_N_PUSH):
            mask = ovp < (H_push - ns)
            val = tl.load(hidden_buf_ptr + sp64 * H_push + (ns + ovp),
                          mask=mask, other=0.0)
            tl.store(dst_base + ns + ovp, val, mask=mask)
        # pack the gate grad as the trailing channel of this row
        tl.store(dst_base + H_push, tl.load(grad_gate_ptr + sp64))


@triton.jit
def _mega_push_rows_tiled(
    pid, num_progs,
    hidden_buf_ptr,
    write_rank_by_src_ptr, write_off_by_src_ptr,
    peer_mem_ptr,
    grad_gate_ptr,
    tile_row0_ptr, tile_rows_ptr,
    signal_mem_ptr, signal_epoch,
    num_tiles_m, num_tiles_n,
    H_push, M,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    """MOE_MEGA_TILE_B3=1 variant of _mega_push_rows: iterate M-tiles in
    strided order, dl.wait EVERY (tile_m, tile_n) readiness slot of the tile
    (the P1 merged-window idiom — no atomic last-finisher counting), then push
    that tile's rows. Producer-first shape: the cube scope signals each task
    as its store lands and never waits, so there is no circular wait (probe 3
    proved the local cube->vector fence/signal/dl.wait chain in a vector
    scope, w8 910B1 2026-09-08). Must run inside a vector scope behind the
    sub_vec_id()==0 gate (dl.wait/consume_token must not run twice)."""
    ovp = tl.arange(0, BLOCK_N_PUSH)
    row_stride = H_push + GATE_PAD
    for tile_m in range(pid, num_tiles_m, num_progs):
        row_start = tl.load(tile_row0_ptr + tile_m)
        rows = tl.load(tile_rows_ptr + tile_m)
        token = 0
        for nt in range(num_tiles_n):
            token += dl.wait(
                signal_mem_ptr + (tile_m * num_tiles_n + nt) * 16,
                1, "gpu", "acquire", waitValue=signal_epoch)
        ready_ptr = dl.consume_token(hidden_buf_ptr, token)
        for r in range(rows):
            sp = row_start + r
            sp64 = sp.to(tl.int64)
            dst_rank = tl.load(write_rank_by_src_ptr + sp64)
            dst_off = tl.load(write_off_by_src_ptr + sp64).to(tl.int64)
            dst_base = dl.symm_at(peer_mem_ptr, dst_rank) + dst_off * row_stride
            for ns in range(0, H_push, BLOCK_N_PUSH):
                mask = ovp < (H_push - ns)
                val = tl.load(ready_ptr + sp64 * H_push + (ns + ovp),
                              mask=mask, other=0.0)
                tl.store(dst_base + ns + ovp, val, mask=mask)
            # pack the gate grad as the trailing channel of this row
            tl.store(dst_base + H_push, tl.load(grad_gate_ptr + sp))


@triton.jit
def _mega_reduce(
    pid, num_progs,
    inv_sort_idxs_ptr,
    peer_mem_ptr,
    output_ptr,
    grad_routing_ptr,
    B, topk, H_push,
    stride_om, stride_on,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD: tl.constexpr,
):
    """Topk-sum reduce peer_mem -> grad_hidden + gather the packed gate
    channel -> grad_routing_weights. Verbatim port of
    _kernel_combine_fc1_bwd_reduce (combine_fc1_bwd.py:349-386); runtime
    bounds already. Must run inside a vector scope (ungated)."""
    ovr = tl.arange(0, BLOCK_N_PUSH)
    row_stride = H_push + GATE_PAD
    for ti in range(pid, B, num_progs):
        ti64 = ti.to(tl.int64)
        # gather the per-(token,slot) gate channel packed at row offset H_push
        for j in range(topk):
            fi = (ti * topk + j).to(tl.int64)
            sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
            tl.store(grad_routing_ptr + fi,
                     tl.load(peer_mem_ptr + sp * row_stride + H_push))
        for ns in range(0, H_push, BLOCK_N_PUSH):
            mask = ovr < (H_push - ns)
            acc = tl.zeros((BLOCK_N_PUSH,), dtype=tl.float32)
            for j in range(topk):
                fi = ti * topk + j
                sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                acc += tl.load(peer_mem_ptr + sp * row_stride + (ns + ovr),
                               mask=mask, other=0.0)
            oo = ti64 * stride_om + (ns + ovr) * stride_on
            tl.store(output_ptr + oo, acc.to(output_ptr.dtype.element_ty), mask=mask)


# ============================================================================
# MoonEP M3 grad_reduce phases (GRAD_REDUCE) — the ReplicaGradTransport chain
# (sink -> barrier -> owner-pull -> barrier -> zero) inlined as mega tail
# phases; the host-side seed/sink copies cannot ride this launch any other
# way, because they read grad tensors the SAME launch only produces at B5.
# ============================================================================
@triton.jit
def _mega_grad_transpose(
    dst_ptr, src_ptr,
    H_dim, F2,
    TM: tl.constexpr, TN: tl.constexpr,
):
    """dst[H, F2] = transpose(src[F2, H]), blocked; the store casts to dst's
    element type, so one body serves the fp32 accumulator seed AND the bf16
    slot sink.  NO tl.trans: this backend's TritonToUnstructure pass fails to
    lower it here (w2 compile, 910B1) and no other kernel in the repo uses
    it — the transposed read is expressed as index arithmetic instead (the
    P4c reduce's strided gather/scatter pattern, masked blocked tiles)."""
    om = tl.arange(0, TM)   # H axis of dst
    on_ = tl.arange(0, TN)  # F2 axis of dst
    for h0 in range(0, H_dim, TM):
        hm = (h0 + om) < H_dim
        for f0 in range(0, F2, TN):
            fn = (f0 + on_) < F2
            v = tl.load(
                src_ptr + (f0 + on_)[None, :] * H_dim + (h0 + om)[:, None],
                mask=hm[:, None] & fn[None, :], other=0.0)
            tl.store(dst_ptr + (h0 + om)[:, None] * F2 + (f0 + on_)[None, :],
                     v, mask=hm[:, None] & fn[None, :])


@triton.jit
def _mega_grad_accum(
    acc_ptr, source_ptr,
    home, chunk_start, count, elems,
    ACC_BLK: tl.constexpr,
):
    """acc[home, chunk_start:chunk_start+count] += source[0:count] in fp32,
    contiguous walk (the fused transport's _fused_accumulate_chunk)."""
    offs = tl.arange(0, ACC_BLK)
    base = home.to(tl.int64) * elems + chunk_start
    for start in range(0, count, ACC_BLK):
        local = start + offs
        m = local < count
        v = tl.load(source_ptr + local, mask=m, other=0.0).to(tl.float32)
        o = base + local.to(tl.int64)
        tl.store(acc_ptr + o, tl.load(acc_ptr + o, mask=m, other=0.0) + v, mask=m)


@triton.jit
def _mega_grad_seed_sink(
    pid, ncores,
    grad_fc1_ptr,              # physical [2*epn, 2F, H] (P5 output, contiguous)
    grad_fc2_ptr,              # physical [2*epn, H, F] (P3 output, contiguous)
    acc_gate_up_ptr,           # fp32 [epn, H, 2F] out (table layout)
    acc_down_ptr,              # fp32 [epn, H, F] out
    gate_up_slot_ptr,          # bf16 symmetric [epn, H, 2F] (borrowed table)
    down_slot_ptr,             # bf16 symmetric [epn, H, F]
    consumed_ptr, consumed_count,
    EPN, H_dim, F_dim,
    TM: tl.constexpr, TN: tl.constexpr, BLK: tl.constexpr,
):
    """P6a: seed the fp32 accumulators with the HOME segments (gate/up
    transposed into the table layout, down cast in place) and SINK this
    rank's replica segments into its own symmetric slots (gate/up transposed,
    down contiguous).  Experts and slots strided across programs — pure local
    copies, no cross-rank dependency.  Must run inside a vector scope."""
    n_down = H_dim * F_dim
    offs = tl.arange(0, BLK)
    # seed: home expert e is physical row e; acc rows are per-expert disjoint
    for e in range(pid, EPN, ncores):
        e64 = e.to(tl.int64)
        _mega_grad_transpose(
            acc_gate_up_ptr + e64 * H_dim * (2 * F_dim),
            grad_fc1_ptr + e64 * (2 * F_dim) * H_dim,
            H_dim, 2 * F_dim, TM, TN)
        for start in range(0, n_down, BLK):
            m = (start + offs) < n_down
            v = tl.load(grad_fc2_ptr + e64 * n_down + start + offs,
                        mask=m, other=0.0)
            tl.store(acc_down_ptr + e64 * n_down + start + offs,
                     v.to(tl.float32), mask=m)
    # sink: consumed slot s carries the gradient of physical expert EPN + s
    for i in range(pid, consumed_count, ncores):
        slot = tl.load(consumed_ptr + i)
        slot64 = slot.to(tl.int64)
        phys64 = (EPN + slot).to(tl.int64)   # int64: 2*epn*2F*H can top 2^31
        _mega_grad_transpose(
            gate_up_slot_ptr + slot64 * H_dim * (2 * F_dim),
            grad_fc1_ptr + phys64 * (2 * F_dim) * H_dim,
            H_dim, 2 * F_dim, TM, TN)
        for start in range(0, n_down, BLK):
            m = (start + offs) < n_down
            v = tl.load(grad_fc2_ptr + phys64 * n_down + start + offs,
                        mask=m, other=0.0)
            tl.store(down_slot_ptr + slot64 * n_down + start + offs, v, mask=m)


@triton.jit
def _mega_grad_owner_pull(
    pid, ncores,
    acc_gate_up_ptr, acc_down_ptr,
    gate_up_slot_ptr, down_slot_ptr,
    desc_peer_ptr, desc_slot_ptr, home_offsets_ptr,
    staging_gu_ptr, staging_dn_ptr,   # [ncores, chunk] per-program staging rows
    gu_elems, dn_elems,
    EPN: tl.constexpr,
    GU_CHUNK: tl.constexpr, DN_CHUNK: tl.constexpr, ACC_BLK: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
):
    """P6b: owner-pull — HOME experts strided across programs; each expert's
    (peer, slot)-ordered descriptors are pulled chunk-wise (getmem, blocking)
    and fp32-accumulated onto the seed.  Per-expert accumulator rows are
    disjoint across programs, so the fp32 order is bit-identical to the fused
    transport kernel's.  The self-owned slot branch is defensive only (the
    planner forbids self-copies).  Must run inside a vector scope."""
    gu_row = staging_gu_ptr + pid.to(tl.int64) * GU_CHUNK
    dn_row = staging_dn_ptr + pid.to(tl.int64) * DN_CHUNK
    for home in range(pid, EPN, ncores):
        start = tl.load(home_offsets_ptr + home)
        end = tl.load(home_offsets_ptr + home + 1)
        for ordinal in range(start, end):
            peer = tl.load(desc_peer_ptr + ordinal)
            slot = tl.load(desc_slot_ptr + ordinal)
            slot64 = slot.to(tl.int64)
            for cs in range(0, gu_elems, GU_CHUNK):
                cnt = tl.minimum(GU_CHUNK, gu_elems - cs)
                src = gate_up_slot_ptr + slot64 * gu_elems + cs
                if peer == LOCAL_RANK:
                    _mega_grad_accum(
                        acc_gate_up_ptr, src, home, cs, cnt, gu_elems, ACC_BLK)
                else:
                    libshmem_device.getmem(gu_row, src, cnt * 2, peer)
                    _mega_grad_accum(
                        acc_gate_up_ptr, gu_row, home, cs, cnt, gu_elems,
                        ACC_BLK)
            for cs in range(0, dn_elems, DN_CHUNK):
                cnt = tl.minimum(DN_CHUNK, dn_elems - cs)
                src = down_slot_ptr + slot64 * dn_elems + cs
                if peer == LOCAL_RANK:
                    _mega_grad_accum(
                        acc_down_ptr, src, home, cs, cnt, dn_elems, ACC_BLK)
                else:
                    libshmem_device.getmem(dn_row, src, cnt * 2, peer)
                    _mega_grad_accum(
                        acc_down_ptr, dn_row, home, cs, cnt, dn_elems, ACC_BLK)


# ============================================================================
# the mega kernel
# ============================================================================
@triton.jit(do_not_specialize=["signal_epoch", "b3_epoch", "b1_epoch"])
def kernel_moe_backward_mega(
    # ---- P1: dispatch + fc2 dgrad (verbatim step-1 operands) ----
    gco_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    H: tl.constexpr, stride_gm,
    fc2_ptr, grad_swiglu_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    N1, K1,
    stride_we1, stride_wk1, stride_wn1,
    signal_epoch,
    # ---- P2: swiglu/situ backward (dC == grad_swiglu_ptr, stride N1) ----
    AB_ptr,                     # fc1_output [M, 2*ffn] contiguous (stride K4)
    scale_ptr,                  # recv_weights_sorted [M]
    dAB_ptr,                    # grad_fc1_output out [M, 2*ffn] (== inp4_ptr)
    dscale_ptr,                 # grad_gate out [M] (== grad_gate_ptr)
    ffn, n_rows,
    situ_beta, situ_linear_beta,
    # ---- P3: fc2 wgrad (grad_out == peer_mem alias [M, H], strides H/1) ----
    orig_in3_ptr, stride_om3, stride_ok3,      # swiglu_out_weighted [M, ffn]
    grad_fc2_ptr, stride_we3, stride_wn3, stride_wk3,
    split_cum_ptr, expert_counts_ptr,          # shared by P3/P5
    N3, K3, num_tn3, num_tk3, w3_total, max_rows_w,
    # ---- P4a: fc1 dgrad GEMM (inp == dAB_ptr, strides im/ik) ----
    fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
    hidden_buf_ptr,
    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
    N4, K4, num_tiles_n4, num_tiles_m4,
    stride_im4, stride_ik4,
    # ---- P4b: reverse-A2A push ----
    write_rank_by_src_ptr, write_off_by_src_ptr,
    M4,
    # ---- P4c: topk reduce ----
    inv_sort_ptr, grad_hidden_ptr, grad_routing_ptr,
    B4, topk4,
    # ---- P5: fc1 wgrad (grad_out == dAB_ptr [M, 2*ffn], strides K4/1) ----
    orig_in5_ptr, stride_om5, stride_ok5,      # recv_hidden_sorted [M, H]
    grad_fc1_ptr, stride_we5, stride_wn5, stride_wk5,
    N5, K5, num_tn5, num_tk5, w5_split, w5_total,
    # ---- constexpr tiles / flags ----
    D_BM: tl.constexpr, D_BN: tl.constexpr, D_BK: tl.constexpr,
    PUSH_BLOCK_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
    MAX_BWD_TILES: tl.constexpr, LOCAL_RANK: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr,
    W_BM: tl.constexpr, W_BN: tl.constexpr, W_BK: tl.constexpr,
    W_NS: tl.constexpr,
    C_BM: tl.constexpr, C_BN: tl.constexpr, C_BK: tl.constexpr,
    C_NS: tl.constexpr,
    BLOCK_N_PUSH: tl.constexpr, GATE_PAD_C: tl.constexpr,
    P1_ON: tl.constexpr, P23_ON: tl.constexpr,
    P4_ON: tl.constexpr, P5_ON: tl.constexpr,
    # ---- B3 signalization (MOE_MEGA_TILE_B3=1): per-tile readiness ----
    b3_signal_ptr, b3_epoch,
    TILE_B3: tl.constexpr,
    # ---- B1 signalization (MOE_MEGA_TILE_B1=1): per-window grad_swiglu ----
    b1_signal_ptr, b1_epoch, num_n_tiles1,
    TILE_B1: tl.constexpr,
    # ---- P4a+P4b fusion (MOE_MEGA_FUSE_P4=1): self-produce-self-push ----
    FUSE_P4: tl.constexpr,
    # ---- MoonEP physical saved (use_moonep): dual weight tables + P6 tail ----
    replica_fc2_ptr, stride_rwe1, stride_rwk1, stride_rwn1,   # P1 replica fc2
    replica_w4_ptr, stride_rwe4, stride_rwk4, stride_rwn4,    # P4a replica gate/up
    tile_home_bound4, home_base4,
    # ---- MoonEP M3 grad_reduce (grad_transport): P6 tail operands ----
    acc_gate_up_ptr, acc_down_ptr,           # fp32 [epn, H, 2F] / [epn, H, F]
    gate_up_slot_ptr, down_slot_ptr,         # borrowed symmetric bf16 tables
    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
    consumed6_ptr, consumed_count6,
    staging_gu_ptr, staging_dn_ptr,          # [ncore, chunk] per-program rows
    gu6_elems, dn6_elems,
    HOME_E: tl.constexpr, ACTIVE_E: tl.constexpr,
    GRAD_REDUCE: tl.constexpr,
    EPN6: tl.constexpr,
    GU_CHUNK: tl.constexpr, DN_CHUNK: tl.constexpr, ACC_BLK: tl.constexpr,
    TM6: tl.constexpr, TN6: tl.constexpr, BLK6: tl.constexpr,
):
    """One launch for the whole non-MoonEP MoE backward — see the module
    docstring for the phase/barrier map and the M0 probe evidence.  Grid MUST
    be (ncore(), 1, 1): barrier_all is the mixed-scope cross-rank collective
    and every program reaches all four barriers unconditionally."""
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)

    # ---------------- P1: dispatch + fc2 input-grad ----------------
    if P1_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _dispatch_grad_source_tiles(
                    pid, num_cores,
                    gco_ptr, peer_mem_ptr, signal_mem_ptr,
                    send_bucket_starts_ptr, send_counts_re_ptr,
                    send_bucket_dst_starts_ptr,
                    signal_epoch, H, stride_gm,
                    LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK,
                    MAX_BWD_TILES, PUSH_BLOCK_M, BLOCK_H_PUSH)
        with al.scope(core_mode="cube", disable_auto_sync=True):
            # Home sweep stops at HOME_E (=EPR without MoonEP): fc2_ptr is the
            # HOME-only table, so consuming replica slots here too would read
            # weights past its end AND race the replica sweep's programs on the
            # same grad_swiglu rows (the standalone step-1 kernel's split).
            _fc2_bwd_gemm_merged_tiles_wait(
                pid, num_cores,
                peer_mem_ptr, signal_mem_ptr, fc2_ptr, grad_swiglu_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
                signal_epoch,
                N1, K1,
                H, 1, stride_we1, stride_wk1, stride_wn1, N1, 1,
                D_BM, D_BN, D_BK, PUSH_BLOCK_M,
                WORLD_SIZE, EXPERTS_PER_RANK,
                0, HOME_E, 0,
                MAX_BWD_TILES, tl.bfloat16,
                b1_signal_ptr, b1_epoch, SIGNAL_ON=TILE_B1,
                LOCAL_RANK=LOCAL_RANK)
            # MoonEP dual weight table: the replica slot range [HOME_E,
            # ACTIVE_E) re-runs the sweep against replica_fc2 re-based at
            # HOME_E (the standalone step-1 kernel's second launch, inlined).
            if ACTIVE_E > HOME_E:
                _fc2_bwd_gemm_merged_tiles_wait(
                    pid, num_cores,
                    peer_mem_ptr, signal_mem_ptr, replica_fc2_ptr,
                    grad_swiglu_ptr,
                    recv_per_expert_ptr, recv_expert_offs_ptr,
                    recv_counts_re_ptr,
                    signal_epoch,
                    N1, K1,
                    H, 1, stride_rwe1, stride_rwk1, stride_rwn1, N1, 1,
                    D_BM, D_BN, D_BK, PUSH_BLOCK_M,
                    WORLD_SIZE, EXPERTS_PER_RANK,
                    HOME_E, ACTIVE_E, HOME_E,
                    MAX_BWD_TILES, tl.bfloat16,
                    b1_signal_ptr, b1_epoch, SIGNAL_ON=TILE_B1,
                    LOCAL_RANK=LOCAL_RANK)
    # B1: publish every rank's P1 remote puts; grad_swiglu GM-visible.
    # TILE_B1 replaces the barrier: P1's cube GEMM SETs a local slot per
    # (expert, n_tile, m_window) grad_swiglu tile, the P2 windowed consumer
    # merged-waits its slots, and P3 re-waits the dispatch slots itself
    # (SET values persist — a second consumer just re-reads them).  Uniform
    # constexpr, so the remaining barriers stay unconditional.
    if not TILE_B1:
        libshmem_device.barrier_all()

    # ---------------- P2 (vector) ∥ P3 (cube) ----------------
    if P23_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if TILE_B1:
                # dl.wait/consume_token must not run twice per program —
                # sub_vec0 gate (probe 3's consumer shape).
                if sub_vec_id() == 0:
                    _mega_swiglu_bwd_windowed(
                        pid, num_cores,
                        grad_swiglu_ptr, N1,
                        AB_ptr, K4,
                        ffn, scale_ptr, dAB_ptr, dscale_ptr,
                        recv_per_expert_ptr, recv_expert_offs_ptr,
                        b1_signal_ptr, b1_epoch,
                        num_n_tiles1,
                        situ_beta, situ_linear_beta,
                        EPR=EXPERTS_PER_RANK, BLOCK_M=D_BM,
                        BLOCK_SIZE=BLOCK_SIZE, ACTIVATION=ACTIVATION,
                        HAS_LINEAR_BETA=HAS_LINEAR_BETA)
            else:
                _mega_swiglu_bwd(
                    pid, num_cores,
                    grad_swiglu_ptr, N1,
                    AB_ptr, K4,
                    ffn, scale_ptr, dAB_ptr, dscale_ptr, n_rows,
                    situ_beta, situ_linear_beta,
                    BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA)
        with al.scope(core_mode="cube", disable_auto_sync=True):
            # grad_fc2_out_sorted IS peer_mem's first [M, H] rows (stride H).
            _mega_wgrad_sweep(
                pid, num_cores,
                peer_mem_ptr, H, 1,
                orig_in3_ptr, stride_om3, stride_ok3,
                grad_fc2_ptr, stride_we3, stride_wn3, stride_wk3,
                split_cum_ptr, expert_counts_ptr,
                N3, K3, num_tn3, num_tk3, 0, w3_total, max_rows_w,
                W_BM, W_BN, W_BK, W_NS,
                WAIT_DISP=TILE_B1,
                signal_mem_ptr=signal_mem_ptr, signal_epoch_val=signal_epoch,
                recv_counts_re_ptr=recv_counts_re_ptr,
                WORLD_SIZE_C=WORLD_SIZE, EPR_C=EXPERTS_PER_RANK,
                MAX_BWD_TILES_C=MAX_BWD_TILES, TILE_M_C=64)
    # B2: all ranks finish READING peer_mem (P3) before ANY rank's P4b may
    # overwrite it; P2 outputs (dAB/dscale) published for P4a/P4b.
    libshmem_device.barrier_all()

    # ---------------- P4a+P4b: fc1 input-grad GEMM + reverse-A2A push ------
    if FUSE_P4:
        # MOE_MEGA_FUSE_P4=1: each program owns a strided set of m-tiles and
        # SELF-PRODUCES then SELF-PUSHES each one — cube scope computes all
        # n-tiles of tile_m and signals its slot, the adjacent vector scope
        # waits that SAME slot and pushes the tile's rows.  B3 disappears
        # entirely (the handoff is intra-program, never cross-program), and
        # iteration i+1's cube GEMM overlaps iteration i's vector push — the
        # P2∥P3 adjacent-scope recipe applied per tile, the structural
        # equivalent of MoonEP's AIC/AIV wave pipeline (on A5's independent
        # engine arrays this is true concurrent transport).  No barrier sits
        # inside the loop and B1/B2/B4 remain unconditional; probe 4 gates
        # the scope-alternation-in-loop + self-signal chain.  MoonEP dual
        # weight table = TWO m-tile loops (home [0, tile_home_bound4) then
        # replica), never a runtime weight-pointer select in the loop body
        # (TritonToUnstructure does not lower it — see _mega_combine_gemm).
        for tile_m in range(pid, tile_home_bound4, num_cores):
            with al.scope(core_mode="cube", disable_auto_sync=True):
                _mega_gemm_mtile(
                    tile_m,
                    dAB_ptr, stride_im4, stride_ik4,
                    fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
                    0,
                    hidden_buf_ptr,
                    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                    N4, K4, num_tiles_n4,
                    b3_signal_ptr, b3_epoch,
                    C_BM, C_BN, C_BK, C_NS, LOCAL_RANK=LOCAL_RANK)
            with al.scope(core_mode="vector", disable_auto_sync=True):
                if sub_vec_id() == 0:
                    _mega_push_mtile(
                        tile_m,
                        hidden_buf_ptr,
                        write_rank_by_src_ptr, write_off_by_src_ptr,
                        peer_mem_ptr, dscale_ptr,
                        tile_row0_ptr, tile_rows_ptr,
                        b3_signal_ptr, b3_epoch,
                        H,
                        BLOCK_N_PUSH, GATE_PAD_C)
        if ACTIVE_E > HOME_E:
            # first replica m-tile on this program's stride class (ids stay
            # ≡ pid mod num_cores across the cut; pid < num_cores keeps
            # tl.cdiv's truncating divide exact for the negative remainder)
            rep_m0 = pid + tl.cdiv(tile_home_bound4 - pid, num_cores) * num_cores
            for tile_m in range(rep_m0, num_tiles_m4, num_cores):
                with al.scope(core_mode="cube", disable_auto_sync=True):
                    _mega_gemm_mtile(
                        tile_m,
                        dAB_ptr, stride_im4, stride_ik4,
                        replica_w4_ptr, stride_rwe4, stride_rwk4, stride_rwn4,
                        home_base4,
                        hidden_buf_ptr,
                        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                        N4, K4, num_tiles_n4,
                        b3_signal_ptr, b3_epoch,
                        C_BM, C_BN, C_BK, C_NS, LOCAL_RANK=LOCAL_RANK)
                with al.scope(core_mode="vector", disable_auto_sync=True):
                    if sub_vec_id() == 0:
                        _mega_push_mtile(
                            tile_m,
                            hidden_buf_ptr,
                            write_rank_by_src_ptr, write_off_by_src_ptr,
                            peer_mem_ptr, dscale_ptr,
                            tile_row0_ptr, tile_rows_ptr,
                            b3_signal_ptr, b3_epoch,
                            H,
                            BLOCK_N_PUSH, GATE_PAD_C)
    else:
        # ---------------- P4a: fc1 input-grad GEMM ----------------
        if P4_ON:
            with al.scope(core_mode="cube", disable_auto_sync=True):
                # MoonEP dual weight table as TWO single-table task-range
                # sweeps (tiles are expert-major, so home tasks are exactly
                # [0, tile_home_bound4*num_tiles_n)); the alternative — a
                # runtime weight-pointer/stride select inside the loop — does
                # not lower (TritonToUnstructure, w2 910B1).
                home_tasks4 = tile_home_bound4 * num_tiles_n4
                _mega_combine_gemm(
                    pid, num_cores,
                    dAB_ptr, stride_im4, stride_ik4,
                    fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
                    0,
                    hidden_buf_ptr,
                    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                    N4, K4, num_tiles_n4, num_tiles_m4,
                    0, home_tasks4,
                    b3_signal_ptr, b3_epoch,
                    C_BM, C_BN, C_BK, C_NS,
                    SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK)
                if ACTIVE_E > HOME_E:
                    # first replica task id on this program's stride class
                    # (ids stay ≡ pid mod num_cores across the cut; pid <
                    # num_cores keeps tl.cdiv's truncating divide exact for
                    # the negative remainder)
                    rep_begin = pid + tl.cdiv(home_tasks4 - pid,
                                              num_cores) * num_cores
                    _mega_combine_gemm(
                        pid, num_cores,
                        dAB_ptr, stride_im4, stride_ik4,
                        replica_w4_ptr, stride_rwe4, stride_rwk4, stride_rwn4,
                        home_base4,
                        hidden_buf_ptr,
                        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                        N4, K4, num_tiles_n4, num_tiles_m4,
                        rep_begin - pid, num_tiles_m4 * num_tiles_n4,
                        b3_signal_ptr, b3_epoch,
                        C_BM, C_BN, C_BK, C_NS,
                        SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK)
        # B3: local cube->vector handoff of hidden_buf. TILE_B3 replaces the
        # barrier with per-(tile,n) readiness signals (uniform constexpr —
        # every program takes the same branch, so the barrier contract is
        # intact and B1/B2/B4 remain unconditional).
        if not TILE_B3:
            libshmem_device.barrier_all()

        # ------- P4b (vec push) ∥ P5a (cube wgrad, first half of tasks) -------
        # The P2∥P3 concurrency recipe: adjacent vector/cube scopes overlap on
        # their engines. P5 needs only P2's dAB (published at B2), so its first
        # task half rides the push window — msprof showed the vector engine
        # ~97% idle across the kernel, so this window was pure loss before.
        if P4_ON:
            with al.scope(core_mode="vector", disable_auto_sync=True):
                if sub_vec_id() == 0:
                    if TILE_B3:
                        _mega_push_rows_tiled(
                            pid, num_cores,
                            hidden_buf_ptr,
                            write_rank_by_src_ptr, write_off_by_src_ptr,
                            peer_mem_ptr, dscale_ptr,
                            tile_row0_ptr, tile_rows_ptr,
                            b3_signal_ptr, b3_epoch,
                            num_tiles_m4, num_tiles_n4,
                            H, M4,
                            BLOCK_N_PUSH, GATE_PAD_C)
                    else:
                        _mega_push_rows(
                            pid, num_cores,
                            hidden_buf_ptr,
                            write_rank_by_src_ptr, write_off_by_src_ptr,
                            peer_mem_ptr, dscale_ptr,
                            H, M4,
                            BLOCK_N_PUSH, GATE_PAD_C)

    # P5a (cube wgrad, first half of tasks): with FUSE_P4 it follows the
    # fused loop, overlapping only the per-program tail pushes (adjacent
    # vec->cube scopes); otherwise it rides the P4b window as before.
    if P5_ON:
        with al.scope(core_mode="cube", disable_auto_sync=True):
            _mega_wgrad_sweep(
                pid, num_cores,
                dAB_ptr, K4, 1,
                orig_in5_ptr, stride_om5, stride_ok5,
                grad_fc1_ptr, stride_we5, stride_wn5, stride_wk5,
                split_cum_ptr, expert_counts_ptr,
                N5, K5, num_tn5, num_tk5, 0, w5_split, max_rows_w,
                W_BM, W_BN, W_BK, W_NS,
                WAIT_DISP=0,
                signal_mem_ptr=signal_mem_ptr, signal_epoch_val=signal_epoch,
                recv_counts_re_ptr=recv_counts_re_ptr,
                WORLD_SIZE_C=WORLD_SIZE, EPR_C=EXPERTS_PER_RANK,
                MAX_BWD_TILES_C=MAX_BWD_TILES, TILE_M_C=64)
    # B4: cross-rank — every push landed before any rank reduces local rows.
    libshmem_device.barrier_all()

    # ------- P4c (vec reduce) ∥ P5b (cube wgrad, second half) — no B5 ----
    # After B4 no rank writes another rank's memory, so no trailing barrier
    # is needed: programs exit when their own reduce + wgrad remainder finish
    # (probe 2a proved a strictly longer FIVE-barrier chain; four is a
    # subset).
    if P4_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_reduce(
                pid, num_cores,
                inv_sort_ptr, peer_mem_ptr,
                grad_hidden_ptr, grad_routing_ptr,
                B4, topk4, H,
                H, 1,
                BLOCK_N_PUSH, GATE_PAD_C)
    if P5_ON:
        with al.scope(core_mode="cube", disable_auto_sync=True):
            _mega_wgrad_sweep(
                pid, num_cores,
                dAB_ptr, K4, 1,
                orig_in5_ptr, stride_om5, stride_ok5,
                grad_fc1_ptr, stride_we5, stride_wn5, stride_wk5,
                split_cum_ptr, expert_counts_ptr,
                N5, K5, num_tn5, num_tk5, w5_split, w5_total, max_rows_w,
                W_BM, W_BN, W_BK, W_NS,
                WAIT_DISP=0,
                signal_mem_ptr=signal_mem_ptr, signal_epoch_val=signal_epoch,
                recv_counts_re_ptr=recv_counts_re_ptr,
                WORLD_SIZE_C=WORLD_SIZE, EPR_C=EXPERTS_PER_RANK,
                MAX_BWD_TILES_C=MAX_BWD_TILES, TILE_M_C=64)

    # ---------------- P6: MoonEP grad_reduce (GRAD_REDUCE) ----------------
    # The ReplicaGradTransport chain inlined as tail phases: seed+sink -> B6
    # (= its barrier #1) -> owner-pull.  B5 ahead of the seed publishes the
    # physical grad_fc1 (P5) / grad_fc2 (P3) stores — unlike the P4c tail
    # this one IS cross-rank (P6b pulls every peer's sunk slots), so it
    # cannot ride kernel exit.  The transport's third stage (zeroing the
    # consumed slots) and its barrier #2 are the kernel exit + a
    # stream-ordered HOST zero after the launch — purely local work, see
    # the module docstring.  GRAD_REDUCE is a uniform constexpr: non-MoonEP
    # launches compile the whole tail — barriers included — out, leaving
    # the chain exactly as probes proved it.
    if GRAD_REDUCE:
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_grad_seed_sink(
                pid, num_cores,
                grad_fc1_ptr, grad_fc2_ptr,
                acc_gate_up_ptr, acc_down_ptr,
                gate_up_slot_ptr, down_slot_ptr,
                consumed6_ptr, consumed_count6,
                EPN6, H, ffn,
                TM6, TN6, BLK6)
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            # sub_vec0 gate (the P4b push pattern): an ungated vector body
            # runs on BOTH vector subcores with the same pid (probe 1) —
            # fine for P6a's idempotent plain stores, fatal here, where
            # _mega_grad_accum's fp32 RMW would double-add every chunk.
            if sub_vec_id() == 0:
                _mega_grad_owner_pull(
                    pid, num_cores,
                    acc_gate_up_ptr, acc_down_ptr,
                    gate_up_slot_ptr, down_slot_ptr,
                    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
                    staging_gu_ptr, staging_dn_ptr,
                    gu6_elems, dn6_elems,
                    EPN=EPN6, GU_CHUNK=GU_CHUNK, DN_CHUNK=DN_CHUNK,
                    ACC_BLK=ACC_BLK, LOCAL_RANK=LOCAL_RANK)


# ============================================================================
# wrapper
# ============================================================================
def _ensure_mega_signal_local(saved, key, slots):
    """Lazy/grow-alloc a LOCAL readiness signal slab on `saved` under `key`:
    one SET slot per task, 16 int32 elements (64B) per slot, mirroring
    _ensure_bwd_signal_mem's discipline. Slot counts move with routing, so an
    existing slab is reused while it fits and regrown (free + re-alloc +
    re-zero) when a routing needs more slots. Local-only target
    (dst == LOCAL_RANK) — the slab never leaves this rank."""
    import shmem as ash
    mem = saved.get(key)
    if mem is not None and mem.numel() >= slots * 16:
        return mem
    if mem is not None:
        ash.aclshmem_free_tensor(mem)
    mem = ash.aclshmem_create_tensor(
        [slots * 16], dtype=torch.int32, device_id=saved["ep_rank"])
    mem.zero_()
    saved[key] = mem
    return mem


def mega_backward_triton(saved, dy, peer_mem, grad_transport=None):
    """MOE_BWD_MEGA=1 backward: the whole 5-step backward in ONE kernel
    launch. Non-MoonEP saved dicts return the SAME 10-key dict as the
    orchestrator's non-MoonEP branch; MoonEP saved dicts (use_moonep) return
    the orchestrator's MoonEP contract (home-slice canonical keys +
    _replica_grad_* views; with grad_transport also the reduced home grads
    and _home_grad_* seed views — see ops/backward.py:328-368).  In both
    layouts grad_fc2_out_sorted is the same zero-copy peer_mem alias step 1
    returns. peer_mem must be the session's FIRST symmetric allocation (heap
    offset 0 — dl.symm_at in P4b).

    grad_transport (MoonEP only): lends the forward's symmetric replica
    tables so the M3 grad_reduce chain rides the SAME launch as P6a/b/c tail
    phases — the host-side sink()/reduce() calls (and their post_sink_hook)
    are REPLACED by the in-kernel chain; sunk/reduced are set on return and
    the fp32 accumulators are cast out stream-ordered after the launch.  The
    slot tables are borrowed-destructively exactly as the host transport
    borrows them."""
    use_moonep = bool(saved.get("use_moonep"))
    # MOE_MEGA_P6=0 compiles the MoonEP dual tables without the grad_reduce
    # tail (bisect/escape hatch; default on).
    grad_reduce = (use_moonep and grad_transport is not None
                   and os.environ.get("MOE_MEGA_P6", "1") != "0")
    device = dy.device
    rank = saved["ep_rank"]
    W = saved["world_size"]

    # P1 prep: expert-major gco (host gather) + cached dispatch maps.
    p1 = _prepare_dispatch_fc2_bwd(saved, dy)
    EPR = p1["E"]; H = p1["H"]; N1 = p1["N"]; K1 = p1["K"]; M = p1["M"]
    ffn = N1
    # MoonEP dual weight tables (P1): on the physical layout EPR == epn + B
    # (the slot/bucket stride) and the replica down table re-bases at
    # home_e.  Non-MoonEP caches carry none of these keys — the home table
    # stands in and ACTIVE_E == HOME_E compiles the second sweep out.
    replica_fc2 = p1.get("replica_fc2", p1["fc2"])
    home_e = int(p1.get("home_experts", EPR))
    active_e = int(p1.get("active_experts", EPR))
    signal_mem = _ensure_bwd_signal_mem(saved, W, EPR, p1["max_bwd_tiles"])
    # SET-mode epoch: producer writes signal_epoch, consumer waits the same
    # value; bump after launch so the next call sees a fresh one (SET
    # overwrites — no slot zeroing). Same keys MegaMoEFunction.backward
    # injects/persists via `state`.
    signal_epoch = saved.get("_bwd_tile_signal_epoch", 1)
    saved["_bwd_tile_signal_epoch"] = signal_epoch + 1

    # wgrad m-loop bound + the READ-buffer pad row count, needed before the
    # output allocations below: rows past each expert's split_size are masked
    # in-kernel, but the masked lanes still VALIDATE their base addresses —
    # for late/small experts split_begin + max_rows runs past the buffers and
    # traps aicore 507015 (w2 hot-expert, 910B1 2026-09-09).  Both in-kernel
    # clamp forms faulted the kernel deterministically (see the
    # DO-NOT-TOUCH note in _mega_wgrad_sweep), so the buffers are PADDED
    # host-side instead: max_rows_w extra rows keep every masked-lane address
    # mapped.  peer_mem — the one buffer this wrapper cannot re-alloc —
    # relies on its recv-budget slack: total_recv + max_rows_w <= budget rows
    # holds for every non-adversarial routing (total_recv ~= tokens*topk <<
    # budget = tokens*topk*world).
    p4 = _combine_static_maps(saved)
    # floored at 1 so the in-kernel m-loop never degenerates to a zero-trip
    # shape (all-experts-empty routings; see the miscompile note in
    # _mega_wgrad_sweep)
    max_rows_w = max(1, int(p4["expert_counts"].max().item()))
    pad_rows_w = max_rows_w

    # outputs (fresh, contiguous — strides passed to the kernel; the four
    # wgrad-sweep read targets carry the pad rows, returned keys are [:M]
    # views)
    fc1_output = saved["fc1_output"].contiguous()      # [M, 2*ffn]
    grad_swiglu = torch.empty(
        M + pad_rows_w, ffn, dtype=dy.dtype, device=device)[:M]
    grad_fc1_output = torch.empty(
        M + pad_rows_w, 2 * ffn, dtype=fc1_output.dtype, device=device)[:M]
    grad_gate = torch.empty(M, dtype=fc1_output.dtype, device=device)
    orig_in3 = torch.empty(
        M + pad_rows_w, ffn, dtype=dy.dtype, device=device)
    orig_in3[:M].copy_(saved["swiglu_out_weighted"])
    grad_fc2 = torch.empty(EPR, H, ffn, dtype=dy.dtype, device=device)
    orig_in5 = torch.empty(
        M + pad_rows_w, H, dtype=dy.dtype, device=device)
    orig_in5[:M].copy_(saved["recv_hidden_sorted"])
    grad_fc1 = torch.empty(EPR, 2 * ffn, H, dtype=dy.dtype, device=device)
    hidden_buf = torch.empty(
        M + pad_rows_w, H, dtype=dy.dtype, device=device)

    # P4 prep: cached combine maps + per-GEMM-tile expert/row tables.
    # A3 (910B-class) L0C halves 256KB -> 128KB: a 256x256 accumulator (the
    # A5 default BM/BN pair) trips bishengir `cc overflow ... 2097152 bits
    # while 1048576 bits available` at COMPILE time. 128KB accs are all fine
    # (A3 sweep, functional w8 2026-09-08: D 128x256 / C 128x256 / D,C
    # 256x128 / W 128x256 pass; only 256x256 fails), so on 910B parts just
    # the BM default steps down (both GEMMs, see the dbm note below) — BN
    # stays 256 to keep the MTE2 weight-reuse amortization the A5 tuning was
    # after. The no-l0c launch option below stays keyed to cbm*cbn > 128*256
    # (A3 defaults never reach it).
    try:
        _l0c_128k = str(NPUUtils().get_arch()).startswith("Ascend910")
    except Exception:
        _l0c_128k = False
    cbm, cbn, cbk, cns = _combine_gemm_tile()
    cbm = int(os.environ.get(
        "MOE_COMBINE_GEMM_BM", "128" if _l0c_128k else "256"))   # see dbm note above
    tiles = _gemm_tile_maps(saved, cbm)
    num_tn4 = (H + cbn - 1) // cbn
    grad_hidden = torch.empty(p4["B"], H, dtype=dy.dtype, device=device)
    grad_routing = torch.empty(
        p4["B"] * p4["topk"], dtype=dy.dtype, device=device)

    # wgrad tiles: BM=64 keeps the direct [M,N] transposed read UB-safe;
    # MOE_FUSED_WGRAD_BLOCK_M overrides (same knob as the fused step2+3).
    # BN/BK/num_stages are mega-local knobs (default the fused tile): msprof
    # showed the cube MTE2 (GM->L1 feed) pipe ~93% busy on kimi t4k — wider
    # N/K tiles amortize the transposed feed (L0C=256KB bounds BN*BK*4B).
    wbm = int(os.environ.get("MOE_FUSED_WGRAD_BLOCK_M", str(FUSED_WBM)))
    wbn = int(os.environ.get("MOE_MEGA_WGRAD_BN", str(FUSED_WBN)))
    wbk = int(os.environ.get("MOE_MEGA_WGRAD_BK", str(FUSED_WBK)))
    wns = int(os.environ.get("MOE_MEGA_WGRAD_NS", "2"))
    dbm, dbn, dbk = _dispatch_gemm_tile()
    # Mega-local GEMM BM defaults: 256 (the standalone kernels' getter
    # defaults to 128). msprof on the optimized kernel showed the cube MTE2
    # (GM->L1 feed) 90.6% busy on the busiest rank — BM 128->256 on BOTH
    # GEMMs halves the per-tile weight re-reads (kimi t4k w8 sweep:
    # 46.36 -> 43.42 ms/iter; w8 functional + f0b probe2/3 green).  CAVEAT:
    # dbm=256 is only fast together with cbm=256 (which flips the no-l0c
    # launch option below) — dbm=256 with cbm=128 measured 60.4 ms/iter.
    # On A3 (_l0c_128k above) the same 256x256 acc will not even compile,
    # so both defaults step down to 128x256 together.
    dbm = int(os.environ.get(
        "MOE_DISPATCH_GEMM_BM", "128" if _l0c_128k else "256"))
    num_tn3 = (H + wbn - 1) // wbn
    num_tk3 = (ffn + wbk - 1) // wbk
    num_tn5 = (2 * ffn + wbn - 1) // wbn
    num_tk5 = (H + wbk - 1) // wbk
    w3_total = EPR * num_tn3 * num_tk3
    w5_total = EPR * num_tn5 * num_tk5
    w5_split = w5_total // 2   # P5a/P5b half-and-half over the P4b/P4c windows

    # step-2 activation derivative selection (ops/backward.py semantics)
    activation = saved.get("activation", "swiglu")
    if activation in (None, "swiglu"):
        act, beta, lbeta, has_lb = 0, 1.0, 1.0, False
    elif activation == "situglu":
        act = 1
        beta = 1.0 if saved.get("situ_beta") is None else float(saved["situ_beta"])
        lbeta = (1.0 if saved.get("situ_linear_beta") is None
                 else float(saved["situ_linear_beta"]))
        has_lb = saved.get("situ_linear_beta") is not None
    else:
        raise ValueError(f"unknown activation for the backward: {activation!r}")

    def _flag(key):
        return os.environ.get(key, "1") != "0"

    # B3 signalization / P4 fusion: per-tile readiness slots + SET-mode epoch
    # (bumped per call — SET overwrites, so no slot re-zero between calls).
    # FUSE_P4 uses one slot per m-tile (self-produce-self-push, no B3 at
    # all); TILE_B3 uses one slot per (m,n) task.  Allocate the larger TILE_B3
    # footprint in both cases — the slab is grow-on-demand and the epoch key
    # is shared, so switching modes between calls stays monotonic and safe.
    tile_b3 = os.environ.get("MOE_MEGA_TILE_B3", "0") == "1"
    fuse_p4 = os.environ.get("MOE_MEGA_FUSE_P4", "0") == "1"
    if fuse_p4 or tile_b3:
        b3_signal = _ensure_mega_signal_local(
            saved, "_mega_b3_signal_mem", tiles["num_tiles_m"] * num_tn4)
        b3_epoch = saved.get("_mega_b3_signal_epoch", 1)
        saved["_mega_b3_signal_epoch"] = b3_epoch + 1
    else:
        # dead args under TILE_B3=0/FUSE_P4=0 (constexpr-guarded uses), but
        # the launch signature is fixed — hand it the (valid symmetric) P1
        # slab.
        b3_signal = signal_mem
        b3_epoch = 0

    # B1 signalization: one slot per (expert, n_tile, m_window) grad_swiglu
    # tile.  Slot stride max_win is derived IN-KERNEL (producer and consumer
    # run the same recv_per_expert walk); the slab only needs the hard upper
    # bound sum_e cdiv(size_e, dbm) <= cdiv(M, dbm) + 1.  Requires P1 and P23
    # (the producer/consumer phases) — otherwise the env knob is inert.
    tile_b1 = (os.environ.get("MOE_MEGA_TILE_B1", "0") == "1"
               and _flag("MOE_MEGA_P1") and _flag("MOE_MEGA_P23"))
    if tile_b1:
        num_n_tiles1 = (N1 + dbn - 1) // dbn
        b1_signal = _ensure_mega_signal_local(
            saved, "_mega_b1_signal_mem",
            EPR * num_n_tiles1 * ((M + dbm - 1) // dbm + 1))
        b1_epoch = saved.get("_mega_b1_signal_epoch", 1)
        saved["_mega_b1_signal_epoch"] = b1_epoch + 1
    else:
        b1_signal = signal_mem   # dead (constexpr-guarded uses)
        b1_epoch = 0
        num_n_tiles1 = 1

    # P4a dual weight tables (MoonEP): replica_weight is the plan's packed
    # replica gate/up table as a stride-only [B, 2F, H] view; tiles are
    # expert-major so tile_home_bound splits home/replica tiles with one
    # range cut.  Non-MoonEP: the home table + full bound (never split).
    replica_w4 = p4["replica_weight"]
    tile_home_bound4 = tiles["tile_home_bound"]
    home_base4 = int(p4["home_experts"])

    # P6 prep (grad_reduce): the by-home descriptors/consumed slots/staging
    # exactly as launch_grad_reduce_transport derives them, minus the host
    # seed/sink (the kernel's P6a replaces both — see the module docstring).
    # The fp32 accumulators are UNSEEDED device buffers until P6a fills them.
    if grad_reduce:
        epn6 = grad_transport.experts_per_rank
        if EPR != 2 * epn6:
            raise ValueError(
                "the in-kernel grad_reduce assumes the full replica budget "
                f"(E == 2*epn), got E={EPR} epn={epn6}")
        if (grad_transport.gate_up_expert_shape != (H, 2 * ffn)
                or grad_transport.down_expert_shape != (H, ffn)):
            raise ValueError(
                "the borrowed replica tables "
                f"{grad_transport.gate_up_expert_shape}/"
                f"{grad_transport.down_expert_shape} disagree with the mega "
                f"shapes {(H, 2 * ffn)}/{(H, ffn)}")
        desc_peer6, desc_slot6, _desc_home6, home_off6 = (
            build_owner_pull_descriptors_by_home(
                grad_transport.experts_to_copy_cpu, rank, epn6))
        desc_peer6 = desc_peer6.to(device)
        desc_slot6 = desc_slot6.to(device)
        home_off6 = home_off6.to(device)
        consumed = grad_transport.consumed_slots()
        consumed6 = (
            torch.tensor(consumed, dtype=torch.int32, device=device)
            if consumed
            else torch.empty(0, dtype=torch.int32, device=device))
        consumed_count6 = len(consumed)
        acc_gate_up = torch.empty(
            (epn6, H, 2 * ffn), dtype=torch.float32, device=device)
        acc_down = torch.empty(
            (epn6, H, ffn), dtype=torch.float32, device=device)
        gate_up_slot = grad_transport.buffers.gate_up
        down_slot = grad_transport.buffers.down
        gu6_elems = H * 2 * ffn
        dn6_elems = H * ffn
        # one staging row per program per table, sized by the transport's
        # chunk geometry (elements == bytes // 2 for bf16)
        gu_chunk6 = min(grad_transport.transport_staging_chunk_bytes // 2,
                        gu6_elems)
        dn_chunk6 = min(grad_transport.transport_staging_chunk_bytes // 2,
                        dn6_elems)
        staging_gu = torch.empty(
            (ncore(), gu_chunk6), dtype=torch.bfloat16, device=device)
        staging_dn = torch.empty(
            (ncore(), dn_chunk6), dtype=torch.bfloat16, device=device)
        tm6, tn6 = 64, 64
        blk6 = int(grad_transport.acc_block)
    else:
        # dead args (GRAD_REDUCE=0 compiles every P6 use out); the launch
        # signature is fixed, so hand it valid tensors of the plain outputs.
        epn6 = home_e
        consumed_count6 = 0
        dead_i32 = torch.empty(0, dtype=torch.int32, device=device)
        acc_gate_up = grad_fc1
        acc_down = grad_fc2
        gate_up_slot = grad_fc1
        down_slot = grad_fc2
        desc_peer6 = desc_slot6 = home_off6 = consumed6 = dead_i32
        staging_gu = staging_dn = grad_hidden
        gu6_elems = dn6_elems = 1
        gu_chunk6 = dn_chunk6 = 1
        tm6 = tn6 = blk6 = 1

    launch_options = (
        {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"}
        if cbm * cbn > 128 * 256 else {}
    )
    kernel_moe_backward_mega[(ncore(), 1, 1)](
        # P1
        p1["gco"], peer_mem, signal_mem,
        p1["send_bucket_starts"], p1["send_counts_re"], p1["send_bucket_dst_starts"],
        H, p1["gco"].stride(0),
        p1["fc2"], grad_swiglu,
        p1["recv_per_expert"], p1["recv_expert_offs"], p1["recv_counts_re"],
        N1, K1,
        p1["fc2"].stride(0), p1["fc2"].stride(1), p1["fc2"].stride(2),
        signal_epoch,
        # P2
        fc1_output,
        saved["recv_weights_sorted"],
        grad_fc1_output,
        grad_gate,
        ffn, M, beta, lbeta,
        # P3
        orig_in3, orig_in3.stride(0), orig_in3.stride(1),
        grad_fc2, grad_fc2.stride(0), grad_fc2.stride(1), grad_fc2.stride(2),
        p4["split_size_cum_per_expert"], p4["expert_counts"],
        H, ffn, num_tn3, num_tk3, w3_total, max_rows_w,
        # P4a
        p4["weight"], p4["we"], p4["wk"], p4["wn"],
        hidden_buf,
        tiles["tile_expert"], tiles["tile_row0"], tiles["tile_rows"],
        H, 2 * ffn, num_tn4, tiles["num_tiles_m"],
        grad_fc1_output.stride(0), grad_fc1_output.stride(1),
        # P4b
        p4["write_rank_by_src"], p4["write_off_by_src"],
        M,
        # P4c
        p4["inv_sort"], grad_hidden, grad_routing,
        p4["B"], p4["topk"],
        # P5
        orig_in5, orig_in5.stride(0), orig_in5.stride(1),
        grad_fc1, grad_fc1.stride(0), grad_fc1.stride(1), grad_fc1.stride(2),
        2 * ffn, H, num_tn5, num_tk5, w5_split, w5_total,
        # MoonEP dual tables + P6 tail operands (dead-arg pattern when off).
        # Passed BY KEYWORD: the signature's constexpr block (D_BM..FUSE_P4)
        # sits between the P5 group and these, so positional binding would
        # land on the tile constexprs.
        replica_fc2_ptr=replica_fc2,
        stride_rwe1=replica_fc2.stride(0), stride_rwk1=replica_fc2.stride(1),
        stride_rwn1=replica_fc2.stride(2),
        replica_w4_ptr=replica_w4, stride_rwe4=p4["rwe"],
        stride_rwk4=p4["rwk"], stride_rwn4=p4["rwn"],
        tile_home_bound4=tile_home_bound4, home_base4=home_base4,
        acc_gate_up_ptr=acc_gate_up, acc_down_ptr=acc_down,
        gate_up_slot_ptr=gate_up_slot, down_slot_ptr=down_slot,
        desc_peer6_ptr=desc_peer6, desc_slot6_ptr=desc_slot6,
        home_off6_ptr=home_off6,
        consumed6_ptr=consumed6, consumed_count6=consumed_count6,
        staging_gu_ptr=staging_gu, staging_dn_ptr=staging_dn,
        gu6_elems=gu6_elems, dn6_elems=dn6_elems,
        # constexpr
        D_BM=dbm, D_BN=dbn, D_BK=dbk, PUSH_BLOCK_M=64,
        WORLD_SIZE=W, EXPERTS_PER_RANK=EPR,
        MAX_BWD_TILES=p1["max_bwd_tiles"], LOCAL_RANK=rank,
        BLOCK_H_PUSH=256,
        BLOCK_SIZE=triton.next_power_of_2(ffn),
        ACTIVATION=act, HAS_LINEAR_BETA=has_lb,
        W_BM=wbm, W_BN=wbn, W_BK=wbk, W_NS=wns,
        C_BM=cbm, C_BN=cbn, C_BK=cbk, C_NS=max(cns, 1),
        BLOCK_N_PUSH=_push_block(), GATE_PAD_C=GATE_PAD,
        P1_ON=_flag("MOE_MEGA_P1"), P23_ON=_flag("MOE_MEGA_P23"),
        P4_ON=_flag("MOE_MEGA_P4"), P5_ON=_flag("MOE_MEGA_P5"),
        b3_signal_ptr=b3_signal, b3_epoch=b3_epoch, TILE_B3=tile_b3,
        b1_signal_ptr=b1_signal, b1_epoch=b1_epoch, num_n_tiles1=num_n_tiles1,
        TILE_B1=tile_b1,
        FUSE_P4=fuse_p4 and _flag("MOE_MEGA_P4"),
        HOME_E=home_e, ACTIVE_E=active_e,
        GRAD_REDUCE=grad_reduce, EPN6=epn6,
        GU_CHUNK=gu_chunk6, DN_CHUNK=dn_chunk6,
        ACC_BLK=blk6, TM6=tm6, TN6=tn6, BLK6=blk6,
        num_warps=8, **launch_options)

    # expert-major peer_mem IS the sorted layout -> identity view (no gather),
    # exactly the alias step 1 returns.
    total_recv = p1["total_recv"]
    grad_fc2_out_sorted = peer_mem.view(-1)[:total_recv * H].view(
        total_recv, H).contiguous()
    if use_moonep:
        # MoonEP contract (ops/backward.py:328-368): canonical keys expose the
        # home segment; the replica segments ride as views for the caller's
        # oracle/transport.  grad_fc1/grad_fc2 here are the PHYSICAL
        # [epn+B, ...] tensors P3/P5 wrote (and P6 seeded/sunk from) — the
        # same pre-reduction tensors the host transport hands back.
        if grad_reduce:
            # the in-kernel P6 chain replaces transport.sink()+reduce():
            # mark both stages done (post_sink_hook cannot fire from inside a
            # launch — tests that capture sunk slots must not run mega mode)
            # and cast the fp32 accumulators out in the gradient layout,
            # stream-ordered behind the launch (the host transport does the
            # same copies after its own launch).
            grad_transport.sunk = True
            grad_transport.reduced = True
            # the transport's third stage (zero the consumed slots) runs
            # HOST-side here, stream-ordered behind the launch — purely
            # local work (every remote getmem reader retired inside the
            # launch); the in-kernel P6c tail's zero stores proved
            # unreliable on this backend (see the module docstring).
            zero_consumed_replica_slots(grad_transport.buffers, consumed)
            grad_fc1_reduced = torch.empty(
                (epn6, 2 * ffn, H), dtype=dy.dtype, device=device
            ).copy_(acc_gate_up.transpose(1, 2))
            grad_fc2_reduced = acc_down.to(dy.dtype)
            grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1_reduced, 2, dim=1)
            return dict(
                grad_hidden=grad_hidden, grad_routing_weights=grad_routing.view(
                    p4["B"], p4["topk"]),
                grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2,
                grad_fc2=grad_fc2_reduced,
                grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
                grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
                grad_fc1=grad_fc1,
                _replica_grad_gate_up=grad_fc1[home_e:],
                _replica_grad_down=grad_fc2[home_e:],
                # pre-reduction seeds (views): what the owner started from
                _home_grad_fc1=grad_fc1[:home_e],
                _home_grad_fc2=grad_fc2[:home_e],
            )
        grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1[:home_e], 2, dim=1)
        return dict(
            grad_hidden=grad_hidden, grad_routing_weights=grad_routing.view(
                p4["B"], p4["topk"]),
            grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2,
            grad_fc2=grad_fc2[:home_e],
            grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
            grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
            grad_fc1=grad_fc1,
            _replica_grad_gate_up=grad_fc1[home_e:],
            _replica_grad_down=grad_fc2[home_e:],
        )
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing.view(
            p4["B"], p4["topk"]),
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
        grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
        grad_fc1=grad_fc1,
    )
