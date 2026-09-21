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
#    P1   dispatch A2A (vec, putmem+signal) + fc2 dgrad (cube, dl.wait);
#         MOE_SAVED_RECOMPUTE=1's act-row recompute rides the same vector
#         scope (ungated, published to P3 by B1)                        ]
#    B1   barrier_all  — publish P1's remote puts + grad_swiglu.          ] step 1
#         (MOE_MEGA_TILE_B1=1 replaces it with per-window SET slots — 910B1
#         only; 950DT lowering boundary, see the knob paragraph below):
#    P2   swiglu/situ backward (vec, ungated)                             ]
#    P3   fc2 wgrad (cube, BM=64 direct transposed read of peer_mem)      ] step2+3
#    B2   barrier_all  — publishes P2's outputs (dAB/dscale) for P4a/P4b;
#                        step 2 removed its OTHER job (P4b overwriting
#                        peer_mem's dispatch area while P3 still reads it)
#                        by giving the return push its own slab
#    P4a  fc1 dgrad GEMM (cube) -> hidden_buf                             ]
#    B3   barrier_all  — local cube->vector hidden_buf handoff            ] step 4
#    P4b  reverse A2A push (vec, sub_vec0-gated dl.symm_at remote stores
#         into combine_buf, the dedicated symmetric return slab — step 2) ]
#      ∥ P5a  fc1 wgrad first half (cube) — the P2∥P3 adjacent-scope      ] step 5
#             concurrency recipe; P5 needs only B2's dAB, and msprof
#             showed the vector engine ~97% idle across the kernel.
#             The split is STRUCTURAL (2026-09-17 revert of a whole-sweep
#             merge): P4c must follow B4 (every push landed) and the only
#             cube work that can cover it is dW1's tail, so one whole
#             sweep either exposes the reduce (before B4) or the push
#             (after B4).  Any split with W - r >= push and r >= reduce
#             keeps both vec phases hidden at wall = P4a + W; the 50/50
#             point sits on both sides of the measured t4k rank0 operating
#             point (push 4.91ms vs W/2 4.95ms)
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
#  M3 ReplicaGradTransport chain rides as TRANSPORT WAVE phases (2026-09-10
#  restructure; the chain is split by table and hidden in windows that
#  already existed — the old form ran it serially after B5 as
#  [seed+sink both -> B6 -> pull both]):
#
#    w1   re-dispatch sweep (MOE_SAVED_RECOMPUTE — moved here from the P1
#         window 2026-09-16: the P1 vec leg was the hot rank's window bound
#         while this window's vector engine sits idle under the cube-bound
#         fc1-dgrad GEMM) + seed acc_down + sink down slots (vec) — grad_fc2
#         is final at B2, so the seed/sink rides the same B2->B3 window
#         BESIDE the cube P4a; slab stores/sunk slots published by B3(/B4)
#    B5   barrier_all  — publish P5's physical grad_fc1 (the wave's last
#         producer dependency)
#    w2   ∥ (vec, one scope): sink gate/up slots (ungated, idempotent plain
#         stores) + owner-pull + fp32-accumulate the DOWN slots (sub_vec0:
#         getmem + RMW) — comm latency hides under the local transposes
#    B6   barrier_all  — = transport barrier #1 (the w2 gate/up slots
#         published)
#    w3   owner-pull + accumulate GATE/UP (vec, sub_vec0) — no trailing
#         barrier: after B6 no rank writes another rank's memory, programs
#         exit when their own pulls finish
#
#    (per-table (peer,slot) descriptor order is untouched by the split, so
#    the fp32 accumulators stay bit-identical to the fused transport kernel)
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
#  MOE_MEGA_TILE_B1=1 (B1 -> per-(expert, n_tile, m_window) grad_swiglu
#  SET slots; P2 merged-waits its windows, P3 re-waits the dispatch slots
#  itself) stays DEFAULT-OFF — 950DT LOWERING BOUNDARY (2026-09-10): the
#  WAIT_DISP=1 instantiation of _mega_wgrad_sweep does not lower on this
#  toolchain at all — bishengir's "merged native A5 regbase pipeline"
#  (buildFinalHIVMPipelines/PlanMemoryRegBase) emits pointer_cast ops with
#  EMPTY address spaces at the P3 transposed m-loop load fed by
#  dl.consume_token ("addrs of PointerCastOp should not be empty"); the
#  whole gate suite is compile-red 7/7 (moonep w2h x3 / w4h / w8, non-moonep
#  functional w2/w4).  Not the dual-def (single-def rebind tried), not the
#  no-l0c A5 launch path (cbm=128 tried), not m-loop pipelining
#  (MOE_MEGA_WGRAD_NS=1 tried) — the consume_token pointer x transposed
#  dot-operand orientation x regbase staging is the poison triad; P1's
#  row-major consume load lowers fine.  Validated GREEN on 910B1's older
#  bishengir (the TILE_B1 functional gate), so the knob stays for 910B1.
#  (One first-run-of-the-day anomaly compiled and numerically mismatched
#  instead — unreproduced, suspected stale cache; every subsequent run
#  fails at compile.)
#  MOE_MEGA_COMBINE_BUF (DEFAULT-ON, 2026-09-10 wave-evolution step 2):
#  the return push (P4b/P4c and the FUSE_P4 push) targets a DEDICATED
#  symmetric combine_buf instead of overwriting peer_mem's dispatch area —
#  the send/recv reuse hazard that forced B2's cross-rank edge, and the
#  precondition for wave-overlapping the return push (step 3).  symm_at
#  resolves non-first symmetric slabs on 950DT (probe 6b; the
#  combine_fc1_bwd.py offset-0 note is 910B1-era) — MOE_MEGA_COMBINE_BUF=0
#  restores the peer_mem target (the 910B1 form).
#
#  KNOWN CODEGEN LIMITATION of the two variant knobs (910B1, pre-existing
#  @91ed770): on CONCENTRATED routings (the symmetric hot-expert gate, w2/w4
#  — one expert owns every received row) the FULL kernel miscompiles P1's
#  fc2-dgrad row addressing: grad_swiglu comes out as bit-exact copies of
#  correct rows placed at wrong rows (pure data-placement, zero arithmetic
#  error; deterministic; TILE_B3 hits the home sweep on the owner rank,
#  FUSE_P4 the replica sweep on the replica rank), and everything reading
#  row-indexed companions (P2's AB, P3's orig_in3) corrupts downstream.
#  Evidence that this is codegen, not source logic: with P2-P6 compiled out
#  (MOE_MEGA_P23=0 MOE_MEGA_P4=0 MOE_MEGA_P5=0 MOE_MEGA_P6=0) P1 alone is
#  bit-identical AND correct in all three configs (peer_mem and grad_swiglu,
#  both ranks); MOE_MEGA_P6=0 alone already makes all configs bit-identical;
#  symmetric-heap divergence was ruled out (routing-independent slab sizing
#  changed nothing).  Same bishengir whole-kernel instability family as the
#  _mega_wgrad_sweep DO-NOT-TOUCH note.  Near-uniform routings are green on
#  both variants (w8/nm gates); both knobs stay default-off.  (The
#  MOE_MEGA_P6=0 bisect cited above is 910B1-only; on 950DT the =0
#  instantiation miscompiles P3/P5 itself — see the wrapper note.)
#  MoonEP: use_moonep saved dicts are supported with the dual weight tables
#  always on; the P6 grad_reduce tail turns on iff the caller lends a
#  grad_transport (ops.backward passes it whenever the forward lent the
#  replica tables — the M3 path).  The transport's own knobs
#  (transport_staging_chunk_bytes / acc_block) carry over as GU_CHUNK /
#  DN_CHUNK / ACC_BLK / BLK6; MOONEP_GRAD_TRANSPORT=legacy is inert here
#  (the in-kernel chain replaces both transport modes).
# ============================================================================

import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .common import NPUUtils, _phase_stamp, ncore
from .dispatch_fc2_bwd import (
    _prepare_dispatch_fc2_bwd,
    _dispatch_grad_source_tiles,
    _fc2_bwd_gemm_merged_tiles_wait,
    _dispatch_gemm_tile,
    _ensure_bwd_signal_mem,
    _sys_cnt_tick,
    _wait_bounded_report,
)
from .combine_fc1_bwd import (
    _combine_static_maps,
    _gemm_tile_maps,
    _combine_gemm_tile,
    _push_block,
    GATE_PAD,
)
from .fused_swiglu_bwd_fc2_wgrad import FUSED_WBM, FUSED_WBN, FUSED_WBK
from .replica_weight_prefetch import (
    _kernel_replica_repush_store,
    _kernel_compact_local_replica_descriptors,
    _push_compact_replica_weight_descriptors,
    _push_compact_replica_weight_descriptors_mte,
    _push_compact_replica_weight_descriptors_store,
    _push_replica_weights_udma_peer,
    _udma_put_nbi,
    _udma_put_signal_nbi,
    _udma_quiet,
)
from ..runtime.replica_weight_prefetch import replica_weight_push_geometry
from .replica_grad_reduce import (
    build_owner_pull_descriptors_by_home,
    launch_replica_grad_barrier,
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
    fc1_scale_ptr,
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
):
    """One row of the step-2 swiglu/situ backward (kernels/swiglu_bwd.py:32-70
    verbatim); dC_ready_ptr is grad_swiglu either behind a consume_token (the
    TILE_B1 windowed consumer) or the plain pointer.  FC1_FP8 dequantizes the
    saved gate/up halves at the load (!59's single-kernel save format: FP8
    E4M3 payload, one FP32 scale per row per FC1_GROUP_N-wide column group,
    gate groups then up groups over the [rows, 2F] row)."""
    offs = tl.arange(0, BLOCK_SIZE)
    r64 = row.to(tl.int64)
    a_ptr = AB_ptr + r64 * AB_stride          # gate half
    b_ptr = a_ptr + ffn                        # up half
    dc_ptr = dC_ready_ptr + r64 * dC_stride
    mask = offs < ffn
    dc = tl.load(dc_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if FC1_FP8:
        scale_row = fc1_scale_ptr + r64 * ((2 * ffn) // FC1_GROUP_N)
        a = a * tl.load(scale_row + offs // FC1_GROUP_N,
                        mask=mask, other=0.0)
        b = b * tl.load(scale_row + (ffn + offs) // FC1_GROUP_N,
                        mask=mask, other=0.0)
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
    # store in dAB's element type (BF16), never AB's: under FC1_FP8 the load
    # side dequantized into fp32, and the gradient must NOT round back to FP8
    tl.store(dAB_ptr + r64 * AB_stride + offs, da.to(dAB_ptr.dtype.element_ty), mask=mask)
    tl.store(dAB_ptr + r64 * AB_stride + ffn + offs, db.to(dAB_ptr.dtype.element_ty), mask=mask)
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
    fc1_scale_ptr,
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
):
    """Step-2 body (kernels/swiglu_bwd.py:32-70 verbatim): ungated vector
    work, rows partitioned by pid over the mega grid (= ncore())."""
    for row in range(pid, n_rows, nprogs):
        _mega_swiglu_bwd_row(
            row, dC_ptr, dC_stride, AB_ptr, AB_stride, ffn,
            scale_ptr, dAB_ptr, dscale_ptr,
            situ_beta, situ_linear_beta,
            BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA,
            fc1_scale_ptr, FC1_FP8, FC1_GROUP_N)


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
    fc1_scale_ptr,
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
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
                        BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA,
                        fc1_scale_ptr, FC1_FP8, FC1_GROUP_N)
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
    tile_m_begin, tile_m_end,
    signal_mem_ptr, signal_epoch,           # B3 readiness slots (SIGNAL_ON)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SIGNAL_ON: tl.constexpr, LOCAL_RANK: tl.constexpr,
    # MOE_MEGA_REPREFETCH=1 replica-call only: wait this m-tile's replica
    # gate/up weight slot before first dereference (P1's in-launch re-push
    # publishes it; deadline cushion = the whole P1 + P2||P3 windows).
    replica_weight_ready_ptr=None, replica_weight_epoch=0,
    WAIT_REPLICA_WEIGHTS: tl.constexpr = 0,
    wait_dbg_ptr=None, WAIT_DEBUG: tl.constexpr = 0,
):
    """fc1 input-grad GEMM ``hidden_buf[m, :] = grad_fc1_output[m, :] @
    weight[e]`` over the M-TILE range [tile_m_begin, tile_m_end) against ONE
    weight table re-based at expert_base (bounds RUNTIME — the constexpr-bound
    production variant would recompile whenever routing moves a tile).
    Persistent STRIDED task partition (same imbalance fix as
    _mega_wgrad_sweep), from _kernel_combine_fc1_bwd_gemm_group
    (combine_fc1_bwd.py:114-171).  The MoonEP dual weight table is TWO calls
    of this helper (home m-tiles then replica m-tiles — the RANGE is in
    m-tile units exactly like the standalone's FIRST/LAST_TILE_M), NOT a
    runtime select on the weight pointer: an arith.select on pointers/strides
    inside the GEMM loop does not lower (CANN TritonToUnstructure fails, w2
    910B1). Must run inside a cube scope.

    2026-09-12 BUGFIX (kimi t4k 507015): this helper originally took a GLOBAL
    task range and decoded ``tile_m = task % num_tiles_m`` (n-major over ALL
    m-tiles), while the MoonEP call site cut the range at
    ``tile_home_bound * num_tiles_n`` — a bound that is only the home tile
    set under M-MAJOR enumeration.  Under the n-major decode the home sweep
    picked up replica m-tiles (weight row ``expert_id`` read up to epn rows
    PAST the epn-row home table) and the replica sweep picked up home tiles
    (NEGATIVE weight rows), i.e. both sweeps read the wrong table for ~half
    the tiles with offsets of up to ±epn expert rows (±176 MB at kimi t4k) —
    silent wrong values where the VA happened to be mapped (tokens=512) and
    MTE invalid-GM 507015 where it was not (tokens=4096).  The m-tile range
    restores the standalone's exact split semantics; for the non-MoonEP full
    range [0, num_tiles_m) the (task -> tile, program) mapping is bit-for-bit
    the old one.

    SIGNAL_ON=1 (MOE_MEGA_TILE_B3=1) fires a local readiness signal after
    each (tile_m, tile_n) task's store lands: fence() orders the FixPipe
    store before signal_op SET (probe 3, w8 910B1 2026-09-08), so the P4b
    push can dl.wait per M-tile instead of crossing the B3 barrier. Slot
    layout tile_m*num_tiles_n + tile_n matches the push-side merged wait."""
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    group_tiles = tile_m_end - tile_m_begin
    total_tasks = group_tiles * num_tiles_n
    for task_id in range(pid, total_tasks, ncores):
        tile_m = tile_m_begin + task_id % group_tiles
        tile_n = task_id // group_tiles
        expert_id = tl.load(tile_expert_ptr + tile_m)
        row_start = tl.load(tile_row0_ptr + tile_m)
        rem = tl.load(tile_rows_ptr + tile_m)
        n_start = tile_n * BLOCK_N
        mm = om < rem
        mn = on_ < (N - n_start)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if WAIT_REPLICA_WEIGHTS:
            # slot ids are table-local: expert_base == home_base on the
            # replica call, so (expert_id - expert_base) IS the replica slot
            replica_slot = expert_id - expert_base
            if WAIT_DEBUG:
                _wait_bounded_report(
                    replica_weight_ready_ptr + replica_slot * 16,
                    replica_weight_epoch, 3, replica_slot, expert_id,
                    pid, wait_dbg_ptr)
            else:
                dl.wait(
                    replica_weight_ready_ptr + replica_slot * 16,
                    1, "gpu", "acquire", waitValue=replica_weight_epoch)
            sweep_weight_ptr = weight_ptr
        else:
            sweep_weight_ptr = weight_ptr
        wb = (expert_id.to(tl.int64) - expert_base) * stride_we
        row_base = row_start.to(tl.int64) + om.to(tl.int64)
        for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            mk = ok < (K - ks)
            ao = row_base[:, None] * stride_im + (ks + ok[None, :]) * stride_ik
            a = tl.load(inp_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
            bo = wb + (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
            b = tl.load(sweep_weight_ptr + bo, mask=mk[:, None] & mn[None, :], other=0.0)
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
    combine_buf_ptr,
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
        dst_base = dl.symm_at(combine_buf_ptr, dst_rank) + dst_off * row_stride
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
    combine_buf_ptr,
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
        dst_base = dl.symm_at(combine_buf_ptr, dst_rank) + dst_off * row_stride
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
    combine_buf_ptr,
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
            dst_base = dl.symm_at(combine_buf_ptr, dst_rank) + dst_off * row_stride
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
    combine_buf_ptr,
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
                     tl.load(combine_buf_ptr + sp * row_stride + H_push))
        for ns in range(0, H_push, BLOCK_N_PUSH):
            mask = ovr < (H_push - ns)
            acc = tl.zeros((BLOCK_N_PUSH,), dtype=tl.float32)
            for j in range(topk):
                fi = ti * topk + j
                sp = tl.load(inv_sort_idxs_ptr + fi).to(tl.int64)
                acc += tl.load(combine_buf_ptr + sp * row_stride + (ns + ovr),
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
def _mega_grad_transpose_tile(
    dst_ptr, src_ptr,
    H_dim, F2,
    tile,
    TM: tl.constexpr, TN: tl.constexpr,
):
    """ONE (h0, f0) tile of dst[H, F2] = transpose(src[F2, H]); the store casts
    to dst's element type, so one body serves the fp32 accumulator seed AND
    the bf16 slot sink.  NO tl.trans: this backend's TritonToUnstructure pass
    fails to lower it here (w2 compile, 910B1) and no other kernel in the repo
    uses it — the transposed read is expressed as index arithmetic instead
    (the P4c reduce's strided gather/scatter pattern, masked blocked tiles).

    2026-09-13 re-partition: the old double-loop body walked every tile of one
    row inside a single call (row = expert/slot strided across programs); the
    tile index decomposition here is h-major (tile // cdiv(F2,TN) = h0) so a
    FLAT task space of (row, tile) can be strided across all ncore()
    programs — at kimi t4k (epn=4, 32 programs) the row-strided form left 4
    programs doing every transpose while 28 sat at the transport barriers."""
    nt = tl.cdiv(F2, TN)
    h0 = (tile // nt) * TM
    f0 = (tile - (tile // nt) * nt) * TN
    om = h0 + tl.arange(0, TM)     # H axis of dst
    on_ = f0 + tl.arange(0, TN)    # F2 axis of dst
    hm = om < H_dim
    fn = on_ < F2
    v = tl.load(
        src_ptr + on_[None, :] * H_dim + om[:, None],
        mask=hm[:, None] & fn[None, :], other=0.0)
    tl.store(dst_ptr + om[:, None] * F2 + on_[None, :],
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
    SINK_GU: tl.constexpr, SINK_DN: tl.constexpr,
):
    """P6a seed+sink, SPLIT BY TABLE for the transport wave (2026-09-10):
    SINK_DN=1 seeds acc_down from the HOME grad_fc2 rows and sinks the
    replica down segments into this rank's own symmetric down slots
    (contiguous bf16 casts) — needs only P3/B2, so it runs in the B2->B3
    window beside the cube P4a (window 1).  SINK_GU=1 seeds acc_gate_up
    (transposed into the table layout) and sinks the replica gate/up
    segments (transposed) — needs P5/B5, and runs in the B5->B6 window
    beside the DOWN owner-pull (window 2).  Pure local copies, no cross-rank
    dependency.  Must run inside a vector scope.

    2026-09-13 re-partition: tasks are FLAT (row, tile/chunk) strided across
    all programs — row 0..EPN-1 seeds home experts, rows EPN.. sink consumed
    slots.  The old row-strided form left only min(rows, ncores) programs
    eligible (epn=4 of 32 at kimi t4k, consumed ~1-2), serializing hundreds
    of MB of transposes on a handful of programs while the rest sat at the
    transport barriers.  Tile/chunk bodies, masks, and per-element value
    order are unchanged."""
    n_down = H_dim * F_dim
    offs = tl.arange(0, BLK)
    rows = EPN + consumed_count
    if SINK_GU:
        gu_tiles = tl.cdiv(H_dim, TM) * tl.cdiv(2 * F_dim, TN)
        for task in range(pid, rows * gu_tiles, ncores):
            row = task // gu_tiles
            tile = task - row * gu_tiles
            if row < EPN:
                # seed: home expert e is physical row e; acc rows disjoint
                row64 = row.to(tl.int64)
                _mega_grad_transpose_tile(
                    acc_gate_up_ptr + row64 * H_dim * (2 * F_dim),
                    grad_fc1_ptr + row64 * (2 * F_dim) * H_dim,
                    H_dim, 2 * F_dim, tile, TM, TN)
            else:
                # sink: consumed slot s carries the gradient of physical
                # expert EPN + s (int64: 2*epn*2F*H can top 2^31)
                slot = tl.load(consumed_ptr + (row - EPN))
                slot64 = slot.to(tl.int64)
                phys64 = (EPN + slot).to(tl.int64)
                _mega_grad_transpose_tile(
                    gate_up_slot_ptr + slot64 * H_dim * (2 * F_dim),
                    grad_fc1_ptr + phys64 * (2 * F_dim) * H_dim,
                    H_dim, 2 * F_dim, tile, TM, TN)
    if SINK_DN:
        dn_chunks = tl.cdiv(n_down, BLK)
        for task in range(pid, rows * dn_chunks, ncores):
            row = task // dn_chunks
            start = (task - row * dn_chunks) * BLK
            m = (start + offs) < n_down
            if row < EPN:
                row64 = row.to(tl.int64)
                v = tl.load(grad_fc2_ptr + row64 * n_down + start + offs,
                            mask=m, other=0.0)
                tl.store(acc_down_ptr + row64 * n_down + start + offs,
                         v.to(tl.float32), mask=m)
            else:
                slot = tl.load(consumed_ptr + (row - EPN))
                slot64 = slot.to(tl.int64)
                phys64 = (EPN + slot).to(tl.int64)
                v = tl.load(grad_fc2_ptr + phys64 * n_down + start + offs,
                            mask=m, other=0.0)
                tl.store(down_slot_ptr + slot64 * n_down + start + offs,
                         v, mask=m)


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
    PULL_GU: tl.constexpr, PULL_DN: tl.constexpr,
):
    """P6b owner-pull, SPLIT BY TABLE for the transport wave (2026-09-10):
    PULL_DN=1 runs in the B5->B6 window (peers' down slots were sunk in the
    B2->B3 window and published by B3/B4 — nothing left to wait for at B5),
    PULL_GU=1 in the B6->exit window (gate/up slots are only published at
    B6).  Each expert's (peer, slot)-ordered descriptors are pulled
    chunk-wise (getmem, blocking) and fp32-accumulated onto the seed — the
    per-table descriptor ORDER is untouched, so the fp32 result stays
    bit-identical to the fused transport kernel's.  The self-owned slot
    branch is defensive only (the planner forbids self-copies).  Must run
    inside a vector scope behind the sub_vec0 gate (the fp32 RMW would
    double-add on the second subcore).

    2026-09-13 re-partition: tasks are FLAT (home, chunk) strided across all
    programs — each task walks its home's FULL descriptor list in ordinal
    order but moves only its own chunk.  Single writer per (home, chunk)
    accumulator range (no cross-program RMW overlap, no atomics), and the
    per-element fp32 addition order is exactly the old home-strided form's
    (seed, then descriptor ordinals ascending).  The old home-strided form
    gave EPN programs work max (4 of 32 at kimi t4k) and each serialized a
    blocking getmem chain over every chunk — msprof showed the busiest
    program at 28ms with GM_to_UB 4.8% utilized."""
    gu_row = staging_gu_ptr + pid.to(tl.int64) * GU_CHUNK
    dn_row = staging_dn_ptr + pid.to(tl.int64) * DN_CHUNK
    if PULL_GU:
        gu_chunks = tl.cdiv(gu_elems, GU_CHUNK)
        for task in range(pid, EPN * gu_chunks, ncores):
            home = task // gu_chunks
            cs = (task - home * gu_chunks) * GU_CHUNK
            cnt = tl.minimum(GU_CHUNK, gu_elems - cs)
            start = tl.load(home_offsets_ptr + home)
            end = tl.load(home_offsets_ptr + home + 1)
            for ordinal in range(start, end):
                peer = tl.load(desc_peer_ptr + ordinal)
                slot = tl.load(desc_slot_ptr + ordinal)
                slot64 = slot.to(tl.int64)
                src = gate_up_slot_ptr + slot64 * gu_elems + cs
                if peer == LOCAL_RANK:
                    _mega_grad_accum(
                        acc_gate_up_ptr, src, home, cs, cnt, gu_elems,
                        ACC_BLK)
                else:
                    # BISECT arm2 (2026-09-21): back to getmem — the direct
                    # remote LOAD is the w8 507014 suspect; getmem was the
                    # 09-10/09-17-era w8-green form.
                    libshmem_device.getmem(gu_row, src, cnt * 2, peer)
                    _mega_grad_accum(
                        acc_gate_up_ptr, gu_row, home, cs, cnt, gu_elems,
                        ACC_BLK)
    if PULL_DN:
        dn_chunks = tl.cdiv(dn_elems, DN_CHUNK)
        for task in range(pid, EPN * dn_chunks, ncores):
            home = task // dn_chunks
            cs = (task - home * dn_chunks) * DN_CHUNK
            cnt = tl.minimum(DN_CHUNK, dn_elems - cs)
            start = tl.load(home_offsets_ptr + home)
            end = tl.load(home_offsets_ptr + home + 1)
            for ordinal in range(start, end):
                peer = tl.load(desc_peer_ptr + ordinal)
                slot = tl.load(desc_slot_ptr + ordinal)
                slot64 = slot.to(tl.int64)
                src = down_slot_ptr + slot64 * dn_elems + cs
                if peer == LOCAL_RANK:
                    _mega_grad_accum(
                        acc_down_ptr, src, home, cs, cnt, dn_elems,
                        ACC_BLK)
                else:
                    # BISECT arm2 (2026-09-21): see the gate/up note.
                    libshmem_device.getmem(dn_row, src, cnt * 2, peer)
                    _mega_grad_accum(
                        acc_down_ptr, dn_row, home, cs, cnt, dn_elems,
                        ACC_BLK)


# ----------------------------------------------------------------------------
# MOE_MEGA_TIMING=1: per-program SYS_CNT phase stamps.  10 checkpoints along
# the kernel body (see the stamp calls); the wrapper hands every launch a
# [ncore, MEGA_TS_SLOTS] int64 buffer, TIMING=0 compiles every stamp out (the
# default binary is unchanged).  Caveat: stamps read the clock on the issuing
# (scalar) stream — a stamp right after an al.scope measures ISSUE completion,
# not engine drain; the checkpoints after barrier_all are exact (the barrier
# is a full sync).  Bracketing dl.wait regions (P1) is exact for the same
# reason: the wait blocks the issuing stream.
# ----------------------------------------------------------------------------
MEGA_TS_SLOTS = 10


# ============================================================================
# MOE_SAVED_RECOMPUTE=1: backward-side activation recompute (2026-09-14
# design "后向最新带重算方案"; fused into the mega launch 9/15).  The mega
# kernel itself rebuilds the two large saved activations it would otherwise
# read from the forward's fixed-address workspace blocks — ONE launch, no
# pre-pass kernel:
#   * recv_hidden_sorted  [M, H] — recomputed by re-dispatching the saved
#     single PRE-DISPATCH token copy over the same (dst, expert) maps P1
#     uses ("the fc1 input recomputes as one all2all").  The re-dispatch
#     reuses the FORWARD dispatch producer's mechanism (dispatch_fc1's
#     _dispatch_one_source_tile_task): each send slot gathers its source
#     row from the single token copy (send_src_idx = bwd_expert_sort //
#     topk, the forward's send_src_idx equivalent) and stores it straight
#     into the destination rank's symmetric re-dispatch slab — no host-side
#     topk-x gathered copy (hco), no landing in peer_mem, no clone-out;
#     P5's wgrad B matrix reads the slab directly.
#   * swiglu_out_weighted [M, F] — recomputed from fc1_output and the saved
#     routing weights (H = act(F1) * F2, Hp = p * H, p kept — one scalar per
#     row, cheap to save).
# fc1_output is NOT recomputed: the forward keeps saving it and the backward
# reads it from memory (the fp8 + framework-side offload variant is
# orthogonal and needs no code here).  Window split (2026-09-16 move): the
# act-row recompute rides the P1 window's vector scope UNGATED beside P1's
# own gco putmem sweep (idempotent stores, the P6 window's proven mixed
# shape), overlapping P1's cube fc2-dgrad, with B1 publishing the rows
# cross-program for P3.  The re-dispatch sweep rides the B2->B3 window's
# leading vector scope BESIDE the cube P4a (the fc1-dgrad GEMM) — moved
# out of the P1 window, where the vec leg (gco putmem + re-dispatch, both
# sub_vec0) was the hot rank's critical path, into a window whose vector
# engine sits idle under the cube-bound GEMM.  B3 publishes the remote
# slab stores cross-rank and P5a (the only consumer) sits behind B3, so no
# barrier is added or moved (the wrapper forces TILE_B3/FUSE_P4 off under
# recompute — both would remove B3).  The wrapper launches the ORIGINAL
# kernel_moe_backward_mega (kept verbatim in this file) for the default =0
# path and this variant for =1; the variant's SAVED_RECOMPUTE constexpr
# compiles both recompute bodies out as a guard.  The forward is not
# adapted: its workspace still carries both keys, the mega path just
# stops consuming them.
# ============================================================================
@triton.jit
def _redispatch_hidden_direct(
    pid, nprogs,
    hidden_ptr, stride_rm,
    send_src_idx_ptr, redis_buf_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    H: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """The forward dispatch producer's DIRECT_REMOTE_STORE branch
    (dispatch_fc1._dispatch_one_source_tile_task), re-aimed at the
    re-dispatch slab: per send slot the source row is GATHERED FROM THE
    SINGLE pre-dispatch token copy (send_src_idx — no host-side topk-x
    materialization) and stored straight into the destination rank's
    symmetric slab via dl.symm_at, 64x1024 store blocks like the forward.
    Same (dst, expert) bucket walk and work-id decode as the P1 gco sweep
    (_dispatch_grad_source_tiles), so rows land in the identical expert-major
    receive layout recv_hidden_sorted has — P5 reads the slab as a drop-in
    B matrix.  No signal chain: the window's closing B3 barrier publishes
    the remote stores (module docstring rule; dl.symm_at resolves non-first
    slabs on 950DT, probe 6b) and P5a — the only consumer — sits behind
    B3.  Rides the B2->B3 window's leading vec scope beside the cube P4a
    (moved out of the P1 window 2026-09-16; see the recompute block
    comment)."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    for work_id in range(pid, num_tasks, nprogs):
        dst_rank = work_id % WORLD_SIZE
        expert_id = work_id // WORLD_SIZE
        task_id = dst_rank * EXPERTS_PER_RANK + expert_id
        task_start = tl.load(send_bucket_starts_ptr + task_id)
        task_count = tl.load(send_counts_re_ptr + task_id)
        task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
        remote = dl.symm_at(redis_buf_ptr, dst_rank)
        for tile_start in range(0, task_count, BLOCK_M):
            tile_rows = tile_start + rows
            row_mask = tile_rows < task_count
            source_rows = tl.load(
                send_src_idx_ptr + task_start + tile_rows,
                mask=row_mask, other=0).to(tl.int64)
            dst_rows = (task_dst_start + tile_rows).to(tl.int64)
            for col_start in range(0, H, BLOCK_N):
                cc = col_start + cols
                col_mask = cc < H
                values = tl.load(
                    hidden_ptr + source_rows[:, None] * stride_rm + cc[None, :],
                    mask=row_mask[:, None] & col_mask[None, :], other=0.0)
                tl.store(
                    remote + dst_rows[:, None] * H + cc[None, :], values,
                    mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _recompute_act_rows(
    pid, nprogs,
    fc1_out_ptr, stride_om,
    scale_ptr,
    actw_ptr, stride_am,
    ffn, n_rows,
    situ_beta, situ_linear_beta,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    fc1_scale_ptr,
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
):
    """weighted_swiglu's row formula verbatim (tl.math.tanh, fp32
    intermediates, BF16 store) over the SAVED fc1_output — bit-identical to
    the forward product except the scale, which arrives as the BF16 saved
    recv_weights_sorted instead of the forward's FP32 routing weights
    (ulp-level, gate tolerance covers it).  FC1_FP8 dequantizes the saved
    gate/up halves at the load (same contract as _mega_swiglu_bwd_row).
    Rides the P1 window's vector scope UNGATED beside the sub_vec0 remote
    bodies: pure-function stores, so running on both vector subcores with
    the same pid-strided rows is idempotent (the P6 seed_sink shape)."""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < ffn
    for row in range(pid, n_rows, nprogs):
        r64 = row.to(tl.int64)
        gate = tl.load(fc1_out_ptr + r64 * stride_om + offs,
                       mask=mask, other=0.0).to(tl.float32)
        up = tl.load(fc1_out_ptr + r64 * stride_om + ffn + offs,
                     mask=mask, other=0.0).to(tl.float32)
        if FC1_FP8:
            scale_row = fc1_scale_ptr + r64 * ((2 * ffn) // FC1_GROUP_N)
            gate = gate * tl.load(scale_row + offs // FC1_GROUP_N,
                                  mask=mask, other=0.0)
            up = up * tl.load(scale_row + (ffn + offs) // FC1_GROUP_N,
                              mask=mask, other=0.0)
        if ACTIVATION == 0:
            act = gate * tl.sigmoid(gate) * up
        else:
            situ_a = situ_beta * tl.math.tanh(gate / situ_beta) * tl.sigmoid(gate)
            if HAS_LINEAR_BETA:
                up = situ_linear_beta * tl.math.tanh(up / situ_linear_beta)
            act = situ_a * up
        sc = tl.load(scale_ptr + r64).to(tl.float32)
        act = act * sc
        tl.store(actw_ptr + r64 * stride_am + offs,
                 act.to(actw_ptr.dtype.element_ty), mask=mask)


# ============================================================================
# the ORIGINAL mega kernel — kernel_moe_backward_mega (kept verbatim,
# 2026-09-15): the pre-recompute binary and still the DEFAULT
# MOE_SAVED_RECOMPUTE=0 launch.  The fused variant below is this exact
# body plus the P0 re-dispatch + act recompute spliced into the P1
# window and the six tail args; its SAVED_RECOMPUTE=0 instantiation
# compiles both bodies out and is equivalent to this kernel.  Keep the
# shared phase bodies in sync when editing either copy.
# ============================================================================
@triton.jit(do_not_specialize=[
    "signal_epoch", "b3_epoch", "b1_epoch",
    "repref_epoch", "repref_desc_count",   # re-prefetch routing geometry
])
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
    # ---- fc1_output save format (!59): FP8 E4M3 + per-row/per-group scales
    # dequantized at the two load sites; False keeps the BF16 staged save ----
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
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
    # ---- step-2 wave evolution: dedicated symmetric return slab ----
    # (the P4b/P4c/FUSE_P4 push+reduce target; combine_buf under
    #  MOE_MEGA_COMBINE_BUF, peer_mem in the =0 fallback)
    combine_buf_ptr,
    # ---- fc1 FP8 save scales [M, 2F // FC1_GROUP_N] FP32 (dead arg when
    # FC1_FP8=False — the launch signature is fixed) ----
    fc1_scale_ptr,
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
    # ---- MoonEP re-prefetch (MOE_MEGA_REPREFETCH=1, pooled tables) ----
    # P1's idle second vector subcore re-pushes THIS layer's replica tables
    # from the home weights (values are bit-exact at backward time — the
    # optimizer has not stepped); the P1 replica sweep (down) and the P4a
    # replica m-tiles (gate/up) dl.wait their slots.  Dead-arg pattern when
    # off: pointers stay valid tensors, epochs 0, pushes compiled out.
    # Post-R1 (2026-09-21) dead-arg group: repref_etc/desc_ids/desc_count,
    # rref_gu/dn_src, *_u64_ptr, rref_rb_ptr and the RREF_* constexprs are
    # unused in the body — the re-push moved to the standalone
    # _kernel_replica_repush_store launch and the consumer wait to the
    # pre-launch collective barrier (REPREFETCH_WAIT is always 0).  They
    # stay in the signature ON PURPOSE: removing them changes the binary
    # and re-rolls the 950DT whole-kernel miscompile dice on validated
    # code (mega_bwd.py:147-164).
    repref_etc_ptr, repref_desc_ids_ptr, repref_desc_count,
    repref_gate_ready_ptr, repref_down_ready_ptr, repref_epoch,
    rref_gu_src_ptr, rref_dn_src_ptr,
    repref_gate_ready_u64_ptr, repref_down_ready_u64_ptr,
    wait_dbg_ptr, rref_rb_ptr,
    REPREFETCH: tl.constexpr,
    REPREFETCH_WAIT: tl.constexpr,
    WAIT_DEBUG: tl.constexpr,
    RREF_GU_ELEMS: tl.constexpr, RREF_GU_CHUNK: tl.constexpr, RREF_GU_NCHUNK: tl.constexpr,
    RREF_DN_ELEMS: tl.constexpr, RREF_DN_CHUNK: tl.constexpr, RREF_DN_NCHUNK: tl.constexpr,
    # ---- MOE_MEGA_TIMING=1: SYS_CNT stamps (dead-arg pattern when off) ----
    ts_ptr, wait1_ptr,
    TS_SLOTS: tl.constexpr, TIMING: tl.constexpr,
):
    """One launch for the whole non-MoonEP MoE backward — see the module
    docstring for the phase/barrier map and the M0 probe evidence.  Grid MUST
    be (ncore(), 1, 1): barrier_all is the mixed-scope cross-rank collective
    and every program reaches all four barriers unconditionally."""
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    if TIMING:
        _phase_stamp(ts_ptr, 0, pid, TS_SLOTS)   # entry

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
        if TIMING:
            _phase_stamp(ts_ptr, 1, pid, TS_SLOTS)   # P1 vec dispatch issued
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
                LOCAL_RANK=LOCAL_RANK,
                wait_acc_ptr=wait1_ptr, WAIT_ACC=TIMING)
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
                    LOCAL_RANK=LOCAL_RANK,
                    wait_acc_ptr=wait1_ptr, WAIT_ACC=TIMING,
                    replica_weight_ready_ptr=repref_down_ready_ptr,
                    replica_weight_epoch=repref_epoch,
                    WAIT_REPLICA_WEIGHTS=REPREFETCH_WAIT,
                    wait_dbg_ptr=wait_dbg_ptr, WAIT_DEBUG=WAIT_DEBUG)
        if TIMING:
            _phase_stamp(ts_ptr, 2, pid, TS_SLOTS)   # P1 cube sweep done
    # B1: publish every rank's P1 remote puts; grad_swiglu GM-visible.
    # TILE_B1 replaces the barrier: P1's cube GEMM SETs a local slot per
    # (expert, n_tile, m_window) grad_swiglu tile, the P2 windowed consumer
    # merged-waits its slots, and P3 re-waits the dispatch slots itself
    # (SET values persist — a second consumer just re-reads them).  Uniform
    # constexpr, so the remaining barriers stay unconditional.
    if not TILE_B1:
        libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 3, pid, TS_SLOTS)   # post-B1

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
                        HAS_LINEAR_BETA=HAS_LINEAR_BETA,
                        fc1_scale_ptr=fc1_scale_ptr,
                        FC1_FP8=FC1_FP8, FC1_GROUP_N=FC1_GROUP_N)
            else:
                _mega_swiglu_bwd(
                    pid, num_cores,
                    grad_swiglu_ptr, N1,
                    AB_ptr, K4,
                    ffn, scale_ptr, dAB_ptr, dscale_ptr, n_rows,
                    situ_beta, situ_linear_beta,
                    BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA,
                    fc1_scale_ptr, FC1_FP8, FC1_GROUP_N)
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
    if TIMING:
        _phase_stamp(ts_ptr, 4, pid, TS_SLOTS)   # P2∥P3 window done
    # B2: publishes P2's outputs (dAB for P4a/P5, dscale for P4b) across
    # programs.  Step 2 removed this barrier's cross-rank job — with the
    # dedicated combine_buf the return push no longer overwrites peer_mem's
    # dispatch area while remote P3s still read it (the =0 fallback keeps
    # that hazard and this barrier remains its only protection).
    libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 5, pid, TS_SLOTS)   # post-B2

    # ---- transport wave window 1 (GRAD_REDUCE): fc2-grad seed+sink ∥ P4a ----
    # grad_fc2 is final at P3/B2, and this window's vector engine is idle
    # (P4a — or the FUSE_P4 loop's first iteration — is cube-only), so the
    # DOWN half of the old P6a rides here as the leading [vec] scope of a
    # P2∥P3-style adjacent pair.  The sunk slots are cross-rank published by
    # the very next hard barrier (B3, or B4 when B3 is signalized/unrolled)
    # and the only consumer is the B5->B6 owner-pull — no new barrier needed.
    if GRAD_REDUCE:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_grad_seed_sink(
                pid, num_cores,
                grad_fc1_ptr, grad_fc2_ptr,
                acc_gate_up_ptr, acc_down_ptr,
                gate_up_slot_ptr, down_slot_ptr,
                consumed6_ptr, consumed_count6,
                EPN6, H, ffn,
                TM6, TN6, BLK6,
                SINK_GU=0, SINK_DN=1)

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
                        combine_buf_ptr, dscale_ptr,
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
                            combine_buf_ptr, dscale_ptr,
                            tile_row0_ptr, tile_rows_ptr,
                            b3_signal_ptr, b3_epoch,
                            H,
                            BLOCK_N_PUSH, GATE_PAD_C)
    else:
        # ---------------- P4a: fc1 input-grad GEMM ----------------
        if P4_ON:
            with al.scope(core_mode="cube", disable_auto_sync=True):
                # MoonEP dual weight table as TWO single-table M-TILE-range
                # sweeps (home m-tiles [0, tile_home_bound4) then replica);
                # the alternative — a runtime weight-pointer/stride select
                # inside the loop — does not lower (TritonToUnstructure, w2
                # 910B1).
                _mega_combine_gemm(
                    pid, num_cores,
                    dAB_ptr, stride_im4, stride_ik4,
                    fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
                    0,
                    hidden_buf_ptr,
                    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                    N4, K4, num_tiles_n4, num_tiles_m4,
                    0, tile_home_bound4,
                    b3_signal_ptr, b3_epoch,
                    C_BM, C_BN, C_BK, C_NS,
                    SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK)
                if ACTIVE_E > HOME_E:
                    # replica m-tiles [tile_home_bound4, num_tiles_m4) against
                    # the replica gate/up table — the m-tile range IS the split
                    # (the old global-task-range cut mixed home and replica
                    # tiles across the two sweeps; see the helper's 2026-09-12
                    # bugfix note)
                    _mega_combine_gemm(
                        pid, num_cores,
                        dAB_ptr, stride_im4, stride_ik4,
                        replica_w4_ptr, stride_rwe4, stride_rwk4, stride_rwn4,
                        home_base4,
                        hidden_buf_ptr,
                        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                        N4, K4, num_tiles_n4, num_tiles_m4,
                        tile_home_bound4, num_tiles_m4,
                        b3_signal_ptr, b3_epoch,
                        C_BM, C_BN, C_BK, C_NS,
                        SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK,
                        replica_weight_ready_ptr=repref_gate_ready_ptr,
                        replica_weight_epoch=repref_epoch,
                        WAIT_REPLICA_WEIGHTS=REPREFETCH_WAIT,
                        wait_dbg_ptr=wait_dbg_ptr, WAIT_DEBUG=WAIT_DEBUG)
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
                            combine_buf_ptr, dscale_ptr,
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
                            combine_buf_ptr, dscale_ptr,
                            H, M4,
                            BLOCK_N_PUSH, GATE_PAD_C)

    # P5a (cube wgrad, first half of tasks): with FUSE_P4 it follows the
    # fused loop, overlapping only the per-program tail pushes (adjacent
    # vec->cube scopes); otherwise it rides the P4b window as before.
    if TIMING:
        _phase_stamp(ts_ptr, 6, pid, TS_SLOTS)   # P4a/FUSE_P4 (+P4b) done
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
    if TIMING:
        _phase_stamp(ts_ptr, 7, pid, TS_SLOTS)   # P5a done
    # B4: cross-rank — every push landed before any rank reduces local rows.
    libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 8, pid, TS_SLOTS)   # post-B4

    # ------- P4c (vec reduce) ∥ P5b (cube wgrad, second half) — no B5 ----
    # After B4 no rank writes another rank's memory, so no trailing barrier
    # is needed: programs exit when their own reduce + wgrad remainder finish
    # (probe 2a proved a strictly longer FIVE-barrier chain; four is a
    # subset).
    if P4_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_reduce(
                pid, num_cores,
                inv_sort_ptr, combine_buf_ptr,
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
    if TIMING:
        _phase_stamp(ts_ptr, 9, pid, TS_SLOTS)   # exit (pre-P6 tail)

    # ---------------- P6: MoonEP grad_reduce (GRAD_REDUCE) ----------------
    # The ReplicaGradTransport chain inlined as tail phases, restructured
    # into the transport wave (2026-09-10): the old serial chain [seed+sink
    # BOTH tables -> B6 -> pull BOTH tables] is split by table and hidden in
    # windows that already existed —
    #   window 1 (B2->B3, ∥ cube P4a, above): seed acc_down + sink down
    #   window 2 (B5->B6, below): sink gate/up (ungated, idempotent) ∥
    #     owner-pull + accumulate the DOWN slots (sub_vec0-gated RMW)
    #   window 3 (B6->exit, below): owner-pull + accumulate GATE/UP
    # B5 publishes the wave's last producer dependency (P5's grad_fc1 — the
    # down slots were published by B3/B4 already); B6 publishes the window-2
    # gate/up slots for window 3.  Barrier count is unchanged (B1..B6) and
    # each table's (peer, slot) descriptor order is untouched, so the fp32
    # accumulators stay bit-identical to the fused transport kernel's.  The
    # transport's third stage (zeroing the consumed slots) and its barrier
    # #2 remain the kernel exit + a stream-ordered HOST zero after the
    # launch — purely local work, see the module docstring.  GRAD_REDUCE is
    # a uniform constexpr: non-MoonEP launches compile the whole tail —
    # barriers included — out, leaving the chain exactly as probes proved
    # it.
    if GRAD_REDUCE:
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            # pull FIRST: the getmem stream (blocking per chunk) starts
            # while other programs are still in the ungated sink below —
            # comm latency hides under the local transposes at engine level.
            if sub_vec_id() == 0:
                _mega_grad_owner_pull(
                    pid, num_cores,
                    acc_gate_up_ptr, acc_down_ptr,
                    gate_up_slot_ptr, down_slot_ptr,
                    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
                    staging_gu_ptr, staging_dn_ptr,
                    gu6_elems, dn6_elems,
                    EPN=EPN6, GU_CHUNK=GU_CHUNK, DN_CHUNK=DN_CHUNK,
                    ACC_BLK=ACC_BLK, LOCAL_RANK=LOCAL_RANK,
                    PULL_GU=0, PULL_DN=1)
            # ungated like the old P6a: idempotent plain stores, so running
            # on BOTH vector subcores with the same pid is harmless.
            _mega_grad_seed_sink(
                pid, num_cores,
                grad_fc1_ptr, grad_fc2_ptr,
                acc_gate_up_ptr, acc_down_ptr,
                gate_up_slot_ptr, down_slot_ptr,
                consumed6_ptr, consumed_count6,
                EPN6, H, ffn,
                TM6, TN6, BLK6,
                SINK_GU=1, SINK_DN=0)
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _mega_grad_owner_pull(
                    pid, num_cores,
                    acc_gate_up_ptr, acc_down_ptr,
                    gate_up_slot_ptr, down_slot_ptr,
                    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
                    staging_gu_ptr, staging_dn_ptr,
                    gu6_elems, dn6_elems,
                    EPN=EPN6, GU_CHUNK=GU_CHUNK, DN_CHUNK=DN_CHUNK,
                    ACC_BLK=ACC_BLK, LOCAL_RANK=LOCAL_RANK,
                    PULL_GU=1, PULL_DN=0)
        # barrier #2 (2026-09-20, zero-race fix): retire every rank's w3
        # getmem readers BEFORE kernel exit — the host-side
        # zero_consumed_replica_slots is stream-ordered only behind THIS
        # rank's exit and was zeroing slots while peers still pulled them
        # (owner grad_fc1 partial loss; down was covered by B6).
        # (w8 bisect arm1 cleared it of hang causation: hangs persist
        # without it, results/plan-a-20260918/w8-phase1_1789928339.)
        libshmem_device.barrier_all()


# ============================================================================
# the mega kernel — fused backward-recompute variant
# (MOE_SAVED_RECOMPUTE=1): ONE launch carries the whole backward plus
# the saved-activation recompute riding the P1 window
#
# do_not_specialize: every ROUTING-DERIVED int must be listed here.  Triton's
# int specialization (divisible-by-16 / equal-to-1) is part of the cache key,
# so any of these crossing a ÷16 boundary mid-run mints a fresh key and pays
# a full ~6.5s JIT recompile inside the launch — one rank compiling stalls
# the whole job's iteration (msprof 2026-09-16: device idle ~6.5s windows,
# op_summary shows the launch stretched to 6518-6668ms; /root/.triton/cache
# kept minting variants through the run, e.g. max_rows_w flipping div16).
# Shape-fixed ints (N/K dims, num_tn*/num_tk*, strides, H, ffn) keep their
# specialization hints — they never vary, so their keys are stable, and the
# div16 hints can matter for address codegen.  The original
# kernel_moe_backward_mega above stays verbatim (user directive), and its
# callers hit the same trap only under non-recompute runs with shifting
# routing — fix it there too if that path is ever performance-relevant.
# ============================================================================
@triton.jit(do_not_specialize=[
    "signal_epoch", "b3_epoch", "b1_epoch",
    "repref_epoch", "repref_desc_count",         # re-prefetch routing geometry
    "n_rows", "max_rows_w", "w3_total",          # P2/P3 routing geometry
    "num_tiles_m4", "M4", "tile_home_bound4",    # P4 tile/split geometry
    "w5_split", "w5_total",                      # P5 wgrad split geometry
])
def kernel_moe_backward_mega_recompute(
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
    # ---- fc1_output save format (!59): FP8 E4M3 + per-row/per-group scales
    # dequantized at the two load sites; False keeps the BF16 staged save ----
    FC1_FP8: tl.constexpr, FC1_GROUP_N: tl.constexpr,
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
    # ---- step-2 wave evolution: dedicated symmetric return slab ----
    # (the P4b/P4c/FUSE_P4 push+reduce target; combine_buf under
    #  MOE_MEGA_COMBINE_BUF, peer_mem in the =0 fallback)
    combine_buf_ptr,
    # ---- fc1 FP8 save scales [M, 2F // FC1_GROUP_N] FP32 (dead arg when
    # FC1_FP8=False — the launch signature is fixed) ----
    fc1_scale_ptr,
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
    # ---- MoonEP re-prefetch (MOE_MEGA_REPREFETCH=1, pooled tables) ----
    # P1's idle second vector subcore re-pushes THIS layer's replica tables
    # from the home weights (values are bit-exact at backward time — the
    # optimizer has not stepped); the P1 replica sweep (down) and the P4a
    # replica m-tiles (gate/up) dl.wait their slots.  Dead-arg pattern when
    # off: pointers stay valid tensors, epochs 0, pushes compiled out.
    # Post-R1 (2026-09-21) dead-arg group: repref_etc/desc_ids/desc_count,
    # rref_gu/dn_src, *_u64_ptr, rref_rb_ptr and the RREF_* constexprs are
    # unused in the body — the re-push moved to the standalone
    # _kernel_replica_repush_store launch and the consumer wait to the
    # pre-launch collective barrier (REPREFETCH_WAIT is always 0).  They
    # stay in the signature ON PURPOSE: removing them changes the binary
    # and re-rolls the 950DT whole-kernel miscompile dice on validated
    # code (mega_bwd.py:147-164).
    repref_etc_ptr, repref_desc_ids_ptr, repref_desc_count,
    repref_gate_ready_ptr, repref_down_ready_ptr, repref_epoch,
    rref_gu_src_ptr, rref_dn_src_ptr,
    repref_gate_ready_u64_ptr, repref_down_ready_u64_ptr,
    wait_dbg_ptr, rref_rb_ptr,
    REPREFETCH: tl.constexpr,
    REPREFETCH_WAIT: tl.constexpr,
    WAIT_DEBUG: tl.constexpr,
    RREF_GU_ELEMS: tl.constexpr, RREF_GU_CHUNK: tl.constexpr, RREF_GU_NCHUNK: tl.constexpr,
    RREF_DN_ELEMS: tl.constexpr, RREF_DN_CHUNK: tl.constexpr, RREF_DN_NCHUNK: tl.constexpr,
    # ---- MOE_MEGA_TIMING=1: SYS_CNT stamps (dead-arg pattern when off) ----
    ts_ptr, wait1_ptr,
    TS_SLOTS: tl.constexpr, TIMING: tl.constexpr,
    # ---- MOE_SAVED_RECOMPUTE=1: P0 re-dispatch + act recompute operands
    # (fused into the launch — the re-dispatch sweep rides the B2->B3
    # window, the act rows the P1 window; dead-arg pattern when
    # SAVED_RECOMPUTE=0 compiles both bodies out — see the recompute block
    # comment above) ----
    redis_src_ptr, stride_rm,              # pre-dispatch token copy [B, H]
    send_src_idx_ptr,                      # send slot -> source row (sort//topk)
    redis_buf_ptr,                         # symmetric slab [rows, H] = P5's B
    SAVED_RECOMPUTE: tl.constexpr,
    REDIS_BM: tl.constexpr, REDIS_BN: tl.constexpr,
):
    """One launch for the whole non-MoonEP MoE backward — see the module
    docstring for the phase/barrier map and the M0 probe evidence.  Grid MUST
    be (ncore(), 1, 1): barrier_all is the mixed-scope cross-rank collective
    and every program reaches all four barriers unconditionally."""
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    if TIMING:
        _phase_stamp(ts_ptr, 0, pid, TS_SLOTS)   # entry

    # ---------------- P1: dispatch + fc2 input-grad ----------------
    # (+ SAVED_RECOMPUTE: the act-row recompute rides this window's vector
    # scope UNGATED beside the gco putmem sweep, overlapping the cube
    # fc2-dgrad and published to P3 by B1 — see the recompute block comment.
    # The re-dispatch sweep moved to the B2->B3 window beside the cube P4a
    # (2026-09-16): the P1 vec leg (gco + re-dispatch, both sub_vec0) was
    # this window's critical path on the hot rank, while the fc1-dgrad
    # window's vector engine sits idle under the cube-bound GEMM.  gco's
    # putmem sweep stays FIRST: its signal chain unblocks the cube dgrad
    # waiters, the critical path.)
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
            if SAVED_RECOMPUTE:
                _recompute_act_rows(
                    pid, num_cores,
                    AB_ptr, K4,
                    scale_ptr,
                    orig_in3_ptr, stride_om3,
                    ffn, n_rows, situ_beta, situ_linear_beta,
                    BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA,
                    fc1_scale_ptr, FC1_FP8, FC1_GROUP_N)
        if TIMING:
            _phase_stamp(ts_ptr, 1, pid, TS_SLOTS)   # P1 vec dispatch issued
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
                LOCAL_RANK=LOCAL_RANK,
                wait_acc_ptr=wait1_ptr, WAIT_ACC=TIMING)
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
                    LOCAL_RANK=LOCAL_RANK,
                    wait_acc_ptr=wait1_ptr, WAIT_ACC=TIMING,
                    replica_weight_ready_ptr=repref_down_ready_ptr,
                    replica_weight_epoch=repref_epoch,
                    WAIT_REPLICA_WEIGHTS=REPREFETCH_WAIT,
                    wait_dbg_ptr=wait_dbg_ptr, WAIT_DEBUG=WAIT_DEBUG)
        if TIMING:
            _phase_stamp(ts_ptr, 2, pid, TS_SLOTS)   # P1 cube sweep done
    # B1: publish every rank's P1 remote puts; grad_swiglu GM-visible.
    # TILE_B1 replaces the barrier: P1's cube GEMM SETs a local slot per
    # (expert, n_tile, m_window) grad_swiglu tile, the P2 windowed consumer
    # merged-waits its slots, and P3 re-waits the dispatch slots itself
    # (SET values persist — a second consumer just re-reads them).  Uniform
    # constexpr, so the remaining barriers stay unconditional.
    if not TILE_B1:
        libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 3, pid, TS_SLOTS)   # post-B1

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
                        HAS_LINEAR_BETA=HAS_LINEAR_BETA,
                        fc1_scale_ptr=fc1_scale_ptr,
                        FC1_FP8=FC1_FP8, FC1_GROUP_N=FC1_GROUP_N)
            else:
                _mega_swiglu_bwd(
                    pid, num_cores,
                    grad_swiglu_ptr, N1,
                    AB_ptr, K4,
                    ffn, scale_ptr, dAB_ptr, dscale_ptr, n_rows,
                    situ_beta, situ_linear_beta,
                    BLOCK_SIZE, ACTIVATION, HAS_LINEAR_BETA,
                    fc1_scale_ptr, FC1_FP8, FC1_GROUP_N)
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
    if TIMING:
        _phase_stamp(ts_ptr, 4, pid, TS_SLOTS)   # P2∥P3 window done
    # B2: publishes P2's outputs (dAB for P4a/P5, dscale for P4b) across
    # programs.  Step 2 removed this barrier's cross-rank job — with the
    # dedicated combine_buf the return push no longer overwrites peer_mem's
    # dispatch area while remote P3s still read it (the =0 fallback keeps
    # that hazard and this barrier remains its only protection).
    libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 5, pid, TS_SLOTS)   # post-B2

    # ---- recompute re-dispatch + transport wave window 1: leading vec scope
    # beside the cube P4a (the fc1-dgrad window) ----
    # MOE_SAVED_RECOMPUTE (2026-09-16 move): the re-dispatch sweep rides here
    # instead of the P1 window's vector scope — the P1 vec leg (gco putmem +
    # re-dispatch, both behind the sub_vec0 gate) was the hot rank's P1
    # critical path (the v2 recompute A/B at kimi t4k measured +3.9 ms/layer
    # non-MoonEP for the whole recompute addition), while this window's
    # vector engine sits idle under the cube-bound fc1-dgrad GEMM (the same
    # idle-vec argument that put the DOWN seed+sink here).  The sweep has no
    # producer dependency (it reads only the forward's saved token copy and
    # the host send tables) and its only consumer is P5a, behind B3 — the
    # barrier that publishes the remote slab stores cross-rank (the wrapper
    # forces TILE_B3/FUSE_P4 off under recompute: both remove that barrier).
    # GRAD_REDUCE's fc2-grad seed+sink rides the same scope as before:
    # grad_fc2 is final at P3/B2, the sunk slots are cross-rank published by
    # the very next hard barrier (B3, or B4 when B3 is signalized/unrolled)
    # and the only consumer is the B5->B6 owner-pull — no new barrier needed.
    if SAVED_RECOMPUTE:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _redispatch_hidden_direct(
                    pid, num_cores,
                    redis_src_ptr, stride_rm,
                    send_src_idx_ptr, redis_buf_ptr,
                    send_bucket_starts_ptr, send_counts_re_ptr,
                    send_bucket_dst_starts_ptr,
                    H, WORLD_SIZE, EXPERTS_PER_RANK,
                    REDIS_BM, REDIS_BN)
    if GRAD_REDUCE:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_grad_seed_sink(
                pid, num_cores,
                grad_fc1_ptr, grad_fc2_ptr,
                acc_gate_up_ptr, acc_down_ptr,
                gate_up_slot_ptr, down_slot_ptr,
                consumed6_ptr, consumed_count6,
                EPN6, H, ffn,
                TM6, TN6, BLK6,
                SINK_GU=0, SINK_DN=1)

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
                        combine_buf_ptr, dscale_ptr,
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
                            combine_buf_ptr, dscale_ptr,
                            tile_row0_ptr, tile_rows_ptr,
                            b3_signal_ptr, b3_epoch,
                            H,
                            BLOCK_N_PUSH, GATE_PAD_C)
    else:
        # ---------------- P4a: fc1 input-grad GEMM ----------------
        if P4_ON:
            with al.scope(core_mode="cube", disable_auto_sync=True):
                # MoonEP dual weight table as TWO single-table M-TILE-range
                # sweeps (home m-tiles [0, tile_home_bound4) then replica);
                # the alternative — a runtime weight-pointer/stride select
                # inside the loop — does not lower (TritonToUnstructure, w2
                # 910B1).
                _mega_combine_gemm(
                    pid, num_cores,
                    dAB_ptr, stride_im4, stride_ik4,
                    fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
                    0,
                    hidden_buf_ptr,
                    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                    N4, K4, num_tiles_n4, num_tiles_m4,
                    0, tile_home_bound4,
                    b3_signal_ptr, b3_epoch,
                    C_BM, C_BN, C_BK, C_NS,
                    SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK)
                if ACTIVE_E > HOME_E:
                    # replica m-tiles [tile_home_bound4, num_tiles_m4) against
                    # the replica gate/up table — the m-tile range IS the split
                    # (the old global-task-range cut mixed home and replica
                    # tiles across the two sweeps; see the helper's 2026-09-12
                    # bugfix note)
                    _mega_combine_gemm(
                        pid, num_cores,
                        dAB_ptr, stride_im4, stride_ik4,
                        replica_w4_ptr, stride_rwe4, stride_rwk4, stride_rwn4,
                        home_base4,
                        hidden_buf_ptr,
                        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                        N4, K4, num_tiles_n4, num_tiles_m4,
                        tile_home_bound4, num_tiles_m4,
                        b3_signal_ptr, b3_epoch,
                        C_BM, C_BN, C_BK, C_NS,
                        SIGNAL_ON=TILE_B3, LOCAL_RANK=LOCAL_RANK,
                        replica_weight_ready_ptr=repref_gate_ready_ptr,
                        replica_weight_epoch=repref_epoch,
                        WAIT_REPLICA_WEIGHTS=REPREFETCH_WAIT,
                        wait_dbg_ptr=wait_dbg_ptr, WAIT_DEBUG=WAIT_DEBUG)
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
                            combine_buf_ptr, dscale_ptr,
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
                            combine_buf_ptr, dscale_ptr,
                            H, M4,
                            BLOCK_N_PUSH, GATE_PAD_C)

    # P5a (cube wgrad, first half of tasks): with FUSE_P4 it follows the
    # fused loop, overlapping only the per-program tail pushes (adjacent
    # vec->cube scopes); otherwise it rides the P4b window as before.
    if TIMING:
        _phase_stamp(ts_ptr, 6, pid, TS_SLOTS)   # P4a/FUSE_P4 (+P4b) done
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
    if TIMING:
        _phase_stamp(ts_ptr, 7, pid, TS_SLOTS)   # P5a done
    # B4: cross-rank — every push landed before any rank reduces local rows.
    libshmem_device.barrier_all()
    if TIMING:
        _phase_stamp(ts_ptr, 8, pid, TS_SLOTS)   # post-B4

    # ------- P4c (vec reduce) ∥ P5b (cube wgrad, second half) — no B5 ----
    # After B4 no rank writes another rank's memory, so no trailing barrier
    # is needed: programs exit when their own reduce + wgrad remainder finish
    # (probe 2a proved a strictly longer FIVE-barrier chain; four is a
    # subset).
    if P4_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _mega_reduce(
                pid, num_cores,
                inv_sort_ptr, combine_buf_ptr,
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
    if TIMING:
        _phase_stamp(ts_ptr, 9, pid, TS_SLOTS)   # exit (pre-P6 tail)

    # ---------------- P6: MoonEP grad_reduce (GRAD_REDUCE) ----------------
    # The ReplicaGradTransport chain inlined as tail phases, restructured
    # into the transport wave (2026-09-10): the old serial chain [seed+sink
    # BOTH tables -> B6 -> pull BOTH tables] is split by table and hidden in
    # windows that already existed —
    #   window 1 (B2->B3, ∥ cube P4a, above): seed acc_down + sink down
    #   window 2 (B5->B6, below): sink gate/up (ungated, idempotent) ∥
    #     owner-pull + accumulate the DOWN slots (sub_vec0-gated RMW)
    #   window 3 (B6->exit, below): owner-pull + accumulate GATE/UP
    # B5 publishes the wave's last producer dependency (P5's grad_fc1 — the
    # down slots were published by B3/B4 already); B6 publishes the window-2
    # gate/up slots for window 3.  Barrier count is unchanged (B1..B6) and
    # each table's (peer, slot) descriptor order is untouched, so the fp32
    # accumulators stay bit-identical to the fused transport kernel's.  The
    # transport's third stage (zeroing the consumed slots) and its barrier
    # #2 remain the kernel exit + a stream-ordered HOST zero after the
    # launch — purely local work, see the module docstring.  GRAD_REDUCE is
    # a uniform constexpr: non-MoonEP launches compile the whole tail —
    # barriers included — out, leaving the chain exactly as probes proved
    # it.
    if GRAD_REDUCE:
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            # pull FIRST: the getmem stream (blocking per chunk) starts
            # while other programs are still in the ungated sink below —
            # comm latency hides under the local transposes at engine level.
            if sub_vec_id() == 0:
                _mega_grad_owner_pull(
                    pid, num_cores,
                    acc_gate_up_ptr, acc_down_ptr,
                    gate_up_slot_ptr, down_slot_ptr,
                    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
                    staging_gu_ptr, staging_dn_ptr,
                    gu6_elems, dn6_elems,
                    EPN=EPN6, GU_CHUNK=GU_CHUNK, DN_CHUNK=DN_CHUNK,
                    ACC_BLK=ACC_BLK, LOCAL_RANK=LOCAL_RANK,
                    PULL_GU=0, PULL_DN=1)
            # ungated like the old P6a: idempotent plain stores, so running
            # on BOTH vector subcores with the same pid is harmless.
            _mega_grad_seed_sink(
                pid, num_cores,
                grad_fc1_ptr, grad_fc2_ptr,
                acc_gate_up_ptr, acc_down_ptr,
                gate_up_slot_ptr, down_slot_ptr,
                consumed6_ptr, consumed_count6,
                EPN6, H, ffn,
                TM6, TN6, BLK6,
                SINK_GU=1, SINK_DN=0)
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _mega_grad_owner_pull(
                    pid, num_cores,
                    acc_gate_up_ptr, acc_down_ptr,
                    gate_up_slot_ptr, down_slot_ptr,
                    desc_peer6_ptr, desc_slot6_ptr, home_off6_ptr,
                    staging_gu_ptr, staging_dn_ptr,
                    gu6_elems, dn6_elems,
                    EPN=EPN6, GU_CHUNK=GU_CHUNK, DN_CHUNK=DN_CHUNK,
                    ACC_BLK=ACC_BLK, LOCAL_RANK=LOCAL_RANK,
                    PULL_GU=1, PULL_DN=0)
        # barrier #2 (2026-09-20, zero-race fix): see the non-recompute
        # kernel's note (w8 bisect arm1 cleared it of hang causation).
        libshmem_device.barrier_all()


# ============================================================================
# wrapper
# ============================================================================
def _ensure_mega_combine_buf(saved, elems):
    """Lazy/grow-alloc the DEDICATED symmetric return slab (wave-evolution
    step 2): P4b's pushes land here instead of overwriting peer_mem's
    dispatch area. bf16 like peer_mem, grow-on-demand with re-zero, mirroring
    _ensure_mega_signal_local's discipline.  Rows are fully rewritten each
    launch (write_off covers every (dst, slot) exactly once), so a reused
    slab is never re-zeroed — stale tails are never read."""
    import shmem as ash
    mem = saved.get("_mega_combine_buf")
    if mem is not None and mem.numel() >= elems:
        return mem
    if mem is not None:
        ash.aclshmem_free_tensor(mem)
    mem = ash.aclshmem_create_tensor(
        [elems], dtype=torch.bfloat16, device_id=saved["ep_rank"])
    mem.zero_()
    saved["_mega_combine_buf"] = mem
    return mem


def _ensure_mega_redispatch_buf(saved, elems):
    """Lazy/grow-alloc the symmetric re-dispatch slab (MOE_SAVED_RECOMPUTE=1):
    P0's direct remote stores land here and P5's wgrad sweep reads it as the
    recv_hidden B matrix. bf16 like peer_mem, grow-on-demand, mirroring
    _ensure_mega_combine_buf's discipline.  Read rows [0, M) are fully
    rewritten each launch (send_bucket_dst_starts covers every (dst, slot)
    row exactly once); the [M, M+pad) tail only backs P5's masked-lane
    address validation (garbage semantics, same as the padded torch buffer
    it replaces).  elems MUST come from a cross-rank MAX — every rank stores
    onto its peers, so a rank-local size skews the symmetric heap (see the
    b3_signal sizing note in the wrapper)."""
    import shmem as ash
    mem = saved.get("_mega_redispatch_buf")
    if mem is not None and mem.numel() >= elems:
        return mem
    if mem is not None:
        ash.aclshmem_free_tensor(mem)
    mem = ash.aclshmem_create_tensor(
        [elems], dtype=torch.bfloat16, device_id=saved["ep_rank"])
    mem.zero_()
    saved["_mega_redispatch_buf"] = mem
    return mem


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


def mega_backward_triton(saved, dy, peer_mem, grad_transport=None,
                         hidden_states=None):
    """MOE_BWD_MEGA=1 backward: the whole 5-step backward in ONE kernel
    launch. Non-MoonEP saved dicts return the SAME 10-key dict as the
    orchestrator's non-MoonEP branch; MoonEP saved dicts (use_moonep) return
    the orchestrator's MoonEP contract (home-slice canonical keys +
    _replica_grad_* views; with grad_transport also the reduced home grads
    and _home_grad_* seed views — see ops/backward.py:328-368).  In both
    layouts grad_fc2_out_sorted is the same zero-copy peer_mem alias step 1
    returns. peer_mem must be the session's FIRST symmetric allocation (heap
    offset 0 — the P1 putmem target and the P4b push under the
    MOE_MEGA_COMBINE_BUF=0 fallback; the default combine_buf push sits at a
    later offset, proven by probe 6b on 950DT).

    grad_transport (MoonEP only): lends the forward's symmetric replica
    tables so the M3 grad_reduce chain rides the SAME launch as P6a/b/c tail
    phases — the host-side sink()/reduce() calls (and their post_sink_hook)
    are REPLACED by the in-kernel chain; sunk/reduced are set on return and
    the fp32 accumulators are cast out stream-ordered after the launch.  The
    slot tables are borrowed-destructively exactly as the host transport
    borrows them.

    hidden_states: the forward INPUT token copy [B, H] (pre-dispatch).  Only
    MOE_SAVED_RECOMPUTE=1 reads it — the fused re-dispatch sweep (riding the
    B2->B3 window beside the cube P4a since the 2026-09-16 move out of P1)
    re-dispatches it over P1's send maps with the forward producer's
    per-token gather mechanism (dl.symm_at direct stores into a dedicated
    symmetric slab, which P5 then reads as its B matrix), and the act rows
    are rewritten from fc1_output in the P1 window.  ONE launch carries
    the whole backward plus the recompute; the two saved workspace keys are
    simply not consumed and the forward is not adapted (its workspace still
    carries them)."""
    use_moonep = bool(saved.get("use_moonep"))
    # MOE_MEGA_P6=0 compiles the MoonEP dual tables without the grad_reduce
    # tail (bisect/escape hatch; default on).  950DT CAVEAT (2026-09-10): the
    # GRAD_REDUCE=0 instantiation is itself miscompiled on this toolchain —
    # P3/P5 wgrad outputs corrupt (91k mismatches at w8, deterministic,
    # grad_hidden/grad_routing bit-exact) while the GRAD_REDUCE=1 binary is
    # green at w8 in the same device state.  The knob's 910B1 bisect evidence
    # (below, KNOWN CODEGEN LIMITATION) does NOT transfer; on 950DT it is not
    # a valid escape hatch or bisect control.
    grad_reduce = (use_moonep and grad_transport is not None
                   and os.environ.get("MOE_MEGA_P6", "1") != "0")
    device = dy.device
    rank = saved["ep_rank"]
    W = saved["world_size"]
    # attribution probe: wrapper entry — splits the inter-launch gap into
    # [prev launch -> entry] = framework autograd segment (pure host) vs
    # [entry -> post-.item()] = wrapper prep + device drain of the queue.
    if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
        print(f"[mega-ent r{rank} t={time.time():.2f}]", flush=True)

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
    # P1 signal slots: allocate AND stride at the ROUTING-INDEPENDENT tile
    # bound (cdiv(B*topk, 64) — a (source, expert) bucket cannot exceed one
    # source rank's whole dispatch). Passing the bound as the kernel
    # constexpr keeps the slot stride IDENTICAL across routings: the
    # routing-derived stride recompiled the kernel on every new value AND,
    # against the once-alloc signal slab, wrote slots past the first-alloc
    # slab's end when a fatter routing arrived (integrated kimi mock w8,
    # 2026-09-15: first-alloc strides 9/10, micro5 pads 2.5-3.9k rows, the
    # 4KB stride-overflow landing in the re-dispatch slab head — the slab
    # starts at exactly sig_off + sig_n*4). Consumers iterate ACTUAL tile
    # counts from the recv tables, so the wider stride costs nothing.
    sig_tiles_bound = max(
        1, -(-int(saved["batch_size"]) * int(saved["topk"]) // 64))
    signal_mem = _ensure_bwd_signal_mem(saved, W, EPR, sig_tiles_bound)
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
    # attribution probe: right after the .item() device sync — everything
    # queued by this rank's autograd up to here (prev mega kernel, framework
    # dense backward, p1 gather) is now retired; the wall clock it took to
    # get HERE past [mega-ent] is the device drain time.
    if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
        print(f"[mega-it r{rank} t={time.time():.2f}]", flush=True)

    # outputs (fresh, contiguous — strides passed to the kernel; the four
    # wgrad-sweep read targets carry the pad rows, returned keys are [:M]
    # views)
    fc1_output = saved["fc1_output"].contiguous()      # [M, 2*ffn]
    # !59's single-kernel forward saves fc1_output as FP8 E4M3 plus one FP32
    # scale per row and per FC1_GROUP_N-wide column group ("fc1_output_scale"
    # / "fc1_scale_group_size").  The mega kernel dequantizes AT ITS TWO LOAD
    # SITES (the P2 swiglu-bwd rows and the P1-window act recompute) — no
    # host-side BF16 materialization, the half-width save survives end to
    # end.  Staged-path saves stay BF16 and compile the dequant out
    # (FC1_FP8=False).
    fc1_fp8 = fc1_output.dtype == torch.float8_e4m3fn
    if fc1_fp8:
        fc1_scale = saved["fc1_output_scale"].contiguous()
        fc1_group_n = int(saved["fc1_scale_group_size"])
        if (
            fc1_scale.dtype != torch.float32
            or fc1_output.shape[1] % fc1_group_n
            or tuple(fc1_scale.shape)
            != (M, fc1_output.shape[1] // fc1_group_n)
        ):
            raise ValueError(
                "fc1_output_scale must be FP32 [rows, 2F // "
                "fc1_scale_group_size] matching the FP8 fc1_output")
    else:
        # dead args (FC1_FP8=False compiles the loads out); the launch
        # signature is fixed
        fc1_scale = fc1_output
        fc1_group_n = 1
    # MOE_SAVED_RECOMPUTE=1 (the 9/14 backward-side recompute design, fused
    # into this launch 9/15): the two large saved activations below are NOT
    # read — orig_in5 becomes the symmetric re-dispatch slab the B2->B3
    # window's leading vec sweep fills beside the cube P4a (the forward
    # producer's per-token gather mechanism, no host-side topk-x copy; moved
    # out of the P1 window 2026-09-16), and orig_in3's rows are recomputed
    # from fc1_output in the P1 window (see the recompute block comment
    # above the mega kernel).  fc1_output itself stays saved and read from
    # memory.  The forward is not adapted: its workspace still allocates
    # both keys, the mega path just stops consuming them.
    saved_recompute = os.environ.get("MOE_SAVED_RECOMPUTE", "0") == "1"
    if saved_recompute:
        if hidden_states is None:
            raise ValueError(
                "MOE_SAVED_RECOMPUTE=1 needs the forward input hidden_states "
                "(the saved pre-dispatch token copy) passed into the backward")
        if tuple(hidden_states.shape) != (saved["batch_size"], H):
            raise ValueError(
                f"hidden_states {tuple(hidden_states.shape)} does not match "
                f"the saved forward input {(saved['batch_size'], H)}")
    grad_swiglu = torch.empty(
        M + pad_rows_w, ffn, dtype=dy.dtype, device=device)[:M]
    # grad outputs stay BF16 (the backward contract dtype) even when
    # fc1_output is the FP8 save: grad_fc1_output/grad_gate are fresh GEMM /
    # reduction operands downstream (P4a/P5 read dAB, the return dict hands
    # both out), not mirrors of AB's dtype.
    grad_fc1_output = torch.empty(
        M + pad_rows_w, 2 * ffn, dtype=torch.bfloat16, device=device)[:M]
    grad_gate = torch.empty(M, dtype=torch.bfloat16, device=device)
    orig_in3 = torch.empty(
        M + pad_rows_w, ffn, dtype=dy.dtype, device=device)
    if not saved_recompute:
        orig_in3[:M].copy_(saved["swiglu_out_weighted"])
    grad_fc2 = torch.empty(EPR, H, ffn, dtype=dy.dtype, device=device)
    if saved_recompute:
        # P5's B matrix = the symmetric re-dispatch slab.  Pre-sized to the
        # ROUTING-INDEPENDENT bound on the first allocation and never grown:
        # M <= B*topk*W (every route lands on some rank) and
        # pad(=max_rows_w) <= B*topk, so rows <= B*topk*(W+1) covers every
        # routing.  The grow-on-demand form (cross-rank MAX of M+pad per
        # call, free+re-alloc when a later routing fatter pads arrives) hung
        # the FIRST same-layer mega launch after the first regrow in the
        # integrated framework loop (kimi mock w8, 2026-09-15: micro5 regrow
        # 41889792->43352064 elems, micro6 kernel spin -> aicore 507014;
        # heap offsets stayed rank-aligned across the regrow, so the failure
        # is the regrow itself, not offset divergence).  The bound is uniform
        # across ranks by construction (B/topk/W identical), so no
        # all_reduce is needed to keep the symmetric heap aligned.
        # [M, M+pad) tail only backs P5's masked-lane address validation,
        # garbage like the padded torch buffer it replaces.
        _rows_bound = int(saved["batch_size"]) * int(saved["topk"]) * (W + 1)
        orig_in5 = _ensure_mega_redispatch_buf(saved, _rows_bound * H)
        in5_sm, in5_sk = H, 1
        # the forward producer's source table: expert-major send slot -> row
        # in the single pre-dispatch copy (== bwd_expert_sort // topk, since
        # flat route ids are repeat_interleave order — the exact gather the
        # hco build used, minus the topk-x materialization)
        send_src_idx = (p1["bwd_expert_sort"].to(torch.int64)
                        // saved["topk"]).to(torch.int32).contiguous()
        redis_src = hidden_states.to(dy.dtype).contiguous()
    else:
        orig_in5 = torch.empty(
            M + pad_rows_w, H, dtype=dy.dtype, device=device)
        orig_in5[:M].copy_(saved["recv_hidden_sorted"])
        in5_sm, in5_sk = orig_in5.stride(0), orig_in5.stride(1)
        # dead args (SAVED_RECOMPUTE=0 compiles the P0/act bodies out)
        send_src_idx = p1["send_counts_re"]
        redis_src = fc1_output
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
    # P5a/P5b half-and-half over the P4b/P4c windows.  The split is what
    # keeps BOTH vec phases covered (2026-09-17: a whole-sweep merge was
    # tried and reverted — it hid dW1 under the push but left the post-B4
    # reduce exposed; P4c cannot move before B4, and past B4 only dW1's
    # tail can cover it, so the sweep must straddle B4.  With
    # w5_total - w5_split >= push and w5_split >= reduce the wall stays
    # cube-bound at P4a + W with nothing exposed; the 50/50 point
    # satisfies both at the measured t4k rank0 numbers, push 4.91ms vs
    # W/2 4.95ms).
    w5_split = w5_total // 2

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

    # MOE_MEGA_TIMING=1: per-program SYS_CNT stamps + P1 wait-accumulation
    # (see MEGA_TS_SLOTS above).  Buffers are allocated/reused unconditionally
    # (tiny) so the launch always hands the kernel valid pointers (dead-arg
    # pattern); TIMING=0 compiles every stamp and the wait bracket out.
    timing_on = os.environ.get("MOE_MEGA_TIMING", "0") == "1"
    ts_buf = saved.get("_mega_ts_buf")
    if ts_buf is None or tuple(ts_buf.shape) != (ncore(), MEGA_TS_SLOTS):
        ts_buf = torch.zeros(
            ncore(), MEGA_TS_SLOTS, dtype=torch.int64, device=device)
        saved["_mega_ts_buf"] = ts_buf
    wait1_buf = saved.get("_mega_wait1_buf")
    if wait1_buf is None or tuple(wait1_buf.shape) != (ncore(),):
        wait1_buf = torch.zeros(ncore(), dtype=torch.int64, device=device)
        saved["_mega_wait1_buf"] = wait1_buf
    saved["_mega_timing_last"] = (ts_buf, wait1_buf)

    # B3 signalization / P4 fusion: per-tile readiness slots + SET-mode epoch
    # (bumped per call — SET overwrites, so no slot re-zero between calls).
    # FUSE_P4 uses one slot per m-tile (self-produce-self-push, no B3 at
    # all); TILE_B3 uses one slot per (m,n) task.  Allocate the larger TILE_B3
    # footprint in both cases — the slab is grow-on-demand and the epoch key
    # is shared, so switching modes between calls stays monotonic and safe.
    #
    # ROUTING-INDEPENDENT sizing (w2 hot-expert, 910B1 2026-09-09): the slot
    # count must NOT come from tiles["num_tiles_m"]/M — those are per-rank
    # RECEIVED rows (hot routing: 1 tile on the owner, 0 on idle ranks), and
    # aclshmem_create_tensor bumps each rank's symmetric heap independently:
    # a divergent size skews every later heap offset on that rank (and a
    # 0-slot call allocates a degenerate [0] tensor).  Every other symmetric
    # slab here is sized from cross-rank collective values for this reason —
    # max_bwd_tiles is an all_reduce MAX (dispatch_fc2_bwd.py), peer_mem an
    # all_reduce MAX (make_moonep_backward_peer_mem).  Bound the row count
    # the same way: no rank receives more than every (source, expert) bucket
    # at the global max, W*EPR buckets of <= max_bwd_tiles*64 rows each.
    # NOTE: this discipline is hardening only — the symmetric hot-expert gate
    # (w2/w4) still FAILS under the variants with uniform sizing, byte-for-
    # byte identically, so heap divergence is NOT that failure's cause (root
    # cause: the codegen limitation documented in the module docstring).
    m_bound = W * EPR * p1["max_bwd_tiles"] * 64
    # MOE_MEGA_REPREFETCH=1 (pooled tables, the 2026-09-17 Plan A): re-push
    # this layer's replica weights INSIDE the launch — P1's idle second
    # vector subcore issues the owner pushes beside the gco dispatch
    # (deadline-aware order: down first, its P1 replica-sweep consumers wait
    # in-window; gate/up's consumers only run at P4a).  v1 is mutually
    # exclusive with the P4-phase restructuring knobs below (their
    # combinations reshape the P4a/B3/B4 phases this rides on), so it forces
    # both off.  Default off = the status-quo semantics (the backward reads
    # whatever the forward left in the tables; correct only with per-layer
    # tables, i.e. pool OFF).
    reprefetch = (use_moonep and active_e > home_e
                  and os.environ.get("MOE_MEGA_REPREFETCH", "0") == "1")
    # E6 research knob (2026-09-18, fault-domain split): push side still
    # runs, consumer dl.wait compiled out. Values will be WRONG — numerics
    # meaningless under this knob; pass/hang is the only signal.
    reprefetch_nowait = (reprefetch and
                         os.environ.get("MOE_MEGA_REPREFETCH_NOWAIT", "0") == "1")
    # R1 (2026-09-21): the consumer dl.wait is ALWAYS compiled out — the
    # wait's publication edge moved to a collective barrier launch queued
    # between the standalone re-push kernel and the mega kernel (see the
    # reprefetch block below), so the mega binary stays bit-identical to
    # the REPREFETCH=0 one (the w8 8/8-green binary).  The in-kernel wait
    # branch was one of the REPREFETCH=1 destabilizers of the w4/w8
    # whole-kernel miscompile family (A+getmem 2/4, A+NOWAIT 2/4 hangs;
    # results/plan-a-20260918/w8-phase1_1789930236/...884).
    reprefetch_wait = False
    if reprefetch and (
        os.environ.get("MOE_MEGA_TILE_B3", "0") == "1"
        or os.environ.get("MOE_MEGA_FUSE_P4", "0") == "1"
    ):
        print(f"[mega r{rank}] MOE_MEGA_REPREFETCH=1 forces "
              f"MOE_MEGA_TILE_B3/FUSE_P4 off (v1 mutual exclusion)",
              flush=True)
        os.environ["MOE_MEGA_TILE_B3"] = "0"
        os.environ["MOE_MEGA_FUSE_P4"] = "0"
    # SAVED_RECOMPUTE forces both B3-removing knobs off: the re-dispatch
    # sweep rides the B2->B3 window and its remote slab stores are published
    # cross-rank BY the B3 barrier — TILE_B3 (per-tile signals) and FUSE_P4
    # (self-produce-self-push) would leave P5a no publication edge for the
    # slab (both default-off perf knobs; the tile_b1 force-off precedent).
    tile_b3 = (os.environ.get("MOE_MEGA_TILE_B3", "0") == "1"
               and not saved_recompute)
    fuse_p4 = (os.environ.get("MOE_MEGA_FUSE_P4", "0") == "1"
               and not saved_recompute)
    if fuse_p4 or tile_b3:
        tiles_bound = (m_bound + cbm - 1) // cbm + EPR
        b3_signal = _ensure_mega_signal_local(
            saved, "_mega_b3_signal_mem", tiles_bound * num_tn4)
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
    # bound sum_e cdiv(size_e, dbm) <= cdiv(M, dbm) + 1 — sized from the
    # routing-independent m_bound above (same symmetric-heap alignment rule).
    # Requires P1 and P23 (the producer/consumer phases) — otherwise the env
    # knob is inert.  DEFAULT-OFF: see the 950DT lowering boundary in the
    # module docstring (the wave-evolution step-1 default-on attempt of
    # 2026-09-10 was reverted the same day).
    tile_b1 = (os.environ.get("MOE_MEGA_TILE_B1", "0") == "1"
               and _flag("MOE_MEGA_P1") and _flag("MOE_MEGA_P23"))
    if saved_recompute:
        # The act-row recompute rides the P1 window's vector scope and
        # publishes through B1: the act rows are read CROSS-PROGRAM by P3,
        # and TILE_B1's signal chain only covers grad_swiglu tiles — so the
        # barrier must stay.  Same reason the window knobs are requirements,
        # not suggestions: P1 off leaves the act recompute nowhere to run.
        # (The re-dispatch sweep rides the B2->B3 window instead — its knob
        # constraints are enforced at the tile_b3/fuse_p4 definitions above.)
        if not (_flag("MOE_MEGA_P1") and _flag("MOE_MEGA_P23")):
            raise ValueError(
                "MOE_SAVED_RECOMPUTE=1 rides the P1 window's vector scope "
                "and publishes through B1 for P3 — it needs MOE_MEGA_P1 and "
                "MOE_MEGA_P23 on")
        tile_b1 = False
    if tile_b1:
        num_n_tiles1 = (N1 + dbn - 1) // dbn
        b1_signal = _ensure_mega_signal_local(
            saved, "_mega_b1_signal_mem",
            EPR * num_n_tiles1 * ((m_bound + dbm - 1) // dbm + 1))
        b1_epoch = saved.get("_mega_b1_signal_epoch", 1)
        saved["_mega_b1_signal_epoch"] = b1_epoch + 1
    else:
        b1_signal = signal_mem   # dead (constexpr-guarded uses)
        b1_epoch = 0
        num_n_tiles1 = 1

    # step-2 wave evolution: the return push (P4b/P4c and FUSE_P4's push)
    # targets a DEDICATED symmetric slab instead of overwriting peer_mem's
    # dispatch area — the send/recv reuse hazard that forced B2's cross-rank
    # edge, and the precondition for wave-overlapping the return push (step
    # 3).  symm_at resolves non-first slabs on 950DT (probe 6b, 2026-09-10;
    # the combine_fc1_bwd.py offset-0 note is 910B1-era); =0 restores the
    # peer_mem form.  Sizing from the cross-rank MAX of B*topk return rows:
    # every rank pushes onto its peers, so a rank-local count would skew the
    # symmetric heap (the same discipline as every other slab here).
    if os.environ.get("MOE_MEGA_COMBINE_BUF", "1") == "1":
        _rows = torch.tensor(
            [p4["B"] * p4["topk"]], dtype=torch.int64, device=device)
        # attribution probe (attempt-4 perf anomaly): bracket the per-launch
        # 1-elem sizing all_reduce — the 5-op baseline backward has no such
        # collective and a degraded ~730ms/call would exactly explain the
        # stable +17.5s/iter plateau from iter3 on.
        if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
            print(f"[mega-ar r{rank} t={time.time():.2f}]", flush=True)
        dist.all_reduce(_rows, op=dist.ReduceOp.MAX, group=saved["ep_group"])
        if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
            print(f"[mega-ar+done r{rank} t={time.time():.2f}]", flush=True)
        combine_buf = _ensure_mega_combine_buf(
            saved, int(_rows.item()) * (H + GATE_PAD))
    else:
        combine_buf = peer_mem

    # MOE_MEGA_HEAP_PROBE=1: per-launch symmetric-heap audit (host-side
    # pointer arithmetic only, no device sync).  Prints every symmetric
    # slab's offset RELATIVE TO peer_mem — cross-rank DIRECT_REMOTE_STORE
    # addressing assumes these offsets are identical on every rank, so any
    # divergence between the printed offsets across ranks means stores land
    # in the wrong peer slab (the suspected iter-3 hang mechanism: clobbered
    # phase signals -> barrier spin -> aicore timeout).
    if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
        _base = peer_mem.data_ptr()

        def _off(t):
            return t.data_ptr() - _base if t is not None else None

        print(
            f"[mega-heap r{rank} t={time.time():.2f}] M={M} pad={pad_rows_w} "
            f"peer_off={_off(peer_mem)} sig_off={_off(signal_mem)} "
            f"sig_n={signal_mem.numel()} "
            f"redis_off={_off(orig_in5) if saved_recompute else '-'} "
            f"redis_n={orig_in5.numel() if saved_recompute else 0} "
            f"comb_off={_off(combine_buf)} comb_n={combine_buf.numel()} "
            f"rfc2_off={_off(replica_fc2)} "
            f"ep={signal_epoch} b3e={b3_epoch} b1e={b1_epoch} "
            f"gr={int(grad_reduce)}",
            flush=True)

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

    # per-program readback scratch for the readback-gated SET (dead-arg
    # when REPREFETCH is off)
    rref_rb = saved.get("_mega_rref_rb")
    if rref_rb is None or rref_rb.numel() < ncore() * 64:
        rref_rb = torch.zeros(ncore() * 64, dtype=torch.bfloat16,
                              device=device)
        saved["_mega_rref_rb"] = rref_rb

    # ---- MOE_MEGA_WAIT_DEBUG (2026-09-20): bounded-spin diagnostics ----
    wait_debug = os.environ.get("MOE_MEGA_WAIT_DEBUG", "0") == "1"
    if wait_debug:
        wait_dbg = saved.get("_mega_wait_dbg")
        if wait_dbg is None or wait_dbg.numel() < ncore() * 8:
            wait_dbg = torch.zeros(ncore() * 8, dtype=torch.int32,
                                   device=device)
            saved["_mega_wait_dbg"] = wait_dbg
        else:
            wait_dbg.zero_()
    else:
        wait_dbg = signal_mem

    # ---- MOE_MEGA_REPREFETCH operands (dead-arg pattern when off) ----
    # Push geometry mirrors the forward config defaults (gate/up 16MB, down
    # 4MB chunks), env-tunable for the RMA-contention sweep.
    gu_elems = 2 * ffn * H
    dn_elems = ffn * H
    gu_chunk, gu_nchunk = replica_weight_push_geometry(
        gu_elems,
        chunk_bytes=int(os.environ.get(
            "MOE_REPREF_GU_CHUNK_BYTES", str(16 * 1024 * 1024))),
    )
    dn_chunk, dn_nchunk = replica_weight_push_geometry(
        dn_elems,
        chunk_bytes=int(os.environ.get(
            "MOE_REPREF_DN_CHUNK_BYTES", str(4 * 1024 * 1024))),
    )
    rref_gu_src = p4["weight"]
    rref_dn_src = p1["fc2"]
    if reprefetch:
        try:
            repref_buffers = saved["replica_buffers"]
            repref_gate_ready = saved["replica_gate_ready"]
            repref_down_ready = saved["replica_down_ready"]
        except KeyError as exc:
            raise RuntimeError(
                "MOE_MEGA_REPREFETCH=1 needs the pooling-era saved contract "
                "(replica_buffers / replica_gate_ready / replica_down_ready); "
                "re-run the forward on a current operator") from exc
        # table-level monotonic mint: same sequence the forward pushes of
        # EVERY pooled layer took — a waiter's epoch can only be satisfied
        # by this launch's own pushes
        repref_epoch = repref_buffers.next_push_epoch()
        # the flat, layout-preserving RMA pushes need the HOME tables'
        # natural contiguous storage: fc1_combined is a transpose stride
        # view of the packed [E, H, 2F] table (its base pointer IS the
        # natural layout), a non-contiguous home storage would scramble the
        # element order into the [B, H, 2F] replica slots
        if not saved["fc1_combined"].transpose(1, 2).is_contiguous():
            raise ValueError(
                "MOE_MEGA_REPREFETCH=1 requires a contiguous home gate/up "
                "table (flat layout-preserving RMA push)")
        repref_etc = saved["experts_to_copy"]
        repref_etc_cpu = saved["experts_to_copy_cpu"]
        lo = rank * home_e
        owned = (repref_etc_cpu >= lo) & (repref_etc_cpu < lo + home_e)
        repref_desc_count = int(
            owned.sum().item() - owned[rank].sum().item())
        # E7 research knob (2026-09-18, fault-domain split): zero the push
        # traffic while keeping the sub_vec1 branch, compaction and quiet —
        # NOPUSH+NOWAIT hanging would indict the branch's mere presence
        # (lattice perturbation), green indicts the push traffic/addressing.
        if os.environ.get("MOE_MEGA_REPREFETCH_NOPUSH", "0") == "1":
            print(f"[mega r{rank}] MOE_MEGA_REPREFETCH_NOPUSH=1: descriptor "
                  f"count {repref_desc_count} -> 0 (E7; pair with NOWAIT)",
                  flush=True)
            repref_desc_count = 0
        if os.environ.get("MOE_MEGA_REPREFETCH_ZEROS", "0") == "1":
            # E9 discriminator: push ZEROS instead of weights (host-side
            # source swap; home_e * max(gu,dn) elems covers every home_slot
            # base).  w2-diagnostic only (full-size alloc).
            rref_gu_src = rref_dn_src = torch.zeros(
                home_e * max(gu_elems, dn_elems), dtype=torch.bfloat16,
                device=device)
            print(f"[mega r{rank}] MOE_MEGA_REPREFETCH_ZEROS=1: E9 zeros "
                  f"source ({home_e * max(gu_elems, dn_elems)} elems)",
                  flush=True)
        repref_gate_ready_u64 = repref_gate_ready.view(torch.uint64)
        repref_down_ready_u64 = repref_down_ready.view(torch.uint64)
        desc_ids = saved.get("_mega_repref_desc_ids")
        if desc_ids is None or desc_ids.numel() < W * home_e:
            desc_ids = torch.empty(
                W * home_e, dtype=torch.int32, device=device)
            saved["_mega_repref_desc_ids"] = desc_ids
        # slot-major owner-local descriptor compaction on the device — the
        # forward's exact pre-launch step (_begin_replica_prefetch),
        # stream-ordered ahead of the mega launch with nothing consuming it
        # in between: no exposure
        _kernel_compact_local_replica_descriptors[1, 1, 1](
            repref_etc, desc_ids,
            LOCAL_RANK=rank, WORLD_SIZE=W, EXPERTS_PER_RANK=home_e)
        # OUT-OF-KERNEL re-push (2026-09-20): run as a standalone launch
        # before the mega kernel — the push code's mere presence in the
        # mega binary trips the w4 whole-kernel miscompile family
        # (A/AN/ANP all hang ~30-50% regardless of knobs/transports).
        _kernel_replica_repush_store[(ncore(), 1, 1)](
            rref_gu_src, rref_dn_src,
            replica_w4, replica_fc2,
            repref_gate_ready, repref_down_ready,
            repref_etc, desc_ids, repref_desc_count,
            repref_epoch,
            LOCAL_RANK=rank, EPR=home_e,
            GU_ELEMS=gu_elems, DN_ELEMS=dn_elems,
            rb_ptr=rref_rb)
        # R1 (2026-09-21): collective fence between the re-push and the mega
        # kernel — every rank's re-push must complete (and be published)
        # before any rank's mega reads its replica slots.  This replaces the
        # in-kernel consumer dl.wait (REPREFETCH_WAIT is now always 0, mega
        # binary ≡ REPREFETCH=0) with a stream-ordered publication edge over
        # two small standalone kernels (transport-proven barrier).
        launch_replica_grad_barrier(ncore())
        if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
            # push-target addressing audit: every remote putmem/signal_op
            # resolves the peer's address by SYMMETRIC OFFSET — these four
            # offsets must be identical on every rank, else re-push writes
            # land in the peer's wrong slab (clobbered signals -> barrier
            # spin -> 507014).
            _b = peer_mem.data_ptr()
            print(
                f"[mega-heap2 r{rank} t={time.time():.2f}] "
                f"rfc2_off={replica_fc2.data_ptr() - _b} "
                f"rw4_off={replica_w4.data_ptr() - _b} "
                f"grdy_off={repref_gate_ready.data_ptr() - _b} "
                f"drdy_off={repref_down_ready.data_ptr() - _b} "
                f"epoch={repref_epoch} desc_n={repref_desc_count}",
                flush=True)
            # descriptor-compaction integrity audit: the compact kernel's
            # masked data-dependent scatter store is a documented 910B1
            # hazard (balanced_routing.py:51-59).  Rebuild the expected
            # owner-local descriptor list on the host and diff — a drop
            # here means some slot's push+SET never happens and vanilla
            # (wait-on) consumers spin forever.
            _flat = repref_etc_cpu.reshape(-1).tolist()
            _want = sorted(
                d for d in range(W * home_e)
                if 0 <= _flat[d] < W * home_e
                and _flat[d] // home_e == rank and d // home_e != rank
            )
            _got = sorted(
                int(v) for v in desc_ids[:repref_desc_count].cpu())
            print(
                f"[mega-heap3 r{rank} t={time.time():.2f}] "
                f"desc_match={_got == _want} want={len(_want)} "
                f"got={len(_got)} "
                f"missing={sorted(set(_want) - set(_got))[:8]}",
                flush=True)
            # consumer-demand vs push-coverage audit: every replica slot
            # this rank's backward m-tiles will dl.wait on must be covered
            # by a pushable ETC entry — an uncovered slot means the wait is
            # unsatisfiable BY CONSTRUCTION (w4 multilayer deadlock
            # candidate: all-core spin at dl.wait + B1).
            _hb = home_base4
            _demand = sorted({
                int(e) - _hb
                for e in tiles["tile_expert"].cpu().tolist()
                if int(e) >= _hb
            })
            _row = repref_etc_cpu[rank].tolist()
            _pushable = sorted(
                s for s in range(home_e)
                if 0 <= _row[s] < W * home_e and _row[s] // home_e != rank
            )
            print(
                f"[mega-heap4 r{rank} t={time.time():.2f}] "
                f"slot_demand={_demand} pushable={_pushable} "
                f"uncovered={[s for s in _demand if s not in _pushable]}",
                flush=True)
    else:
        repref_epoch = 0
        repref_desc_count = 0
        repref_etc = desc_ids = p1["recv_per_expert"]
        repref_gate_ready = repref_down_ready = signal_mem
        repref_gate_ready_u64 = repref_down_ready_u64 = \
            signal_mem[:2].view(torch.uint64)

    launch_options = (
        {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"}
        if cbm * cbn > 128 * 256 else {}
    )
    # Shared launch bindings: everything after the P5 group binds BY
    # KEYWORD (the signature's constexpr block (D_BM..FUSE_P4) sits between
    # the P5 group and the MoonEP tail, so positional binding would land on
    # the tile constexprs).  The two kernels share the entire pre-recompute
    # signature, so one args/kwargs pair serves both launch sites.
    mega_args = (
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
        # P5 (orig_in5 = the re-dispatch slab under SAVED_RECOMPUTE, the
        # padded torch clone of recv_hidden_sorted otherwise)
        orig_in5, in5_sm, in5_sk,
        grad_fc1, grad_fc1.stride(0), grad_fc1.stride(1), grad_fc1.stride(2),
        2 * ffn, H, num_tn5, num_tk5, w5_split, w5_total,
    )
    mega_kwargs = dict(
        fc1_scale_ptr=fc1_scale,
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
        MAX_BWD_TILES=sig_tiles_bound, LOCAL_RANK=rank,
        BLOCK_H_PUSH=256,
        BLOCK_SIZE=triton.next_power_of_2(ffn),
        ACTIVATION=act, HAS_LINEAR_BETA=has_lb,
        FC1_FP8=fc1_fp8, FC1_GROUP_N=fc1_group_n,
        W_BM=wbm, W_BN=wbn, W_BK=wbk, W_NS=wns,
        C_BM=cbm, C_BN=cbn, C_BK=cbk, C_NS=max(cns, 1),
        BLOCK_N_PUSH=_push_block(), GATE_PAD_C=GATE_PAD,
        P1_ON=_flag("MOE_MEGA_P1"), P23_ON=_flag("MOE_MEGA_P23"),
        P4_ON=_flag("MOE_MEGA_P4"), P5_ON=_flag("MOE_MEGA_P5"),
        b3_signal_ptr=b3_signal, b3_epoch=b3_epoch, TILE_B3=tile_b3,
        b1_signal_ptr=b1_signal, b1_epoch=b1_epoch, num_n_tiles1=num_n_tiles1,
        TILE_B1=tile_b1,
        FUSE_P4=fuse_p4 and _flag("MOE_MEGA_P4"),
        combine_buf_ptr=combine_buf,
        HOME_E=home_e, ACTIVE_E=active_e,
        GRAD_REDUCE=grad_reduce, EPN6=epn6,
        GU_CHUNK=gu_chunk6, DN_CHUNK=dn_chunk6,
        ACC_BLK=blk6, TM6=tm6, TN6=tn6, BLK6=blk6,
        ts_ptr=ts_buf, wait1_ptr=wait1_buf,
        TS_SLOTS=MEGA_TS_SLOTS, TIMING=timing_on,
        # re-prefetch (dead-arg pattern when REPREFETCH=False)
        repref_etc_ptr=repref_etc, repref_desc_ids_ptr=desc_ids,
        repref_desc_count=repref_desc_count,
        repref_gate_ready_ptr=repref_gate_ready,
        repref_down_ready_ptr=repref_down_ready,
        repref_epoch=repref_epoch,
        REPREFETCH=reprefetch,
        REPREFETCH_WAIT=reprefetch_wait,
        wait_dbg_ptr=wait_dbg, WAIT_DEBUG=wait_debug,
        rref_rb_ptr=rref_rb,
        rref_gu_src_ptr=rref_gu_src, rref_dn_src_ptr=rref_dn_src,
        repref_gate_ready_u64_ptr=repref_gate_ready_u64,
        repref_down_ready_u64_ptr=repref_down_ready_u64,
        RREF_GU_ELEMS=gu_elems, RREF_GU_CHUNK=gu_chunk, RREF_GU_NCHUNK=gu_nchunk,
        RREF_DN_ELEMS=dn_elems, RREF_DN_CHUNK=dn_chunk, RREF_DN_NCHUNK=dn_nchunk,
    )
    if saved_recompute:
        kernel_moe_backward_mega_recompute[(ncore(), 1, 1)](
            *mega_args, **mega_kwargs,
            redis_src_ptr=redis_src, stride_rm=redis_src.stride(0),
            send_src_idx_ptr=send_src_idx, redis_buf_ptr=orig_in5,
            SAVED_RECOMPUTE=True, REDIS_BM=64, REDIS_BN=1024,
            num_warps=8, **launch_options)
    else:
        # the ORIGINAL kernel (kept verbatim above): the non-recompute
        # path launches it exactly as before the fused variant existed
        kernel_moe_backward_mega[(ncore(), 1, 1)](
            *mega_args, **mega_kwargs,
            num_warps=8, **launch_options)
    # attribution probe: launch call returned (async) — with [mega-it] this
    # bounds the triton launch host overhead; with the NEXT [mega-ent] it
    # bounds the pure-host autograd segment after this launch.
    if wait_debug:
        _entries = wait_dbg.view(-1, 8).cpu()
        _nz = _entries[_entries[:, 0] > 0]
        if len(_nz):
            for _e in _nz.tolist():
                print(f"[mega-waitdbg r{rank}] TIMEOUT site={_e[0]} "
                      f"slot={_e[1]} want={_e[2]} observed={_e[3]} "
                      f"expert={_e[4]} spins={_e[5]}", flush=True)

    if os.environ.get("MOE_MEGA_HEAP_PROBE") == "1":
        print(f"[mega-post r{rank} t={time.time():.2f}]", flush=True)

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
            # barrier #3 (2026-09-20): publish the zeroing before any next
            # forward's owner-push — an unordered next push could otherwise
            # be wiped by this zeroing (the mirror race of barrier #2).
            # (w8 bisect arm1 cleared it of hang causation.)
            launch_replica_grad_barrier(grad_transport.num_barrier_programs)
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
