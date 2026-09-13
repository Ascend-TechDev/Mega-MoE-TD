# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""M0 probes for the one-kernel fused MoE backward (``MOE_BWD_MEGA``).

The fused backward kernel chains five phases inside ONE launch separated by
in-kernel ``libshmem_device.barrier_all()`` fences.  Two of those fences carry
handoffs that no in-repo kernel has proven yet:

* **cube -> vector**: phase 4a writes ``hidden_buf`` from ``al.scope(
  core_mode="cube")`` GEMM stores; phase 4b's vector scope then reads those
  rows.  Today every cube->vector handoff crosses a KERNEL boundary via a
  stream event (``combine_fc1_bwd.py:17-21``); the only documented in-kernel
  limitation is about cube-side GM *atomics*, not plain stores — but plain
  stores + ``barrier_all`` visibility is exactly what these probes must prove
  before the mega kernel is built on it.
* **remote putmem -> local loads across the barrier**: phase 4b pushes rows
  into PEER ``peer_mem`` and phase 4c reads them back after the fence.

Probe 1 validates both edges on a small self-contained mixed kernel:
vector-scope ``putmem`` of a per-program pattern row to rank ``(r+1)%W`` plus
a real per-program ``tl.dot`` GEMM tile, one barrier, then a vector-scope
verification pass that re-reads EVERY GEMM row (written by other programs'
cube scopes) and the pattern rows rank ``(r-1+W)%W`` pushed remotely.

Probe 1b repeats probe 1 with ``disable_auto_sync`` dropped from the scopes
(the scope-exit auto-sync fence is the first fallback if 1a fails).

Probe 1c repeats probe 1a with ``barrier_all_vec()`` instead of
``barrier_all()`` — only needed if 1a fails; the ``barrier_all_vec``
participation contract is documented for pure-vector kernels
(``replica_grad_reduce.py:50-59``) and may wedge inside a mixed kernel, so
this node is env-skippable (``MOE_PROBE_SKIP_BARRIER_VEC=1``) and the hang is
caught by ``DIST_TEST_TIMEOUT_S``.

Probe 2a chains FIVE ``barrier_all()`` calls with alternating mixed-scope work
between them — the exact barrier count of the mega kernel — as LITERAL source
phases with all cross-phase data flowing through GM, exactly how the mega
kernel is written.  (Probe 2b, the device-``for``-loop form with an SSA cube
accumulator carried across barriers, WEDGED all 8 ranks at w8 on 2026-09-04;
it is kept as the wedge record and skipped by default — collectives must not
sit inside a device-side loop.  Probe 2c is a vector-only bisect aid.)

Probe 3 gates the B1/B3 *signalization* plan (replacing those barriers with
per-tile signals, the "pipeline end-to-end" borrow): a cube-scope GEMM stores
its row-block, ``fence()`` (or ``quiet()`` via ``MOE_PROBE3_QUIET=1``), then
``signal_op SET`` to THIS rank's slot; a vector scope — where ``dl.wait`` has
never been exercised, P1 only waits in cube scopes — consumes each producer's
slot with ``dl.wait/consume_token`` and verifies that producer's rows.  NO
barrier anywhere; producer-first source order avoids the circular wait.
Pass = B1/B3 signalization is mechanically sound; wedge/fail = fall back to
the barrier chain (the shipped mega kernel is the fallback state).

Probe 4 gates the P4a+P4b FUSION (``MOE_MEGA_FUSE_P4``, the "self-produce-
self-push" tile loop): alternating cube/vector scopes INSIDE a device-side
``for`` loop — every iteration computes a dot tile in a cube scope (K-loop
with num_stages, mirroring the fused GEMM body), stores it, ``fence()`` (or
``quiet()`` via ``MOE_PROBE4_QUIET=1``), ``signal_op SET`` to THIS rank's
per-(pid, it) slot; the adjacent vector scope then ``dl.wait``s that slot,
``consume_token``s, and verifies the tile.  No barrier anywhere.  The pieces
are individually proven — probe 3: the local cube->vector fence/signal/wait
chain (single-shot); P1 production (dispatch_fc2_bwd.py:327-333): wait +
consume inside a task loop with multiple consumes per program — but always
within ONE scope; nothing alternates scopes per loop iteration.  A wedge or
mismatch means the fusion must stay two-phase (P4a; barrier/TILE_B3; P4b).

Probe 5 gates MOE_MEGA_UB_P4 (the forwardOne borrow): the tile crosses from
the cube scope to the SAME program's vector scope through an ON-CHIP UB
double buffer — ``al.fixpipe(acc, ub[p])`` + ``al.sync_block_set`` on the
cube side, ``al.sync_block_wait`` + ``bl.to_tensor(subview)`` + release on
the vector side, with 2-deep ping-pong backpressure (the producer waits the
consumer's release from two iterations back).  The tile never touches GM
and no signal slot exists — this is the FC1->activation handoff of the
single-kernel forward (origin/forwardOne ``fused_forward.py``) transplanted
into the mega backward's alternating-scope task loop.  A wedge or a stale
read means P4a+P4b must keep the GM ``hidden_buf`` handoff (probe 4's
regime) — the UB path would eliminate the tile's GM round-trip entirely.

Probe 5 RESULT (2026-09-09, 910B1 x8): the toolchain HARD-GATES ``al.fixpipe``
to Ascend910_95 — the extension's semantic layer raises "this feature is only
supported on Ascend910_95" (Fixpipe docstring: "L0C to UB (for Ascend910_95
series)"), and ``copy_from_ub_to_l1``/buffer ``copy`` carry the same gate — so
the UB direct handoff is A5-ONLY and cannot compile on any other part.  The
probe therefore asserts the gate on non-910_95 (the boundary is a TESTED
fact, not a red X) and keeps the numeric leg ready for A5.  ``sync_block_*``
and ``bl.alloc/subview/to_tensor`` are NOT gated; on A3 the realizable subset
of forwardOne's pipeline is the FUSE_P4 GM-handoff wave (probe 4).

Probe 6 gates the combine_buf step of the wave evolution (2026-09-10): the
backward's P4b return push writes the DESTINATION rank's buffer through
``dl.symm_at`` + ``tl.store``, and production only ever does that against
peer_mem, which the wrapper guarantees is the session's FIRST symmetric
allocation (heap offset 0 — the ``dl.symm_at only resolves at heap offset 0``
note in combine_fc1_bwd.py).  A separate return buffer (forwardOne's
combine_buf arrangement, the precondition for wave-overlapping the return
push with later dispatch) sits at a NON-zero heap offset, and BOTH remote
write forms are unproven there: symm_at never ran against a non-first slab,
and putmem (the P1 form) only ever wrote offset-0 peer_mem (P6b getmem only
READS offset-N).  With a dummy slab occupying offset 0 (misresolution lands
in it and is detected), the two forms run as SEPARATE kernels so a faulting
form wedges only its own test: 6a = putmem, 6b = symm_at + store.  A 6b
wedge/mismatch/corruption is the boundary record — combine_buf's push then
takes 6a's putmem form; 6a failing too means no remote-write form works at
offset N and the step is blocked at the transport level.

Decision tree (plan R1)
-----------------------
| result                              | decision                                   |
|-------------------------------------|--------------------------------------------|
| 1a passes                           | ship the mega kernel with barrier_all()    |
|                                     | and disable_auto_sync=True scopes          |
| 1a fails cube->vector only          | drop disable_auto_sync on the PRODUCING    |
|                                     | scope (probe 1b regime)                    |
| 1a fails remote-put visibility only | add explicit libshmem_device.fence()       |
|                                     | before the barrier in P1a/P4b              |
| 1a + 1b fail, 1c passes             | switch the mega kernel to barrier_all_vec  |
| all fail                            | de-scope to two launches (P1..P4a /        |
|                                     | P4b..P5), stream order replaces B3-B5      |

Static finding baked into this file: ``al.scope`` kwargs are read from the
AST as LITERALS ONLY (``cann/extension/code_generator.py:63-69`` only accepts
``ast.Constant`` values) — ``disable_auto_sync=SOME_CONSTEXPR`` is silently
dropped.  The sync-mode A/B therefore uses a constexpr ``if`` around two
literal ``with`` statements, and the mega kernel must keep literal
``disable_auto_sync=True`` at every scope.

Second compile-time constraint (found by probe 2a's first draft): reducing a
``tl.dot`` result INSIDE the cube scope (``tl.sum(tl.dot(...))``) trips a
bishengir sync-solver assertion (``CUBE_OR_VECTOR`` core type,
SyncSolverIRTranslator.cpp:509).  Cube-scope bodies must store dot tiles to
GM; reductions belong to a vector-scope phase (as all production kernels
already do).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl  # noqa: F401  (peer_mem symmetric mapping)
from triton_dist.language.extra import libshmem_device
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from mega_moe.kernels.common import ncore
from triton.backends.ascend.driver import NPUUtils
from tests import _moe_testkit as kit

# Probe-1 GEMM shape: one BLOCK_M row-block per program (grid == ncore()), so
# every element of gemm_out is written by exactly one program's cube scope and
# the verification pass exercises cross-program visibility on every row.
PROBE_BLOCK_M = 64
PROBE_N = 256
PROBE_K = 256
PROBE_BLOCK_N = 256
PROBE_BLOCK_K = 64
# Per-program remote pattern block (int32 elements pushed with one putmem).
PROBE_PATN = 16
# Probe-2 scratch columns per program (int32 increments + one cube dot result).
CHAIN_COLS = 16
GEMM_RTOL = 2e-2
GEMM_ATOL = 1e-2


@triton.jit
def _probe_pattern_value(rank, pid, e):
    """Deterministic pattern: rank*1000003 + pid*131 + e*7 + 12345 (< 2^31)."""
    return rank * 1000003 + pid * 131 + e * 7 + 12345


@triton.jit
def _probe_push_body(
    pid,
    pat_ptr, peer_mem_ptr,
    dst_rank,
    NPROG: tl.constexpr, PATN: tl.constexpr,
):
    """Vector-scope producer: one putmem of this program's pattern block into
    rank dst_rank's peer_mem slot ``dst_rank*NPROG + pid`` + fence."""
    libshmem_device.putmem(
        peer_mem_ptr + (dst_rank * NPROG + pid) * PATN,
        pat_ptr + pid * PATN,
        PATN * 4, dst_rank)
    libshmem_device.fence()


@triton.jit
def _probe_gemm_body(
    pid,
    a_ptr, b_ptr, gemm_out_ptr,
    N, K,
    stride_am, stride_bk, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
):
    """Cube-scope producer: gemm_out[pid*BLOCK_M, :] = A[...] @ B (row-block
    per program, K-loop, no masks — the probe shapes divide evenly)."""
    om = tl.arange(0, BLOCK_M)
    ok = tl.arange(0, BLOCK_K)
    on_ = tl.arange(0, BLOCK_N)
    row0 = pid * BLOCK_M
    for nt in range(NUM_N_TILES):
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ks in range(0, K, BLOCK_K):
            a = tl.load(a_ptr + (row0 + om)[:, None] * stride_am + (ks + ok)[None, :])
            b = tl.load(b_ptr + (ks + ok)[:, None] * stride_bk + (nt * BLOCK_N + on_)[None, :])
            acc += tl.dot(a, b)
        tl.store(gemm_out_ptr + (row0 + om)[:, None] * stride_om + (nt * BLOCK_N + on_)[None, :],
                 acc.to(gemm_out_ptr.dtype.element_ty))


@triton.jit
def _probe_verify_body(
    pid,
    gemm_out_ptr, golden_ptr, peer_mem_ptr, flags_ptr,
    T, N, src_rank,
    stride_om,
    BLOCK_N: tl.constexpr, PATN: tl.constexpr, NPROG: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    RTOL: tl.constexpr, ATOL: tl.constexpr,
):
    """Vector-scope consumer (after the barrier): re-read EVERY gemm_out row
    (all row-blocks were written by other/all programs' cube scopes) against
    the fp32 golden, and the pattern block rank src_rank pushed into OUR
    peer_mem.  Slot arithmetic: rank d's peer_mem region [d*NPROG,
    (d+1)*NPROG) holds data destined FOR d (written by rank (d-1+W)%W), so the
    reader must index by ITS OWN rank — NOT by src_rank (that block in our
    buffer was never written and stays zero).  Writes flags[pid, 0] = gemm
    mismatch count, [pid, 1] = pattern mismatch count."""
    offs_n = tl.arange(0, BLOCK_N)
    bad_gemm = 0  # python-int init + tensor accumulate (ready_token idiom)
    for r in range(T):
        out = tl.load(gemm_out_ptr + r * stride_om + offs_n).to(tl.float32)
        gold = tl.load(golden_ptr + r * N + offs_n)
        diff = tl.abs(out - gold)
        bad_gemm += tl.sum((diff > (ATOL + RTOL * tl.abs(gold))).to(tl.int32))
    tl.store(flags_ptr + pid * 2, bad_gemm)
    offs_e = tl.arange(0, PATN)
    got = tl.load(peer_mem_ptr + (LOCAL_RANK * NPROG + pid) * PATN + offs_e)
    want = _probe_pattern_value(src_rank, pid, offs_e)
    tl.store(flags_ptr + pid * 2 + 1, tl.sum((got != want).to(tl.int32)))


@triton.jit(do_not_specialize=["dst_rank", "src_rank"])
def kernel_probe_mixed_visibility(
    a_ptr, b_ptr, gemm_out_ptr, golden_ptr,
    pat_ptr, peer_mem_ptr, flags_ptr,
    T, N, K, dst_rank, src_rank,
    stride_am, stride_bk, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr,
    NPROG: tl.constexpr, PATN: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    RTOL: tl.constexpr, ATOL: tl.constexpr,
    BARRIER_VEC: tl.constexpr, ASYNC_SCOPES: tl.constexpr,
):
    """Probe 1: mixed vector(putmem)+cube(GEMM) scopes -> ONE in-kernel barrier
    -> vector verification of both handoff edges.  See module docstring."""
    pid = tl.program_id(axis=0)

    # NOTE: al.scope kwargs are AST literals only (code_generator.py:63-69) —
    # a constexpr disable_auto_sync argument is silently dropped, so the sync
    # A/B is a constexpr if over two literal `with` statements.
    if ASYNC_SCOPES:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _probe_push_body(pid, pat_ptr, peer_mem_ptr, dst_rank,
                                 NPROG=NPROG, PATN=PATN)
    else:
        with al.scope(core_mode="vector"):
            if sub_vec_id() == 0:
                _probe_push_body(pid, pat_ptr, peer_mem_ptr, dst_rank,
                                 NPROG=NPROG, PATN=PATN)

    if ASYNC_SCOPES:
        with al.scope(core_mode="cube", disable_auto_sync=True):
            _probe_gemm_body(pid, a_ptr, b_ptr, gemm_out_ptr, N, K,
                             stride_am, stride_bk, stride_om,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                             NUM_N_TILES=NUM_N_TILES)
    else:
        with al.scope(core_mode="cube"):
            _probe_gemm_body(pid, a_ptr, b_ptr, gemm_out_ptr, N, K,
                             stride_am, stride_bk, stride_om,
                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                             NUM_N_TILES=NUM_N_TILES)

    if BARRIER_VEC:
        libshmem_device.barrier_all_vec()
    else:
        libshmem_device.barrier_all()

    # Verification runs on BOTH vector subcores (the mega kernel's swiglu
    # phase runs ungated exactly like this; duplicate loads are idempotent).
    if ASYNC_SCOPES:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            _probe_verify_body(pid, gemm_out_ptr, golden_ptr, peer_mem_ptr,
                               flags_ptr, T, N, src_rank, stride_om,
                               BLOCK_N=BLOCK_N, PATN=PATN, NPROG=NPROG,
                               LOCAL_RANK=LOCAL_RANK,
                               RTOL=RTOL, ATOL=ATOL)
    else:
        with al.scope(core_mode="vector"):
            _probe_verify_body(pid, gemm_out_ptr, golden_ptr, peer_mem_ptr,
                               flags_ptr, T, N, src_rank, stride_om,
                               BLOCK_N=BLOCK_N, PATN=PATN, NPROG=NPROG,
                               LOCAL_RANK=LOCAL_RANK,
                               RTOL=RTOL, ATOL=ATOL)


@triton.jit
def kernel_probe_barrier_chain_unrolled(
    x_ptr, y_ptr,           # [16,16] bf16 dot inputs (cube phase)
    scratch_ptr,            # int32 [NPROG, CHAIN_COLS] increments (vector phase)
    dot_out_ptr,            # fp32 [NPROG, 5, 16, 16] cube dot tiles (DCE guard)
    NPROG: tl.constexpr, CHAIN_COLS: tl.constexpr,
):
    """Probe 2a (primary): FIVE barrier_all() calls with mixed-scope work
    between them — structurally identical to the mega kernel: the phases are
    LITERAL source blocks (no device-side loop) and every cross-phase value
    flows through GM, never as an SSA value carried across a barrier.  Each
    phase = vector increment (sub_vec0-gated) + cube dot stored to its own
    dot_out column + barrier_all.  A wedge hangs the worker (caught by
    DIST_TEST_TIMEOUT_S); a lost increment fails the scratch==5 check."""
    pid = tl.program_id(axis=0)
    om = tl.arange(0, 16)
    offs = tl.arange(0, CHAIN_COLS)

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
            tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
        yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
        tl.store(dot_out_ptr + (pid * 5 + 0) * 256 + om[:, None] * 16 + om[None, :],
                 tl.dot(x, yv))
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
            tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
        yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
        tl.store(dot_out_ptr + (pid * 5 + 1) * 256 + om[:, None] * 16 + om[None, :],
                 tl.dot(x, yv))
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
            tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
        yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
        tl.store(dot_out_ptr + (pid * 5 + 2) * 256 + om[:, None] * 16 + om[None, :],
                 tl.dot(x, yv))
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
            tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
        yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
        tl.store(dot_out_ptr + (pid * 5 + 3) * 256 + om[:, None] * 16 + om[None, :],
                 tl.dot(x, yv))
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
            tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
        yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
        tl.store(dot_out_ptr + (pid * 5 + 4) * 256 + om[:, None] * 16 + om[None, :],
                 tl.dot(x, yv))
    libshmem_device.barrier_all()


@triton.jit
def kernel_probe_barrier_chain_vec_only(
    scratch_ptr,            # int32 [NPROG, CHAIN_COLS] increments
    NPROG: tl.constexpr, CHAIN_COLS: tl.constexpr,
):
    """Probe 2c (bisect aid): five barrier_all() calls with VECTOR-scope work
    only — isolates the (grid programs x barrier count) axis from the mixed
    scope/cube axis.  If 2a wedges but 2c passes, the culprit is the cube-side
    phase structure, not the barrier chain itself."""
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, CHAIN_COLS)
    for phase in range(5):
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
                tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
        libshmem_device.barrier_all()


@triton.jit
def kernel_probe_barrier_chain(
    x_ptr, y_ptr,           # [16,16] bf16 dot inputs (cube phase)
    scratch_ptr,            # int32 [NPROG, CHAIN_COLS] increments (vector phase)
    dot_out_ptr,            # fp32 [NPROG] cube acc landing (DCE guard)
    NPROG: tl.constexpr, CHAIN_COLS: tl.constexpr,
):
    """Probe 2b (wedge record, off by default): the loop form — five
    barrier_all() calls inside a DEVICE-side ``for phase in range(5)`` loop
    with the cube accumulator carried across barriers as an SSA value.  This
    wedged all 8 ranks at w8 (2026-09-04, workers killed at 240s) while the
    literal-unrolled 2a form passes, so collectives must NOT sit inside a
    device-side loop.  The mega kernel writes its phases as literal source
    blocks and carries nothing across barriers, which is exactly what 2a
    proves safe.  Run with MOE_PROBE_RUN_CHAIN_LOOP=1."""
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, CHAIN_COLS)
    om = tl.arange(0, 16)
    acc = tl.zeros((16, 16), dtype=tl.float32)
    for phase in range(5):
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                # non-atomic RMW on this program's OWN row: one lane, safe
                cur = tl.load(scratch_ptr + pid * CHAIN_COLS + offs)
                tl.store(scratch_ptr + pid * CHAIN_COLS + offs, cur + 1)
        with al.scope(core_mode="cube", disable_auto_sync=True):
            x = tl.load(x_ptr + om[:, None] * 16 + om[None, :])
            yv = tl.load(y_ptr + om[:, None] * 16 + om[None, :])
            acc += tl.dot(x, yv)
        libshmem_device.barrier_all()
    tl.store(dot_out_ptr + pid, tl.sum(acc))


def _fold_and_raise(failures, label, rank, device, ep_group):
    """MIN-fold the local verdict across the EP group and raise on any failure
    (local copy of tests/fstage/test_f0b_probes.py:_fold_and_raise)."""
    ok = not failures
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if rank == 0 and failures:
        print(f"[FAIL] {label}: " + "; ".join(failures), flush=True)
    if not bool(flag.item()):
        detail = "; ".join(failures) if failures else "a peer rank failed"
        raise AssertionError(f"{label}: {detail}")


def _require_runtime(label):
    if kit.ash is None:
        raise RuntimeError(f"{label} requires torch_npu and ACLSHMEM")


def _run_probe_mixed(rank: int, world_size: int, *, barrier_vec: bool,
                     async_scopes: bool, label: str) -> None:
    """Probe-1 driver shared by variants 1a/1b/1c."""
    _require_runtime(label)
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        # peer_mem FIRST symmetric allocation (dl.symm_at offset-0 discipline)
        peer_mem = kit.ash.aclshmem_create_tensor(
            [world_size * nprog * PROBE_PATN], dtype=torch.int32, device_id=rank)
        try:
            peer_mem.zero_()
            torch.manual_seed(2115 + rank)
            T = nprog * PROBE_BLOCK_M
            a = torch.randn(T, PROBE_K, dtype=torch.bfloat16, device=device)
            b = torch.randn(PROBE_K, PROBE_N, dtype=torch.bfloat16, device=device)
            gemm_out = torch.empty(T, PROBE_N, dtype=torch.bfloat16, device=device)
            golden = a.float() @ b.float()
            e = torch.arange(PROBE_PATN, dtype=torch.int32, device=device)
            pids = torch.arange(nprog, dtype=torch.int32, device=device)
            pat = (rank * 1000003 + pids[:, None] * 131 + e[None, :] * 7
                   + 12345).to(torch.int32).contiguous()
            flags = torch.full((nprog, 2), -1, dtype=torch.int32, device=device)
            dist.barrier()

            kernel_probe_mixed_visibility[(nprog, 1, 1)](
                a, b, gemm_out, golden, pat, peer_mem, flags,
                T, PROBE_N, PROBE_K,
                (rank + 1) % world_size, (rank - 1 + world_size) % world_size,
                a.stride(0), b.stride(0), gemm_out.stride(0),
                BLOCK_M=PROBE_BLOCK_M, BLOCK_N=PROBE_BLOCK_N,
                BLOCK_K=PROBE_BLOCK_K, NUM_N_TILES=PROBE_N // PROBE_BLOCK_N,
                NPROG=nprog, PATN=PROBE_PATN,
                LOCAL_RANK=rank,
                RTOL=GEMM_RTOL, ATOL=GEMM_ATOL,
                BARRIER_VEC=barrier_vec, ASYNC_SCOPES=async_scopes,
                num_warps=8)
            torch.npu.synchronize()

            failures = []
            bad_gemm = int(flags[:, 0].sum().item())
            bad_pat = int(flags[:, 1].sum().item())
            unverified = int((flags[:, 0] < 0).sum().item())
            if bad_gemm:
                worst = int(flags[:, 0].max().item())
                failures.append(
                    f"cube->vector GM visibility across the barrier failed: "
                    f"{bad_gemm} mismatched elements (worst program {worst})")
            if bad_pat:
                failures.append(
                    f"remote putmem visibility across the barrier failed: "
                    f"{bad_pat} mismatched pattern ints")
            if unverified:
                failures.append(
                    f"{unverified} programs never stored a gemm verdict "
                    f"(kernel did not run to completion?)")
            _fold_and_raise(failures, label, rank, device, ep_group)
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


def run_mega_probe1_mixed_visibility(rank: int, world_size: int) -> None:
    """1a (primary): barrier_all() + disable_auto_sync=True — the mega design."""
    _run_probe_mixed(rank, world_size, barrier_vec=False, async_scopes=True,
                     label=f"mega-probe1a-mixed-vis-w{world_size}")


def run_mega_probe1b_auto_sync(rank: int, world_size: int) -> None:
    """1b (fallback regime): same probe without disable_auto_sync."""
    _run_probe_mixed(rank, world_size, barrier_vec=False, async_scopes=False,
                     label=f"mega-probe1b-auto-sync-w{world_size}")


def run_mega_probe1c_barrier_vec(rank: int, world_size: int) -> None:
    """1c (only if 1a fails): barrier_all_vec() inside the mixed kernel."""
    _run_probe_mixed(rank, world_size, barrier_vec=True, async_scopes=True,
                     label=f"mega-probe1c-barrier-vec-w{world_size}")


def run_mega_probe2_barrier_chain(rank: int, world_size: int) -> None:
    """Probe 2a (primary): five literal in-kernel barrier_all() phases with
    mixed-scope work between — the mega kernel's structure."""
    _require_runtime("mega probe2")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    label = f"mega-probe2a-barrier-chain-w{world_size}"

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        torch.manual_seed(2215 + rank)
        x = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
        y = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
        scratch = torch.zeros(nprog, CHAIN_COLS, dtype=torch.int32, device=device)
        dot_out = torch.zeros(nprog, 5, 16, 16, dtype=torch.float32, device=device)
        dist.barrier()

        kernel_probe_barrier_chain_unrolled[(nprog, 1, 1)](
            x, y, scratch, dot_out,
            NPROG=nprog, CHAIN_COLS=CHAIN_COLS, num_warps=8)
        torch.npu.synchronize()

        failures = []
        if not bool((scratch == 5).all().item()):
            bad = int((scratch != 5).sum().item())
            failures.append(
                f"barrier chain lost vector-phase increments: {bad} scratch "
                f"cells != 5 (min={int(scratch.min().item())}, "
                f"max={int(scratch.max().item())})")
        if not bool(torch.isfinite(dot_out).all().item()):
            failures.append("cube-phase dot accumulator went non-finite")
        _fold_and_raise(failures, label, rank, device, ep_group)


def run_mega_probe2c_barrier_vec_only(rank: int, world_size: int) -> None:
    """Probe 2c (bisect aid, skip by default): vector-only work between the
    five barriers — run it if 2a wedges to split the loop/grid axis from the
    mixed-scope axis."""
    _require_runtime("mega probe2c")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    label = f"mega-probe2c-barrier-vec-only-w{world_size}"

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        scratch = torch.zeros(nprog, CHAIN_COLS, dtype=torch.int32, device=device)
        dist.barrier()

        kernel_probe_barrier_chain_vec_only[(nprog, 1, 1)](
            scratch, NPROG=nprog, CHAIN_COLS=CHAIN_COLS, num_warps=8)
        torch.npu.synchronize()

        failures = []
        if not bool((scratch == 5).all().item()):
            bad = int((scratch != 5).sum().item())
            failures.append(
                f"vector-only barrier chain lost increments: {bad} scratch "
                f"cells != 5 (min={int(scratch.min().item())}, "
                f"max={int(scratch.max().item())})")
        _fold_and_raise(failures, label, rank, device, ep_group)


# ---------------------------------------------------------------------------
# Probe 3: LOCAL cube->vector per-tile signal (no barrier at all)
# ---------------------------------------------------------------------------
# The remote direction of this pipeline is production-proven (P1: vector
# putmem+fence+signal_op SET -> cube dl.wait/consume_token, dispatch_fc2_bwd).
# The B1/B3 signalization plan needs the LOCAL direction, which nothing in the
# repo exercises yet: a CUBE scope's plain GM stores (FixPipe path) -> fence()
# -> signal_op SET to THIS rank's slot -> a VECTOR scope's dl.wait (also
# unproven in a vector scope; P1 only waits in cube scopes) -> GM loads of the
# producer's rows.  Producer-first source order avoids the circular wait (every
# program's first phase requires nothing), mirroring how a signalized P4a->P4b
# would sit in the mega kernel.
@triton.jit(do_not_specialize=["signal_epoch"])
def kernel_probe_local_signal(
    a_ptr, b_ptr, gemm_out_ptr, golden_ptr, signal_mem_ptr, flags_ptr,
    T, N, K, signal_epoch,
    stride_am, stride_bk, stride_om,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_N_TILES: tl.constexpr, NPROG: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    RTOL: tl.constexpr, ATOL: tl.constexpr, QUIET_FENCE: tl.constexpr,
):
    """Probe 3: cube GEMM row-block + local signal, then vector verify gated
    per producer signal — NO barrier_all anywhere.  flags[pid, 0] = mismatched
    element count across ALL producers' rows, [pid, 1] = waits completed."""
    pid = tl.program_id(axis=0)

    # producer first: nothing it does can block on another program
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _probe_gemm_body(pid, a_ptr, b_ptr, gemm_out_ptr, N, K,
                         stride_am, stride_bk, stride_om,
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                         NUM_N_TILES=NUM_N_TILES)
        # QUIET_FENCE is the fallback rung if fence() does not order the
        # FixPipe store before the signal (quiet drains all engines).
        if QUIET_FENCE:
            libshmem_device.quiet()
        else:
            libshmem_device.fence()
        libshmem_device.signal_op(
            signal_mem_ptr + pid * 16, signal_epoch,
            libshmem_device.ACLSHMEM_SIGNAL_SET, LOCAL_RANK)

    # consumer: wait each producer's slot in order, then verify THAT
    # producer's row-block immediately (pipelined consumption, the shape a
    # signalized B1/B3 would use).  sub_vec0-gated: dl.wait/consume_token must
    # not run twice per program (token accounting).
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            offs_n = tl.arange(0, BLOCK_N)
            bad = 0  # python-int init + tensor accumulate (ready_token idiom)
            token = 0
            for src in range(NPROG):
                token += dl.wait(signal_mem_ptr + src * 16, 1, "gpu",
                                 "acquire", waitValue=signal_epoch)
                ready_ptr = dl.consume_token(gemm_out_ptr, token)
                row0 = src * BLOCK_M
                for r in range(BLOCK_M):
                    out = tl.load(ready_ptr + (row0 + r) * stride_om
                                  + offs_n).to(tl.float32)
                    gold = tl.load(golden_ptr + (row0 + r) * N + offs_n)
                    diff = tl.abs(out - gold)
                    bad += tl.sum((diff > (ATOL + RTOL * tl.abs(gold)))
                                  .to(tl.int32))
            tl.store(flags_ptr + pid * 2, bad)
            tl.store(flags_ptr + pid * 2 + 1, NPROG)


def run_mega_probe3_local_signal(rank: int, world_size: int) -> None:
    """Probe 3 driver: local cube->vector signal visibility, no barrier.
    MOE_PROBE3_QUIET=1 swaps fence() for quiet() (fallback rung)."""
    _require_runtime("mega probe3")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    quiet = os.environ.get("MOE_PROBE3_QUIET") == "1"
    label = f"mega-probe3-local-signal-w{world_size}"

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        signal_mem = kit.ash.aclshmem_create_tensor(
            [nprog * 16], dtype=torch.int32, device_id=rank)
        try:
            signal_mem.zero_()
            torch.manual_seed(2315 + rank)
            T = nprog * PROBE_BLOCK_M
            a = torch.randn(T, PROBE_K, dtype=torch.bfloat16, device=device)
            b = torch.randn(PROBE_K, PROBE_N, dtype=torch.bfloat16, device=device)
            gemm_out = torch.empty(T, PROBE_N, dtype=torch.bfloat16, device=device)
            golden = a.float() @ b.float()
            flags = torch.full((nprog, 2), -1, dtype=torch.int32, device=device)
            dist.barrier()

            kernel_probe_local_signal[(nprog, 1, 1)](
                a, b, gemm_out, golden, signal_mem, flags,
                T, PROBE_N, PROBE_K, 1,
                a.stride(0), b.stride(0), gemm_out.stride(0),
                BLOCK_M=PROBE_BLOCK_M, BLOCK_N=PROBE_BLOCK_N,
                BLOCK_K=PROBE_BLOCK_K, NUM_N_TILES=PROBE_N // PROBE_BLOCK_N,
                NPROG=nprog, LOCAL_RANK=rank,
                RTOL=GEMM_RTOL, ATOL=GEMM_ATOL, QUIET_FENCE=quiet,
                num_warps=8)
            torch.npu.synchronize()

            failures = []
            bad_gemm = int(flags[:, 0].sum().item())
            unverified = int((flags[:, 1] < 0).sum().item())
            if bad_gemm:
                worst = int(flags[:, 0].max().item())
                failures.append(
                    f"local cube->vector signal visibility failed: {bad_gemm} "
                    f"mismatched elements (worst program {worst}); "
                    f"fence({'quiet' if quiet else 'fence'} did not order "
                    f"the cube store before the signal)")
            if unverified:
                failures.append(
                    f"{unverified} programs never stored a verdict "
                    f"(kernel did not run to completion?)")
            _fold_and_raise(failures, label, rank, device, ep_group)
        finally:
            kit.ash.aclshmem_free_tensor(signal_mem)


# ---------------------------------------------------------------------------
# Probe 4: fused tile loop — alternating cube/vector scopes INSIDE a device
# side for-loop, with the per-iteration LOCAL self-signal chain (cube store
# -> fence -> signal_op own slot; vector wait/consume/load).  Gates
# MOE_MEGA_FUSE_P4: probe 3 proved the chain single-shot, P1 proves
# wait+consume inside a task loop, but both keep ONE scope for the whole
# loop — per-iteration scope ALTERNATION is the only unproven piece.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["signal_epoch"])
def kernel_probe_fused_loop(
    a_ptr, b_ptr, out_ptr, golden_ptr, signal_mem_ptr, flags_ptr,
    NITER, signal_epoch,
    NPROG: tl.constexpr, LOCAL_RANK: tl.constexpr,
    RTOL: tl.constexpr, ATOL: tl.constexpr, QUIET_FENCE: tl.constexpr,
):
    """Probe 4: per iteration a cube scope computes one 16x16 dot tile
    (K-loop with num_stages, the fused GEMM body's shape), stores, fence/
    quiet, signal_op SET to THIS rank's slot (pid*NITER+it); the adjacent
    vector scope (sub_vec0) dl.waits that slot, consume_token, verifies the
    tile vs the fp32 golden.  NO barrier anywhere.  flags[pid, 0] = mismatched
    elements over all iterations, flags[pid, 1] = iterations completed."""
    pid = tl.program_id(axis=0)
    om = tl.arange(0, 16)
    bad = 0  # python-int init + tensor accumulate (probe-3 idiom)
    token = 0
    for it in range(NITER):
        with al.scope(core_mode="cube", disable_auto_sync=True):
            base = (pid * NITER + it) * 256
            acc = tl.zeros((16, 16), dtype=tl.float32)
            for ks in tl.range(0, 16, 8, num_stages=2):
                ok8 = ks + tl.arange(0, 8)
                x = tl.load(a_ptr + base + om[:, None] * 16 + ok8[None, :])
                yv = tl.load(b_ptr + ok8[:, None] * 16 + om[None, :])
                acc += tl.dot(x, yv)
            tl.store(out_ptr + base + om[:, None] * 16 + om[None, :],
                     acc.to(out_ptr.dtype.element_ty))
            # QUIET_FENCE is the fallback rung if fence() does not order the
            # FixPipe store before the per-iteration signal (probe 3 rung).
            if QUIET_FENCE:
                libshmem_device.quiet()
            else:
                libshmem_device.fence()
            libshmem_device.signal_op(
                signal_mem_ptr + (pid * NITER + it) * 16, signal_epoch,
                libshmem_device.ACLSHMEM_SIGNAL_SET, LOCAL_RANK)
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                # recompute `base` here: no SSA value needs to cross a scope
                # boundary when it is this cheap (loop index only).
                base = (pid * NITER + it) * 256
                token += dl.wait(
                    signal_mem_ptr + (pid * NITER + it) * 16,
                    1, "gpu", "acquire", waitValue=signal_epoch)
                ready_ptr = dl.consume_token(out_ptr, token)
                got = tl.load(
                    ready_ptr + base + om[:, None] * 16 + om[None, :]
                ).to(tl.float32)
                gold = tl.load(
                    golden_ptr + base + om[:, None] * 16 + om[None, :])
                diff = tl.abs(got - gold)
                bad += tl.sum((diff > (ATOL + RTOL * tl.abs(gold)))
                              .to(tl.int32))
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            tl.store(flags_ptr + pid * 2, bad)
            tl.store(flags_ptr + pid * 2 + 1, NITER)


def run_mega_probe4_fused_loop(rank: int, world_size: int) -> None:
    """Probe 4 driver: fused scope-alternation loop + per-iteration local
    self-signal (the MOE_MEGA_FUSE_P4 gate).  MOE_PROBE4_QUIET=1 swaps
    fence() for quiet(); MOE_PROBE4_ITERS overrides the iteration count."""
    _require_runtime("mega probe4")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    niter = int(os.environ.get("MOE_PROBE4_ITERS", "8"))
    quiet = os.environ.get("MOE_PROBE4_QUIET") == "1"
    label = f"mega-probe4-fused-loop-w{world_size}"

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        signal_mem = kit.ash.aclshmem_create_tensor(
            [nprog * niter * 16], dtype=torch.int32, device_id=rank)
        try:
            signal_mem.zero_()
            torch.manual_seed(2415 + rank)
            a = torch.randn(
                nprog * niter, 16, 16, dtype=torch.bfloat16, device=device)
            b = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
            out = torch.empty_like(a)
            golden = a.float() @ b.float()   # (N,16,16)@(16,16) broadcasts
            flags = torch.full((nprog, 2), -1, dtype=torch.int32, device=device)
            dist.barrier()

            kernel_probe_fused_loop[(nprog, 1, 1)](
                a, b, out, golden, signal_mem, flags,
                niter, 1,
                NPROG=nprog, LOCAL_RANK=rank,
                RTOL=GEMM_RTOL, ATOL=GEMM_ATOL, QUIET_FENCE=quiet,
                num_warps=8)
            torch.npu.synchronize()

            failures = []
            bad = int(flags[:, 0].sum().item())
            done = int(flags[:, 1].max().item())
            unverified = int((flags[:, 1] < 0).sum().item())
            if bad:
                worst = int(flags[:, 0].max().item())
                failures.append(
                    f"fused-loop self-signal visibility failed: {bad} "
                    f"mismatched elements (worst program {worst}); "
                    f"fence({'quiet' if quiet else 'fence'}) did not order "
                    f"the cube store before the per-iteration signal")
            if unverified or done != niter:
                failures.append(
                    f"{unverified} programs never stored a verdict or the "
                    f"loop short-ran (max iterations seen {done}, want "
                    f"{niter}) — scope alternation inside a device loop "
                    f"wedged or miscompiled")
            _fold_and_raise(failures, label, rank, device, ep_group)
        finally:
            kit.ash.aclshmem_free_tensor(signal_mem)


def run_mega_probe2b_barrier_chain_loop(rank: int, world_size: int) -> None:
    """Probe 2b (wedge record, off by default): the device-loop form that
    wedged all ranks at w8 — kept to document the constraint, re-enabled with
    MOE_PROBE_RUN_CHAIN_LOOP=1."""
    _require_runtime("mega probe2b")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    label = f"mega-probe2b-barrier-chain-loop-w{world_size}"

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        torch.manual_seed(2215 + rank)
        x = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
        y = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
        scratch = torch.zeros(nprog, CHAIN_COLS, dtype=torch.int32, device=device)
        dot_out = torch.zeros(nprog, dtype=torch.float32, device=device)
        dist.barrier()

        kernel_probe_barrier_chain[(nprog, 1, 1)](
            x, y, scratch, dot_out,
            NPROG=nprog, CHAIN_COLS=CHAIN_COLS, num_warps=8)
        torch.npu.synchronize()

        failures = []
        if not bool((scratch == 5).all().item()):
            bad = int((scratch != 5).sum().item())
            failures.append(
                f"barrier chain lost vector-phase increments: {bad} scratch "
                f"cells != 5 (min={int(scratch.min().item())}, "
                f"max={int(scratch.max().item())})")
        if not bool(torch.isfinite(dot_out).all().item()):
            failures.append("cube-phase dot accumulator went non-finite")
        _fold_and_raise(failures, label, rank, device, ep_group)


# ---------------------------------------------------------------------------
# pytest entries (8-card; a wedge inside a variant is caught by
# DIST_TEST_TIMEOUT_S and the conftest straggler kill)
# ---------------------------------------------------------------------------

@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe1_mixed_visibility(dist_test):
    dist_test(run_mega_probe1_mixed_visibility, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe1b_auto_sync(dist_test):
    dist_test(run_mega_probe1b_auto_sync, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe1c_barrier_vec(dist_test):
    if os.environ.get("MOE_PROBE_SKIP_BARRIER_VEC") == "1":
        pytest.skip("MOE_PROBE_SKIP_BARRIER_VEC=1 (barrier_all_vec wedge known)")
    dist_test(run_mega_probe1c_barrier_vec, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe2_barrier_chain(dist_test):
    """Probe 2a: five literal barrier_all() phases, mixed scopes (primary)."""
    dist_test(run_mega_probe2_barrier_chain, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe2c_barrier_vec_only(dist_test):
    if os.environ.get("MOE_PROBE_SKIP_VEC_ONLY") == "1":
        pytest.skip("MOE_PROBE_SKIP_VEC_ONLY=1 (bisect aid, off by default)")
    dist_test(run_mega_probe2c_barrier_vec_only, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe2b_barrier_chain_loop(dist_test):
    if os.environ.get("MOE_PROBE_RUN_CHAIN_LOOP") != "1":
        pytest.skip("loop-form chain wedged all ranks at w8; "
                    "set MOE_PROBE_RUN_CHAIN_LOOP=1 to reproduce")


@triton.jit
def kernel_probe5_ub_pingpong(
    a_ptr, b_ptr, golden_ptr, flags_ptr,
    NITER,
    NPROG: tl.constexpr, LOCAL_RANK: tl.constexpr,
    RTOL: tl.constexpr, ATOL: tl.constexpr,
):
    """Probe 5: per iteration a cube scope computes one 16x16 dot tile
    (probe 4's math) and hands it to the SAME program's vector scope through
    an on-chip UB double buffer -- al.fixpipe(acc, ub[p]) +
    al.sync_block_set('cube','vector',8+p); the vector scope (sub_vec0)
    sync_block_waits, bl.to_tensor's the buffer back, verifies it against the
    golden, then releases the slot with sync_block_set('vector','cube',10+p);
    the cube side waits that release before overwriting the same ping-pong
    slot two iterations later (2-deep backpressure).  The tile NEVER touches
    GM and NO signal slot exists -- this is the forwardOne single-kernel
    forward's FC1->activation handoff mechanism (fused_forward.py,
    origin/forwardOne) gated in the mega backward's task-loop shape.
    flags[pid, 0] = mismatched elements, flags[pid, 1] = iterations done."""
    pid = tl.program_id(axis=0)
    om = tl.arange(0, 16)
    ub0 = bl.alloc(tl.bfloat16, [16, 16], al.ascend_address_space.UB)
    ub1 = bl.alloc(tl.bfloat16, [16, 16], al.ascend_address_space.UB)
    bad = 0  # python-int init + tensor accumulate (probe-3 idiom)
    for it in range(NITER):
        with al.scope(core_mode="cube", disable_auto_sync=True):
            base = (pid * NITER + it) * 256
            if it >= 2:
                # backpressure: buffer (it % 2) was last consumed by the
                # vector scope of iteration it-2 -- wait its release.  The
                # buffer SELECT must be a device branch with each arm
                # hardcoding its buffer (a buffer is not an SSA value one
                # can tl.select; the forwardOne form).
                if it % 2 == 0:
                    al.sync_block_wait("vector", "cube", 10)
                else:
                    al.sync_block_wait("vector", "cube", 11)
            acc = tl.zeros((16, 16), dtype=tl.float32)
            for ks in tl.range(0, 16, 8, num_stages=2):
                ok8 = ks + tl.arange(0, 8)
                x = tl.load(a_ptr + base + om[:, None] * 16 + ok8[None, :])
                yv = tl.load(b_ptr + ok8[:, None] * 16 + om[None, :])
                acc += tl.dot(x, yv)
            if it % 2 == 0:
                al.fixpipe(acc, ub0)
                al.sync_block_set("cube", "vector", 8)
            else:
                al.fixpipe(acc, ub1)
                al.sync_block_set("cube", "vector", 9)
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                # recompute from the loop index only (probe-4 idiom: no SSA
                # value needs to cross a scope boundary when it is cheap)
                base = (pid * NITER + it) * 256
                # forwardOne reads UB in 8-row chunks (fused_forward.py's
                # `for row_chunk in range(0, pair_block_m, 8)` loop), wait
                # BEFORE the chunk loop and release AFTER it -- match the
                # production form exactly.
                if it % 2 == 0:
                    al.sync_block_wait("cube", "vector", 8)
                else:
                    al.sync_block_wait("cube", "vector", 9)
                for row_chunk in range(0, 16, 8):
                    # both subviews built unconditionally; only the
                    # to_tensor sits in the device branch (the forwardOne
                    # form -- a view is cheap, a buffer is not an SSA value)
                    view0 = ub0.subview([row_chunk, 0], [8, 16], [1, 1])
                    view1 = ub1.subview([row_chunk, 0], [8, 16], [1, 1])
                    if it % 2 == 0:
                        got = bl.to_tensor(view0,
                                           writable=False).to(tl.float32)
                    else:
                        got = bl.to_tensor(view1,
                                           writable=False).to(tl.float32)
                    rows8 = row_chunk + tl.arange(0, 8)
                    gold = tl.load(
                        golden_ptr + base + rows8[:, None] * 16
                        + om[None, :])
                    diff = tl.abs(got - gold)
                    bad += tl.sum((diff > (ATOL + RTOL * tl.abs(gold)))
                                  .to(tl.int32))
                if it % 2 == 0:
                    al.sync_block_set("vector", "cube", 10)
                else:
                    al.sync_block_set("vector", "cube", 11)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            tl.store(flags_ptr + pid * 2, bad)
            tl.store(flags_ptr + pid * 2 + 1, NITER)


def run_mega_probe5_ub_pingpong(rank: int, world_size: int) -> None:
    """Probe 5 driver: the forwardOne UB ping-pong handoff (fixpipe +
    sync_block, zero GM round-trip and zero signal slots) inside the mega
    backward's alternating-scope task loop (the MOE_MEGA_UB_P4 gate).

    ARCH BOUNDARY (found 2026-09-09, 910B1 x8): the toolchain HARD-GATES
    al.fixpipe to Ascend910_95 — semantic.py raises "this feature is only
    supported on Ascend910_95" (triton/language/extra/cann/extension/
    core.py, Fixpipe docstring: "L0C to UB (for Ascend910_95 series)");
    copy_from_ub_to_l1 / buffer copy carry the same gate.  forwardOne's UB
    direct handoff is therefore A5-ONLY — on every other part this probe
    ASSERTS the gate fires (a tested boundary, not a failure) and the
    numeric validation below runs only on 910_95.  sync_block_set/wait and
    bl.alloc/subview/to_tensor themselves are NOT gated — the A3 subset of
    forwardOne's pipeline is the FUSE_P4 GM-handoff wave (probe 4).

    MOE_PROBE5_ITERS overrides the iteration count (default 16 -- enough for
    seven full ping-pong cycles of backpressure)."""
    _require_runtime("mega probe5")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    niter = int(os.environ.get("MOE_PROBE5_ITERS", "16"))
    label = f"mega-probe5-ub-pingpong-w{world_size}"
    arch = str(NPUUtils().get_arch())
    is_910_95 = "910_95" in arch

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        # no symmetric allocation: the whole handoff is on-chip (the session
        # wrapper only keeps the probe environment uniform with 1-4).
        torch.manual_seed(2416 + rank)
        a = torch.randn(
            nprog * niter, 16, 16, dtype=torch.bfloat16, device=device)
        b = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
        golden = a.float() @ b.float()   # (N,16,16)@(16,16) broadcasts
        flags = torch.full((nprog, 2), -1, dtype=torch.int32, device=device)
        dist.barrier()

        if not is_910_95:
            # boundary leg: the fixpipe semantic gate must fire, identically,
            # on every rank — anything else (a wedge, a DIFFERENT error, or a
            # surprise success) is a real failure.
            try:
                kernel_probe5_ub_pingpong[(nprog, 1, 1)](
                    a, b, golden, flags,
                    niter,
                    NPROG=nprog, LOCAL_RANK=rank,
                    RTOL=GEMM_RTOL, ATOL=GEMM_ATOL,
                    num_warps=8)
                torch.npu.synchronize()
            except Exception as e:  # noqa: BLE001 — verdict on the message
                if "only supported on Ascend910_95" in str(e):
                    print(f"[{label}] rank {rank}: fixpipe 910_95 gate "
                          f"fires as documented on {arch} — UB handoff is "
                          f"A5-only, boundary OK")
                    return
                raise
            raise RuntimeError(
                f"[{label}] rank {rank}: fixpipe COMPILED on {arch} — the "
                f"910_95 gate is gone; run the numeric leg on this part "
                f"(set is_910_95) and re-baseline this probe")

        kernel_probe5_ub_pingpong[(nprog, 1, 1)](
            a, b, golden, flags,
            niter,
            NPROG=nprog, LOCAL_RANK=rank,
            RTOL=GEMM_RTOL, ATOL=GEMM_ATOL,
            num_warps=8)
        torch.npu.synchronize()

        failures = []
        bad = int(flags[:, 0].sum().item())
        done = int(flags[:, 1].max().item())
        unverified = int((flags[:, 1] < 0).sum().item())
        if bad:
            worst = int(flags[:, 0].max().item())
            failures.append(
                f"UB ping-pong handoff failed: {bad} mismatched elements "
                f"(worst program {worst}) -- fixpipe did not publish the "
                f"cube tile to UB, or bl.to_tensor read a stale buffer "
                f"(sync_block ordering)")
        if unverified or done != niter:
            failures.append(
                f"{unverified} programs never stored a verdict or the "
                f"loop short-ran (max iterations seen {done}, want "
                f"{niter}) -- the sync_block backpressure chain wedged or "
                f"miscompiled")
        _fold_and_raise(failures, label, rank, device, ep_group)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe3_local_signal(dist_test):
    """Probe 3: local cube->vector per-tile signal, no barrier (B1/B3
    signalization gate).  A wedge (signal_op-to-self or dl.wait in a vector
    scope unsupported) is caught by DIST_TEST_TIMEOUT_S."""
    dist_test(run_mega_probe3_local_signal, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe4_fused_loop(dist_test):
    """Probe 4: alternating cube/vector scopes in a device-side loop +
    per-iteration local self-signal (MOE_MEGA_FUSE_P4 gate).  A wedge (scope
    machinery per iteration) is caught by DIST_TEST_TIMEOUT_S."""
    dist_test(run_mega_probe4_fused_loop, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe5_ub_pingpong(dist_test):
    """Probe 5: UB fixpipe ping-pong cube->vector handoff in a device-side
    loop — no GM round-trip, no signal slots (the MOE_MEGA_UB_P4 gate;
    forwardOne's mechanism).  On non-910_95 parts the fixpipe semantic gate
    is EXPECTED to fire at compile ("only supported on Ascend910_95",
    found 910B1 2026-09-09) — the boundary leg asserts exactly that and
    passes; the numeric leg runs only on A5.  A wedge (sync_block / bl
    machinery under per-iteration scope alternation) is caught by
    DIST_TEST_TIMEOUT_S."""
    dist_test(run_mega_probe5_ub_pingpong, world_size=8)


# ---------------------------------------------------------------------------
# Probe 6: remote writes to a symmetric slab at a NON-zero heap offset (the
# combine_buf gate of the 2026-09-10 wave evolution).  A dummy symmetric
# tensor occupies offset 0; buf sits behind it, exactly like a combine_buf
# would sit behind peer_mem.  The FIRST draft coupled both write forms in
# ONE kernel before a shared barrier — a faulting store wedged both legs
# and the probe wedged/failed/greened nondeterministically at w2 (2026-09-
# 10); the forms are therefore SEPARATE kernels so each verdict stands
# alone (a wedge in 6b is then the boundary itself, recorded by the
# DIST_TEST_TIMEOUT_S deadline):
#   6a putmem -> remote offset-N rows (the P1-dispatch write form; P1 only
#      ever putmems offset-0 peer_mem, P6b getmem-READS offset-N — the
#      write direction is unproven),
#   6b dl.symm_at(buf) + tl.store (the P4b return-push form; production
#      symm_at only ever targets the FIRST symmetric allocation).
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["dst_rank", "src_rank"])
def kernel_probe6a_putmem_offset(
        dummy_ptr, buf_ptr, pat_ptr, flags_ptr,
        dst_rank, src_rank,
        PATN: tl.constexpr, NPROG: tl.constexpr, PUT_ON: tl.constexpr):
    """putmem leg: rows [NPROG, 2*NPROG) of the peer's offset-N buf carry
    this rank's pattern; flags[pid, 0] = mismatches, [pid, 1] = nonzero
    elements that appeared in OUR offset-0 dummy (cross-slab corruption
    signature).  PUT_ON=0 is the CONTROL shape (MOE_PROBE6_CONTROL=1): the
    identical session/scopes/barrier with the putmem compiled out — if the
    control wedges too the probe's structure is at fault, not the write
    form (that bisect ran 2026-09-10 after the first 6a wedged)."""
    pid = tl.program_id(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            if PUT_ON:
                libshmem_device.putmem(
                    buf_ptr + (NPROG + pid) * PATN, pat_ptr + pid * PATN,
                    PATN * 4, dst_rank)
                libshmem_device.fence()
    libshmem_device.barrier_all()
    with al.scope(core_mode="vector", disable_auto_sync=True):
        # offs/want derived INSIDE the scope: no tensor SSA value may cross
        # the barrier (probe 2b's wedge lesson).
        offs = tl.arange(0, PATN)
        want = _probe_pattern_value(src_rank, pid, offs)
        got_put = tl.load(buf_ptr + (NPROG + pid) * PATN + offs)
        tl.store(flags_ptr + pid * 2,
                 tl.sum((got_put != want).to(tl.int32)))
        dummy_vals = tl.load(dummy_ptr + pid * PATN + offs)
        tl.store(flags_ptr + pid * 2 + 1,
                 tl.sum((dummy_vals != 0).to(tl.int32)))


@triton.jit(do_not_specialize=["dst_rank", "src_rank"])
def kernel_probe6b_symm_at_offset(
        dummy_ptr, buf_ptr, pat_ptr, flags_ptr,
        dst_rank, src_rank,
        PATN: tl.constexpr, NPROG: tl.constexpr):
    """symm_at leg: rows [0, NPROG) of the peer's offset-N buf carry pid+1
    through dl.symm_at + tl.store (the P4b form); flags[pid, 0] = mismatches,
    [pid, 1] = nonzero elements in OUR offset-0 dummy (the misresolution
    signature — symm_at silently resolving the heap base instead of the
    slab would land the peer's stores in our dummy)."""
    pid = tl.program_id(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            offs = tl.arange(0, PATN)
            remote = dl.symm_at(buf_ptr, dst_rank)
            tl.store(remote + pid * PATN + offs, pid + 1)
            libshmem_device.fence()
    libshmem_device.barrier_all()
    with al.scope(core_mode="vector", disable_auto_sync=True):
        offs = tl.arange(0, PATN)
        got_symm = tl.load(buf_ptr + pid * PATN + offs)
        tl.store(flags_ptr + pid * 2,
                 tl.sum((got_symm != pid + 1).to(tl.int32)))
        dummy_vals = tl.load(dummy_ptr + pid * PATN + offs)
        tl.store(flags_ptr + pid * 2 + 1,
                 tl.sum((dummy_vals != 0).to(tl.int32)))


def _run_probe6(rank: int, world_size: int, *, symm: bool, label: str) -> None:
    """Shared 6a/6b driver: identical geometry, only the write form differs."""
    _require_runtime("mega probe6")
    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    nprog = ncore()
    dst_rank = (rank + 1) % world_size
    src_rank = (rank - 1 + world_size) % world_size

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        # allocation ORDER is the point: dummy owns heap offset 0, buf sits
        # at a nonzero offset exactly like a combine_buf behind peer_mem
        dummy = kit.ash.aclshmem_create_tensor(
            [nprog * PROBE_PATN], dtype=torch.int32, device_id=rank)
        buf = kit.ash.aclshmem_create_tensor(
            [2 * nprog * PROBE_PATN], dtype=torch.int32, device_id=rank)
        try:
            dummy.zero_()
            buf.zero_()
            # host mirror of _probe_pattern_value (rank, pid, e)
            pat = torch.tensor(
                [[rank * 1000003 + p * 131 + e * 7 + 12345
                  for e in range(PROBE_PATN)]
                 for p in range(nprog)],
                dtype=torch.int32, device=device)
            flags = torch.full((nprog, 2), -1, dtype=torch.int32, device=device)
            dist.barrier()

            if symm:
                kernel_probe6b_symm_at_offset[(nprog, 1, 1)](
                    dummy, buf, pat, flags, dst_rank, src_rank,
                    PATN=PROBE_PATN, NPROG=nprog, num_warps=8)
            else:
                # control mode: PUT_ON=0 keeps the session/allocation/
                # scope/barrier shape and compiles out only the putmem
                kernel_probe6a_putmem_offset[(nprog, 1, 1)](
                    dummy, buf, pat, flags, dst_rank, src_rank,
                    PATN=PROBE_PATN, NPROG=nprog,
                    PUT_ON=(os.environ.get("MOE_PROBE6_CONTROL") != "1"),
                    num_warps=8)
            torch.npu.synchronize()

            bad = int(flags[:, 0].sum().item())
            bad_dummy = int(flags[:, 1].sum().item())
            unverified = int((flags[:, 0] < 0).sum().item())
            control = (not symm
                       and os.environ.get("MOE_PROBE6_CONTROL") == "1")
            failures = []
            if unverified:
                failures.append(
                    f"{unverified} programs never stored a verdict "
                    f"(kernel wedged or short-ran)")
            form = "dl.symm_at+store" if symm else "putmem"
            if control:
                # PUT_ON=0: the only verdict is completion — the numeric
                # mismatch (buf stays zero) is expected
                print(f"[{label}] rank {rank}: CONTROL completed (putmem "
                      f"compiled out; dummy corruption {bad_dummy} would "
                      f"still be a finding)", flush=True)
            elif bad or bad_dummy:
                print(f"[{label}] rank {rank}: BOUNDARY — the {form} form "
                      f"does not cleanly write the offset-N slab "
                      f"({bad} mismatched, offset-0 dummy corruption "
                      f"{bad_dummy})", flush=True)
                if not symm:
                    # the putmem leg is the FALLBACK form — if it is broken
                    # too the environment cannot build combine_buf at all
                    failures.append(
                        f"putmem control leg failed ({bad} mismatched, "
                        f"{bad_dummy} dummy corruptions) — both remote-write "
                        f"forms are broken at offset N")
            else:
                print(f"[{label}] rank {rank}: the {form} form cleanly "
                      f"writes the offset-N slab", flush=True)
            _fold_and_raise(failures, label, rank, device, ep_group)
        finally:
            kit.ash.aclshmem_free_tensor(buf)
            kit.ash.aclshmem_free_tensor(dummy)


def run_mega_probe6a_putmem_offset(rank: int, world_size: int) -> None:
    _run_probe6(rank, world_size, symm=False,
                label=f"mega-probe6a-putmem-offset-w{world_size}")


def run_mega_probe6b_symm_at_offset(rank: int, world_size: int) -> None:
    _run_probe6(rank, world_size, symm=True,
                label=f"mega-probe6b-symm-at-offset-w{world_size}")


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe6a_putmem_offset(dist_test):
    """Probe 6a: putmem into a remote offset-N symmetric slab (the P1 write
    form at a heap offset it has never run against).  Wedges are caught by
    DIST_TEST_TIMEOUT_S."""
    dist_test(run_mega_probe6a_putmem_offset, world_size=2)


@pytest.mark.dist
@pytest.mark.functional
def test_mega_probe6b_symm_at_offset(dist_test):
    """Probe 6b: dl.symm_at + tl.store against a remote offset-N symmetric
    slab (the P4b return-push form; production symm_at only targets the
    FIRST symmetric allocation).  A wedge here IS the boundary record —
    combine_buf's push must then take 6a's putmem form."""
    dist_test(run_mega_probe6b_symm_at_offset, world_size=2)
