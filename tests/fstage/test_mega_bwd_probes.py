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
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from mega_moe.kernels.common import ncore
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
