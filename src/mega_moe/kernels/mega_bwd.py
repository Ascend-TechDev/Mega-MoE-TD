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
#  MOE_MEGA_WGRAD_NS wgrad tile knobs. Only the non-MoonEP layout is
#  supported.
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

from .common import ncore
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


# ============================================================================
# jit phase helpers (scopes live in the mega kernel, the dispatch_fc2_bwd
# pattern — helpers are scope-free bodies the caller wraps)
# ============================================================================
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
    offs = tl.arange(0, BLOCK_SIZE)
    for row in range(pid, n_rows, nprogs):
        r64 = row.to(tl.int64)
        a_ptr = AB_ptr + r64 * AB_stride          # gate half
        b_ptr = a_ptr + ffn                        # up half
        dc_ptr = dC_ptr + r64 * dC_stride
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
def _mega_wgrad_sweep(
    pid, ncores,
    grad_out_ptr, stride_outm, stride_outn,   # [M, N] expert-major (NO host transpose)
    orig_in_ptr, stride_om, stride_ok,        # [M, K]
    grad_w_ptr, stride_we, stride_wn, stride_wk,   # [E, N, K] out
    split_size_cum_per_expert_ptr, expert_counts_ptr,
    N, K, num_tiles_n, num_tiles_k, task_begin, task_end,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
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

    Must run inside a cube scope; no reductions here (cube-scope dot
    reductions trip the bishengir CUBE_OR_VECTOR assertion)."""
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
        n_start = tn * BLOCK_N
        k_start = tk * BLOCK_K
        nmask = (n_start + offs_n) < N
        kmask = (k_start + offs_k) < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m in tl.range(0, split_size, BLOCK_M, num_stages=NUM_STAGES):
            mm = m + offs_m
            mmask = mm < split_size
            row64 = (split_begin + mm).to(tl.int64)
            a_off = row64[None, :] * stride_outm + (n_start + offs_n[:, None]) * stride_outn
            a = tl.load(grad_out_ptr + a_off, mask=nmask[:, None] & mmask[None, :], other=0.0)
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
    weight_ptr, stride_we, stride_wk, stride_wn,   # fc1_combined [E, 2*ffn, H] (K, N)
    hidden_buf_ptr,                         # [M, H] out (row stride = N)
    tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
    N, K, num_tiles_n, num_tiles_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """fc1 input-grad GEMM ``hidden_buf[m, :] = grad_fc1_output[m, :] @
    fc1_combined[e]`` over ALL M-tiles (FIRST=0/LAST=num_tiles_m as RUNTIME
    bounds — the constexpr-bound production variant would recompile whenever
    routing moves a tile). Persistent STRIDED task partition (same imbalance
    fix as _mega_wgrad_sweep), from _kernel_combine_fc1_bwd_gemm_group
    (combine_fc1_bwd.py:114-171); WEIGHT_EXPERT_BASE=0 (non-MoonEP single
    home table). Must run inside a cube scope."""
    om = tl.arange(0, BLOCK_M)
    on_ = tl.arange(0, BLOCK_N)
    ok = tl.arange(0, BLOCK_K)
    group_tiles = num_tiles_m
    total_tasks = group_tiles * num_tiles_n
    for task_id in range(pid, total_tasks, ncores):
        tile_m = task_id % group_tiles
        tile_n = task_id // group_tiles
        expert_id = tl.load(tile_expert_ptr + tile_m)
        row_start = tl.load(tile_row0_ptr + tile_m)
        rem = tl.load(tile_rows_ptr + tile_m)
        n_start = tile_n * BLOCK_N
        mm = om < rem
        mn = on_ < (N - n_start)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        wb = expert_id.to(tl.int64) * stride_we
        row_base = row_start.to(tl.int64) + om.to(tl.int64)
        for ks in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            mk = ok < (K - ks)
            ao = row_base[:, None] * stride_im + (ks + ok[None, :]) * stride_ik
            a = tl.load(inp_ptr + ao, mask=mm[:, None] & mk[None, :], other=0.0)
            bo = (ks + ok[:, None]) * stride_wk + (n_start + on_[None, :]) * stride_wn
            b = tl.load(weight_ptr + wb + bo, mask=mk[:, None] & mn[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = row_base[:, None] * N + (n_start + on_[None, :])
        tl.store(hidden_buf_ptr + co, acc.to(hidden_buf_ptr.dtype.element_ty),
                 mask=mm[:, None] & mn[None, :])


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
# the mega kernel
# ============================================================================
@triton.jit(do_not_specialize=["signal_epoch"])
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
    N3, K3, num_tn3, num_tk3, w3_total,
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
            _fc2_bwd_gemm_merged_tiles_wait(
                pid, num_cores,
                peer_mem_ptr, signal_mem_ptr, fc2_ptr, grad_swiglu_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
                signal_epoch,
                N1, K1,
                H, 1, stride_we1, stride_wk1, stride_wn1, N1, 1,
                D_BM, D_BN, D_BK, PUSH_BLOCK_M,
                WORLD_SIZE, EXPERTS_PER_RANK,
                0, EXPERTS_PER_RANK, 0,
                MAX_BWD_TILES, tl.bfloat16)
    # B1: publish every rank's P1 remote puts; grad_swiglu GM-visible.
    libshmem_device.barrier_all()

    # ---------------- P2 (vector) ∥ P3 (cube) ----------------
    if P23_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
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
                N3, K3, num_tn3, num_tk3, 0, w3_total,
                W_BM, W_BN, W_BK, W_NS)
    # B2: all ranks finish READING peer_mem (P3) before ANY rank's P4b may
    # overwrite it; P2 outputs (dAB/dscale) published for P4a/P4b.
    libshmem_device.barrier_all()

    # ---------------- P4a: fc1 input-grad GEMM ----------------
    if P4_ON:
        with al.scope(core_mode="cube", disable_auto_sync=True):
            _mega_combine_gemm(
                pid, num_cores,
                dAB_ptr, stride_im4, stride_ik4,
                fc1_combined_ptr, stride_we4, stride_wk4, stride_wn4,
                hidden_buf_ptr,
                tile_expert_ptr, tile_row0_ptr, tile_rows_ptr,
                N4, K4, num_tiles_n4, num_tiles_m4,
                C_BM, C_BN, C_BK, C_NS)
    # B3: local cube->vector handoff of hidden_buf.
    libshmem_device.barrier_all()

    # ------- P4b (vec push) ∥ P5a (cube wgrad, first half of tasks) -------
    # The P2∥P3 concurrency recipe: adjacent vector/cube scopes overlap on
    # their engines. P5 needs only P2's dAB (published at B2), so its first
    # task half rides the push window — msprof showed the vector engine
    # ~97% idle across the kernel, so this window was pure loss before.
    if P4_ON:
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                _mega_push_rows(
                    pid, num_cores,
                    hidden_buf_ptr,
                    write_rank_by_src_ptr, write_off_by_src_ptr,
                    peer_mem_ptr, dscale_ptr,
                    H, M4,
                    BLOCK_N_PUSH, GATE_PAD_C)
    if P5_ON:
        with al.scope(core_mode="cube", disable_auto_sync=True):
            _mega_wgrad_sweep(
                pid, num_cores,
                dAB_ptr, K4, 1,
                orig_in5_ptr, stride_om5, stride_ok5,
                grad_fc1_ptr, stride_we5, stride_wn5, stride_wk5,
                split_cum_ptr, expert_counts_ptr,
                N5, K5, num_tn5, num_tk5, 0, w5_split,
                W_BM, W_BN, W_BK, W_NS)
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
                N5, K5, num_tn5, num_tk5, w5_split, w5_total,
                W_BM, W_BN, W_BK, W_NS)


# ============================================================================
# wrapper
# ============================================================================
def mega_backward_triton(saved, dy, peer_mem):
    """MOE_BWD_MEGA=1 backward: the whole non-MoonEP 5-step backward in ONE
    kernel launch. Returns the SAME 10-key dict as the orchestrator's
    non-MoonEP branch (grad_fc2_out_sorted is the same zero-copy peer_mem
    alias step 1 returns). peer_mem must be the session's FIRST symmetric
    allocation (heap offset 0 — dl.symm_at in P4b)."""
    if saved.get("use_moonep"):
        raise ValueError("MOE_BWD_MEGA=1 supports only non-MoonEP saved dicts")
    device = dy.device
    rank = saved["ep_rank"]
    W = saved["world_size"]

    # P1 prep: expert-major gco (host gather) + cached dispatch maps.
    p1 = _prepare_dispatch_fc2_bwd(saved, dy)
    EPR = p1["E"]; H = p1["H"]; N1 = p1["N"]; K1 = p1["K"]; M = p1["M"]
    ffn = N1
    signal_mem = _ensure_bwd_signal_mem(saved, W, EPR, p1["max_bwd_tiles"])
    # SET-mode epoch: producer writes signal_epoch, consumer waits the same
    # value; bump after launch so the next call sees a fresh one (SET
    # overwrites — no slot zeroing). Same keys MegaMoEFunction.backward
    # injects/persists via `state`.
    signal_epoch = saved.get("_bwd_tile_signal_epoch", 1)
    saved["_bwd_tile_signal_epoch"] = signal_epoch + 1

    # outputs (fresh, contiguous — strides passed to the kernel)
    fc1_output = saved["fc1_output"].contiguous()      # [M, 2*ffn]
    grad_swiglu = torch.empty(M, ffn, dtype=dy.dtype, device=device)
    grad_fc1_output = torch.empty_like(fc1_output)     # [M, 2*ffn]
    grad_gate = torch.empty(M, dtype=fc1_output.dtype, device=device)
    orig_in3 = saved["swiglu_out_weighted"].contiguous()   # [M, ffn]
    grad_fc2 = torch.empty(EPR, H, ffn, dtype=dy.dtype, device=device)
    orig_in5 = saved["recv_hidden_sorted"].contiguous()    # [M, H]
    grad_fc1 = torch.empty(EPR, 2 * ffn, H, dtype=dy.dtype, device=device)
    hidden_buf = torch.empty(M, H, dtype=dy.dtype, device=device)

    # P4 prep: cached combine maps + per-GEMM-tile expert/row tables.
    p4 = _combine_static_maps(saved)
    cbm, cbn, cbk, cns = _combine_gemm_tile()
    cbm = int(os.environ.get("MOE_COMBINE_GEMM_BM", "256"))   # see dbm note above
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
    dbm = int(os.environ.get("MOE_DISPATCH_GEMM_BM", "256"))
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
        H, ffn, num_tn3, num_tk3, w3_total,
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
        num_warps=8, **launch_options)

    # expert-major peer_mem IS the sorted layout -> identity view (no gather),
    # exactly the alias step 1 returns.
    total_recv = p1["total_recv"]
    grad_fc2_out_sorted = peer_mem.view(-1)[:total_recv * H].view(
        total_recv, H).contiguous()
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing.view(
            p4["B"], p4["topk"]),
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
        grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
        grad_fc1=grad_fc1,
    )
