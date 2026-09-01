# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Ascend (CANN/NPU) fused EP MoE dispatch + FC1 kernel — default schedule only.

Architecture:

  Phase 1 — In-kernel dispatch:
    Each AI core iterates local tokens and writes them through the remote
    symmetric MTE mapping (peer_mem).
    This replaces host-side ``all_to_all_single``.

  The all-core pipeline runs Vector dispatch and Cube FC1 concurrently on
    every AI core.  A dispatch task notifies a source-local tile after its
    remote stores are visible; FC1 waits only for the tile it is about to use.

The Vector stream walks nonempty source/expert buckets (expert-major order)
and publishes per-tile SET readiness slots; the Cube stream consumes merged
expert M windows over the N-tile pool as their overlapping source tiles
become ready.  ``FINAL_BARRIER`` remains the only per-call option.

Reference: Ascend ``01-ascend-allgather-gemm.py`` tutorial.
"""

import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .shmem_transport import mte_put
from .shmem_udma import (
    udma_bfloat16_put_nbi,
    udma_bfloat16_put_signal_nbi,
    udma_quiet,
)


@triton.jit
def _submit_one_replica_panel_put_signal(
    destination_ptr,
    source_ptr,
    panel_elements,
    ready_address,
    signal_epoch,
    destination_rank,
    CHUNK_ELEMENTS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Push one panel and let its final WQE publish remote readiness."""
    for chunk_id in tl.static_range(0, NUM_CHUNKS - 1):
        chunk_start = chunk_id * CHUNK_ELEMENTS
        chunk_elements = tl.minimum(
            CHUNK_ELEMENTS,
            panel_elements - chunk_start,
        )
        udma_bfloat16_put_nbi(
            destination_ptr + chunk_start,
            source_ptr + chunk_start,
            chunk_elements,
            destination_rank,
        )
    if NUM_CHUNKS > 1:
        # UDMA WQEs are NO-ordering even on one QP.  Complete every prefix
        # chunk before using the final chunk's notify as panel readiness.
        udma_quiet(destination_rank)
    final_chunk = NUM_CHUNKS - 1
    final_start = final_chunk * CHUNK_ELEMENTS
    final_elements = tl.minimum(
        CHUNK_ELEMENTS,
        panel_elements - final_start,
    )
    udma_bfloat16_put_signal_nbi(
        destination_ptr + final_start,
        source_ptr + final_start,
        final_elements,
        ready_address,
        signal_epoch,
        destination_rank,
    )


@triton.jit
def _push_replica_weight_panels_for_destination(
    destination_rank,
    source_gate_up_ptr,
    source_down_ptr,
    replica_gate_up_ptr,
    replica_down_ptr,
    gate_up_ready_ptr,
    down_ready_ptr,
    consumed_epoch_ptr,
    experts_to_copy_ptr,
    signal_epoch,
    previous_epoch,
    replica_slot_count,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    GATE_PANEL_ELEMENTS: tl.constexpr,
    GATE_CHUNK_ELEMENTS: tl.constexpr,
    GATE_NUM_CHUNKS: tl.constexpr,
    DOWN_PANEL_ELEMENTS: tl.constexpr,
    DOWN_CHUNK_ELEMENTS: tl.constexpr,
    DOWN_NUM_CHUNKS: tl.constexpr,
):
    """Push model-layout gate/up once, then the two contiguous down panels."""
    for panel_kind in tl.static_range(0, 3):
        replica_slot = 0
        while replica_slot < replica_slot_count:
            descriptor = destination_rank * EXPERTS_PER_RANK + replica_slot
            global_expert = tl.load(experts_to_copy_ptr + descriptor)
            owned = (
                (global_expert >= 0)
                & (global_expert // EXPERTS_PER_RANK == LOCAL_RANK)
            )
            if owned:
                if (panel_kind == 0) & (previous_epoch > 0):
                    remote_consumed = libshmem_device.remote_ptr(
                        consumed_epoch_ptr + replica_slot * 16,
                        destination_rank,
                    )
                    libshmem_device.signal_wait_until(
                        remote_consumed,
                        libshmem_device.ACLSHMEM_CMP_EQ,
                        previous_epoch,
                    )
                owner_local_expert = global_expert % EXPERTS_PER_RANK
                if panel_kind == 0:
                    gate_up_elements: tl.constexpr = 2 * GATE_PANEL_ELEMENTS
                    source_panel = (
                        source_gate_up_ptr
                        + owner_local_expert * gate_up_elements
                    )
                    destination_panel = (
                        replica_gate_up_ptr
                        + replica_slot * gate_up_elements
                    )
                    _submit_one_replica_panel_put_signal(
                        destination_panel,
                        source_panel,
                        gate_up_elements,
                        gate_up_ready_ptr + replica_slot * 16,
                        signal_epoch,
                        destination_rank,
                        GATE_CHUNK_ELEMENTS,
                        GATE_NUM_CHUNKS,
                    )
                else:
                    panel = panel_kind - 1
                    source_panel = (
                        source_down_ptr
                        + owner_local_expert * (2 * DOWN_PANEL_ELEMENTS)
                        + panel * DOWN_PANEL_ELEMENTS
                    )
                    destination_panel = (
                        replica_down_ptr
                        + replica_slot * (2 * DOWN_PANEL_ELEMENTS)
                        + panel * DOWN_PANEL_ELEMENTS
                    )
                    _submit_one_replica_panel_put_signal(
                        destination_panel,
                        source_panel,
                        DOWN_PANEL_ELEMENTS,
                        down_ready_ptr + (replica_slot * 2 + panel) * 16,
                        signal_epoch,
                        destination_rank,
                        DOWN_CHUNK_ELEMENTS,
                        DOWN_NUM_CHUNKS,
                    )
            replica_slot += 1


@triton.jit(
    do_not_specialize=[
        "signal_epoch",
        "replica_weight_epoch",
        "previous_replica_epoch",
        "replica_slot_count",
    ]
)
def _kernel_dispatch_fc1(
    # ---- Input / output ----
    input_ptr,
    peer_mem_ptr,
    routing_weight_ptr,
    routing_weight_recv_ptr,
    signal_mem_ptr,
    weight_ptr,
    replica_weight_ptr,
    replica_weight_ready_ptr,
    replica_down_weight_ptr,
    replica_down_ready_ptr,
    source_gate_up_ptr,
    source_down_ptr,
    consumed_epoch_ptr,
    experts_to_copy_ptr,
    replica_slot_count,
    output_ptr,

    # ---- Dispatch token metadata ----
    send_src_idx_ptr,
    send_route_idx_ptr,
    send_bucket_dst_starts_ptr,

    # ---- Compact dispatch / receive metadata ----
    send_bucket_starts_ptr,
    send_counts_re_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    recv_counts_re_ptr,

    signal_epoch,
    replica_weight_epoch,
    previous_replica_epoch,

    # ---- Dimensions ----
    hidden: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,

    # ---- Strides ----
    stride_input_m,
    stride_input_k,
    stride_weight_0,
    stride_weight_1,
    stride_weight_2,
    stride_replica_weight_0,
    stride_replica_weight_k,
    stride_replica_weight_n,
    stride_output_m,
    stride_output_n,

    # ---- Meta-parameters ----
    NUM_PROGRAM_CORES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HOME_EXPERTS_PER_RANK: tl.constexpr,
    USE_REPLICA_WEIGHTS: tl.constexpr,
    PUSH_REPLICA_WEIGHTS: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    ACTIVE_EXPERTS_PER_RANK: tl.constexpr,
    GATE_PANEL_ELEMENTS: tl.constexpr,
    GATE_CHUNK_ELEMENTS: tl.constexpr,
    GATE_NUM_CHUNKS: tl.constexpr,
    DOWN_PANEL_ELEMENTS: tl.constexpr,
    DOWN_CHUNK_ELEMENTS: tl.constexpr,
    DOWN_NUM_CHUNKS: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    FINAL_BARRIER: tl.constexpr,
    DISPATCH_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Overlap remote dispatch and FC1 on the all-core pipeline.

    The Vector side dispatches source-local tiles (expert-major task order)
    while the Cube side consumes independently-ready receive tiles on the same
    logical PID.  Each source-local M tile is published with a SET(epoch) slot;
    FC1 acquires every dispatch tile whose rows overlap the merged expert
    window it is about to compute.
    """
    pid = tl.program_id(axis=0)
    dtype = tl.bfloat16

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _dispatch_count_derived_source_tiles(
                pid, NUM_PROGRAM_CORES,
                input_ptr, peer_mem_ptr,
                routing_weight_ptr, routing_weight_recv_ptr,
                signal_mem_ptr,
                send_src_idx_ptr, send_route_idx_ptr,
                send_bucket_dst_starts_ptr,
                send_bucket_starts_ptr, send_counts_re_ptr,
                signal_epoch, hidden, stride_input_m,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK,
                ACTIVE_EXPERTS_PER_RANK,
                MAX_SOURCE_TILES, DISPATCH_BLOCK_SIZE_M)
        elif pid < WORLD_SIZE:
            if PUSH_REPLICA_WEIGHTS:
                if pid != LOCAL_RANK:
                    _push_replica_weight_panels_for_destination(
                        pid,
                        source_gate_up_ptr,
                        source_down_ptr,
                        replica_weight_ptr,
                        replica_down_weight_ptr,
                        replica_weight_ready_ptr,
                        replica_down_ready_ptr,
                        consumed_epoch_ptr,
                        experts_to_copy_ptr,
                        replica_weight_epoch,
                        previous_replica_epoch,
                        replica_slot_count,
                        LOCAL_RANK,
                        HOME_EXPERTS_PER_RANK,
                        GATE_PANEL_ELEMENTS,
                        GATE_CHUNK_ELEMENTS,
                        GATE_NUM_CHUNKS,
                        DOWN_PANEL_ELEMENTS,
                        DOWN_CHUNK_ELEMENTS,
                        DOWN_NUM_CHUNKS,
                    )
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _triton_grouped_gemm_expert_n_merged_tiles_wait(
            pid, NUM_PROGRAM_CORES,
            peer_mem_ptr, signal_mem_ptr, weight_ptr,
            replica_weight_ready_ptr, output_ptr,
            recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
            signal_epoch, replica_weight_epoch,
            N, K, stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            DISPATCH_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K,
            WORLD_SIZE, EXPERTS_PER_RANK,
            0, HOME_EXPERTS_PER_RANK, 0,
            MAX_SOURCE_TILES, False, dtype)
        if USE_REPLICA_WEIGHTS:
            _triton_grouped_gemm_expert_n_merged_tiles_wait(
                pid, NUM_PROGRAM_CORES,
                peer_mem_ptr, signal_mem_ptr, replica_weight_ptr,
                replica_weight_ready_ptr, output_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
                signal_epoch, replica_weight_epoch,
                N, K, stride_input_m, stride_input_k,
                stride_replica_weight_0, stride_replica_weight_n,
                stride_replica_weight_k,
                stride_output_m, stride_output_n,
                DISPATCH_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_M,
                BLOCK_SIZE_N, BLOCK_SIZE_K,
                WORLD_SIZE, EXPERTS_PER_RANK,
                HOME_EXPERTS_PER_RANK, ACTIVE_EXPERTS_PER_RANK,
                HOME_EXPERTS_PER_RANK,
                MAX_SOURCE_TILES, True, dtype)

    # Standalone dispatch must finish globally before its single receive
    # buffer can be reused.  A complete forward may defer this collective to
    # FC2/combine: all local input dependencies are already covered by expert
    # readiness, and the later combine kernel has its own rank-wide barrier.
    if FINAL_BARRIER:
        libshmem_device.barrier_all()


@triton.jit
def _find_nth_nonempty_task(
    counts_ptr,
    active_task_id,
    NUM_TASKS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Map a dense active-task ordinal to its sparse count-matrix index.

    Tasks are visited in expert-major order (expert, then destination rank) so
    every destination receives early expert buckets concurrently instead of
    every source rank sending to rank 0 first.
    """
    selected_task = 0
    num_seen = 0
    for schedule_task in range(0, NUM_TASKS):
        expert_id = schedule_task // WORLD_SIZE
        dst_rank = schedule_task % WORLD_SIZE
        candidate_task = dst_rank * EXPERTS_PER_RANK + expert_id
        task_count = tl.load(counts_ptr + candidate_task)
        is_nonempty = task_count > 0
        is_selected = is_nonempty & (num_seen == active_task_id)
        selected_task = tl.where(is_selected, candidate_task, selected_task)
        num_seen += tl.where(is_nonempty, 1, 0)
    return selected_task


@triton.jit
def _dispatch_one_source_tile_task(
    task_id, task_lane, task_cores,
    input_ptr, peer_mem_ptr,
    routing_weight_ptr, routing_weight_recv_ptr,
    signal_mem_ptr,
    send_src_idx_ptr, send_route_idx_ptr, send_bucket_dst_starts_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr,
    signal_epoch,
    hidden: tl.constexpr,
    stride_input_m,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    DISPATCH_BLOCK_SIZE_M: tl.constexpr,
):
    """Dispatch the source-local tiles assigned to one lane of a nonempty bucket."""
    task_start = tl.load(send_bucket_starts_ptr + task_id)
    task_count = tl.load(send_counts_re_ptr + task_id)
    task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
    dst_rank = task_id // EXPERTS_PER_RANK
    expert_id = task_id % EXPERTS_PER_RANK
    num_source_tiles = tl.cdiv(task_count, DISPATCH_BLOCK_SIZE_M)

    for source_tile in range(task_lane, num_source_tiles, task_cores):
        tile_start = source_tile * DISPATCH_BLOCK_SIZE_M
        tile_count = tl.minimum(
            DISPATCH_BLOCK_SIZE_M,
            task_count - tile_start,
        )
        for tile_token in range(tile_count):
            send_idx = task_start + tile_start + tile_token
            src_idx = tl.load(send_src_idx_ptr + send_idx)
            dst_offs = (
                task_dst_start + tile_start + tile_token
            ).to(tl.int64)
            src_idx64 = src_idx.to(tl.int64)
            src_base = input_ptr + src_idx64 * stride_input_m
            dst_base = peer_mem_ptr + dst_offs * stride_input_m
            mte_put(
                dst_base,
                src_base,
                hidden,
                dst_rank,
                BLOCK_ELEMENTS=4096,
            )
            route_idx = tl.load(send_route_idx_ptr + send_idx)
            mte_put(
                routing_weight_recv_ptr + dst_offs,
                routing_weight_ptr + route_idx,
                1,
                dst_rank,
                BLOCK_ELEMENTS=1,
            )

        libshmem_device.fence()
        signal_slot = (
            (LOCAL_RANK * EXPERTS_PER_RANK + expert_id) * MAX_SOURCE_TILES
            + source_tile
        )
        libshmem_device.signal_op(
            signal_mem_ptr + signal_slot * 16,
            signal_epoch,
            libshmem_device.ACLSHMEM_SIGNAL_SET,
            dst_rank,
        )


@triton.jit
def _dispatch_count_derived_source_tiles(
    pid, num_cores: tl.constexpr,
    input_ptr, peer_mem_ptr,
    routing_weight_ptr, routing_weight_recv_ptr,
    signal_mem_ptr,
    send_src_idx_ptr, send_route_idx_ptr, send_bucket_dst_starts_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr,
    signal_epoch,
    hidden: tl.constexpr,
    stride_input_m,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    ACTIVE_EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Assign cores only to nonempty buckets, then stripe each bucket's tiles."""
    num_tasks: tl.constexpr = WORLD_SIZE * ACTIVE_EXPERTS_PER_RANK
    num_active_tasks = 0
    for schedule_task in range(0, num_tasks):
        expert_id = schedule_task // WORLD_SIZE
        dst_rank = schedule_task % WORLD_SIZE
        task_id = dst_rank * EXPERTS_PER_RANK + expert_id
        task_count = tl.load(send_counts_re_ptr + task_id)
        num_active_tasks += tl.where(task_count > 0, 1, 0)

    if num_active_tasks > 0:
        if num_cores >= num_active_tasks:
            active_task_id = pid % num_active_tasks
            task_lane = pid // num_active_tasks
            task_cores = (
                num_cores + num_active_tasks - 1 - active_task_id
            ) // num_active_tasks
            task_id = _find_nth_nonempty_task(
                send_counts_re_ptr, active_task_id, num_tasks,
                WORLD_SIZE, EXPERTS_PER_RANK)
            _dispatch_one_source_tile_task(
                task_id, task_lane, task_cores,
                input_ptr, peer_mem_ptr,
                routing_weight_ptr, routing_weight_recv_ptr,
                signal_mem_ptr,
                send_src_idx_ptr, send_route_idx_ptr,
                send_bucket_dst_starts_ptr,
                send_bucket_starts_ptr, send_counts_re_ptr,
                signal_epoch, hidden, stride_input_m,
                LOCAL_RANK, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
                BLOCK_SIZE_M)
        else:
            for active_task_id in range(pid, num_active_tasks, num_cores):
                task_id = _find_nth_nonempty_task(
                    send_counts_re_ptr, active_task_id, num_tasks,
                    WORLD_SIZE, EXPERTS_PER_RANK)
                _dispatch_one_source_tile_task(
                    task_id, 0, 1,
                    input_ptr, peer_mem_ptr,
                    routing_weight_ptr, routing_weight_recv_ptr,
                    signal_mem_ptr,
                    send_src_idx_ptr, send_route_idx_ptr,
                    send_bucket_dst_starts_ptr,
                    send_bucket_starts_ptr, send_counts_re_ptr,
                    signal_epoch, hidden, stride_input_m,
                    LOCAL_RANK, EXPERTS_PER_RANK,
                    MAX_SOURCE_TILES, BLOCK_SIZE_M)


@triton.jit
def _triton_grouped_gemm_expert_n_merged_tiles_wait(
    pid, ncore: tl.constexpr,
    input_ptr, signal_mem_ptr, weight_ptr, replica_weight_ready_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    signal_epoch, replica_weight_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    DISPATCH_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    FIRST_EXPERT: tl.constexpr,
    LAST_EXPERT: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    WAIT_REPLICA_WEIGHTS: tl.constexpr,
    dtype: tl.constexpr,
):
    """Consume merged expert M windows as their source tiles become ready.

    GEMM ownership is keyed only by ``(local_expert, n_tile)``.  Source rank
    participates solely in the readiness dependency calculation, so source
    fragments no longer force independent full-weight scans.  Expert-major
    task order lets all Cube cores cover one expert's N tiles together.
    """
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    first_task = FIRST_EXPERT * num_n_tiles
    last_task = LAST_EXPERT * num_n_tiles

    for task_id in range(pid + first_task, last_task, ncore):
        expert_id = task_id // num_n_tiles
        n_tile = task_id % num_n_tiles
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)

        if expert_size > 0:
            if WAIT_REPLICA_WEIGHTS:
                replica_slot = expert_id - WEIGHT_EXPERT_BASE
                weight_ready_token = dl.wait(
                    replica_weight_ready_ptr
                    + replica_slot * 16,
                    1,
                    "gpu",
                    "acquire",
                    waitValue=replica_weight_epoch,
                )
                ready_weight_ptr = dl.consume_token(
                    weight_ptr, weight_ready_token
                )
            else:
                ready_weight_ptr = weight_ptr
            num_m_windows = tl.cdiv(expert_size, GEMM_BLOCK_SIZE_M)
            for m_window in range(0, num_m_windows):
                window_start = m_window * GEMM_BLOCK_SIZE_M
                window_size = tl.minimum(
                    GEMM_BLOCK_SIZE_M,
                    expert_size - window_start,
                )
                window_end = window_start + window_size
                source_start = 0
                ready_token = 0

                # peer_mem is expert-major then source-major.  Acquire every
                # dispatch tile whose rows overlap this merged expert window.
                for source_id in range(0, WORLD_SIZE):
                    source_size = tl.load(
                        recv_counts_re_ptr
                        + source_id * EXPERTS_PER_RANK
                        + expert_id
                    )
                    source_end = source_start + source_size
                    overlap_start = tl.maximum(window_start, source_start)
                    overlap_end = tl.minimum(window_end, source_end)
                    if overlap_start < overlap_end:
                        first_source_tile = (
                            overlap_start - source_start
                        ) // DISPATCH_BLOCK_SIZE_M
                        last_source_tile = (
                            overlap_end - source_start - 1
                        ) // DISPATCH_BLOCK_SIZE_M
                        for source_tile in range(
                            first_source_tile,
                            last_source_tile + 1,
                        ):
                            signal_slot = (
                                (source_id * EXPERTS_PER_RANK + expert_id)
                                * MAX_SOURCE_TILES
                                + source_tile
                            )
                            token = dl.wait(
                                signal_mem_ptr + signal_slot * 16,
                                1,
                                "gpu",
                                "acquire",
                                waitValue=signal_epoch,
                            )
                            ready_token += token
                    source_start = source_end

                ready_input_ptr = dl.consume_token(input_ptr, ready_token)
                _triton_grouped_gemm_one_mn_tile_tail(
                    ready_input_ptr, ready_weight_ptr, output_ptr,
                    expert_id, expert_off + window_start,
                    window_size, n_tile, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    GEMM_BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    WEIGHT_EXPERT_BASE, dtype)


@triton.jit
def _triton_grouped_gemm_one_mn_tile(
    input_ptr, weight_ptr, output_ptr,
    expert_id, m_off, m_size, n_tile, N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Compute one expert output tile instead of serializing the full N axis."""
    if m_size > 0:
        m_off64 = m_off.to(tl.int64)
        m_offs = m_off64 + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        m_mask = m_offs < m_off64 + m_size
        n_offs = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        weight_expert = (
            expert_id.to(tl.int64) - WEIGHT_EXPERT_BASE
        )
        weight_base = weight_ptr + weight_expert * stride_weight_0
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
            k_offs = k_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offs < K
            a_ptrs = (
                input_ptr
                + m_offs[:, None] * stride_input_m
                + k_offs[None, :] * stride_input_k
            )
            a = tl.load(
                a_ptrs,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            b_ptrs = (
                weight_base
                + k_offs[:, None] * stride_weight_2
                + n_offs[None, :] * stride_weight_1
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(a, b)

        c_ptrs = (
            output_ptr
            + m_offs[:, None] * stride_output_m
            + n_offs[None, :] * stride_output_n
        )
        tl.store(
            c_ptrs,
            acc.to(dtype),
            mask=m_mask[:, None] & n_mask[None, :],
        )


@triton.jit
def _triton_grouped_gemm_one_mn_tile_tail(
    input_ptr, weight_ptr, output_ptr,
    expert_id, m_off, m_size, n_tile, N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    dtype: tl.constexpr,
):
    """Dispatch one N tile to a smaller legal Cube M tile for tails."""
    if BLOCK_SIZE_M >= 64:
        if m_size <= BLOCK_SIZE_M // 4:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M // 4, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WEIGHT_EXPERT_BASE, dtype)
        elif m_size <= BLOCK_SIZE_M // 2:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M // 2, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WEIGHT_EXPERT_BASE, dtype)
        else:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WEIGHT_EXPERT_BASE, dtype)
    elif BLOCK_SIZE_M == 32:
        if m_size <= 16:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                16, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WEIGHT_EXPERT_BASE, dtype)
        else:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WEIGHT_EXPERT_BASE, dtype)
    else:
        _triton_grouped_gemm_one_mn_tile(
            input_ptr, weight_ptr, output_ptr,
            expert_id, m_off, m_size, n_tile, N, K,
            stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            WEIGHT_EXPERT_BASE, dtype)


__all__ = ["_kernel_dispatch_fc1"]
