# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Ascend (CANN/NPU) fused EP MoE dispatch + FC1 kernel — default schedule only.

Architecture:

  Phase 1 — In-kernel dispatch:
    Each AI core iterates local tokens and batches ACLSHMEM ``putmem_nbi``
    writes into remote ranks' symmetric memory (peer_mem).
    This replaces host-side ``all_to_all_single``.

  The all-core pipeline runs Vector dispatch and Cube FC1 concurrently on
    every AI core.  A dispatch task notifies a source-local tile after its
    remote stores are visible; FC1 waits only for the tile it is about to use.

This is the single-schedule counterpart of the former multi-strategy kernel.
It supports only the default ``MoEForwardConfig`` path:

  * ``dispatch_fc1_schedule = "allcore_expert_n_tile"``

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


@triton.jit(do_not_specialize=["signal_epoch"])
def _kernel_dispatch_fc1(
    # ---- Input / output ----
    input_ptr,
    peer_mem_ptr,
    routing_weight_ptr,
    routing_weight_recv_ptr,
    signal_mem_ptr,
    weight_ptr,
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
    stride_output_m,
    stride_output_n,

    # ---- Meta-parameters ----
    NUM_PROGRAM_CORES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
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

    # All AI cores expose independent Vector and Cube instruction streams.
    # The Vector side dispatches source-local tiles while the Cube side
    # consumes independently-ready receive tiles on the same logical PID.
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
                MAX_SOURCE_TILES, DISPATCH_BLOCK_SIZE_M)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _triton_grouped_gemm_expert_n_merged_tiles_wait(
            pid, NUM_PROGRAM_CORES,
            peer_mem_ptr, signal_mem_ptr, weight_ptr, output_ptr,
            recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
            signal_epoch,
            N, K, stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            DISPATCH_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K,
            WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)

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
            dst_offs = task_dst_start + tile_start + tile_token
            src_base = input_ptr + src_idx * stride_input_m
            dst_base = peer_mem_ptr + dst_offs * stride_input_m
            libshmem_device.putmem_nbi(
                dst_base, src_base, hidden * 2, dst_rank)
            route_idx = tl.load(send_route_idx_ptr + send_idx)
            if tile_token == tile_count - 1:
                # Blocking putmem submits this final write and then drains all
                # outstanding operations to dst_rank, including the payload
                # and routing-weight NBI writes issued earlier in this tile.
                libshmem_device.putmem(
                    routing_weight_recv_ptr + dst_offs,
                    routing_weight_ptr + route_idx,
                    4,
                    dst_rank,
                )
            else:
                libshmem_device.putmem_nbi(
                    routing_weight_recv_ptr + dst_offs,
                    routing_weight_ptr + route_idx,
                    4,
                    dst_rank,
                )

        # The final blocking write above is the transport-aware completion
        # point; keep the existing pipe/cache fence before publishing the tile.
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
    MAX_SOURCE_TILES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Assign cores only to nonempty buckets, then stripe each bucket's tiles."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    num_active_tasks = 0
    for task_id in range(0, num_tasks):
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
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    signal_epoch,
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
    MAX_SOURCE_TILES: tl.constexpr,
    dtype: tl.constexpr,
):
    """Consume merged expert M windows as their source tiles become ready.

    GEMM ownership is keyed only by ``(local_expert, n_tile)``.  Source rank
    participates solely in the readiness dependency calculation, so source
    fragments no longer force independent full-weight scans.  Expert-major
    task order lets all Cube cores cover one expert's N tiles together.
    """
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_tasks = EXPERTS_PER_RANK * num_n_tiles

    for task_id in range(pid, num_tasks, ncore):
        expert_id = task_id // num_n_tiles
        n_tile = task_id % num_n_tiles
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)

        if expert_size > 0:
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
                    ready_input_ptr, weight_ptr, output_ptr,
                    expert_id, expert_off + window_start,
                    window_size, n_tile, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    GEMM_BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


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
    dtype: tl.constexpr,
):
    """Compute one expert output tile instead of serializing the full N axis."""
    if m_size > 0:
        m_offs = m_off + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offs < m_off + m_size
        n_offs = n_tile * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offs < N

        weight_base = weight_ptr + expert_id.to(tl.int64) * stride_weight_0
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
                BLOCK_SIZE_M // 4, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        elif m_size <= BLOCK_SIZE_M // 2:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M // 2, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        else:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
    elif BLOCK_SIZE_M == 32:
        if m_size <= 16:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                16, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        else:
            _triton_grouped_gemm_one_mn_tile(
                input_ptr, weight_ptr, output_ptr,
                expert_id, m_off, m_size, n_tile, N, K,
                stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
    else:
        _triton_grouped_gemm_one_mn_tile(
            input_ptr, weight_ptr, output_ptr,
            expert_id, m_off, m_size, n_tile, N, K,
            stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


__all__ = ["_kernel_dispatch_fc1"]
