# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Ascend (CANN/NPU) fused EP MoE dispatch + FC1 kernel.

Architecture:

  Phase 1 — In-kernel dispatch:
    Each AI core iterates local tokens and uses ACLSHMEM ``putmem`` to write
    them directly into remote ranks' symmetric memory (peer_mem).
    This replaces host-side ``all_to_all_single``.

  The optimized all-core path runs Vector dispatch and Cube FC1 concurrently
  on every AI core.  A dispatch task notifies a source-local tile after its
  remote stores are visible; FC1 waits only for the tile it is about to use.
  Fixed producer/consumer roles remain available as fallback schedules.

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
    send_staging_ptr,
    peer_mem_ptr,
    routing_weight_ptr,
    routing_staging_ptr,
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
    N_DISPATCH_CORES: tl.constexpr,
    NUM_CONSUMER_CORES: tl.constexpr,
    NUM_PROGRAM_CORES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    TILE_READINESS: tl.constexpr,
    COUNT_DERIVED_SCHEDULE: tl.constexpr,
    ALL_CORE_PIPELINE: tl.constexpr,
    DIRECT_EXPERT_DISPATCH: tl.constexpr,
    TILE_BULK_DISPATCH: tl.constexpr,
    EXPERT_N_TILE_CONSUMER: tl.constexpr,
    MN_TILE_FC1: tl.constexpr,
    N_TILE_FC1: tl.constexpr,
    HAS_ROUTING_WEIGHT: tl.constexpr,
    FINAL_BARRIER: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Overlap remote dispatch and fc1 with selectable core scheduling.

    Tile mode publishes each source-local M tile with a SET(epoch) slot and
    starts its GEMM independently.  The optimized all-core path runs dispatch
    on every core's Vector stream and GEMM on every core's Cube stream.  Fixed
    producer/consumer roles and expert readiness remain as A/B fallbacks.
    """
    pid = tl.program_id(axis=0)
    dtype = tl.bfloat16

    if ALL_CORE_PIPELINE:
        # All AI cores expose independent Vector and Cube instruction streams.
        # The Vector side dispatches source-local tiles while the Cube side
        # consumes independently-ready receive tiles on the same logical PID.
        with al.scope(core_mode="vector", disable_auto_sync=True):
            if sub_vec_id() == 0:
                if DIRECT_EXPERT_DISPATCH:
                    _dispatch_direct_expert_buckets(
                        pid, NUM_PROGRAM_CORES,
                        send_staging_ptr, peer_mem_ptr,
                        routing_staging_ptr, routing_weight_recv_ptr,
                        signal_mem_ptr,
                        send_bucket_dst_starts_ptr,
                        send_bucket_starts_ptr, send_counts_re_ptr,
                        hidden, stride_input_m,
                        WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
                        True,
                        HAS_ROUTING_WEIGHT)
                else:
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
                        MAX_SOURCE_TILES, BLOCK_SIZE_M,
                        HAS_ROUTING_WEIGHT, TILE_BULK_DISPATCH,
                        EXPERT_N_TILE_CONSUMER)
        with al.scope(core_mode="cube", disable_auto_sync=True):
            if EXPERT_N_TILE_CONSUMER:
                _triton_grouped_gemm_expert_n_merged_tiles_wait(
                    pid, NUM_PROGRAM_CORES,
                    peer_mem_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                    recv_per_expert_ptr, recv_expert_offs_ptr,
                    recv_counts_re_ptr, signal_epoch,
                    N, K, stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    WORLD_SIZE, EXPERTS_PER_RANK,
                    MAX_SOURCE_TILES, dtype)
            elif DIRECT_EXPERT_DISPATCH:
                expert_counter_base = WORLD_SIZE * EXPERTS_PER_RANK * MAX_SOURCE_TILES
                if N_TILE_FC1:
                    _triton_grouped_gemm_expert_n_tiles_wait(
                        pid, NUM_PROGRAM_CORES,
                        peer_mem_ptr, signal_mem_ptr + expert_counter_base * 16,
                        weight_ptr, output_ptr,
                        recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
                        N, K, stride_input_m, stride_input_k,
                        stride_weight_0, stride_weight_1, stride_weight_2,
                        stride_output_m, stride_output_n,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                        WORLD_SIZE, EXPERTS_PER_RANK, dtype)
                elif MN_TILE_FC1:
                    _triton_grouped_gemm_expert_mn_tiles_wait(
                        pid, NUM_PROGRAM_CORES,
                        peer_mem_ptr, signal_mem_ptr + expert_counter_base * 16,
                        weight_ptr, output_ptr,
                        recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
                        N, K, stride_input_m, stride_input_k,
                        stride_weight_0, stride_weight_1, stride_weight_2,
                        stride_output_m, stride_output_n,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                        WORLD_SIZE, EXPERTS_PER_RANK, dtype)
                else:
                    _triton_grouped_gemm_expert_tiles_wait(
                        pid, NUM_PROGRAM_CORES,
                        peer_mem_ptr, signal_mem_ptr + expert_counter_base * 16,
                        weight_ptr, output_ptr,
                        recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
                        N, K, stride_input_m, stride_input_k,
                        stride_weight_0, stride_weight_1, stride_weight_2,
                        stride_output_m, stride_output_n,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                        WORLD_SIZE, EXPERTS_PER_RANK, dtype)
            else:
                _triton_grouped_gemm_count_derived_source_tiles_wait(
                    pid, NUM_PROGRAM_CORES,
                    peer_mem_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                    N, K, stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)
    elif pid < N_DISPATCH_CORES:
        with al.scope(core_mode="vector"):
            # A mixed tl.dot kernel has multiple Vector sub-cores per AI Core.
            # Exactly one sub-core owns communication, and the explicit Vector
            # scope prevents the Cube side from duplicating SHMEM operations.
            if sub_vec_id() == 0:
                if COUNT_DERIVED_SCHEDULE:
                    _dispatch_count_derived_source_tiles(
                        pid, N_DISPATCH_CORES,
                        input_ptr, peer_mem_ptr,
                        routing_weight_ptr, routing_weight_recv_ptr,
                        signal_mem_ptr,
                        send_src_idx_ptr, send_route_idx_ptr,
                        send_bucket_dst_starts_ptr,
                        send_bucket_starts_ptr, send_counts_re_ptr,
                        signal_epoch, hidden, stride_input_m,
                        LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK,
                        MAX_SOURCE_TILES, BLOCK_SIZE_M,
                        HAS_ROUTING_WEIGHT, False, False)
                else:
                    for task_idx in range(pid, WORLD_SIZE * EXPERTS_PER_RANK, N_DISPATCH_CORES):
                        task_start = tl.load(send_bucket_starts_ptr + task_idx)
                        task_count = tl.load(send_counts_re_ptr + task_idx)
                        task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_idx)
                        dst_rank = task_idx // EXPERTS_PER_RANK
                        signal_slot = task_idx % EXPERTS_PER_RANK
                        if TILE_READINESS:
                            # A source bucket is itself contiguous in the target
                            # expert region.  Publish each source-local M tile as
                            # soon as its rows are visible so Cube consumers do not
                            # wait for the rest of the bucket or other sources.
                            num_source_tiles = tl.cdiv(task_count, BLOCK_SIZE_M)
                            for source_tile in range(num_source_tiles):
                                tile_start = source_tile * BLOCK_SIZE_M
                                tile_count = tl.minimum(BLOCK_SIZE_M, task_count - tile_start)
                                for tile_token in range(tile_count):
                                    send_idx = task_start + tile_start + tile_token
                                    src_idx = tl.load(send_src_idx_ptr + send_idx)
                                    dst_offs = task_dst_start + tile_start + tile_token
                                    src_base = input_ptr + src_idx * stride_input_m
                                    dst_base = peer_mem_ptr + dst_offs * stride_input_m
                                    libshmem_device.putmem(dst_base, src_base, hidden * 2, dst_rank)
                                    if HAS_ROUTING_WEIGHT:
                                        route_idx = tl.load(send_route_idx_ptr + send_idx)
                                        libshmem_device.putmem(
                                            routing_weight_recv_ptr + dst_offs,
                                            routing_weight_ptr + route_idx,
                                            4,
                                            dst_rank,
                                        )
                                libshmem_device.fence()
                                tile_signal_slot = (
                                    (LOCAL_RANK * EXPERTS_PER_RANK + signal_slot) * MAX_SOURCE_TILES
                                    + source_tile
                                )
                                libshmem_device.signal_op(
                                    signal_mem_ptr + tile_signal_slot * 16,
                                    signal_epoch,
                                    libshmem_device.ACLSHMEM_SIGNAL_SET,
                                    dst_rank,
                                )
                        else:
                            for task_token_idx in range(task_count):
                                send_idx = task_start + task_token_idx
                                src_idx = tl.load(send_src_idx_ptr + send_idx)
                                dst_offs = task_dst_start + task_token_idx
                                src_base = input_ptr + src_idx * stride_input_m
                                dst_base = peer_mem_ptr + dst_offs * stride_input_m
                                libshmem_device.putmem(dst_base, src_base, hidden * 2, dst_rank)
                                if HAS_ROUTING_WEIGHT:
                                    route_idx = tl.load(send_route_idx_ptr + send_idx)
                                    libshmem_device.putmem(
                                        routing_weight_recv_ptr + dst_offs,
                                        routing_weight_ptr + route_idx,
                                        4,
                                        dst_rank,
                                    )

                            # Empty buckets also contribute one ADD so the expert
                            # counter advances by WORLD_SIZE on every epoch.
                            if task_count > 0:
                                libshmem_device.fence()
                            expert_counter_base = WORLD_SIZE * EXPERTS_PER_RANK * MAX_SOURCE_TILES
                            libshmem_device.signal_op(
                                signal_mem_ptr + (expert_counter_base + signal_slot) * 16,
                                1,
                                libshmem_device.ACLSHMEM_SIGNAL_ADD,
                                dst_rank,
                            )
    else:
        consumer_pid = pid - N_DISPATCH_CORES
        if TILE_READINESS:
            if COUNT_DERIVED_SCHEDULE:
                _triton_grouped_gemm_count_derived_source_tiles_wait(
                    consumer_pid, NUM_CONSUMER_CORES,
                    peer_mem_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                    N, K, stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)
            else:
                _triton_grouped_gemm_source_tiles_wait(
                    consumer_pid,
                    peer_mem_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                    N, K, stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    NUM_CONSUMER_CORES, WORLD_SIZE, EXPERTS_PER_RANK,
                    MAX_SOURCE_TILES, dtype)
        else:
            expert_counter_base = WORLD_SIZE * EXPERTS_PER_RANK * MAX_SOURCE_TILES
            _triton_grouped_gemm_tiled_wait(
                consumer_pid, NUM_CONSUMER_CORES,
                peer_mem_ptr, signal_mem_ptr + expert_counter_base * 16,
                weight_ptr, output_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr,
                signal_epoch,
                N, K, stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WORLD_SIZE, EXPERTS_PER_RANK, dtype)

    # Standalone dispatch must finish globally before its single receive
    # buffer can be reused.  A complete forward may defer this collective to
    # FC2/combine: all local input dependencies are already covered by expert
    # readiness, and the later combine kernel has its own rank-wide barrier.
    if FINAL_BARRIER:
        libshmem_device.barrier_all()


@triton.jit
def _triton_grouped_gemm_tiled_wait(
    pid, ncore,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    dtype: tl.constexpr,
):
    """Generate expert M tiles in-kernel and acquire each expert once."""
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_expert_tiles = tl.cdiv(expert_size, BLOCK_SIZE_M)

        if pid < num_expert_tiles:
            ready_input_ptr = input_ptr
            token = dl.wait(
                signal_mem_ptr + expert_id * 16,
                1,
                "gpu",
                "acquire",
                waitValue=signal_epoch * WORLD_SIZE,
            )
            ready_input_ptr = dl.consume_token(ready_input_ptr, token)

            for local_tile in range(pid, num_expert_tiles, ncore):
                m_off = expert_off + local_tile * BLOCK_SIZE_M
                m_remaining = expert_size - local_tile * BLOCK_SIZE_M
                m_size = tl.minimum(m_remaining, BLOCK_SIZE_M)
                _triton_grouped_gemm_one_tile(
                    ready_input_ptr, weight_ptr, output_ptr,
                    expert_id, m_off, m_size, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


@triton.jit
def _triton_grouped_gemm_expert_tiles_wait(
    pid, ncore: tl.constexpr,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    dtype: tl.constexpr,
):
    """Balance combined expert M tiles over all Cube cores after expert readiness."""
    tile_base = 0
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_expert_tiles = tl.cdiv(expert_size, BLOCK_SIZE_M)
        for local_tile in range(0, num_expert_tiles):
            tile_id = tile_base + local_tile
            if tile_id % ncore == pid:
                token = dl.wait(
                    signal_mem_ptr + expert_id * 16,
                    1,
                    "gpu",
                    "acquire",
                    waitValue=signal_epoch * WORLD_SIZE,
                )
                ready_input_ptr = dl.consume_token(input_ptr, token)
                m_off = expert_off + local_tile * BLOCK_SIZE_M
                m_size = tl.minimum(
                    expert_size - local_tile * BLOCK_SIZE_M,
                    BLOCK_SIZE_M,
                )
                _triton_grouped_gemm_one_tile(
                    ready_input_ptr, weight_ptr, output_ptr,
                    expert_id, m_off, m_size, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        tile_base += num_expert_tiles


@triton.jit
def _triton_grouped_gemm_expert_mn_tiles_wait(
    pid, ncore: tl.constexpr,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    dtype: tl.constexpr,
):
    """Balance individual expert M/N output tiles over every Cube core.

    The older expert scheduler assigns a complete M tile to one core and then
    computes every N tile serially on that core.  DSV4 has only about two M
    tiles per local expert but 24 N tiles, so expert readiness exposes work to
    only a few of the 24 Cube cores.  This mapping makes the N dimension part
    of the global task id and lets all cores consume a newly-ready expert.
    """
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    task_base = 0
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_m_tiles = tl.cdiv(expert_size, BLOCK_SIZE_M)
        num_expert_tasks = num_m_tiles * num_n_tiles

        # Rotate ownership by the cumulative task base so rank-MAX imbalance
        # does not leave the same cores with every expert's tail.  A core
        # acquires an expert once, then processes all of its strided M/N tasks.
        first_task = (pid - task_base % ncore + ncore) % ncore
        if first_task < num_expert_tasks:
            token = dl.wait(
                signal_mem_ptr + expert_id * 16,
                1,
                "gpu",
                "acquire",
                waitValue=signal_epoch * WORLD_SIZE,
            )
            ready_input_ptr = dl.consume_token(input_ptr, token)
            for expert_task in range(first_task, num_expert_tasks, ncore):
                m_tile = expert_task // num_n_tiles
                n_tile = expert_task % num_n_tiles
                m_off = expert_off + m_tile * BLOCK_SIZE_M
                m_size = tl.minimum(
                    expert_size - m_tile * BLOCK_SIZE_M,
                    BLOCK_SIZE_M,
                )
                _triton_grouped_gemm_one_mn_tile(
                    ready_input_ptr, weight_ptr, output_ptr,
                    expert_id, m_off, m_size, n_tile, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        task_base += num_expert_tasks


@triton.jit
def _triton_grouped_gemm_expert_n_tiles_wait(
    pid, ncore: tl.constexpr,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    dtype: tl.constexpr,
):
    """Assign an expert N tile to one core, then traverse its M tiles.

    When an expert spans multiple M tiles, keeping one N tile on the same core
    lets later dots reuse the same transposed W1 slice.  Independent N tasks
    retain parallelism without encoding any model or world-size policy here.
    """
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    n_task_base = 0
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        num_full_m_tiles = expert_size // BLOCK_SIZE_M
        tail_size = expert_size - num_full_m_tiles * BLOCK_SIZE_M
        num_expert_n_tasks = tl.where(expert_size > 0, num_n_tiles, 0)

        first_n_task = (pid - n_task_base % ncore + ncore) % ncore
        if first_n_task < num_expert_n_tasks:
            token = dl.wait(
                signal_mem_ptr + expert_id * 16,
                1,
                "gpu",
                "acquire",
                waitValue=signal_epoch * WORLD_SIZE,
            )
            ready_input_ptr = dl.consume_token(input_ptr, token)
            for n_tile in range(first_n_task, num_n_tiles, ncore):
                for m_tile in range(0, num_full_m_tiles):
                    m_off = expert_off + m_tile * BLOCK_SIZE_M
                    _triton_grouped_gemm_one_mn_tile(
                        ready_input_ptr, weight_ptr, output_ptr,
                        expert_id, m_off, BLOCK_SIZE_M, n_tile, N, K,
                        stride_input_m, stride_input_k,
                        stride_weight_0, stride_weight_1, stride_weight_2,
                        stride_output_m, stride_output_n,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
                if tail_size > 0:
                    tail_m_off = expert_off + num_full_m_tiles * BLOCK_SIZE_M
                    _triton_grouped_gemm_one_mn_tile_tail(
                        ready_input_ptr, weight_ptr, output_ptr,
                        expert_id, tail_m_off, tail_size, n_tile, N, K,
                        stride_input_m, stride_input_k,
                        stride_weight_0, stride_weight_1, stride_weight_2,
                        stride_output_m, stride_output_n,
                        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)
        n_task_base += num_expert_n_tasks


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
    BLOCK_SIZE_M: tl.constexpr,
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
            num_m_windows = tl.cdiv(expert_size, BLOCK_SIZE_M)
            for m_window in range(0, num_m_windows):
                window_start = m_window * BLOCK_SIZE_M
                window_size = tl.minimum(
                    BLOCK_SIZE_M,
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
                        ) // BLOCK_SIZE_M
                        last_source_tile = (
                            overlap_end - source_start - 1
                        ) // BLOCK_SIZE_M
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
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


@triton.jit
def _find_nth_nonempty_task(
    counts_ptr,
    active_task_id,
    NUM_TASKS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    EXPERT_MAJOR_TASK_ORDER: tl.constexpr,
):
    """Map a dense active-task ordinal to its sparse count-matrix index."""
    selected_task = 0
    num_seen = 0
    for schedule_task in range(0, NUM_TASKS):
        if EXPERT_MAJOR_TASK_ORDER:
            expert_id = schedule_task // WORLD_SIZE
            dst_rank = schedule_task % WORLD_SIZE
            candidate_task = dst_rank * EXPERTS_PER_RANK + expert_id
        else:
            candidate_task = schedule_task
        task_count = tl.load(counts_ptr + candidate_task)
        is_nonempty = task_count > 0
        is_selected = is_nonempty & (num_seen == active_task_id)
        selected_task = tl.where(is_selected, candidate_task, selected_task)
        num_seen += tl.where(is_nonempty, 1, 0)
    return selected_task


@triton.jit
def _dispatch_direct_expert_buckets(
    pid, num_cores: tl.constexpr,
    send_staging_ptr, peer_mem_ptr,
    routing_staging_ptr, routing_weight_recv_ptr,
    signal_mem_ptr,
    send_bucket_dst_starts_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr,
    hidden: tl.constexpr,
    stride_input_m,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    PUBLISH_READINESS: tl.constexpr,
    HAS_ROUTING_WEIGHT: tl.constexpr,
):
    """Bulk-dispatch one pre-gathered bucket, then publish readiness."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    expert_counter_base: tl.constexpr = num_tasks * MAX_SOURCE_TILES
    # The metadata matrix is destination-major, but visiting it in that order
    # makes every source rank send to rank 0 first and the last rank last.  The
    # rank-MAX critical path then loses almost all dispatch/FC1 overlap.  Walk
    # an expert-major work ordinal instead so every destination receives early
    # expert buckets concurrently while preserving the exact metadata layout.
    for work_id in range(pid, num_tasks, num_cores):
        dst_rank = work_id % WORLD_SIZE
        expert_id = work_id // WORLD_SIZE
        task_id = dst_rank * EXPERTS_PER_RANK + expert_id
        task_start = tl.load(send_bucket_starts_ptr + task_id)
        task_count = tl.load(send_counts_re_ptr + task_id)
        task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
        if task_count > 0:
            if PUBLISH_READINESS:
                libshmem_device.putmem(
                    peer_mem_ptr + task_dst_start * stride_input_m,
                    send_staging_ptr + task_start * hidden,
                    task_count * hidden * 2,
                    dst_rank,
                )
                if HAS_ROUTING_WEIGHT:
                    libshmem_device.putmem(
                        routing_weight_recv_ptr + task_dst_start,
                        routing_staging_ptr + task_start,
                        task_count * 4,
                        dst_rank,
                    )
                libshmem_device.fence()
            else:
                libshmem_device.putmem(
                    peer_mem_ptr + task_dst_start * stride_input_m,
                    send_staging_ptr + task_start * hidden,
                    task_count * hidden * 2,
                    dst_rank,
                )
                if HAS_ROUTING_WEIGHT:
                    libshmem_device.putmem(
                        routing_weight_recv_ptr + task_dst_start,
                        routing_staging_ptr + task_start,
                        task_count * 4,
                        dst_rank,
                    )
        if PUBLISH_READINESS:
            libshmem_device.signal_op(
                signal_mem_ptr + (expert_counter_base + expert_id) * 16,
                1,
                libshmem_device.ACLSHMEM_SIGNAL_ADD,
                dst_rank,
            )


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
    BLOCK_SIZE_M: tl.constexpr,
    HAS_ROUTING_WEIGHT: tl.constexpr,
    TILE_BULK_DISPATCH: tl.constexpr,
):
    """Dispatch the source-local tiles assigned to one lane of a nonempty bucket."""
    task_start = tl.load(send_bucket_starts_ptr + task_id)
    task_count = tl.load(send_counts_re_ptr + task_id)
    task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
    dst_rank = task_id // EXPERTS_PER_RANK
    expert_id = task_id % EXPERTS_PER_RANK
    num_source_tiles = tl.cdiv(task_count, BLOCK_SIZE_M)

    for source_tile in range(task_lane, num_source_tiles, task_cores):
        tile_start = source_tile * BLOCK_SIZE_M
        tile_count = tl.minimum(BLOCK_SIZE_M, task_count - tile_start)
        if TILE_BULK_DISPATCH:
            send_offs = task_start + tile_start
            dst_offs = task_dst_start + tile_start
            libshmem_device.putmem(
                peer_mem_ptr + dst_offs * stride_input_m,
                input_ptr + send_offs * stride_input_m,
                tile_count * hidden * 2,
                dst_rank,
            )
            if HAS_ROUTING_WEIGHT:
                libshmem_device.putmem(
                    routing_weight_recv_ptr + dst_offs,
                    routing_weight_ptr + send_offs,
                    tile_count * 4,
                    dst_rank,
                )
        else:
            for tile_token in range(tile_count):
                send_idx = task_start + tile_start + tile_token
                src_idx = tl.load(send_src_idx_ptr + send_idx)
                dst_offs = task_dst_start + tile_start + tile_token
                src_base = input_ptr + src_idx * stride_input_m
                dst_base = peer_mem_ptr + dst_offs * stride_input_m
                libshmem_device.putmem(dst_base, src_base, hidden * 2, dst_rank)
                if HAS_ROUTING_WEIGHT:
                    route_idx = tl.load(send_route_idx_ptr + send_idx)
                    libshmem_device.putmem(
                        routing_weight_recv_ptr + dst_offs,
                        routing_weight_ptr + route_idx,
                        4,
                        dst_rank,
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
    MAX_SOURCE_TILES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_ROUTING_WEIGHT: tl.constexpr,
    TILE_BULK_DISPATCH: tl.constexpr,
    EXPERT_MAJOR_TASK_ORDER: tl.constexpr,
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
                WORLD_SIZE, EXPERTS_PER_RANK, EXPERT_MAJOR_TASK_ORDER)
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
                BLOCK_SIZE_M, HAS_ROUTING_WEIGHT, TILE_BULK_DISPATCH)
        else:
            for active_task_id in range(pid, num_active_tasks, num_cores):
                task_id = _find_nth_nonempty_task(
                    send_counts_re_ptr, active_task_id, num_tasks,
                    WORLD_SIZE, EXPERTS_PER_RANK, EXPERT_MAJOR_TASK_ORDER)
                _dispatch_one_source_tile_task(
                    task_id, 0, 1,
                    input_ptr, peer_mem_ptr,
                    routing_weight_ptr, routing_weight_recv_ptr,
                    signal_mem_ptr,
                    send_src_idx_ptr, send_route_idx_ptr,
                    send_bucket_dst_starts_ptr,
                    send_bucket_starts_ptr, send_counts_re_ptr,
                    signal_epoch, hidden, stride_input_m,
                    LOCAL_RANK, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
                    BLOCK_SIZE_M, HAS_ROUTING_WEIGHT, TILE_BULK_DISPATCH)


@triton.jit
def _triton_grouped_gemm_source_tile_task(
    task_id, task_lane, task_cores,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    dtype: tl.constexpr,
):
    """Consume source-local M tiles as their individual SET slots become ready."""
    source_id = task_id // EXPERTS_PER_RANK
    expert_id = task_id % EXPERTS_PER_RANK
    source_size = tl.load(
        recv_counts_re_ptr + source_id * EXPERTS_PER_RANK + expert_id)

    source_off = tl.load(recv_expert_offs_ptr + expert_id)
    for prior_source in range(0, WORLD_SIZE):
        if prior_source < source_id:
            source_off += tl.load(
                recv_counts_re_ptr + prior_source * EXPERTS_PER_RANK + expert_id)

    num_source_tiles = tl.cdiv(source_size, BLOCK_SIZE_M)
    for source_tile in range(task_lane, num_source_tiles, task_cores):
        signal_slot = (
            (source_id * EXPERTS_PER_RANK + expert_id) * MAX_SOURCE_TILES
            + source_tile
        )
        token = dl.wait(
            signal_mem_ptr + signal_slot * 16,
            1,
            "gpu",
            "acquire",
            waitValue=signal_epoch,
        )
        ready_input_ptr = dl.consume_token(input_ptr, token)
        m_off = source_off + source_tile * BLOCK_SIZE_M
        m_size = tl.minimum(source_size - source_tile * BLOCK_SIZE_M, BLOCK_SIZE_M)
        _triton_grouped_gemm_one_tile(
            ready_input_ptr, weight_ptr, output_ptr,
            expert_id, m_off, m_size, N, K,
            stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


@triton.jit
def _triton_grouped_gemm_source_tiles_wait(
    pid,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_CORES: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    dtype: tl.constexpr,
):
    """Statically spread source/expert tasks, then dynamically wait per M tile."""
    num_source_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    if NUM_CONSUMER_CORES >= num_source_tasks:
        task_id = pid % num_source_tasks
        task_lane = pid // num_source_tasks
        task_cores = (NUM_CONSUMER_CORES + num_source_tasks - 1 - task_id) // num_source_tasks
        _triton_grouped_gemm_source_tile_task(
            task_id, task_lane, task_cores,
            input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
            recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
            N, K, stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)
    else:
        for task_id in range(pid, num_source_tasks, NUM_CONSUMER_CORES):
            _triton_grouped_gemm_source_tile_task(
                task_id, 0, 1,
                input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                N, K, stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)


@triton.jit
def _triton_grouped_gemm_count_derived_source_tiles_wait(
    pid, num_cores: tl.constexpr,
    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
    N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    dtype: tl.constexpr,
):
    """Assign Cube cores across nonempty source/expert tasks from receive counts."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    num_active_tasks = 0
    for task_id in range(0, num_tasks):
        task_count = tl.load(recv_counts_re_ptr + task_id)
        num_active_tasks += tl.where(task_count > 0, 1, 0)

    if num_active_tasks > 0:
        if num_cores >= num_active_tasks:
            active_task_id = pid % num_active_tasks
            task_lane = pid // num_active_tasks
            task_cores = (
                num_cores + num_active_tasks - 1 - active_task_id
            ) // num_active_tasks
            task_id = _find_nth_nonempty_task(
                recv_counts_re_ptr, active_task_id, num_tasks,
                WORLD_SIZE, EXPERTS_PER_RANK, False)
            _triton_grouped_gemm_source_tile_task(
                task_id, task_lane, task_cores,
                input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                N, K, stride_input_m, stride_input_k,
                stride_weight_0, stride_weight_1, stride_weight_2,
                stride_output_m, stride_output_n,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)
        else:
            for active_task_id in range(pid, num_active_tasks, num_cores):
                task_id = _find_nth_nonempty_task(
                    recv_counts_re_ptr, active_task_id, num_tasks,
                    WORLD_SIZE, EXPERTS_PER_RANK, False)
                _triton_grouped_gemm_source_tile_task(
                    task_id, 0, 1,
                    input_ptr, signal_mem_ptr, weight_ptr, output_ptr,
                    recv_counts_re_ptr, recv_expert_offs_ptr, signal_epoch,
                    N, K, stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                    WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, dtype)


@triton.jit
def _triton_grouped_gemm_one_tile(
    input_ptr, weight_ptr, output_ptr,
    expert_id, m_off, m_size, N, K,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    dtype: tl.constexpr,
):
    """Compute all M/N blocks belonging to one expert tile."""

    if m_size > 0:
        num_m_blocks = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)
        m_end = m_off + m_size  # exclusive upper bound of valid M rows

        # Packed DSV4 W1 has stride_e = 2F * H = 44,040,192 elements.
        # With 96 local experts (W4), expert_id * stride_e exceeds signed
        # int32 from expert 49 onward even though each operand fits int32.
        # Widen before multiplication so pointer arithmetic remains correct
        # for W4/W2 expert shards.
        weight_base = weight_ptr + expert_id.to(tl.int64) * stride_weight_0

        for m_block in range(num_m_blocks):
            m_start = m_block * BLOCK_SIZE_M
            # direct comparison masks (no tl.minimum) — matches tutorial 01.
            m_offs = m_off + m_start + tl.arange(0, BLOCK_SIZE_M)
            m_mask = m_offs < m_end

            for n_block in range(num_n_blocks):
                n_start = n_block * BLOCK_SIZE_N
                n_offs = n_start + tl.arange(0, BLOCK_SIZE_N)
                n_mask = n_offs < N

                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                for k_block in range(tl.cdiv(K, BLOCK_SIZE_K)):
                    k_start = k_block * BLOCK_SIZE_K
                    k_offs = k_start + tl.arange(0, BLOCK_SIZE_K)
                    k_mask = k_offs < K

                    # Load input [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    a_ptrs = (input_ptr +
                              m_offs[:, None] * stride_input_m +
                              k_offs[None, :] * stride_input_k)
                    a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

                    # Load weight [BLOCK_SIZE_K, BLOCK_SIZE_N]
                    # weight[expert, N, K] → read as [K, N]
                    b_ptrs = (weight_base +
                              k_offs[:, None] * stride_weight_2 +
                              n_offs[None, :] * stride_weight_1)
                    b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

                    acc += tl.dot(a, b)

                c = acc.to(dtype)
                c_ptrs = (output_ptr +
                          m_offs[:, None] * stride_output_m +
                          n_offs[None, :] * stride_output_n)
                tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


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
