# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Single-launch home-expert Mega-MoE forward for Ascend.

This kernel deliberately favors a simple, auditable barrier schedule over
overlap. Routing, dispatch/FC1, weighted activation, FC2, reverse transport,
and local top-k reduction all execute in one physical all-core launch.
"""

import triton
import triton.language as tl
from triton.language.extra.cann.extension import sub_vec_id
import triton.language.extra.cann.extension as al
from triton_dist.language.extra import libshmem_device

from .dispatch_fc1 import (
    _dispatch_count_derived_source_tiles,
    _triton_grouped_gemm_expert_n_merged_tiles_wait,
)
from .fc2_combine import _fc2_gemm_one_mn_tile
from .weighted_swiglu import _weighted_activation_rows

_ROUTE_BLOCK: tl.constexpr = 256
_ACTIVATION_BLOCK_M: tl.constexpr = 8
_ACTIVATION_BLOCK_N: tl.constexpr = 128
_REDUCE_BLOCK_N: tl.constexpr = 256


@triton.jit
def _zero_route_workspaces(
    pid,
    core_bucket_cursor_ptr,
    route_to_send_ptr,
    num_routes,
    NUM_PROGRAM_CORES: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
):
    bucket_offsets = tl.arange(0, NUM_BINS_PAD)
    tl.store(
        core_bucket_cursor_ptr + pid * NUM_BINS_PAD + bucket_offsets,
        0,
    )

    num_route_tiles = tl.cdiv(num_routes, ROUTE_BLOCK)
    for tile_id in range(pid, num_route_tiles, NUM_PROGRAM_CORES):
        route_ids = tile_id * ROUTE_BLOCK + tl.arange(0, ROUTE_BLOCK)
        tl.store(
            route_to_send_ptr + route_ids,
            -1,
            mask=route_ids < num_routes,
        )


@triton.jit
def _count_routes_by_core(
    pid,
    selected_experts_ptr,
    core_bucket_cursor_ptr,
    num_routes,
    NUM_PROGRAM_CORES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    """Build one private histogram row per physical AICore.

    Route chunks are contiguous and ordered by ``pid``.  That property lets a
    later prefix over the private rows reproduce stable argsort order without
    atomics in the scatter phase.
    """
    routes_per_core = tl.cdiv(num_routes, NUM_PROGRAM_CORES)
    route_id = pid * routes_per_core
    route_end = tl.minimum(route_id + routes_per_core, num_routes)
    while route_id < route_end:
        expert = tl.load(selected_experts_ptr + route_id)
        if (expert >= 0) & (expert < NUM_EXPERTS):
            count_ptr = (
                core_bucket_cursor_ptr
                + pid * NUM_BINS_PAD
                + expert
            )
            count = tl.load(count_ptr)
            tl.store(count_ptr, count + 1)
        route_id += 1


@triton.jit
def _publish_count_row(
    core_bucket_cursor_ptr,
    counts_mem_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_PROGRAM_CORES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    local_row_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    for bucket in range(0, NUM_BINS_PAD):
        count = 0
        if bucket < NUM_EXPERTS:
            for core_id in range(0, NUM_PROGRAM_CORES):
                count += tl.load(
                    core_bucket_cursor_ptr
                    + core_id * NUM_BINS_PAD
                    + bucket
                )
        tl.store(local_row_ptr + bucket, count)

    for peer_rank in range(0, WORLD_SIZE):
        if peer_rank != LOCAL_RANK:
            libshmem_device.putmem(
                local_row_ptr,
                local_row_ptr,
                NUM_BINS_PAD * 4,
                peer_rank,
            )
    libshmem_device.fence()


@triton.jit
def _build_destination_metadata(
    dst_rank,
    counts_mem_ptr,
    send_bucket_starts_ptr,
    send_bucket_dst_starts_ptr,
    recv_counts_re_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stats_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    local_row_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    first_bucket = dst_rank * EXPERTS_PER_RANK
    send_running = 0
    for bucket in range(0, first_bucket):
        send_running += tl.load(local_row_ptr + bucket)

    expert_base = 0
    for local_expert in range(0, EXPERTS_PER_RANK):
        bucket = first_bucket + local_expert
        local_count = tl.load(local_row_ptr + bucket)
        target_total = 0
        source_prefix = 0
        for source_rank in range(0, WORLD_SIZE):
            source_count = tl.load(
                counts_mem_ptr + source_rank * NUM_BINS_PAD + bucket
            )
            target_total += source_count
            if source_rank < LOCAL_RANK:
                source_prefix += source_count
            if dst_rank == LOCAL_RANK:
                tl.store(
                    recv_counts_re_ptr
                    + source_rank * EXPERTS_PER_RANK
                    + local_expert,
                    source_count,
                )

        tl.store(send_bucket_starts_ptr + bucket, send_running)
        tl.store(
            send_bucket_dst_starts_ptr + bucket,
            expert_base + source_prefix,
        )
        if dst_rank == LOCAL_RANK:
            tl.store(recv_per_expert_ptr + local_expert, target_total)
            tl.store(recv_expert_offs_ptr + local_expert, expert_base)

        send_running += local_count
        expert_base += target_total

    if dst_rank == LOCAL_RANK:
        tl.store(recv_expert_offs_ptr + EXPERTS_PER_RANK, expert_base)
        tl.store(stats_ptr, expert_base)
    tl.store(stats_ptr + 2 + dst_rank, expert_base)


@triton.jit
def _convert_counts_to_stable_cursors(
    pid,
    core_bucket_cursor_ptr,
    send_bucket_starts_ptr,
    NUM_PROGRAM_CORES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    for bucket in range(pid, NUM_EXPERTS, NUM_PROGRAM_CORES):
        cursor = tl.load(send_bucket_starts_ptr + bucket)
        for core_id in range(0, NUM_PROGRAM_CORES):
            count_ptr = (
                core_bucket_cursor_ptr
                + core_id * NUM_BINS_PAD
                + bucket
            )
            count = tl.load(count_ptr)
            tl.store(count_ptr, cursor)
            cursor += count


@triton.jit
def _build_pull_metadata(
    source_rank,
    counts_mem_ptr,
    recv_expert_offs_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
):
    remote_send_cursor = 0
    for bucket in range(0, NUM_EXPERTS):
        route_count = tl.load(
            counts_mem_ptr + source_rank * NUM_BINS_PAD + bucket
        )
        destination_rank = bucket // EXPERTS_PER_RANK
        local_expert = bucket % EXPERTS_PER_RANK
        if destination_rank == LOCAL_RANK:
            source_local_start = tl.load(
                recv_expert_offs_ptr + local_expert
            )
            for prior_source in range(0, WORLD_SIZE):
                if prior_source < source_rank:
                    source_local_start += tl.load(
                        counts_mem_ptr
                        + prior_source * NUM_BINS_PAD
                        + bucket
                    )
            segment_id = local_expert * WORLD_SIZE + source_rank
            tl.store(
                pull_tile_rank_ptr + segment_id,
                tl.where(route_count > 0, source_rank, -1),
            )
            tl.store(
                pull_tile_src_start_ptr + segment_id,
                source_local_start,
            )
            tl.store(
                pull_tile_dst_start_ptr + segment_id,
                remote_send_cursor,
            )
            tl.store(pull_tile_row_count_ptr + segment_id, route_count)
        remote_send_cursor += route_count


@triton.jit
def _scatter_stable_routes(
    pid,
    selected_experts_ptr,
    core_bucket_cursor_ptr,
    send_token_indices_ptr,
    send_route_indices_ptr,
    route_to_send_ptr,
    num_routes,
    NUM_PROGRAM_CORES: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    TOPK: tl.constexpr,
):
    routes_per_core = tl.cdiv(num_routes, NUM_PROGRAM_CORES)
    route_id = pid * routes_per_core
    route_end = tl.minimum(route_id + routes_per_core, num_routes)
    while route_id < route_end:
        expert = tl.load(selected_experts_ptr + route_id)
        if (expert >= 0) & (expert < NUM_EXPERTS):
            cursor_ptr = (
                core_bucket_cursor_ptr
                + pid * NUM_BINS_PAD
                + expert
            )
            send_row = tl.load(cursor_ptr)
            tl.store(cursor_ptr, send_row + 1)
            tl.store(send_token_indices_ptr + send_row, route_id // TOPK)
            tl.store(send_route_indices_ptr + send_row, route_id)
            tl.store(route_to_send_ptr + route_id, send_row)
        route_id += 1


@triton.jit
def _fc2_all_experts(
    pid,
    weighted_activation_ptr,
    down_weight_ptr,
    peer_mem_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    NUM_PROGRAM_CORES: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    num_n_tiles: tl.constexpr = N // BLOCK_N
    num_tasks: tl.constexpr = EXPERTS_PER_RANK * num_n_tiles
    for task_id in range(pid, num_tasks, NUM_PROGRAM_CORES):
        expert_id = task_id // num_n_tiles
        n_tile = task_id % num_n_tiles
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        if expert_size > 0:
            num_m_windows = tl.cdiv(expert_size, BLOCK_M)
            for m_window in range(0, num_m_windows):
                row_start = expert_off + m_window * BLOCK_M
                row_count = tl.minimum(
                    BLOCK_M,
                    expert_size - m_window * BLOCK_M,
                )
                _fc2_gemm_one_mn_tile(
                    weighted_activation_ptr,
                    down_weight_ptr,
                    peer_mem_ptr,
                    expert_id,
                    row_start,
                    row_count,
                    n_tile,
                    N,
                    K,
                    stride_input_m,
                    stride_input_k,
                    stride_weight_e,
                    stride_weight_n,
                    stride_weight_k,
                    N,
                    1,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    0,
                )


@triton.jit
def _reverse_put_fc2_rows(
    pid,
    combine_buf_ptr,
    peer_mem_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    NUM_PROGRAM_CORES: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    HIDDEN: tl.constexpr,
):
    num_segments: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    for segment_id in range(pid, num_segments, NUM_PROGRAM_CORES):
        encoded_rank = tl.load(pull_tile_rank_ptr + segment_id)
        if encoded_rank >= 0:
            src_start = tl.load(
                pull_tile_src_start_ptr + segment_id
            ).to(tl.int64)
            dst_start = tl.load(
                pull_tile_dst_start_ptr + segment_id
            ).to(tl.int64)
            row_count = tl.load(
                pull_tile_row_count_ptr + segment_id
            ).to(tl.int64)
            libshmem_device.putmem(
                combine_buf_ptr + dst_start * HIDDEN,
                peer_mem_ptr + src_start * HIDDEN,
                row_count * HIDDEN * 2,
                encoded_rank,
            )
    libshmem_device.fence()


@triton.jit
def _reduce_topk_rows(
    pid,
    combine_buf_ptr,
    route_to_send_ptr,
    output_ptr,
    num_tokens,
    capacity_ok,
    NUM_PROGRAM_CORES: tl.constexpr,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols_in_block = tl.arange(0, BLOCK_N)
    for token_id in range(pid, num_tokens, NUM_PROGRAM_CORES):
        route_base = token_id * TOPK
        for col_start in range(0, HIDDEN, BLOCK_N):
            cols = col_start + cols_in_block
            mask_n = cols < HIDDEN
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            if capacity_ok:
                for topk_slot in tl.static_range(0, TOPK):
                    send_row = tl.load(
                        route_to_send_ptr + route_base + topk_slot
                    )
                    valid = send_row >= 0
                    safe_row = tl.where(valid, send_row, 0).to(tl.int64)
                    value = tl.load(
                        combine_buf_ptr + safe_row * HIDDEN + cols,
                        mask=mask_n & valid,
                        other=0.0,
                    ).to(tl.float32)
                    acc += value
            tl.store(
                output_ptr + token_id.to(tl.int64) * HIDDEN + cols,
                acc.to(tl.bfloat16),
                mask=mask_n,
            )


@triton.jit(do_not_specialize=["num_routes", "signal_epoch"])
def _kernel_fused_forward(
    hidden_states_ptr,
    selected_experts_ptr,
    routing_weights_ptr,
    gate_up_weight_ptr,
    down_weight_ptr,
    peer_mem_ptr,
    routing_weight_recv_ptr,
    signal_mem_ptr,
    combine_buf_ptr,
    fc1_output_ptr,
    weighted_activation_ptr,
    output_ptr,
    counts_mem_ptr,
    send_bucket_starts_ptr,
    send_bucket_dst_starts_ptr,
    recv_counts_re_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stats_ptr,
    core_bucket_cursor_ptr,
    send_token_indices_ptr,
    send_route_indices_ptr,
    route_to_send_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    num_routes,
    signal_epoch,
    situ_beta,
    situ_linear_beta,
    stride_hidden_m,
    stride_hidden_k,
    stride_gate_up_e,
    stride_gate_up_n,
    stride_gate_up_k,
    stride_down_e,
    stride_down_n,
    stride_down_k,
    NUM_PROGRAM_CORES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    FFN: tl.constexpr,
    MAX_RECEIVED_ROUTES: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    DISPATCH_BLOCK_M: tl.constexpr,
    FC1_BLOCK_M: tl.constexpr,
    FC1_BLOCK_N: tl.constexpr,
    FC1_BLOCK_K: tl.constexpr,
    FC2_BLOCK_M: tl.constexpr,
    FC2_BLOCK_N: tl.constexpr,
    FC2_BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # R0: initialize all ordinary routing workspaces.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _zero_route_workspaces(
                pid,
                core_bucket_cursor_ptr,
                route_to_send_ptr,
                num_routes,
                NUM_PROGRAM_CORES,
                NUM_BINS_PAD,
                _ROUTE_BLOCK,
            )
    libshmem_device.barrier_all()

    # R1: private per-core histograms, followed by one published symmetric row.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _count_routes_by_core(
                pid,
                selected_experts_ptr,
                core_bucket_cursor_ptr,
                num_routes,
                NUM_PROGRAM_CORES,
                NUM_EXPERTS,
                NUM_BINS_PAD,
            )
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid == 0):
            _publish_count_row(
                core_bucket_cursor_ptr,
                counts_mem_ptr,
                LOCAL_RANK,
                WORLD_SIZE,
                NUM_PROGRAM_CORES,
                NUM_EXPERTS,
                NUM_BINS_PAD,
            )
    libshmem_device.barrier_all()

    # R2: derive the same expert/source-major metadata as the reference path.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid < WORLD_SIZE):
            _build_destination_metadata(
                pid,
                counts_mem_ptr,
                send_bucket_starts_ptr,
                send_bucket_dst_starts_ptr,
                recv_counts_re_ptr,
                recv_per_expert_ptr,
                recv_expert_offs_ptr,
                stats_ptr,
                LOCAL_RANK,
                WORLD_SIZE,
                EXPERTS_PER_RANK,
                NUM_BINS_PAD,
            )
    libshmem_device.barrier_all()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _convert_counts_to_stable_cursors(
                pid,
                core_bucket_cursor_ptr,
                send_bucket_starts_ptr,
                NUM_PROGRAM_CORES,
                NUM_EXPERTS,
                NUM_BINS_PAD,
            )
            if pid < WORLD_SIZE:
                _build_pull_metadata(
                    pid,
                    counts_mem_ptr,
                    recv_expert_offs_ptr,
                    pull_tile_rank_ptr,
                    pull_tile_src_start_ptr,
                    pull_tile_dst_start_ptr,
                    pull_tile_row_count_ptr,
                    LOCAL_RANK,
                    WORLD_SIZE,
                    EXPERTS_PER_RANK,
                    NUM_EXPERTS,
                    NUM_BINS_PAD,
                )
            if pid == 0:
                max_received = 0
                for dst_rank in range(0, WORLD_SIZE):
                    max_received = tl.maximum(
                        max_received,
                        tl.load(stats_ptr + 2 + dst_rank),
                    )
                tl.store(stats_ptr + 1, max_received)
    libshmem_device.barrier_all()

    # R3: stable direct scatter into the send permutation and its inverse.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _scatter_stable_routes(
                pid,
                selected_experts_ptr,
                core_bucket_cursor_ptr,
                send_token_indices_ptr,
                send_route_indices_ptr,
                route_to_send_ptr,
                num_routes,
                NUM_PROGRAM_CORES,
                NUM_EXPERTS,
                NUM_BINS_PAD,
                TOPK,
            )
    libshmem_device.barrier_all()

    capacity_ok = tl.load(stats_ptr + 1) <= MAX_RECEIVED_ROUTES
    local_counts_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD

    # D/FC1: dispatch on Vector while Cube consumes signalled receive tiles.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (sub_vec_id() == 0) & capacity_ok:
            _dispatch_count_derived_source_tiles(
                pid,
                NUM_PROGRAM_CORES,
                hidden_states_ptr,
                peer_mem_ptr,
                routing_weights_ptr,
                routing_weight_recv_ptr,
                signal_mem_ptr,
                send_token_indices_ptr,
                send_route_indices_ptr,
                send_bucket_dst_starts_ptr,
                send_bucket_starts_ptr,
                local_counts_ptr,
                signal_epoch,
                HIDDEN,
                stride_hidden_m,
                LOCAL_RANK,
                WORLD_SIZE,
                EXPERTS_PER_RANK,
                EXPERTS_PER_RANK,
                MAX_SOURCE_TILES,
                DISPATCH_BLOCK_M,
            )
    with al.scope(core_mode="cube", disable_auto_sync=True):
        if capacity_ok:
            _triton_grouped_gemm_expert_n_merged_tiles_wait(
                pid,
                NUM_PROGRAM_CORES,
                peer_mem_ptr,
                signal_mem_ptr,
                gate_up_weight_ptr,
                signal_mem_ptr,
                fc1_output_ptr,
                recv_per_expert_ptr,
                recv_expert_offs_ptr,
                recv_counts_re_ptr,
                signal_epoch,
                0,
                2 * FFN,
                HIDDEN,
                stride_hidden_m,
                stride_hidden_k,
                stride_gate_up_e,
                stride_gate_up_n,
                stride_gate_up_k,
                2 * FFN,
                1,
                DISPATCH_BLOCK_M,
                FC1_BLOCK_M,
                FC1_BLOCK_N,
                FC1_BLOCK_K,
                WORLD_SIZE,
                EXPERTS_PER_RANK,
                0,
                EXPERTS_PER_RANK,
                0,
                MAX_SOURCE_TILES,
                False,
                tl.bfloat16,
            )
    libshmem_device.barrier_all()

    # A: apply the weighted activation after every FC1 tile is complete.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (sub_vec_id() == 0) & capacity_ok:
            _weighted_activation_rows(
                fc1_output_ptr,
                routing_weight_recv_ptr,
                weighted_activation_ptr,
                0,
                tl.load(recv_expert_offs_ptr + EXPERTS_PER_RANK),
                FFN,
                situ_beta,
                situ_linear_beta,
                _ACTIVATION_BLOCK_M,
                _ACTIVATION_BLOCK_N,
                ACTIVATION,
                HAS_LINEAR_BETA,
            )
    libshmem_device.barrier_all()

    # FC2: consume the fully materialized weighted activation.
    with al.scope(core_mode="cube", disable_auto_sync=True):
        if capacity_ok:
            _fc2_all_experts(
                pid,
                weighted_activation_ptr,
                down_weight_ptr,
                peer_mem_ptr,
                recv_per_expert_ptr,
                recv_expert_offs_ptr,
                FFN,
                1,
                stride_down_e,
                stride_down_n,
                stride_down_k,
                NUM_PROGRAM_CORES,
                EXPERTS_PER_RANK,
                HIDDEN,
                FFN,
                FC2_BLOCK_M,
                FC2_BLOCK_N,
                FC2_BLOCK_K,
            )
    libshmem_device.barrier_all()

    # C: return expert rows to each source rank's stable-send workspace.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (sub_vec_id() == 0) & capacity_ok:
            _reverse_put_fc2_rows(
                pid,
                combine_buf_ptr,
                peer_mem_ptr,
                pull_tile_rank_ptr,
                pull_tile_src_start_ptr,
                pull_tile_dst_start_ptr,
                pull_tile_row_count_ptr,
                NUM_PROGRAM_CORES,
                WORLD_SIZE,
                EXPERTS_PER_RANK,
                HIDDEN,
            )
    libshmem_device.barrier_all()

    # Final local route restoration and top-k sum.  Overflow produces zeros on
    # every rank, avoiding asymmetric early exits and out-of-bounds RMA.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _reduce_topk_rows(
                pid,
                combine_buf_ptr,
                route_to_send_ptr,
                output_ptr,
                num_routes // TOPK,
                capacity_ok,
                NUM_PROGRAM_CORES,
                HIDDEN,
                TOPK,
                _REDUCE_BLOCK_N,
            )
    libshmem_device.barrier_all()


__all__ = [
    "_kernel_fused_forward",
]
