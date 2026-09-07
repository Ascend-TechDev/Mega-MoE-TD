# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Single-launch home-expert Mega-MoE forward for Ascend.

Routing, dispatch/FC1, weighted activation, FC2, reverse transport, and local
top-k reduction execute in one physical all-core launch.
"""

import triton
import triton.language as tl
import triton_dist.language as dl
import triton.extension.buffer.language as bl
from triton.language.extra.cann.extension import sub_vec_id
import triton.language.extra.cann.extension as al
from triton_dist.language.extra import libshmem_device

from .dispatch_fc1 import _dispatch_count_derived_source_tiles
from .fc2_combine import _fc2_gemm_one_mn_tile

_ROUTE_BLOCK: tl.constexpr = 256
_SCATTER_BLOCK: tl.constexpr = 128
_REDUCE_BLOCK_N: tl.constexpr = 1024


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
def _zero_pipeline_counters(pid, pipeline_signal_ptr,
                            NUM_PROGRAM_CORES: tl.constexpr,
                            NUM_COUNTERS: tl.constexpr):
    for counter_id in range(pid, NUM_COUNTERS, NUM_PROGRAM_CORES):
        tl.store(pipeline_signal_ptr + counter_id * 16, 0)


@triton.jit
def _count_routes_by_core(pid, selected_experts_ptr, core_bucket_cursor_ptr,
                          num_routes, NUM_PROGRAM_CORES: tl.constexpr,
                          NUM_EXPERTS: tl.constexpr,
                          NUM_BINS_PAD: tl.constexpr,
                          BLOCK_SIZE: tl.constexpr):
    """Build one private histogram row per physical AICore.

    Route chunks are contiguous and ordered by ``pid``.  That property lets a
    later prefix over the private rows reproduce stable argsort order without
    atomics in the scatter phase.
    """
    routes_per_core = tl.cdiv(num_routes, NUM_PROGRAM_CORES)
    route_start = pid * routes_per_core
    route_end = tl.minimum(route_start + routes_per_core, num_routes)
    block_offsets = tl.arange(0, BLOCK_SIZE)
    bin_offsets = tl.arange(0, NUM_BINS_PAD)
    counts = tl.zeros((NUM_BINS_PAD, ), dtype=tl.int32)
    for block_start in range(route_start, route_end, BLOCK_SIZE):
        route_ids = block_start + block_offsets
        mask = route_ids < route_end
        experts = tl.load(selected_experts_ptr + route_ids,
                          mask=mask,
                          other=-1)
        valid = mask & (experts >= 0) & (experts < NUM_EXPERTS)
        safe_experts = tl.where(valid, experts, 0)
        block_counts = tl.histogram(safe_experts, NUM_BINS_PAD)
        invalid_count = tl.sum((~valid).to(tl.int32), axis=0)
        block_counts = tl.where(bin_offsets == 0, block_counts - invalid_count,
                                block_counts)
        counts += block_counts
    tl.store(core_bucket_cursor_ptr + pid * NUM_BINS_PAD + bin_offsets, counts)


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
        dst_rank, counts_mem_ptr, send_bucket_starts_ptr,
        send_bucket_dst_starts_ptr, recv_counts_re_ptr, recv_per_expert_ptr,
        recv_expert_offs_ptr, stats_ptr, LOCAL_RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        NUM_BINS_PAD: tl.constexpr):
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
            source_count = tl.load(counts_mem_ptr +
                                   source_rank * NUM_BINS_PAD + bucket)
            target_total += source_count
            if source_rank < LOCAL_RANK:
                source_prefix += source_count
            if dst_rank == LOCAL_RANK:
                tl.store(
                    recv_counts_re_ptr + source_rank * EXPERTS_PER_RANK +
                    local_expert, source_count)
        tl.store(send_bucket_starts_ptr + bucket, send_running)
        tl.store(send_bucket_dst_starts_ptr + bucket,
                 expert_base + source_prefix)
        if dst_rank == LOCAL_RANK:
            tl.store(recv_per_expert_ptr + local_expert, target_total)
            if EXPERTS_PER_RANK > 1:
                tl.store(recv_expert_offs_ptr + local_expert, expert_base)
        send_running += local_count
        expert_base += target_total
    if dst_rank == LOCAL_RANK:
        if EXPERTS_PER_RANK > 1:
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
def _build_pull_destination_starts(
    source_rank,
    counts_mem_ptr,
    pull_tile_dst_start_ptr,
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
            segment_id = local_expert * WORLD_SIZE + source_rank
            tl.store(
                pull_tile_dst_start_ptr + segment_id,
                remote_send_cursor,
            )
        remote_send_cursor += route_count


@triton.jit
def _scatter_stable_routes(pid, selected_experts_ptr, core_bucket_cursor_ptr,
                           send_token_indices_ptr, send_route_indices_ptr,
                           route_to_send_ptr, num_routes,
                           NUM_PROGRAM_CORES: tl.constexpr,
                           NUM_EXPERTS: tl.constexpr,
                           NUM_BINS_PAD: tl.constexpr, TOPK: tl.constexpr,
                           BLOCK_SIZE: tl.constexpr):
    routes_per_core = tl.cdiv(num_routes, NUM_PROGRAM_CORES)
    route_start = pid * routes_per_core
    route_end = tl.minimum(route_start + routes_per_core, num_routes)
    route_offsets = tl.arange(0, BLOCK_SIZE)
    bin_offsets = tl.arange(0, NUM_BINS_PAD)
    cursors = tl.load(core_bucket_cursor_ptr + pid * NUM_BINS_PAD +
                      bin_offsets)
    for block_start in range(route_start, route_end, BLOCK_SIZE):
        route_ids = block_start + route_offsets
        route_mask = route_ids < route_end
        experts = tl.load(selected_experts_ptr + route_ids,
                          mask=route_mask,
                          other=-1)
        valid = route_mask & (experts >= 0) & (experts < NUM_EXPERTS)
        matches = (experts[None, :] == bin_offsets[:, None]) & valid[None, :]
        matches_i32 = matches.to(tl.int32)
        within_block = tl.cumsum(matches_i32, axis=1) - matches_i32
        send_rows = tl.sum((cursors[:, None] + within_block) * matches_i32,
                           axis=0)
        tl.store(send_token_indices_ptr + send_rows,
                 route_ids // TOPK,
                 mask=valid)
        tl.store(send_route_indices_ptr + send_rows, route_ids, mask=valid)
        tl.store(route_to_send_ptr + route_ids, send_rows, mask=valid)
        cursors += tl.sum(matches_i32, axis=1)
    tl.store(core_bucket_cursor_ptr + pid * NUM_BINS_PAD + bin_offsets,
             cursors)


@triton.jit
def _partition_pipeline_fc1_activation_group_ub(
        pid, input_ptr, signal_mem_ptr, pipeline_signal_ptr, weight_ptr,
        routing_weight_ptr, output_ptr, recv_per_expert_ptr,
        recv_expert_offs_ptr, recv_counts_re_ptr, signal_epoch, group_id,
        situ_beta, situ_linear_beta, stride_input_m,
        stride_input_k, stride_weight_e, stride_weight_n, stride_weight_k,
        LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, MAX_SOURCE_TILES: tl.constexpr,
        MAX_PIPELINE_GROUPS: tl.constexpr, FFN: tl.constexpr, K: tl.constexpr,
        DISPATCH_BLOCK_M: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_WINDOWS: tl.constexpr, FC1_CORES: tl.constexpr,
        ACTIVATION: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr,
        FULL_GROUP: tl.constexpr):
    """Pipeline FC1 gate/up tiles through UB directly into activation output."""
    lanes_per_expert: tl.constexpr = FC1_CORES // EXPERTS_PER_RANK
    pair_block_m: tl.constexpr = BLOCK_M // 2
    pair_block_n: tl.constexpr = BLOCK_N // 2
    num_n_tiles: tl.constexpr = (FFN + pair_block_n - 1) // pair_block_n
    max_row_parts: tl.constexpr = (GROUP_WINDOWS * BLOCK_M + pair_block_m -
                                   1) // pair_block_m
    local_tiles: tl.constexpr = (max_row_parts * num_n_tiles +
                                 lanes_per_expert - 1) // lanes_per_expert
    gate_buffer_0 = bl.alloc(tl.bfloat16, [pair_block_m, pair_block_n],
                             al.ascend_address_space.UB)
    up_buffer_0 = bl.alloc(tl.bfloat16, [pair_block_m, pair_block_n],
                           al.ascend_address_space.UB)
    gate_buffer_1 = bl.alloc(tl.bfloat16, [pair_block_m, pair_block_n],
                             al.ascend_address_space.UB)
    up_buffer_1 = bl.alloc(tl.bfloat16, [pair_block_m, pair_block_n],
                           al.ascend_address_space.UB)
    group_ready_token = 0
    with al.scope(core_mode='cube', disable_auto_sync=True):
        cube_expert_id = tl.minimum(pid // lanes_per_expert,
                                    EXPERTS_PER_RANK - 1)
        cube_expert_lane = pid % lanes_per_expert
        cube_expert_off = tl.load(recv_expert_offs_ptr + cube_expert_id)
        cube_group_start = group_id * GROUP_WINDOWS * BLOCK_M
        cube_expert_size = tl.load(recv_per_expert_ptr + cube_expert_id)
        cube_group_size = GROUP_WINDOWS * BLOCK_M if FULL_GROUP else tl.minimum(
            GROUP_WINDOWS *
            BLOCK_M, tl.maximum(cube_expert_size - cube_group_start, 0))
        cube_group_tiles = max_row_parts * num_n_tiles if FULL_GROUP else (
            cube_group_size + pair_block_m - 1) // pair_block_m * num_n_tiles
        cube_active = pid < FC1_CORES if FULL_GROUP else (pid < FC1_CORES) & (
            cube_group_start < cube_expert_size)
        group_ready_slot = (cube_expert_id * MAX_PIPELINE_GROUPS +
                            group_id) * lanes_per_expert
        if cube_active & (cube_expert_lane == 0):
            cube_group_end = cube_group_start + cube_group_size
            source_start = 0
            dispatch_ready_token = 0
            for source_id in range(0, WORLD_SIZE):
                source_size = tl.load(recv_counts_re_ptr +
                                      source_id * EXPERTS_PER_RANK +
                                      cube_expert_id)
                source_end = source_start + source_size
                overlap_start = tl.maximum(cube_group_start, source_start)
                overlap_end = tl.minimum(cube_group_end, source_end)
                if overlap_start < overlap_end:
                    first_source_tile = (overlap_start -
                                         source_start) // DISPATCH_BLOCK_M
                    last_source_tile = (overlap_end - source_start -
                                        1) // DISPATCH_BLOCK_M
                    signal_slot = (
                        source_id * EXPERTS_PER_RANK +
                        cube_expert_id) * MAX_SOURCE_TILES + first_source_tile
                    dispatch_ready_token += dl.wait(
                        signal_mem_ptr + signal_slot * 16,
                        last_source_tile - first_source_tile + 1,
                        'gpu',
                        'acquire',
                        waitValue=signal_epoch)
                source_start = source_end
            ready_pipeline_signal_ptr = dl.consume_token(
                pipeline_signal_ptr, dispatch_ready_token)
            dl.notify(ready_pipeline_signal_ptr + group_ready_slot * 16,
                      LOCAL_RANK,
                      signal=signal_epoch,
                      sig_op='set',
                      comm_scope='intra_node')
        if cube_active:
            group_ready_token += dl.wait(pipeline_signal_ptr +
                                         group_ready_slot * 16,
                                         1,
                                         'gpu',
                                         'acquire',
                                         waitValue=signal_epoch)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        vector_expert_id = tl.minimum(pid // lanes_per_expert,
                                      EXPERTS_PER_RANK - 1)
        vector_expert_lane = pid % lanes_per_expert
        vector_expert_off = tl.load(recv_expert_offs_ptr + vector_expert_id)
        vector_group_start = group_id * GROUP_WINDOWS * BLOCK_M
        vector_expert_size = tl.load(recv_per_expert_ptr + vector_expert_id)
        vector_group_size = GROUP_WINDOWS * BLOCK_M if FULL_GROUP else tl.minimum(
            GROUP_WINDOWS *
            BLOCK_M, tl.maximum(vector_expert_size - vector_group_start, 0))
        vector_group_tiles = max_row_parts * num_n_tiles if FULL_GROUP else (
            vector_group_size + pair_block_m - 1) // pair_block_m * num_n_tiles
        vector_active = pid < FC1_CORES if FULL_GROUP else (
            pid < FC1_CORES) & (vector_group_start < vector_expert_size)
    for pipeline_step in range(0, local_tiles + 1):
        with al.scope(core_mode='cube', disable_auto_sync=True):
            row_part = pipeline_step % max_row_parts
            n_tile = cube_expert_lane + pipeline_step // max_row_parts * lanes_per_expert
            cube_group_row_parts = max_row_parts if FULL_GROUP else (
                cube_group_size + pair_block_m - 1) // pair_block_m
            cube_tile_valid = cube_active & (pipeline_step < local_tiles) & (
                row_part < cube_group_row_parts) & (n_tile < num_n_tiles)
            if cube_tile_valid:
                input_group_offset = (cube_expert_off.to(
                    tl.int64) + cube_group_start.to(tl.int64)) * stride_input_m
                ready_input_ptr = dl.consume_token(
                    input_ptr + input_group_offset, group_ready_token)
                if pipeline_step >= 2:
                    if pipeline_step % 2 == 0:
                        al.sync_block_wait('vector', 'cube', 10)
                    else:
                        al.sync_block_wait('vector', 'cube', 11)
                local_row_start = row_part * pair_block_m
                row_count = pair_block_m if FULL_GROUP else tl.minimum(
                    pair_block_m, cube_group_size - local_row_start)
                offs_m = tl.arange(0, pair_block_m)
                offs_n = tl.arange(0, pair_block_n)
                offs_k = tl.arange(0, BLOCK_K)
                rows = local_row_start + offs_m
                gate_cols = n_tile * pair_block_n + offs_n
                up_cols = FFN + gate_cols
                mask_m = offs_m < row_count
                mask_n = gate_cols < FFN
                weight_base = weight_ptr + cube_expert_id.to(
                    tl.int64) * stride_weight_e
                ready_weight_base = weight_base
                gate_acc = tl.zeros((pair_block_m, pair_block_n),
                                    dtype=tl.float32)
                up_acc = tl.zeros((pair_block_m, pair_block_n),
                                  dtype=tl.float32)
                for k_start in range(0, K, BLOCK_K):
                    red = k_start + offs_k
                    if K % BLOCK_K == 0:
                        if FULL_GROUP & (FFN % pair_block_n == 0):
                            a = tl.load(ready_input_ptr +
                                        rows[:, None] * stride_input_m +
                                        red[None, :] * stride_input_k)
                            gate_weight = tl.load(
                                ready_weight_base +
                                gate_cols[None, :] * stride_weight_n +
                                red[:, None] * stride_weight_k)
                            up_weight = tl.load(ready_weight_base +
                                                up_cols[None, :] *
                                                stride_weight_n +
                                                red[:, None] * stride_weight_k)
                        else:
                            a = tl.load(ready_input_ptr +
                                        rows[:, None] * stride_input_m +
                                        red[None, :] * stride_input_k,
                                        mask=mask_m[:, None],
                                        other=0.0)
                            gate_weight = tl.load(
                                ready_weight_base +
                                gate_cols[None, :] * stride_weight_n +
                                red[:, None] * stride_weight_k,
                                mask=mask_n[None, :],
                                other=0.0)
                            up_weight = tl.load(
                                ready_weight_base +
                                up_cols[None, :] * stride_weight_n +
                                red[:, None] * stride_weight_k,
                                mask=mask_n[None, :],
                                other=0.0)
                    else:
                        mask_k = red < K
                        a = tl.load(ready_input_ptr +
                                    rows[:, None] * stride_input_m +
                                    red[None, :] * stride_input_k,
                                    mask=mask_m[:, None] & mask_k[None, :],
                                    other=0.0)
                        gate_weight = tl.load(
                            ready_weight_base +
                            gate_cols[None, :] * stride_weight_n +
                            red[:, None] * stride_weight_k,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0)
                        up_weight = tl.load(
                            ready_weight_base +
                            up_cols[None, :] * stride_weight_n +
                            red[:, None] * stride_weight_k,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0)
                    gate_acc += tl.dot(a, gate_weight)
                    up_acc += tl.dot(a, up_weight)
                if pipeline_step % 2 == 0:
                    al.fixpipe(gate_acc, gate_buffer_0)
                    al.fixpipe(up_acc, up_buffer_0)
                    al.sync_block_set('cube', 'vector', 8)
                else:
                    al.fixpipe(gate_acc, gate_buffer_1)
                    al.fixpipe(up_acc, up_buffer_1)
                    al.sync_block_set('cube', 'vector', 9)
            if True & (pipeline_step == local_tiles):
                if cube_active & (cube_expert_lane < num_n_tiles):
                    lane_n_tiles = (num_n_tiles - cube_expert_lane +
                                    lanes_per_expert - 1) // lanes_per_expert
                    lane_tiles = cube_group_row_parts * lane_n_tiles
                    if (lane_tiles - 1) % 2 == 0:
                        al.sync_block_wait('vector', 'cube', 10)
                    else:
                        al.sync_block_wait('vector', 'cube', 11)
        with al.scope(core_mode='vector', disable_auto_sync=True):
            previous_step = pipeline_step - 1
            row_part = previous_step % max_row_parts
            n_tile = vector_expert_lane + previous_step // max_row_parts * lanes_per_expert
            vector_group_row_parts = max_row_parts if FULL_GROUP else (
                vector_group_size + pair_block_m - 1) // pair_block_m
            previous_valid = vector_active & (pipeline_step > 0) & (
                row_part < vector_group_row_parts) & (n_tile < num_n_tiles)
            if previous_valid & (sub_vec_id() == 0):
                local_row_start = row_part * pair_block_m
                row_count = pair_block_m if FULL_GROUP else tl.minimum(
                    pair_block_m, vector_group_size - local_row_start)
                output_group_row = vector_expert_off.to(
                    tl.int64) + vector_group_start.to(tl.int64)
                output_group_ptr = output_ptr + output_group_row * FFN
                routing_group_ptr = routing_weight_ptr + output_group_row
                col_start = n_tile * pair_block_n
                if previous_step % 2 == 0:
                    al.sync_block_wait('cube', 'vector', 8)
                else:
                    al.sync_block_wait('cube', 'vector', 9)
                for row_chunk in range(0, pair_block_m, 8):
                    gate_view_0 = gate_buffer_0.subview([row_chunk, 0],
                                                        [8, pair_block_n],
                                                        [1, 1])
                    up_view_0 = up_buffer_0.subview([row_chunk, 0],
                                                    [8, pair_block_n], [1, 1])
                    gate_view_1 = gate_buffer_1.subview([row_chunk, 0],
                                                        [8, pair_block_n],
                                                        [1, 1])
                    up_view_1 = up_buffer_1.subview([row_chunk, 0],
                                                    [8, pair_block_n], [1, 1])
                    if previous_step % 2 == 0:
                        gate = bl.to_tensor(gate_view_0,
                                            writable=False).to(tl.float32)
                        up = bl.to_tensor(up_view_0,
                                          writable=False).to(tl.float32)
                    else:
                        gate = bl.to_tensor(gate_view_1,
                                            writable=False).to(tl.float32)
                        up = bl.to_tensor(up_view_1,
                                          writable=False).to(tl.float32)
                    if ACTIVATION == 0:
                        activated = gate * tl.sigmoid(gate) * up
                    else:
                        situ_a = situ_beta * tl.math.tanh(
                            gate / situ_beta) * tl.sigmoid(gate)
                        if HAS_LINEAR_BETA:
                            up = situ_linear_beta * tl.math.tanh(
                                up / situ_linear_beta)
                        activated = situ_a * up
                    local_rows = row_chunk + tl.arange(0, 8)
                    cols = col_start + tl.arange(0, pair_block_n)
                    mask_m = local_rows < row_count
                    mask_n = cols < FFN
                    if FULL_GROUP:
                        routing_weight = tl.load(routing_group_ptr +
                                                 local_row_start +
                                                 local_rows).to(tl.float32)
                    else:
                        routing_weight = tl.load(routing_group_ptr +
                                                 local_row_start + local_rows,
                                                 mask=mask_m,
                                                 other=0.0).to(tl.float32)
                    activated *= routing_weight[:, None]
                    if FULL_GROUP & (FFN % pair_block_n == 0):
                        tl.store(
                            output_group_ptr +
                            (local_row_start + local_rows)[:, None] * FFN +
                            cols[None, :], activated)
                    else:
                        tl.store(
                            output_group_ptr +
                            (local_row_start + local_rows)[:, None] * FFN +
                            cols[None, :],
                            activated,
                            mask=mask_m[:, None] & mask_n[None, :])
                if previous_step % 2 == 0:
                    al.sync_block_set('vector', 'cube', 10)
                else:
                    al.sync_block_set('vector', 'cube', 11)
            if pipeline_step == local_tiles:
                if vector_active & (sub_vec_id() == 0):
                    fc1_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * lanes_per_expert
                    signal_index = vector_expert_id * MAX_PIPELINE_GROUPS + group_id
                    activation_slot = fc1_signal_slots + signal_index * lanes_per_expert + vector_expert_lane
                    dl.notify(pipeline_signal_ptr + activation_slot * 16,
                              LOCAL_RANK,
                              signal=signal_epoch,
                              sig_op='set',
                              comm_scope='intra_node')


@triton.jit
def _partitioned_pipeline_fc2_group(
        pid, input_ptr, weight_ptr, output_ptr, recv_per_expert_ptr,
        recv_expert_offs_ptr, pipeline_signal_ptr, signal_epoch, group_id,
        stride_input_m, stride_input_k, stride_weight_e, stride_weight_n,
        stride_weight_k, stride_output_m, NUM_PROGRAM_CORES: tl.constexpr,
        LOCAL_RANK: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        MAX_PIPELINE_GROUPS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_WINDOWS: tl.constexpr, FC1_CORES: tl.constexpr,
        FC2_FIRST_CORE: tl.constexpr, ACTIVATION_WORKERS: tl.constexpr,
        WAIT_ACTIVATION: tl.constexpr, SET_EVENT: tl.constexpr):
    fc2_cores: tl.constexpr = NUM_PROGRAM_CORES - FC2_FIRST_CORE
    num_n_tiles: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    num_group_tiles: tl.constexpr = GROUP_WINDOWS * num_n_tiles
    num_lane_tiles: tl.constexpr = (num_group_tiles + fc2_cores -
                                    1) // fc2_cores
    fc1_lanes: tl.constexpr = FC1_CORES // EXPERTS_PER_RANK
    activation_lanes: tl.constexpr = ACTIVATION_WORKERS // EXPERTS_PER_RANK
    fc1_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * fc1_lanes
    activation_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * activation_lanes
    fc2_start_base: tl.constexpr = fc1_signal_slots + activation_signal_slots
    fc2_completion_base: tl.constexpr = fc2_start_base + MAX_PIPELINE_GROUPS * fc2_cores
    if pid >= FC2_FIRST_CORE:
        fc2_pid = pid - FC2_FIRST_CORE
        group_start = group_id * GROUP_WINDOWS * BLOCK_M
        for expert_id in range(0, EXPERTS_PER_RANK):
            expert_size = tl.load(recv_per_expert_ptr + expert_id)
            if group_start < expert_size:
                ready_input_ptr = input_ptr
                if WAIT_ACTIVATION:
                    signal_index = expert_id * MAX_PIPELINE_GROUPS + group_id
                    activation_ready = dl.wait(
                        pipeline_signal_ptr +
                        (fc1_signal_slots + signal_index * activation_lanes) *
                        16,
                        activation_lanes,
                        'gpu',
                        'acquire',
                        waitValue=signal_epoch)
                    ready_input_ptr = dl.consume_token(input_ptr,
                                                       activation_ready)
                expert_off = tl.load(recv_expert_offs_ptr + expert_id)
                for lane_tile in tl.static_range(0, num_lane_tiles):
                    group_tile = fc2_pid + lane_tile * fc2_cores
                    if group_tile < num_group_tiles:
                        group_window = group_tile // num_n_tiles
                        n_tile = group_tile % num_n_tiles
                        window_start = group_start + group_window * BLOCK_M
                        if window_start < expert_size:
                            row_count = tl.minimum(BLOCK_M,
                                                   expert_size - window_start)
                            _fc2_gemm_one_mn_tile(
                                ready_input_ptr, weight_ptr, output_ptr,
                                expert_id, expert_off + window_start,
                                row_count, n_tile, N, K, stride_input_m,
                                stride_input_k, stride_weight_e,
                                stride_weight_n, stride_weight_k,
                                stride_output_m, 1, BLOCK_M, BLOCK_N, BLOCK_K,
                                0)
        if SET_EVENT:
            libshmem_device.fence()
            completion_slot = fc2_completion_base + group_id * fc2_cores + fc2_pid
            dl.notify(pipeline_signal_ptr + completion_slot * 16,
                      LOCAL_RANK,
                      signal=signal_epoch,
                      sig_op='set',
                      comm_scope='intra_node')


@triton.jit
def _partition_pipeline_reverse_group_parallel(
        worker_id, combine_buf_ptr, fc2_output_ptr, recv_per_expert_ptr,
        recv_expert_offs_ptr, recv_counts_re_ptr, pull_tile_dst_start_ptr,
        pipeline_signal_ptr, signal_epoch, group_id,
        NUM_VECTOR_WORKERS: tl.constexpr, FC2_CORES: tl.constexpr,
        FC1_WORKERS: tl.constexpr,
        ACTIVATION_WORKERS: tl.constexpr, LOCAL_RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        MAX_PIPELINE_GROUPS: tl.constexpr, HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr, GROUP_WINDOWS: tl.constexpr):
    """Return one group with a fixed set of Vector workers per source rank."""
    workers_per_source: tl.constexpr = NUM_VECTOR_WORKERS // WORLD_SIZE
    num_work_items: tl.constexpr = workers_per_source * WORLD_SIZE
    fc1_lanes: tl.constexpr = FC1_WORKERS // EXPERTS_PER_RANK
    activation_lanes: tl.constexpr = ACTIVATION_WORKERS // EXPERTS_PER_RANK
    fc1_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * fc1_lanes
    activation_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * activation_lanes
    fc2_signal_base: tl.constexpr = fc1_signal_slots + activation_signal_slots + MAX_PIPELINE_GROUPS * FC2_CORES
    reverse_signal_base: tl.constexpr = fc2_signal_base + MAX_PIPELINE_GROUPS * FC2_CORES
    if worker_id < num_work_items:
        source_rank = (worker_id % WORLD_SIZE).to(tl.int32)
        source_lane = worker_id // WORLD_SIZE
        group_start = group_id * GROUP_WINDOWS * BLOCK_M
        fc2_ready = dl.wait(
            pipeline_signal_ptr
            + (fc2_signal_base + group_id * FC2_CORES) * 16,
            FC2_CORES,
            'gpu',
            'acquire',
            waitValue=signal_epoch,
        )
        ready_fc2_ptr = dl.consume_token(fc2_output_ptr, fc2_ready)
        for expert_id in range(0, EXPERTS_PER_RANK):
            expert_size = tl.load(recv_per_expert_ptr + expert_id)
            if group_start < expert_size:
                source_size = tl.load(
                    recv_counts_re_ptr
                    + source_rank * EXPERTS_PER_RANK
                    + expert_id
                )
                source_start = 0
                for prior_rank in range(0, WORLD_SIZE):
                    prior_size = tl.load(
                        recv_counts_re_ptr
                        + prior_rank * EXPERTS_PER_RANK
                        + expert_id
                    )
                    source_start += tl.where(
                        prior_rank < source_rank, prior_size, 0
                    )
                source_end = source_start + source_size
                group_end = tl.minimum(
                    group_start + GROUP_WINDOWS * BLOCK_M, expert_size
                )
                overlap_start = tl.maximum(group_start, source_start)
                overlap_end = tl.minimum(group_end, source_end)
                if overlap_start < overlap_end:
                    overlap_rows = overlap_end - overlap_start
                    part_begin = (
                        overlap_rows * source_lane // workers_per_source
                    )
                    part_end = (
                        overlap_rows * (source_lane + 1) // workers_per_source
                    )
                    part_rows = part_end - part_begin
                    if part_rows > 0:
                        expert_off = tl.load(
                            recv_expert_offs_ptr + expert_id
                        ).to(tl.int64)
                        segment_id = expert_id * WORLD_SIZE + source_rank
                        destination_start = tl.load(
                            pull_tile_dst_start_ptr + segment_id
                        ).to(tl.int64)
                        source_row = (
                            expert_off + overlap_start + part_begin
                        ).to(tl.int64)
                        destination_row = (
                            destination_start
                            + overlap_start
                            - source_start
                            + part_begin
                        ).to(tl.int64)
                        libshmem_device.putmem(
                            combine_buf_ptr + destination_row * HIDDEN,
                            ready_fc2_ptr + source_row * HIDDEN,
                            part_rows * HIDDEN * 2,
                            source_rank,
                        )
        libshmem_device.fence()
        reverse_signal_slot = reverse_signal_base + (
            group_id * WORLD_SIZE + LOCAL_RANK
        ) * workers_per_source + source_lane
        dl.notify(pipeline_signal_ptr + reverse_signal_slot * 16,
                  source_rank,
                  signal=signal_epoch,
                  sig_op='set',
                  comm_scope='intra_node')


@triton.jit
def _wait_partition_pipeline_reverse_groups(
        counts_mem_ptr, pipeline_signal_ptr, signal_epoch,
        NUM_VECTOR_WORKERS: tl.constexpr, FC2_CORES: tl.constexpr,
        FC1_WORKERS: tl.constexpr,
        ACTIVATION_WORKERS: tl.constexpr, WORLD_SIZE: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, NUM_BINS_PAD: tl.constexpr,
        MAX_PIPELINE_GROUPS: tl.constexpr, BLOCK_M: tl.constexpr,
        GROUP_WINDOWS: tl.constexpr):
    fc1_lanes: tl.constexpr = FC1_WORKERS // EXPERTS_PER_RANK
    activation_lanes: tl.constexpr = ACTIVATION_WORKERS // EXPERTS_PER_RANK
    fc1_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * fc1_lanes
    activation_signal_slots: tl.constexpr = EXPERTS_PER_RANK * MAX_PIPELINE_GROUPS * activation_lanes
    fc2_signal_base: tl.constexpr = fc1_signal_slots + activation_signal_slots + MAX_PIPELINE_GROUPS * FC2_CORES
    reverse_signal_base: tl.constexpr = fc2_signal_base + MAX_PIPELINE_GROUPS * FC2_CORES
    signals_per_source: tl.constexpr = NUM_VECTOR_WORKERS // WORLD_SIZE
    reverse_ready = 0
    for destination_rank in range(0, WORLD_SIZE):
        destination_groups = 0
        for local_expert in range(0, EXPERTS_PER_RANK):
            global_expert = destination_rank * EXPERTS_PER_RANK + local_expert
            expert_routes = 0
            for source_rank in range(0, WORLD_SIZE):
                expert_routes += tl.load(counts_mem_ptr +
                                         source_rank * NUM_BINS_PAD +
                                         global_expert)
            destination_groups = tl.maximum(
                destination_groups,
                tl.cdiv(expert_routes, GROUP_WINDOWS * BLOCK_M))
        for group_id in range(0, destination_groups):
            reverse_signal_slot = reverse_signal_base + (
                group_id * WORLD_SIZE + destination_rank) * signals_per_source
            reverse_ready += dl.wait(pipeline_signal_ptr +
                                     reverse_signal_slot * 16,
                                     signals_per_source,
                                     'gpu',
                                     'acquire',
                                     waitValue=signal_epoch)
    return reverse_ready


@triton.jit
def _reduce_topk_rows(pid, combine_buf_ptr, route_to_send_ptr, output_ptr,
                      num_tokens, capacity_ok, NUM_PROGRAM_CORES: tl.constexpr,
                      HIDDEN: tl.constexpr, TOPK: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    cols_in_block = tl.arange(0, BLOCK_N)
    for token_id in range(pid, num_tokens, NUM_PROGRAM_CORES):
        route_base = token_id * TOPK
        for col_start in range(0, HIDDEN, BLOCK_N):
            cols = col_start + cols_in_block
            mask_n = cols < HIDDEN
            acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)
            if capacity_ok:
                for topk_slot in tl.static_range(0, TOPK):
                    send_row = tl.load(route_to_send_ptr + route_base +
                                       topk_slot)
                    valid = send_row >= 0
                    safe_row = tl.where(valid, send_row, 0).to(tl.int64)
                    value = tl.load(combine_buf_ptr + safe_row * HIDDEN + cols,
                                    mask=mask_n & valid,
                                    other=0.0).to(tl.float32)
                    acc += value
            tl.store(output_ptr + token_id.to(tl.int64) * HIDDEN + cols,
                     acc.to(tl.bfloat16),
                     mask=mask_n)


@triton.jit(do_not_specialize=['num_routes', 'signal_epoch'])
def _kernel_fused_forward(
        hidden_states_ptr, selected_experts_ptr, routing_weights_ptr,
        gate_up_weight_ptr, down_weight_ptr, peer_mem_ptr,
        routing_weight_recv_ptr, signal_mem_ptr, pipeline_signal_ptr,
        combine_buf_ptr, fc2_output_ptr, weighted_activation_ptr, output_ptr,
        counts_mem_ptr,
        send_bucket_starts_ptr, send_bucket_dst_starts_ptr, recv_counts_re_ptr,
        recv_per_expert_ptr, recv_expert_offs_ptr, stats_ptr,
        core_bucket_cursor_ptr, send_token_indices_ptr, send_route_indices_ptr,
        route_to_send_ptr, pull_tile_dst_start_ptr, num_routes, signal_epoch,
        situ_beta, situ_linear_beta,
        stride_hidden_m: tl.constexpr, stride_hidden_k: tl.constexpr,
        stride_gate_up_e: tl.constexpr, stride_gate_up_n: tl.constexpr,
        stride_gate_up_k: tl.constexpr, stride_down_e: tl.constexpr,
        stride_down_n: tl.constexpr, stride_down_k: tl.constexpr,
        NUM_PROGRAM_CORES: tl.constexpr, LOCAL_RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, TOPK: tl.constexpr,
        HIDDEN: tl.constexpr, FFN: tl.constexpr,
        MAX_RECEIVED_ROUTES: tl.constexpr, NUM_BINS_PAD: tl.constexpr,
        MAX_SOURCE_TILES: tl.constexpr, MAX_PIPELINE_GROUPS: tl.constexpr,
        DISPATCH_BLOCK_M: tl.constexpr, FC1_BLOCK_M: tl.constexpr,
        FC1_BLOCK_N: tl.constexpr, FC1_BLOCK_K: tl.constexpr,
        FC2_BLOCK_M: tl.constexpr, FC2_BLOCK_N: tl.constexpr,
        FC2_BLOCK_K: tl.constexpr, ACTIVATION: tl.constexpr,
        HAS_LINEAR_BETA: tl.constexpr,
        PIPELINE_GROUP_WINDOWS: tl.constexpr):
    """Production routing-to-reduction pipeline for the single-kernel path."""
    pid = tl.program_id(axis=0)
    pipeline_fc1_cores: tl.constexpr = (
        NUM_PROGRAM_CORES // EXPERTS_PER_RANK
    ) * EXPERTS_PER_RANK
    pipeline_fc2_first_core: tl.constexpr = 0
    pipeline_fc2_cores: tl.constexpr = NUM_PROGRAM_CORES
    pipeline_group_tile_m: tl.constexpr = (
        FC1_BLOCK_M if FC1_BLOCK_M >= FC2_BLOCK_M else FC2_BLOCK_M
    )
    pipeline_group_rows: tl.constexpr = (
        PIPELINE_GROUP_WINDOWS * pipeline_group_tile_m
    )
    pipeline_fc1_group_windows: tl.constexpr = (
        pipeline_group_rows // FC1_BLOCK_M
    )
    pipeline_fc2_group_windows: tl.constexpr = (
        pipeline_group_rows // FC2_BLOCK_M
    )
    reverse_workers_per_source: tl.constexpr = (
        2 * NUM_PROGRAM_CORES // WORLD_SIZE
    )
    pipeline_counter_count: tl.constexpr = MAX_PIPELINE_GROUPS * (
        2 * pipeline_fc1_cores
        + 2 * pipeline_fc2_cores
        + WORLD_SIZE * reverse_workers_per_source
    )
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 0:
            _zero_route_workspaces(pid, core_bucket_cursor_ptr,
                                   route_to_send_ptr, num_routes,
                                   NUM_PROGRAM_CORES, NUM_BINS_PAD,
                                   _ROUTE_BLOCK)
            _zero_pipeline_counters(
                pid, pipeline_signal_ptr, NUM_PROGRAM_CORES,
                pipeline_counter_count)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 0:
            _count_routes_by_core(pid, selected_experts_ptr,
                                  core_bucket_cursor_ptr, num_routes,
                                  NUM_PROGRAM_CORES, NUM_EXPERTS, NUM_BINS_PAD,
                                  _ROUTE_BLOCK)
    libshmem_device.barrier_all()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid == 0):
            _publish_count_row(core_bucket_cursor_ptr, counts_mem_ptr,
                               LOCAL_RANK, WORLD_SIZE, NUM_PROGRAM_CORES,
                               NUM_EXPERTS, NUM_BINS_PAD)
    libshmem_device.barrier_all()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid < WORLD_SIZE):
            _build_destination_metadata(
                pid, counts_mem_ptr, send_bucket_starts_ptr,
                send_bucket_dst_starts_ptr, recv_counts_re_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, stats_ptr,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK, NUM_BINS_PAD)
    libshmem_device.barrier_all()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 0:
            _convert_counts_to_stable_cursors(pid, core_bucket_cursor_ptr,
                                              send_bucket_starts_ptr,
                                              NUM_PROGRAM_CORES, NUM_EXPERTS,
                                              NUM_BINS_PAD)
            if pid < WORLD_SIZE:
                _build_pull_destination_starts(
                    pid, counts_mem_ptr, pull_tile_dst_start_ptr, LOCAL_RANK,
                    WORLD_SIZE, EXPERTS_PER_RANK, NUM_EXPERTS, NUM_BINS_PAD)
            if pid == 0:
                max_received = 0
                for dst_rank in range(0, WORLD_SIZE):
                    max_received = tl.maximum(
                        max_received, tl.load(stats_ptr + 2 + dst_rank))
                tl.store(stats_ptr + 1, max_received)
    libshmem_device.barrier_all()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 0:
            _scatter_stable_routes(pid, selected_experts_ptr,
                                   core_bucket_cursor_ptr,
                                   send_token_indices_ptr,
                                   send_route_indices_ptr, route_to_send_ptr,
                                   num_routes, NUM_PROGRAM_CORES, NUM_EXPERTS,
                                   NUM_BINS_PAD, TOPK, _SCATTER_BLOCK)
    libshmem_device.barrier_all()
    capacity_ok = tl.load(stats_ptr + 1) <= MAX_RECEIVED_ROUTES
    local_counts_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if capacity_ok:
            _dispatch_count_derived_source_tiles(
                pid * 2 + sub_vec_id(), 2 * NUM_PROGRAM_CORES,
                hidden_states_ptr, peer_mem_ptr, routing_weights_ptr,
                routing_weight_recv_ptr, signal_mem_ptr,
                send_token_indices_ptr, send_route_indices_ptr,
                send_bucket_dst_starts_ptr, send_bucket_starts_ptr,
                local_counts_ptr, signal_epoch, HIDDEN, stride_hidden_m,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK, EXPERTS_PER_RANK,
                MAX_SOURCE_TILES, DISPATCH_BLOCK_M, True)
    pipeline_input_ptr = peer_mem_ptr
    max_pipeline_groups = 0
    common_full_groups = tl.load(recv_per_expert_ptr) // pipeline_group_rows
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        max_pipeline_groups = tl.maximum(max_pipeline_groups,
                                         tl.cdiv(expert_size,
                                                 pipeline_group_rows))
        common_full_groups = tl.minimum(common_full_groups,
                                        expert_size // pipeline_group_rows)
    max_pipeline_groups = tl.where(capacity_ok, max_pipeline_groups, 0)
    common_full_groups = tl.where(capacity_ok, common_full_groups, 0)
    sequential_full_groups = common_full_groups
    for full_group in range(0, sequential_full_groups):
        _partition_pipeline_fc1_activation_group_ub(
            pid, pipeline_input_ptr, signal_mem_ptr, pipeline_signal_ptr,
            gate_up_weight_ptr, routing_weight_recv_ptr,
            weighted_activation_ptr, recv_per_expert_ptr, recv_expert_offs_ptr,
            recv_counts_re_ptr, signal_epoch, full_group, situ_beta,
            situ_linear_beta, stride_hidden_m, stride_hidden_k,
            stride_gate_up_e, stride_gate_up_n, stride_gate_up_k, LOCAL_RANK,
            WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
            MAX_PIPELINE_GROUPS, FFN, HIDDEN, DISPATCH_BLOCK_M, FC1_BLOCK_M,
            FC1_BLOCK_N, FC1_BLOCK_K, pipeline_fc1_group_windows,
            pipeline_fc1_cores, ACTIVATION, HAS_LINEAR_BETA, True)
    tail_groups = max_pipeline_groups - common_full_groups
    for tail_index in range(0, tail_groups):
        tail_group = common_full_groups + tail_index
        _partition_pipeline_fc1_activation_group_ub(
            pid, pipeline_input_ptr, signal_mem_ptr, pipeline_signal_ptr,
            gate_up_weight_ptr, routing_weight_recv_ptr,
            weighted_activation_ptr, recv_per_expert_ptr, recv_expert_offs_ptr,
            recv_counts_re_ptr, signal_epoch, tail_group, situ_beta,
            situ_linear_beta, stride_hidden_m, stride_hidden_k,
            stride_gate_up_e, stride_gate_up_n, stride_gate_up_k, LOCAL_RANK,
            WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
            MAX_PIPELINE_GROUPS, FFN, HIDDEN, DISPATCH_BLOCK_M, FC1_BLOCK_M,
            FC1_BLOCK_N, FC1_BLOCK_K, pipeline_fc1_group_windows,
            pipeline_fc1_cores, ACTIVATION, HAS_LINEAR_BETA, False)
    for fc2_group in range(0, max_pipeline_groups):
        with al.scope(core_mode='cube', disable_auto_sync=True):
            _partitioned_pipeline_fc2_group(
                pid, weighted_activation_ptr, down_weight_ptr, fc2_output_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, pipeline_signal_ptr,
                signal_epoch, fc2_group, FFN, 1, stride_down_e, stride_down_n,
                stride_down_k, HIDDEN, NUM_PROGRAM_CORES, LOCAL_RANK,
                EXPERTS_PER_RANK, MAX_PIPELINE_GROUPS, HIDDEN, FFN,
                FC2_BLOCK_M, FC2_BLOCK_N, FC2_BLOCK_K,
                pipeline_fc2_group_windows,
                pipeline_fc1_cores, pipeline_fc2_first_core,
                pipeline_fc1_cores, True, True)
        with al.scope(core_mode='vector', disable_auto_sync=True):
            if capacity_ok & (fc2_group > 0):
                _partition_pipeline_reverse_group_parallel(
                    pid * 2 + sub_vec_id(), combine_buf_ptr, fc2_output_ptr,
                    recv_per_expert_ptr, recv_expert_offs_ptr,
                    recv_counts_re_ptr, pull_tile_dst_start_ptr,
                    pipeline_signal_ptr, signal_epoch, fc2_group - 1,
                    2 * NUM_PROGRAM_CORES, pipeline_fc2_cores,
                    pipeline_fc1_cores, pipeline_fc1_cores,
                    LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK,
                    MAX_PIPELINE_GROUPS, HIDDEN, FC2_BLOCK_M,
                    pipeline_fc2_group_windows)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if capacity_ok & (max_pipeline_groups > 0):
            _partition_pipeline_reverse_group_parallel(
                pid * 2 + sub_vec_id(), combine_buf_ptr, fc2_output_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
                pull_tile_dst_start_ptr, pipeline_signal_ptr, signal_epoch,
                max_pipeline_groups - 1, 2 * NUM_PROGRAM_CORES,
                pipeline_fc2_cores, pipeline_fc1_cores, pipeline_fc1_cores,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK,
                MAX_PIPELINE_GROUPS, HIDDEN, FC2_BLOCK_M,
                pipeline_fc2_group_windows)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        reverse_ready = 0
        if capacity_ok:
            reverse_ready += _wait_partition_pipeline_reverse_groups(
                counts_mem_ptr, pipeline_signal_ptr, signal_epoch,
                2 * NUM_PROGRAM_CORES, pipeline_fc2_cores,
                pipeline_fc1_cores, pipeline_fc1_cores, WORLD_SIZE,
                EXPERTS_PER_RANK, NUM_BINS_PAD, MAX_PIPELINE_GROUPS,
                FC2_BLOCK_M, pipeline_fc2_group_windows)
        ready_combine_ptr = dl.consume_token(combine_buf_ptr, reverse_ready)
        _reduce_topk_rows(pid * 2 + sub_vec_id(), ready_combine_ptr,
                          route_to_send_ptr, output_ptr, num_routes // TOPK,
                          capacity_ok, 2 * NUM_PROGRAM_CORES, HIDDEN, TOPK,
                          _REDUCE_BLOCK_N)
