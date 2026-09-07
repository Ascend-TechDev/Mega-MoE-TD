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

from .dispatch_fc1 import _dispatch_one_source_tile_task
from .fc2_combine import _fc2_gemm_one_mn_tile

_ROUTE_BLOCK: tl.constexpr = 256
_SCATTER_BLOCK: tl.constexpr = 128
_REDUCE_BLOCK_N: tl.constexpr = 4096


@triton.jit
def _mixed_forward_barrier():
    # The 9.1 mixed SHMEM barrier can expose incomplete count publication.
    # Keep cross-rank polling Vector-only, with explicit Cube/Vector joins.
    al.sync_block_all('all', 15)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        libshmem_device.barrier_all_vec()
    al.sync_block_all('all', 15)


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
    # The first rank barrier separates per-call ADD resets from publication.
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

    # Scalar stores must reach GM before MTE reads the published count row.
    libshmem_device.fence()
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
    # Bound the cumsum's bin-by-route matrix for full-expert configurations.
    bin_block: tl.constexpr = 32 if NUM_BINS_PAD > 32 else NUM_BINS_PAD
    for bin_start in range(0, NUM_BINS_PAD, bin_block):
        bin_offsets = bin_start + tl.arange(0, bin_block)
        cursors = tl.load(core_bucket_cursor_ptr + pid * NUM_BINS_PAD +
                          bin_offsets)
        for block_start in range(route_start, route_end, BLOCK_SIZE):
            route_ids = block_start + route_offsets
            route_mask = route_ids < route_end
            experts = tl.load(selected_experts_ptr + route_ids,
                              mask=route_mask,
                              other=-1)
            valid = route_mask & (experts >= 0) & (experts < NUM_EXPERTS)
            if NUM_BINS_PAD > bin_block:
                valid &= (experts >= bin_start) & (experts < bin_start + bin_block)
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
def _wait_dispatch_row_range(
        signal_mem_ptr, recv_counts_re_ptr, expert_id, row_start, row_end,
        signal_epoch, WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        MAX_SOURCE_TILES: tl.constexpr, DISPATCH_BLOCK_M: tl.constexpr):
    source_start = 0
    ready_token = 0
    for source_id in range(0, WORLD_SIZE):
        source_size = tl.load(recv_counts_re_ptr +
                              source_id * EXPERTS_PER_RANK + expert_id)
        source_end = source_start + source_size
        overlap_start = tl.maximum(row_start, source_start)
        overlap_end = tl.minimum(row_end, source_end)
        if overlap_start < overlap_end:
            first_tile = (overlap_start - source_start) // DISPATCH_BLOCK_M
            last_tile = (overlap_end - source_start - 1) // DISPATCH_BLOCK_M
            signal_slot = ((source_id * EXPERTS_PER_RANK + expert_id)
                           * MAX_SOURCE_TILES + first_tile)
            ready_token += dl.wait(
                signal_mem_ptr + signal_slot * 16, last_tile - first_tile + 1,
                'gpu', 'acquire', waitValue=signal_epoch)
        source_start = source_end
    return ready_token


@triton.jit
def _wait_fc1_vector_ack(SLOT: tl.constexpr):
    # Dual Vector subcores acknowledge the same event before Cube reuses UB.
    al.sync_block_wait('vector', 'cube', 10 + SLOT,
                       al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)


@triton.jit
def _partition_pipeline_fc1_activation_group_ub(
        pid, input_ptr, signal_mem_ptr, weight_ptr, routing_weight_ptr, output_ptr,
        recv_expert_offs_ptr, recv_counts_re_ptr, signal_epoch, situ_beta, situ_linear_beta,
        stride_input_m, stride_input_k, stride_weight_e, stride_weight_n, stride_weight_k,
        WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr, MAX_SOURCE_TILES: tl.constexpr,
        FFN: tl.constexpr, K: tl.constexpr, DISPATCH_BLOCK_M: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_WINDOWS: tl.constexpr,
        FC1_CORES: tl.constexpr, ACTIVATION: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr,
        FULL_GROUP: tl.constexpr, wave_expert, wave_row_start, wave_rows):
    """Pipeline FC1 gate/up tiles through UB directly into activation output."""
    pair_block_m: tl.constexpr = BLOCK_M
    pair_block_n: tl.constexpr = BLOCK_N // 2
    vector_block_m: tl.constexpr = pair_block_m // 2
    activation_block_m: tl.constexpr = 32 if vector_block_m >= 32 else vector_block_m
    num_n_tiles: tl.constexpr = (FFN + pair_block_n - 1) // pair_block_n
    # The supported M*N <= 65536 bound leaves room for both UB slots.
    gate_buffer_0 = bl.alloc(tl.bfloat16, [vector_block_m, pair_block_n],
                             al.ascend_address_space.UB)
    up_buffer_0 = bl.alloc(tl.bfloat16, [vector_block_m, pair_block_n],
                           al.ascend_address_space.UB)
    gate_buffer_1 = bl.alloc(tl.bfloat16, [vector_block_m, pair_block_n],
                             al.ascend_address_space.UB)
    up_buffer_1 = bl.alloc(tl.bfloat16, [vector_block_m, pair_block_n],
                           al.ascend_address_space.UB)
    group_ready_token = 0
    cube_expert_id = wave_expert
    cube_expert_off = tl.load(recv_expert_offs_ptr + wave_expert)
    cube_group_start = wave_row_start
    cube_group_size = wave_rows
    cube_group_row_parts = tl.cdiv(wave_rows, pair_block_m)
    # The caller supplies a nonempty expert range and a lane in [0, cores).
    cube_lane_tiles = tl.cdiv(
        cube_group_row_parts * num_n_tiles - pid, FC1_CORES)
    vector_expert_off = cube_expert_off
    vector_group_start = cube_group_start
    vector_group_size = cube_group_size
    vector_group_row_parts = cube_group_row_parts
    vector_lane_tiles = cube_lane_tiles
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if vector_lane_tiles > 0:
            # Cube loads/MAC can overlap both Vector return workers.
            # Each UB owner releases its storage before the next Fixpipe.
            al.sync_block_set('vector', 'cube', 7, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)
    pipeline_tiles = cube_lane_tiles
    for pipeline_step in range(0, pipeline_tiles + 1):
        with al.scope(core_mode='cube', disable_auto_sync=True):
            tile_id = pid + pipeline_step * FC1_CORES
            row_part = tile_id % cube_group_row_parts
            n_tile = tile_id // cube_group_row_parts
            if pipeline_step < cube_lane_tiles:
                input_group_offset = (cube_expert_off.to(
                    tl.int64) + cube_group_start.to(tl.int64)) * stride_input_m
                local_row_start = row_part * pair_block_m
                row_count = pair_block_m if FULL_GROUP else tl.minimum(
                    pair_block_m, cube_group_size - local_row_start)
                if FULL_GROUP:
                    # With a power-of-two wave, a strided lane revisits
                    # its rows after W / gcd(W, cores) steps, often one.
                    core_power_two: tl.constexpr = FC1_CORES & -FC1_CORES
                    first_sweep_steps: tl.constexpr = (
                        GROUP_WINDOWS // core_power_two
                        if GROUP_WINDOWS > core_power_two else 1)
                    first_sweep = pipeline_step < first_sweep_steps
                else:
                    first_sweep = pipeline_step < cube_group_row_parts
                if first_sweep:
                    group_ready_token += _wait_dispatch_row_range(
                        signal_mem_ptr, recv_counts_re_ptr, cube_expert_id,
                        cube_group_start + local_row_start,
                        cube_group_start + local_row_start + row_count,
                        signal_epoch, WORLD_SIZE, EXPERTS_PER_RANK,
                        MAX_SOURCE_TILES, DISPATCH_BLOCK_M)
                # The Ascend consume-token ABI expects the allocation base;
                # a subview's offset is not preserved by its pointer return.
                ready_input_ptr = dl.consume_token(
                    input_ptr, group_ready_token) + input_group_offset
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
                                weight_base +
                                gate_cols[None, :] * stride_weight_n +
                                red[:, None] * stride_weight_k)
                            up_weight = tl.load(weight_base +
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
                                weight_base +
                                gate_cols[None, :] * stride_weight_n +
                                red[:, None] * stride_weight_k,
                                mask=mask_n[None, :],
                                other=0.0)
                            up_weight = tl.load(
                                weight_base +
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
                            weight_base +
                            gate_cols[None, :] * stride_weight_n +
                            red[:, None] * stride_weight_k,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0)
                        up_weight = tl.load(
                            weight_base +
                            up_cols[None, :] * stride_weight_n +
                            red[:, None] * stride_weight_k,
                            mask=mask_k[:, None] & mask_n[None, :],
                            other=0.0)
                    gate_acc += tl.dot(a, gate_weight)
                    up_acc += tl.dot(a, up_weight)
                # Only the UB overwrite depends on the previous consumer;
                # the next tile's loads and MAC can run before that handoff.
                if pipeline_step == 0:
                    al.sync_block_wait('vector', 'cube', 7, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)
                if pipeline_step >= 2:
                    if pipeline_step % 2 == 0:
                        _wait_fc1_vector_ack(0)
                    else:
                        _wait_fc1_vector_ack(1)
                if pipeline_step % 2 == 0:
                    al.fixpipe(gate_acc, gate_buffer_0, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                    al.fixpipe(up_acc, up_buffer_0, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                    al.sync_block_set('cube', 'vector', 8, al.PIPE.PIPE_FIX,
                                      al.PIPE.PIPE_V)
                else:
                    al.fixpipe(gate_acc, gate_buffer_1, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                    al.fixpipe(up_acc, up_buffer_1, dual_dst_mode=al.FixpipeDualDstMode.ROW_SPLIT)
                    al.sync_block_set('cube', 'vector', 9, al.PIPE.PIPE_FIX,
                                      al.PIPE.PIPE_V)
            if pipeline_step == pipeline_tiles:
                if cube_lane_tiles > 0:
                    # Drain every live buffer before reusing the group events.
                    if (cube_lane_tiles - 1) % 2 == 0:
                        _wait_fc1_vector_ack(0)
                    else:
                        _wait_fc1_vector_ack(1)
                    if cube_lane_tiles > 1:
                        if (cube_lane_tiles - 2) % 2 == 0:
                            _wait_fc1_vector_ack(0)
                        else:
                            _wait_fc1_vector_ack(1)
        with al.scope(core_mode='vector', disable_auto_sync=True):
            previous_step = pipeline_step - 1
            tile_id = pid + previous_step * FC1_CORES
            row_part = tile_id % vector_group_row_parts
            n_tile = tile_id // vector_group_row_parts
            previous_valid = (pipeline_step >= 1) & (
                previous_step < vector_lane_tiles)
            if previous_valid:
                local_row_start = row_part * pair_block_m
                row_count = pair_block_m if FULL_GROUP else tl.minimum(
                    pair_block_m, vector_group_size - local_row_start)
                output_group_row = vector_expert_off.to(
                    tl.int64) + vector_group_start.to(tl.int64)
                output_group_ptr = output_ptr + output_group_row * FFN
                routing_group_ptr = routing_weight_ptr + output_group_row
                col_start = n_tile * pair_block_n
                vector_row_base = (sub_vec_id() * vector_block_m)
                # UB conversion runs on V before the routing-weight MTE2 load.
                if previous_step % 2 == 0:
                    al.sync_block_wait('cube', 'vector', 8, al.PIPE.PIPE_FIX,
                                       al.PIPE.PIPE_V)
                else:
                    al.sync_block_wait('cube', 'vector', 9, al.PIPE.PIPE_FIX,
                                       al.PIPE.PIPE_V)
                activation_rows = vector_block_m
                for row_chunk in range(0, activation_rows, activation_block_m):
                    gate_view_0 = gate_buffer_0.subview([row_chunk, 0],
                                                        [activation_block_m, pair_block_n],
                                                        [1, 1])
                    up_view_0 = up_buffer_0.subview([row_chunk, 0],
                                                    [activation_block_m, pair_block_n], [1, 1])
                    gate_view_1 = gate_buffer_1.subview([row_chunk, 0],
                                                        [activation_block_m, pair_block_n],
                                                        [1, 1])
                    up_view_1 = up_buffer_1.subview([row_chunk, 0],
                                                    [activation_block_m, pair_block_n], [1, 1])
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
                    local_rows = vector_row_base + row_chunk + tl.arange(0, activation_block_m)
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
                    al.sync_block_set('vector', 'cube', 10,
                                      al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)
                else:
                    al.sync_block_set('vector', 'cube', 11,
                                      al.PIPE.PIPE_MTE3, al.PIPE.PIPE_FIX)


@triton.jit
def _return_rows_direct(combine_buf_ptr, fc2_output_ptr, source_row,
                        destination_row, num_rows, source_rank,
                        HIDDEN: tl.constexpr):
    block_size: tl.constexpr = 8192
    offsets = tl.arange(0, block_size)
    remote_output = dl.symm_at(combine_buf_ptr, source_rank)
    source_start = source_row.to(tl.int64) * HIDDEN
    destination_start = destination_row.to(tl.int64) * HIDDEN
    num_elements = num_rows * HIDDEN
    for chunk_start in range(0, num_elements, block_size):
        elements = chunk_start + offsets
        valid = elements < num_elements
        values = tl.load(fc2_output_ptr + source_start + elements,
                         mask=valid, other=0.0)
        tl.store(remote_output + destination_start + elements, values, mask=valid)


@triton.jit
def _build_dynamic_wave_offsets(
        destination_rank, counts_mem_ptr, wave_expert_offsets_ptr,
        WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        NUM_BINS_PAD: tl.constexpr, BLOCK_M: tl.constexpr):
    row_offset = 0
    block_offset = 0
    table = wave_expert_offsets_ptr + destination_rank * (EXPERTS_PER_RANK + 1) * 2
    for expert in range(EXPERTS_PER_RANK):
        # EPR1 has invariant zero bases, initialized with the workspace.
        # A constant GM store can keep its zero in UB across native putmem,
        # whose scratch overwrites that UB slot on the metadata publisher.
        if EXPERTS_PER_RANK > 1:
            tl.store(table + expert * 2, block_offset)
            tl.store(table + expert * 2 + 1, row_offset)
        rows = 0
        for source in range(WORLD_SIZE):
            rows += tl.load(counts_mem_ptr + source * NUM_BINS_PAD
                            + destination_rank * EXPERTS_PER_RANK + expert)
        row_offset += rows
        block_offset += tl.cdiv(rows, BLOCK_M)
    tl.store(table + EXPERTS_PER_RANK * 2, block_offset)
    tl.store(table + EXPERTS_PER_RANK * 2 + 1, row_offset)


@triton.jit
def _dynamic_wave_expert_range(
        wave_expert_offsets_ptr, rank, expert, wave,
        EXPERTS_PER_RANK: tl.constexpr, BLOCK_M: tl.constexpr,
        WAVE_WINDOWS: tl.constexpr):
    entry = wave_expert_offsets_ptr + (rank * (EXPERTS_PER_RANK + 1) + expert) * 2
    first_block = tl.load(entry)
    expert_offset = tl.load(entry + 1)
    expert_rows = tl.load(entry + 3) - expert_offset
    begin = tl.minimum(expert_rows, tl.maximum(
        wave * WAVE_WINDOWS - first_block, 0) * BLOCK_M)
    end = tl.minimum(expert_rows, tl.maximum(
        (wave + 1) * WAVE_WINDOWS - first_block, 0) * BLOCK_M)
    return begin, end, expert_offset, first_block


@triton.jit
def _dispatch_dynamic_wave(
        pid, wave, hidden_states_ptr, peer_mem_ptr, routing_weights_ptr,
        routing_weight_recv_ptr, signal_mem_ptr, send_token_indices_ptr,
        send_route_indices_ptr, send_bucket_dst_starts_ptr,
        send_bucket_starts_ptr, local_counts_ptr, wave_expert_offsets_ptr,
        signal_epoch, stride_hidden_m,
        NUM_CORES: tl.constexpr, LOCAL_RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        HIDDEN: tl.constexpr, BLOCK_M: tl.constexpr,
        WAVE_WINDOWS: tl.constexpr, MAX_SOURCE_TILES: tl.constexpr,
        DISPATCH_BLOCK_M: tl.constexpr):
    destination = pid % WORLD_SIZE
    lane = pid // WORLD_SIZE
    lanes = tl.cdiv(NUM_CORES - destination, WORLD_SIZE)
    for expert in range(EXPERTS_PER_RANK):
        begin, end, expert_offset, _ = _dynamic_wave_expert_range(
            wave_expert_offsets_ptr, destination, expert, wave,
            EXPERTS_PER_RANK, BLOCK_M, WAVE_WINDOWS)
        if begin < end:
            bucket = destination * EXPERTS_PER_RANK + expert
            source_begin = tl.load(send_bucket_dst_starts_ptr + bucket) - expert_offset
            source_rows = tl.load(local_counts_ptr + bucket)
            # A source tile belongs to the wave containing its first row.
            # Copy the entire tile there; later waves reuse its ready signal.
            first_tile = tl.cdiv(tl.maximum(begin - source_begin, 0), DISPATCH_BLOCK_M)
            last_tile = tl.cdiv(tl.minimum(tl.maximum(end - source_begin, 0),
                                          source_rows), DISPATCH_BLOCK_M)
            if first_tile < last_tile:
                _dispatch_one_source_tile_task(
                    bucket, lane, lanes, hidden_states_ptr, peer_mem_ptr,
                    routing_weights_ptr, routing_weight_recv_ptr, signal_mem_ptr,
                    send_token_indices_ptr, send_route_indices_ptr,
                    send_bucket_dst_starts_ptr, send_bucket_starts_ptr,
                    local_counts_ptr, signal_epoch, HIDDEN, stride_hidden_m,
                    LOCAL_RANK, EXPERTS_PER_RANK, MAX_SOURCE_TILES,
                    DISPATCH_BLOCK_M, True, first_tile, last_tile)


@triton.jit
def _fc2_dynamic_wave(
        pid, wave, input_ptr, weight_ptr, output_ptr, pipeline_signal_ptr, wave_expert_offsets_ptr,
        signal_epoch, stride_weight_e, stride_weight_n, stride_weight_k, NUM_CORES: tl.constexpr,
        LOCAL_RANK: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr, MAX_WAVES: tl.constexpr,
        HIDDEN: tl.constexpr, FFN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, WAVE_WINDOWS: tl.constexpr):
    completion_base: tl.constexpr = MAX_WAVES * NUM_CORES
    n_tiles: tl.constexpr = (HIDDEN + BLOCK_N - 1) // BLOCK_N
    activation_ready = dl.wait(
        pipeline_signal_ptr + wave * NUM_CORES * 16,
        NUM_CORES, 'gpu', 'acquire', waitValue=2)
    wave_input = dl.consume_token(input_ptr, activation_ready)
    for expert in range(EXPERTS_PER_RANK):
        begin, end, expert_offset, first_block = _dynamic_wave_expert_range(
            wave_expert_offsets_ptr, LOCAL_RANK, expert, wave,
            EXPERTS_PER_RANK, BLOCK_M, WAVE_WINDOWS)
        if begin < end:
            row_tiles = tl.cdiv(end - begin, BLOCK_M)
            tile_base = (first_block + begin // BLOCK_M) * n_tiles
            first_tile = (pid + NUM_CORES - tile_base % NUM_CORES) % NUM_CORES
            for tile in range(first_tile, row_tiles * n_tiles, NUM_CORES):
                row_start = begin + tile // n_tiles * BLOCK_M
                _fc2_gemm_one_mn_tile(
                    wave_input, weight_ptr, output_ptr, expert,
                    expert_offset + row_start, tl.minimum(BLOCK_M, end - row_start),
                    tile % n_tiles, HIDDEN, FFN, FFN, 1, stride_weight_e,
                    stride_weight_n, stride_weight_k, HIDDEN, 1,
                    BLOCK_M, BLOCK_N, BLOCK_K, 0)
    libshmem_device.fence()
    libshmem_device.signal_op(
        pipeline_signal_ptr + (completion_base + wave * NUM_CORES + pid) * 16,
        signal_epoch, libshmem_device.ACLSHMEM_SIGNAL_SET, LOCAL_RANK)


@triton.jit
def _return_dynamic_wave(
        worker, wave, combine_buf_ptr, fc2_output_ptr, recv_counts_re_ptr,
        pull_tile_dst_start_ptr, pipeline_signal_ptr, wave_expert_offsets_ptr, signal_epoch,
        NUM_CORES: tl.constexpr, LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, MAX_WAVES: tl.constexpr, HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr, WAVE_WINDOWS: tl.constexpr):
    completion_base: tl.constexpr = MAX_WAVES * NUM_CORES
    return_base: tl.constexpr = 2 * MAX_WAVES * NUM_CORES
    worker = worker.to(tl.int32)
    source = worker % WORLD_SIZE
    lane = worker // WORLD_SIZE
    lanes = tl.cdiv(2 * NUM_CORES - source, WORLD_SIZE)
    ready = dl.wait(
        pipeline_signal_ptr + (completion_base + wave * NUM_CORES) * 16,
        NUM_CORES, 'gpu', 'acquire', waitValue=signal_epoch)
    wave_fc2 = dl.consume_token(fc2_output_ptr, ready)
    for expert in range(EXPERTS_PER_RANK):
        begin, end, expert_offset, _ = _dynamic_wave_expert_range(
            wave_expert_offsets_ptr, LOCAL_RANK, expert, wave,
            EXPERTS_PER_RANK, BLOCK_M, WAVE_WINDOWS)
        if begin < end:
            source_begin = 0
            for prior in range(WORLD_SIZE):
                count = tl.load(recv_counts_re_ptr + prior * EXPERTS_PER_RANK + expert)
                source_begin += tl.where(prior < source, count, 0)
            source_rows = tl.load(recv_counts_re_ptr + source * EXPERTS_PER_RANK + expert)
            overlap_begin = tl.maximum(begin, source_begin)
            overlap_rows = tl.maximum(tl.minimum(end, source_begin + source_rows)
                                      - overlap_begin, 0)
            part_begin = overlap_rows * lane // lanes
            part_end = overlap_rows * (lane + 1) // lanes
            if part_begin < part_end:
                destination_start = tl.load(
                    pull_tile_dst_start_ptr + expert * WORLD_SIZE + source)
                source_row = expert_offset.to(tl.int64) + overlap_begin + part_begin
                destination_row = (destination_start.to(tl.int64) + overlap_begin
                                   - source_begin + part_begin)
                _return_rows_direct(
                    combine_buf_ptr, wave_fc2, source_row, destination_row,
                    part_end - part_begin, source, HIDDEN)
    libshmem_device.fence()
    libshmem_device.signal_op(
        pipeline_signal_ptr + (return_base + wave * WORLD_SIZE + LOCAL_RANK) * 16,
        1, libshmem_device.ACLSHMEM_SIGNAL_ADD, source)


@triton.jit
def _wait_dynamic_wave_returns(
        wave_expert_offsets_ptr, pipeline_signal_ptr, NUM_CORES: tl.constexpr,
        LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
        MAX_WAVES: tl.constexpr, WAVE_WINDOWS: tl.constexpr):
    return_base: tl.constexpr = 2 * MAX_WAVES * NUM_CORES
    ready = 0
    for destination in range(WORLD_SIZE):
        blocks = tl.load(wave_expert_offsets_ptr
                         + (destination * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK) * 2)
        waves = tl.cdiv(blocks, WAVE_WINDOWS)
        for wave in range(waves):
            expected = tl.cdiv(2 * NUM_CORES - LOCAL_RANK, WORLD_SIZE)
            ready += dl.wait(
                pipeline_signal_ptr + (return_base + wave * WORLD_SIZE + destination) * 16,
                1, 'gpu', 'acquire', waitValue=expected)
    return ready


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


@triton.jit
def _run_dynamic_wave_pipeline(
        pid, hidden_states_ptr, gate_up_weight_ptr, down_weight_ptr, routing_weights_ptr,
        peer_mem_ptr, routing_weight_recv_ptr, signal_mem_ptr, pipeline_signal_ptr,
        weighted_activation_ptr, fc2_output_ptr, combine_buf_ptr, output_ptr, local_counts_ptr,
        send_bucket_starts_ptr, send_bucket_dst_starts_ptr, send_token_indices_ptr,
        send_route_indices_ptr, route_to_send_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
        pull_tile_dst_start_ptr, wave_expert_offsets_ptr, num_routes, signal_epoch, capacity_ok,
        situ_beta, situ_linear_beta, stride_hidden_m, stride_hidden_k, stride_gate_up_e,
        stride_gate_up_n, stride_gate_up_k, stride_down_e, stride_down_n, stride_down_k,
        NUM_CORES: tl.constexpr, LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, HIDDEN: tl.constexpr, FFN: tl.constexpr, TOPK: tl.constexpr,
        MAX_SOURCE_TILES: tl.constexpr, MAX_WAVES: tl.constexpr, DISPATCH_BLOCK_M: tl.constexpr,
        BLOCK_M: tl.constexpr, FC1_BLOCK_N: tl.constexpr, FC1_BLOCK_K: tl.constexpr,
        FC2_BLOCK_N: tl.constexpr, FC2_BLOCK_K: tl.constexpr, WAVE_WINDOWS: tl.constexpr,
        ACTIVATION: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr):
    global_waves = 0
    for rank in range(WORLD_SIZE):
        blocks = tl.load(wave_expert_offsets_ptr
                         + (rank * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK) * 2)
        global_waves = tl.maximum(global_waves, tl.cdiv(blocks, WAVE_WINDOWS))
    local_blocks = tl.load(wave_expert_offsets_ptr
                           + (LOCAL_RANK * (EXPERTS_PER_RANK + 1) + EXPERTS_PER_RANK) * 2)
    local_waves = tl.where(capacity_ok, tl.cdiv(local_blocks, WAVE_WINDOWS), 0)
    global_waves = tl.where(capacity_ok, global_waves, 0)
    fc1_n_tiles: tl.constexpr = (FFN + FC1_BLOCK_N // 2 - 1) // (FC1_BLOCK_N // 2)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 1:
            for prepared_wave in range(2):
                if prepared_wave < global_waves:
                    _dispatch_dynamic_wave(
                        pid, prepared_wave, hidden_states_ptr, peer_mem_ptr,
                        routing_weights_ptr, routing_weight_recv_ptr, signal_mem_ptr,
                        send_token_indices_ptr, send_route_indices_ptr,
                        send_bucket_dst_starts_ptr, send_bucket_starts_ptr,
                        local_counts_ptr, wave_expert_offsets_ptr, signal_epoch,
                        stride_hidden_m, NUM_CORES, LOCAL_RANK, WORLD_SIZE,
                        EXPERTS_PER_RANK, HIDDEN, BLOCK_M, WAVE_WINDOWS,
                        MAX_SOURCE_TILES, DISPATCH_BLOCK_M)
    # Empty receive ranks still dispatch every global wave. Keeping dispatch
    # ahead of all return waits prevents a cross-rank dependency cycle.
    for step in range(global_waves + 2):
        with al.scope(core_mode='vector', disable_auto_sync=True):
            if (sub_vec_id() == 1) & (step > 0) & (step + 1 < global_waves):
                _dispatch_dynamic_wave(
                    pid, step + 1, hidden_states_ptr, peer_mem_ptr,
                    routing_weights_ptr, routing_weight_recv_ptr, signal_mem_ptr,
                    send_token_indices_ptr, send_route_indices_ptr,
                    send_bucket_dst_starts_ptr, send_bucket_starts_ptr,
                    local_counts_ptr, wave_expert_offsets_ptr, signal_epoch,
                    stride_hidden_m, NUM_CORES, LOCAL_RANK, WORLD_SIZE,
                    EXPERTS_PER_RANK, HIDDEN, BLOCK_M, WAVE_WINDOWS,
                    MAX_SOURCE_TILES, DISPATCH_BLOCK_M)
        if step < local_waves:
            for expert in range(EXPERTS_PER_RANK):
                begin, end, _, first_block = _dynamic_wave_expert_range(
                    wave_expert_offsets_ptr, LOCAL_RANK, expert, step,
                    EXPERTS_PER_RANK, BLOCK_M, WAVE_WINDOWS)
                if begin < end:
                    tile_base = (first_block + begin // BLOCK_M) * fc1_n_tiles
                    lane = (pid + NUM_CORES - tile_base % NUM_CORES) % NUM_CORES
                    full_rows = end - begin == WAVE_WINDOWS * BLOCK_M
                    if full_rows:
                        _partition_pipeline_fc1_activation_group_ub(
                            lane, peer_mem_ptr, signal_mem_ptr, gate_up_weight_ptr,
                            routing_weight_recv_ptr, weighted_activation_ptr, recv_expert_offs_ptr,
                            recv_counts_re_ptr, signal_epoch, situ_beta, situ_linear_beta,
                            stride_hidden_m, stride_hidden_k, stride_gate_up_e, stride_gate_up_n,
                            stride_gate_up_k, WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, FFN,
                            HIDDEN, DISPATCH_BLOCK_M, BLOCK_M, FC1_BLOCK_N, FC1_BLOCK_K,
                            WAVE_WINDOWS, NUM_CORES, ACTIVATION, HAS_LINEAR_BETA, True, expert,
                            begin, end - begin)
                    else:
                        _partition_pipeline_fc1_activation_group_ub(
                            lane, peer_mem_ptr, signal_mem_ptr, gate_up_weight_ptr,
                            routing_weight_recv_ptr, weighted_activation_ptr, recv_expert_offs_ptr,
                            recv_counts_re_ptr, signal_epoch, situ_beta, situ_linear_beta,
                            stride_hidden_m, stride_hidden_k, stride_gate_up_e, stride_gate_up_n,
                            stride_gate_up_k, WORLD_SIZE, EXPERTS_PER_RANK, MAX_SOURCE_TILES, FFN,
                            HIDDEN, DISPATCH_BLOCK_M, BLOCK_M, FC1_BLOCK_N, FC1_BLOCK_K,
                            WAVE_WINDOWS, NUM_CORES, ACTIVATION, HAS_LINEAR_BETA, False, expert,
                            begin, end - begin)
            with al.scope(core_mode='vector', disable_auto_sync=True):
                libshmem_device.fence()
                libshmem_device.signal_op(
                    pipeline_signal_ptr + (step * NUM_CORES + pid) * 16,
                    1, libshmem_device.ACLSHMEM_SIGNAL_ADD, LOCAL_RANK)
        with al.scope(core_mode='cube', disable_auto_sync=True):
            if (step > 0) & (step - 1 < local_waves):
                _fc2_dynamic_wave(
                    pid, step - 1, weighted_activation_ptr, down_weight_ptr, fc2_output_ptr,
                    pipeline_signal_ptr, wave_expert_offsets_ptr, signal_epoch, stride_down_e,
                    stride_down_n, stride_down_k, NUM_CORES, LOCAL_RANK, EXPERTS_PER_RANK, MAX_WAVES,
                    HIDDEN, FFN, BLOCK_M, FC2_BLOCK_N, FC2_BLOCK_K, WAVE_WINDOWS)
        with al.scope(core_mode='vector', disable_auto_sync=True):
            if (step >= 2) & (step - 2 < local_waves):
                _return_dynamic_wave(
                    pid * 2 + sub_vec_id(), step - 2, combine_buf_ptr,
                    fc2_output_ptr, recv_counts_re_ptr, pull_tile_dst_start_ptr, pipeline_signal_ptr,
                    wave_expert_offsets_ptr, signal_epoch, NUM_CORES, LOCAL_RANK, WORLD_SIZE,
                    EXPERTS_PER_RANK, MAX_WAVES, HIDDEN, BLOCK_M, WAVE_WINDOWS)
    with al.scope(core_mode='vector', disable_auto_sync=True):
        return_ready = 0
        if capacity_ok:
            return_ready = _wait_dynamic_wave_returns(
                wave_expert_offsets_ptr, pipeline_signal_ptr, NUM_CORES, LOCAL_RANK,
                WORLD_SIZE, EXPERTS_PER_RANK, MAX_WAVES, WAVE_WINDOWS)
        ready_combine = dl.consume_token(combine_buf_ptr, return_ready)
        reduce_block_n: tl.constexpr = _REDUCE_BLOCK_N if HIDDEN >= 2048 else 1024
        _reduce_topk_rows(
            pid * 2 + sub_vec_id(), ready_combine, route_to_send_ptr, output_ptr,
            num_routes // TOPK, capacity_ok, 2 * NUM_CORES, HIDDEN, TOPK, reduce_block_n)


@triton.jit(do_not_specialize=['num_routes', 'signal_epoch'])
def _kernel_fused_forward(
        hidden_states_ptr, selected_experts_ptr, routing_weights_ptr, gate_up_weight_ptr,
        down_weight_ptr, peer_mem_ptr, routing_weight_recv_ptr, signal_mem_ptr, pipeline_signal_ptr,
        combine_buf_ptr, fc2_output_ptr, weighted_activation_ptr, output_ptr, counts_mem_ptr,
        send_bucket_starts_ptr, send_bucket_dst_starts_ptr, recv_counts_re_ptr, recv_per_expert_ptr,
        recv_expert_offs_ptr, stats_ptr, core_bucket_cursor_ptr, send_token_indices_ptr,
        send_route_indices_ptr, route_to_send_ptr, pull_tile_dst_start_ptr, wave_expert_offsets_ptr,
        num_routes, signal_epoch, situ_beta, situ_linear_beta, stride_hidden_m: tl.constexpr,
        stride_hidden_k: tl.constexpr, stride_gate_up_e: tl.constexpr,
        stride_gate_up_n: tl.constexpr, stride_gate_up_k: tl.constexpr, stride_down_e: tl.constexpr,
        stride_down_n: tl.constexpr, stride_down_k: tl.constexpr, NUM_PROGRAM_CORES: tl.constexpr,
        LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr, NUM_EXPERTS: tl.constexpr,
        EXPERTS_PER_RANK: tl.constexpr, TOPK: tl.constexpr, HIDDEN: tl.constexpr, FFN: tl.constexpr,
        MAX_RECEIVED_ROUTES: tl.constexpr, NUM_BINS_PAD: tl.constexpr,
        MAX_SOURCE_TILES: tl.constexpr, MAX_PIPELINE_GROUPS: tl.constexpr,
        DISPATCH_BLOCK_M: tl.constexpr, FC1_BLOCK_M: tl.constexpr, FC1_BLOCK_N: tl.constexpr,
        FC1_BLOCK_K: tl.constexpr, FC2_BLOCK_N: tl.constexpr,
        FC2_BLOCK_K: tl.constexpr, ACTIVATION: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr,
        PIPELINE_GROUP_WINDOWS: tl.constexpr):
    """Production routing-to-reduction pipeline for the single-kernel path."""
    pid = tl.program_id(axis=0)
    pipeline_counter_count: tl.constexpr = MAX_PIPELINE_GROUPS * (
        2 * NUM_PROGRAM_CORES + WORLD_SIZE)
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
    _mixed_forward_barrier()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid == 0):
            _publish_count_row(core_bucket_cursor_ptr, counts_mem_ptr,
                               LOCAL_RANK, WORLD_SIZE, NUM_PROGRAM_CORES,
                               NUM_EXPERTS, NUM_BINS_PAD)
    _mixed_forward_barrier()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if (sub_vec_id() == 0) & (pid < WORLD_SIZE):
            _build_destination_metadata(
                pid, counts_mem_ptr, send_bucket_starts_ptr,
                send_bucket_dst_starts_ptr, recv_counts_re_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, stats_ptr,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK, NUM_BINS_PAD)
            _build_dynamic_wave_offsets(
                pid, counts_mem_ptr, wave_expert_offsets_ptr, WORLD_SIZE,
                EXPERTS_PER_RANK, NUM_BINS_PAD, FC1_BLOCK_M)
    _mixed_forward_barrier()
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
    _mixed_forward_barrier()
    with al.scope(core_mode='vector', disable_auto_sync=True):
        if sub_vec_id() == 0:
            _scatter_stable_routes(pid, selected_experts_ptr,
                                   core_bucket_cursor_ptr,
                                   send_token_indices_ptr,
                                   send_route_indices_ptr, route_to_send_ptr,
                                   num_routes, NUM_PROGRAM_CORES, NUM_EXPERTS,
                                   NUM_BINS_PAD, TOPK, _SCATTER_BLOCK)
    _mixed_forward_barrier()
    capacity_ok = tl.load(stats_ptr + 1) <= MAX_RECEIVED_ROUTES
    local_counts_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    _run_dynamic_wave_pipeline(
        pid, hidden_states_ptr, gate_up_weight_ptr, down_weight_ptr, routing_weights_ptr,
        peer_mem_ptr, routing_weight_recv_ptr, signal_mem_ptr, pipeline_signal_ptr,
        weighted_activation_ptr, fc2_output_ptr, combine_buf_ptr, output_ptr, local_counts_ptr,
        send_bucket_starts_ptr, send_bucket_dst_starts_ptr, send_token_indices_ptr,
        send_route_indices_ptr, route_to_send_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
        pull_tile_dst_start_ptr, wave_expert_offsets_ptr, num_routes, signal_epoch, capacity_ok,
        situ_beta, situ_linear_beta, stride_hidden_m, stride_hidden_k, stride_gate_up_e,
        stride_gate_up_n, stride_gate_up_k, stride_down_e, stride_down_n, stride_down_k,
        NUM_PROGRAM_CORES, LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK, HIDDEN, FFN, TOPK,
        MAX_SOURCE_TILES, MAX_PIPELINE_GROUPS, DISPATCH_BLOCK_M, FC1_BLOCK_M, FC1_BLOCK_N,
        FC1_BLOCK_K, FC2_BLOCK_N, FC2_BLOCK_K, PIPELINE_GROUP_WINDOWS, ACTIVATION, HAS_LINEAR_BETA)
