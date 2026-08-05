# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BF16 Ascend Triton kernel for MoE FC2 and distributed combine.

The input rows are already grouped as local-expert major, then source-rank
major.  Routing weights were applied by weighted SwiGLU before this stage.
This module therefore performs only FC2, route-output transport, and the final
top-k reduction.  The direct path pulls contiguous source/expert buckets into
a local workspace before reducing; it never issues fine-grained remote loads
from the token/top-k loop.
"""

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id


_ROUTE_BLOCK = 256


@triton.jit
def _fill_int_kernel(output_ptr, size, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(output_ptr + offs, -1, mask=offs < size)


@triton.jit
def _scatter_route_to_send_kernel(send_route_idx_ptr, route_to_send_ptr, num_send, BLOCK: tl.constexpr):
    sorted_offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = sorted_offs < num_send
    route_ids = tl.load(send_route_idx_ptr + sorted_offs, mask=mask, other=0)
    tl.store(route_to_send_ptr + route_ids, sorted_offs, mask=mask)


@triton.jit
def _prepare_fc2_combine_metadata_kernel(
    recv_counts_re_ptr,
    recv_expert_offs_ptr,
    counts_mem_ptr,
    send_bucket_starts_ptr,
    send_bucket_receive_offsets_ptr,
    fc2_tile_expert_ptr,
    fc2_tile_row_start_ptr,
    fc2_tile_row_count_ptr,
    reverse_tile_rank_ptr,
    reverse_tile_src_start_ptr,
    reverse_tile_dst_start_ptr,
    reverse_tile_row_count_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BUILD_FC2_TILES: tl.constexpr,
    DIRECT_PULL: tl.constexpr,
):
    """Build compact FC2 and route-transport tiles on one Vector core."""
    if BUILD_FC2_TILES:
        fc2_cursor = 0
        for expert_id in range(0, EXPERTS_PER_RANK):
            expert_count = tl.load(recv_counts_re_ptr + expert_id)
            # recv_counts_re is [source, expert]; sum all sources for this expert.
            for source_rank in range(1, WORLD_SIZE):
                expert_count += tl.load(
                    recv_counts_re_ptr
                    + source_rank * EXPERTS_PER_RANK
                    + expert_id
                )
            expert_start = tl.load(recv_expert_offs_ptr + expert_id)
            num_tiles = tl.cdiv(expert_count, BLOCK_M)
            for tile_id in range(0, num_tiles):
                row_start = expert_start + tile_id * BLOCK_M
                row_count = tl.minimum(
                    BLOCK_M,
                    expert_count - tile_id * BLOCK_M,
                )
                tl.store(fc2_tile_expert_ptr + fc2_cursor, expert_id)
                tl.store(fc2_tile_row_start_ptr + fc2_cursor, row_start)
                tl.store(fc2_tile_row_count_ptr + fc2_cursor, row_count)
                fc2_cursor += 1

    transport_cursor = 0
    if DIRECT_PULL:
        # On the home/source rank, every stable-send bucket maps one-to-one to
        # a contiguous source segment in the destination rank's expert-major
        # FC2 output.  Pull those segments into local stable-send order so the
        # later top-k loop performs local loads only.
        local_count_row = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
        for bucket in range(0, WORLD_SIZE * EXPERTS_PER_RANK):
            route_count = tl.load(local_count_row + bucket)
            local_send_start = tl.load(send_bucket_starts_ptr + bucket)
            remote_receive_start = tl.load(
                send_bucket_receive_offsets_ptr + bucket
            )
            destination_rank = bucket // EXPERTS_PER_RANK
            num_tiles = tl.cdiv(route_count, BLOCK_M)
            for tile_id in range(0, num_tiles):
                tile_delta = tile_id * BLOCK_M
                row_count = tl.minimum(BLOCK_M, route_count - tile_delta)
                tl.store(
                    reverse_tile_rank_ptr + transport_cursor,
                    destination_rank,
                )
                tl.store(
                    reverse_tile_src_start_ptr + transport_cursor,
                    remote_receive_start + tile_delta,
                )
                tl.store(
                    reverse_tile_dst_start_ptr + transport_cursor,
                    local_send_start + tile_delta,
                )
                tl.store(
                    reverse_tile_row_count_ptr + transport_cursor,
                    row_count,
                )
                transport_cursor += 1
    else:
        for source_rank in range(0, WORLD_SIZE):
            # On the home/source rank, returned rows use the same stable
            # global-expert order as its original send list.  Compute the
            # prefix preceding this destination once, then advance by expert.
            remote_bucket_start = 0
            for prior_bucket in range(0, LOCAL_RANK * EXPERTS_PER_RANK):
                remote_bucket_start += tl.load(
                    counts_mem_ptr
                    + source_rank * NUM_BINS_PAD
                    + prior_bucket
                )

            for expert_id in range(0, EXPERTS_PER_RANK):
                source_count = tl.load(
                    recv_counts_re_ptr
                    + source_rank * EXPERTS_PER_RANK
                    + expert_id
                )
                source_row_start = tl.load(
                    recv_expert_offs_ptr + expert_id
                )
                for prior_source in range(0, WORLD_SIZE):
                    if prior_source < source_rank:
                        source_row_start += tl.load(
                            recv_counts_re_ptr
                            + prior_source * EXPERTS_PER_RANK
                            + expert_id
                        )

                num_tiles = tl.cdiv(source_count, BLOCK_M)
                for tile_id in range(0, num_tiles):
                    tile_delta = tile_id * BLOCK_M
                    row_count = tl.minimum(
                        BLOCK_M, source_count - tile_delta
                    )
                    tl.store(
                        reverse_tile_rank_ptr + transport_cursor,
                        source_rank,
                    )
                    tl.store(
                        reverse_tile_src_start_ptr + transport_cursor,
                        source_row_start + tile_delta,
                    )
                    tl.store(
                        reverse_tile_dst_start_ptr + transport_cursor,
                        remote_bucket_start + tile_delta,
                    )
                    tl.store(
                        reverse_tile_row_count_ptr + transport_cursor,
                        row_count,
                    )
                    transport_cursor += 1

                remote_bucket_start += source_count


@triton.jit
def _fc2_gemm_one_mn_tile(
    input_ptr,
    weight_ptr,
    fc2_buf_ptr,
    expert_id,
    row_start,
    row_count,
    n_tile,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_fc2_m,
    stride_fc2_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one FC2 M/N tile for both task-scheduling variants."""
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rows = row_start + offs_m
    cols = n_tile * BLOCK_N + offs_n
    mask_m = offs_m < row_count
    mask_n = cols < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    weight_base = weight_ptr + expert_id.to(tl.int64) * stride_weight_e
    for k_start in range(0, K, BLOCK_K):
        red = k_start + offs_k
        mask_k = red < K
        a_ptrs = (
            input_ptr
            + rows[:, None] * stride_input_m
            + red[None, :] * stride_input_k
        )
        b_ptrs = (
            weight_base
            + cols[None, :] * stride_weight_n
            + red[:, None] * stride_weight_k
        )
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(a, b)
    out_ptrs = (
        fc2_buf_ptr
        + rows[:, None] * stride_fc2_m
        + cols[None, :] * stride_fc2_n
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _kernel_fc2_combine(
    input_ptr,
    weight_ptr,
    fc2_buf_ptr,
    peer_mem_ptr,
    route_to_send_ptr,
    output_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    fc2_tile_expert_ptr,
    fc2_tile_row_start_ptr,
    fc2_tile_row_count_ptr,
    reverse_tile_rank_ptr,
    reverse_tile_src_start_ptr,
    reverse_tile_dst_start_ptr,
    reverse_tile_row_count_ptr,
    num_fc2_slots,
    num_reverse_slots,
    batch_size,
    num_send,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_fc2_m,
    stride_fc2_n,
    stride_output_m,
    stride_output_n,
    N: tl.constexpr,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N_PUSH: tl.constexpr,
    BLOCK_N_REDUCE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    FC2_EXPERT_N_PERSISTENT: tl.constexpr,
    DIRECT_PULL: tl.constexpr,
    REVERSE_VECTOR_WORKERS: tl.constexpr,
    REDUCE_VECTOR_WORKERS: tl.constexpr,
):
    """FC2 -> bucket transport -> local top-k reduction."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)
    fc2_output_ptr = fc2_buf_ptr
    if DIRECT_PULL:
        # Direct mode leaves FC2 rows in symmetric memory, then bulk-pulls
        # contiguous bucket segments into the ordinary local workspace.
        fc2_output_ptr = peer_mem_ptr

    # Phase 1: BF16 FC2 with FP32 accumulation.
    with al.scope(core_mode="cube", disable_auto_sync=True):
        # ``launch_fc2_combine`` requires BLOCK_N to divide N.  Keep this as
        # pure constexpr arithmetic: the Ascend frontend materializes
        # ``tl.cdiv`` as a semantic value, which cannot then be multiplied by
        # the constexpr expert count in the persistent schedule.
        num_n_tiles: tl.constexpr = N // BLOCK_N
        if FC2_EXPERT_N_PERSISTENT:
            # Expert-major ownership: one task keeps a fixed (expert, N tile)
            # while traversing every merged M window for that expert.
            total_fc2_tasks: tl.constexpr = EXPERTS_PER_RANK * num_n_tiles
            for task_id in range(pid, total_fc2_tasks, ncore):
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
                            input_ptr,
                            weight_ptr,
                            fc2_output_ptr,
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
                            stride_fc2_m,
                            stride_fc2_n,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                        )
        else:
            # Baseline N-major ownership over prebuilt (expert, M window)
            # descriptors.  Keep this exact schedule for isolated A/B runs.
            total_fc2_tasks = num_fc2_slots * num_n_tiles
            for task_id in range(pid, total_fc2_tasks, ncore):
                tile_id = task_id % num_fc2_slots
                n_tile = task_id // num_fc2_slots
                expert_id = tl.load(fc2_tile_expert_ptr + tile_id)
                if expert_id >= 0:
                    row_start = tl.load(fc2_tile_row_start_ptr + tile_id)
                    row_count = tl.load(fc2_tile_row_count_ptr + tile_id)
                    _fc2_gemm_one_mn_tile(
                        input_ptr,
                        weight_ptr,
                        fc2_output_ptr,
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
                        stride_fc2_m,
                        stride_fc2_n,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                    )

    if not DIRECT_PULL:
        # Keep the legacy reverse-push A/B path fused.  Direct pull is split
        # into a pure-Vector kernel below because mixing blocking getmem with
        # the Cube pipeline is not runtime-safe on the current backend.
        # Reverse push still needs its rank-wide dependency inside this kernel.
        libshmem_device.barrier_all()
        with al.scope(core_mode="vector", disable_auto_sync=True):
            transport_sub_id = sub_vec_id().to(tl.int32)
            if transport_sub_id < REVERSE_VECTOR_WORKERS:
                transport_worker_id = pid * REVERSE_VECTOR_WORKERS + transport_sub_id
                num_transport_workers = ncore * REVERSE_VECTOR_WORKERS
                transport_cols = tl.arange(0, BLOCK_N_PUSH)
                for tile_id in range(
                    transport_worker_id,
                    num_reverse_slots,
                    num_transport_workers,
                ):
                    peer_rank = tl.load(reverse_tile_rank_ptr + tile_id)
                    if peer_rank >= 0:
                        src_start = tl.load(
                            reverse_tile_src_start_ptr + tile_id
                        ).to(tl.int64)
                        dst_start = tl.load(
                            reverse_tile_dst_start_ptr + tile_id
                        ).to(tl.int64)
                        row_count = tl.load(reverse_tile_row_count_ptr + tile_id)
                        remote_peer = dl.symm_at(peer_mem_ptr, peer_rank)
                        for row_delta in range(0, row_count):
                            for col_start in range(0, N, BLOCK_N_PUSH):
                                cols = col_start + transport_cols
                                mask = cols < N
                                values = tl.load(
                                    fc2_buf_ptr
                                    + (src_start + row_delta) * stride_fc2_m
                                    + cols * stride_fc2_n,
                                    mask=mask,
                                    other=0.0,
                                )
                                tl.store(
                                    remote_peer
                                    + (dst_start + row_delta) * N
                                    + cols,
                                    values,
                                    mask=mask,
                                )

        libshmem_device.barrier_all()

        with al.scope(core_mode="vector", disable_auto_sync=True):
            reduce_sub_id = sub_vec_id().to(tl.int32)
            if reduce_sub_id < REDUCE_VECTOR_WORKERS:
                reduce_worker_id = pid * REDUCE_VECTOR_WORKERS + reduce_sub_id
                num_reduce_workers = ncore * REDUCE_VECTOR_WORKERS
                reduce_cols = tl.arange(0, BLOCK_N_REDUCE)
                for token_id in range(
                    reduce_worker_id,
                    batch_size,
                    num_reduce_workers,
                ):
                    token_id64 = token_id.to(tl.int64)
                    for col_start in range(0, N, BLOCK_N_REDUCE):
                        cols = col_start + reduce_cols
                        mask_n = cols < N
                        acc = tl.zeros((BLOCK_N_REDUCE,), dtype=tl.float32)
                        for topk_slot in range(0, TOPK):
                            route_id = token_id * TOPK + topk_slot
                            send_row = tl.load(route_to_send_ptr + route_id)
                            valid = (send_row >= 0) & (send_row < num_send)
                            safe_row = tl.where(valid, send_row, 0).to(tl.int64)
                            values = tl.load(
                                peer_mem_ptr + safe_row * N + cols,
                                mask=mask_n & valid,
                                other=0.0,
                            ).to(tl.float32)
                            acc += values
                        tl.store(
                            output_ptr
                            + token_id64 * stride_output_m
                            + cols * stride_output_n,
                            acc.to(tl.bfloat16),
                            mask=mask_n,
                        )

        # Reverse push reduces directly from peer_mem, so do not let a faster
        # rank overwrite a slower rank's input during the next dispatch.
        libshmem_device.barrier_all()


@triton.jit
def _kernel_direct_pull_transport(
    fc2_buf_ptr,
    peer_mem_ptr,
    reverse_tile_rank_ptr,
    reverse_tile_src_start_ptr,
    reverse_tile_dst_start_ptr,
    reverse_tile_row_count_ptr,
    num_reverse_slots,
    stride_fc2_m,
    N: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Pull FC2 descriptors into an ordinary local workspace."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)

    # Reaching this separate launch proves that the local FC2 kernel has fully
    # exited.  Synchronize here, rather than before that exit, so every peer's
    # symmetric FC2 rows and Cube-store epilogue are visible before the first
    # remote get starts.
    libshmem_device.barrier_all_vec()

    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            for peer_owner in range(pid, WORLD_SIZE, ncore):
                for tile_id in range(0, num_reverse_slots):
                    peer_rank = tl.load(reverse_tile_rank_ptr + tile_id)
                    if peer_rank == peer_owner:
                        src_start = tl.load(
                            reverse_tile_src_start_ptr + tile_id
                        ).to(tl.int64)
                        dst_start = tl.load(
                            reverse_tile_dst_start_ptr + tile_id
                        ).to(tl.int64)
                        row_count = tl.load(
                            reverse_tile_row_count_ptr + tile_id
                        )
                        libshmem_device.getmem(
                            fc2_buf_ptr + dst_start * stride_fc2_m,
                            peer_mem_ptr + src_start * N,
                            row_count * N * 2,
                            peer_rank,
                        )

    # All remote readers must leave symmetric FC2 storage before any rank can
    # return and reuse it for the next dispatch.
    libshmem_device.barrier_all_vec()


@triton.jit
def _kernel_local_topk_reduce(
    fc2_buf_ptr,
    route_to_send_ptr,
    output_ptr,
    batch_size,
    num_send,
    stride_fc2_m,
    stride_fc2_n,
    stride_output_m,
    stride_output_n,
    N: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N_REDUCE: tl.constexpr,
):
    """Reduce locally staged route rows after the pull kernel has exited."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)

    with al.scope(core_mode="vector", disable_auto_sync=True):
        # A pure AIV launch has one Vector core per program.  Unlike a mixed
        # Cube/Vector kernel, there is no second sub-vector worker to address
        # with sub_vec_id(); distribute tokens directly over the launch grid.
        reduce_cols = tl.arange(0, BLOCK_N_REDUCE)
        for token_id in range(pid, batch_size, ncore):
            token_id64 = token_id.to(tl.int64)
            for col_start in range(0, N, BLOCK_N_REDUCE):
                cols = col_start + reduce_cols
                mask_n = cols < N
                acc = tl.zeros((BLOCK_N_REDUCE,), dtype=tl.float32)
                for topk_slot in range(0, TOPK):
                    route_id = token_id * TOPK + topk_slot
                    send_row = tl.load(route_to_send_ptr + route_id)
                    valid = (send_row >= 0) & (send_row < num_send)
                    safe_row = tl.where(valid, send_row, 0).to(tl.int64)
                    values = tl.load(
                        fc2_buf_ptr
                        + safe_row * stride_fc2_m
                        + cols * stride_fc2_n,
                        mask=mask_n & valid,
                        other=0.0,
                    ).to(tl.float32)
                    acc += values
                tl.store(
                    output_ptr
                    + token_id64 * stride_output_m
                    + cols * stride_output_n,
                    acc.to(tl.bfloat16),
                    mask=mask_n,
                )


def build_route_to_send(send_route_idx: torch.Tensor, route_to_send: torch.Tensor) -> torch.Tensor:
    """Build flattened-route -> stable send-row mapping, using -1 for drops."""
    if send_route_idx.dtype != torch.int32 or route_to_send.dtype != torch.int32:
        raise TypeError("route indices and route_to_send must use torch.int32")
    if send_route_idx.ndim != 1 or route_to_send.ndim != 1:
        raise ValueError("route indices and route_to_send must both be 1D")
    if send_route_idx.device != route_to_send.device:
        raise ValueError("route indices and route_to_send must be on the same device")
    if not send_route_idx.is_contiguous() or not route_to_send.is_contiguous():
        raise ValueError("route indices and route_to_send must be contiguous")
    num_routes = route_to_send.numel()
    if send_route_idx.numel() > num_routes:
        raise ValueError("the valid send count cannot exceed the flattened route count")
    if num_routes:
        _fill_int_kernel[(triton.cdiv(num_routes, _ROUTE_BLOCK), )](
            route_to_send, num_routes, BLOCK=_ROUTE_BLOCK)
    num_send = send_route_idx.numel()
    if num_send:
        _scatter_route_to_send_kernel[(triton.cdiv(num_send, _ROUTE_BLOCK), )](
            send_route_idx, route_to_send, num_send, BLOCK=_ROUTE_BLOCK)
    return route_to_send


def prepare_fc2_combine_metadata(
    recv_counts_re: torch.Tensor,
    recv_expert_offs: torch.Tensor,
    counts_mem: torch.Tensor,
    send_bucket_starts: torch.Tensor,
    send_bucket_receive_offsets: torch.Tensor,
    fc2_tile_expert: torch.Tensor,
    fc2_tile_row_start: torch.Tensor,
    fc2_tile_row_count: torch.Tensor,
    reverse_tile_rank: torch.Tensor,
    reverse_tile_src_start: torch.Tensor,
    reverse_tile_dst_start: torch.Tensor,
    reverse_tile_row_count: torch.Tensor,
    num_fc2_slots: int,
    num_reverse_slots: int,
    *,
    local_rank: int,
    world_size: int,
    experts_per_rank: int,
    num_bins_pad: int,
    block_m: int,
    direct_pull: bool,
    build_fc2_tiles: bool = True,
) -> None:
    named_tensors = {
        "recv_counts_re": recv_counts_re,
        "recv_expert_offs": recv_expert_offs,
        "counts_mem": counts_mem,
        "send_bucket_starts": send_bucket_starts,
        "send_bucket_receive_offsets": send_bucket_receive_offsets,
        "fc2_tile_expert": fc2_tile_expert,
        "fc2_tile_row_start": fc2_tile_row_start,
        "fc2_tile_row_count": fc2_tile_row_count,
        "reverse_tile_rank": reverse_tile_rank,
        "reverse_tile_src_start": reverse_tile_src_start,
        "reverse_tile_dst_start": reverse_tile_dst_start,
        "reverse_tile_row_count": reverse_tile_row_count,
    }
    device = recv_counts_re.device
    for name, tensor in named_tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not 0 <= local_rank < world_size:
        raise ValueError("local_rank must be in [0, world_size)")
    if world_size <= 0 or experts_per_rank <= 0:
        raise ValueError("world_size and experts_per_rank must be positive")
    if num_bins_pad < world_size * experts_per_rank:
        raise ValueError("num_bins_pad is too small for the global expert buckets")
    if recv_counts_re.numel() < world_size * experts_per_rank:
        raise ValueError("recv_counts_re is smaller than [world_size, experts_per_rank]")
    if recv_expert_offs.numel() < experts_per_rank + 1:
        raise ValueError("recv_expert_offs must contain experts_per_rank + 1 entries")
    if counts_mem.numel() < world_size * num_bins_pad:
        raise ValueError("counts_mem is smaller than the padded count cube")
    num_experts = world_size * experts_per_rank
    if (
        send_bucket_starts.numel() < num_experts
        or send_bucket_receive_offsets.numel() < num_experts
    ):
        raise ValueError(
            "send bucket metadata must contain one entry per global expert"
        )
    for name, tensor in (
        ("fc2_tile_expert", fc2_tile_expert),
        ("fc2_tile_row_start", fc2_tile_row_start),
        ("fc2_tile_row_count", fc2_tile_row_count),
    ):
        if tensor.numel() < num_fc2_slots:
            raise ValueError(f"{name} has fewer than num_fc2_slots entries")
    for name, tensor in (
        ("reverse_tile_rank", reverse_tile_rank),
        ("reverse_tile_src_start", reverse_tile_src_start),
        ("reverse_tile_dst_start", reverse_tile_dst_start),
        ("reverse_tile_row_count", reverse_tile_row_count),
    ):
        if tensor.numel() < num_reverse_slots:
            raise ValueError(f"{name} has fewer than num_reverse_slots entries")
    if num_fc2_slots <= 0 or num_reverse_slots <= 0:
        raise ValueError("metadata slot counts must be positive")
    if block_m < 16 or block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two no smaller than 16")
    if not isinstance(direct_pull, bool):
        raise TypeError("direct_pull must be a bool")
    if not isinstance(build_fc2_tiles, bool):
        raise TypeError("build_fc2_tiles must be a bool")

    if build_fc2_tiles:
        _fill_int_kernel[(triton.cdiv(num_fc2_slots, _ROUTE_BLOCK), )](
            fc2_tile_expert,
            num_fc2_slots,
            BLOCK=_ROUTE_BLOCK,
        )
    _fill_int_kernel[(triton.cdiv(num_reverse_slots, _ROUTE_BLOCK), )](
        reverse_tile_rank,
        num_reverse_slots,
        BLOCK=_ROUTE_BLOCK,
    )

    _prepare_fc2_combine_metadata_kernel[(1, )](
        recv_counts_re,
        recv_expert_offs,
        counts_mem,
        send_bucket_starts,
        send_bucket_receive_offsets,
        fc2_tile_expert,
        fc2_tile_row_start,
        fc2_tile_row_count,
        reverse_tile_rank,
        reverse_tile_src_start,
        reverse_tile_dst_start,
        reverse_tile_row_count,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BINS_PAD=num_bins_pad,
        BLOCK_M=block_m,
        BUILD_FC2_TILES=build_fc2_tiles,
        DIRECT_PULL=direct_pull,
        multibuffer=True,
    )


def launch_fc2_combine(
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
    fc2_buf: torch.Tensor,
    peer_mem: torch.Tensor,
    route_to_send: torch.Tensor,
    output: torch.Tensor,
    received_routes_per_expert: torch.Tensor,
    received_expert_offsets: torch.Tensor,
    fc2_tile_expert: torch.Tensor,
    fc2_tile_row_start: torch.Tensor,
    fc2_tile_row_count: torch.Tensor,
    reverse_tile_rank: torch.Tensor,
    reverse_tile_src_start: torch.Tensor,
    reverse_tile_dst_start: torch.Tensor,
    reverse_tile_row_count: torch.Tensor,
    num_fc2_slots: int,
    num_reverse_slots: int,
    num_send: int,
    *,
    topk: int,
    num_cores: int,
    block_m: int,
    block_n: int,
    block_k: int,
    world_size: int,
    expert_n_persistent: bool,
    direct_pull: bool,
    reverse_vector_workers: int,
    reduce_vector_workers: int,
) -> torch.Tensor:
    """Launch FC2/combine with a logical ``[E, N, K]`` BF16 weight view."""
    if weighted_activation.ndim != 2 or down_weight.ndim != 3:
        raise ValueError("weighted_activation must be [M, K] and down_weight must be [E, N, K]")
    if weighted_activation.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise TypeError("weighted_activation and down_weight must use torch.bfloat16")
    if weighted_activation.shape[1] != down_weight.shape[2]:
        raise ValueError("weighted_activation K must match down_weight K")
    M, K = weighted_activation.shape
    N = down_weight.shape[1]
    required_fc2_buf_rows = num_send if direct_pull else M
    if (
        fc2_buf.ndim != 2
        or fc2_buf.shape[0] < required_fc2_buf_rows
        or fc2_buf.shape[1] != N
    ):
        raise ValueError(
            "fc2_buf must provide one row per sent route for direct pull, "
            "or one row per received route for reverse push"
        )
    if fc2_buf.dtype != torch.bfloat16:
        raise TypeError("fc2_buf must use torch.bfloat16")
    if peer_mem.ndim != 1 or peer_mem.dtype != torch.bfloat16:
        raise ValueError("peer_mem must be a flat BF16 symmetric tensor")
    if output.ndim != 2 or output.shape[1] != N or output.dtype != torch.bfloat16:
        raise ValueError("output must be a BF16 tensor shaped [tokens, N]")
    if route_to_send.ndim != 1 or route_to_send.dtype != torch.int32:
        raise ValueError("route_to_send must be a flat torch.int32 tensor")
    if route_to_send.numel() != output.shape[0] * topk:
        raise ValueError("route_to_send length must equal output tokens * topk")
    if not 0 <= num_send <= route_to_send.numel():
        raise ValueError("num_send must be in [0, route_to_send.numel()]")
    required_peer_rows = M if direct_pull else num_send
    if peer_mem.numel() < required_peer_rows * N:
        raise ValueError("peer_mem is too small for the FC2/combine rows")
    if direct_pull and fc2_buf.data_ptr() == peer_mem.data_ptr():
        raise ValueError(
            "bulk direct-pull workspace must not alias symmetric FC2 rows"
        )
    experts_per_rank = down_weight.shape[0]
    if experts_per_rank <= 0:
        raise ValueError("down_weight must contain at least one local expert")
    if (
        received_routes_per_expert.ndim != 1
        or received_routes_per_expert.numel() < experts_per_rank
    ):
        raise ValueError(
            "received_routes_per_expert must contain one entry per local expert"
        )
    if (
        received_expert_offsets.ndim != 1
        or received_expert_offsets.numel() < experts_per_rank + 1
    ):
        raise ValueError(
            "received_expert_offsets must contain experts_per_rank + 1 entries"
        )
    tensors = (
        weighted_activation,
        down_weight,
        fc2_buf,
        peer_mem,
        route_to_send,
        output,
        received_routes_per_expert,
        received_expert_offsets,
        fc2_tile_expert,
        fc2_tile_row_start,
        fc2_tile_row_count,
        reverse_tile_rank,
        reverse_tile_src_start,
        reverse_tile_dst_start,
        reverse_tile_row_count,
    )
    if any(tensor.device != weighted_activation.device for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be contiguous")
    metadata_tensors = (
        route_to_send,
        received_routes_per_expert,
        received_expert_offsets,
        fc2_tile_expert,
        fc2_tile_row_start,
        fc2_tile_row_count,
        reverse_tile_rank,
        reverse_tile_src_start,
        reverse_tile_dst_start,
        reverse_tile_row_count,
    )
    for tensor in metadata_tensors:
        if tensor.dtype != torch.int32:
            raise TypeError("all FC2/combine metadata tensors must use torch.int32")
    if any(
        tensor.numel() < num_fc2_slots
        for tensor in (
            fc2_tile_expert,
            fc2_tile_row_start,
            fc2_tile_row_count,
        )
    ):
        raise ValueError("FC2 metadata workspace is smaller than num_fc2_slots")
    if any(
        tensor.numel() < num_reverse_slots
        for tensor in (
            reverse_tile_rank,
            reverse_tile_src_start,
            reverse_tile_dst_start,
            reverse_tile_row_count,
        )
    ):
        raise ValueError("reverse metadata workspace is smaller than num_reverse_slots")
    if topk <= 0 or num_cores <= 0:
        raise ValueError("topk and num_cores must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not isinstance(expert_n_persistent, bool):
        raise TypeError("expert_n_persistent must be a bool")
    if not isinstance(direct_pull, bool):
        raise TypeError("direct_pull must be a bool")
    if (
        type(reverse_vector_workers) is not int
        or reverse_vector_workers not in (1, 2)
    ):
        raise ValueError("reverse_vector_workers must be 1 or 2")
    if (
        type(reduce_vector_workers) is not int
        or reduce_vector_workers not in (1, 2)
    ):
        raise ValueError("reduce_vector_workers must be 1 or 2")
    for name, block in (
        ("block_m", block_m),
        ("block_n", block_n),
        ("block_k", block_k),
    ):
        if block < 16 or block & (block - 1):
            raise ValueError(f"{name} must be a power of two no smaller than 16")
    if N % block_n or K % block_k:
        raise ValueError(
            "FC2 block_n and block_k must divide N and K on the current Ascend backend"
        )

    push_width = 512 if weighted_activation.shape[0] >= 1024 else 256
    _kernel_fc2_combine[(num_cores, 1, 1)](
        weighted_activation,
        down_weight,
        fc2_buf,
        peer_mem,
        route_to_send,
        output,
        received_routes_per_expert,
        received_expert_offsets,
        fc2_tile_expert,
        fc2_tile_row_start,
        fc2_tile_row_count,
        reverse_tile_rank,
        reverse_tile_src_start,
        reverse_tile_dst_start,
        reverse_tile_row_count,
        num_fc2_slots,
        num_reverse_slots,
        output.shape[0],
        num_send,
        weighted_activation.stride(0),
        weighted_activation.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        down_weight.stride(2),
        fc2_buf.stride(0),
        fc2_buf.stride(1),
        output.stride(0),
        output.stride(1),
        N=down_weight.shape[1],
        K=down_weight.shape[2],
        # Direct pull runs reduction in the third launch below.  Normalize the
        # constexprs used only by the reverse-push branch so unrelated tuning
        # choices do not create duplicate Cube binaries.  The real top-k and
        # worker settings remain active in reverse push and local reduction.
        TOPK=(1 if direct_pull else topk),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_N_PUSH=(1 if direct_pull else push_width),
        BLOCK_N_REDUCE=(1 if direct_pull else push_width),
        EXPERTS_PER_RANK=experts_per_rank,
        FC2_EXPERT_N_PERSISTENT=expert_n_persistent,
        DIRECT_PULL=direct_pull,
        # Direct pull uses one owner program per peer in its separate kernel;
        # reverse-push worker count must not create dead direct-pull variants.
        REVERSE_VECTOR_WORKERS=(
            1 if direct_pull else reverse_vector_workers
        ),
        REDUCE_VECTOR_WORKERS=(1 if direct_pull else reduce_vector_workers),
    )
    if direct_pull:
        _kernel_direct_pull_transport[(num_cores, 1, 1)](
            fc2_buf,
            peer_mem,
            reverse_tile_rank,
            reverse_tile_src_start,
            reverse_tile_dst_start,
            reverse_tile_row_count,
            num_reverse_slots,
            fc2_buf.stride(0),
            N=down_weight.shape[1],
            WORLD_SIZE=world_size,
        )
        # Keep reduction in a separate launch.  Blocking getmem only fences the
        # Vector core that issued it; the kernel boundary makes every pulled GM
        # row visible before all Vector sub-cores consume the local workspace.
        _kernel_local_topk_reduce[
            (num_cores * reduce_vector_workers, 1, 1)
        ](
            fc2_buf,
            route_to_send,
            output,
            output.shape[0],
            num_send,
            fc2_buf.stride(0),
            fc2_buf.stride(1),
            output.stride(0),
            output.stride(1),
            N=down_weight.shape[1],
            TOPK=topk,
            BLOCK_N_REDUCE=push_width,
        )
    return output


__all__ = [
    "build_route_to_send",
    "prepare_fc2_combine_metadata",
    "launch_fc2_combine",
]
