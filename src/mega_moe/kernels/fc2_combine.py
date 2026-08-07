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
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id


_META_BLOCK = 256
_ROUTE_BLOCK = 256


@triton.jit
def _fill_route_to_send_kernel(route_to_send_ptr, num_routes, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(route_to_send_ptr + offs, -1, mask=offs < num_routes)


@triton.jit
def _scatter_route_to_send_kernel(send_route_idx_ptr, route_to_send_ptr, num_send, BLOCK: tl.constexpr):
    sorted_offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = sorted_offs < num_send
    route_ids = tl.load(send_route_idx_ptr + sorted_offs, mask=mask, other=0)
    tl.store(route_to_send_ptr + route_ids, sorted_offs, mask=mask)


@triton.jit
def _prepare_fc2_combine_metadata_kernel(
    counts_mem_ptr,
    send_bucket_starts_ptr,
    send_bucket_receive_offsets_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    num_pull_slots,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    META_BLOCK: tl.constexpr,
):
    """Build compact descriptors for pulling remote FC2 bucket rows."""
    meta_offs = tl.arange(0, META_BLOCK)
    for start in range(0, num_pull_slots, META_BLOCK):
        offs = start + meta_offs
        tl.store(pull_tile_rank_ptr + offs, -1, mask=offs < num_pull_slots)

    transport_cursor = 0
    # On the home/source rank, every stable-send bucket maps one-to-one to a
    # contiguous source segment in the destination rank's expert-major FC2
    # output.  Pull those segments into local stable-send order so the later
    # top-k loop performs local loads only.
    local_count_row = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    for bucket in range(0, WORLD_SIZE * EXPERTS_PER_RANK):
        route_count = tl.load(local_count_row + bucket)
        local_send_start = tl.load(send_bucket_starts_ptr + bucket)
        remote_receive_start = tl.load(send_bucket_receive_offsets_ptr + bucket)
        destination_rank = bucket // EXPERTS_PER_RANK
        num_tiles = tl.cdiv(route_count, BLOCK_M)
        for tile_id in range(0, num_tiles):
            tile_delta = tile_id * BLOCK_M
            row_count = tl.minimum(BLOCK_M, route_count - tile_delta)
            tl.store(pull_tile_rank_ptr + transport_cursor, destination_rank)
            tl.store(
                pull_tile_src_start_ptr + transport_cursor,
                remote_receive_start + tile_delta,
            )
            tl.store(
                pull_tile_dst_start_ptr + transport_cursor,
                local_send_start + tile_delta,
            )
            tl.store(pull_tile_row_count_ptr + transport_cursor, row_count)
            transport_cursor += 1


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
    """Compute one FC2 M/N tile for the expert/N-persistent schedule."""
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
def _kernel_fc2_expert_n_persistent(
    input_ptr,
    weight_ptr,
    peer_mem_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stride_input_m,
    stride_input_k,
    stride_weight_e,
    stride_weight_n,
    stride_weight_k,
    stride_peer_m,
    stride_peer_n,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Expert/N-persistent FC2, writing expert-major rows to symmetric GM."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)

    with al.scope(core_mode="cube", disable_auto_sync=True):
        # ``launch_fc2_combine`` requires BLOCK_N to divide N.  Keep this as
        # pure constexpr arithmetic: the Ascend frontend materializes
        # ``tl.cdiv`` as a semantic value, which cannot then be multiplied by
        # the constexpr expert count in the persistent schedule.
        num_n_tiles: tl.constexpr = N // BLOCK_N
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
                        stride_peer_m,
                        stride_peer_n,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                    )


@triton.jit
def _kernel_direct_pull_transport(
    fc2_buf_ptr,
    peer_mem_ptr,
    pull_tile_rank_ptr,
    pull_tile_src_start_ptr,
    pull_tile_dst_start_ptr,
    pull_tile_row_count_ptr,
    num_pull_slots,
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
                for tile_id in range(0, num_pull_slots):
                    peer_rank = tl.load(pull_tile_rank_ptr + tile_id)
                    if peer_rank == peer_owner:
                        src_start = tl.load(
                            pull_tile_src_start_ptr + tile_id
                        ).to(tl.int64)
                        dst_start = tl.load(
                            pull_tile_dst_start_ptr + tile_id
                        ).to(tl.int64)
                        row_count = tl.load(
                            pull_tile_row_count_ptr + tile_id
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
        _fill_route_to_send_kernel[(triton.cdiv(num_routes, _ROUTE_BLOCK), )](
            route_to_send, num_routes, BLOCK=_ROUTE_BLOCK)
    num_send = send_route_idx.numel()
    if num_send:
        _scatter_route_to_send_kernel[(triton.cdiv(num_send, _ROUTE_BLOCK), )](
            send_route_idx, route_to_send, num_send, BLOCK=_ROUTE_BLOCK)
    return route_to_send


def prepare_fc2_combine_metadata(
    counts_mem: torch.Tensor,
    send_bucket_starts: torch.Tensor,
    send_bucket_receive_offsets: torch.Tensor,
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    num_pull_slots: int,
    *,
    local_rank: int,
    world_size: int,
    experts_per_rank: int,
    num_bins_pad: int,
    block_m: int,
) -> None:
    named_tensors = {
        "counts_mem": counts_mem,
        "send_bucket_starts": send_bucket_starts,
        "send_bucket_receive_offsets": send_bucket_receive_offsets,
        "pull_tile_rank": pull_tile_rank,
        "pull_tile_src_start": pull_tile_src_start,
        "pull_tile_dst_start": pull_tile_dst_start,
        "pull_tile_row_count": pull_tile_row_count,
    }
    device = counts_mem.device
    for name, tensor in named_tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if world_size <= 0 or experts_per_rank <= 0:
        raise ValueError("world_size and experts_per_rank must be positive")
    if not 0 <= local_rank < world_size:
        raise ValueError("local_rank must be in [0, world_size)")
    if num_bins_pad < world_size * experts_per_rank:
        raise ValueError("num_bins_pad is too small for the global expert buckets")
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
        ("pull_tile_rank", pull_tile_rank),
        ("pull_tile_src_start", pull_tile_src_start),
        ("pull_tile_dst_start", pull_tile_dst_start),
        ("pull_tile_row_count", pull_tile_row_count),
    ):
        if tensor.numel() < num_pull_slots:
            raise ValueError(f"{name} has fewer than num_pull_slots entries")
    if num_pull_slots <= 0:
        raise ValueError("num_pull_slots must be positive")
    if block_m < 16 or block_m & (block_m - 1):
        raise ValueError("block_m must be a power of two no smaller than 16")

    _prepare_fc2_combine_metadata_kernel[(1, )](
        counts_mem,
        send_bucket_starts,
        send_bucket_receive_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        num_pull_slots,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BINS_PAD=num_bins_pad,
        BLOCK_M=block_m,
        META_BLOCK=_META_BLOCK,
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
    pull_tile_rank: torch.Tensor,
    pull_tile_src_start: torch.Tensor,
    pull_tile_dst_start: torch.Tensor,
    pull_tile_row_count: torch.Tensor,
    num_pull_slots: int,
    num_send: int,
    *,
    topk: int,
    num_program_cores: int,
    block_m: int,
    block_n: int,
    block_k: int,
    world_size: int,
) -> torch.Tensor:
    """Launch expert/N-persistent FC2, direct pull, then two-AIV reduction."""
    if weighted_activation.ndim != 2 or down_weight.ndim != 3:
        raise ValueError("weighted_activation must be [M, K] and down_weight must be [E, N, K]")
    if weighted_activation.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise TypeError("weighted_activation and down_weight must use torch.bfloat16")
    if weighted_activation.shape[1] != down_weight.shape[2]:
        raise ValueError("weighted_activation K must match down_weight K")
    M, K = weighted_activation.shape
    N = down_weight.shape[1]
    if (
        fc2_buf.ndim != 2
        or fc2_buf.shape[0] < num_send
        or fc2_buf.shape[1] != N
    ):
        raise ValueError("fc2_buf must provide one row per sent route")
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
    if peer_mem.numel() < M * N:
        raise ValueError("peer_mem is too small for the symmetric FC2 rows")
    if fc2_buf.data_ptr() == peer_mem.data_ptr():
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
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
    )
    if any(tensor.device != weighted_activation.device for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all FC2/combine tensors must be contiguous")
    metadata_tensors = (
        route_to_send,
        received_routes_per_expert,
        received_expert_offsets,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
    )
    for tensor in metadata_tensors:
        if tensor.dtype != torch.int32:
            raise TypeError("all FC2/combine metadata tensors must use torch.int32")
    if any(
        tensor.numel() < num_pull_slots
        for tensor in (
            pull_tile_rank,
            pull_tile_src_start,
            pull_tile_dst_start,
            pull_tile_row_count,
        )
    ):
        raise ValueError("pull metadata workspace is smaller than num_pull_slots")
    if num_pull_slots <= 0:
        raise ValueError("num_pull_slots must be positive")
    if topk <= 0 or num_program_cores <= 0:
        raise ValueError("topk and num_program_cores must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
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

    reduce_width = 512 if weighted_activation.shape[0] >= 1024 else 256
    _kernel_fc2_expert_n_persistent[(num_program_cores, 1, 1)](
        weighted_activation,
        down_weight,
        peer_mem,
        received_routes_per_expert,
        received_expert_offsets,
        weighted_activation.stride(0),
        weighted_activation.stride(1),
        down_weight.stride(0),
        down_weight.stride(1),
        down_weight.stride(2),
        fc2_buf.stride(0),
        fc2_buf.stride(1),
        N=down_weight.shape[1],
        K=down_weight.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EXPERTS_PER_RANK=experts_per_rank,
    )
    _kernel_direct_pull_transport[(num_program_cores, 1, 1)](
        fc2_buf,
        peer_mem,
        pull_tile_rank,
        pull_tile_src_start,
        pull_tile_dst_start,
        pull_tile_row_count,
        num_pull_slots,
        fc2_buf.stride(0),
        N=down_weight.shape[1],
        WORLD_SIZE=world_size,
    )
    # Blocking getmem only fences the Vector core that issued it; the kernel
    # boundary makes every pulled GM row visible before both AIV sub-cores per
    # AI Core consume the local workspace.  Derive the 2-AIV grid from the
    # caller-provided AI Core program count so this remains portable.
    _kernel_local_topk_reduce[(num_program_cores * 2, 1, 1)](
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
        BLOCK_N_REDUCE=reduce_width,
    )
    return output


__all__ = [
    "build_route_to_send",
    "prepare_fc2_combine_metadata",
    "launch_fc2_combine",
]
