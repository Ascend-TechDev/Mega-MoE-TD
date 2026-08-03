# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""BF16 Ascend Triton kernel for MoE FC2 and reverse combine.

The input rows are already grouped as local-expert major, then source-rank
major.  Routing weights were applied by weighted SwiGLU before this stage.
This module therefore performs only FC2, the reverse all-to-all, and the final
top-k reduction.
"""

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
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
    recv_counts_re_ptr,
    recv_expert_offs_ptr,
    counts_mem_ptr,
    fc2_tile_expert_ptr,
    fc2_tile_row_start_ptr,
    fc2_tile_row_count_ptr,
    reverse_tile_rank_ptr,
    reverse_tile_src_start_ptr,
    reverse_tile_dst_start_ptr,
    reverse_tile_row_count_ptr,
    num_fc2_slots,
    num_reverse_slots,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    META_BLOCK: tl.constexpr,
):
    """Build compact FC2 and reverse-A2A tile descriptions on one Vector core."""
    meta_offs = tl.arange(0, META_BLOCK)
    for start in range(0, num_fc2_slots, META_BLOCK):
        offs = start + meta_offs
        tl.store(fc2_tile_expert_ptr + offs, -1, mask=offs < num_fc2_slots)
    for start in range(0, num_reverse_slots, META_BLOCK):
        offs = start + meta_offs
        tl.store(reverse_tile_rank_ptr + offs, -1, mask=offs < num_reverse_slots)

    fc2_cursor = 0
    for expert_id in range(0, EXPERTS_PER_RANK):
        expert_count = tl.load(recv_counts_re_ptr + expert_id)
        # recv_counts_re is [source, expert]; sum all sources for this expert.
        for source_rank in range(1, WORLD_SIZE):
            expert_count += tl.load(
                recv_counts_re_ptr + source_rank * EXPERTS_PER_RANK + expert_id)
        expert_start = tl.load(recv_expert_offs_ptr + expert_id)
        num_tiles = tl.cdiv(expert_count, BLOCK_M)
        for tile_id in range(0, num_tiles):
            row_start = expert_start + tile_id * BLOCK_M
            row_count = tl.minimum(BLOCK_M, expert_count - tile_id * BLOCK_M)
            tl.store(fc2_tile_expert_ptr + fc2_cursor, expert_id)
            tl.store(fc2_tile_row_start_ptr + fc2_cursor, row_start)
            tl.store(fc2_tile_row_count_ptr + fc2_cursor, row_count)
            fc2_cursor += 1

    reverse_cursor = 0
    for source_rank in range(0, WORLD_SIZE):
        # On the home/source rank, returned rows use the same stable global-
        # expert order as its original send list.  Compute the prefix preceding
        # this destination rank once, then advance across local experts.
        remote_bucket_start = 0
        for prior_bucket in range(0, LOCAL_RANK * EXPERTS_PER_RANK):
            remote_bucket_start += tl.load(
                counts_mem_ptr + source_rank * NUM_BINS_PAD + prior_bucket)

        for expert_id in range(0, EXPERTS_PER_RANK):
            source_count = tl.load(
                recv_counts_re_ptr + source_rank * EXPERTS_PER_RANK + expert_id)
            source_row_start = tl.load(recv_expert_offs_ptr + expert_id)
            for prior_source in range(0, WORLD_SIZE):
                if prior_source < source_rank:
                    source_row_start += tl.load(
                        recv_counts_re_ptr + prior_source * EXPERTS_PER_RANK + expert_id)

            num_tiles = tl.cdiv(source_count, BLOCK_M)
            for tile_id in range(0, num_tiles):
                tile_delta = tile_id * BLOCK_M
                row_count = tl.minimum(BLOCK_M, source_count - tile_delta)
                tl.store(reverse_tile_rank_ptr + reverse_cursor, source_rank)
                tl.store(reverse_tile_src_start_ptr + reverse_cursor, source_row_start + tile_delta)
                tl.store(reverse_tile_dst_start_ptr + reverse_cursor, remote_bucket_start + tile_delta)
                tl.store(reverse_tile_row_count_ptr + reverse_cursor, row_count)
                reverse_cursor += 1

            remote_bucket_start += source_count


@triton.jit
def _kernel_fc2_combine(
    input_ptr,
    weight_ptr,
    fc2_buf_ptr,
    peer_mem_ptr,
    route_to_send_ptr,
    output_ptr,
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
):
    """FC2 -> reverse symmetric write -> top-k reduction."""
    pid = tl.program_id(0)
    ncore = tl.num_programs(0)

    # Phase 1: BF16 FC2 with FP32 accumulation.
    with al.scope(core_mode="cube", disable_auto_sync=True):
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        num_n_tiles: tl.constexpr = tl.cdiv(N, BLOCK_N)
        total_fc2_tasks = num_fc2_slots * num_n_tiles
        for task_id in range(pid, total_fc2_tasks, ncore):
            tile_id = task_id % num_fc2_slots
            n_tile = task_id // num_fc2_slots
            expert_id = tl.load(fc2_tile_expert_ptr + tile_id)
            if expert_id >= 0:
                row_start = tl.load(fc2_tile_row_start_ptr + tile_id)
                row_count = tl.load(fc2_tile_row_count_ptr + tile_id)
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
                    a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
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

    # Every rank must finish FC2 before any reverse-A2A reader starts.
    libshmem_device.barrier_all()

    # Phase 2: reverse A2A.  One Vector sub-core owns each remote store.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            push_cols = tl.arange(0, BLOCK_N_PUSH)
            for tile_id in range(pid, num_reverse_slots, ncore):
                dst_rank = tl.load(reverse_tile_rank_ptr + tile_id)
                if dst_rank >= 0:
                    src_start = tl.load(reverse_tile_src_start_ptr + tile_id).to(tl.int64)
                    dst_start = tl.load(reverse_tile_dst_start_ptr + tile_id).to(tl.int64)
                    row_count = tl.load(reverse_tile_row_count_ptr + tile_id)
                    remote_peer = dl.symm_at(peer_mem_ptr, dst_rank)
                    for row_delta in range(0, row_count):
                        for col_start in range(0, N, BLOCK_N_PUSH):
                            cols = col_start + push_cols
                            mask = cols < N
                            values = tl.load(
                                fc2_buf_ptr + (src_start + row_delta) * stride_fc2_m + cols * stride_fc2_n,
                                mask=mask,
                                other=0.0,
                            )
                            tl.store(
                                remote_peer + (dst_start + row_delta) * N + cols,
                                values,
                                mask=mask,
                            )

    # Remote rows must be visible before their home rank reduces them.
    libshmem_device.barrier_all()

    # Phase 3: restore flattened route slots and reduce top-k in FP32.
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
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
                            peer_mem_ptr + safe_row * N + cols,
                            mask=mask_n & valid,
                            other=0.0,
                        ).to(tl.float32)
                        acc += values
                    tl.store(
                        output_ptr + token_id64 * stride_output_m + cols * stride_output_n,
                        acc.to(tl.bfloat16),
                        mask=mask_n,
                    )

    # The next full forward reuses peer_mem for dispatch.  Do not let a faster
    # rank overwrite a slower rank's reduction input.
    libshmem_device.barrier_all()


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
            route_to_send, num_routes, BLOCK=_ROUTE_BLOCK, use_bytecode=True)
    num_send = send_route_idx.numel()
    if num_send:
        _scatter_route_to_send_kernel[(triton.cdiv(num_send, _ROUTE_BLOCK), )](
            send_route_idx, route_to_send, num_send, BLOCK=_ROUTE_BLOCK, use_bytecode=True)
    return route_to_send


def prepare_fc2_combine_metadata(
    recv_counts_re: torch.Tensor,
    recv_expert_offs: torch.Tensor,
    counts_mem: torch.Tensor,
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
) -> None:
    named_tensors = {
        "recv_counts_re": recv_counts_re,
        "recv_expert_offs": recv_expert_offs,
        "counts_mem": counts_mem,
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

    _prepare_fc2_combine_metadata_kernel[(1, )](
        recv_counts_re,
        recv_expert_offs,
        counts_mem,
        fc2_tile_expert,
        fc2_tile_row_start,
        fc2_tile_row_count,
        reverse_tile_rank,
        reverse_tile_src_start,
        reverse_tile_dst_start,
        reverse_tile_row_count,
        num_fc2_slots,
        num_reverse_slots,
        LOCAL_RANK=local_rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BINS_PAD=num_bins_pad,
        BLOCK_M=block_m,
        META_BLOCK=_META_BLOCK,
        use_bytecode=True,
    )


def launch_fc2_combine(
    weighted_activation: torch.Tensor,
    down_weight: torch.Tensor,
    fc2_buf: torch.Tensor,
    peer_mem: torch.Tensor,
    route_to_send: torch.Tensor,
    output: torch.Tensor,
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
) -> torch.Tensor:
    """Launch the BF16-only production FC2/combine kernel."""
    if weighted_activation.ndim != 2 or down_weight.ndim != 3:
        raise ValueError("weighted_activation must be [M, K] and down_weight must be [E, N, K]")
    if weighted_activation.dtype != torch.bfloat16 or down_weight.dtype != torch.bfloat16:
        raise TypeError("weighted_activation and down_weight must use torch.bfloat16")
    if weighted_activation.shape[1] != down_weight.shape[2]:
        raise ValueError("weighted_activation K must match down_weight K")
    M, K = weighted_activation.shape
    N = down_weight.shape[1]
    if fc2_buf.ndim != 2 or fc2_buf.shape[0] < M or fc2_buf.shape[1] != N:
        raise ValueError("fc2_buf must provide at least [M, N] BF16 elements")
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
    if peer_mem.numel() < num_send * N:
        raise ValueError("peer_mem is too small for the returned send rows")

    tensors = (
        weighted_activation,
        down_weight,
        fc2_buf,
        peer_mem,
        route_to_send,
        output,
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
    for tensor in tensors[6:]:
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
        TOPK=topk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCK_N_PUSH=push_width,
        BLOCK_N_REDUCE=push_width,
        use_bytecode=True,
    )
    return output


__all__ = [
    "build_route_to_send",
    "prepare_fc2_combine_metadata",
    "launch_fc2_combine",
]
