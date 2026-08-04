# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stable route filtering, count exchange, and typed dispatch metadata."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device

from .workspace import MoEForwardContext


@dataclass
class MoERoutingPlan:
    """Single-in-flight routing metadata shared by dispatch and combine.

    Invalid expert ids are represented by ``valid_route_mask`` and are omitted
    from all communication.  The remaining routes are stable-sorted by global
    expert id, preserving the exact inverse-order contract used by combine.
    """

    num_input_tokens: int
    num_received_routes: int
    selected_experts: torch.Tensor
    send_token_indices: torch.Tensor
    send_bucket_starts: torch.Tensor
    send_bucket_receive_offsets: torch.Tensor
    send_counts_by_rank_expert: torch.Tensor
    num_sent_routes: int
    received_routes_per_expert: torch.Tensor
    received_expert_offsets: torch.Tensor
    receive_counts_by_source_expert: torch.Tensor
    stable_sort_indices: torch.Tensor
    valid_route_mask: torch.Tensor


@triton.jit(do_not_specialize=["num_valid"])
def _kernel_build_routing_metadata(
    sorted_key_ptr,
    counts_mem_ptr,
    send_bucket_starts_ptr,
    send_bucket_dst_starts_ptr,
    recv_counts_re_ptr,
    recv_per_expert_ptr,
    recv_expert_offs_ptr,
    stats_ptr,
    num_valid,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    NUM_BINS_PAD: tl.constexpr,
    HISTOGRAM_BLOCK_SIZE: tl.constexpr,
):
    """Build compact dispatch metadata on one Vector AI core per rank."""
    bin_offs = tl.arange(0, NUM_BINS_PAD)
    local_counts = tl.zeros((NUM_BINS_PAD,), dtype=tl.int32)
    invalid_lanes = 0

    # The Ascend histogram lowering used here does not reliably support its
    # optional mask.  Padding lanes map to bin zero and are subtracted once.
    for route_start in range(0, num_valid, HISTOGRAM_BLOCK_SIZE):
        route_offs = route_start + tl.arange(0, HISTOGRAM_BLOCK_SIZE)
        route_mask = route_offs < num_valid
        route_keys = tl.load(
            sorted_key_ptr + route_offs, mask=route_mask, other=0
        ).to(tl.int32)
        local_counts += tl.histogram(route_keys, NUM_BINS_PAD)
        invalid_lanes += HISTOGRAM_BLOCK_SIZE - tl.sum(route_mask.to(tl.int32))
    local_counts -= tl.where(bin_offs == 0, invalid_lanes, 0)

    local_row_ptr = counts_mem_ptr + LOCAL_RANK * NUM_BINS_PAD
    tl.store(local_row_ptr + bin_offs, local_counts)
    for peer_rank in range(WORLD_SIZE):
        if peer_rank != LOCAL_RANK:
            libshmem_device.putmem(
                local_row_ptr,
                local_row_ptr,
                NUM_BINS_PAD * 4,
                peer_rank,
            )

    # This metadata kernel is AIV-only.  On the current CANN/Triton stack the
    # generic barrier waits for Cube-side participants and deadlocks; use the
    # Vector-only collective so every rank has the same participating domain.
    libshmem_device.barrier_all_vec()

    send_running = 0
    recv_running = 0
    max_required = 0
    for dst_rank in range(WORLD_SIZE):
        expert_base = 0
        required_for_dst = 0
        for local_expert in range(EXPERTS_PER_RANK):
            bucket = dst_rank * EXPERTS_PER_RANK + local_expert
            local_count = tl.load(local_row_ptr + bucket)
            tl.store(send_bucket_starts_ptr + bucket, send_running)

            target_total = 0
            source_prefix = 0
            for source_rank in range(WORLD_SIZE):
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

            tl.store(
                send_bucket_dst_starts_ptr + bucket,
                expert_base + source_prefix,
            )

            if dst_rank == LOCAL_RANK:
                tl.store(recv_per_expert_ptr + local_expert, target_total)
                tl.store(recv_expert_offs_ptr + local_expert, recv_running)
                recv_running += target_total

            send_running += local_count
            expert_base += target_total
            required_for_dst += target_total

        max_required = tl.maximum(max_required, required_for_dst)

    tl.store(recv_expert_offs_ptr + EXPERTS_PER_RANK, recv_running)
    tl.store(stats_ptr, recv_running)
    tl.store(stats_ptr + 1, max_required)

    # Protect the shared count cube from a faster rank's next invocation.
    libshmem_device.barrier_all_vec()


def build_routing_plan(
    context: MoEForwardContext,
    selected_experts: torch.Tensor,
) -> MoERoutingPlan:
    """Build one stable, compact routing plan for dispatch and combine."""
    world_size = context.world_size
    experts_per_rank = context.experts_per_rank
    num_experts = context.num_experts
    flat_experts = selected_experts.reshape(-1)
    valid_route_mask = (flat_experts >= 0) & (flat_experts < num_experts)
    num_valid = int(valid_route_mask.sum().item())
    row_tokens = context.row_token_indices[:flat_experts.numel()]
    if num_valid == flat_experts.numel():
        kept_experts = flat_experts
        kept_token_indices = row_tokens
    else:
        kept_experts = flat_experts[valid_route_mask]
        kept_token_indices = row_tokens[valid_route_mask]

    # Float32 avoids the Ascend integer ArgSort AICPU fallback and exactly
    # represents the small expert-id range.
    stable_sort_indices = torch.argsort(
        kept_experts.to(torch.float32), stable=True
    )
    sorted_experts = kept_experts[stable_sort_indices].contiguous()
    send_token_indices = kept_token_indices[stable_sort_indices].to(
        torch.int32
    ).contiguous()

    _kernel_build_routing_metadata[(1, 1, 1)](
        sorted_experts,
        context.metadata_counts_mem,
        context.metadata_send_bucket_starts,
        context.metadata_send_bucket_dst_starts,
        context.metadata_recv_counts_re,
        context.metadata_recv_per_expert,
        context.metadata_recv_expert_offs,
        context.metadata_stats,
        num_valid,
        LOCAL_RANK=context.rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
        NUM_BUCKETS=num_experts,
        NUM_BINS_PAD=context.metadata_num_bins,
        HISTOGRAM_BLOCK_SIZE=2048,
    )

    num_received_routes = int(context.metadata_stats[0].item())
    max_received_routes = context.peer_mem.numel() // context.hidden_size
    if max_received_routes >= torch.iinfo(torch.int32).max:
        raise ValueError("dispatch receive offsets exceed int32 range")
    if context.receive_capacity_factor < world_size:
        required_received_routes = int(context.metadata_stats[1].item())
        if required_received_routes > max_received_routes:
            raise ValueError(
                f"peer buffer capacity {max_received_routes} routes is smaller than "
                f"the required receive size {required_received_routes}"
            )

    local_count_start = context.rank * context.metadata_num_bins
    send_counts = context.metadata_counts_mem[
        local_count_start:local_count_start + num_experts
    ].view(world_size, experts_per_rank)
    return MoERoutingPlan(
        num_input_tokens=selected_experts.shape[0],
        num_received_routes=num_received_routes,
        selected_experts=selected_experts,
        send_token_indices=send_token_indices,
        send_bucket_starts=context.metadata_send_bucket_starts,
        send_bucket_receive_offsets=context.metadata_send_bucket_dst_starts,
        send_counts_by_rank_expert=send_counts,
        num_sent_routes=num_valid,
        received_routes_per_expert=context.metadata_recv_per_expert,
        received_expert_offsets=context.metadata_recv_expert_offs,
        receive_counts_by_source_expert=context.metadata_recv_counts_re,
        stable_sort_indices=stable_sort_indices,
        valid_route_mask=valid_route_mask,
    )


__all__ = ["MoERoutingPlan", "build_routing_plan"]
