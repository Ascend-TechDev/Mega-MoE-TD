# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""ACLSHMEM buffers owned by one standalone Mega-MoE forward instance."""

from dataclasses import dataclass
from typing import Optional

import torch
import triton


@dataclass
class MoEForwardContext:
    """Single-in-flight symmetric buffers and reusable routing workspaces.

    A routing plan must be consumed before the next plan is built because the
    device metadata tensors are intentionally reused in place.
    """

    hidden_size: int
    num_experts: int
    experts_per_rank: int
    world_size: int
    rank: int
    receive_capacity_factor: float
    max_source_tiles: int

    peer_mem: Optional[torch.Tensor] = None
    routing_weight_mem: Optional[torch.Tensor] = None
    signal_mem: Optional[torch.Tensor] = None
    row_token_indices: Optional[torch.Tensor] = None
    row_route_indices: Optional[torch.Tensor] = None
    metadata_counts_mem: Optional[torch.Tensor] = None
    metadata_num_bins: int = 0
    metadata_send_bucket_starts: Optional[torch.Tensor] = None
    metadata_send_bucket_dst_starts: Optional[torch.Tensor] = None
    metadata_recv_counts_re: Optional[torch.Tensor] = None
    metadata_recv_per_expert: Optional[torch.Tensor] = None
    metadata_recv_expert_offs: Optional[torch.Tensor] = None
    metadata_stats: Optional[torch.Tensor] = None

    def finalize(self):
        """Release symmetric allocations and drop ordinary workspaces."""
        import shmem as ash

        if self.peer_mem is not None:
            ash.aclshmem_free_tensor(self.peer_mem)
            self.peer_mem = None
        if self.routing_weight_mem is not None:
            ash.aclshmem_free_tensor(self.routing_weight_mem)
            self.routing_weight_mem = None
        if self.signal_mem is not None:
            ash.aclshmem_free_tensor(self.signal_mem)
            self.signal_mem = None
        if self.metadata_counts_mem is not None:
            ash.aclshmem_free_tensor(self.metadata_counts_mem)
            self.metadata_counts_mem = None
        self.row_token_indices = None
        self.row_route_indices = None
        self.metadata_send_bucket_starts = None
        self.metadata_send_bucket_dst_starts = None
        self.metadata_recv_counts_re = None
        self.metadata_recv_per_expert = None
        self.metadata_recv_expert_offs = None
        self.metadata_stats = None


def create_moe_forward_context(
    *,
    max_tokens_per_rank: int,
    hidden_size: int,
    top_k: int,
    num_experts: int,
    rank: int,
    world_size: int,
    receive_capacity_factor: float,
    dispatch_fc1_block_size_m: int,
) -> MoEForwardContext:
    """Allocate BF16 token buffers and an FP32 routing-weight buffer."""
    import shmem as ash

    ash_rank = ash.my_pe()
    ash_world_size = ash.pe_count()
    if ash_rank != rank or ash_world_size != world_size:
        raise ValueError(
            "Ascend Mega-MoE requires the EP process group to match the complete "
            "ACLSHMEM world"
        )

    experts_per_rank = num_experts // world_size
    max_received_routes = int(max_tokens_per_rank * top_k * receive_capacity_factor)
    max_source_tiles = (
        max_tokens_per_rank * top_k + dispatch_fc1_block_size_m - 1
    ) // dispatch_fc1_block_size_m
    context = MoEForwardContext(
        hidden_size=hidden_size,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
        rank=rank,
        receive_capacity_factor=receive_capacity_factor,
        max_source_tiles=max_source_tiles,
    )

    context.peer_mem = ash.aclshmem_create_tensor(
        [max_received_routes * hidden_size],
        dtype=torch.bfloat16,
        device_id=rank,
    )
    context.peer_mem.zero_()

    # Router weights remain FP32 through transport.  Activations and expert
    # weights are BF16 throughout this tutorial.
    context.routing_weight_mem = ash.aclshmem_create_tensor(
        [max_received_routes],
        dtype=torch.float32,
        device_id=rank,
    )
    context.routing_weight_mem.zero_()

    source_tile_slots = world_size * experts_per_rank * max_source_tiles
    # Expert-ready fallback schedules use one ADD counter per local expert.
    # Keep these slots disjoint from the source-tile SET epochs.
    expert_counter_slots = experts_per_rank
    signal_slots = source_tile_slots + expert_counter_slots
    context.signal_mem = ash.aclshmem_create_tensor(
        [signal_slots * 16],
        dtype=torch.int32,
        device_id=rank,
    )
    context.signal_mem.zero_()

    metadata_num_bins = triton.next_power_of_2(num_experts)
    context.metadata_num_bins = metadata_num_bins
    context.metadata_counts_mem = ash.aclshmem_create_tensor(
        [world_size * metadata_num_bins],
        dtype=torch.int32,
        device_id=rank,
    )
    context.metadata_counts_mem.zero_()

    device = context.peer_mem.device
    context.metadata_send_bucket_starts = torch.empty(
        num_experts, dtype=torch.int32, device=device
    )
    context.metadata_send_bucket_dst_starts = torch.empty(
        num_experts, dtype=torch.int32, device=device
    )
    context.metadata_recv_counts_re = torch.empty(
        (world_size, experts_per_rank), dtype=torch.int32, device=device
    )
    context.metadata_recv_per_expert = torch.empty(
        experts_per_rank, dtype=torch.int32, device=device
    )
    context.metadata_recv_expert_offs = torch.empty(
        experts_per_rank + 1, dtype=torch.int32, device=device
    )
    # [local receive routes, maximum receive routes required by any rank]
    context.metadata_stats = torch.empty(2, dtype=torch.int32, device=device)

    num_routes = max_tokens_per_rank * top_k
    context.row_token_indices = (
        torch.arange(num_routes, dtype=torch.int32, device=device) // top_k
    )
    context.row_route_indices = torch.arange(
        num_routes, dtype=torch.int32, device=device
    )
    return context


__all__ = ["MoEForwardContext", "create_moe_forward_context"]
