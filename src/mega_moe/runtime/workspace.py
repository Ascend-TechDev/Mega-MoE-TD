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
    replica_budget: int
    physical_experts_per_rank: int
    world_size: int
    rank: int
    receive_capacity_factor: float
    max_source_tiles: int

    peer_mem: Optional[torch.Tensor] = None
    routing_weight_mem: Optional[torch.Tensor] = None
    signal_mem: Optional[torch.Tensor] = None
    replica_gate_ready: Optional[torch.Tensor] = None
    replica_down_ready: Optional[torch.Tensor] = None
    replica_down_descriptors: Optional[torch.Tensor] = None
    row_token_indices: Optional[torch.Tensor] = None
    row_route_indices: Optional[torch.Tensor] = None
    planning_counts_mem: Optional[torch.Tensor] = None
    planning_num_bins: int = 0
    planning_expert_count: Optional[torch.Tensor] = None
    planning_transfers: Optional[torch.Tensor] = None
    planning_allocation: Optional[torch.Tensor] = None
    planning_alloc_cumsum: Optional[torch.Tensor] = None
    planning_experts_to_copy: Optional[torch.Tensor] = None
    planning_inverse_experts_to_copy: Optional[torch.Tensor] = None
    planning_replica_counts: Optional[torch.Tensor] = None
    metadata_counts_mem: Optional[torch.Tensor] = None
    metadata_num_bins: int = 0
    metadata_send_bucket_starts: Optional[torch.Tensor] = None
    metadata_send_bucket_dst_starts: Optional[torch.Tensor] = None
    metadata_recv_counts_re: Optional[torch.Tensor] = None
    metadata_recv_per_expert: Optional[torch.Tensor] = None
    metadata_recv_expert_offs: Optional[torch.Tensor] = None
    metadata_stats: Optional[torch.Tensor] = None
    metadata_local_expert_starts: Optional[torch.Tensor] = None
    balanced_send_token_indices: Optional[torch.Tensor] = None
    balanced_send_route_indices: Optional[torch.Tensor] = None
    ep_group: object = None

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
        self.replica_gate_ready = None
        self.replica_down_ready = None
        self.replica_down_descriptors = None
        if self.planning_counts_mem is not None:
            ash.aclshmem_free_tensor(self.planning_counts_mem)
            self.planning_counts_mem = None
        self.planning_expert_count = None
        self.planning_transfers = None
        self.planning_allocation = None
        self.planning_alloc_cumsum = None
        self.planning_experts_to_copy = None
        self.planning_inverse_experts_to_copy = None
        self.planning_replica_counts = None
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
        self.metadata_local_expert_starts = None
        self.balanced_send_token_indices = None
        self.balanced_send_route_indices = None


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
    enable_moonep: bool = False,
    ep_group=None,
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
    if type(enable_moonep) is not bool:
        raise TypeError("enable_moonep must be a bool")

    experts_per_rank = num_experts // world_size
    replica_budget = experts_per_rank if enable_moonep else 0
    physical_experts_per_rank = experts_per_rank + replica_budget
    max_received_routes = int(max_tokens_per_rank * top_k * receive_capacity_factor)
    max_source_tiles = (
        max_tokens_per_rank * top_k + dispatch_fc1_block_size_m - 1
    ) // dispatch_fc1_block_size_m
    context = MoEForwardContext(
        hidden_size=hidden_size,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        replica_budget=replica_budget,
        physical_experts_per_rank=physical_experts_per_rank,
        world_size=world_size,
        rank=rank,
        receive_capacity_factor=receive_capacity_factor,
        max_source_tiles=max_source_tiles,
        ep_group=ep_group,
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

    # Dispatch readiness and replica-weight readiness share one eagerly
    # initialized symmetric allocation.  A slot occupies 16 int32 values to
    # match the ACLSHMEM signal ABI used by the existing dispatch pipeline.
    dispatch_signal_slots = (
        world_size * physical_experts_per_rank * max_source_tiles
    )
    replica_signal_slots = 2 * replica_budget
    signal_slots = dispatch_signal_slots + replica_signal_slots
    context.signal_mem = ash.aclshmem_create_tensor(
        [signal_slots * 16],
        dtype=torch.int32,
        device_id=rank,
    )
    context.signal_mem.zero_()
    if replica_budget:
        gate_start = dispatch_signal_slots * 16
        down_start = gate_start + replica_budget * 16
        context.replica_gate_ready = context.signal_mem[
            gate_start:down_start
        ]
        context.replica_down_ready = context.signal_mem[
            down_start:down_start + replica_budget * 16
        ]

    num_dispatch_buckets = world_size * physical_experts_per_rank
    if enable_moonep:
        # One extra published bin stores each source rank's original route
        # count, so every rank can reject non-dropless/asymmetric inputs from
        # the same complete table without a separate HCCL collective.
        context.planning_num_bins = triton.next_power_of_2(num_experts + 1)
        context.planning_counts_mem = ash.aclshmem_create_tensor(
            [world_size * context.planning_num_bins],
            dtype=torch.int32,
            device_id=rank,
        )
        context.planning_counts_mem.zero_()

    metadata_num_bins = triton.next_power_of_2(num_dispatch_buckets)
    context.metadata_num_bins = metadata_num_bins
    context.metadata_counts_mem = ash.aclshmem_create_tensor(
        [world_size * metadata_num_bins],
        dtype=torch.int32,
        device_id=rank,
    )
    context.metadata_counts_mem.zero_()

    device = context.peer_mem.device
    if enable_moonep:
        # The two planner kernels reuse these ordinary device workspaces every
        # step.  Only tpe_all is symmetric; all ranks deterministically rebuild
        # the same allocation and replica tables from it.
        context.planning_expert_count = torch.empty(
            num_experts, dtype=torch.int64, device=device
        )
        context.planning_transfers = torch.empty(
            (world_size, world_size), dtype=torch.int64, device=device
        )
        context.planning_allocation = torch.empty(
            (world_size, num_experts), dtype=torch.int64, device=device
        )
        context.planning_alloc_cumsum = torch.empty(
            (num_experts, world_size), dtype=torch.int32, device=device
        )
        context.planning_experts_to_copy = torch.empty(
            (world_size, replica_budget), dtype=torch.int32, device=device
        )
        context.planning_inverse_experts_to_copy = torch.empty(
            (world_size, num_experts), dtype=torch.int32, device=device
        )
        context.planning_replica_counts = torch.empty(
            world_size, dtype=torch.int32, device=device
        )
        context.metadata_local_expert_starts = torch.empty(
            num_experts, dtype=torch.int32, device=device
        )
        # Compact owner-local down-copy descriptors are ordinary
        # (non-symmetric) workspace.  A single owner can supply replicas to
        # every peer, so reserve the complete ETC-table cardinality rather than
        # only one rank's replica budget.
        context.replica_down_descriptors = torch.empty(
            num_experts, dtype=torch.int32, device=device
        )
    context.metadata_send_bucket_starts = torch.empty(
        num_dispatch_buckets, dtype=torch.int32, device=device
    )
    context.metadata_send_bucket_dst_starts = torch.empty(
        num_dispatch_buckets, dtype=torch.int32, device=device
    )
    context.metadata_recv_counts_re = torch.empty(
        (world_size, physical_experts_per_rank),
        dtype=torch.int32,
        device=device,
    )
    context.metadata_recv_per_expert = torch.empty(
        physical_experts_per_rank, dtype=torch.int32, device=device
    )
    context.metadata_recv_expert_offs = torch.empty(
        physical_experts_per_rank + 1, dtype=torch.int32, device=device
    )
    # [local receive routes, maximum receive routes required by any rank,
    #  one temporary receive total per destination rank]
    context.metadata_stats = torch.empty(
        2 + world_size,
        dtype=torch.int32,
        device=device,
    )

    num_routes = max_tokens_per_rank * top_k
    context.row_token_indices = (
        torch.arange(num_routes, dtype=torch.int32, device=device) // top_k
    )
    context.row_route_indices = torch.arange(
        num_routes, dtype=torch.int32, device=device
    )
    if enable_moonep:
        context.balanced_send_token_indices = torch.empty(
            num_routes, dtype=torch.int32, device=device
        )
        context.balanced_send_route_indices = torch.empty(
            num_routes, dtype=torch.int32, device=device
        )
    return context


__all__ = ["MoEForwardContext", "create_moe_forward_context"]
