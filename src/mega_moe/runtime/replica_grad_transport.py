# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Symmetric-slot transport for MoonEP replica expert weight gradients.

The M2 physical backward leaves the replica weight gradients in ordinary device
tensors.  :class:`ReplicaGradTransport` moves them through the *forward's*
symmetric replica weight tables (no second symmetric allocation, no second
gather of the redundant gradients): the producing rank sinks its replica
segments into its own slots, and every owner pulls the slots holding copies of
its home experts back and accumulates them onto its home seed.

Borrowing the tables is destructive — ``FusedMoEForward.lend_replica_weight_tables_for_grad``
invalidates the replica weight cache at hand-off, so a gradient left in a slot
can never be sold as a weight even if the backward aborts midway.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from ..kernels.replica_grad_reduce import (
    build_owner_pull_descriptors,
    launch_owner_pull_accumulate,
    launch_replica_grad_barrier,
    zero_consumed_replica_slots,
)
from .replica_weight_prefetch import (
    ReplicaWeightBuffers,
    _validate_barrier_grid,
    replica_weight_push_geometry,
)


@dataclass
class ReplicaGradTransport:
    """One backward's handle on the forward's symmetric replica weight tables.

    ``buffers`` is a *borrowed reference* to the operator's tables (never a new
    allocation); the ``experts_to_copy_*`` snapshots are clones, because the
    planner's metadata workspace is reused in place.  ``num_barrier_programs``
    must equal the physical AICore count (see
    :func:`mega_moe.runtime.replica_weight_prefetch._validate_barrier_grid`).
    """

    buffers: ReplicaWeightBuffers
    experts_to_copy_cpu: torch.Tensor
    experts_to_copy_device: Optional[torch.Tensor]
    rank: int
    world_size: int
    experts_per_rank: int
    num_barrier_programs: int
    gate_up_chunk_bytes: int = 16 * 1024 * 1024
    down_chunk_bytes: int = 4 * 1024 * 1024
    acc_block: int = 4096
    post_sink_hook: Optional[Callable[["ReplicaGradTransport"], None]] = None
    sunk: bool = field(default=False, compare=False)
    reduced: bool = field(default=False, compare=False)

    def __post_init__(self):
        if self.buffers.closed:
            raise RuntimeError("the borrowed replica weight buffers were finalized")
        if self.buffers.experts_per_rank != self.experts_per_rank:
            raise ValueError("the borrowed tables do not match experts_per_rank")
        if self.buffers.rank != self.rank:
            raise ValueError("the borrowed tables belong to another rank")
        if self.experts_to_copy_cpu.device.type != "cpu":
            raise ValueError("experts_to_copy_cpu must be a CPU tensor")
        if tuple(self.experts_to_copy_cpu.shape) != (
            self.world_size, self.experts_per_rank
        ):
            raise ValueError(
                "experts_to_copy_cpu must have shape "
                f"{(self.world_size, self.experts_per_rank)}"
            )
        _validate_barrier_grid(self.num_barrier_programs)

    # ---- geometry ------------------------------------------------------
    @property
    def gate_up_expert_shape(self) -> tuple[int, int]:
        """Per-expert ``[H, 2F]`` shape of the borrowed gate/up table."""
        return self.buffers.gate_up_expert_shape

    @property
    def down_expert_shape(self) -> tuple[int, int]:
        """Per-expert ``[H, F]`` shape of the borrowed down table."""
        return self.buffers.down_expert_shape

    @property
    def local_experts_to_copy(self) -> torch.Tensor:
        """This rank's replica row of the global ``experts_to_copy`` table."""
        return self.experts_to_copy_cpu[self.rank]

    def consumed_slots(self) -> list[int]:
        """Replica slots this rank owns, i.e. the ones :meth:`sink` writes."""
        return [
            slot
            for slot, expert in enumerate(self.local_experts_to_copy.tolist())
            if expert >= 0
        ]

    # ---- stages --------------------------------------------------------
    def sink(self, grad_fc1_phys, grad_fc2_phys) -> None:
        """Copy this rank's replica weight-grad segments into its slots.

        Only slots with ``experts_to_copy[rank][slot] >= 0`` are touched; the
        physical gradient layout is ``[epn + B, 2F, H]`` / ``[epn + B, H, F]``
        while the tables pack ``[H, 2F]`` / ``[H, F]``, so the gate/up segment
        is transposed on the way in.  Both sides are BF16, which makes the copy
        exact and leaves the precision question to the owner-side fp32 reduce.
        """
        if self.sunk:
            raise RuntimeError("the replica grad transport was already sunk")
        home_experts = self.experts_per_rank
        gate_up_table = self.buffers.gate_up
        down_table = self.buffers.down
        for name, grad, expected_shape in (
            # the gate/up gradient is [2F, H]; the table slot packs [H, 2F]
            ("grad_fc1_phys", grad_fc1_phys,
             (home_experts * 2, gate_up_table.shape[2], gate_up_table.shape[1])),
            ("grad_fc2_phys", grad_fc2_phys, (home_experts * 2, *down_table.shape[1:])),
        ):
            if grad.ndim != 3 or tuple(grad.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got "
                    f"{tuple(grad.shape)}"
                )
            if grad.dtype != torch.bfloat16 or not grad.is_contiguous():
                raise ValueError(f"{name} must be a contiguous bf16 tensor")
        for slot in self.consumed_slots():
            # gate/up gradient is [2F, H]; the table slot is packed [H, 2F].
            gate_up_table[slot].copy_(grad_fc1_phys[home_experts + slot].t())
            down_table[slot].copy_(grad_fc2_phys[home_experts + slot])
        self.sunk = True
        if self.post_sink_hook is not None:
            self.post_sink_hook(self)

    def reduce(self, grad_fc1_phys, grad_fc2_phys):
        """Return the final BF16 home weight gradients as ``(fc1, fc2)``.

        Stage order (every rank must call this exactly once per backward):

        * seed the fp32 accumulators with the physical home segments;
        * sink the replica segments (if the caller has not done it already);
        * barrier #1 publishes the slots, the owner-pull kernel accumulates
          every replica slot onto its owner's seed, barrier #2 retires the
          readers, the consumed slots are zeroed, and barrier #3 publishes the
          zeroing so the next forward's owner-push cannot race it.

        All three barriers are launched unconditionally: the descriptor count
        is rank-dependent, but the barrier participation is not.
        """
        if self.reduced:
            raise RuntimeError("the replica grad transport was already reduced")
        home_experts = self.experts_per_rank
        # Seed first: the home segment is the fp32 base every replica slot adds
        # to.  The gate/up accumulator is seeded in the *table's* [H, 2F] layout
        # so the device-side accumulate stays a contiguous walk; the transposes
        # live here, on the host, where they are exact.
        device = grad_fc1_phys.device
        acc_gate_up = torch.empty(
            (home_experts, *self.gate_up_expert_shape),
            dtype=torch.float32,
            device=device,
        ).copy_(grad_fc1_phys[:home_experts].transpose(1, 2))
        acc_down = grad_fc2_phys[:home_experts].float()
        if not self.sunk:
            self.sink(grad_fc1_phys, grad_fc2_phys)

        desc_peer, desc_slot, desc_home = build_owner_pull_descriptors(
            self.experts_to_copy_cpu, self.rank, home_experts
        )
        desc_peer = desc_peer.to(device)
        desc_slot = desc_slot.to(device)
        desc_home = desc_home.to(device)
        gate_up_elements = int(acc_gate_up[0].numel())
        down_elements = int(acc_down[0].numel())
        hidden_dim = self.down_expert_shape[0]
        ffn_dim = self.down_expert_shape[1]
        if self.gate_up_expert_shape != (hidden_dim, 2 * ffn_dim):
            raise ValueError("the borrowed gate/up and down tables disagree")
        gate_up_chunk_elements, gate_up_num_chunks = replica_weight_push_geometry(
            gate_up_elements, chunk_bytes=self.gate_up_chunk_bytes
        )
        down_chunk_elements, down_num_chunks = replica_weight_push_geometry(
            down_elements, chunk_bytes=self.down_chunk_bytes
        )

        launch_replica_grad_barrier(self.num_barrier_programs)
        launch_owner_pull_accumulate(
            acc_gate_up=acc_gate_up,
            acc_down=acc_down,
            gate_up_table=self.buffers.gate_up,
            down_table=self.buffers.down,
            desc_peer=desc_peer,
            desc_slot=desc_slot,
            desc_home=desc_home,
            rank=self.rank,
            gate_up_chunk_elements=gate_up_chunk_elements,
            gate_up_num_chunks=gate_up_num_chunks,
            down_chunk_elements=down_chunk_elements,
            down_num_chunks=down_num_chunks,
            acc_block=self.acc_block,
        )
        launch_replica_grad_barrier(self.num_barrier_programs)
        zero_consumed_replica_slots(self.buffers, self.consumed_slots())
        launch_replica_grad_barrier(self.num_barrier_programs)
        self.reduced = True
        # Back to the gradient's [2F, H] layout the caller chunks gate from up.
        reduced_gate_up = torch.empty(
            (home_experts, 2 * ffn_dim, hidden_dim),
            dtype=torch.bfloat16,
            device=device,
        ).copy_(acc_gate_up.transpose(1, 2))
        return reduced_gate_up, acc_down.to(torch.bfloat16)


__all__ = ["ReplicaGradTransport"]
