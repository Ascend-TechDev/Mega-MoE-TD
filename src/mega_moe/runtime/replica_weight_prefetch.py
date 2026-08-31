# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Owner-staging lifecycle for dispatch-fused MoonEP panel pushes."""

from dataclasses import dataclass
from math import prod
from typing import Optional

import torch


_BF16_BYTES = 2
_ACLSHMEM_UDMA_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_WEIGHT_CHUNKS = 4096


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_chunk_bytes(chunk_bytes: int) -> int:
    """Return BF16 elements per RMA chunk after checking the uint32 ABI."""
    _require_positive_int("chunk_bytes", chunk_bytes)
    if chunk_bytes > _ACLSHMEM_UDMA_MAX_BYTES:
        raise ValueError(
            "chunk_bytes exceeds the ACLSHMEM UDMA 256 MiB request limit"
        )
    if chunk_bytes % _BF16_BYTES:
        raise ValueError("chunk_bytes must be aligned to the BF16 element size")
    return chunk_bytes // _BF16_BYTES


def _num_chunks(elements: int, chunk_elements: int) -> int:
    _require_positive_int("elements", elements)
    _require_positive_int("chunk_elements", chunk_elements)
    chunks = (elements + chunk_elements - 1) // chunk_elements
    if chunks > _MAX_WEIGHT_CHUNKS:
        raise ValueError(
            "chunk_bytes would require too many compile-time RMA chunks; "
            "use a larger chunk size"
        )
    return chunks


def replica_weight_push_geometry(
    panel_elements: int,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> tuple[int, int]:
    """Return ``(chunk_elements, num_chunks)`` for one output-N panel PUT."""
    chunk_elements = _validate_chunk_bytes(chunk_bytes)
    return chunk_elements, _num_chunks(panel_elements, chunk_elements)


def _validate_local_weight_pair(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    named_weights = {
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
    }
    for name, weight in named_weights.items():
        if weight.ndim != 3:
            raise ValueError(f"{name} must have [expert, K, N] layout")
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"{name} must use torch.bfloat16, got {weight.dtype}")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if any(dimension <= 0 for dimension in weight.shape):
            raise ValueError(f"{name} dimensions must all be positive")

    if gate_up_weight.device != down_weight.device:
        raise ValueError("gate_up_weight and down_weight must be on one device")
    if gate_up_weight.shape[0] != down_weight.shape[0]:
        raise ValueError("gate_up_weight and down_weight expert counts differ")
    if gate_up_weight.shape[1] != down_weight.shape[1]:
        raise ValueError("gate_up_weight and down_weight hidden sizes differ")
    if gate_up_weight.shape[2] != 2 * down_weight.shape[2]:
        raise ValueError(
            "gate_up_weight output size must be twice down_weight's FFN size"
        )
    if gate_up_weight.shape[1] % 2:
        raise ValueError("hidden_size must be even for two down-weight N panels")

    experts_per_rank = gate_up_weight.shape[0]
    hidden_size = gate_up_weight.shape[1]
    ffn_size = down_weight.shape[2]
    gate_up_elements = prod(gate_up_weight.shape[1:])
    down_elements = prod(down_weight.shape[1:])
    return (
        experts_per_rank,
        hidden_size,
        ffn_size,
        gate_up_elements,
        down_elements,
    )


@dataclass
class ReplicaWeightBuffers:
    """Ordinary compact owner staging plus symmetric consumer destinations."""

    source_gate_up_mem: Optional[torch.Tensor]
    source_down_mem: Optional[torch.Tensor]
    gate_up_mem: Optional[torch.Tensor]
    down_mem: Optional[torch.Tensor]
    previous_experts_to_copy: Optional[torch.Tensor]
    source_slot_by_global_expert: Optional[torch.Tensor]
    source_capacity: int
    replica_capacity: int
    experts_per_rank: int
    hidden_size: int
    ffn_size: int

    @property
    def closed(self) -> bool:
        return (
            self.source_gate_up_mem is None
            or self.source_down_mem is None
            or self.gate_up_mem is None
            or self.down_mem is None
        )

    @property
    def source_gate_up(self) -> torch.Tensor:
        if self.source_gate_up_mem is None:
            raise RuntimeError("replica source staging has been finalized")
        return self.source_gate_up_mem.view(
            self.source_capacity, 2, self.hidden_size, self.ffn_size
        )

    @property
    def source_down(self) -> torch.Tensor:
        if self.source_down_mem is None:
            raise RuntimeError("replica source staging has been finalized")
        return self.source_down_mem.view(
            self.source_capacity, self.hidden_size, self.ffn_size
        )

    @property
    def gate_up(self) -> torch.Tensor:
        if self.gate_up_mem is None:
            raise RuntimeError("replica destination buffers have been finalized")
        return self.gate_up_mem.view(
            self.replica_capacity, 2, self.hidden_size, self.ffn_size
        )

    @property
    def down(self) -> torch.Tensor:
        if self.down_mem is None:
            raise RuntimeError("replica destination buffers have been finalized")
        return self.down_mem.view(
            self.replica_capacity, self.hidden_size, self.ffn_size
        )

    def finalize(self) -> None:
        """Collectively release symmetric destinations; ordinary staging drops."""
        import shmem as ash

        self.previous_experts_to_copy = None
        self.source_slot_by_global_expert = None
        self.source_down_mem = None
        self.source_gate_up_mem = None
        if self.down_mem is not None:
            ash.aclshmem_free_tensor(self.down_mem)
            self.down_mem = None
        if self.gate_up_mem is not None:
            ash.aclshmem_free_tensor(self.gate_up_mem)
            self.gate_up_mem = None


def allocate_replica_weight_buffers(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    source_capacity: int,
    replica_capacity: int,
    rank: int,
    world_size: int,
) -> ReplicaWeightBuffers:
    """Allocate local compact staging and equal-sized symmetric destinations."""
    _require_positive_int("world_size", world_size)
    _require_positive_int("source_capacity", source_capacity)
    _require_positive_int("replica_capacity", replica_capacity)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be an integer in [0, world_size)")
    (
        experts_per_rank,
        hidden_size,
        ffn_size,
        gate_up_elements,
        down_elements,
    ) = _validate_local_weight_pair(gate_up_weight, down_weight)
    if source_capacity > experts_per_rank:
        raise ValueError("source_capacity cannot exceed experts_per_rank")
    if replica_capacity > experts_per_rank:
        raise ValueError("replica_capacity cannot exceed experts_per_rank")

    import shmem as ash

    if ash.my_pe() != rank or ash.pe_count() != world_size:
        raise ValueError(
            "replica prefetch requires the EP group to match the ACLSHMEM world"
        )

    device = gate_up_weight.device
    source_gate_up_mem = torch.empty(
        source_capacity * gate_up_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    source_down_mem = torch.empty(
        source_capacity * down_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    gate_up_mem = ash.aclshmem_create_tensor(
        [replica_capacity * gate_up_elements],
        dtype=torch.bfloat16,
        device_id=rank,
    )
    try:
        down_mem = ash.aclshmem_create_tensor(
            [replica_capacity * down_elements],
            dtype=torch.bfloat16,
            device_id=rank,
        )
    except Exception:
        ash.aclshmem_free_tensor(gate_up_mem)
        raise

    return ReplicaWeightBuffers(
        source_gate_up_mem=source_gate_up_mem,
        source_down_mem=source_down_mem,
        gate_up_mem=gate_up_mem,
        down_mem=down_mem,
        previous_experts_to_copy=torch.full(
            (world_size, experts_per_rank),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        source_slot_by_global_expert=torch.empty(
            world_size * experts_per_rank,
            dtype=torch.int32,
            device=device,
        ),
        source_capacity=source_capacity,
        replica_capacity=replica_capacity,
        experts_per_rank=experts_per_rank,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
    )


def wait_previous_replica_consumed_async(
    previous_experts_to_copy: torch.Tensor,
    consumed_epoch: torch.Tensor,
    previous_epoch: int,
    *,
    rank: int,
    world_size: int,
    experts_per_rank: int,
) -> None:
    """Wait until prior consumers release every staging source used by owner."""
    if previous_epoch <= 0:
        return
    from mega_moe.kernels.replica_weight_prefetch import (
        _kernel_wait_previous_replica_consumed,
    )

    _kernel_wait_previous_replica_consumed[(1, 1, 1)](
        previous_experts_to_copy,
        consumed_epoch,
        previous_epoch,
        experts_per_rank,
        LOCAL_RANK=rank,
        WORLD_SIZE=world_size,
        EXPERTS_PER_RANK=experts_per_rank,
    )


def publish_replica_consumed_async(
    experts_to_copy: torch.Tensor,
    down_ready: torch.Tensor,
    consumed_epoch: torch.Tensor,
    signal_epoch: int,
    replica_slot_count: int,
    *,
    rank: int,
    experts_per_rank: int,
) -> None:
    """Publish consumer slot reuse only after every final panel is usable."""
    from mega_moe.kernels.replica_weight_prefetch import (
        _kernel_publish_replica_consumed,
    )

    _kernel_publish_replica_consumed[(1, 1, 1)](
        experts_to_copy,
        down_ready,
        consumed_epoch,
        signal_epoch,
        replica_slot_count,
        LOCAL_RANK=rank,
        EXPERTS_PER_RANK=experts_per_rank,
    )


def fence_replica_panel_pushes_async(*, rank: int, world_size: int) -> None:
    """Drain every owner PUT QP for teardown or symmetric-buffer resize."""
    from mega_moe.kernels.replica_weight_prefetch import (
        _kernel_quiet_replica_panel_pushes,
    )
    from mega_moe.runtime.udma_pipe_s import udma_pipe_s_bisheng_options

    _kernel_quiet_replica_panel_pushes[(world_size, 1, 1)](
        LOCAL_RANK=rank,
        WORLD_SIZE=world_size,
        bisheng_options=udma_pipe_s_bisheng_options(),
    )


def launch_replica_source_pack_async(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    buffers: ReplicaWeightBuffers,
    source_ready: torch.Tensor,
    signal_epoch: int,
    *,
    rank: int,
    num_vector_programs: int,
) -> None:
    """Queue parallel owner packing and per-panel source-ready publication."""
    _require_positive_int("num_vector_programs", num_vector_programs)
    from mega_moe.kernels.replica_weight_prefetch import (
        _kernel_pack_requested_replica_weight_panels,
    )

    _kernel_pack_requested_replica_weight_panels[
        (num_vector_programs, 1, 1)
    ](
        gate_up_weight,
        down_weight,
        buffers.source_gate_up,
        buffers.source_down,
        buffers.source_slot_by_global_expert,
        source_ready,
        signal_epoch,
        NUM_PROGRAMS=num_vector_programs,
        LOCAL_RANK=rank,
        EXPERTS_PER_RANK=buffers.experts_per_rank,
        HIDDEN_SIZE=buffers.hidden_size,
        FFN_SIZE=buffers.ffn_size,
        COPY_BLOCK_ELEMENTS=4096,
        GATE_BLOCK_K=16,
        GATE_BLOCK_N=256,
    )


__all__ = [
    "ReplicaWeightBuffers",
    "allocate_replica_weight_buffers",
    "fence_replica_panel_pushes_async",
    "launch_replica_source_pack_async",
    "publish_replica_consumed_async",
    "replica_weight_push_geometry",
    "wait_previous_replica_consumed_async",
]
