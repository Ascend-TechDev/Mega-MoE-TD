# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Lifecycle and synchronous owner-push prefetch for replica expert weights."""

from dataclasses import dataclass
from math import prod
from typing import Optional

import torch


_BF16_BYTES = 2
_ACLSHMEM_PUTMEM_MAX_BYTES = (1 << 32) - 1
_DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_WEIGHT_CHUNKS = 4096


def _require_positive_int(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _validate_barrier_grid(num_barrier_programs: int) -> None:
    _require_positive_int("num_barrier_programs", num_barrier_programs)
    from triton.backends.ascend.driver import NPUUtils

    physical_aicore_count = NPUUtils().get_aicore_num()
    if num_barrier_programs != physical_aicore_count:
        raise ValueError(
            "num_barrier_programs must equal the physical AICore count "
            f"{physical_aicore_count}, got {num_barrier_programs}"
        )


def _validate_chunk_bytes(chunk_bytes: int) -> int:
    """Return BF16 elements per RMA chunk after checking the uint32 ABI."""
    _require_positive_int("chunk_bytes", chunk_bytes)
    if chunk_bytes > _ACLSHMEM_PUTMEM_MAX_BYTES:
        raise ValueError(
            "chunk_bytes exceeds the ACLSHMEM uint32 byte-count ABI"
        )
    if chunk_bytes % _BF16_BYTES:
        raise ValueError("chunk_bytes must be aligned to the BF16 element size")
    return chunk_bytes // _BF16_BYTES


def _num_chunks(elements_per_expert: int, chunk_elements: int) -> int:
    _require_positive_int("elements_per_expert", elements_per_expert)
    _require_positive_int("chunk_elements", chunk_elements)
    chunks = (elements_per_expert + chunk_elements - 1) // chunk_elements
    if chunks > _MAX_WEIGHT_CHUNKS:
        raise ValueError(
            "chunk_bytes would require too many compile-time RMA chunks; "
            "use a larger chunk size"
        )
    return chunks


def replica_weight_push_geometry(
    elements_per_expert: int,
    *,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> tuple[int, int]:
    """Return ``(chunk_elements, num_chunks)`` for an in-kernel push."""
    chunk_elements = _validate_chunk_bytes(chunk_bytes)
    return chunk_elements, _num_chunks(elements_per_expert, chunk_elements)


def _validate_local_weight_pair(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> tuple[int, int, int]:
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

    experts_per_rank = gate_up_weight.shape[0]
    gate_up_elements = prod(gate_up_weight.shape[1:])
    down_elements = prod(down_weight.shape[1:])
    return experts_per_rank, gate_up_elements, down_elements


@dataclass
class ReplicaWeightBuffers:
    """Two fixed-``B=epn`` symmetric replica tables owned by one operator."""

    gate_up_mem: Optional[torch.Tensor]
    down_mem: Optional[torch.Tensor]
    experts_per_rank: int
    gate_up_expert_shape: tuple[int, int]
    down_expert_shape: tuple[int, int]
    rank: int

    @property
    def closed(self) -> bool:
        return self.gate_up_mem is None or self.down_mem is None

    @property
    def gate_up(self) -> torch.Tensor:
        if self.gate_up_mem is None:
            raise RuntimeError("replica weight buffers have been finalized")
        return self.gate_up_mem.view(
            self.experts_per_rank, *self.gate_up_expert_shape
        )

    @property
    def down(self) -> torch.Tensor:
        if self.down_mem is None:
            raise RuntimeError("replica weight buffers have been finalized")
        return self.down_mem.view(
            self.experts_per_rank, *self.down_expert_shape
        )

    @property
    def allocated_bytes(self) -> int:
        if self.closed:
            return 0
        return (self.gate_up_mem.numel() + self.down_mem.numel()) * _BF16_BYTES

    def finalize(self) -> None:
        """Collectively release both symmetric tables; the call is idempotent."""
        import shmem as ash

        # Free in reverse allocation order.  Every rank must invoke finalize in
        # the same order after all kernels that reference these buffers finish.
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
    rank: int,
    world_size: int,
) -> ReplicaWeightBuffers:
    """Allocate equal-shaped symmetric replica tables on every EP rank.

    All ranks must call this function in the same ACLSHMEM allocation order.
    The replica budget is fixed to ``experts_per_rank``.
    """
    _require_positive_int("world_size", world_size)
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("rank must be an integer in [0, world_size)")
    experts_per_rank, gate_up_elements, down_elements = (
        _validate_local_weight_pair(gate_up_weight, down_weight)
    )

    import shmem as ash

    if ash.my_pe() != rank or ash.pe_count() != world_size:
        raise ValueError(
            "replica prefetch requires the EP group to match the ACLSHMEM world"
        )

    gate_up_mem = ash.aclshmem_create_tensor(
        [experts_per_rank * gate_up_elements],
        dtype=torch.bfloat16,
        device_id=rank,
    )
    try:
        down_mem = ash.aclshmem_create_tensor(
            [experts_per_rank * down_elements],
            dtype=torch.bfloat16,
            device_id=rank,
        )
    except Exception:
        ash.aclshmem_free_tensor(gate_up_mem)
        raise

    return ReplicaWeightBuffers(
        gate_up_mem=gate_up_mem,
        down_mem=down_mem,
        experts_per_rank=experts_per_rank,
        gate_up_expert_shape=tuple(gate_up_weight.shape[1:]),
        down_expert_shape=tuple(down_weight.shape[1:]),
        rank=rank,
    )


def fence_replica_weight_prefetch_async(*, num_barrier_programs: int) -> None:
    """Queue the collective fence that makes all replica tables visible.

    Launch this on the prefetch stream after that stream's local weight pushes
    and after a caller-stream event proving dispatch has completed.  Deferring
    the all-core barrier prevents it from competing with dispatch for Vector
    resources while retaining down-copy/dispatch overlap.
    """
    _validate_barrier_grid(num_barrier_programs)
    from mega_moe.kernels.replica_weight_prefetch import (
        _kernel_replica_weight_prefetch_barrier,
    )

    _kernel_replica_weight_prefetch_barrier[
        (num_barrier_programs, 1, 1)
    ]()


__all__ = [
    "ReplicaWeightBuffers",
    "allocate_replica_weight_buffers",
    "fence_replica_weight_prefetch_async",
]
