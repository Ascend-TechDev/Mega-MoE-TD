# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Lifecycle and synchronous owner-push prefetch for replica expert weights."""

import os
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
    # Local NPU ordinal for every symmetric allocation owned by these buffers
    # (tables themselves and the grad-push scratch).  ``rank`` stays the
    # ACLSHMEM global PE; multi-node splits the two
    # (see mega_moe.runtime.device).
    local_device: Optional[int] = None
    # Table-level SET-epoch counter (pooling era, 2026-09-17): EVERY push
    # into these slots — forward prefetch or backward re-prefetch, any layer
    # sharing the pool entry — mints the next value here.  Monotonic
    # numbering per slot-set is the SET-wait correctness contract: a
    # consumer's dl.wait(epoch) may only be satisfied by the push that SET
    # that value, which holds iff no two pushes into one ready slab ever
    # share a number.  The historical per-operator counter breaks under
    # pooling (two layers mint overlapping epochs -> a stale slot passes the
    # new wait -> silently wrong weights; same family as the mega_persistent
    # epoch-reset bug).
    push_epoch: int = 0
    # MOE_MEGA_GRAD_TRANSPORT=udma scratch (2026-09-22 combo plan): the
    # owner-pull's getmem is corrupt under MTE|UDMA, so the mega backward
    # inverts to a credit-protocol UDMA push; these symmetric slabs are the
    # push destinations and handshake words.  Cached on the (pooled) buffers
    # so the ACLSHMEM allocation order stays rank-collective and the
    # epoch-monotonic SET words need no cross-call zeroing.
    grad_push_scratch: Optional[dict] = None

    def ensure_grad_push_scratch(
        self, gu_tasks, gu_chunk, dn_tasks, dn_chunk, device,
        ord_stride=1,
    ) -> dict:
        """Allocate (once) the symmetric grad-push staging and word slabs.

        Staging rows are (task, ordinal)-indexed (``(home * chunks + chunk)
        * ord_stride + ordinal``) — credit-free since the 2026-09-22
        zengwang fix (signal_op credit SETs dropped 33-41% under framework
        load while put_signal_nbi arrivals landed 100%), every ordinal owns
        a private row so pushers never wait.  Arrival words are int32 pairs
        viewed as uint64 for the UDMA tail SET and read as the low int32 by
        the local ``dl.wait`` — values are ``epoch * 256 + ordinal``,
        strictly increasing across calls, which is why zero-init once
        suffices.  ``cred_*`` stay allocated for the frozen kernel
        signature but are written by nobody.
        """
        key = (gu_tasks, gu_chunk, dn_tasks, dn_chunk, ord_stride)
        cached = self.grad_push_scratch
        if cached is not None:
            if cached["key"] != key:
                raise ValueError(
                    "grad push scratch geometry changed on live buffers: "
                    f"{cached['key']} -> {key}"
                )
            return cached
        import shmem as ash

        dev = self.rank if self.local_device is None else self.local_device

        def _alloc(count, dtype):
            tensor = ash.aclshmem_create_tensor(
                [count], dtype=dtype, device_id=dev)
            tensor.zero_()
            return tensor

        # int32 pairs (u64-viewable), 2 words per (task, ordinal) per family
        gu_cells = gu_tasks * ord_stride
        dn_cells = dn_tasks * ord_stride
        scratch = {
            "key": key,
            "staging_gu": _alloc(gu_cells * gu_chunk, torch.bfloat16),
            "staging_dn": _alloc(dn_cells * dn_chunk, torch.bfloat16),
            "arr_gu": _alloc(2 * gu_cells, torch.int32),
            "arr_dn": _alloc(2 * dn_cells, torch.int32),
            "cred_gu": _alloc(2 * gu_cells, torch.int32),
            "cred_dn": _alloc(2 * dn_cells, torch.int32),
        }
        self.grad_push_scratch = scratch
        return scratch

    def next_push_epoch(self, floor: int = 0) -> int:
        """Mint the next push epoch, never at or below ``floor``.

        ``floor`` lets the caller honor reservations made against other epoch
        sequences that share these slabs (the single-kernel path's
        ``_tile_signal_epoch`` aliasing guard lifts the tile sequence above
        minted replica epochs and reserves the next one).
        """
        value = max(self.push_epoch, floor) + 1
        if value >= torch.iinfo(torch.int32).max:
            raise RuntimeError(
                "replica weight push epoch exhausted; recreate the operator "
                "(or pool)")
        self.push_epoch = value
        return value

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
        if self.grad_push_scratch is not None:
            for name in ("cred_dn", "cred_gu", "arr_dn", "arr_gu",
                         "staging_dn", "staging_gu"):
                ash.aclshmem_free_tensor(self.grad_push_scratch[name])
            self.grad_push_scratch = None


def allocate_replica_weight_buffers(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    local_device: Optional[int] = None,
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

    dev = rank if local_device is None else local_device
    gate_up_mem = ash.aclshmem_create_tensor(
        [experts_per_rank * gate_up_elements],
        dtype=torch.bfloat16,
        device_id=dev,
    )
    try:
        down_mem = ash.aclshmem_create_tensor(
            [experts_per_rank * down_elements],
            dtype=torch.bfloat16,
            device_id=dev,
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
        local_device=dev,
    )


# ---------------------------------------------------------------------------
# Session-level replica-table pool (MEGAMOE_REPLICA_POOL=1)
#
# The integrated net runs one operator instance per MoE layer, and every
# instance used to allocate its OWN pair of symmetric replica tables
# (~0.67 GB/layer at the kimi mock shape) held for the process lifetime —
# the whole-net memory blow-up.  Under pooling all same-shape layers share
# ONE pair: the pool below is keyed by shape, entries are reference counted,
# and the real allocation (with its rank-collective ordering) happens only on
# the first miss.  Layer serialization (standard autograd: at most one
# layer's forward OR backward uses the tables at any instant) makes a pool
# depth of 1 sufficient; 1F1B/interleaved schedules would need a deeper
# pool and are explicitly out of scope.
#
# Sharing the TABLES is only correct because the backward re-prefetches
# (MOE_MEGA_REPREFETCH): by a layer's backward time the pooled content is
# some other layer's weights, so the mega backward must re-push this
# layer's weights before reading them.  Enabling the pool for a multi-layer
# model without the backward re-prefetch reads foreign weights silently.
# ----------------------------------------------------------------------------

# key -> [ReplicaWeightBuffers, live refcount]
_REPLICA_POOL: dict[tuple, list] = {}


def replica_pool_enabled() -> bool:
    return os.environ.get("MEGAMOE_REPLICA_POOL", "0") == "1"


def acquire_replica_weight_buffers(
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    local_device: Optional[int] = None,
) -> tuple[ReplicaWeightBuffers, bool]:
    """Take a pooled handle; returns ``(buffers, fresh)``.

    ``fresh`` marks a real allocation: only that caller runs the collective
    drain/barrier dance after the call (every rank visits layer shapes in
    the same order, so hit/miss decisions are rank-uniform and the ACLSHMEM
    allocation order stays collective-safe).
    """
    experts_per_rank, _, _ = _validate_local_weight_pair(
        gate_up_weight, down_weight
    )
    key = (
        world_size,
        experts_per_rank,
        tuple(gate_up_weight.shape[1:]),
        tuple(down_weight.shape[1:]),
    )
    entry = _REPLICA_POOL.get(key)
    if entry is not None and not entry[0].closed:
        entry[1] += 1
        return entry[0], False
    # Pool key intentionally omits local_device: it is constant per process.
    buffers = allocate_replica_weight_buffers(
        gate_up_weight, down_weight, rank=rank, world_size=world_size,
        local_device=local_device,
    )
    _REPLICA_POOL[key] = [buffers, 1]
    return buffers, True


def release_replica_weight_buffers(buffers: ReplicaWeightBuffers) -> None:
    """Drop one pooled reference; the last one frees the symmetric tables."""
    for key, entry in _REPLICA_POOL.items():
        if entry[0] is buffers:
            entry[1] -= 1
            if entry[1] <= 0:
                del _REPLICA_POOL[key]
                buffers.finalize()
            return
    # Never pooled (or the pool was drained underneath): plain finalize.
    buffers.finalize()


def drain_replica_weight_pool() -> None:
    """Force-finalize every pooled entry regardless of refcounts (tests)."""
    for _key, entry in list(_REPLICA_POOL.items()):
        if not entry[0].closed:
            entry[0].finalize()
    _REPLICA_POOL.clear()


def replica_pool_snapshot() -> dict[tuple, int]:
    """Live pool state as ``{shape key: live refcount}`` (regression tests)."""
    return {key: entry[1] for key, entry in _REPLICA_POOL.items()}


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
    "acquire_replica_weight_buffers",
    "release_replica_weight_buffers",
    "drain_replica_weight_pool",
    "replica_pool_enabled",
    "replica_pool_snapshot",
]
