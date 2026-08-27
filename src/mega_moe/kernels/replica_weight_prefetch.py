# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Owner-push ACLSHMEM transport for MoonEP replica expert weights.

The source table is ordinary device memory.  Only the destination replica
table must be an ACLSHMEM symmetric allocation.  Every rank scans the same
``experts_to_copy[destination, replica_slot]`` table and pushes the experts it
owns into the corresponding destination slots.
"""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton_dist.language.extra import libshmem_device


@triton.jit
def _kernel_compact_local_replica_descriptors(
    experts_to_copy_ptr,
    descriptor_ids_ptr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Build the owner-local remote descriptor list in slot-major order."""
    destination_rank = tl.arange(0, WORLD_SIZE)
    write_count = 0
    with al.scope(core_mode="vector", disable_auto_sync=True):
        for replica_slot in range(0, EXPERTS_PER_RANK):
            descriptor = destination_rank * EXPERTS_PER_RANK + replica_slot
            global_expert = tl.load(experts_to_copy_ptr + descriptor)
            valid = (
                (global_expert >= 0)
                & (global_expert < WORLD_SIZE * EXPERTS_PER_RANK)
                & (global_expert // EXPERTS_PER_RANK == LOCAL_RANK)
                & (destination_rank != LOCAL_RANK)
            )
            prefix = tl.cumsum(valid.to(tl.int32), axis=0) - valid.to(tl.int32)
            tl.store(
                descriptor_ids_ptr + write_count + prefix,
                descriptor,
                mask=valid,
            )
            write_count += tl.sum(valid.to(tl.int32), axis=0)


@triton.jit
def _push_one_replica_weight_chunk(
    local_weight_ptr,
    replica_weight_ptr,
    source_base,
    destination_base,
    destination_rank,
    chunk_id,
    WEIGHT_ELEMENTS_PER_EXPERT: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
):
    """Issue one independently schedulable RMA chunk."""
    chunk_start = chunk_id * CHUNK_ELEMENTS
    chunk_elements = tl.minimum(
        CHUNK_ELEMENTS,
        WEIGHT_ELEMENTS_PER_EXPERT - chunk_start,
    )
    destination = replica_weight_ptr + destination_base + chunk_start
    source = local_weight_ptr + source_base + chunk_start
    libshmem_device.putmem(
        destination,
        source,
        chunk_elements * 2,
        destination_rank,
    )


@triton.jit
def _push_one_replica_weight(
    local_weight_ptr,
    replica_weight_ptr,
    source_base,
    destination_base,
    destination_rank,
    WEIGHT_ELEMENTS_PER_EXPERT: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    NUM_WEIGHT_CHUNKS: tl.constexpr,
):
    """Issue every RMA chunk for one replica descriptor."""
    for chunk_id in tl.static_range(0, NUM_WEIGHT_CHUNKS):
        # ``tl.static_range`` unrolls this loop.  Do not annotate the
        # per-iteration value as a named constexpr: Ascend Triton rejects the
        # second unrolled assignment when an expert spans multiple chunks.
        _push_one_replica_weight_chunk(
            local_weight_ptr,
            replica_weight_ptr,
            source_base,
            destination_base,
            destination_rank,
            chunk_id,
            WEIGHT_ELEMENTS_PER_EXPERT,
            CHUNK_ELEMENTS,
        )


@triton.jit
def _push_compact_replica_weight_descriptor(
    descriptor_ordinal,
    local_weight_ptr,
    replica_weight_ptr,
    experts_to_copy_ptr,
    descriptor_ids_ptr,
    ready_mem_ptr,
    signal_epoch,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    WEIGHT_ELEMENTS_PER_EXPERT: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    NUM_WEIGHT_CHUNKS: tl.constexpr,
):
    """Push one complete compact descriptor and publish remote readiness."""
    descriptor = tl.load(descriptor_ids_ptr + descriptor_ordinal)
    destination_rank = descriptor // EXPERTS_PER_RANK
    replica_slot = descriptor % EXPERTS_PER_RANK
    global_expert = tl.load(experts_to_copy_ptr + descriptor)
    home_slot = global_expert % EXPERTS_PER_RANK
    source_base = home_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
    destination_base = replica_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
    _push_one_replica_weight(
        local_weight_ptr,
        replica_weight_ptr,
        source_base,
        destination_base,
        destination_rank,
        WEIGHT_ELEMENTS_PER_EXPERT,
        CHUNK_ELEMENTS,
        NUM_WEIGHT_CHUNKS,
    )
    libshmem_device.fence()
    libshmem_device.signal_op(
        ready_mem_ptr + replica_slot * 16,
        signal_epoch,
        libshmem_device.ACLSHMEM_SIGNAL_SET,
        destination_rank,
    )


@triton.jit
def _push_compact_replica_weight_descriptors(
    pid,
    num_programs: tl.constexpr,
    local_weight_ptr,
    replica_weight_ptr,
    experts_to_copy_ptr,
    descriptor_ids_ptr,
    descriptor_count,
    ready_mem_ptr,
    signal_epoch,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    WEIGHT_ELEMENTS_PER_EXPERT: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    NUM_WEIGHT_CHUNKS: tl.constexpr,
):
    """Statically stripe a compact owner-local descriptor list."""
    descriptor_ordinal = pid
    while descriptor_ordinal < descriptor_count:
        _push_compact_replica_weight_descriptor(
            descriptor_ordinal,
            local_weight_ptr,
            replica_weight_ptr,
            experts_to_copy_ptr,
            descriptor_ids_ptr,
            ready_mem_ptr,
            signal_epoch,
            LOCAL_RANK,
            EXPERTS_PER_RANK,
            WEIGHT_ELEMENTS_PER_EXPERT,
            CHUNK_ELEMENTS,
            NUM_WEIGHT_CHUNKS,
        )
        descriptor_ordinal += num_programs


@triton.jit
def _kernel_replica_weight_prefetch_barrier():
    """Fence all weight puts with the backend-required barrier-sized grid."""
    libshmem_device.barrier_all_vec()


__all__ = [
    "_kernel_replica_weight_prefetch_barrier",
]
