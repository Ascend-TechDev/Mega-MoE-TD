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
import triton_dist.language as dl
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
def _push_compact_replica_weight_descriptors_mte(
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
    """MTE put-with-notify variant (2026-09-20, user direction): prefix
    chunks are putmem_nbi, the final chunk is ONE putmem_signal_nbi WQE
    carrying payload+SET together — the old putmem+fence+signal_op form
    lost the ordering because signal_op rides a DIFFERENT channel than the
    putmem stream (E9 zeros probe: SET landed before payload, the gate/up
    payload overwrote sunk gradients = the 0.47 signature).  Same engine
    family as the P1 dispatch's putmem traffic, so no MTE/UDMA
    cross-engine fault (probe_repush_udma.py context-3, x20 concurrent).
    quiet() drains the prefix stream before the fused final (bench
    put_pull's proven putmem_nbi+quiet pairing)."""
    descriptor_ordinal = pid
    while descriptor_ordinal < descriptor_count:
        descriptor = tl.load(descriptor_ids_ptr + descriptor_ordinal)
        destination_rank = descriptor // EXPERTS_PER_RANK
        replica_slot = descriptor % EXPERTS_PER_RANK
        global_expert = tl.load(experts_to_copy_ptr + descriptor)
        home_slot = global_expert % EXPERTS_PER_RANK
        source_base = home_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
        destination_base = replica_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
        dst = replica_weight_ptr + destination_base
        src = local_weight_ptr + source_base
        for chunk_id in tl.static_range(0, NUM_WEIGHT_CHUNKS - 1):
            offset = chunk_id * CHUNK_ELEMENTS
            libshmem_device.putmem_nbi(
                dst + offset, src + offset, CHUNK_ELEMENTS * 2,
                destination_rank)
        if NUM_WEIGHT_CHUNKS > 1:
            libshmem_device.quiet()
        tail: tl.constexpr = (NUM_WEIGHT_CHUNKS - 1) * CHUNK_ELEMENTS
        libshmem_device.putmem_signal_nbi(
            dst + tail, src + tail,
            (WEIGHT_ELEMENTS_PER_EXPERT - tail) * 2,
            ready_mem_ptr + replica_slot * 16,
            signal_epoch,
            libshmem_device.ACLSHMEM_SIGNAL_SET,
            destination_rank,
        )
        descriptor_ordinal += num_programs


@triton.jit
def _store_remote_panel(dst_ptr, src_ptr, peer,
                        ELEMS: tl.constexpr, BLK: tl.constexpr = 8192):
    """Direct remote stores via dl.symm_at + tl.store (the forward
    dispatch's put_store idiom): no engine on the push side, so no
    MTE/UDMA concurrency hazard."""
    offs = tl.arange(0, BLK)
    remote = dl.symm_at(dst_ptr, peer)
    for c in range(0, ELEMS, BLK):
        v = tl.load(src_ptr + c + offs)
        tl.store(remote + c + offs, v)


@triton.jit
def _push_compact_replica_weight_descriptors_store(
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
        rb_ptr=None,
):
    """Store-path variant of _push_compact_replica_weight_descriptors
    (2026-09-18 root-cause fix): the putmem form let the SET outrun its
    payload on this backend (E9 zeros probe — consumers passed the epoch
    wait on stale content, and the gate/up payload overwrote sunk
    gradients at w2 = the 0.47 signature), and the UDMA put-with-notify
    form faults/wedges when concurrent with the dispatch putmems.
    Direct remote stores + fence + SET is the forward dispatch's proven
    ordering idiom and touches no engine."""
    descriptor_ordinal = pid
    while descriptor_ordinal < descriptor_count:
        descriptor = tl.load(descriptor_ids_ptr + descriptor_ordinal)
        destination_rank = descriptor // EXPERTS_PER_RANK
        replica_slot = descriptor % EXPERTS_PER_RANK
        global_expert = tl.load(experts_to_copy_ptr + descriptor)
        home_slot = global_expert % EXPERTS_PER_RANK
        source_base = home_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
        destination_base = replica_slot.to(tl.int64) * WEIGHT_ELEMENTS_PER_EXPERT
        _store_remote_panel(replica_weight_ptr + destination_base,
                            local_weight_ptr + source_base,
                            destination_rank, WEIGHT_ELEMENTS_PER_EXPERT)
        libshmem_device.fence()
        # (readback gate REMOVED for the payload-vs-verification
        # disambiguation run, 2026-09-20)
        libshmem_device.signal_op(
            ready_mem_ptr + replica_slot * 16,
            signal_epoch,
            libshmem_device.ACLSHMEM_SIGNAL_SET,
            destination_rank,
        )
        descriptor_ordinal += num_programs




@triton.jit
def _kernel_replica_repush_store(
        gu_src_ptr, dn_src_ptr,
        replica_gu_ptr, replica_dn_ptr,
        gate_ready_ptr, down_ready_ptr,
        experts_to_copy_ptr, descriptor_ids_ptr, descriptor_count,
        signal_epoch,
        LOCAL_RANK: tl.constexpr,
        EPR: tl.constexpr,
        GU_ELEMS: tl.constexpr, DN_ELEMS: tl.constexpr,
        rb_ptr,
):
    """Standalone re-push kernel (2026-09-20): the plan-A backward re-push
    moved OUT of the mega kernel — the push code's mere PRESENCE in the
    mega binary trips the w4 whole-kernel miscompile family (A/AN/ANP all
    hang ~30-50% regardless of knobs/transports).  Stream-ordered before
    the mega launch; the consumers' epoch waits observe its SETs like the
    forward's standalone prefetch barrier (proven precedent)."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    _push_compact_replica_weight_descriptors_store(
        pid, num_programs,
        dn_src_ptr, replica_dn_ptr,
        experts_to_copy_ptr, descriptor_ids_ptr, descriptor_count,
        down_ready_ptr, signal_epoch,
        LOCAL_RANK, EPR, DN_ELEMS, rb_ptr=rb_ptr)
    _push_compact_replica_weight_descriptors_store(
        pid, num_programs,
        gu_src_ptr, replica_gu_ptr,
        experts_to_copy_ptr, descriptor_ids_ptr, descriptor_count,
        gate_ready_ptr, signal_epoch,
        LOCAL_RANK, EPR, GU_ELEMS, rb_ptr=rb_ptr)


@triton.jit
def _kernel_replica_weight_prefetch_barrier():
    """Fence all weight puts with the backend-required barrier-sized grid."""
    libshmem_device.barrier_all_vec()



# UDMA put-with-notify bindings — the !59 forward's working primitive family
# (fused_moonep.py's _udma_panel/_single_moonep_push).  Bound via getattr so
# MTE-only installs keep importing this module; the mega_bwd wrapper validates
# availability when MOE_MEGA_REPREFETCH=1.
_udma_put_nbi = getattr(libshmem_device, "udma_put_nbi", None)
_udma_put_signal_nbi = getattr(libshmem_device, "udma_put_signal_nbi", None)
_udma_quiet = getattr(libshmem_device, "udma_quiet", None)


@triton.jit
def _udma_push_panel(destination, source, ready_u64_ptr, epoch, peer,
                     ELEMENTS: tl.constexpr, CHUNK_ELEMENTS: tl.constexpr):
    """One table panel over the peer's QP (fused_moonep.py's _udma_panel):
    WQEs have NO ordering, even on one QP, so prefix chunks are unordered
    put_nbi, a per-peer quiet drains them, and the final chunk carries
    payload+notify in ONE WQE — the SET can never outrun its payload."""
    CHUNKS: tl.constexpr = triton.cdiv(ELEMENTS, CHUNK_ELEMENTS)
    for chunk in range(CHUNKS - 1):
        offset = chunk * CHUNK_ELEMENTS
        _udma_put_nbi(destination + offset, source + offset,
                      CHUNK_ELEMENTS, peer)
    if CHUNKS > 1:
        _udma_quiet(peer)
    tail_offset: tl.constexpr = (CHUNKS - 1) * CHUNK_ELEMENTS
    # epoch arrives as a plain python int when the caller's value was
    # constexpr-specialized (triton specializes ==1) — isinstance resolves at
    # compile time, both arms yield a uint64 scalar.
    if isinstance(epoch, int):
        epoch_u64 = tl.full([], epoch, tl.uint64)
    else:
        epoch_u64 = epoch.to(tl.uint64)
    _udma_put_signal_nbi(
        destination + tail_offset, source + tail_offset,
        ELEMENTS - tail_offset, ready_u64_ptr, epoch_u64, peer)


@triton.jit
def _push_replica_weights_udma_peer(
        peer,
        gu_src_ptr, dn_src_ptr,
        replica_gu_ptr, replica_dn_ptr,
        gate_ready_u64_ptr, down_ready_u64_ptr,
        experts_to_copy_ptr, signal_epoch,
        LOCAL_RANK: tl.constexpr, EPN: tl.constexpr,
        GU_ELEMS: tl.constexpr, GU_CHUNK: tl.constexpr,
        DN_ELEMS: tl.constexpr, DN_CHUNK: tl.constexpr):
    """Push every expert of mine that `peer` replica-cached (ETC row scan —
    the forward's _single_moonep_push ownership: one program owns one peer's
    QP).  Ready slots keep the pooling ABI: 64B per slot (16 int32 == 8
    uint64), the UDMA notify lands in the slot's first two words and the
    consumers dl.wait its low int32 at ready + slot*16."""
    for slot in range(EPN):
        expert = tl.load(experts_to_copy_ptr + peer * EPN + slot)
        if (expert >= 0) & (expert // EPN == LOCAL_RANK):
            local = (expert % EPN).to(tl.int64)
            _udma_push_panel(replica_dn_ptr + slot * DN_ELEMS,
                             dn_src_ptr + local * DN_ELEMS,
                             down_ready_u64_ptr + slot * 8,
                             signal_epoch, peer, DN_ELEMS, DN_CHUNK)
            _udma_push_panel(replica_gu_ptr + slot * GU_ELEMS,
                             gu_src_ptr + local * GU_ELEMS,
                             gate_ready_u64_ptr + slot * 8,
                             signal_epoch, peer, GU_ELEMS, GU_CHUNK)


__all__ = [
    "_kernel_replica_repush_store",
    "_kernel_replica_weight_prefetch_barrier",
]
