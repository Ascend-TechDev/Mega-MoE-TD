# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Device planning and PIPE_S owner pushes in the single forward launch."""

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device

from .moonep_planning import _kernel_moonep_b0_b1

# Older Triton hashes dependencies even in constexpr-dead branches. Bind
# optional APIs without attribute lookups in JIT bodies so MoonEP-off remains
# usable with the existing MTE-only installation. The host validates UDMA-on.
_udma_put_nbi = getattr(libshmem_device, "udma_put_nbi", None)
_udma_put_signal_nbi = getattr(libshmem_device, "udma_put_signal_nbi", None)
_udma_quiet = getattr(libshmem_device, "udma_quiet", None)


@triton.jit
def _single_moonep_b0(counts_ptr, expert_count_ptr, transfers_ptr,
                      R: tl.constexpr, E: tl.constexpr, EPN: tl.constexpr,
                      ROW_STRIDE: tl.constexpr, BLOCK_E: tl.constexpr):
    _kernel_moonep_b0_b1(counts_ptr, expert_count_ptr, transfers_ptr,
                        R, E, EPN, ROW_STRIDE, BLOCK_E, False)
    # MoonEP assumes equal, dropless inputs. Other inputs retain home routing;
    # deciding from the replicated table gives every rank the same fallback.
    total = tl.load(counts_ptr + E)
    balanced_input = True
    for source in range(R):
        valid = 0
        for e0 in range(0, E, BLOCK_E):
            e = e0 + tl.arange(0, BLOCK_E)
            valid += tl.sum(tl.load(counts_ptr + source * ROW_STRIDE + e,
                                   mask=e < E, other=0), 0)
        declared = tl.load(counts_ptr + source * ROW_STRIDE + E)
        balanced_input &= (valid == declared) & (declared == total)
    if not balanced_input:
        r = tl.arange(0, R)
        tl.store(transfers_ptr + r[:, None] * R + r[None, :],
                 tl.zeros((R, R), tl.int64))


@triton.jit
def _single_moonep_scatter(
        pid, selected_ptr, cursors_ptr, raw_counts_ptr, alloc_cumsum_ptr,
        inverse_ptr, send_starts_ptr, send_tokens_ptr, send_routes_ptr,
        route_to_send_ptr, num_routes, NUM_CORES: tl.constexpr,
        LOCAL_RANK: tl.constexpr, R: tl.constexpr, E: tl.constexpr,
        EPN: tl.constexpr, CURSOR_STRIDE: tl.constexpr,
        RAW_STRIDE: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    per_core = tl.cdiv(num_routes, NUM_CORES)
    begin = pid * per_core
    end = tl.minimum(begin + per_core, num_routes)
    lanes = tl.arange(0, BLOCK)
    BIN_BLOCK: tl.constexpr = min(32, triton.next_power_of_2(E))
    for e0 in range(0, E, BIN_BLOCK):
        bins = e0 + tl.arange(0, BIN_BLOCK)
        cursors = tl.load(cursors_ptr + pid * CURSOR_STRIDE + bins,
                          mask=bins < E, other=0)
        for block_start in range(begin, end, BLOCK):
            route = block_start + lanes
            expert = tl.load(selected_ptr + route, mask=route < end, other=-1)
            valid = ((route < end) & (expert >= e0)
                     & (expert < tl.minimum(e0 + BIN_BLOCK, E)))
            safe_expert = tl.minimum(tl.maximum(expert, 0), E - 1)
            matches = ((expert[None, :] == bins[:, None]) & valid[None, :]).to(tl.int32)
            within = tl.cumsum(matches, 1) - matches
            ordinal = tl.sum((cursors[:, None] + within) * matches, 0)
            source_lo = tl.full((BLOCK,), 0, tl.int32)
            for source in tl.static_range(LOCAL_RANK):
                source_lo += tl.load(raw_counts_ptr + source * RAW_STRIDE + safe_expert)
            global_ordinal = ordinal + source_lo
            destination = tl.full((BLOCK,), 0, tl.int32)
            for rank in tl.static_range(R - 1):
                hi = tl.load(alloc_cumsum_ptr + safe_expert * R + rank)
                destination += (global_ordinal >= hi).to(tl.int32)
            previous = tl.maximum(destination - 1, 0)
            allocation_lo = tl.load(alloc_cumsum_ptr + safe_expert * R + previous)
            allocation_lo = tl.where(destination > 0, allocation_lo, 0)
            replica = tl.load(inverse_ptr + destination * E + safe_expert)
            slot = tl.where(destination == safe_expert // EPN,
                            safe_expert % EPN, EPN + replica)
            bucket = destination * (2 * EPN) + slot
            start = tl.load(send_starts_ptr + bucket)
            row = start + global_ordinal - tl.maximum(source_lo, allocation_lo)
            tl.store(send_tokens_ptr + row, route // TOPK, mask=valid)
            tl.store(send_routes_ptr + row, route, mask=valid)
            tl.store(route_to_send_ptr + route, row, mask=valid)
            cursors += tl.sum(matches, 1)


@triton.jit
def _udma_panel(destination, source, ready, epoch, peer,
                ELEMENTS: tl.constexpr, CHUNK_ELEMENTS: tl.constexpr):
    CHUNKS: tl.constexpr = triton.cdiv(ELEMENTS, CHUNK_ELEMENTS)
    for chunk in range(CHUNKS - 1):
        offset = chunk * CHUNK_ELEMENTS
        _udma_put_nbi(destination + offset, source + offset,
                                     CHUNK_ELEMENTS, peer)
    if CHUNKS > 1:
        # WQEs have NO ordering, even on one QP. A final-chunk notification
        # proves panel readiness only after all prefix chunks have completed.
        _udma_quiet(peer)
    tail_offset: tl.constexpr = (CHUNKS - 1) * CHUNK_ELEMENTS
    _udma_put_signal_nbi(
        destination + tail_offset, source + tail_offset, ELEMENTS - tail_offset,
        ready, epoch.to(tl.uint64), peer)


@triton.jit
def _single_moonep_push(
        peer, gate_up_ptr, down_ptr, replica_gate_ptr, replica_down_ptr,
        gate_ready_ptr, down_ready_ptr, experts_to_copy_ptr, signal_epoch,
        LOCAL_RANK: tl.constexpr, EPN: tl.constexpr, HIDDEN: tl.constexpr,
        FFN: tl.constexpr, CHUNK_ELEMENTS: tl.constexpr):
    # Exactly one Vector owns each peer's QP. PIPE_S submission leaves both
    # Vector subcores available for the existing UB activation pipeline.
    for panel in tl.static_range(3):
        for slot in range(EPN):
            expert = tl.load(experts_to_copy_ptr + peer * EPN + slot)
            if (expert >= 0) & (expert // EPN == LOCAL_RANK):
                local = expert % EPN
                if panel == 0:
                    elements: tl.constexpr = 2 * HIDDEN * FFN
                    _udma_panel(replica_gate_ptr + slot * elements,
                                gate_up_ptr + local * elements,
                                gate_ready_ptr + slot * 8, signal_epoch,
                                peer, elements, CHUNK_ELEMENTS)
                else:
                    elements: tl.constexpr = HIDDEN * FFN // 2
                    _udma_panel(replica_down_ptr + (2 * slot + panel - 1) * elements,
                                down_ptr + (2 * local + panel - 1) * elements,
                                down_ready_ptr + (2 * slot + panel - 1) * 8,
                                signal_epoch, peer, elements, CHUNK_ELEMENTS)
