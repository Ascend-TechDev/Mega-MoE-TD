# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Device-side MoonEP B.0-B.3 planning for Mega-MoE routing.

The original MoonEP planning implementation builds seven transport-specific
tables.  Mega's dispatch path needs only ``alloc_cumsum`` and
``experts_to_copy``; an inverse replica table is emitted at the same time to
avoid a host-side inversion pass.

Unlike the bring-up implementation, the kernels below pad Triton vector axes
explicitly.  Production expert counts such as E=896 and E/R=112 or 224 are
therefore supported even though they are not powers of two.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


@triton.jit
def _kernel_moonep_b0_b1(
    tpe_all_ptr,
    expert_count_ptr,
    transfers_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    EPN: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    FINAL_BARRIER: tl.constexpr = True,
):
    """Build MoonEP global counts and the B.1 transfer matrix."""
    offs_r = tl.arange(0, R)

    # B.0: global expert counts and home-group loads.  The caller validates
    # the published route totals once, before entering this collective kernel.
    group_tokens = tl.zeros((R,), dtype=tl.int64)
    for e0 in tl.static_range(0, E, BLOCK_E):
        offs_e = e0 + tl.arange(0, BLOCK_E)
        expert_count = tl.zeros((BLOCK_E,), dtype=tl.int64)
        for source in tl.static_range(0, R):
            counts = tl.load(
                tpe_all_ptr + source * ROW_STRIDE + offs_e
            ).to(tl.int64)
            expert_count += counts
        tl.store(expert_count_ptr + offs_e, expert_count)
        home = offs_e // EPN
        group_tokens += tl.sum(
            tl.where(
                offs_r[:, None] == home[None, :],
                expert_count[None, :],
                0,
            ),
            axis=1,
        )
    capacity = tl.load(tpe_all_ptr + E).to(tl.int64)

    # B.1: greedily fill the most underloaded destination from the most
    # overloaded home group.  Both argmax and argmin ties choose low rank ids.
    balance = group_tokens - capacity
    transfers = tl.zeros((R, R), dtype=tl.int64)
    if R > 1:
        active = True
        for _ in tl.static_range(0, R - 1):
            largest = tl.max(balance, axis=0)
            smallest = tl.min(balance, axis=0)
            step = active & (largest > 0) & (smallest < 0)
            source = tl.min(
                tl.where(balance == largest, offs_r, R), axis=0
            )
            destination = tl.min(
                tl.where(balance == smallest, offs_r, R), axis=0
            )
            moved = -smallest
            transfers = tl.where(
                step
                & (offs_r[:, None] == source)
                & (offs_r[None, :] == destination),
                moved,
                transfers,
            )
            balance = tl.where(
                step & (offs_r == source), balance - moved, balance
            )
            balance = tl.where(step & (offs_r == destination), 0, balance)
            active = step

    tl.store(
        transfers_ptr + offs_r[:, None] * R + offs_r[None, :], transfers
    )
    # This is the last planner access to the symmetric input.  Do not let a
    # faster rank publish its next tpe row while a peer still reads this one.
    if FINAL_BARRIER:
        libshmem_device.barrier_all_vec()


@triton.jit
def _kernel_moonep_b2(
    expert_count_ptr,
    transfers_ptr,
    allocation_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    EPN: tl.constexpr,
    BLE: tl.constexpr,
):
    """Cut one home group per program according to MoonEP B.2."""
    home = tl.program_id(axis=0)
    offs_r = tl.arange(0, R)
    offs_le = tl.arange(0, BLE)
    valid_le = offs_le < EPN
    expert_ids = home * EPN + offs_le
    initial_counts = tl.load(
        expert_count_ptr + expert_ids,
        mask=valid_le & (expert_ids < E),
        other=0,
    )
    # Global scratch avoids carrying vector selects through a dynamic scf.for,
    # which the current Ascend block-pointer pass cannot lower.  Each program
    # owns one disjoint home-group column slice and one transfer-matrix row.
    tl.store(
        allocation_ptr
        + offs_r[:, None] * E
        + expert_ids[None, :],
        tl.zeros((R, BLE), dtype=tl.int64),
        mask=valid_le[None, :] & (expert_ids[None, :] < E),
    )
    tl.store(
        allocation_ptr + home * E + expert_ids,
        initial_counts,
        mask=valid_le & (expert_ids < E),
    )
    if R > 1:
        # Every successful step exhausts either one destination quota or one
        # expert remainder, so R+EPN iterations are sufficient.  Completed
        # iterations leave all state untouched through scalar store masks.
        for _ in range(R + EPN):
            quotas = tl.load(transfers_ptr + home * R + offs_r)
            remaining = tl.load(
                allocation_ptr + home * E + expert_ids,
                mask=valid_le & (expert_ids < E),
                other=-1,
            )
            largest_quota = tl.max(quotas, axis=0)
            largest_remaining = tl.max(remaining, axis=0)
            step = (largest_quota > 0) & (largest_remaining > 0)
            destination = tl.min(
                tl.where(quotas == largest_quota, offs_r, R), axis=0
            )
            local_expert = tl.min(
                tl.where(
                    remaining == largest_remaining, offs_le, BLE
                ),
                axis=0,
            )
            moved = tl.minimum(largest_quota, largest_remaining)
            safe_destination = tl.where(step, destination, 0)
            safe_local_expert = tl.where(step, local_expert, 0)
            allocation_offset = (
                safe_destination * E + home * EPN + safe_local_expert
            )
            previous = tl.load(
                allocation_ptr + allocation_offset, mask=step, other=0
            )
            tl.store(
                allocation_ptr + allocation_offset,
                previous + moved,
                mask=step,
            )
            tl.store(
                allocation_ptr + home * E + home * EPN + safe_local_expert,
                largest_remaining - moved,
                mask=step,
            )
            tl.store(
                transfers_ptr + home * R + safe_destination,
                largest_quota - moved,
                mask=step,
            )


@triton.jit
def _kernel_moonep_alloc_cumsum(
    allocation_ptr,
    alloc_cumsum_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Transpose the inclusive destination prefix into Mega's [E,R]."""
    offs_e = tl.program_id(axis=0) * BLOCK_E + tl.arange(0, BLOCK_E)
    valid_e = offs_e < E
    running = tl.zeros((BLOCK_E,), dtype=tl.int64)
    for destination in tl.static_range(0, R):
        allocation = tl.load(
            allocation_ptr + destination * E + offs_e,
            mask=valid_e,
            other=0,
        )
        running += allocation
        tl.store(
            alloc_cumsum_ptr + offs_e * R + destination,
            running.to(tl.int32),
            mask=valid_e,
        )


@triton.jit
def _kernel_moonep_b3(
    allocation_ptr,
    experts_to_copy_ptr,
    inverse_ptr,
    replica_counts_ptr,
    R: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    EPN: tl.constexpr,
    B: tl.constexpr,
):
    """Select remote experts and build ETC plus its inverse in one pass."""
    destination = tl.program_id(axis=0)
    offs_e = tl.arange(0, BE)
    valid_e = offs_e < E
    is_home = (offs_e >= destination * EPN) & (
        offs_e < (destination + 1) * EPN
    )
    remote_counts = tl.load(
        allocation_ptr + destination * E + offs_e,
        mask=valid_e & ~is_home,
        other=0,
    )
    replica_count = tl.sum((remote_counts > 0).to(tl.int32), axis=0)
    tl.store(replica_counts_ptr + destination, replica_count)

    # Taking the top-B of count*(E+1)+expert_id reproduces both MoonEP
    # rules: larger allocations win, and equal allocations choose the larger
    # expert. Keys are unique, so iterative argmax equals a descending sort;
    # al.sort is avoided because its device runtime only links on A5.
    key_base: tl.constexpr = E + 1
    keys = tl.where(
        remote_counts > 0,
        remote_counts * key_base + offs_e,
        -1,
    )
    slot_of = tl.full((BE,), -1, tl.int32)
    for slot in range(B):
        best = tl.max(keys, axis=0)
        top = tl.argmax(keys, axis=0)
        expert = tl.where(best >= 0, top, -1)
        tl.store(
            experts_to_copy_ptr + destination * B + slot,
            expert.to(tl.int32),
        )
        # Record the slot and knock the winner out arithmetically: vector
        # tl.where and data-dependent scalar stores inside a loop are both
        # rejected by this build's TritonToLinalg pass.
        hit = ((offs_e == top) & (best >= 0)).to(tl.int64)
        slot_of = slot_of + hit.to(tl.int32) * (slot - slot_of)
        keys = keys - hit * (keys + 1)
    tl.store(
        inverse_ptr + destination * E + offs_e,
        slot_of,
        mask=valid_e,
    )


def launch_moonep_b0_b3(
    tpe_all: torch.Tensor,
    expert_count: torch.Tensor,
    transfers: torch.Tensor,
    allocation: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    experts_to_copy: torch.Tensor,
    inverse_experts_to_copy: torch.Tensor,
    replica_counts: torch.Tensor,
    *,
    world_size: int,
    num_experts: int,
    experts_per_rank: int,
    row_stride: int,
) -> None:
    """Queue device-side MoonEP B.0-B.3 on the current NPU stream."""
    replica_budget = experts_per_rank
    if num_experts % world_size or experts_per_rank != num_experts // world_size:
        raise ValueError("MoonEP requires E to be divisible by world size")
    expected = {
        "tpe_all": ((world_size, row_stride), torch.int32),
        "expert_count": ((num_experts,), torch.int64),
        "transfers": ((world_size, world_size), torch.int64),
        "allocation": ((world_size, num_experts), torch.int64),
        "alloc_cumsum": ((num_experts, world_size), torch.int32),
        "experts_to_copy": ((world_size, replica_budget), torch.int32),
        "inverse_experts_to_copy": ((world_size, num_experts), torch.int32),
        "replica_counts": ((world_size,), torch.int32),
    }
    tensors = {
        "tpe_all": tpe_all,
        "expert_count": expert_count,
        "transfers": transfers,
        "allocation": allocation,
        "alloc_cumsum": alloc_cumsum,
        "experts_to_copy": experts_to_copy,
        "inverse_experts_to_copy": inverse_experts_to_copy,
        "replica_counts": replica_counts,
    }
    device = tpe_all.device
    for name, tensor in tensors.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} must have shape {shape} and dtype {dtype}"
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {device}")
    if row_stride <= num_experts:
        raise ValueError("tpe_all row stride must include the route-count bin")
    if world_size & (world_size - 1):
        raise ValueError("MoonEP Triton planning requires power-of-two world size")

    block_e = min(32, _next_pow2(num_experts))
    while num_experts % block_e:
        block_e //= 2
    _kernel_moonep_b0_b1[(1, 1, 1)](
        tpe_all,
        expert_count,
        transfers,
        R=world_size,
        E=num_experts,
        EPN=experts_per_rank,
        ROW_STRIDE=row_stride,
        BLOCK_E=block_e,
    )
    _kernel_moonep_b2[(world_size, 1, 1)](
        expert_count,
        transfers,
        allocation,
        R=world_size,
        E=num_experts,
        EPN=experts_per_rank,
        BLE=_next_pow2(experts_per_rank),
    )
    _kernel_moonep_alloc_cumsum[
        (triton.cdiv(num_experts, block_e), 1, 1)
    ](
        allocation,
        alloc_cumsum,
        R=world_size,
        E=num_experts,
        BLOCK_E=block_e,
    )
    _kernel_moonep_b3[(world_size, 1, 1)](
        allocation,
        experts_to_copy,
        inverse_experts_to_copy,
        replica_counts,
        R=world_size,
        E=num_experts,
        BE=_next_pow2(num_experts),
        EPN=experts_per_rank,
        B=replica_budget,
    )


__all__ = ["launch_moonep_b0_b3"]
