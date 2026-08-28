# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Owner-pull ACLSHMEM reduction of MoonEP replica expert weight gradients.

The physical (M2) backward produces one weight gradient per *physical* slot
``[home | replica]``.  A replica slot's gradient belongs to the rank that owns
the logical expert living in that slot, so it has to travel back to the owner.
Instead of allocating a second symmetric pair of tables, M3 reuses the forward's
replica *weight* tables (:class:`mega_moe.runtime.replica_weight_prefetch.ReplicaWeightBuffers`):
each rank sinks its replica weight gradients into its own symmetric slots, and
every owner then pulls the slots that hold copies of its home experts back with
``getmem`` and accumulates them, in ``(peer, slot)`` lexicographic order, in
fp32 onto its home weight-gradient seed.  That is exactly the reduction the
test-side oracle performs over HCCL, so the two can be cross-checked.

The stage order every EP rank must execute identically is

    sink -> barrier #1 -> owner-pull -> barrier #2 -> zero consumed slots
         -> barrier #3

with all three barriers launched unconditionally: a rank that owns no replica
and holds none still has to arrive, otherwise the collective hangs.
"""

import torch
import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device


# Element slices pulled per accumulate step.  Small enough to keep the bf16
# staging read plus the fp32 load/store pair well inside Unified Buffer.
_ACC_BLOCK = 4096


@triton.jit
def _kernel_replica_grad_barrier():
    """Fence replica-gradient RMA with the backend-required barrier-sized grid.

    ``barrier_all_vec`` is tied to one participating Vector block per AI Core,
    so this must stay a separate AICore-sized launch — never embedded in the
    owner-pull grid (mirrors ``_kernel_replica_weight_prefetch_barrier`` and
    ``_kernel_combine_fc1_bwd_barrier``).
    """
    libshmem_device.barrier_all_vec()


@triton.jit
def _accumulate_pulled_chunk(
    acc_ptr,
    source_ptr,
    home_expert,
    count,
    ELEMENTS_PER_EXPERT: tl.constexpr,
    ACC_BLOCK: tl.constexpr,
):
    """acc[home_expert] += source[0:count] in fp32, one ACC_BLOCK slice at a time.

    Both sides are walked contiguously: the accumulator is laid out exactly
    like the symmetric table it reduces (``[H, 2F]`` gate/up, ``[H, F]`` down),
    so the host-side seed and result carry the layout conversion instead of the
    kernel.  A strided (transposing) accumulate was measured to drop a small,
    run-varying subset of elements on this backend, which the contiguous walk
    does not.
    """
    offsets = tl.arange(0, ACC_BLOCK)
    acc_base = home_expert.to(tl.int64) * ELEMENTS_PER_EXPERT
    for start in range(0, count, ACC_BLOCK):
        local = start + offsets
        mask = local < count
        value = tl.load(source_ptr + local, mask=mask, other=0.0).to(tl.float32)
        acc_offsets = acc_base + local.to(tl.int64)
        accumulated = tl.load(acc_ptr + acc_offsets, mask=mask, other=0.0)
        tl.store(acc_ptr + acc_offsets, accumulated + value, mask=mask)


@triton.jit
def _pull_and_accumulate_one_table(
    acc_ptr,
    slot_table_ptr,
    staging_ptr,
    peer,
    slot,
    home_expert,
    LOCAL_RANK: tl.constexpr,
    ELEMENTS_PER_EXPERT: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    ACC_BLOCK: tl.constexpr,
):
    """Pull one replica slot of one table and fp32-accumulate it on the owner.

    A self-owned slot (``peer == LOCAL_RANK``) is read directly — the planner
    currently forbids self-copies, so this is only a defensive path that keeps
    the kernel's semantics identical to the test-side oracle.  Remote slots are
    pulled chunk by chunk into the staging workspace; ``getmem`` is blocking, so
    the accumulate may read the staging buffer as soon as it returns.
    """
    slot_base = slot.to(tl.int64) * ELEMENTS_PER_EXPERT
    for chunk_id in tl.static_range(0, NUM_CHUNKS):
        # ``tl.static_range`` unrolls this loop; keep ``chunk_id`` an anonymous
        # value (a named constexpr breaks the second unrolled assignment).
        chunk_start = chunk_id * CHUNK_ELEMENTS
        count = tl.minimum(CHUNK_ELEMENTS, ELEMENTS_PER_EXPERT - chunk_start)
        if count > 0:
            source = slot_table_ptr + slot_base + chunk_start
            if peer == LOCAL_RANK:
                _accumulate_pulled_chunk(
                    acc_ptr, source, home_expert, count,
                    ELEMENTS_PER_EXPERT, ACC_BLOCK,
                )
            else:
                libshmem_device.getmem(staging_ptr, source, count * 2, peer)
                _accumulate_pulled_chunk(
                    acc_ptr, staging_ptr, home_expert, count,
                    ELEMENTS_PER_EXPERT, ACC_BLOCK,
                )


@triton.jit
def _kernel_owner_pull_accumulate(
    pid,
    num_programs,
    acc_gate_up_ptr,
    acc_down_ptr,
    gate_up_slot_ptr,
    down_slot_ptr,
    desc_peer_ptr,
    desc_slot_ptr,
    desc_home_ptr,
    desc_count,
    staging_gate_up_ptr,
    staging_down_ptr,
    LOCAL_RANK: tl.constexpr,
    GATE_UP_ELEMENTS_PER_EXPERT: tl.constexpr,
    GATE_UP_CHUNK_ELEMENTS: tl.constexpr,
    GATE_UP_NUM_CHUNKS: tl.constexpr,
    DOWN_ELEMENTS_PER_EXPERT: tl.constexpr,
    DOWN_CHUNK_ELEMENTS: tl.constexpr,
    DOWN_NUM_CHUNKS: tl.constexpr,
    ACC_BLOCK: tl.constexpr,
):
    """Walk the owner's descriptor list and pull every replica slot it names.

    The list is ``(peer, slot)``-ordered, so a single-program launch reproduces
    the lexicographic fp32 accumulation of the reference reduction exactly.
    Several ranks may replicate the same logical expert, which makes the
    addition order load-bearing (fp32 is not associative); the launcher
    therefore pins ``num_programs`` to one until the accumulation is
    partitioned by home expert instead of by descriptor.
    """
    for ordinal in range(pid, desc_count, num_programs):
        peer = tl.load(desc_peer_ptr + ordinal)
        slot = tl.load(desc_slot_ptr + ordinal)
        home_expert = tl.load(desc_home_ptr + ordinal)
        _pull_and_accumulate_one_table(
            acc_gate_up_ptr, gate_up_slot_ptr, staging_gate_up_ptr,
            peer, slot, home_expert,
            LOCAL_RANK,
            GATE_UP_ELEMENTS_PER_EXPERT, GATE_UP_CHUNK_ELEMENTS,
            GATE_UP_NUM_CHUNKS, ACC_BLOCK,
        )
        _pull_and_accumulate_one_table(
            acc_down_ptr, down_slot_ptr, staging_down_ptr,
            peer, slot, home_expert,
            LOCAL_RANK,
            DOWN_ELEMENTS_PER_EXPERT, DOWN_CHUNK_ELEMENTS, DOWN_NUM_CHUNKS,
            ACC_BLOCK,
        )


def launch_replica_grad_barrier(num_programs: int) -> None:
    """Queue one AICore-sized collective fence for the replica grad transport."""
    _kernel_replica_grad_barrier[(num_programs, 1, 1)]()


def build_owner_pull_descriptors(experts_to_copy_cpu, rank, experts_per_rank):
    """Return ``(peer, slot, home_expert)`` CPU int32 tensors for one owner.

    Scanning the complete ``[world_size, replica_budget]`` table in row-major
    order yields the descriptors already sorted by ``(peer, slot)``.  A slot is
    named when it holds one of this rank's home experts; the count may legally
    be zero (nobody replicated this rank's experts), and self-owned slots are
    kept in the list so the kernel can take its local-read branch.
    """
    if experts_to_copy_cpu.device.type != "cpu":
        raise ValueError("experts_to_copy_cpu must be a CPU tensor")
    if experts_to_copy_cpu.dtype != torch.int32:
        raise ValueError("experts_to_copy_cpu must use torch.int32")
    if experts_to_copy_cpu.ndim != 2:
        raise ValueError("experts_to_copy_cpu must have shape [world_size, B]")
    world_size, replica_slots = experts_to_copy_cpu.shape
    if replica_slots != experts_per_rank:
        raise ValueError(
            "the replica budget must equal experts_per_rank, got "
            f"{replica_slots} slots for {experts_per_rank} home experts"
        )
    owner_start = rank * experts_per_rank
    owner_end = owner_start + experts_per_rank
    peers: list[int] = []
    slots: list[int] = []
    homes: list[int] = []
    for peer, row in enumerate(experts_to_copy_cpu.tolist()):
        for slot, expert in enumerate(row):
            if owner_start <= expert < owner_end:
                peers.append(peer)
                slots.append(slot)
                homes.append(expert - owner_start)

    def _as_tensor(values):
        if values:
            return torch.tensor(values, dtype=torch.int32)
        return torch.empty(0, dtype=torch.int32)

    return _as_tensor(peers), _as_tensor(slots), _as_tensor(homes)


def zero_consumed_replica_slots(buffers, consumed_slots) -> None:
    """Zero the replica slots this rank sank a gradient into, in both tables.

    Runs on the current stream between barrier #2 and barrier #3 so the next
    forward's owner-push can never race a stale gradient left in a slot.
    """
    if not consumed_slots:
        return
    index = torch.tensor(
        list(consumed_slots), dtype=torch.int64, device=buffers.gate_up.device
    )
    buffers.gate_up.index_fill_(0, index, 0)
    buffers.down.index_fill_(0, index, 0)


def launch_owner_pull_accumulate(
    *,
    acc_gate_up,
    acc_down,
    gate_up_table,
    down_table,
    desc_peer,
    desc_slot,
    desc_home,
    rank,
    gate_up_chunk_elements,
    gate_up_num_chunks,
    down_chunk_elements,
    down_num_chunks,
    num_programs=1,
    acc_block=_ACC_BLOCK,
):
    """Launch the owner-pull accumulation over one descriptor list.

    The staging workspace is sized to one chunk (never more than one full
    expert), so a single-program launch reuses it for every descriptor.  The
    accumulators must already share the tables' per-expert layout, because the
    kernel accumulates contiguously in that layout.
    """
    if num_programs != 1:
        raise ValueError(
            "the owner-pull accumulation must launch as a single program while "
            "the fp32 accumulation order follows the descriptor list"
        )
    for name, accumulator in (
        ("acc_gate_up", acc_gate_up), ("acc_down", acc_down)
    ):
        if accumulator.dtype != torch.float32 or not accumulator.is_contiguous():
            raise ValueError(f"{name} must be a contiguous fp32 tensor")
    for name, table, accumulator in (
        ("gate_up_table", gate_up_table, acc_gate_up),
        ("down_table", down_table, acc_down),
    ):
        if table.dtype != torch.bfloat16 or not table.is_contiguous():
            raise ValueError(f"{name} must be a contiguous bf16 table")
        if table.device != accumulator.device or tuple(table.shape) != tuple(
            accumulator.shape
        ):
            raise ValueError(
                f"{name} has shape {tuple(table.shape)} on {table.device}, "
                f"expected {tuple(accumulator.shape)} on {accumulator.device}"
            )
    for name, descriptor in (
        ("desc_peer", desc_peer), ("desc_slot", desc_slot),
        ("desc_home", desc_home),
    ):
        if descriptor.dtype != torch.int32 or descriptor.ndim != 1:
            raise ValueError(f"{name} must be a 1-D int32 tensor")
        if descriptor.device != acc_gate_up.device:
            raise ValueError(f"{name} must live on the accumulator device")
    if desc_peer.numel() != desc_slot.numel() or desc_peer.numel() != (
        desc_home.numel()
    ):
        raise ValueError("the descriptor columns must have equal length")

    device = acc_gate_up.device
    gate_up_elements = int(acc_gate_up[0].numel())
    down_elements = int(acc_down[0].numel())
    staging_gate_up = torch.empty(
        min(gate_up_chunk_elements, gate_up_elements),
        dtype=torch.bfloat16,
        device=device,
    )
    staging_down = torch.empty(
        min(down_chunk_elements, down_elements),
        dtype=torch.bfloat16,
        device=device,
    )
    _kernel_owner_pull_accumulate[(num_programs, 1, 1)](
        0,
        num_programs,
        acc_gate_up,
        acc_down,
        gate_up_table,
        down_table,
        desc_peer,
        desc_slot,
        desc_home,
        int(desc_peer.numel()),
        staging_gate_up,
        staging_down,
        LOCAL_RANK=rank,
        GATE_UP_ELEMENTS_PER_EXPERT=gate_up_elements,
        GATE_UP_CHUNK_ELEMENTS=gate_up_chunk_elements,
        GATE_UP_NUM_CHUNKS=gate_up_num_chunks,
        DOWN_ELEMENTS_PER_EXPERT=down_elements,
        DOWN_CHUNK_ELEMENTS=down_chunk_elements,
        DOWN_NUM_CHUNKS=down_num_chunks,
        ACC_BLOCK=acc_block,
    )


__all__ = [
    "build_owner_pull_descriptors",
    "launch_owner_pull_accumulate",
    "launch_replica_grad_barrier",
    "zero_consumed_replica_slots",
]
