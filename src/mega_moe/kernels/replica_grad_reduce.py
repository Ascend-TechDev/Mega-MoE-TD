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

Two implementations of the cross-rank chain live here:

* the *legacy* path — three standalone barrier launches around the
  single-program owner-pull kernel and a host-side zero (:func:`launch_replica_grad_barrier`
  + :func:`launch_owner_pull_accumulate` + :func:`zero_consumed_replica_slots`);
* the *fused* path (:func:`launch_grad_reduce_transport`) — the whole
  ``barrier #1 -> owner-pull -> barrier #2 -> zero -> barrier #3`` chain as
  ONE kernel launch on the AICore-sized barrier grid, with the owner-pull
  partitioned by home expert so the per-expert ``(peer, slot)`` accumulation
  order (and therefore the fp32 result) stays bit-identical to the legacy
  single-program walk.  This mirrors the persistent-CTA ``GradReduceKernel``
  shape of the MoonEP reference (one kernel, internal grid barriers, local
  zeroing after the readers' barrier); the in-kernel ``barrier_all_vec``
  between real work phases follows the planning-kernel precedent.
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


@triton.jit
def _fused_accumulate_chunk(
    acc_ptr,
    source_ptr,
    home_expert,
    count,
    elements_per_expert,
    ACC_BLOCK: tl.constexpr,
):
    """acc[home_expert] += source[0:count] in fp32, contiguous walk.

    Runtime-elements twin of :func:`_accumulate_pulled_chunk`: the fused
    kernel drives its chunk loop with a runtime bound so the pull body is not
    unrolled once per (large) expert table, which keeps the JIT'd kernel
    small for the Kimi-sized expert shapes.
    """
    offsets = tl.arange(0, ACC_BLOCK)
    acc_base = home_expert.to(tl.int64) * elements_per_expert
    for start in range(0, count, ACC_BLOCK):
        local = start + offsets
        mask = local < count
        value = tl.load(source_ptr + local, mask=mask, other=0.0).to(tl.float32)
        acc_offsets = acc_base + local.to(tl.int64)
        accumulated = tl.load(acc_ptr + acc_offsets, mask=mask, other=0.0)
        tl.store(acc_ptr + acc_offsets, accumulated + value, mask=mask)


@triton.jit
def _fused_pull_one_table(
    acc_ptr,
    slot_table_ptr,
    staging_row_ptr,
    peer,
    slot,
    home_expert,
    elements_per_expert,
    LOCAL_RANK: tl.constexpr,
    CHUNK_ELEMENTS: tl.constexpr,
    ACC_BLOCK: tl.constexpr,
):
    """Pull one replica slot of one table into this program's staging row.

    Same chunked ``getmem`` -> staging -> fp32 accumulate contract as
    :func:`_pull_and_accumulate_one_table`, except the chunk trip count is
    dynamic (runtime ``elements_per_expert``) and the staging buffer is this
    program's private row, so every program of the fused grid can pull a
    different home expert concurrently without a staging race.
    """
    slot_base = slot.to(tl.int64) * elements_per_expert
    for chunk_start in range(0, elements_per_expert, CHUNK_ELEMENTS):
        count = tl.minimum(CHUNK_ELEMENTS, elements_per_expert - chunk_start)
        if count > 0:
            source = slot_table_ptr + slot_base + chunk_start
            if peer == LOCAL_RANK:
                _fused_accumulate_chunk(
                    acc_ptr, source, home_expert, count,
                    elements_per_expert, ACC_BLOCK,
                )
            else:
                libshmem_device.getmem(staging_row_ptr, source, count * 2, peer)
                _fused_accumulate_chunk(
                    acc_ptr, staging_row_ptr, home_expert, count,
                    elements_per_expert, ACC_BLOCK,
                )


@triton.jit
def _fused_zero_rows(
    slot_table_ptr,
    slot,
    elements_per_expert,
    ACC_BLOCK: tl.constexpr,
):
    """Zero one slot row of one symmetric table with contiguous stores."""
    base = slot.to(tl.int64) * elements_per_expert
    offsets = tl.arange(0, ACC_BLOCK)
    zeros = tl.zeros((ACC_BLOCK,), dtype=slot_table_ptr.dtype.element_ty)
    for start in range(0, elements_per_expert, ACC_BLOCK):
        local = start + offsets
        mask = local < elements_per_expert
        tl.store(slot_table_ptr + base + local.to(tl.int64), zeros, mask=mask)


@triton.jit
def _kernel_grad_reduce_transport(
    pid,
    num_programs,
    acc_gate_up_ptr,
    acc_down_ptr,
    gate_up_slot_ptr,
    down_slot_ptr,
    desc_peer_ptr,
    desc_slot_ptr,
    desc_home_ptr,
    home_offsets_ptr,
    desc_count,
    consumed_ptr,
    consumed_count,
    staging_gate_up_ptr,
    staging_down_ptr,
    gate_up_elements_per_expert,
    down_elements_per_expert,
    LOCAL_RANK: tl.constexpr,
    EPN: tl.constexpr,
    GATE_UP_CHUNK_ELEMENTS: tl.constexpr,
    DOWN_CHUNK_ELEMENTS: tl.constexpr,
    ACC_BLOCK: tl.constexpr,
):
    """One launch for the whole cross-rank replica grad transport.

    Phases, mirroring the reference persistent ``GradReduceKernel``:

    * barrier #1 publishes every rank's sunk slots (the host-side sink copies
      are already stream-ordered ahead of this kernel);
    * the owner-pull walks the *by-home-expert* descriptor list: program
      ``pid`` owns home experts ``pid, pid + num_programs, ...``, and each
      expert's descriptors stay in ``(peer, slot)`` lexicographic order, so
      the fp32 accumulation is bit-identical to the legacy single-program
      walk (per-expert contributions are disjoint rows of the accumulator,
      which is what makes the partition legal for non-associative fp32);
    * barrier #2 retires every reader before any slot is cleared;
    * each rank zeroes only the slots it sank a gradient into;
    * barrier #3 publishes the zeroing so the next forward's owner-push can
      never race a stale gradient.

    The grid MUST be the AICore-sized barrier grid — every program of every
    rank arrives at all three ``barrier_all_vec`` calls unconditionally, even
    when this rank owns no replica and holds none (empty phase loops).
    """
    staging_gate_up_row = staging_gate_up_ptr + pid.to(tl.int64) * (
        GATE_UP_CHUNK_ELEMENTS
    )
    staging_down_row = staging_down_ptr + pid.to(tl.int64) * DOWN_CHUNK_ELEMENTS
    # barrier #1: all ranks' sunk slots are visible to every peer.
    libshmem_device.barrier_all_vec()
    for home in range(pid, EPN, num_programs):
        start = tl.load(home_offsets_ptr + home)
        end = tl.load(home_offsets_ptr + home + 1)
        for ordinal in range(start, end):
            peer = tl.load(desc_peer_ptr + ordinal)
            slot = tl.load(desc_slot_ptr + ordinal)
            _fused_pull_one_table(
                acc_gate_up_ptr, gate_up_slot_ptr, staging_gate_up_row,
                peer, slot, home, gate_up_elements_per_expert,
                LOCAL_RANK, GATE_UP_CHUNK_ELEMENTS, ACC_BLOCK,
            )
            _fused_pull_one_table(
                acc_down_ptr, down_slot_ptr, staging_down_row,
                peer, slot, home, down_elements_per_expert,
                LOCAL_RANK, DOWN_CHUNK_ELEMENTS, ACC_BLOCK,
            )
    # barrier #2: every reader is done before any consumed slot is cleared.
    libshmem_device.barrier_all_vec()
    for i in range(pid, consumed_count, num_programs):
        slot = tl.load(consumed_ptr + i)
        _fused_zero_rows(
            gate_up_slot_ptr, slot, gate_up_elements_per_expert, ACC_BLOCK
        )
        _fused_zero_rows(
            down_slot_ptr, slot, down_elements_per_expert, ACC_BLOCK
        )
    # barrier #3: the zeroing is published before any next forward re-push.
    libshmem_device.barrier_all_vec()


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


def build_owner_pull_descriptors_by_home(experts_to_copy_cpu, rank, experts_per_rank):
    """Return by-home-expert pull descriptors for the fused transport kernel.

    Produces ``(peer, slot, home, home_offsets)`` CPU int32 tensors where the
    descriptor triples are grouped by local home expert (``home`` ascending)
    and, inside every home expert, still ``(peer, slot)`` lexicographic — the
    very same order the flat :func:`build_owner_pull_descriptors` scan yields,
    so the fused per-expert accumulation reproduces the legacy fp32 result
    bit-for-bit.  ``home_offsets`` has ``experts_per_rank + 1`` entries;
    descriptor ``i`` belongs to home expert ``le`` iff
    ``home_offsets[le] <= i < home_offsets[le + 1]``.
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
    # One row-major (peer, slot) scan, bucketed per local home expert: the
    # bucket append order preserves the lexicographic order inside each expert.
    buckets: list[list[tuple[int, int]]] = [
        [] for _ in range(experts_per_rank)
    ]
    for peer, row in enumerate(experts_to_copy_cpu.tolist()):
        for slot, expert in enumerate(row):
            if owner_start <= expert < owner_end:
                buckets[expert - owner_start].append((peer, slot))
    peers: list[int] = []
    slots: list[int] = []
    homes: list[int] = []
    home_offsets = [0]
    for local_expert, bucket in enumerate(buckets):
        for peer, slot in bucket:
            peers.append(peer)
            slots.append(slot)
            homes.append(local_expert)
        home_offsets.append(len(peers))

    def _as_tensor(values):
        if values:
            return torch.tensor(values, dtype=torch.int32)
        return torch.empty(0, dtype=torch.int32)

    return (
        _as_tensor(peers),
        _as_tensor(slots),
        _as_tensor(homes),
        torch.tensor(home_offsets, dtype=torch.int32),
    )


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


def launch_grad_reduce_transport(
    *,
    acc_gate_up,
    acc_down,
    gate_up_table,
    down_table,
    desc_peer,
    desc_slot,
    desc_home,
    home_offsets,
    consumed_idx,
    rank,
    num_programs,
    gate_up_chunk_bytes,
    down_chunk_bytes,
    acc_block=_ACC_BLOCK,
):
    """Launch the whole cross-rank replica grad transport as ONE kernel.

    Replaces the legacy launch sequence (barrier -> single-program owner-pull
    -> barrier -> host zero -> barrier, six launches plus host work between
    them) with a single AICore-sized launch whose internal
    ``barrier_all_vec`` calls carry the two fences.  The owner-pull is
    partitioned by home expert across the grid's programs; each program pulls
    through its own staging row, sized ``chunk_bytes`` per program per table.

    Args follow the legacy launcher's contract, plus the by-home-expert
    descriptor columns from :func:`build_owner_pull_descriptors_by_home`
    (with ``home_offsets``), and ``consumed_idx`` — the int32 device tensor of
    replica slots this rank sank a gradient into (may be empty).  The
    accumulators must already carry the fp32 home-segment seed in the tables'
    per-expert layout, exactly as the legacy path requires.  ``num_programs``
    must equal the physical AICore count (the barrier grid).
    """
    for name, accumulator in (
        ("acc_gate_up", acc_gate_up), ("acc_down", acc_down)
    ):
        if accumulator.dtype != torch.float32 or not accumulator.is_contiguous():
            raise ValueError(f"{name} must be a contiguous fp32 tensor")
    experts_per_rank, hidden_dim, ffn_packed = acc_gate_up.shape
    if tuple(acc_down.shape) != (experts_per_rank, hidden_dim, ffn_packed // 2):
        raise ValueError(
            "the down accumulator must pack half the gate/up expert width, "
            f"got {tuple(acc_down.shape)} for gate/up "
            f"{tuple(acc_gate_up.shape)}"
        )
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
    if home_offsets.dtype != torch.int32 or home_offsets.ndim != 1:
        raise ValueError("home_offsets must be a 1-D int32 tensor")
    if home_offsets.device != acc_gate_up.device:
        raise ValueError("home_offsets must live on the accumulator device")
    if tuple(home_offsets.shape) != (experts_per_rank + 1,):
        raise ValueError(
            "home_offsets must have shape "
            f"{(experts_per_rank + 1,)}, got {tuple(home_offsets.shape)}"
        )
    if consumed_idx.dtype != torch.int32 or consumed_idx.ndim != 1:
        raise ValueError("consumed_idx must be a 1-D int32 tensor")
    if consumed_idx.device != acc_gate_up.device:
        raise ValueError("consumed_idx must live on the accumulator device")
    if type(num_programs) is not int or num_programs <= 0:
        raise ValueError("num_programs must be a positive integer")

    device = acc_gate_up.device
    gate_up_elements = int(acc_gate_up[0].numel())
    down_elements = int(acc_down[0].numel())
    for name, chunk_bytes in (
        ("gate_up_chunk_bytes", gate_up_chunk_bytes),
        ("down_chunk_bytes", down_chunk_bytes),
    ):
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError(f"{name} must be a positive integer")
    # One staging row per program per table; a program only ever pulls one
    # chunk at a time, so the row is sized by the (validated) chunk geometry.
    gate_up_chunk_elements = min(
        gate_up_chunk_bytes // 2, gate_up_elements
    )
    down_chunk_elements = min(down_chunk_bytes // 2, down_elements)
    if gate_up_chunk_elements <= 0 or down_chunk_elements <= 0:
        raise ValueError("the staging chunk must hold at least one element")
    staging_gate_up = torch.empty(
        (num_programs, gate_up_chunk_elements),
        dtype=torch.bfloat16,
        device=device,
    )
    staging_down = torch.empty(
        (num_programs, down_chunk_elements),
        dtype=torch.bfloat16,
        device=device,
    )
    _kernel_grad_reduce_transport[(num_programs, 1, 1)](
        0,
        num_programs,
        acc_gate_up,
        acc_down,
        gate_up_table,
        down_table,
        desc_peer,
        desc_slot,
        desc_home,
        home_offsets,
        int(desc_peer.numel()),
        consumed_idx,
        int(consumed_idx.numel()),
        staging_gate_up,
        staging_down,
        gate_up_elements,
        down_elements,
        LOCAL_RANK=rank,
        EPN=experts_per_rank,
        GATE_UP_CHUNK_ELEMENTS=gate_up_chunk_elements,
        DOWN_CHUNK_ELEMENTS=down_chunk_elements,
        ACC_BLOCK=acc_block,
    )


__all__ = [
    "build_owner_pull_descriptors",
    "build_owner_pull_descriptors_by_home",
    "launch_grad_reduce_transport",
    "launch_owner_pull_accumulate",
    "launch_replica_grad_barrier",
    "zero_consumed_replica_slots",
]
