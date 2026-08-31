# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Private torch + HCCL physical-slot MoonEP forward used by fused backward.

The production forward scatters tokens over MoonEP *physical* slots (home
experts plus replica slots), but it never materializes the intermediates the
5-op triton backward needs.  This module replays the exact same routing plan
with torch + HCCL and produces ``saved_phys``: the ``saved`` contract of
:func:`mega_moe.ops._torch_forward.moe_forward` re-interpreted over the
physical ``[home | replica]`` slot table.

Physical slot semantics
-----------------------
With ``epn`` home experts per rank and a replica budget ``B = epn``:

* slot ``s < epn``            -> home expert ``ep_rank * epn + s``
* slot ``epn + b`` (``b<B``)  -> logical expert ``experts_to_copy[ep_rank][b]``
  (``-1`` marks an empty slot; its weights stay zero and it receives no rows)

``saved_phys`` key contract (consumed by the M2/M3 physical backward)
---------------------------------------------------------------------
Legacy keys (same names/shapes/dtypes as the home-layout ``saved``):

* dims/group: ``batch_size hidden_dim ffn_dim num_experts experts_per_rank
  topk world_size ep_rank ep_group M total_send total_recv`` plus ``output``
  and ``dy=None``
* dispatch invariants:
  - ``sort_idxs``      int64 [total_send]  send position -> flat route id,
    i.e. the plan's ``(destination, physical slot)`` bucket-stable send order
  - ``inv_sort``       int64 [total_send]  inverse of ``sort_idxs``
  - ``local_sort_idxs`` int64 [total_recv] slot-sorted position -> arrival
    position (arrival buffer is source-major); a permutation of arange
  - ``inv_local``      int64 [total_recv]  inverse of ``local_sort_idxs``
  - ``splits_send_list`` / ``splits_recv_list``  python int lists, per
    destination-rank / per source-rank route counts
* physical groups: ``expert_counts`` int32 ``[epn + B]`` physical slot counts
  (== ``plan.received_routes_per_expert``), ``split_size_cum_per_expert`` and
  the ``meta_*`` / ``num_tiles_total`` tile metadata over that range
* activations in physical ``(slot, source)`` row order: ``recv_hidden_sorted
  fc1_output gate up swiglu_out_weighted recv_weights_sorted fc2_out``
* weights: ``fc1_1`` ``fc1_2`` ``fc1_combined`` ``fc2`` hold the HOME segment
  only, in the legacy ``[E, N, K]`` layouts (replica weights live in the
  dedicated packed keys below)

MoonEP additions (all plan-derived tensors are ``clone()`` snapshots, so the
single-in-flight planning workspace may be reused afterwards):

* ``use_moonep``                  ``True``
* ``home_experts_per_rank``       ``epn``
* ``physical_experts_per_rank``   ``epn + B``
* ``active_physical_experts_per_rank``
* ``experts_to_copy``             int32 ``[world_size, B]`` device snapshot
* ``experts_to_copy_cpu``         int32 ``[world_size, B]`` cpu snapshot
* ``replica_gate_up``             bf16 ``[B, H, 2F]`` packed ``[K, N]`` table
* ``replica_down``                bf16 ``[B, H, F]``
* ``plan_send_counts_by_rank_expert``     int32 ``[world_size, epn + B]``
* ``plan_send_bucket_starts``             int32 ``[world_size * (epn + B)]``
* ``plan_send_bucket_dst_starts``         int32 ``[world_size * (epn + B)]``
* ``plan_recv_counts_by_source_expert``   int32 ``[world_size, epn + B]``
* ``plan_received_expert_offsets``        int32 ``[epn + B + 1]``
"""

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from ._torch_forward import (
    _a2a,
    _gated_activation,
    grouped_matmul,
    prepare_moe_metadata,
)


# ----------------------------------------------------------------------------
# Replica weight transport (functional-test owner-push over HCCL p2p)
# ----------------------------------------------------------------------------

def gather_replica_weights_via_hccl(
    experts_to_copy_cpu,
    home_gate_up,
    home_down,
    ep_group,
):
    """Fill this rank's replica tables from the owning ranks over HCCL p2p.

    ``experts_to_copy_cpu`` is the planner's complete ``[world_size, B]`` CPU
    table, identical on every rank.  For each non-empty replica slot the owner
    (``expert // experts_per_rank``) pushes its home row to the destination
    rank with ``dist.isend``/``dist.irecv``.  Empty slots (``-1``) keep their
    zero initialization.  Returns ``(replica_gate_up [B, H, 2F],
    replica_down [B, H, F])`` in the production packed ``[K, N]`` layout.
    """
    if experts_to_copy_cpu.ndim != 2:
        raise ValueError("experts_to_copy_cpu must have shape [world_size, B]")
    world_size, replica_slots = experts_to_copy_cpu.shape
    for name, weight in (("home_gate_up", home_gate_up), ("home_down", home_down)):
        if weight.ndim != 3:
            raise ValueError(f"{name} must have [experts, K, N] layout")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    experts_per_rank = home_gate_up.shape[0]
    if replica_slots != experts_per_rank:
        raise ValueError(
            "the replica budget must equal experts_per_rank, got "
            f"{replica_slots} slots for {experts_per_rank} home experts"
        )
    if home_down.shape[0] != experts_per_rank:
        raise ValueError("home_down and home_gate_up expert counts differ")
    if home_gate_up.shape[1] != home_down.shape[1]:
        raise ValueError("home_gate_up and home_down hidden sizes differ")
    if home_gate_up.shape[2] != 2 * home_down.shape[2]:
        raise ValueError("home_gate_up output size must be twice down's FFN size")

    device = home_gate_up.device
    replica_gate_up = torch.zeros_like(home_gate_up)
    replica_down = torch.zeros_like(home_down)
    rank = dist.get_rank(ep_group)

    # Both sides walk the same global (destination, slot) order, so every send
    # finds its posted match.  Receives are posted before the owning send of
    # the same pair whenever this rank is the destination.
    experts_to_copy = experts_to_copy_cpu.to(torch.int64).tolist()
    requests = []
    for destination in range(world_size):
        for slot in range(replica_slots):
            expert = experts_to_copy[destination][slot]
            if expert < 0:
                continue
            owner = expert // experts_per_rank
            local_row = expert % experts_per_rank
            if destination == rank:
                if owner == rank:
                    replica_gate_up[slot].copy_(home_gate_up[local_row])
                    replica_down[slot].copy_(home_down[local_row])
                else:
                    requests.append(
                        dist.irecv(
                            replica_gate_up[slot], src=owner, group=ep_group
                        )
                    )
                    requests.append(
                        dist.irecv(
                            replica_down[slot], src=owner, group=ep_group
                        )
                    )
            elif owner == rank:
                requests.append(
                    dist.isend(
                        home_gate_up[local_row], dst=destination, group=ep_group
                    )
                )
                requests.append(
                    dist.isend(
                        home_down[local_row], dst=destination, group=ep_group
                    )
                )
    for request in requests:
        request.wait()
    return replica_gate_up, replica_down


# ----------------------------------------------------------------------------
# Physical-slot forward
# ----------------------------------------------------------------------------

def _dual_table_grouped_matmul(
    activations,
    home_weight,
    replica_weight,
    expert_counts,
    home_experts,
    *,
    transpose,
):
    """Grouped matmul over physical slots with a home/replica weight table.

    Rows are grouped by physical slot, so the home table (``[epn, ...]``)
    serves slots ``[0, epn)`` and the replica table serves ``[epn, epn + B)``.
    """
    home_counts = expert_counts[:home_experts]
    replica_counts = expert_counts[home_experts:]
    home_rows = int(home_counts.sum().item())
    home_part = grouped_matmul(
        activations[:home_rows], home_weight, home_counts, transpose=transpose
    )
    replica_part = grouped_matmul(
        activations[home_rows:], replica_weight, replica_counts, transpose=transpose
    )
    return torch.cat((home_part, replica_part), dim=0)


def _arrival_to_slot_permutation(receive_counts_by_source_expert, slot_starts):
    """Map the source-major arrival buffer onto ``(slot, source)`` row order.

    The all-to-all arrival buffer groups rows by source rank and, inside one
    source, by physical slot.  The grouped GEMMs need rows grouped by physical
    slot with source ranks in increasing order inside each slot.  Both layouts
    move whole ``(source, slot)`` blocks, so the permutation is a block
    reshuffle computed with plain torch ops.
    """
    counts = receive_counts_by_source_expert
    source_totals = counts.sum(dim=1)
    source_starts = source_totals.cumsum(dim=0) - source_totals
    within_source = counts.cumsum(dim=1) - counts
    within_slot = counts.cumsum(dim=0) - counts
    arrival_start = (source_starts.unsqueeze(1) + within_source).reshape(-1)
    sorted_start = (slot_starts.unsqueeze(0) + within_slot).reshape(-1)
    group_sizes = counts.reshape(-1)

    total = int(group_sizes.sum().item())
    permutation = torch.empty(total, dtype=torch.int64, device=counts.device)
    if total:
        group_ids = torch.repeat_interleave(
            torch.arange(group_sizes.numel(), device=counts.device), group_sizes
        )
        group_base = group_sizes.cumsum(dim=0) - group_sizes
        lane_offsets = (
            torch.arange(total, device=counts.device) - group_base[group_ids]
        )
        permutation[sorted_start[group_ids] + lane_offsets] = (
            arrival_start[group_ids] + lane_offsets
        )
    return permutation


def build_physical_saved_from_plan(
    plan,
    hidden_states,
    routing_weights,
    home_gate_up_weight,
    home_down_weight,
    replica_gate_up,
    replica_down,
    *,
    ep_group,
    activation="swiglu",
    situ_beta=1.0,
    situ_linear_beta=None,
):
    """Replay one MoonEP routing plan in torch and materialize ``saved_phys``.

    Dispatch follows the plan's ``(destination, physical slot)`` bucket-stable
    send order, the received rows are regrouped to ``(slot, source)`` order,
    and the grouped FC1/SwiGLU/FC2 run against the dual home/replica weight
    tables.  The combine reverses both stages and reduces the plain top-k sum.
    Returns ``(output [tokens, hidden], saved_phys)``.
    """
    if plan.experts_to_copy is None or plan.experts_to_copy_cpu is None:
        raise ValueError("build_physical_saved_from_plan requires a MoonEP plan")
    dtype = hidden_states.dtype
    device = hidden_states.device
    world_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    batch_size, hidden_dim = hidden_states.shape
    topk = routing_weights.shape[1]
    experts_per_rank = home_gate_up_weight.shape[0]
    if plan.physical_experts_per_rank != 2 * experts_per_rank:
        raise ValueError(
            "the plan's physical slot stride must be 2 * experts_per_rank, got "
            f"{plan.physical_experts_per_rank}"
        )
    if home_gate_up_weight.shape[1] != hidden_dim:
        raise ValueError("home_gate_up_weight hidden size must match hidden_states")
    ffn_dim = home_gate_up_weight.shape[2] // 2
    for name, weight in (
        ("home_gate_up_weight", home_gate_up_weight),
        ("home_down_weight", home_down_weight),
        ("replica_gate_up", replica_gate_up),
        ("replica_down", replica_down),
    ):
        if weight.dtype != dtype or weight.device != device:
            raise ValueError(f"{name} must use {dtype} on {device}")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if home_gate_up_weight.shape != replica_gate_up.shape:
        raise ValueError("replica_gate_up must match home_gate_up_weight shape")
    if home_down_weight.shape != replica_down.shape:
        raise ValueError("replica_down must match home_down_weight shape")
    if tuple(home_down_weight.shape) != (
        experts_per_rank, hidden_dim, ffn_dim
    ):
        raise ValueError(
            "home_down_weight must have shape "
            f"{(experts_per_rank, hidden_dim, ffn_dim)}"
        )
    if routing_weights.shape != (
        plan.num_input_tokens,
        topk,
    ):
        raise ValueError("routing_weights must match the plan's token count")

    # ---- Snapshot every plan tensor: the metadata workspace is reused in
    # place by the next build_routing_plan call.
    send_counts_snapshot = plan.send_counts_by_rank_expert.to(torch.int64)
    recv_counts_snapshot = plan.receive_counts_by_source_expert.to(torch.int64)
    send_bucket_starts = plan.send_bucket_starts.clone()
    send_bucket_dst_starts = plan.send_bucket_receive_offsets.clone()
    received_expert_offsets = plan.received_expert_offsets.clone()
    expert_counts_phys = plan.received_routes_per_expert.clone()
    experts_to_copy = plan.experts_to_copy.clone()
    experts_to_copy_cpu = plan.experts_to_copy_cpu.clone()
    send_token_indices = plan.send_token_indices.to(torch.int64).clone()
    send_route_indices = plan.send_route_indices.to(torch.int64).clone()

    splits_send_list = send_counts_snapshot.sum(dim=1).tolist()
    splits_recv_list = recv_counts_snapshot.sum(dim=1).tolist()
    total_send = int(sum(splits_send_list))
    total_recv = int(sum(splits_recv_list))
    if total_send != send_route_indices.numel():
        raise ValueError("send counts disagree with the plan's send order size")
    if total_recv != int(expert_counts_phys.sum().item()):
        raise ValueError("receive counts disagree with the physical slot counts")

    # ---- Dispatch: gather in the plan's send order and exchange payloads.
    hidden_send = hidden_states[send_token_indices]
    weights_send = routing_weights.reshape(-1)[send_route_indices].to(dtype)

    tokens_recv = _a2a(hidden_send, splits_recv_list, splits_send_list, ep_group)
    weights_recv = _a2a(weights_send, splits_recv_list, splits_send_list, ep_group)

    # ---- Regroup the source-major arrival buffer to (slot, source) order.
    slot_starts = received_expert_offsets.to(torch.int64)[:-1]
    column_totals = recv_counts_snapshot.sum(dim=0)
    if not torch.equal(
        slot_starts, column_totals.cumsum(dim=0) - column_totals
    ):
        raise ValueError(
            "plan received_expert_offsets disagree with the receive counts"
        )
    local_sort_idxs = _arrival_to_slot_permutation(recv_counts_snapshot, slot_starts)
    inv_local = torch.argsort(local_sort_idxs)
    recv_hidden_sorted = tokens_recv[local_sort_idxs]
    recv_weights_sorted = weights_recv[local_sort_idxs]

    meta = prepare_moe_metadata(expert_counts_phys)
    (split_size_cum_per_expert, meta_expert_ids, meta_split_cum,
     meta_tile_num, meta_tile_num_cum, num_tiles_total) = meta

    # ---- Dual-table grouped FC1 -> weighted SwiGLU -> dual-table grouped FC2.
    fc1_out = _dual_table_grouped_matmul(
        recv_hidden_sorted,
        home_gate_up_weight,
        replica_gate_up,
        expert_counts_phys,
        experts_per_rank,
        transpose=False,
    )
    gate, up = fc1_out.chunk(2, dim=-1)
    swiglu_out = _gated_activation(gate, up, activation, situ_beta, situ_linear_beta)
    swiglu_out_weighted = (
        swiglu_out * recv_weights_sorted.float().unsqueeze(-1)
    ).to(dtype)
    fc2_out = _dual_table_grouped_matmul(
        swiglu_out_weighted,
        home_down_weight,
        replica_down,
        expert_counts_phys,
        experts_per_rank,
        transpose=True,
    )

    # ---- Combine: undo the slot regroup, reverse-A2A, undo the send sort.
    fc2_out_unsorted = fc2_out[inv_local]
    combined_out_flat = _a2a(fc2_out_unsorted, splits_send_list, splits_recv_list, ep_group)
    sort_idxs = send_route_indices
    inv_sort = torch.argsort(sort_idxs)
    combined_full = combined_out_flat[inv_sort]
    output = combined_full.view(batch_size, topk, hidden_dim).sum(dim=1)

    # Legacy-layout home weight views for the physical backward contract.
    fc1_1 = home_gate_up_weight[:, :, :ffn_dim].transpose(1, 2).contiguous()
    fc1_2 = home_gate_up_weight[:, :, ffn_dim:].transpose(1, 2).contiguous()
    fc1_combined = torch.cat((fc1_1, fc1_2), dim=1)

    saved_phys = dict(
        output=output, dy=None,
        # dims / group
        batch_size=batch_size, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
        num_experts=experts_per_rank * world_size,
        experts_per_rank=experts_per_rank, topk=topk,
        world_size=world_size, ep_rank=ep_rank, ep_group=ep_group,
        # dispatch / sort invariants (physical semantics)
        sort_idxs=sort_idxs, local_sort_idxs=local_sort_idxs,
        inv_local=inv_local, inv_sort=inv_sort,
        splits_send_list=splits_send_list, splits_recv_list=splits_recv_list,
        total_send=total_send, total_recv=total_recv, M=total_recv,
        expert_counts=expert_counts_phys,
        split_size_cum_per_expert=split_size_cum_per_expert,
        meta_expert_ids=meta_expert_ids, meta_split_cum=meta_split_cum,
        meta_tile_num=meta_tile_num, meta_tile_num_cum=meta_tile_num_cum,
        num_tiles_total=num_tiles_total,
        # fwd activations in physical (slot, source) row order
        recv_hidden_sorted=recv_hidden_sorted, fc1_output=fc1_out,
        gate=gate, up=up, swiglu_out_weighted=swiglu_out_weighted,
        recv_weights_sorted=recv_weights_sorted, fc2_out=fc2_out,
        # home-segment weights in the legacy saved layouts
        fc1_1=fc1_1, fc1_2=fc1_2, fc2=home_down_weight, fc1_combined=fc1_combined,
        selected_experts=plan.selected_experts,
        # MoonEP physical-slot additions (cloned plan snapshots)
        use_moonep=True,
        home_experts_per_rank=experts_per_rank,
        physical_experts_per_rank=plan.physical_experts_per_rank,
        active_physical_experts_per_rank=plan.active_physical_experts_per_rank,
        experts_to_copy=experts_to_copy,
        experts_to_copy_cpu=experts_to_copy_cpu,
        replica_gate_up=replica_gate_up,
        replica_down=replica_down,
        plan_send_counts_by_rank_expert=send_counts_snapshot.to(torch.int32),
        plan_send_bucket_starts=send_bucket_starts,
        plan_send_bucket_dst_starts=send_bucket_dst_starts,
        plan_recv_counts_by_source_expert=recv_counts_snapshot.to(torch.int32),
        plan_received_expert_offsets=received_expert_offsets,
    )
    return output, saved_phys


__all__ = [
    "_arrival_to_slot_permutation",
    "build_physical_saved_from_plan",
    "gather_replica_weights_via_hccl",
]
