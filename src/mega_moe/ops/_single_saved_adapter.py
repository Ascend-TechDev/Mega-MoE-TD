# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Backward-contract enrichment for the single-kernel forward's saved dict.

``FusedMoEForward._forward_single_kernel(..., return_saved=True)`` (the
``enable_single_kernel_forward`` path) returns a MINIMAL contract: the raw
``fc1_output``, the per-received-row routing weights, zero-copy input
references, and the receive-layout tables
(``recv_expert_offsets`` / ``recv_counts_by_source_expert`` /
``send_route_indices``).  The backward — the one-launch mega kernel under
``MOE_BWD_MEGA=1`` + ``MOE_SAVED_RECOMPUTE=1`` — rebuilds its dispatch maps
from ``selected_experts`` itself and recomputes the two big activations
in-launch, but still needs the combine-side plan tables, the weight
references, and the scalar section.  This module derives all of that from
the minimal contract in closed form — no activation is rebuilt or permuted
here.

Layout alignment (plan B, 2026-09-16): the single-kernel planner's
``_build_destination_metadata`` computes ``send_bucket_dst_starts =
expert_base + source_prefix`` with expert-major ``expert_base`` and
ascending-source ``source_prefix`` — the exact algebra of the backward's
``_dispatch_static_maps`` — so the SEGMENT structure (expert-major,
source-minor) is identical on both sides.  The WITHIN-segment order also
differs (``_scatter_stable_routes`` walks contiguous per-core route slices
with per-core bucket cursors, giving core-major-then-flat order, while the
backward's own reconstruction is a stable argsort) — but instead of bending
the forward artifacts to the backward (the original adapter
``index_select``-reordered ``fc1_output`` / ``recv_weights_sorted`` by
``argsort(send_route_indices)``, one ~[M, 2F] gather copy per layer per
step), the backward now bends to the forward: this adapter exports the
forward's send table as ``forward_send_route`` / ``sort_idxs`` and
``_dispatch_static_maps`` adopts it as the canonical send order, so the
re-dispatch itself reproduces the forward placement and both saved
activations pass through un-reordered.  Every backward map (gco gather,
re-dispatch source rows, push-back offsets, reduce scatter) is POSITIONAL
over the send table, so swapping the table keeps them mutually consistent.
Two cheap tripwires below still verify the segment algebra and the send
table's layout at runtime — never silence them.

MoonEP (``enable_moonep``, 2026-09-22): the single-kernel planner already
produces every layout table in PHYSICAL slot order (home | replica,
EPR = epn + budget), so the same closed forms run unchanged over the
physical stride.  The send side is NOT derivable from expert bincounts (a
route's destination is the planner's choice), so it is snapshotted from the
kernel's own metadata count row and cross-checked against the send bucket
starts (tripwire 2b becomes a per-bucket expert-identity check against the
planner's ``experts_to_copy``).  The MoonEP saved sections mirror
``_attach_moonep_plan_sections`` / ``build_physical_saved_from_plan``, with
the replica weight tables / ready slabs borrowed LIVE (pooled content is
re-pushed in-launch by the backward under MOE_MEGA_REPREFETCH, default on),
and ``op._replica_experts_cache`` is staged so the auto-lend in
``MegaMoEFunction.backward`` succeeds.

Droless routing is REQUIRED: the backward's map build asserts
``bincount(selected_experts).sum() == total_send``, so dropped routes
(capacity_factor < world_size) fail here first with a clear message.
"""

import torch

from ._moonep_torch_forward import _arrival_to_slot_permutation
from ._native_saved import _weight_reference_section
from ._torch_forward import BLOCK_SIZE_M

_REQUIRED_CONTRACT_KEYS = (
    "fc1_output", "recv_weights_sorted", "selected_experts", "routing_weights",
    "hidden_states", "recv_expert_offsets", "recv_counts_by_source_expert",
    "send_route_indices", "total_send", "total_recv", "M",
)


def enrich_single_kernel_saved(op, saved, *, hidden_states, gate_up_weight,
                               down_weight):
    """Expand the single-kernel minimal contract into the backward's saved dict.

    Mutates and returns ``saved``.  ``hidden_states`` / ``gate_up_weight`` /
    ``down_weight`` are the ``apply``-time forward inputs (autograd-held, one
    distinct set per layer) — the weight references are built from them, never
    from the operator's internal staging buffers (those are copy_-refreshed
    every call and would go stale between layers).
    """
    use_moonep = bool(op.enable_moonep)
    if int(op.world_size) * int(op.experts_per_rank) > 32:
        # workspace.py pads the bins to next_power_of_2(E); above 32 the
        # scatter's multi-bin-block loop (bin_block=32) corrupts the send
        # tables — observed at E=128 as duplicate slots + uninitialized
        # entries in send_route_indices and an aivec trap in isolation
        # (upstream latent defect: no upstream case exercises E>32; kimi
        # integration shapes are E=32).  Fail fast instead of reordering
        # garbage.
        raise NotImplementedError(
            f"single-kernel saved contract validated only for E<=32 (got "
            f"E={int(op.world_size) * int(op.experts_per_rank)}): the "
            "kernel scatter's multi-bin-block path corrupts the send "
            "tables above 32 bins"
        )
    missing = [k for k in _REQUIRED_CONTRACT_KEYS if k not in saved]
    if missing:
        raise ValueError(
            f"the single-kernel saved contract is missing keys: {missing}"
        )
    device = hidden_states.device
    W = int(op.world_size)
    epn = int(op.experts_per_rank)
    # Slot stride of every layout table below: the MoonEP physical table
    # (home | replica) under enable_moonep, the home table otherwise.
    EPR = int(op.physical_experts_per_rank) if use_moonep else epn
    total_send = int(saved["total_send"])
    total_recv = int(saved["total_recv"])

    flat = saved["selected_experts"].to(torch.int64).reshape(-1)
    if total_send != flat.numel():
        raise ValueError(
            "droless routing required for the single-kernel backward "
            f"contract: {total_send} routes kept of {flat.numel()} — raise "
            "receive_capacity_factor to >= world_size"
        )

    recv_counts64 = saved["recv_counts_by_source_expert"].to(
        torch.int64)                                        # [W, EPR] (source, expert)
    recv_expert_offs = saved["recv_expert_offsets"].to(
        torch.int64)                                        # [EPR + 1]

    # Tripwire 1: the expert-major receive prefix must match the counts the
    # contract carries (both sides derive it by cumsum over per-expert totals).
    expected_offs = torch.zeros_like(recv_expert_offs)
    expected_offs[1:] = recv_counts64.sum(0).cumsum(0)
    if not torch.equal(recv_expert_offs, expected_offs):
        raise RuntimeError(
            "single-kernel receive layout diverged from the backward's "
            "expert-major algebra (recv_expert_offsets != cumsum of "
            "recv_counts_by_source_expert sums)"
        )

    # Tripwire 2: the send table this adapter is about to hand the backward
    # as its canonical send order (plan B).  Two structural requirements:
    # (1) it is a permutation of 0..total_send-1 — the re-dispatch's source
    #     gather (send_src_idx = sort_idxs // topk) and the reduce's
    #     inv_sort scatter are only well-defined over a true permutation;
    # (2) it is bucket-segmented — flat expert ids non-decreasing along the
    #     table.  The backward's sweeps walk (dst, expert) buckets through
    #     send_bucket_starts, so a table laid out in any other order would
    #     push rows into the wrong receive slots.  The forward's
    #     `_scatter_stable_routes` bucket cursors guarantee both.  (The old
    #     adapter instead index_select-reordered fc1_output /
    #     recv_weights_sorted INTO the backward's stable-argsort order —
    #     plan B flips the direction and drops both copies.)
    send_route = saved["send_route_indices"].to(torch.int64)
    sorted_route = torch.sort(send_route).values
    if not torch.equal(
        sorted_route,
        torch.arange(total_send, dtype=torch.int64, device=device),
    ):
        raise RuntimeError(
            "single-kernel send_route_indices is not a permutation of "
            "0..total_send-1 — it cannot serve as the backward's send "
            f"order: len={send_route.numel()} total_send={total_send} "
            f"min={int(sorted_route[0]) if sorted_route.numel() else -1} "
            f"max={int(sorted_route[-1]) if sorted_route.numel() else -1} "
            f"distinct={int(torch.unique(send_route).numel())} "
            f"head={send_route[:8].tolist()}"
        )
    # MoonEP plan snapshot: the device planner's products, settled by the
    # forward's metadata .item() sync before this adapter runs.  Cloned here
    # because a framework host sharing one operator across same-shape layers
    # lets a LATER forward rewrite the planning workspaces before this
    # layer's backward (the _single_kernel_snapshot contract).
    experts_to_copy = experts_to_copy_cpu = None
    active_phys = 0
    if use_moonep:
        ctxn = op.context
        if op._replica_weight_buffers is None:
            raise RuntimeError(
                "single-kernel MoonEP forward returned without replica "
                "weight buffers; cannot build the physical backward contract"
            )
        experts_to_copy = ctxn.planning_experts_to_copy.clone()
        experts_to_copy_cpu = ctxn.planning_experts_to_copy.cpu().contiguous()
        # world-max replica count, mirroring build_routing_plan
        # (runtime/routing.py): active is rank-uniform by construction.
        active_phys = epn + int(ctxn.planning_replica_counts.max().item())
        replica_budget = int(experts_to_copy.shape[1])
        if EPR != epn + replica_budget:
            raise RuntimeError(
                "physical expert stride disagrees with the plan table: "
                f"EPR={EPR} != epn({epn}) + budget({replica_budget})"
            )

    expert_seq = flat[send_route]
    if use_moonep:
        # MoonEP tripwire 2b: buckets are (dst, PHYSICAL slot) and a route's
        # flat expert id is NOT monotone along the table (a replica bucket
        # holds its owner's expert id, which may sort anywhere).  The strong
        # invariant instead: every bucket's rows carry exactly the expert
        # that bucket holds — dst*epn+slot for home slots, the copied expert
        # for replica slots — checked against the planner's own count row.
        counts_row = op.context.metadata_counts_mem.view(
            W, op.context.metadata_num_bins
        )[op.rank, : W * EPR].to(torch.int64).reshape(-1)          # [W*EPR]
        if int(counts_row.sum().item()) != total_send:
            raise RuntimeError(
                "MoonEP send count row disagrees with the contract: "
                f"{int(counts_row.sum().item())} != {total_send}"
            )
        # Cross-check two independent kernel products: the metadata count
        # row and the send bucket starts must be each other's cumsum.
        expected_starts = counts_row.cumsum(0) - counts_row
        if not torch.equal(
            op.context.metadata_send_bucket_starts.to(torch.int64),
            expected_starts,
        ):
            raise RuntimeError(
                "MoonEP send bucket starts are not the exclusive cumsum of "
                "the metadata count row — the backward's bucket walk would "
                "scatter rows to wrong slots"
            )
        bucket_ids = torch.arange(W * EPR, dtype=torch.int64, device=device)
        dst_of = bucket_ids // EPR
        slot_of = bucket_ids % EPR
        etc_flat = experts_to_copy.reshape(-1).to(torch.int64)
        replica_expert = etc_flat[
            dst_of * replica_budget + (slot_of - epn).clamp_min(0)
        ]
        expected_expert = torch.where(
            slot_of < epn, dst_of * epn + slot_of, replica_expert
        )
        expected_seq = torch.repeat_interleave(
            expected_expert, counts_row, output_size=total_send
        )
        if not torch.equal(expert_seq, expected_seq):
            raise RuntimeError(
                "single-kernel send_route_indices disagrees with the MoonEP "
                "plan: a (dst, slot) bucket's rows must carry exactly the "
                "expert that slot holds (home: dst*epn+slot; replica: the "
                "planner's experts_to_copy entry)"
            )
    elif bool((expert_seq[1:] < expert_seq[:-1]).any()):
        raise RuntimeError(
            "single-kernel send_route_indices is not bucket-segmented "
            "(flat expert ids decrease along the table): the backward's "
            "(dst, expert) bucket walk would scatter rows to wrong slots"
        )
    # inv_sort: the send position of each route id (int argsort falls back
    # to AICPU on Ascend — float32, routing.py precedent).  This is exactly
    # the old reorder permutation, now consumed by the reduce scatter.
    perm = torch.argsort(send_route.to(torch.float32))
    fc1_output = saved["fc1_output"]
    if int(fc1_output.shape[0]) != total_recv:
        raise RuntimeError(
            "fc1_output rows do not match the contract's receive count: "
            f"{int(fc1_output.shape[0])} != {total_recv}"
        )
    # The canonical send order for the backward's map build (consumed by
    # _dispatch_static_maps as bwd_expert_sort): with this in place,
    # fc1_output / recv_weights_sorted ride along in the forward's own
    # receive order — no index_select, no fresh ~[M, 2F] block per step.
    saved["forward_send_route"] = send_route.to(torch.int32).contiguous()

    # ---- plan tables (closed forms; mirrors _native_saved
    # _snapshot_home_plan_sections, fed from the contract instead of a
    # MoERoutingPlan) -------------------------------------------------------
    expert_counts64 = recv_counts64.sum(0)
    expert_counts = expert_counts64.to(torch.int32)
    split_size_cum_per_expert = torch.zeros(
        EPR + 1, dtype=torch.int32, device=device)
    split_size_cum_per_expert[1:] = expert_counts64.cumsum(0)

    tiles_per_expert = (expert_counts64 + (BLOCK_SIZE_M - 1)) // BLOCK_SIZE_M
    num_tiles = int(tiles_per_expert.sum().item())
    num_tiles_total = torch.tensor([num_tiles], dtype=torch.int32, device=device)
    if num_tiles:
        token_starts = expert_counts64.cumsum(0) - expert_counts64
        tile_starts = tiles_per_expert.cumsum(0) - tiles_per_expert
        expert_ids = torch.arange(EPR, dtype=torch.int64, device=device)
        meta_expert_ids = torch.repeat_interleave(
            expert_ids, tiles_per_expert, output_size=num_tiles)
        meta_split_cum = torch.repeat_interleave(
            token_starts, tiles_per_expert, output_size=num_tiles)
        meta_tile_num_cum = torch.repeat_interleave(
            tile_starts, tiles_per_expert, output_size=num_tiles)
        meta_tile_num = (
            torch.arange(num_tiles, dtype=torch.int64, device=device)
            - meta_tile_num_cum
        )
        meta_expert_ids = meta_expert_ids.to(torch.int32)
        meta_split_cum = meta_split_cum.to(torch.int32)
        meta_tile_num = meta_tile_num.to(torch.int32)
        meta_tile_num_cum = meta_tile_num_cum.to(torch.int32)
    else:
        meta_expert_ids = torch.empty(0, dtype=torch.int32, device=device)
        meta_split_cum = torch.empty(0, dtype=torch.int32, device=device)
        meta_tile_num = torch.empty(0, dtype=torch.int32, device=device)
        meta_tile_num_cum = torch.empty(0, dtype=torch.int32, device=device)

    # Send-side per-destination totals: the same bincount the backward's
    # _dispatch_static_maps runs — done here once, host-side, so the split
    # lists need no extra collective round.  Under MoonEP the destination is
    # the PLANNER's choice (home hit stays home, overflow lands on a replica
    # holder), so the counts are not derivable from expert bincounts — reuse
    # the kernel's own count row validated by the tripwire above.
    if use_moonep:
        send_counts_re = counts_row.reshape(W, EPR)             # [W, phys]
    else:
        send_counts_re = torch.bincount(
            flat, minlength=W * EPR).to(torch.int64).reshape(W, EPR)  # [W, EPR]
    splits_send_list = send_counts_re.sum(dim=1).tolist()
    splits_recv_list = recv_counts64.sum(dim=1).tolist()
    if sum(splits_send_list) != total_send:
        raise RuntimeError(
            "send counts disagree with the contract's send size: "
            f"{sum(splits_send_list)} != {total_send}"
        )
    if sum(splits_recv_list) != total_recv:
        raise RuntimeError(
            "receive counts disagree with the contract's receive count: "
            f"{sum(splits_recv_list)} != {total_recv}"
        )

    # Permutation invariants, from the arrival algebra (int argsort falls
    # back to AICPU on Ascend — every argsort goes through float32,
    # routing.py precedent).  sort_idxs IS the forward's send table (plan
    # B): the re-dispatch P0 walks routes in this order, rows land in the
    # forward's receive layout, and fc1_output / recv_weights_sorted ride
    # along un-reordered.  inv_sort is its inverse (== perm above).
    sort_idxs = send_route.contiguous()
    inv_sort = perm
    slot_starts = recv_expert_offs[:-1]
    local_sort_idxs = _arrival_to_slot_permutation(recv_counts64, slot_starts)
    inv_local = torch.argsort(local_sort_idxs.to(torch.float32))

    saved.update(
        # plan tables
        expert_counts=expert_counts,
        split_size_cum_per_expert=split_size_cum_per_expert,
        num_tiles_total=num_tiles_total,
        meta_expert_ids=meta_expert_ids,
        meta_split_cum=meta_split_cum,
        meta_tile_num=meta_tile_num,
        meta_tile_num_cum=meta_tile_num_cum,
        sort_idxs=sort_idxs,
        local_sort_idxs=local_sort_idxs,
        inv_local=inv_local,
        inv_sort=inv_sort,
        splits_send_list=splits_send_list,
        splits_recv_list=splits_recv_list,
    )
    saved.update(_weight_reference_section(gate_up_weight, down_weight,
                                           int(gate_up_weight.shape[2] // 2)))
    # Full-snapshot marker: after enrichment every tensor in this dict is a
    # clone / dtype-cast copy / closed-form fresh build / caller-held input
    # reference — nothing aliases the operator's planning or mirror
    # workspaces.  MegaMoEFunction.backward relaxes its routing-generation
    # equality guard for marked dicts, which is what lets a framework host
    # share ONE operator (and its ~GB-scale fixed workspaces) across all
    # same-shape MoE layers: layer N's forward may bump the generation
    # before layer N-1's backward runs, and that is fine here (the 5-op
    # saved contract DOES alias those workspaces, so the guard stays
    # strict for it).  MoonEP exception, by design: the replica weight
    # tables / ready slabs / pool object below are LIVE shared workspaces —
    # under table pooling their content at backward time is another layer's
    # weights, which is exactly what MOE_MEGA_REPREFETCH=1 (default on)
    # repairs by re-pushing this layer's tables in-launch before P1 reads
    # them.  Everything the guard actually protects (the plan tables) is
    # still cloned above.
    saved["_single_kernel_snapshot"] = True
    _situ_beta = getattr(op, "situ_beta", None)
    _situ_linear_beta = getattr(op, "situ_linear_beta", None)
    saved.update(
        batch_size=int(hidden_states.shape[0]),
        hidden_dim=int(hidden_states.shape[1]),
        ffn_dim=int(gate_up_weight.shape[2] // 2),
        topk=int(op.top_k),
        world_size=W,
        ep_rank=int(op.rank),
        # ep_rank is the ACLSHMEM global PE (kernel LOCAL_RANK); local_device
        # is the NPU ordinal for backward-side allocations (multi-node split).
        local_device=int(getattr(op, "local_device", op.rank)),
        ep_group=op.ep_group,
        # HOME experts per rank under MoonEP (mirrors saved_phys in
        # _moonep_torch_forward); the physical stride rides separately.
        experts_per_rank=epn,
        activation=str(getattr(op, "activation", None) or "swiglu"),
        situ_beta=1.0 if _situ_beta is None else float(_situ_beta),
        situ_linear_beta=(
            None if _situ_linear_beta is None else float(_situ_linear_beta)
        ),
    )
    if use_moonep:
        saved.update(
            # MoonEP physical-slot sections, mirroring
            # _attach_moonep_plan_sections / build_physical_saved_from_plan.
            use_moonep=True,
            num_experts=epn * W,
            home_experts_per_rank=epn,
            physical_experts_per_rank=EPR,
            active_physical_experts_per_rank=active_phys,
            experts_to_copy=experts_to_copy,
            experts_to_copy_cpu=experts_to_copy_cpu,
            # Live symmetric tables (see the snapshot-marker note above):
            # the backward sinks replica weight GRADIENTS into these slots
            # and MOE_MEGA_REPREFETCH re-pushes this layer's weights into
            # them in-launch.
            replica_gate_up=op._replica_weight_buffers.gate_up,
            replica_down=op._replica_weight_buffers.down,
            replica_buffers=op._replica_weight_buffers,
            replica_gate_ready=op.context.replica_gate_ready,
            replica_down_ready=op.context.replica_down_ready,
            # Physical plan tables consumed by _dispatch_static_maps_moonep.
            # plan_recv_* / plan_received_* ARE the contract tables (already
            # clones in physical slot order); the send side re-serves the
            # tripwire-validated count row and the kernel's bucket starts.
            plan_send_counts_by_rank_expert=(
                send_counts_re.to(torch.int32).contiguous()
            ),
            plan_send_bucket_starts=(
                op.context.metadata_send_bucket_starts.clone()
            ),
            plan_send_bucket_dst_starts=(
                op.context.metadata_send_bucket_dst_starts.clone()
            ),
            plan_recv_counts_by_source_expert=(
                saved["recv_counts_by_source_expert"]
            ),
            plan_received_expert_offsets=saved["recv_expert_offsets"],
        )
        # MegaMoEFunction.backward auto-lends the replica tables for the
        # grad transport whenever saved["use_moonep"]; lend requires the
        # ETC snapshot this path never staged (the single-kernel forward
        # always invalidates the weight cache instead of caching it).
        op._replica_experts_cache = experts_to_copy_cpu.clone()
    return saved


__all__ = ["enrich_single_kernel_saved"]
