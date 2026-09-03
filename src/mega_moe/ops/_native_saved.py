# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Native ``saved`` capture for the fused Mega-MoE forward (Stage H1).

``FusedMoEForward.forward(..., return_saved=True)`` produces the backward's
``saved`` dict directly out of the fused forward instead of replaying the
routing plan with torch + HCCL.  This module owns that capture.  It is split
into the three short calls the forward makes:

* :func:`snapshot_plan_metadata` (T1) — runs *immediately* after
  ``build_routing_plan`` returns, while the current stream is already drained
  by the planner's ``.item()``.  Clones every plan tensor (the planning
  workspace is single-in-flight and rewritten in place by the next build),
  derives the permutation invariants, and performs the two ``[W]``-element
  D2H transfers for the split lists.  Everything host-syncing happens here on
  purpose: deferring it behind dispatch/FC1/FC2 would stall the whole pipeline.
* :func:`capture_activations` (T2) — activation snapshot inside the dispatch
  window.  **Stage H2 stub** (signature + contract below, not called yet).
* :func:`assemble_native_saved` (T3) — merges the T1 snapshot, the T2
  activations, and the scalar/group section into the final ``saved`` dict.

Home-layout key contract (Stage H1)
-----------------------------------
The alignment target is the torch-replay ``saved`` built by
``mega_moe.ops._torch_forward.moe_forward`` (39 keys, ``_torch_forward.py``
``saved = dict(...)``).  H1 covers its metadata / permutation / scalar keys:

===============================  =========================  ============================
key (H1)                         value                      oracle counterpart
===============================  =========================  ============================
``expert_counts``                int32 ``[E_p]``            ``received_routes_per_expert``
                                 clone                      (the plan has **no**
                                                            ``expert_counts`` field)
``split_size_cum_per_expert``    int32 ``[E_p+1]``          recomputed:
                                                            ``zeros(E+1); [1:] =
                                                            counts64.cumsum(0)``
``num_tiles_total``              int32 ``[1]`` (device)     ``prepare_moe_metadata``
``meta_expert_ids`` /            int32 ``[num_tiles]``      idem, recomputed on device
``meta_split_cum`` /                                        without the host loop
``meta_tile_num`` /
``meta_tile_num_cum``
``sort_idxs``                    int64 ``[total_send]``     ``plan.send_route_indices``
``local_sort_idxs``              int64 ``[M]``              ``_arrival_to_slot_permutation``
``inv_local`` / ``inv_sort``     int64                      float32 ``argsort`` (see
                                                            ``runtime/routing.py``,
                                                            int argsort falls back to
                                                            AICPU on Ascend)
``splits_send_list`` /          python ``list[int]``       2 x ``[W]`` D2H built here
``splits_recv_list``
``M`` / ``total_send`` /         python ``int``             plan counters (no D2H)
``total_recv``
``batch_size`` .. ``ep_group``   python ``int`` / group     operator attributes
``experts_per_rank``
``_routing_generation`` /        python ``int``             defensive gate (H3 asserts)
``_owner_token``
``plan_send_counts_by_rank_expert`` ..                      extra plan snapshots
``plan_received_expert_offsets``                            (not replay keys); named
                                                            after the MoonEP
                                                            ``saved_phys`` convention
===============================  =========================  ============================

Deliberate divergences from the replay oracle (both are contract-safe):

1. **Device residency**: the replay builds ``split_size_cum_per_expert``,
   ``meta_*``, and ``num_tiles_total`` on the *host*; the native capture keeps
   them on the NPU (``num_tiles_total`` stays a device scalar per plan §3.1C).
   Every consumer already routes them through ``.to(device)`` /
   ``.cpu().tolist()``.  Values and dtypes are identical.
2. **Send-order permutation**: the home replay orders its send buffer by
   *destination rank only* (stable), while the fused plan orders it by
   *(destination rank, local expert)* (stable) — ``sort_idxs`` /
   ``inv_sort`` / ``local_sort_idxs`` / ``inv_local`` therefore differ as
   *index arrays* even though the resulting ``(expert, source)`` received row
   order is bit-identical (verified against the real
   ``_arrival_to_slot_permutation``).  The backward's own reconstruction uses
   the plan-order formula (``kernels/dispatch_fc2_bwd.py``, ``bwd_expert_sort``
   = stable float32 argsort of the flat expert ids), so the native values are
   the self-consistent choice.  Oracle tests must compare the derived row
   order (or re-derive the replay from the plan's send order the way
   ``_moonep_torch_forward`` does), not the raw permutation arrays.

Stage H2 adds the remaining oracle keys: the activation section
(``recv_hidden_sorted`` — the one materialized clone, taken in the dispatch
window before FC2 staging overwrites the peer-memory view;
``recv_weights_sorted`` — bf16 cast; ``fc1_output`` — reference) plus
``swiglu_out_weighted``, which is produced inside
``_fc2_combine_shadow_activation`` and returned through its opt-in
``return_weighted_activation`` two-tuple, and the zero-copy weight references
``fc1_1`` / ``fc1_2`` / ``fc2`` / ``fc1_combined`` (§3.1B, never materialized).
The input reference ``selected_experts`` is captured from H1 on (held by
reference like the replay oracle does; the home backward's step 1 rebuilds the
expert-major mapping from it).  ``output`` / ``dy`` / ``num_experts`` / ``gate``
/ ``up`` / ``fc2_out`` are never saved (plan §3.1F).

The MoonEP physical layout (Stage N) reuses this module: its plan metadata
covers the physical ``[home | replica]`` slot range, so
:func:`_attach_moonep_plan_sections` only has to append the replica sections.
"""

import torch

from ..runtime.routing import MoERoutingPlan
from ._moonep_torch_forward import _arrival_to_slot_permutation
from ._torch_forward import BLOCK_SIZE_M


def _resolve_send_route_indices(op, plan: MoERoutingPlan) -> torch.Tensor:
    """Return the plan's send-order route ids, rebuilding them if absent.

    Mirrors the fallback in ``FusedMoEForward.dispatch_fc1`` bit-for-bit so
    T1 and the dispatch kernel always agree on the send order.
    """
    if plan.send_route_indices is not None:
        return plan.send_route_indices
    context = op.context
    route_indices = context.row_route_indices[: plan.selected_experts.numel()]
    if plan.num_sent_routes == plan.selected_experts.numel():
        kept_route_indices = route_indices
    else:
        kept_route_indices = route_indices[plan.valid_route_mask]
    return (
        kept_route_indices[plan.stable_sort_indices]
        .to(torch.int32)
        .contiguous()
    )


def _snapshot_home_plan_sections(op, plan: MoERoutingPlan) -> dict:
    """Clone the home-layout plan metadata and derive its invariants."""
    device = plan.received_routes_per_expert.device

    # ---- C: plan metadata clones (the planning workspace is single-in-flight
    # and rewritten in place by the next build_routing_plan call).
    expert_counts = plan.received_routes_per_expert.clone()
    send_counts_by_rank_expert = plan.send_counts_by_rank_expert.clone()
    send_bucket_starts = plan.send_bucket_starts.clone()
    send_bucket_receive_offsets = plan.send_bucket_receive_offsets.clone()
    receive_counts_by_source_expert = (
        plan.receive_counts_by_source_expert.clone()
    )
    received_expert_offsets = plan.received_expert_offsets.clone()
    send_route_indices = _resolve_send_route_indices(op, plan)

    # split_size_cum_per_expert: same values as prepare_moe_metadata, without
    # its host loop (cumsum in int64, stored int32 to match the saved dtype).
    expert_counts64 = expert_counts.to(torch.int64)
    num_local_experts = expert_counts64.shape[0]
    split_size_cum_per_expert = torch.zeros(
        num_local_experts + 1, dtype=torch.int32, device=device
    )
    split_size_cum_per_expert[1:] = expert_counts64.cumsum(0)

    # Tile metadata, recomputed on device (the replay builder walks a python
    # loop over every tile; these closed forms produce identical values).
    tiles_per_expert = (
        expert_counts64 + (BLOCK_SIZE_M - 1)
    ) // BLOCK_SIZE_M
    num_tiles = int(tiles_per_expert.sum().item())
    num_tiles_total = torch.tensor(
        [num_tiles], dtype=torch.int32, device=device
    )
    if num_tiles:
        token_starts = expert_counts64.cumsum(0) - expert_counts64
        tile_starts = tiles_per_expert.cumsum(0) - tiles_per_expert
        expert_ids = torch.arange(
            num_local_experts, dtype=torch.int64, device=device
        )
        meta_expert_ids = torch.repeat_interleave(
            expert_ids, tiles_per_expert, output_size=num_tiles
        )
        meta_split_cum = torch.repeat_interleave(
            token_starts, tiles_per_expert, output_size=num_tiles
        )
        meta_tile_num_cum = torch.repeat_interleave(
            tile_starts, tiles_per_expert, output_size=num_tiles
        )
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

    # ---- E: the two [world_size] D2H transfers.  They run here because the
    # planner's own .item() has already drained the stream; doing them later
    # would stall on dispatch/FC1/FC2 instead.
    send_counts64 = send_counts_by_rank_expert.to(torch.int64)
    recv_counts64 = receive_counts_by_source_expert.to(torch.int64)
    splits_send_list = send_counts64.sum(dim=1).tolist()
    splits_recv_list = recv_counts64.sum(dim=1).tolist()

    total_send = int(plan.num_sent_routes)
    total_recv = int(plan.num_received_routes)
    if sum(splits_send_list) != total_send:
        raise RuntimeError(
            "send counts disagree with the plan's send order size: "
            f"{sum(splits_send_list)} != {total_send}"
        )
    if sum(splits_recv_list) != total_recv:
        raise RuntimeError(
            "receive counts disagree with the plan's receive count: "
            f"{sum(splits_recv_list)} != {total_recv}"
        )

    # ---- D: permutation invariants.  Integer argsort falls back to AICPU on
    # Ascend, so every argsort goes through float32 (routing.py precedent);
    # the route counts stay far below float32's exact integer range.
    sort_idxs = send_route_indices.to(torch.int64).clone()
    inv_sort = torch.argsort(sort_idxs.to(torch.float32))
    slot_starts = received_expert_offsets.to(torch.int64)[:-1]
    local_sort_idxs = _arrival_to_slot_permutation(recv_counts64, slot_starts)
    inv_local = torch.argsort(local_sort_idxs.to(torch.float32))

    return {
        # C: metadata
        "expert_counts": expert_counts,
        "split_size_cum_per_expert": split_size_cum_per_expert,
        "num_tiles_total": num_tiles_total,
        "meta_expert_ids": meta_expert_ids,
        "meta_split_cum": meta_split_cum,
        "meta_tile_num": meta_tile_num,
        "meta_tile_num_cum": meta_tile_num_cum,
        # C: raw plan snapshots (not replay keys; saved_phys naming)
        "plan_send_counts_by_rank_expert": send_counts_by_rank_expert,
        "plan_send_bucket_starts": send_bucket_starts,
        "plan_send_bucket_dst_starts": send_bucket_receive_offsets,
        "plan_recv_counts_by_source_expert": receive_counts_by_source_expert,
        "plan_received_expert_offsets": received_expert_offsets,
        # D: permutation invariants
        "sort_idxs": sort_idxs,
        "local_sort_idxs": local_sort_idxs,
        "inv_local": inv_local,
        "inv_sort": inv_sort,
        # E: host values
        "splits_send_list": splits_send_list,
        "splits_recv_list": splits_recv_list,
        "total_send": total_send,
        "total_recv": total_recv,
        "M": total_recv,
    }


def _attach_moonep_plan_sections(op, plan: MoERoutingPlan, snapshot: dict):
    """Stage N extension point: append the MoonEP plan snapshot sections.

    The MoonEP layout reuses the home snapshot above verbatim — its plan
    metadata already spans the physical ``[home | replica]`` slot range — and
    adds the replica-specific sections (``experts_to_copy`` snapshots, the
    physical/home expert strides, and the replica weight table references).
    Implemented by Stage N1.
    """
    if not op.enable_moonep:
        return
    raise NotImplementedError(
        "MoonEP plan snapshot sections arrive with Stage N1"
    )


def snapshot_plan_metadata(op, plan: MoERoutingPlan) -> dict:
    """T1 — snapshot one routing plan for the native ``saved`` dict.

    Must run immediately after ``build_routing_plan`` returned (the planner's
    host reads have already synchronized the stream, so the two ``[W]``
    device-to-host transfers and the tile-count read cost no pipeline stall).
    Every returned tensor is a clone or a freshly computed value, so the
    single-in-flight planning workspace may be reused afterwards.

    Returns the metadata / permutation / host-counter sections of the saved
    dict, keyed by their final ``saved`` names.
    """
    if (
        plan.owner_token is not op._routing_owner_token
        or plan.generation != op._routing_generation
    ):
        raise RuntimeError(
            "routing plan belongs to another or superseded operator"
        )
    snapshot = _snapshot_home_plan_sections(op, plan)
    _attach_moonep_plan_sections(op, plan, snapshot)
    return snapshot


def capture_activations(op, dispatch_result, workspace=None):
    """T2 — snapshot the forward activations the backward needs.

    Call site: inside ``FusedMoEForward.forward`` between the ``dispatch_fc1``
    result and the ``_fc2_combine_shadow_activation`` launch.  The dispatch
    kernel has completed on the current stream (its local dependencies are
    covered by the expert readiness signals even with ``final_barrier=False``),
    and FC2 has not started staging — this window is mandatory because FC2's
    device-put workers and backward step 1 both overwrite the peer-memory
    receive view.  Captured:

    * ``recv_hidden_sorted``  — ``empty_like``+``copy_`` of
      ``dispatch_result.dispatched_tokens`` (the one required materialization;
      with ``workspace`` the copy lands in a fixed-address persistent slice
      instead of a fresh allocation);
    * ``recv_weights_sorted`` — ``dispatch_result.received_routing_weights``
      (FP32 workspace) cast to the operator's BF16 activation dtype;
    * ``fc1_output``          — reference to the freshly allocated FC1 output
      (already a workspace slice when the caller passed one in).

    ``swiglu_out_weighted`` is produced inside ``_fc2_combine_shadow_activation``
    and rides that helper's two-tuple return instead of this call.
    """
    dispatched_tokens = dispatch_result.dispatched_tokens
    if workspace is not None:
        m = dispatched_tokens.shape[0]
        recv_hidden_sorted = workspace["recv_hidden_sorted"][:m]
        if tuple(recv_hidden_sorted.shape) != tuple(dispatched_tokens.shape):
            raise ValueError(
                "workspace recv_hidden_sorted slice "
                f"{tuple(recv_hidden_sorted.shape)} does not cover "
                f"dispatched_tokens {tuple(dispatched_tokens.shape)}"
            )
        recv_weights_sorted = workspace["recv_weights_sorted"][:m]
        recv_weights_sorted.copy_(dispatch_result.received_routing_weights)
    else:
        recv_hidden_sorted = torch.empty_like(dispatched_tokens)
        recv_weights_sorted = dispatch_result.received_routing_weights.to(
            op.activation_dtype
        )
    recv_hidden_sorted.copy_(dispatched_tokens)
    return {
        "recv_hidden_sorted": recv_hidden_sorted,
        "recv_weights_sorted": recv_weights_sorted,
        "fc1_output": dispatch_result.fc1_output,
    }


def _weight_reference_section(gate_up_weight, down_weight, ffn_dim: int) -> dict:
    """Zero-copy weight references for the backward (plan §3.1B).

    ``fc1_1`` / ``fc1_2`` / ``fc1_combined`` are stride views of the packed
    ``[E, H, 2F]`` gate/up table and are never materialized (the replay
    materializes ~2.3 GiB per half; the backward only reads ``fc1_1``'s dtype).
    ``fc1_combined`` keeps the transposed stride view —
    ``kernels/combine_fc1_bwd.py`` addresses it through explicit strides.
    Layout check: ``pack_gate_up_weights`` packs gate first / up last along the
    output dim, so ``dim=1`` of the transpose reads ``[gate F; up F]`` exactly
    like the replay's ``cat([fc1_1, fc1_2], dim=1)``.
    """
    return {
        "fc2": down_weight,
        "fc1_combined": gate_up_weight.transpose(1, 2),
        "fc1_1": gate_up_weight[:, :, :ffn_dim].transpose(1, 2),
        "fc1_2": gate_up_weight[:, :, ffn_dim:].transpose(1, 2),
    }


def assemble_native_saved(
    op,
    plan: MoERoutingPlan,
    snapshot: dict,
    activations,
    *,
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    selected_experts: torch.Tensor,
) -> dict:
    """T3 — assemble the native ``saved`` dict from the T1/T2 captures.

    ``activations`` is the :func:`capture_activations` result (plus the
    ``swiglu_out_weighted`` entry the FC2 helper returns).  The weight
    references (§3.1B) are zero-copy views of the forward inputs — the caller
    must not mutate the weights between forward and backward (the standard
    save-for-backward assumption).  ``selected_experts`` is the forward input,
    held by reference (same semantics as the replay oracle).  The defensive
    gate keys (``_routing_generation`` / ``_owner_token``, python ints) let the
    backward reject a saved dict whose operator has since replayed another
    plan.
    """
    ffn_dim = int(gate_up_weight.shape[2] // 2)
    saved = dict(snapshot)
    if activations is not None:
        saved.update(activations)
    saved.update(_weight_reference_section(gate_up_weight, down_weight, ffn_dim))
    # Activation config for the step-2 derivative: the fused forward selects
    # SwiGLU or SiTU-GLU at construction, and the backward must differentiate
    # the same one (op lifts the fields in __init__).
    _situ_beta = getattr(op, "situ_beta", None)
    _situ_linear_beta = getattr(op, "situ_linear_beta", None)
    saved.update(
        batch_size=int(hidden_states.shape[0]),
        hidden_dim=int(hidden_states.shape[1]),
        ffn_dim=ffn_dim,
        topk=int(op.top_k),
        world_size=int(op.world_size),
        ep_rank=int(op.rank),
        ep_group=op.ep_group,
        experts_per_rank=int(op.experts_per_rank),
        selected_experts=selected_experts,
        activation=str(getattr(op, "activation", None) or "swiglu"),
        situ_beta=1.0 if _situ_beta is None else float(_situ_beta),
        situ_linear_beta=(
            None if _situ_linear_beta is None else float(_situ_linear_beta)
        ),
        _routing_generation=int(plan.generation),
        _owner_token=id(op._routing_owner_token),
    )
    return saved


__all__ = [
    "assemble_native_saved",
    "capture_activations",
    "snapshot_plan_metadata",
]
