# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The autograd.Function layer of the Ascend MoE operator.

Home of :class:`MegaMoEFunction` — the Stage H3 merged autograd Function
(master plan §3.2.3) — moved here from :mod:`mega_moe.ops.backward`
(2026-09-15): backward.py keeps the 5-op orchestrator and the torch-forward
autograd shim, while the fused operator's user-facing ``apply`` entry and its
cross-step persistent-state contract live in this module.

``forward`` calls :class:`mega_moe.ops.forward.FusedMoEForward` with
``return_saved=True`` — the backward's saved values are captured inside the
fused pass itself (no torch replay, no HCCL re-dispatch).  ``backward`` runs
:func:`mega_moe.ops.backward.moe_backward_triton` on that dict (the 5-op
orchestrator, which dispatches to the one-launch mega kernel under
``MOE_BWD_MEGA=1``).
"""

import torch

from .backward import moe_backward_triton

# Cross-step runtime state of the ONE-launch mega backward
# (MOE_BWD_MEGA=1) that caches on `saved` but must outlive it: the
# grow-on-demand symmetric slabs (re-dispatch source under
# MOE_SAVED_RECOMPUTE=1, the dedicated combine return slab, the optional
# tile-signal slabs) plus the SET-epoch counters that have to stay
# monotonic over them.  An integrated framework builds a FRESH saved dict
# every forward (return_saved=True per step), so without carrying these on
# the caller-owned `state` every backward would re-allocate its slabs
# (skewing the symmetric heap upward ~235MB/layer/iter at the kimi mock
# shape) and restart the epochs at 1 over stale slot values.
_MEGA_PERSISTENT_KEYS = (
    "_mega_redispatch_buf", "_mega_combine_buf",
    "_mega_b3_signal_mem", "_mega_b1_signal_mem",
    "_mega_b3_signal_epoch", "_mega_b1_signal_epoch",
    "_mega_ts_buf", "_mega_wait1_buf",
)


class MegaMoEFunction(torch.autograd.Function):
    """One-pass fused EP-MoE whose backward runs on the native saved dict.

    ``forward`` calls :class:`mega_moe.ops.forward.FusedMoEForward` with
    ``return_saved=True`` — the backward's saved values are captured inside
    the fused pass itself (no torch replay, no HCCL re-dispatch).  ``backward``
    runs :func:`moe_backward_triton` on that dict.

    Args (``apply``, in order):

        op                ``FusedMoEForward`` that will run the pass (and, for
                          the MoonEP layout, lends the replica grad transport)
        hidden_states     ``[B, H]``                                (grad)
        routing_weights   ``[B, topk]`` contiguous FP32             (grad)
        selected_experts  ``[B, topk]`` int expert ids              (no grad)
        gate_up_weight    packed ``[E_p, H, 2F]`` gate/up table     (grad)
        down_weight       ``[E_p, H, F]``                           (grad)
        peer_mem          shared symmetric buffer at heap offset 0 — must be
                          the session's FIRST symmetric allocation.  ``None``
                          is rejected: the buffer is caller-owned and we never
                          fall back to ``op.context.peer_mem``.
        state             caller-owned persistent namespace carrying the
                          backward's cross-step runtime state (``signal_mem`` /
                          ``epoch``): injected at backward entry, written back
                          after the launch.  Without it every step would
                          re-allocate symmetric tile-signal slots (symmetric
                          heap leak) and restart the SET epoch at 1 (stale
                          signal values would read as fresh).  ``None`` keeps
                          one-shot semantics for a single backward.

    Gates (§3.2 item 4): the saved dict embeds ``_routing_generation`` /
    ``_owner_token`` python ints; ``backward`` rejects a saved whose plan a
    later forward of this operator has already superseded — or that belongs to
    another operator instance — instead of silently reading the rewritten
    planning workspace.  ``moe_backward_triton`` additionally asserts the
    single-in-flight receive buffer (``recv_hidden_sorted`` must not alias
    ``peer_mem``).
    """

    @staticmethod
    def forward(ctx, op, hidden_states, routing_weights, selected_experts,
                gate_up_weight, down_weight, peer_mem, state):
        if peer_mem is None:
            raise ValueError(
                "peer_mem must be passed explicitly: the backward symmetric "
                "buffer must be the session's first ACLSHMEM allocation and "
                "cannot default to op.context.peer_mem"
            )
        with torch.no_grad():
            output, saved = op.forward(
                hidden_states,
                selected_experts,
                gate_up_weight,
                down_weight,
                routing_weights,
                return_saved=True,
            )
        if "recv_counts_by_source_expert" in saved:
            # Single-kernel forward (enable_single_kernel_forward): the saved
            # dict is the MINIMAL contract (fc1_output + receive-layout
            # tables).  Expand it into the full backward contract — closed-form
            # plan tables, weight references from the apply-time inputs (never
            # the operator's per-call-refreshed staging buffers), scalars, and
            # the two layout tripwires (see _single_saved_adapter).
            from ._single_saved_adapter import enrich_single_kernel_saved
            saved = enrich_single_kernel_saved(
                op, saved,
                hidden_states=hidden_states,
                gate_up_weight=gate_up_weight,
                down_weight=down_weight,
            )
        # Optional host swap of the ONE big saved activation: with
        # MEGAMOE_FC1_OFFLOAD=1 fc1_output moves to a pooled pinned host
        # buffer on a side stream (framework async_offload.py SwapTensor
        # idiom — the native saved dict is invisible to
        # saved_tensors_hooks, so the framework mechanism can't do this)
        # and the backward entry H2Ds it back before anything reads it.
        from ._fc1_host_offload import maybe_offload_fc1
        maybe_offload_fc1(saved)
        # The saved intermediates are freshly computed views/clones (not the
        # forward inputs), and the weight views must stay pinned until the
        # backward — stash on ctx instead of save_for_backward, mirroring
        # MegaMoEBackwardFunction.  hidden_states (the pre-dispatch token
        # copy) rides along the same way: only MOE_SAVED_RECOMPUTE=1 reads
        # it (the backward-side re-dispatch recompute), everything else
        # ignores it.
        ctx.op = op
        ctx.saved = saved
        ctx.peer_mem = peer_mem
        ctx.state = state
        ctx.hidden_states = hidden_states
        return output

    @staticmethod
    def backward(ctx, dy):
        op = ctx.op
        saved = ctx.saved
        if saved.get("_owner_token") != id(op._routing_owner_token):
            raise RuntimeError(
                "the saved dict belongs to another operator instance; run "
                "the forward and backward on the same operator"
            )
        if (
            saved.get("_routing_generation") != op._routing_generation
            and not saved.get("_single_kernel_snapshot")
        ):
            raise RuntimeError(
                "the saved dict belongs to a routing plan this operator "
                "has already superseded; run backward before the next "
                "forward on the operator"
            )
        # _single_kernel_snapshot skips the generation check: that dict is a
        # full snapshot (every tensor is a clone / cast copy / fresh build /
        # caller-held input ref — see _single_saved_adapter), so a later
        # forward on the SAME operator (a framework host sharing one operator
        # across same-shape MoE layers, ep_plan.megamoe_shared_op) cannot
        # rewrite anything it reads.  The 5-op saved contract aliases the
        # operator's planning workspaces and keeps the strict check.
        # Host swap back: if the forward offloaded fc1_output, H2D it onto
        # a fresh device tensor now — every consumer below (the mega launch
        # and the orchestrator) runs on this stream, ordered after the copy.
        from ._fc1_host_offload import maybe_reload_fc1
        maybe_reload_fc1(saved)
        # §3.4 state injection: reuse the persistent symmetric tile-signal
        # slots and keep the SET epoch monotonically advancing.  The first
        # epoch must be >= 1 — a freshly zeroed slot already reads as 0.
        if ctx.state is not None:
            if getattr(ctx.state, "signal_mem", None) is not None:
                saved["_bwd_tile_signal_mem"] = ctx.state.signal_mem
            saved["_bwd_tile_signal_epoch"] = max(
                int(getattr(ctx.state, "epoch", 0)), 1
            )
            # mega-bwd slabs/epochs (see _MEGA_PERSISTENT_KEYS): seed the
            # fresh saved dict from the persistent state so the wrapper's
            # _ensure_* helpers find and reuse last step's allocations.
            for key, val in getattr(ctx.state, "mega_persistent", {}).items():
                if val is not None:
                    saved[key] = val
        # MoonEP physical saved (Stage N): sink the replica weight gradients
        # into the borrowed symmetric tables and owner-pull every copy — the
        # §3.3 autograd-side transport hookup.  Lending invalidates the
        # replica weight cache, so no forward may run on this operator until
        # the borrowed transport has been reduced.
        grad_transport = None
        if saved.get("use_moonep"):
            grad_transport = op.lend_replica_weight_tables_for_grad()
        grads = moe_backward_triton(saved, dy, ctx.peer_mem,
                                    grad_transport=grad_transport,
                                    hidden_states=getattr(
                                        ctx, "hidden_states", None))
        # Write the lazily allocated signal memory and the bumped epoch back
        # so the next step reuses both.
        if ctx.state is not None:
            ctx.state.signal_mem = saved.get("_bwd_tile_signal_mem")
            ctx.state.epoch = saved.get("_bwd_tile_signal_epoch", 1)
            ctx.state.mega_persistent = {
                key: saved[key] for key in _MEGA_PERSISTENT_KEYS
                if key in saved
            }
        # The wgrad halves are [E, F, H]; merge back into the packed Kimi
        # [E, H, 2F] layout of the gate_up_weight input.
        grad_gate_up = torch.cat(
            (grads["grad_fc1_1"], grads["grad_fc1_2"]), dim=1
        ).transpose(1, 2)
        # apply-argument order: (op, hidden_states, routing_weights,
        # selected_experts, gate_up_weight, down_weight, peer_mem, state)
        return (
            None,                           # op
            grads["grad_hidden"],           # hidden_states
            grads["grad_routing_weights"],  # routing_weights
            None,                           # selected_experts
            grad_gate_up,                   # gate_up_weight
            grads["grad_fc2"],              # down_weight
            None,                           # peer_mem
            None,                           # state
        )
