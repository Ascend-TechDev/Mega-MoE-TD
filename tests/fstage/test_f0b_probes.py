# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage F0b pre-integration probes (master plan ``MEGAMOE_MASTER_PLAN.md`` §5, row F0b).

Four 8-card probes that gate the MindSpeed-MM ``megamoe`` adaptation layer
(Stage F2).  All four run at the Kimi-K3 reduced-layer EP shape
``hidden=3584, ffn=3072, num_experts=32, topk=8`` (``epn=4`` at world size 8)
with ``receive_capacity_factor == world_size`` (dropless), the configuration
the adaptation layer will pin (plan §4.2).  The case functions accept any EP
world size that divides ``num_experts`` so the orchestrator can smoke them at
w2/w4 before spending the 8-card window; the pytest entries below pin
``world_size=8``.

Probes
------
1. ``probe1``  fused-forward parity for ``activation="situglu"`` with the Kimi
   parameters ``situ_beta=4.0`` / ``situ_linear_beta=25.0`` against an
   *independent* FP32 reference: the grouped-per-expert dispatch math of
   ``tests._moe_baselines.torch_moe_fwd_golden`` (transcribed below,
   parameterized on the activation) with the Kimi ``SituAndMul`` formula
   applied in FP32.  Gate/up weights come from distinct seeds so a gate/up
   swap cannot pass by symmetry (plan F0b ①, risk R8).
2. ``probe2``  ``MegaMoEFunction`` forward+backward at the real shape (with
   the production ``situglu`` activation) vs the hand-written eager 5-op
   backward baseline on the SiTU-converted torch-replay ``saved`` (the five
   canonical grad keys), exactly like ``run_megamoe_situglu_autograd_case``
   but at Kimi shape/8-card (plan F0b ②).
3. ``probe3``  signal_mem/epoch cross-step reuse: two consecutive
   forward+backward steps through ONE persistent caller ``state`` vs two steps
   that each allocate a FRESH state — every canonical grad must be
   bit-identical across and between the two regimes (plan F0b ③, risks
   R4/R5).
4. ``probe4``  workspace-reuse safety of one shared ``FusedMoEForward``
   running two consecutive forwards vs two independent instances: outputs must
   match within the forward output tolerance (plan F0b ④, risk R2) — the
   premise of the F2 ``_FwdOpPool``.

Static findings baked into this file (verified against the source, no NPU)
--------------------------------------------------------------------------
* **``top_k`` is fixed at operator construction.**  ``FusedMoEForward.__init__
 `` stores ``self.top_k`` and ``_validate_topk_indices``
  (``src/mega_moe/ops/forward.py:450-460``) rejects any ``selected_experts``
  whose second dimension differs.  Varying ``top_k`` therefore requires a
  separate operator instance: the F2 ``_FwdOpPool`` shape key MUST include
  ``top_k`` (alongside hidden/num_experts/max_tokens_per_rank).  Probe 4
  instead varies the token count (≤ ``max_tokens_per_rank``) and the routing
  seed — both legal runtime inputs — and documents this restriction.
* **The fused backward's SiTU branch landed (2026-09-01).**
  ``kernel_swiglu_bwd`` and the fused step2+step3 kernel now carry
  ``ACTIVATION``/``HAS_LINEAR_BETA`` selectors with the SiTU derivative
  (mirroring the host repo ``mindspeed_mm/fsdp/ops/glu/situ_triton.py``
  backward), and ``assemble_native_saved`` embeds ``activation``/``situ_beta``/
  ``situ_linear_beta`` for the backward.  Probe 2 therefore runs the production
  ``situglu`` activation end-to-end (forward AND backward) at the real shape —
  the earlier silu-only workaround is gone; probe 3 keeps ``swiglu`` so both
  derivative branches have 8-card coverage.

Memory notes for the 8-card runs (per rank, 910B 64 GiB HBM)
-------------------------------------------------------------
* Symmetric heap (``MOE_FUSED_ASH_SIZE_GB``, default 4 GiB via
  ``kit.get_ash_size_bytes(default_gb=4)``): backward peer_mem
  ``512*8*8*(3584+8)`` rows ≈ 225 MiB + the operator context peer_mem
  (``512*8*8*3584`` bf16) ≈ 225 MiB + combine storage ≈ 30 MiB + signal slots
  < 1 MiB  ⇒  ≈ 0.5 GiB per live operator pair; probe 4 keeps at most one
  operator alive.  Well inside 4 GiB.
* Device HBM (probe 2/3, worst-case receive M=32768 rows): native saved ≈
  0.7 GiB, replay golden saved + eager-baseline intermediates ≈ 1.5 GiB
  transient, weights + Function leaves/grads ≈ 1.1 GiB  ⇒  peak ≈ 3-4 GiB.
  Probe 3 keeps the full probe-2 shape (no shrink needed) and additionally
  leaks two fresh backward ``signal_mem`` allocations (≈ 0.5 MiB each) by
  design — that leak is precisely what the persistent state avoids.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.distributed as dist

from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
from mega_moe.ops._torch_forward import moe_forward
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    backward_torch_baseline,
    compare_backward_gradients,
    make_down_weights,
    make_gate_up_weights,
    make_routing_weights,
    prepare_inputs,
)
from tests._numeric import (
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    assert_close,
    diagnose,
)

# The H3 merged native-saved autograd Function (plan §3.2.3).  Collection must
# keep working on a tree without it, so the import degrades to ``None`` and the
# probe entries skip (same pattern as tests/layer/test_moe_suite.py).
try:
    from mega_moe.ops import MegaMoEFunction  # noqa: E402
except ImportError:
    try:
        from mega_moe.ops.backward import MegaMoEFunction  # noqa: E402
    except ImportError:
        MegaMoEFunction = None


# ---------------------------------------------------------------------------
# Kimi-K3 reduced-layer shape (plan §2.1: latent MoE, experts work at H=3584)
# and the Kimi SiTU activation parameters (config.json: situ_beta=4.0,
# situ_linear_beta=25.0).
# ---------------------------------------------------------------------------
KIMI_HIDDEN = 3584
KIMI_FFN = 3072
KIMI_NUM_EXPERTS = 32
KIMI_TOPK = 8
F0B_TOKENS = 512

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

# The five canonical backward gradient keys shared by
# ``compare_backward_gradients`` and ``moe_backward_triton``'s result dict.
F0B_GRAD_KEYS = (
    "grad_hidden",
    "grad_routing_weights",
    "grad_fc1_1",
    "grad_fc1_2",
    "grad_fc2",
)


def _probe_shape(world_size: int):
    """Return the probe shape tuple, asserting the EP divisibility."""
    if KIMI_NUM_EXPERTS % world_size:
        raise ValueError(
            f"the F0b probes need num_experts={KIMI_NUM_EXPERTS} divisible by "
            f"the EP world size (got {world_size}); Kimi-real epn=4 is w8"
        )
    return (
        F0B_TOKENS,
        KIMI_HIDDEN,
        KIMI_FFN,
        KIMI_NUM_EXPERTS,
        KIMI_TOPK,
        KIMI_NUM_EXPERTS // world_size,
    )


def _situglu_config(world_size: int) -> MoEForwardConfig:
    """Production forward config: Kimi SiTU-GLU, dropless receive capacity."""
    return MoEForwardConfig(
        receive_capacity_factor=float(world_size),
        activation="situglu",
        situ_beta=SITU_BETA,
        situ_linear_beta=SITU_LINEAR_BETA,
    )


def _swiglu_config(world_size: int) -> MoEForwardConfig:
    """Probe-3 config: default swiglu (probe 3 covers the silu derivative;
    probe 2 runs the production situglu)."""
    return MoEForwardConfig(receive_capacity_factor=float(world_size))


# ---------------------------------------------------------------------------
# Independent FP32 SiTU-GLU reference (probe 1 / probe 4 oracle)
# ---------------------------------------------------------------------------

def _situ_and_mul_fp32(gate, up, beta, linear_beta):
    """Kimi ``SituAndMul`` in FP32.

    Transcribed from the HOST repository
    ``mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py:121-142``
    (``class SituAndMul``)::

        situ_a = beta * tanh(gate / beta) * sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * tanh(up / linear_beta)
        return situ_a * up

    The operator kernel agrees term for term
    (``src/mega_moe/kernels/weighted_swiglu.py:61-73``: ``situ_beta *
    tanh(gate / situ_beta) * sigmoid(gate)``, then the optional
    ``situ_linear_beta * tanh(up / situ_linear_beta)`` on the up half, then the
    FP32 routing-weight scale).  Not imported from the host repo on purpose:
    the probe must stay independent of the thing it gates.
    """
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return situ_a * up


@torch.no_grad()
def _torch_moe_fwd_golden_activation(
    hidden_states,
    routing_weights,
    expert_indices,
    w_gate_local,
    w_up_local,
    w2_local,
    num_tot_experts,
    ep_group,
    activation_fn: Callable,
):
    """Independent full post-routing MoE forward golden, activation-parameterized.

    The dispatch/combine math mirrors ``tests._moe_baselines.torch_moe_fwd_golden``
    (count exchange -> token/weight/expert-id all-to-all -> stable local-expert
    grouping -> per-expert FP32 GEMMs -> activation -> FP32 route scale -> FC2 ->
    reverse all-to-all -> undo sorts -> FP32 top-k sum).  That helper hard codes
    ``silu(gate) * up``; this local variant swaps in ``activation_fn(gate, up)``
    so probe 1 can pin the Kimi SiTU-GLU formula without touching the shared
    baselines.  Everything else — dtypes, sort keys, split sizes — is identical.

    ``activation_fn(gate_bf16, up_bf16)`` receives the BF16 FC1 halves and must
    return the FP32 activated product (the FP32 routing weight is applied by
    this golden, mirroring the operator).
    """
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be bfloat16")
    if routing_weights.dtype != torch.float32:
        raise ValueError("routing_weights must be float32")
    if routing_weights.shape != expert_indices.shape:
        raise ValueError("routing_weights and expert_indices must have the same shape")
    if w_gate_local.shape != w_up_local.shape:
        raise ValueError("w_gate_local and w_up_local must have the same shape")
    if any(
        weight.dtype != torch.bfloat16
        for weight in (w_gate_local, w_up_local, w2_local)
    ):
        raise ValueError("all expert weights must be bfloat16")

    device = hidden_states.device
    num_tokens, hidden = hidden_states.shape
    topk = expert_indices.shape[1]
    world_size = dist.get_world_size(group=ep_group)
    experts_per_rank, ffn_dim, weight_hidden = w_gate_local.shape
    if weight_hidden != hidden:
        raise ValueError("gate/up reduction dimension must equal hidden size")
    if w2_local.shape != (experts_per_rank, hidden, ffn_dim):
        raise ValueError(
            "w2_local must have shape [experts_per_rank, hidden, ffn_dim]")
    if num_tot_experts != experts_per_rank * world_size:
        raise ValueError("num_tot_experts must equal experts_per_rank * world_size")

    # ---- Dispatch: count exchange, then token/weight/expert-id all-to-all ----
    hidden_repeated = hidden_states.repeat_interleave(topk, dim=0)
    flat_weights = routing_weights.reshape(-1)
    flat_experts = expert_indices.reshape(-1).long()
    valid_mask = (flat_experts >= 0) & (flat_experts < num_tot_experts)
    valid_indices = torch.where(valid_mask)[0]

    hidden_valid = hidden_repeated[valid_indices]
    weights_valid = flat_weights[valid_indices]
    experts_valid = flat_experts[valid_indices]
    dest_ranks = (experts_valid // experts_per_rank).to(torch.int32)

    rank_sort = torch.argsort(dest_ranks.to(torch.float32), stable=True)
    tokens_send = hidden_valid[rank_sort].contiguous()
    # Routing weights retain FP32 precision across transport.
    weights_send = weights_valid[rank_sort].contiguous()
    experts_send = experts_valid[rank_sort].to(torch.int32).contiguous()
    sorted_dest_ranks = dest_ranks[rank_sort]

    send_counts = torch.bincount(sorted_dest_ranks, minlength=world_size).to(
        torch.int32
    )
    recv_counts = torch.empty(world_size, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()
    total_send = int(send_counts.sum().item())
    total_recv = int(recv_counts.sum().item())

    tokens_recv = torch.empty(
        (total_recv, hidden), dtype=torch.bfloat16, device=device)
    weights_recv = torch.empty((total_recv,), dtype=torch.float32, device=device)
    experts_recv = torch.empty((total_recv,), dtype=torch.int32, device=device)
    dist.all_to_all_single(
        tokens_recv, tokens_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(
        weights_recv, weights_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(
        experts_recv, experts_send, recv_splits, send_splits, group=ep_group)

    # ---- Local experts: stable grouping, FC1, activation, route scale, FC2 --
    local_experts = experts_recv % experts_per_rank
    local_sort = torch.argsort(local_experts.to(torch.float32), stable=True)
    tokens_grouped = tokens_recv[local_sort]
    weights_grouped = weights_recv[local_sort]
    experts_grouped = local_experts[local_sort]

    gate_out = torch.empty(
        (total_recv, ffn_dim), dtype=torch.bfloat16, device=device)
    up_out = torch.empty_like(gate_out)
    for local_expert in range(experts_per_rank):
        expert_mask = experts_grouped == local_expert
        if not bool(expert_mask.any()):
            continue
        expert_input = tokens_grouped[expert_mask].float()
        gate_out[expert_mask] = (
            expert_input @ w_gate_local[local_expert].T.float()
        ).to(torch.bfloat16)
        up_out[expert_mask] = (
            expert_input @ w_up_local[local_expert].T.float()
        ).to(torch.bfloat16)

    # The ONLY deviation from the shared golden: the parameterized activation
    # (probe 1 passes the Kimi SituAndMul FP32 formula).
    weighted_activation = (
        activation_fn(gate_out.float(), up_out.float())
        * weights_grouped.float()[:, None]
    ).to(torch.bfloat16)

    fc2_grouped = torch.empty(
        (total_recv, hidden), dtype=torch.bfloat16, device=device)
    for local_expert in range(experts_per_rank):
        expert_mask = experts_grouped == local_expert
        if not bool(expert_mask.any()):
            continue
        fc2_grouped[expert_mask] = (
            weighted_activation[expert_mask].float()
            @ w2_local[local_expert].T.float()
        ).to(torch.bfloat16)

    # ---- Combine: undo receive grouping, reverse A2A, undo dispatch sort ----
    inverse_local_sort = torch.argsort(local_sort.to(torch.float32))
    fc2_recv_order = fc2_grouped[inverse_local_sort].contiguous()
    combined_rank_sorted = torch.empty(
        (total_send, hidden), dtype=torch.bfloat16, device=device)
    dist.all_to_all_single(
        combined_rank_sorted,
        fc2_recv_order,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=ep_group,
    )

    inverse_rank_sort = torch.argsort(rank_sort.to(torch.float32))
    combined_routes = torch.zeros(
        (num_tokens * topk, hidden), dtype=torch.bfloat16, device=device)
    combined_routes[valid_indices] = combined_rank_sorted[inverse_rank_sort]
    combined_routes = combined_routes.view(num_tokens, topk, hidden)
    output_fp32 = torch.zeros(
        (num_tokens, hidden), dtype=torch.float32, device=device)
    for route_slot in range(topk):
        output_fp32 += combined_routes[:, route_slot].float()
    return output_fp32.to(torch.bfloat16)


def _situ_golden(hs, routing_weights, expert_indices, w_gate, w_up, w2,
                 num_experts, ep_group):
    """Probe-1/4 oracle: the local golden with the Kimi SiTU formula plugged in."""
    return _torch_moe_fwd_golden_activation(
        hs,
        routing_weights,
        expert_indices,
        w_gate,
        w_up,
        w2,
        num_experts,
        ep_group,
        activation_fn=lambda gate, up: _situ_and_mul_fp32(
            gate, up, SITU_BETA, SITU_LINEAR_BETA
        ),
    )


# ---------------------------------------------------------------------------
# Shared probe plumbing
# ---------------------------------------------------------------------------

def _fold_and_raise(failures, label, rank, device, ep_group):
    """MIN-fold the local verdict across the EP group and raise on any failure.

    Collect-then-fold (instead of raising mid-case) so a failing rank cannot
    strand its peers inside a later collective: every rank reaches the fold,
    the first failure message is reported by rank 0, and the case raises once.
    """
    ok = not failures
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if rank == 0 and failures:
        print(f"[FAIL] {label}: " + "; ".join(failures), flush=True)
    if not bool(flag.item()):
        detail = "; ".join(failures) if failures else "a peer rank failed"
        raise AssertionError(f"{label}: {detail}")


def _check_output_close(actual, expected, failures, what):
    """Append a tolerance failure instead of raising (see _fold_and_raise)."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        failures.append(
            f"{what}: shape/dtype kernel={tuple(actual.shape)}/{actual.dtype} "
            f"golden={tuple(expected.shape)}/{expected.dtype}"
        )
        return
    if not bool(torch.isfinite(actual.float()).all()):
        failures.append(f"{what}: output contains non-finite values")
    try:
        assert_close(actual, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
    except AssertionError as exc:
        failures.append(
            f"{what}: {str(exc).splitlines()[0]} | {diagnose(actual, expected)}"
        )


def _run_function_step(op, hidden_states, routing_weights, selected_experts,
                       packed_w1, w2, dy, peer_mem, state):
    """One MegaMoEFunction forward+backward step; returns the four grad leaves."""
    hidden_leaf = hidden_states.clone().requires_grad_(True)
    routing_leaf = routing_weights.clone().requires_grad_(True)
    gate_up_leaf = packed_w1.clone().requires_grad_(True)
    down_leaf = w2.clone().requires_grad_(True)
    output = MegaMoEFunction.apply(
        op,
        hidden_leaf,
        routing_leaf,
        selected_experts,
        gate_up_leaf,
        down_leaf,
        peer_mem,
        state,
    )
    output.backward(dy)
    return hidden_leaf, routing_leaf, gate_up_leaf, down_leaf


def _function_grads(leaves, ffn_dim):
    """Map the Function's autograd grads onto the five canonical check keys.

    ``grad_gate_up`` arrives merged into the Kimi ``[E, H, 2F]`` layout
    (``cat(g1, g2, dim=1).transpose(1, 2)``); the canonical keys expect the
    replay's ``[E, F, H]`` halves (mirrors ``_megamoe_function_grads``).
    """
    hidden_leaf, routing_leaf, gate_up_leaf, down_leaf = leaves
    grad_gate_up = gate_up_leaf.grad
    return dict(
        grad_hidden=hidden_leaf.grad,
        grad_routing_weights=routing_leaf.grad,
        grad_fc1_1=grad_gate_up[:, :, :ffn_dim].transpose(1, 2),
        grad_fc1_2=grad_gate_up[:, :, ffn_dim:].transpose(1, 2),
        grad_fc2=down_leaf.grad,
    )


def _require_runtime(label):
    """Shared NPU/ACLSHMEM/API availability guard for the worker functions."""
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError(f"{label} requires torch_npu and ACLSHMEM")


def _make_case_weights(num_experts, hidden, ffn, world_size, rank, device):
    """Random rank-local weights with DISTINCT gate vs up seeds (probe 1)."""
    dtype = torch.bfloat16
    # make_gate_up_weights draws gate/up from independent tables (seeds 142 vs
    # 143 inside make_w1), so a gate/up swap in the operator or the packed
    # layout cannot cancel out by symmetry.
    w_gate, w_up = make_gate_up_weights(
        num_experts, hidden, ffn, world_size, rank, dtype, device
    )
    w2 = make_down_weights(
        num_experts, hidden, ffn, world_size, rank, dtype, device
    )
    packed_w1 = pack_gate_up_weights(w_gate, w_up)
    return w_gate, w_up, packed_w1, w2


# ---------------------------------------------------------------------------
# Probe 1 — fused situglu forward parity vs an independent FP32 reference
# ---------------------------------------------------------------------------

def run_f0b_probe1_situglu_parity(rank: int, world_size: int) -> None:
    """F0b ①: ``activation="situglu"`` (β=4.0, lβ=25.0) forward parity.

    The fused operator at the Kimi-real shape (H=3584, F=3072, E=32, K=8,
    512 tokens/rank) is compared against a local FP32 golden that re-derives
    the dispatch with its own all-to-alls and applies the transcribed Kimi
    ``SituAndMul`` formula.  Gate/up weights come from different seeds, so a
    gate/up swap cannot pass (the activation is not symmetric in its two
    halves).  Tolerances: the suite's forward output figures
    (``OUTPUT_RTOL=4e-2`` / ``OUTPUT_ATOL=4e-2``).
    """
    _require_runtime("f0b probe1")
    tokens, hidden, ffn, num_experts, topk, epn = _probe_shape(world_size)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"f0b-probe1-situglu-parity-w{world_size}-epn{epn}"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=4)
    ):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=_situglu_config(world_size),
        )
        try:
            failures = []
            w_gate, w_up, packed_w1, w2 = _make_case_weights(
                num_experts, hidden, ffn, world_size, rank, device
            )
            # Distinct-seed sanity: if the two local tables ever collided the
            # swap detector below would be vacuous.
            if bool(
                torch.equal(
                    w_gate[:, :64, :64].float(), w_up[:, :64, :64].float()
                )
            ):
                failures.append(
                    "gate/up weight tables collided; the swap detector is "
                    "vacuous (check make_gate_up_weights seeds)"
                )
            hs, expert_indices = prepare_inputs(
                tokens, hidden, num_experts, topk, dtype, device,
                seed=7101 + rank,
            )
            routing_weights = make_routing_weights(
                tokens, topk, device, seed=7102 + rank
            )
            dist.barrier()

            actual = op.forward(hs, expert_indices, packed_w1, w2, routing_weights)
            dist.barrier()

            expected = _situ_golden(
                hs, routing_weights, expert_indices, w_gate, w_up, w2,
                num_experts, ep_group,
            )
            _check_output_close(
                actual, expected, failures, "situglu output vs fp32 reference"
            )
            _fold_and_raise(failures, label, rank, device, ep_group)
        finally:
            op.finalize()


# ---------------------------------------------------------------------------
# Probe 2 — real-shape MegaMoEFunction forward+backward vs the eager baseline
# ---------------------------------------------------------------------------

def run_f0b_probe2_real_shape_backward(rank: int, world_size: int) -> None:
    """F0b ②: native-saved autograd at the Kimi-real shape (8-card, SiTU).

    Exactly the ``run_megamoe_situglu_autograd_case`` recipe lifted to
    H=3584/F=3072/E=32/K=8/512 tokens at world size 8 with the PRODUCTION
    ``situglu`` activation (β=4.0, lβ=25.0): the eager golden is the
    ``moe_forward`` replay ``saved`` converted to SiTU (``swiglu_out_weighted``
    re-derived from the same pre-activation halves; the baseline told which
    derivative to take) plus the hand-written 5-op ``backward_torch_baseline``,
    and the five canonical grads of ``MegaMoEFunction`` must match it.
    ``peer_mem`` is sized for the worst-case dropless receive
    (``tokens*topk*world_size`` rows, i.e. capacity factor == world size like
    the H3 case) and stays the session's FIRST symmetric allocation.

    Numeric verdict discipline: the repo's ``compare_backward_gradients``
    (elementwise-bad iff ``d > atol + rtol*GLOBAL_gmax``) is authoritative.
    A per-key elementwise ``assert_close`` at the same rtol/atol is NOT added
    on top: at this accumulation depth (F=3076/3072 reductions) sub-ulp bf16
    noise on small-magnitude elements fails elementwise while being well
    inside the suite's global-gmax rule — the w8 first run tripped exactly
    that on ``grad_fc2`` (max 3.1e-2, rel 5e-3) with ``compare_backward_gradients``
    green on all ranks.  (Probe 3's step-1 numeric gate uses the same rule.)
    """
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")
    _require_runtime("f0b probe2")
    tokens, hidden, ffn, num_experts, topk, epn = _probe_shape(world_size)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"f0b-probe2-real-shape-bwd-w{world_size}-epn{epn}"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=4)
    ):
        # peer_mem must stay the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below claims its own heap
        # objects for planning and dispatch.  Rows cover the worst-case
        # per-rank receive of the dropless plan (tokens*topk*world_size).
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk * world_size, tokens * topk, hidden, dtype, rank,
            ep_group,
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=_situglu_config(world_size),
            )
            try:
                failures = []
                w_gate, w_up, packed_w1, w2 = _make_case_weights(
                    num_experts, hidden, ffn, world_size, rank, device
                )
                hs, expert_indices = prepare_inputs(
                    tokens, hidden, num_experts, topk, dtype, device,
                    seed=7201 + rank,
                )
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=7202 + rank
                )
                torch.manual_seed(7203 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                # Eager golden: the replay saved (silu) converted to SiTU —
                # dispatch metadata and the pre-activation halves are
                # activation-independent; only the weighted activation and the
                # differentiated activation change (fp32 SiTU, fp32 route
                # scale, bf16 store — the operator's numeric path).
                dist.barrier()
                with torch.no_grad():
                    _, golden_saved = moe_forward(
                        hs,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    gate = golden_saved["gate"].float()
                    up = golden_saved["up"].float()
                    situ_a = (
                        SITU_BETA
                        * torch.tanh(gate / SITU_BETA)
                        * torch.sigmoid(gate)
                    )
                    up_v = SITU_LINEAR_BETA * torch.tanh(
                        up / SITU_LINEAR_BETA
                    )
                    golden_saved["swiglu_out_weighted"] = (
                        situ_a
                        * up_v
                        * golden_saved["recv_weights_sorted"]
                        .float()
                        .unsqueeze(-1)
                    ).to(dtype)
                    golden_saved["activation"] = "situglu"
                    golden_saved["situ_beta"] = SITU_BETA
                    golden_saved["situ_linear_beta"] = SITU_LINEAR_BETA
                    golden = backward_torch_baseline(golden_saved, dy)
                del golden_saved  # ~1.5 GiB of replay intermediates at rest

                state = SimpleNamespace(signal_mem=None, epoch=0)
                dist.barrier()
                grads = _function_grads(
                    _run_function_step(
                        op, hs, routing_weights, expert_indices, packed_w1,
                        w2, dy, peer_mem, state,
                    ),
                    ffn,
                )
                for name in F0B_GRAD_KEYS:
                    value = grads[name]
                    if value is None or tuple(value.shape) != tuple(
                        golden[name].shape
                    ):
                        failures.append(
                            f"grad {name} shape "
                            f"{None if value is None else tuple(value.shape)} "
                            f"!= golden {tuple(golden[name].shape)}"
                        )
                        continue
                    if not bool(torch.isfinite(value.float()).all()):
                        failures.append(f"grad {name} contains non-finite values")
                all_ok, details = compare_backward_gradients(grads, golden)
                if rank == 0:
                    print(f"[probe2] cmp_grad rows: {details}", flush=True)
                if not all_ok:
                    failures.append(f"compare_backward_gradients: {details}")
                # The Function must persist the cross-step state (§3.4).
                if state.signal_mem is None or int(state.epoch) < 1:
                    failures.append(
                        "the Function did not persist signal_mem/epoch in the "
                        "caller state"
                    )
                _fold_and_raise(failures, label, rank, device, ep_group)
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


# ---------------------------------------------------------------------------
# Probe 3 — signal_mem/epoch cross-step reuse vs fresh allocation (bitwise)
# ---------------------------------------------------------------------------

def run_f0b_probe3_state_reuse_bitwise(rank: int, world_size: int) -> None:
    """F0b ③: persistent ``state`` reuse vs fresh allocation, bit for bit.

    Same shape and lifecycle as probe 2 (kept at the full Kimi shape — the
    per-rank HBM estimate in the module docstring leaves ample headroom on a
    64 GiB 910B).  Four forward+backward steps run through ONE operator:

    * steps 1-2 share one persistent ``state`` (``signal_mem``/``epoch``
      injected at backward entry and written back — the adaptation layer's
      cross-step regime, risks R5/R4);
    * steps 3-4 each build a FRESH state, i.e. every backward lazily
      allocates a new symmetric ``signal_mem`` and restarts the SET epoch at 1
      (the leak/stale-epoch regime the persistent state exists to avoid; the
      two extra allocations are tiny and deliberate).

    Every canonical grad of step 1 must equal step 2 bit-for-bit (epoch reuse
    did not perturb the kernels), every grad of fresh step 3 must equal fresh
    step 4, and — the plan's “连续两次 backward vs 全新分配，逐位一致” — the
    persistent regime's grads must equal the fresh regime's bitwise as well.
    Step 1 is additionally checked against the eager golden so a
    bit-stable-but-wrong kernel cannot pass silently.
    """
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")
    _require_runtime("f0b probe3")
    tokens, hidden, ffn, num_experts, topk, epn = _probe_shape(world_size)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"f0b-probe3-state-reuse-w{world_size}-epn{epn}"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=4)
    ):
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk * world_size, tokens * topk, hidden, dtype, rank,
            ep_group,
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=_swiglu_config(world_size),
            )
            try:
                failures = []
                w_gate, w_up, packed_w1, w2 = _make_case_weights(
                    num_experts, hidden, ffn, world_size, rank, device
                )
                hs, expert_indices = prepare_inputs(
                    tokens, hidden, num_experts, topk, dtype, device,
                    seed=7301 + rank,
                )
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=7302 + rank
                )
                torch.manual_seed(7303 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                dist.barrier()
                with torch.no_grad():
                    _, golden_saved = moe_forward(
                        hs,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    golden = backward_torch_baseline(golden_saved, dy)
                del golden_saved

                # Regime A: one persistent state across two consecutive steps.
                persistent = SimpleNamespace(signal_mem=None, epoch=0)
                grads_p1 = _function_grads(
                    _run_function_step(
                        op, hs, routing_weights, expert_indices, packed_w1,
                        w2, dy, peer_mem, persistent,
                    ),
                    ffn,
                )
                epoch_after_first = int(persistent.epoch)
                grads_p2 = _function_grads(
                    _run_function_step(
                        op, hs, routing_weights, expert_indices, packed_w1,
                        w2, dy, peer_mem, persistent,
                    ),
                    ffn,
                )
                # Regime B: a FRESH state for every step.
                grads_f1 = _function_grads(
                    _run_function_step(
                        op, hs, routing_weights, expert_indices, packed_w1,
                        w2, dy, peer_mem,
                        SimpleNamespace(signal_mem=None, epoch=0),
                    ),
                    ffn,
                )
                grads_f2 = _function_grads(
                    _run_function_step(
                        op, hs, routing_weights, expert_indices, packed_w1,
                        w2, dy, peer_mem,
                        SimpleNamespace(signal_mem=None, epoch=0),
                    ),
                    ffn,
                )

                # Numeric gate (once): step 1 must be a correct gradient.
                # Verdict = compare_backward_gradients only (see the probe-2
                # docstring: a stacked elementwise assert_close over-tightens
                # beyond the suite's global-gmax rule on sub-ulp bf16 noise).
                for name in F0B_GRAD_KEYS:
                    if grads_p1[name] is None or tuple(
                        grads_p1[name].shape
                    ) != tuple(golden[name].shape):
                        failures.append(
                            f"grad {name} shape "
                            f"{None if grads_p1[name] is None else tuple(grads_p1[name].shape)} "
                            f"!= golden {tuple(golden[name].shape)}"
                        )
                all_ok, details = compare_backward_gradients(grads_p1, golden)
                if not all_ok:
                    failures.append(f"compare_backward_gradients: {details}")

                # Bitwise gates: within each regime and across the regimes.
                for name in F0B_GRAD_KEYS:
                    pairs = (
                        ("persistent epoch reuse", grads_p1[name], grads_p2[name]),
                        ("fresh allocation", grads_f1[name], grads_f2[name]),
                        ("persistent vs fresh (step 1)", grads_p1[name], grads_f1[name]),
                        ("persistent vs fresh (step 2)", grads_p2[name], grads_f2[name]),
                    )
                    for what, first, second in pairs:
                        if first is None or second is None:
                            failures.append(f"{what}: grad {name} is None")
                        elif not torch.equal(first, second):
                            failures.append(
                                f"{what} changed grad {name} bitwise: "
                                f"{diagnose(first, second)}"
                            )

                # The persistent state must have advanced the epoch across the
                # two steps and kept the signal memory alive.
                if persistent.signal_mem is None:
                    failures.append(
                        "the persistent state did not retain signal_mem"
                    )
                if int(persistent.epoch) <= epoch_after_first or epoch_after_first < 1:
                    failures.append(
                        "the persistent state did not advance/write back the "
                        f"epoch (after first={epoch_after_first}, "
                        f"after second={persistent.epoch})"
                    )
                _fold_and_raise(failures, label, rank, device, ep_group)
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


# ---------------------------------------------------------------------------
# Probe 4 — one shared operator, two consecutive forwards vs fresh instances
# ---------------------------------------------------------------------------

# Probe 4's runtime-varied cases: (token count, seed base).  ``top_k`` is NOT
# varied — it is fixed at operator construction (FusedMoEForward.__init__ /
# _validate_topk_indices, src/mega_moe/ops/forward.py:450-460), so a top-k
# change needs a second operator instance and the F2 _FwdOpPool shape key
# must include it.  Token count and routing seed are legal runtime inputs
# (tokens <= max_tokens_per_rank) and exercise the same workspace-reuse path.
F0B_PROBE4_CASES = (
    (512, 7401),
    (384, 7501),
)


def run_f0b_probe4_shared_op_forward(rank: int, world_size: int) -> None:
    """F0b ④: shared-operator workspace reuse vs independent instances.

    Two consecutive forwards with different token counts (512 then 384) and
    different routing seeds run through (a) two INDEPENDENT
    ``FusedMoEForward`` instances, each constructed for and running exactly
    one case, and (b) ONE shared instance running both cases back to back.
    The shared outputs must match the independent outputs within the forward
    output tolerances, and each independent output must match the independent
    FP32 SiTU golden (so a consistently-wrong reuse cannot pass).

    The production config is used (``situglu``, β=4.0 / lβ=25.0): the shared
    operator's SECOND forward also exercises its forward-side tile-signal
    epoch reuse, which the F2 pool's 4-layers-one-instance design depends on.

    Lifecycle note: at most one operator is alive at a time (independent A ->
    finalize -> independent B -> finalize -> shared A+B -> finalize).  This
    deliberately exercises sequential construct/finalize cycles inside one
    ACLSHMEM session — itself a precondition for the F2 pool's
    ``destroy_all()``/re-``get()`` lifecycle.
    """
    _require_runtime("f0b probe4")
    tokens_max, hidden, ffn, num_experts, topk, epn = _probe_shape(world_size)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"f0b-probe4-shared-op-fwd-w{world_size}-epn{epn}"

    def build_inputs(case_tokens, seed_base):
        hs, expert_indices = prepare_inputs(
            case_tokens, hidden, num_experts, topk, dtype, device,
            seed=seed_base + rank,
        )
        weights = make_routing_weights(
            case_tokens, topk, device, seed=seed_base + 1 + rank
        )
        return hs, expert_indices, weights

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=4)
    ):
        failures = []
        w_gate, w_up, packed_w1, w2 = _make_case_weights(
            num_experts, hidden, ffn, world_size, rank, device
        )

        # Phase 1: independent instances, one case each, strictly one alive.
        independent_outputs = {}
        for case_index, (case_tokens, seed_base) in enumerate(F0B_PROBE4_CASES):
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens_max,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=_situglu_config(world_size),
            )
            try:
                hs, expert_indices, routing_weights = build_inputs(
                    case_tokens, seed_base
                )
                dist.barrier()
                output = op.forward(
                    hs, expert_indices, packed_w1, w2, routing_weights
                )
                dist.barrier()
                golden = _situ_golden(
                    hs, routing_weights, expert_indices, w_gate, w_up, w2,
                    num_experts, ep_group,
                )
                _check_output_close(
                    output, golden, failures,
                    f"independent op case {case_index} vs fp32 reference",
                )
                independent_outputs[case_index] = output.clone()
            finally:
                op.finalize()

        # Phase 2: ONE shared instance runs both cases back to back.  Its
        # planning workspace, dispatch buffers, combine storage, and forward
        # tile-signal epochs are reused across the two forwards — the exact
        # regime the F2 _FwdOpPool imposes on the four Kimi layers.
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens_max,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=_situglu_config(world_size),
        )
        try:
            for case_index, (case_tokens, seed_base) in enumerate(
                F0B_PROBE4_CASES
            ):
                hs, expert_indices, routing_weights = build_inputs(
                    case_tokens, seed_base
                )
                dist.barrier()
                output = op.forward(
                    hs, expert_indices, packed_w1, w2, routing_weights
                )
                dist.barrier()
                _check_output_close(
                    output,
                    independent_outputs[case_index],
                    failures,
                    f"shared op case {case_index} vs independent instance",
                )
        finally:
            op.finalize()
        _fold_and_raise(failures, label, rank, device, ep_group)


# ---------------------------------------------------------------------------
# pytest entries (8-card, per the F0b gate; the run_* cases also accept the
# smaller divisor worlds for cheap w2/w4 smokes before the 8-card window)
# ---------------------------------------------------------------------------

@pytest.mark.dist
@pytest.mark.functional
def test_f0b_probe1_situglu_parity(dist_test):
    dist_test(run_f0b_probe1_situglu_parity, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_f0b_probe2_real_shape_backward(dist_test):
    if MegaMoEFunction is None:
        pytest.skip(
            "H3 API pending: mega_moe does not export MegaMoEFunction yet"
        )
    dist_test(run_f0b_probe2_real_shape_backward, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_f0b_probe3_state_reuse_bitwise(dist_test):
    if MegaMoEFunction is None:
        pytest.skip(
            "H3 API pending: mega_moe does not export MegaMoEFunction yet"
        )
    dist_test(run_f0b_probe3_state_reuse_bitwise, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_f0b_probe4_shared_op_forward(dist_test):
    dist_test(run_f0b_probe4_shared_op_forward, world_size=8)
