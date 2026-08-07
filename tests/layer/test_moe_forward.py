# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Correctness tests for the standalone Ascend Mega-MoE post-routing forward.

Validates ``FusedMoEForward`` against an independent PyTorch
golden (``torch_dispatch_fc1_golden``) that performs its own host-side
``all_to_all_single`` dispatch and per-expert ``torch.matmul``. The golden never
touches the kernel's ``peer_mem``, so it catches dispatch-routing bugs (offset
collisions, mis-routing, dropped tokens) — which a self-consistent reference
reusing the kernel's dispatched buffers cannot.

The optimized path additionally validates packed gate/up FC1 followed by weighted
SwiGLU.  Its independent Torch golden accepts and transports FP32 routing weights
with the same routes, computes gate/up projections separately, and checks
raw FC1, unweighted SwiGLU, weighted SwiGLU, and the received routing-weight layout.

The full-forward golden follows the NVIDIA reference stage order while remaining a
pure Torch/HCCL implementation: dispatch, FC1, FP32 SwiGLU/routing multiplication,
FC2, reverse all-to-all, route-order restoration, and top-k reduction.  Invalid
negative or out-of-range expert ids are dropped.  Empty-receive and all-drop ranks
still enter every collective, so asymmetric routing cannot deadlock the golden.

Usage:
    source /home/w00845909/distribution/Triton-distributed-ascend/run.sh
    python -m pytest tests/layer/test_moe_forward.py -m dist -v -s
"""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import shmem as ash

from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
import mega_moe.kernels.dispatch_fc1 as dispatch_fc1_module
import mega_moe.kernels.fc2_combine as fc2_combine_module
from mega_moe.kernels.weighted_swiglu import weighted_swiglu_forward
import mega_moe.runtime.routing as routing_metadata_module
from tests._moe_dist_utils import get_ash_size_bytes, init_aclshmem
from tests._numeric import (
    APPROX_ATOL,
    APPROX_RTOL,
    LAYOUT_ATOL,
    LAYOUT_RTOL,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
)

g_ash_size = get_ash_size_bytes(default_gb=1)


# ---------------------------------------------------------------------------
#  Independent golden — own host-side all_to_all dispatch + per-expert matmul
# ---------------------------------------------------------------------------

@torch.no_grad()
def torch_dispatch_fc1_golden(x, exp_indices, w1_local, num_tot_experts, dtype, device,
                              ep_group):
    """Independent golden for dispatch + fc1.

    Does its OWN ``all_to_all_single`` dispatch of tokens and expert ids (never touches
    the kernel's ``peer_mem``), groups received tokens by local expert, and computes fc1
    via per-expert ``torch.matmul`` against this rank's EP weight slice ``w1_local``.

    Returns ``(fc1, exp_per_row, dispatched_tokens)`` in expert-grouped order
    (expert 0's tokens first, then expert 1, ...). ``exp_per_row[i]`` is the
    local expert id that produced row ``i``.
    """
    if dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError("dispatch tokens must be bfloat16")
    if w1_local.dtype != torch.bfloat16:
        raise ValueError("expert weights must be bfloat16")

    T, hidden = x.shape
    topk = exp_indices.shape[1]
    experts_per_rank = w1_local.shape[0]
    world_size = dist.get_world_size(group=ep_group)
    inter = w1_local.shape[1]

    # Filter every invalid route, including negative and out-of-range expert ids.
    flat_e = exp_indices.reshape(-1).long()
    row_tok = torch.arange(T, device=device).repeat_interleave(topk)
    keep = (flat_e >= 0) & (flat_e < num_tot_experts)
    e_keep = flat_e[keep]
    tok_keep = row_tok[keep]
    x_keep = x[tok_keep]
    dest = (e_keep // experts_per_rank).to(torch.int32)
    local_expert = (e_keep % experts_per_rank).to(torch.int32)

    # sort by dest rank (stable -> token-major within a dest)
    sort_idx = torch.argsort(dest, stable=True)
    x_send = x_keep[sort_idx].contiguous()
    local_exp_send = local_expert[sort_idx].contiguous()
    sorted_dest = dest[sort_idx]

    # exchange counts
    send_counts = torch.bincount(sorted_dest, minlength=world_size).to(torch.int32)
    recv_counts = torch.empty(world_size, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    M_local = int(recv_counts.sum().item())
    ssl = send_counts.cpu().tolist()
    srl = recv_counts.cpu().tolist()

    # exchange tokens + expert ids
    x_recv = torch.empty((M_local, hidden), dtype=dtype, device=device)
    dist.all_to_all_single(x_recv, x_send, srl, ssl, group=ep_group)
    local_exp_recv = torch.empty((M_local,), dtype=torch.int32, device=device)
    dist.all_to_all_single(local_exp_recv, local_exp_send, srl, ssl, group=ep_group)

    if M_local == 0:
        return (torch.empty((0, inter), dtype=dtype, device=device),
                torch.empty((0,), dtype=torch.int32, device=device),
                torch.empty((0, hidden), dtype=dtype, device=device))

    # group by local expert (stable -> preserves receive order within an expert)
    recv_order = torch.argsort(local_exp_recv, stable=True)
    x_grouped = x_recv[recv_order]
    exp_grouped = local_exp_recv[recv_order]

    # per-expert matmul: out = x @ w1_local[e].T
    out = torch.empty((M_local, inter), dtype=dtype, device=device)
    for e_local in range(experts_per_rank):
        mask = exp_grouped == e_local
        if mask.sum() == 0:
            continue
        out[mask] = (x_grouped[mask].float() @ w1_local[e_local].T.float()).to(dtype)
    return out, exp_grouped, x_grouped


@torch.no_grad()
def torch_dispatch_fc1_weighted_swiglu_golden(
    x,
    exp_indices,
    routing_weights,
    w_gate_local,
    w_up_local,
    num_tot_experts,
    dtype,
    device,
    ep_group,
):
    """Independent Torch golden for dispatch + packed FC1 + weighted SwiGLU.

    Token data, local expert ids, and routing weights are exchanged with independent
    ``all_to_all_single`` calls.  Routing weights remain FP32 through communication
    and activation arithmetic.  Gate and up projections are computed separately so this golden
    does not share the optimized path's packed-W1 layout.

    Returns a dictionary in deterministic expert-grouped receive order with keys:
    ``fc1_out``, ``swiglu_out``, ``weighted_swiglu_out``,
    ``routing_weight_recv``, ``local_expert_ids``, and ``dispatched_tokens``.
    """
    if routing_weights.shape != exp_indices.shape:
        raise ValueError("routing_weights and exp_indices must have the same shape")
    if routing_weights.dtype != torch.float32:
        raise ValueError("routing_weights must be float32")
    if dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError("dispatch tokens must be bfloat16")
    if w_gate_local.dtype != torch.bfloat16 or w_up_local.dtype != torch.bfloat16:
        raise ValueError("expert weights must be bfloat16")
    if w_gate_local.shape != w_up_local.shape:
        raise ValueError("w_gate_local and w_up_local must have the same shape")

    T, hidden = x.shape
    topk = exp_indices.shape[1]
    experts_per_rank = w_gate_local.shape[0]
    ffn_dim = w_gate_local.shape[1]
    world_size = dist.get_world_size(group=ep_group)

    flat_e = exp_indices.reshape(-1).long()
    flat_weight = routing_weights.reshape(-1)
    row_tok = torch.arange(T, device=device).repeat_interleave(topk)
    keep = (flat_e >= 0) & (flat_e < num_tot_experts)
    e_keep = flat_e[keep]
    tok_keep = row_tok[keep]
    weight_keep = flat_weight[keep]
    x_keep = x[tok_keep]
    dest = (e_keep // experts_per_rank).to(torch.int32)
    local_expert = (e_keep % experts_per_rank).to(torch.int32)

    # All route payloads use the same stable destination-rank permutation.
    sort_idx = torch.argsort(dest, stable=True)
    x_send = x_keep[sort_idx].contiguous()
    local_exp_send = local_expert[sort_idx].contiguous()
    weight_send = weight_keep[sort_idx].contiguous()
    sorted_dest = dest[sort_idx]

    send_counts = torch.bincount(sorted_dest, minlength=world_size).to(torch.int32)
    recv_counts = torch.empty(world_size, dtype=torch.int32, device=device)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    M_local = int(recv_counts.sum().item())
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()

    x_recv = torch.empty((M_local, hidden), dtype=dtype, device=device)
    local_exp_recv = torch.empty((M_local,), dtype=torch.int32, device=device)
    weight_recv = torch.empty((M_local,), dtype=torch.float32, device=device)
    # No early return: even a zero-receive rank must participate in every
    # collective because it can still send routes to another rank.
    dist.all_to_all_single(x_recv, x_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(
        local_exp_recv, local_exp_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(weight_recv, weight_send, recv_splits, send_splits, group=ep_group)

    recv_order = torch.argsort(local_exp_recv, stable=True)
    x_grouped = x_recv[recv_order]
    exp_grouped = local_exp_recv[recv_order]
    weight_grouped = weight_recv[recv_order]

    gate_out = torch.empty((M_local, ffn_dim), dtype=dtype, device=device)
    up_out = torch.empty_like(gate_out)
    for e_local in range(experts_per_rank):
        mask = exp_grouped == e_local
        if not bool(mask.any()):
            continue
        expert_input = x_grouped[mask].float()
        gate_out[mask] = (expert_input @ w_gate_local[e_local].T.float()).to(dtype)
        up_out[mask] = (expert_input @ w_up_local[e_local].T.float()).to(dtype)

    fc1_out = torch.cat((gate_out, up_out), dim=-1)
    swiglu_fp32 = torch.nn.functional.silu(gate_out.float()) * up_out.float()
    swiglu_out = swiglu_fp32.to(dtype)
    weighted_swiglu_out = (swiglu_fp32 * weight_grouped.float()[:, None]).to(dtype)
    return {
        "fc1_out": fc1_out,
        "swiglu_out": swiglu_out,
        "weighted_swiglu_out": weighted_swiglu_out,
        "routing_weight_recv": weight_grouped,
        "local_expert_ids": exp_grouped,
        "dispatched_tokens": x_grouped,
    }


@torch.no_grad()
def torch_moe_fwd_golden(
    hidden_states,
    routing_weights,
    expert_indices,
    w_gate_local,
    w_up_local,
    w2_local,
    num_tot_experts,
    ep_group,
):
    """Independent full post-routing Torch/HCCL MoE forward.

    Shapes are ``hidden_states [T,H]``, route tensors ``[T,K]``, gate/up weights
    ``[E_local,F,H]``, down weights ``[E_local,H,F]``, and output ``[T,H]``.
    Activations and expert weights are BF16.  ``routing_weights`` enters and is
    transported as FP32, and participates in SwiGLU multiplication as FP32.

    This intentionally does not call either the partial golden or any production
    metadata/kernel helper.  In particular, it always executes count, forward
    payload, and reverse payload collectives, including when a rank sends or
    receives zero valid routes.
    """
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be bfloat16")
    if routing_weights.dtype != torch.float32:
        raise ValueError("routing_weights must be float32")
    if routing_weights.shape != expert_indices.shape:
        raise ValueError("routing_weights and expert_indices must have the same shape")
    if w_gate_local.shape != w_up_local.shape:
        raise ValueError("w_gate_local and w_up_local must have the same shape")
    if any(weight.dtype != torch.bfloat16 for weight in (w_gate_local, w_up_local, w2_local)):
        raise ValueError("all expert weights must be bfloat16")

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

    rank_sort = torch.argsort(dest_ranks, stable=True)
    tokens_send = hidden_valid[rank_sort].contiguous()
    # Routing weights retain FP32 precision across HCCL transport.
    weights_send = weights_valid[rank_sort].contiguous()
    experts_send = experts_valid[rank_sort].to(torch.int32).contiguous()
    sorted_dest_ranks = dest_ranks[rank_sort]

    send_counts = torch.bincount(sorted_dest_ranks, minlength=world_size).to(torch.int32)
    recv_counts = torch.empty(world_size, dtype=torch.int32, device=hidden_states.device)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    send_splits = send_counts.cpu().tolist()
    recv_splits = recv_counts.cpu().tolist()
    total_send = int(send_counts.sum().item())
    total_recv = int(recv_counts.sum().item())

    tokens_recv = torch.empty(
        (total_recv, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    weights_recv = torch.empty(
        (total_recv,), dtype=torch.float32, device=hidden_states.device)
    experts_recv = torch.empty(
        (total_recv,), dtype=torch.int32, device=hidden_states.device)
    dist.all_to_all_single(
        tokens_recv, tokens_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(
        weights_recv, weights_send, recv_splits, send_splits, group=ep_group)
    dist.all_to_all_single(
        experts_recv, experts_send, recv_splits, send_splits, group=ep_group)

    # ---- Local experts: stable grouping, FC1, weighted SwiGLU, then FC2 ----
    local_experts = experts_recv % experts_per_rank
    local_sort = torch.argsort(local_experts, stable=True)
    tokens_grouped = tokens_recv[local_sort]
    weights_grouped = weights_recv[local_sort]
    experts_grouped = local_experts[local_sort]

    gate_out = torch.empty(
        (total_recv, ffn_dim), dtype=torch.bfloat16, device=hidden_states.device)
    up_out = torch.empty_like(gate_out)
    for local_expert in range(experts_per_rank):
        expert_mask = experts_grouped == local_expert
        if not bool(expert_mask.any()):
            continue
        expert_input = tokens_grouped[expert_mask].float()
        gate_out[expert_mask] = (
            expert_input @ w_gate_local[local_expert].T.float()).to(torch.bfloat16)
        up_out[expert_mask] = (
            expert_input @ w_up_local[local_expert].T.float()).to(torch.bfloat16)

    weighted_activation = (
        torch.nn.functional.silu(gate_out.float())
        * up_out.float()
        * weights_grouped.float()[:, None]
    ).to(torch.bfloat16)

    fc2_grouped = torch.empty(
        (total_recv, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    for local_expert in range(experts_per_rank):
        expert_mask = experts_grouped == local_expert
        if not bool(expert_mask.any()):
            continue
        fc2_grouped[expert_mask] = (
            weighted_activation[expert_mask].float()
            @ w2_local[local_expert].T.float()
        ).to(torch.bfloat16)

    # ---- Combine: undo receive grouping, reverse A2A, undo dispatch sort ----
    inverse_local_sort = torch.argsort(local_sort)
    fc2_recv_order = fc2_grouped[inverse_local_sort].contiguous()
    combined_rank_sorted = torch.empty(
        (total_send, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    dist.all_to_all_single(
        combined_rank_sorted,
        fc2_recv_order,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=ep_group,
    )

    inverse_rank_sort = torch.argsort(rank_sort)
    combined_routes = torch.zeros(
        (num_tokens * topk, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    combined_routes[valid_indices] = combined_rank_sorted[inverse_rank_sort]
    combined_routes = combined_routes.view(num_tokens, topk, hidden)
    output_fp32 = torch.zeros(
        (num_tokens, hidden), dtype=torch.float32, device=hidden_states.device)
    for route_slot in range(topk):
        output_fp32 += combined_routes[:, route_slot].float()
    return output_fp32.to(torch.bfloat16)


# ---------------------------------------------------------------------------
#  Data helpers
# ---------------------------------------------------------------------------

def make_w1(num_experts, hidden, inter, world_size, rank, dtype, device, seed):
    """Create this rank's deterministic BF16 expert slice without a global table.

    Every global expert is owned by exactly one EP rank, so constructing the
    full ``[num_experts, inter, hidden]`` table independently on every rank is
    unnecessary.  A rank-derived seed keeps ownership bugs observable while
    making the DSV4 correctness smoke practical.
    """
    epr = num_experts // world_size
    torch.manual_seed(seed + rank * 1_000_003)
    w1_local = torch.randn(
        (epr, inter, hidden), dtype=dtype, device=device
    ).mul_((1.0 / hidden) ** 0.5)
    return None, w1_local.contiguous()


def make_gate_up_weights(num_experts, hidden, ffn_dim, world_size, rank, dtype, device):
    """Create distinct rank-local gate/up weights from rank-independent tables."""
    _, w_gate_local = make_w1(
        num_experts, hidden, ffn_dim, world_size, rank, dtype, device, seed=142)
    _, w_up_local = make_w1(
        num_experts, hidden, ffn_dim, world_size, rank, dtype, device, seed=143)
    return w_gate_local, w_up_local


def make_down_weights(num_experts, hidden, ffn_dim, world_size, rank, dtype, device):
    """Create rank-local W2 with layout ``[E_local, hidden, ffn_dim]``."""
    _, w2_local = make_w1(
        num_experts, ffn_dim, hidden, world_size, rank, dtype, device, seed=144)
    return w2_local


def make_routing_weights(n, topk, device, seed):
    """Create FP32 per-route weights with distinct values across top-k slots."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(n, topk, generator=g, dtype=torch.float32)
    return torch.softmax(logits, dim=-1).to(device).contiguous()


def prepare_inputs(n, hidden, num_experts, topk, dtype, device, seed, drop_frac=0.0):
    """Per-rank hidden_states [n, hidden] and expert_index [n, topk] int32.

    ``drop_frac`` > 0 randomly marks that fraction of (token, topk) slots as dropped
    (expert_index = num_experts). Inputs are rank-distinct (seed varies with rank) for
    realistic asymmetric load.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    hs = (torch.randn(n, hidden, generator=g, dtype=torch.float32) * 0.5).to(dtype).to(device)
    logits = torch.rand(n, num_experts, generator=g).to(device)
    expert_index = torch.topk(logits, k=topk, dim=-1).indices.to(torch.int32)
    if drop_frac > 0.0:
        drop = (torch.rand(n, topk, generator=g) < drop_frac).to(device)
        expert_index = expert_index.masked_fill(drop, num_experts)
    return hs, expert_index


# ---------------------------------------------------------------------------
#  Exact grouped-layout comparison
# ---------------------------------------------------------------------------

def _compare_fc1_by_expert(kernel_out, kernel_exp, kernel_dispatch,
                           golden_out, golden_exp, golden_dispatch,
                           experts_per_rank, label, rank, device):
    """Compare the deterministic source-rank/expert/token grouped layout."""
    ok = True
    msgs = []
    if kernel_out.shape != golden_out.shape:
        ok = False
        msgs.append(f"output shape kernel={tuple(kernel_out.shape)} golden={tuple(golden_out.shape)}")
    if kernel_exp.shape != golden_exp.shape:
        ok = False
        msgs.append(f"expert shape kernel={tuple(kernel_exp.shape)} golden={tuple(golden_exp.shape)}")
    if ok:
        try:
            torch.testing.assert_close(
                kernel_exp, golden_exp, rtol=LAYOUT_RTOL, atol=LAYOUT_ATOL
            )
        except AssertionError as exc:
            ok = False
            msgs.append(f"expert layout mismatch: {str(exc).splitlines()[0]}")
        try:
            torch.testing.assert_close(
                kernel_out.float(),
                golden_out.float(),
                rtol=APPROX_RTOL,
                atol=APPROX_ATOL,
            )
        except AssertionError:
            ok = False
            diff = (kernel_out.float() - golden_out.float()).abs()
            zero_rows = int((kernel_out == 0).all(dim=1).sum().item())
            msgs.append(
                f"values max_diff={diff.max().item():.4f} mean={diff.mean().item():.4f} "
                f"zero_rows={zero_rows}/{kernel_out.shape[0]}"
            )
            for expert_idx in range(experts_per_rank):
                expert_mask = golden_exp == expert_idx
                if bool(expert_mask.any()):
                    expert_diff = diff[expert_mask]
                    if float(expert_diff.max().item()) > APPROX_ATOL:
                        msgs.append(
                            f"exp{expert_idx} max={expert_diff.max().item():.4f} "
                            f"mean={expert_diff.mean().item():.4f}"
                        )
        try:
            torch.testing.assert_close(
                kernel_dispatch.float(),
                golden_dispatch.float(),
                rtol=LAYOUT_RTOL,
                atol=LAYOUT_ATOL,
            )
        except AssertionError:
            ok = False
            diff = (kernel_dispatch.float() - golden_dispatch.float()).abs()
            msgs.append(f"dispatch max_diff={diff.max().item():.4f} mean={diff.mean().item():.4f}")
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = bool(flag.item())
    if not ok and msgs:
        print(f"[rank {rank}] {label}: {'; '.join(msgs)}", flush=True)
    if rank == 0:
        suffix = "" if ok else "  |  " + "; ".join(msgs)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}", flush=True)
    return ok


def _compare_weighted_stage(kernel_values, golden_values, label, rank, device):
    """Compare every layout and numerical boundary of the weighted stage."""
    ok = True
    msgs = []
    exact_keys = ("local_expert_ids", "routing_weight_recv", "dispatched_tokens")
    approximate_keys = ("fc1_out", "swiglu_out", "weighted_swiglu_out")

    for key in exact_keys + approximate_keys:
        actual = kernel_values[key]
        expected = golden_values[key]
        if actual.shape != expected.shape:
            ok = False
            msgs.append(
                f"{key} shape kernel={tuple(actual.shape)} golden={tuple(expected.shape)}")
            continue
        rtol = LAYOUT_RTOL if key in exact_keys else APPROX_RTOL
        atol = LAYOUT_ATOL if key in exact_keys else APPROX_ATOL
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError:
            ok = False
            if actual.numel() == 0:
                msgs.append(f"{key} empty-tensor mismatch")
            elif actual.is_floating_point():
                diff = (actual.float() - expected.float()).abs()
                msgs.append(
                    f"{key} max_diff={diff.max().item():.4f} mean={diff.mean().item():.4f}")
            else:
                mismatch = int((actual != expected).sum().item())
                msgs.append(f"{key} mismatches={mismatch}/{actual.numel()}")

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    ok = bool(flag.item())
    if not ok and msgs:
        print(f"[rank {rank}] {label}: {'; '.join(msgs)}", flush=True)
    if rank == 0:
        suffix = "" if ok else "  |  " + "; ".join(msgs)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}", flush=True)
    return ok


def _compare_full_output(actual, expected, label, rank, device, ep_group):
    """Compare a final ``[tokens, hidden]`` BF16 output on every EP rank."""
    ok = True
    msgs = []
    if actual.shape != expected.shape:
        ok = False
        msgs.append(f"shape kernel={tuple(actual.shape)} golden={tuple(expected.shape)}")
    if actual.dtype != torch.bfloat16:
        ok = False
        msgs.append(f"dtype kernel={actual.dtype} expected=torch.bfloat16")
    if actual.shape == expected.shape:
        actual_fp32 = actual.float()
        expected_fp32 = expected.float()
        if not bool(torch.isfinite(actual_fp32).all()):
            ok = False
            msgs.append("kernel output contains non-finite values")
        try:
            torch.testing.assert_close(
                actual_fp32,
                expected_fp32,
                rtol=OUTPUT_RTOL,
                atol=OUTPUT_ATOL,
            )
        except AssertionError:
            ok = False
            diff = (actual_fp32 - expected_fp32).abs()
            if diff.numel() == 0:
                msgs.append("empty output mismatch")
            else:
                msgs.append(
                    f"values max_diff={diff.max().item():.4f} "
                    f"mean={diff.mean().item():.4f}")

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    ok = bool(flag.item())
    if not ok and msgs:
        print(f"[rank {rank}] {label}: {'; '.join(msgs)}", flush=True)
    if rank == 0:
        suffix = "" if ok else "  |  " + "; ".join(msgs)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}", flush=True)
    return ok


# ---------------------------------------------------------------------------
#  Configs & worker
# ---------------------------------------------------------------------------

from config import FORWARD_SHAPES, FORWARD_SHAPES_KIMI


def run_one(layer, hs, exp_idx, w1l, num_experts, label, rank, device, dtype):
    """Run kernel dispatch+fc1 and compare against the independent golden."""
    routing_plan = layer.build_routing_plan(exp_idx)
    routing_weights = torch.ones_like(exp_idx, dtype=torch.float32)
    logical_w1 = w1l.transpose(-1, -2)
    gemm_output = torch.zeros(
        (routing_plan.num_received_routes, logical_w1.shape[1]),
        dtype=dtype,
        device=device,
    )
    dispatch_result = layer.dispatch_fc1(
        hs,
        exp_idx,
        routing_plan,
        w1l,
        fc1_output=gemm_output,
        routing_weights=routing_weights,
    )
    kernel_exp = torch.repeat_interleave(
        torch.arange(layer.experts_per_rank, dtype=torch.int32, device=device),
        routing_plan.received_routes_per_expert,
    )

    g_out, g_exp, g_dispatch = torch_dispatch_fc1_golden(
        hs, exp_idx, logical_w1, num_experts, dtype, device, layer.ep_group)

    return _compare_fc1_by_expert(
        dispatch_result.fc1_output,
        kernel_exp,
        dispatch_result.dispatched_tokens,
        g_out, g_exp, g_dispatch,
        layer.experts_per_rank, label, rank, device)


def run_weighted_one(
    layer,
    hs,
    exp_idx,
    routing_weights,
    w_gate_local,
    w_up_local,
    packed_w1,
    num_experts,
    label,
    rank,
    device,
    dtype,
):
    """Run the production weighted-stage entry and compare all boundaries."""
    weighted_swiglu_out, dispatch_result = layer.dispatch_fc1_weighted_swiglu(
        hs,
        exp_idx,
        routing_weights,
        packed_w1,
    )
    fc1_out = dispatch_result.fc1_output
    recv_weight = dispatch_result.received_routing_weights
    swiglu_out = weighted_swiglu_forward(
        fc1_out, torch.ones_like(recv_weight), layer.num_aicore_programs)

    kernel_exp = torch.repeat_interleave(
        torch.arange(layer.experts_per_rank, dtype=torch.int32, device=device),
        dispatch_result.routing_plan.received_routes_per_expert,
    )
    kernel_values = {
        "fc1_out": fc1_out,
        "swiglu_out": swiglu_out,
        "weighted_swiglu_out": weighted_swiglu_out,
        "routing_weight_recv": recv_weight,
        "local_expert_ids": kernel_exp,
        "dispatched_tokens": dispatch_result.dispatched_tokens,
    }
    golden_values = torch_dispatch_fc1_weighted_swiglu_golden(
        hs,
        exp_idx,
        routing_weights,
        w_gate_local,
        w_up_local,
        num_experts,
        dtype,
        device,
        layer.ep_group,
    )
    return _compare_weighted_stage(kernel_values, golden_values, label, rank, device)


def run_full_one(
    layer,
    hidden_states,
    expert_indices,
    routing_weights,
    w_gate_local,
    w_up_local,
    packed_w1,
    w2_local,
    num_experts,
    label,
    rank,
    device,
):
    """Run the production full-forward entry against the independent golden."""
    actual = layer.forward(
        hidden_states,
        expert_indices,
        packed_w1,
        w2_local,
        routing_weights,
    )
    expected = torch_moe_fwd_golden(
        hidden_states,
        routing_weights,
        expert_indices,
        w_gate_local,
        w_up_local,
        w2_local,
        num_experts,
        layer.ep_group,
    )
    return _compare_full_output(actual, expected, label, rank, device, layer.ep_group)


def run_test(rank, world_size):
    # ---- init aclshmem ----
    init_aclshmem(rank, world_size, g_ash_size)

    device = f"npu:{rank}"
    dtype = torch.bfloat16
    all_passed = True

    only_config = os.environ.get("MOE_FUSED_TEST_CONFIG")
    base_configs = list(FORWARD_SHAPES)
    if os.environ.get("MOE_KIMI") == "1":
        base_configs.extend(FORWARD_SHAPES_KIMI)
    configs = [config for config in base_configs if only_config in (None, config.label)]

    for config in configs:
        label = config.label
        hidden = config.hidden
        inter = config.ffn
        topk = config.topk
        drop_frac = config.drop_frac
        n_per_rank = config.resolved_tokens_per_rank(world_size)
        num_experts = config.resolved_num_experts(world_size)
        if num_experts % world_size != 0:
            raise ValueError("expert count must be divisible by world_size")
        tiling_overrides = {}
        for env_name, parameter_name in (
            (
                "MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M",
                "dispatch_fc1_block_size_m",
            ),
            ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_N", "fc1_gemm_block_size_n"),
            ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_K", "fc1_gemm_block_size_k"),
            (
                "MOE_FUSED_FC2_COMBINE_BLOCK_SIZE_M",
                "fc2_combine_block_size_m",
            ),
            ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_N", "fc2_gemm_block_size_n"),
            ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_K", "fc2_gemm_block_size_k"),
        ):
            if env_name in os.environ:
                tiling_overrides[parameter_name] = int(os.environ[env_name])
        forward_config = MoEForwardConfig(
            num_aicore_programs=int(
                os.environ.get("MOE_FUSED_NUM_AICORE_PROGRAMS", "24")
            ),
            receive_capacity_factor=float(world_size),
            **tiling_overrides,
        )
        optimized_op = FusedMoEForward(
            None,
            max_tokens_per_rank=n_per_rank,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=forward_config,
        )
        try:
            # Build only rank-local expert tables. The standalone FC1 check can
            # reuse the gate table; allocating a fourth DSV4-sized table adds no
            # independent coverage.
            w_gate_local, w_up_local = make_gate_up_weights(
                num_experts, hidden, inter, world_size, rank, dtype, device)
            # Pack once and reuse it across every candidate call.  Repacking a
            # DSV4 W1 for each sub-check creates a 15.75 GiB transient and can
            # exceed 910B1 HBM even though the production model stores one W1.
            packed_w1 = pack_gate_up_weights(w_gate_local, w_up_local)
            w1l = packed_w1
            w2_local = make_down_weights(
                num_experts, hidden, inter, world_size, rank, dtype, device)
            # inputs are rank-distinct (seed varies with rank) for asymmetric load
            hs, exp_idx = prepare_inputs(n_per_rank, hidden, num_experts, topk, dtype, device,
                                         seed=43 + rank * 1000, drop_frac=drop_frac)
            if label == "S-drop":
                # Negative ids are dropped by the same [0, num_experts) rule.
                exp_idx[0, 0] = -1
            routing_weights = make_routing_weights(
                n_per_rank, topk, device, seed=44 + rank * 1000)
            dist.barrier()

            ok = run_one(
                optimized_op, hs, exp_idx, w1l, num_experts,
                f"{label}-overlap", rank, device, dtype)
            all_passed = all_passed and ok

            ok_weighted = run_weighted_one(
                optimized_op,
                hs,
                exp_idx,
                routing_weights,
                w_gate_local,
                w_up_local,
                packed_w1,
                num_experts,
                f"{label}-overlap-weighted-swiglu",
                rank,
                device,
                dtype,
            )
            all_passed = all_passed and ok_weighted

            # Full-forward coverage stays on the small S shapes.  S-drop also
            # includes one negative id, so both invalid-id classes reach the
            # complete dispatch/FC1/FC2/combine path.
            if label in ("S", "S-drop", "DSV4-smoke"):
                ok_full = run_full_one(
                    optimized_op,
                    hs,
                    exp_idx,
                    routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    w2_local,
                    num_experts,
                    f"{label}-full-forward",
                    rank,
                    device,
                )
                all_passed = all_passed and ok_full

            # Every token goes to one destination/expert.  Other ranks are
            # send-only and must still enter the fused kernel's tail barrier.
            skew_idx = torch.full_like(exp_idx, world_size)
            ok_skew = run_one(
                optimized_op, hs, skew_idx, w1l, num_experts,
                f"{label}-overlap-skew", rank, device, dtype)
            all_passed = all_passed and ok_skew

            if label == "S":
                edge_routing_weights = torch.zeros_like(routing_weights)
                edge_routing_weights[:, -1] = 1.0
                ok_weighted_skew = run_weighted_one(
                    optimized_op,
                    hs,
                    skew_idx,
                    edge_routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    num_experts,
                    f"{label}-overlap-weighted-swiglu-skew",
                    rank,
                    device,
                    dtype,
                )
                all_passed = all_passed and ok_weighted_skew
                ok_full_skew = run_full_one(
                    optimized_op,
                    hs,
                    skew_idx,
                    edge_routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    w2_local,
                    num_experts,
                    f"{label}-full-forward-zero-receive",
                    rank,
                    device,
                )
                all_passed = all_passed and ok_full_skew

                # Exercise M_local == 0 and zero dispatch tasks on every rank.
                all_drop_idx = torch.full_like(exp_idx, num_experts)
                ok_all_drop = run_one(
                    optimized_op, hs, all_drop_idx, w1l, num_experts,
                    f"{label}-overlap-all-drop", rank, device, dtype)
                all_passed = all_passed and ok_all_drop
                ok_weighted_all_drop = run_weighted_one(
                    optimized_op,
                    hs,
                    all_drop_idx,
                    routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    num_experts,
                    f"{label}-overlap-weighted-swiglu-all-drop",
                    rank,
                    device,
                    dtype,
                )
                all_passed = all_passed and ok_weighted_all_drop
                ok_full_all_drop = run_full_one(
                    optimized_op,
                    hs,
                    all_drop_idx,
                    routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    w2_local,
                    num_experts,
                    f"{label}-full-forward-all-drop",
                    rank,
                    device,
                )
                all_passed = all_passed and ok_full_all_drop

                # Reuse the same slots with a later epoch; stale signal values
                # must not satisfy the next overlap launch.
                ok_repeat = run_one(
                    optimized_op, hs, exp_idx, w1l, num_experts,
                    f"{label}-overlap-epoch-reuse", rank, device, dtype)
                all_passed = all_passed and ok_repeat
                ok_full_repeat = run_full_one(
                    optimized_op,
                    hs,
                    exp_idx,
                    routing_weights,
                    w_gate_local,
                    w_up_local,
                    packed_w1,
                    w2_local,
                    num_experts,
                    f"{label}-full-forward-reuse-after-all-drop",
                    rank,
                    device,
                )
                all_passed = all_passed and ok_full_repeat

        finally:
            optimized_op.finalize()

    _ = ash.aclshmem_finalize()

    final_flag = torch.tensor([1 if all_passed else 0], dtype=torch.int32, device=device)
    dist.all_reduce(final_flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("ALL PASSED" if bool(final_flag.item()) else "SOME TESTS FAILED", flush=True)
    if not bool(final_flag.item()):
        raise AssertionError("Ascend post-routing MoE golden check failed.")


# ---------------------------------------------------------------------------
#  Pytest
# ---------------------------------------------------------------------------

def test_public_api_has_only_bf16_and_current_combine_arguments():
    """The public API exposes neither dtype nor dead legacy combine knobs."""
    constructor_parameters = inspect.signature(FusedMoEForward).parameters
    assert "dtype" not in constructor_parameters
    assert "config" in constructor_parameters

    combine_parameters = inspect.signature(FusedMoEForward.fc2_combine).parameters
    assert "gate_input" not in combine_parameters
    assert "gemm_BLOCK_SIZE_N" not in combine_parameters


def test_config_keeps_fixed_fc1_schedule_and_best_defaults():
    config = MoEForwardConfig()
    assert config.num_aicore_programs == 24
    assert config.dispatch_fc1_block_size_m == 128
    assert config.fc1_gemm_block_size_n == 256
    assert config.fc1_gemm_block_size_k == 128
    assert config.fc2_combine_block_size_m == 128
    assert config.fc2_gemm_block_size_n == 256
    assert config.fc2_gemm_block_size_k == 128
    assert config.dispatch_fc1_schedule == "allcore_expert_n_tile"
    assert config.resolved_receive_capacity_factor(8) == 8.0

    parameters = inspect.signature(MoEForwardConfig).parameters
    assert set(parameters) == {
        "num_aicore_programs",
        "receive_capacity_factor",
        "dispatch_fc1_block_size_m",
        "fc1_gemm_block_size_n",
        "fc1_gemm_block_size_k",
        "fc2_combine_block_size_m",
        "fc2_gemm_block_size_n",
        "fc2_gemm_block_size_k",
        "dispatch_fc1_schedule",
        "activation",
        "situ_beta",
        "situ_linear_beta",
    }

    assert MoEForwardConfig(
        dispatch_fc1_schedule="allcore_expert_n_tile"
    ).dispatch_fc1_schedule == "allcore_expert_n_tile"
    with pytest.raises(ValueError, match="dispatch_fc1_schedule must be one of"):
        MoEForwardConfig(dispatch_fc1_schedule="unsupported")

    # Target-only activation extension remains available on top of the
    # source-pruned forward schedule/config surface.
    assert config.activation == "swiglu"
    situglu = MoEForwardConfig(
        activation="situglu", situ_beta=2.0, situ_linear_beta=1.5
    )
    assert situglu.activation == "situglu"
    assert situglu.situ_beta == 2.0
    assert situglu.situ_linear_beta == 1.5
    with pytest.raises(ValueError, match="activation must be one of"):
        MoEForwardConfig(activation="relu")
    with pytest.raises(ValueError, match="situ_beta must be positive"):
        MoEForwardConfig(situ_beta=0.0)


def test_direct_pull_workspace_is_sized_by_sent_routes_only():
    """The local FC2 staging buffer does not reserve receive-capacity rows."""
    op = FusedMoEForward.__new__(FusedMoEForward)
    torch.nn.Module.__init__(op)
    op.max_tokens_per_rank = 8
    op.hidden_size = 4
    op.top_k = 2
    op.world_size = 8
    op.experts_per_rank = 112
    op.activation_dtype = torch.bfloat16
    op.config = MoEForwardConfig()
    op.context = SimpleNamespace(peer_mem=torch.empty(4096, dtype=torch.bfloat16))
    op._combine_fc2_buf = None

    op._ensure_combine_buffers()

    assert op._combine_fc2_buf.shape == (16, 4)
    assert op._route_to_send.shape == (16,)
    assert op._max_pull_tile_slots == 1 + 8 * 112
    assert op._pull_tile_rank.shape == (1 + 8 * 112,)


def test_dispatch_kernel_keeps_default_only_pipeline():
    launch_source = inspect.getsource(FusedMoEForward.dispatch_fc1)
    kernel_source = inspect.getsource(dispatch_fc1_module._kernel_dispatch_fc1.fn)
    consumer_source = inspect.getsource(
        dispatch_fc1_module._triton_grouped_gemm_expert_n_merged_tiles_wait.fn
    )
    tile_dispatch_source = inspect.getsource(
        dispatch_fc1_module._dispatch_one_source_tile_task.fn
    )

    assert "_triton_grouped_gemm_expert_n_merged_tiles_wait(" in kernel_source
    assert "_dispatch_count_derived_source_tiles(" in kernel_source
    assert "if sub_vec_id() == 0:" in kernel_source
    assert "FINAL_BARRIER" in kernel_source
    assert "DIRECT_EXPERT_DISPATCH" not in kernel_source
    assert "EXPERT_N_TILE_CONSUMER" not in kernel_source
    assert "MN_TILE_FC1" not in kernel_source
    assert "send_staging_ptr" not in kernel_source
    assert "routing_staging_ptr" not in kernel_source
    assert "source_id" in consumer_source
    assert "overlap_start" in consumer_source
    assert "self.dispatch_fc1_schedule" not in launch_source
    assert "NUM_PROGRAM_CORES=self.num_aicore_programs" in launch_source
    assert "DIRECT_EXPERT_DISPATCH" not in launch_source
    assert "EXPERT_N_TILE_CONSUMER" not in launch_source
    assert "MN_TILE_FC1" not in launch_source
    assert "HAS_ROUTING_WEIGHT" not in launch_source
    assert "HAS_ROUTING_WEIGHT" not in kernel_source
    assert "hidden * 2" in tile_dispatch_source


def test_routing_metadata_keeps_width_specific_910b1_lowering():
    histogram_source = inspect.getsource(
        routing_metadata_module._kernel_build_routing_metadata.fn
    )
    lower_bound_source = inspect.getsource(
        routing_metadata_module._kernel_build_routing_metadata_lower_bound.fn
    )
    metadata_helper_source = inspect.getsource(
        routing_metadata_module._publish_counts_and_build_metadata.fn
    )
    launch_source = inspect.getsource(routing_metadata_module.build_routing_plan)

    assert "tl.histogram(route_keys, NUM_BINS_PAD)" in histogram_source
    assert "left_safe_mid = tl.where(left_active, left_mid, 0)" in lower_bound_source
    assert "if context.metadata_num_bins > 512" in launch_source
    assert "if sub_vec_id() == 0:" in histogram_source
    assert "if sub_vec_id() == 0:" in metadata_helper_source
    assert "NUM_BINS_PAD * 4" in histogram_source
    assert "NUM_BINS_PAD * 4" in metadata_helper_source


def test_fc2_launch_uses_persistent_pull_and_dynamic_reduce_grid(monkeypatch):
    fc2_launches = []
    transport_launches = []
    reduce_launches = []
    fc2_kernel_source = inspect.getsource(
        fc2_combine_module._kernel_fc2_expert_n_persistent.fn
    )
    transport_kernel_source = inspect.getsource(
        fc2_combine_module._kernel_direct_pull_transport.fn
    )
    reduce_kernel_source = inspect.getsource(
        fc2_combine_module._kernel_local_topk_reduce.fn
    )
    launch_source = inspect.getsource(fc2_combine_module.launch_fc2_combine)

    class FakeKernel:
        def __init__(self, sink):
            self.sink = sink

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.sink.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        fc2_combine_module,
        "_kernel_fc2_expert_n_persistent",
        FakeKernel(fc2_launches),
    )
    monkeypatch.setattr(
        fc2_combine_module,
        "_kernel_direct_pull_transport",
        FakeKernel(transport_launches),
    )
    monkeypatch.setattr(
        fc2_combine_module,
        "_kernel_local_topk_reduce",
        FakeKernel(reduce_launches),
    )

    experts = 2
    rows = 5
    reduction = 16
    hidden = 16
    tokens = 2
    topk = 1
    received_routes_per_expert = torch.tensor([2, 3], dtype=torch.int32)
    received_expert_offsets = torch.tensor([0, 2, 5], dtype=torch.int32)
    route_to_send = torch.arange(tokens * topk, dtype=torch.int32)
    pull_metadata = [torch.zeros(4, dtype=torch.int32) for _ in range(4)]
    peer_mem = torch.empty(rows * hidden, dtype=torch.bfloat16)
    fc2_buf = torch.empty((rows, hidden), dtype=torch.bfloat16)
    num_program_cores = 7

    output = fc2_combine_module.launch_fc2_combine(
        torch.zeros((rows, reduction), dtype=torch.bfloat16),
        torch.zeros((experts, hidden, reduction), dtype=torch.bfloat16),
        fc2_buf,
        peer_mem,
        route_to_send,
        torch.empty((tokens, hidden), dtype=torch.bfloat16),
        received_routes_per_expert,
        received_expert_offsets,
        *pull_metadata,
        num_pull_slots=4,
        num_send=tokens * topk,
        topk=topk,
        num_program_cores=num_program_cores,
        block_m=16,
        block_n=16,
        block_k=16,
        world_size=1,
    )

    assert output.shape == (tokens, hidden)
    assert len(fc2_launches) == 1
    grid, args, kwargs = fc2_launches[0]
    assert grid == (num_program_cores, 1, 1)
    assert args[2] is peer_mem
    assert kwargs["EXPERTS_PER_RANK"] == experts
    assert len(transport_launches) == 1
    transport_grid, transport_args, transport_kwargs = transport_launches[0]
    assert transport_grid == (num_program_cores, 1, 1)
    assert transport_args[0] is fc2_buf
    assert transport_args[1] is peer_mem
    assert transport_kwargs["WORLD_SIZE"] == 1
    assert len(reduce_launches) == 1
    reduce_grid, reduce_args, _ = reduce_launches[0]
    assert reduce_grid == (num_program_cores * 2, 1, 1)
    assert reduce_args[0] is fc2_buf
    assert reduce_args[1] is route_to_send
    assert reduce_args[2] is output
    assert "_fc2_gemm_one_mn_tile(" in fc2_kernel_source
    assert transport_kernel_source.count("libshmem_device.barrier_all_vec()") == 2
    assert "libshmem_device.barrier_all()" not in transport_kernel_source
    assert "row_count * N * 2" in transport_kernel_source
    assert "if sub_vec_id() == 0:" in transport_kernel_source
    assert "libshmem_device.getmem(" in transport_kernel_source
    assert "peer_rank == peer_owner" in transport_kernel_source
    assert "libshmem_device.getmem(" not in reduce_kernel_source
    assert "for token_id in range(pid, batch_size, ncore)" in reduce_kernel_source
    assert "reduce_sub_id = sub_vec_id" not in reduce_kernel_source
    assert "num_program_cores * 2" in launch_source


def test_fc2_metadata_builds_only_direct_pull_descriptors(monkeypatch):
    metadata_launches = []
    metadata_source = inspect.getsource(
        fc2_combine_module._prepare_fc2_combine_metadata_kernel.fn
    )
    production_source = inspect.getsource(
        FusedMoEForward._prepare_combine_metadata
    )

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                metadata_launches.append((grid, args, kwargs))

            return launch

    monkeypatch.setattr(
        fc2_combine_module,
        "_prepare_fc2_combine_metadata_kernel",
        FakeKernel(),
    )

    counts_mem = torch.tensor([2, 3], dtype=torch.int32)
    send_starts = torch.tensor([0, 2], dtype=torch.int32)
    receive_starts = torch.tensor([0, 2], dtype=torch.int32)
    pull_metadata = [torch.zeros(4, dtype=torch.int32) for _ in range(4)]

    fc2_combine_module.prepare_fc2_combine_metadata(
        counts_mem,
        send_starts,
        receive_starts,
        *pull_metadata,
        num_pull_slots=4,
        local_rank=0,
        world_size=1,
        experts_per_rank=2,
        num_bins_pad=2,
        block_m=16,
    )

    assert len(metadata_launches) == 1
    grid, _, kwargs = metadata_launches[0]
    assert grid == (1, )
    assert kwargs["BLOCK_M"] == 16
    assert "pull_tile_rank_ptr" in metadata_source
    assert "self._pull_tile_rank" in production_source


def test_pack_gate_up_weights_returns_contiguous_kn_layout():
    gate = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    up = gate + 100
    packed = pack_gate_up_weights(gate, up)

    assert packed.shape == (2, 4, 6)
    assert packed.is_contiguous()
    torch.testing.assert_close(packed[:, :, :3], gate.transpose(1, 2))
    torch.testing.assert_close(packed[:, :, 3:], up.transpose(1, 2))


def _situglu_torch_ref(fc1, rw, activation, beta, lin_beta):
    """Independent FP32 reference for the target activation extension."""
    ffn_dim = fc1.shape[-1] // 2
    gate = fc1[..., :ffn_dim].float()
    up = fc1[..., ffn_dim:].float()
    if activation == "swiglu":
        activated = torch.nn.functional.silu(gate) * up
    else:
        activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if lin_beta is not None:
            up = lin_beta * torch.tanh(up / lin_beta)
        activated = activated * up
    return (activated * rw.float().unsqueeze(-1)).to(fc1.dtype)


@pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="SiTU-GLU kernel correctness requires an NPU device",
)
def test_weighted_swiglu_kernel_supports_swiglu_and_situglu():
    """The target-only SiTU-GLU switch remains correct after source sync."""
    device = "npu:0"
    torch.npu.set_device(0)
    torch.manual_seed(0)
    rows, ffn_dim = 64, 256
    fc1 = (
        (torch.randn(rows, 2 * ffn_dim, dtype=torch.float32) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    rw = (torch.rand(rows, dtype=torch.float32, device=device) + 0.1).contiguous()
    cases = (
        ("swiglu", 1.0, None),
        ("situglu", 1.0, None),
        ("situglu", 1.5, None),
        ("situglu", 2.0, 1.0),
    )
    for activation, beta, linear_beta in cases:
        actual = weighted_swiglu_forward(
            fc1,
            rw,
            24,
            activation=activation,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        expected = _situglu_torch_ref(fc1, rw, activation, beta, linear_beta)
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )

    empty = weighted_swiglu_forward(
        fc1[:0], rw[:0], 24, activation="situglu", situ_beta=1.0
    )
    assert empty.shape == (0, ffn_dim)
    assert empty.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="activation must be 'swiglu' or 'situglu'"):
        weighted_swiglu_forward(fc1, rw, 24, activation="relu")


def test_make_down_weights_returns_contiguous_nk_layout():
    down = make_down_weights(
        num_experts=4,
        hidden=8,
        ffn_dim=6,
        world_size=2,
        rank=0,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert down.shape == (2, 8, 6)
    assert down.is_contiguous()
    assert down.stride() == (48, 6, 1)
    activation = torch.randn(3, 6, dtype=torch.float32)
    assert (activation @ down[0].T.float()).shape == (3, 8)


@pytest.mark.dist
def test_forward_2ranks(dist_test):
    dist_test(run_test, world_size=2)


@pytest.mark.dist
def test_forward_4ranks(dist_test):
    dist_test(run_test, world_size=4)


@pytest.mark.dist
def test_forward_8ranks(dist_test):
    dist_test(run_test, world_size=8)
