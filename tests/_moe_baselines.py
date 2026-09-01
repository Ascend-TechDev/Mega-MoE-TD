# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Single test-side correctness/baseline hub.

The forward functions are independent per-expert Torch/HCCL correctness
oracles; the hand-written backward implementation below is the only test-side
backward reference.  The production differentiable forward it reuses lives in
``mega_moe.ops._torch_forward``.  The grouped performance baseline is kept in
``benchmark.layer._grouped_forward_baseline`` and is intentionally not copied
here.  This module contains no pytest entry points.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from mega_moe.ops._torch_forward import (
    grouped_matmul,
    grouped_transposed_matmul,
    moe_forward,
)
from tests._numeric import cmp_grad
from tests._numeric import (
    APPROX_ATOL,
    APPROX_RTOL,
    LAYOUT_ATOL,
    LAYOUT_RTOL,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
)


__all__ = [
    "backward_torch_baseline",
    "moe_backward_torch",
    "build_backward_saved",
    "compare_backward_gradients",
    "make_backward_inputs",
    "make_down_weights",
    "make_gate_up_weights",
    "make_routing_weights",
    "prepare_inputs",
    "run_full_one",
    "run_one",
    "torch_dispatch_fc1_golden",
    "torch_moe_fwd_golden",
]


def make_backward_inputs(
    ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42
):
    """Build the single shared backward input layout for tests and benchmarks."""
    pe = dist.get_rank(ep_group)
    world_size = dist.get_world_size(ep_group)
    epr = num_experts // world_size
    dtype = torch.bfloat16
    device = f"npu:{pe}"
    torch.manual_seed(seed + pe * 1000)
    hs = torch.randn(ntokens, hidden_dim, dtype=dtype, device=device)
    gw = torch.randn(num_experts, hidden_dim, dtype=dtype, device=device)
    fc1_1 = torch.randn(epr, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc1_2 = torch.randn(epr, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc2 = torch.randn(epr, hidden_dim, ffn_dim, dtype=dtype, device=device)
    dist.broadcast(gw, src=0, group=ep_group)
    logits = hs.float() @ gw.float().T
    rw = torch.softmax(logits, dim=-1).to(dtype)
    topk_w, topk_idx = torch.topk(rw, topk, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    dy = torch.randn_like(hs)
    return hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device


def build_backward_saved(
    ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42
):
    """Build the one canonical saved-state layout used by bwd tests/benchmarks."""
    hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device = make_backward_inputs(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed
    )
    with torch.no_grad():
        _, saved = moe_forward(
            hs,
            topk_w,
            topk_idx,
            fc1_1,
            fc1_2,
            fc2,
            ep_group,
            topk,
            return_saved=True,
        )
    return saved, dy, dtype, device


def combine_bwd_a2a(dy, saved):
    """Step 1a: backward of (topk-sum + reverse-A2A + local-sort).
    dy [B,H] -> grad_fc2_out_sorted [M,H] on the expert rank."""
    B = saved["batch_size"]; H = saved["hidden_dim"]; topk = saved["topk"]
    dtype = dy.dtype; device = dy.device
    ep_group = saved["ep_group"]
    # un-reduce: forward output = combined_full.view(B,topk,H).sum(1) => each topk copy gets dy
    grad_combined_full = dy.repeat_interleave(topk, dim=0)               # [B*topk, H]
    # forward: combined_full = combined_out_flat[inv_sort]  =>  grad_combined_out_flat = grad_combined_full[sort_idxs]
    grad_combined_out_flat = grad_combined_full[saved["sort_idxs"]]       # [total_send, H]
    # backward of reverse-A2A = forward dispatch-A2A (swap split roles)
    grad_fc2_out_unsorted = torch.empty((saved["total_recv"], H), dtype=dtype, device=device)
    dist.all_to_all_single(grad_fc2_out_unsorted, grad_combined_out_flat,
                           output_split_sizes=saved["splits_recv_list"],
                           input_split_sizes=saved["splits_send_list"], group=ep_group)
    # forward: fc2_out_unsorted = fc2_out[inv_local]; recv_hidden_sorted = arrival[local_sort]
    #   => grad_fc2_out_sorted = grad_fc2_out_unsorted[local_sort_idxs]
    grad_fc2_out_sorted = grad_fc2_out_unsorted[saved["local_sort_idxs"]]
    return grad_fc2_out_sorted


def fc2_input_grad(grad_fc2_out_sorted, saved):
    """Step 1b: grad_swiglu [M,ffn] = grad_fc2_out_sorted @ fc2[e]  (input-grad grouped gemm)."""
    return grouped_matmul(grad_fc2_out_sorted, saved["fc2"], saved["expert_counts"], transpose=False)


def swiglu_bwd(grad_swiglu, saved):
    """Step 2: SwiGLU/SiTU-GLU backward (activation from ``saved``).
    fwd: swiglu_out = act(gate) * v(up) ; swiglu_out_weighted = swiglu_out * scale (scale=recv_weights_sorted)
      SwiGLU:  act = silu(gate), v = up
      SiTU:    act = beta*tanh(gate/beta)*sigmoid(gate), v = linear_beta*tanh(up/linear_beta) (or up)
      dGate = grad_swiglu * act'(gate) * v * scale
      dUp   = grad_swiglu * act(gate)  * v' * scale
      dScale = sum(act(gate)*v*grad_swiglu)  (per row)  => grad of recv_weights_sorted
    Returns grad_fc1_output [M,2*ffn] (=cat[dGate,dUp]), grad_gate [M]."""
    gate = saved["gate"].float()
    up = saved["up"].float()
    scale = saved["recv_weights_sorted"].float().unsqueeze(-1)
    g = grad_swiglu.float()
    if saved.get("activation", "swiglu") == "situglu":
        beta = float(saved.get("situ_beta") or 1.0)
        linear_beta = saved.get("situ_linear_beta")
        t = torch.tanh(gate / beta)
        sigmoid_g = torch.sigmoid(gate)
        act_g = beta * t * sigmoid_g
        act_prime = (1 - t * t) * sigmoid_g + beta * t * sigmoid_g * (1 - sigmoid_g)
        if linear_beta is not None:
            linear_beta = float(linear_beta)
            tu = torch.tanh(up / linear_beta)
            v = linear_beta * tu
            v_prime = 1 - tu * tu
        else:
            v = up
            v_prime = torch.ones_like(up)
    else:
        sigmoid_g = torch.sigmoid(gate)
        act_g = gate * sigmoid_g
        act_prime = act_g * (1 - sigmoid_g) + sigmoid_g       # d/dgate silu(gate)
        v = up
        v_prime = torch.ones_like(up)
    dGate = g * act_prime * v * scale
    dUp = g * act_g * v_prime * scale
    grad_fc1_output = torch.cat([dGate, dUp], dim=-1).to(grad_swiglu.dtype)
    grad_gate = (act_g * v * g).sum(dim=-1).to(grad_swiglu.dtype)   # dscale, [M]
    return grad_fc1_output, grad_gate


def fc2_weight_grad(grad_fc2_out_sorted, saved):
    """Step 3: grad_fc2 [E,H,ffn] = grad_fc2_out_sorted^T @ swiglu_out_weighted."""
    return grouped_transposed_matmul(grad_fc2_out_sorted, saved["swiglu_out_weighted"],
                                     saved["expert_counts"])


def fc1_input_grad(grad_fc1_output, saved):
    """Step 4a: grad_recv_hidden_sorted [M,H] = grad_fc1_output @ fc1_combined[e]."""
    return grouped_matmul(grad_fc1_output, saved["fc1_combined"], saved["expert_counts"], transpose=False)


def dispatch_bwd(grad_recv_hidden_sorted, grad_gate, saved):
    """Step 4b: reverse-A2A(expert->home) + topk sum => grad_hidden [B,H];
    grad_gate -> reverse-A2A + gather => grad_routing_weights [B,topk]."""
    B = saved["batch_size"]; H = saved["hidden_dim"]; topk = saved["topk"]
    dtype = grad_recv_hidden_sorted.dtype; device = grad_recv_hidden_sorted.device
    ep_group = saved["ep_group"]
    # unsort to arrival order, then reverse-A2A (expert->home)
    grad_recv_hidden_unsorted = grad_recv_hidden_sorted[saved["inv_local"]]   # [total_recv, H]
    grad_combined_out_flat = torch.empty((saved["total_send"], H), dtype=dtype, device=device)
    dist.all_to_all_single(grad_combined_out_flat, grad_recv_hidden_unsorted,
                           output_split_sizes=saved["splits_send_list"],
                           input_split_sizes=saved["splits_recv_list"], group=ep_group)
    grad_combined_full = grad_combined_out_flat[saved["inv_sort"]]            # [B*topk, H]
    grad_hidden = grad_combined_full.view(B, topk, H).sum(dim=1)              # plain sum (fwd was repeat_interleave)

    # gate (dscale) is 1-to-1: reverse-A2A back to home, then gather by inv_sort (no sum)
    grad_gate_unsorted = grad_gate[saved["inv_local"]]                        # [total_recv]
    grad_sorted_weights = torch.empty((saved["total_send"],), dtype=dtype, device=device)
    dist.all_to_all_single(grad_sorted_weights, grad_gate_unsorted,
                           output_split_sizes=saved["splits_send_list"],
                           input_split_sizes=saved["splits_recv_list"], group=ep_group)
    grad_routing_flat = grad_sorted_weights[saved["inv_sort"]]                # [B*topk]
    grad_routing_weights = grad_routing_flat.view(B, topk)
    return grad_hidden, grad_routing_weights


def fc1_weight_grad(grad_fc1_output, saved):
    """Step 5: grad_fc1 [E,2*ffn,H] = grad_fc1_output^T @ recv_hidden_sorted; chunk into fc1_1/fc1_2."""
    grad_fc1 = grouped_transposed_matmul(grad_fc1_output, saved["recv_hidden_sorted"],
                                         saved["expert_counts"])
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return grad_fc1_1, grad_fc1_2, grad_fc1


def moe_backward_torch(saved, dy):
    """Run all 5 hand-written backward mega-ops. Returns a dict of grads."""
    dy = dy.to(saved["output"].dtype)
    grad_fc2_out_sorted = combine_bwd_a2a(dy, saved)                 # step 1a
    grad_swiglu = fc2_input_grad(grad_fc2_out_sorted, saved)         # step 1b
    grad_fc1_output, grad_gate = swiglu_bwd(grad_swiglu, saved)      # step 2
    grad_fc2 = fc2_weight_grad(grad_fc2_out_sorted, saved)           # step 3
    grad_recv_hidden_sorted = fc1_input_grad(grad_fc1_output, saved) # step 4a
    grad_hidden, grad_routing_weights = dispatch_bwd(                # step 4b
        grad_recv_hidden_sorted, grad_gate, saved)
    grad_fc1_1, grad_fc1_2, grad_fc1 = fc1_weight_grad(grad_fc1_output, saved)  # step 5
    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing_weights,
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        # intermediates exposed for per-mega-op triton correctness checks
        grad_fc2_out_sorted=grad_fc2_out_sorted, grad_swiglu=grad_swiglu,
        grad_fc1_output=grad_fc1_output, grad_gate=grad_gate,
        grad_recv_hidden_sorted=grad_recv_hidden_sorted, grad_fc1=grad_fc1,
    )


def backward_torch_baseline(saved, dy):
    """Run the single hand-written backward correctness baseline."""
    with torch.no_grad():
        return moe_backward_torch(saved, dy)


def compare_backward_gradients(triton_result, torch_result):
    """Return the five canonical gradient checks without printing or timing."""
    checks = (
        "grad_hidden",
        "grad_routing_weights",
        "grad_fc1_1",
        "grad_fc1_2",
        "grad_fc2",
    )
    rows = []
    all_ok = True
    for name in checks:
        ok, max_abs, rel, nbad = cmp_grad(name, triton_result[name], torch_result[name])
        rows.append({"name": name, "ok": bool(ok), "max_abs": max_abs, "relative": rel, "mismatches": nbad})
        all_ok = all_ok and ok
    return bool(all_ok), rows


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
    sort_idx = torch.argsort(dest.to(torch.float32), stable=True)
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
    recv_order = torch.argsort(local_exp_recv.to(torch.float32), stable=True)
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

    rank_sort = torch.argsort(dest_ranks.to(torch.float32), stable=True)
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
    local_sort = torch.argsort(local_experts.to(torch.float32), stable=True)
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
    inverse_local_sort = torch.argsort(local_sort.to(torch.float32))
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

    inverse_rank_sort = torch.argsort(rank_sort.to(torch.float32))
    combined_routes = torch.zeros(
        (num_tokens * topk, hidden), dtype=torch.bfloat16, device=hidden_states.device)
    combined_routes[valid_indices] = combined_rank_sorted[inverse_rank_sort]
    combined_routes = combined_routes.view(num_tokens, topk, hidden)
    output_fp32 = torch.zeros(
        (num_tokens, hidden), dtype=torch.float32, device=hidden_states.device)
    for route_slot in range(topk):
        output_fp32 += combined_routes[:, route_slot].float()
    return output_fp32.to(torch.bfloat16)

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
