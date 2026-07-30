# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  moe_backward_golden.py
#
#  Pure torch + hccl MoE backward golden reference.
#
#  Mirrors the GPU golden (test/nvidia/test_ep_moe_fused.py: torch_moe_fwd) and
#  the Ascend 06 tutorial (06-ascend-moe-grouped-gemm-combine.py) forward, then
#  adds a HAND-WRITTEN backward split into the same 5 mega-ops as the GPU
#  TritonDistFusedEpMoeFunction.backward:
#
#    1. combine-bwd-A2A (dispatch direction) + fc2 input-grad
#    2. SwiGLU backward
#    3. fc2 weight-grad  (transposed grouped gemm)
#    4. fc1 input-grad + combine reverse-A2A + gate-grad
#    5. fc1 weight-grad  (transposed grouped gemm)
#
#  Forward invariants (routing weights applied in SwiGLU; combine reduce is a
#  PLAIN sum over topk — no token dropping, matching 06):
#    dispatch(home->expert) -> sort by local expert -> fc1(gate=H@fc1_1^T, up=H@fc1_2^T)
#    -> SwiGLU(silu(gate)*up * scale) -> fc2(swiglu@fc2^T)
#    -> reverse-A2A(expert->home) -> view(B,topk,H).sum(1)
#
#  Correctness ground truth: a hand-written backward is cross-checked against
#  autograd (output.backward(dy)) on every grad tensor.
#
#  Usage:
#    torchrun --nproc-per-node=2 tutorials/ascend/moe_backward/moe_backward_golden.py
# ============================================================================

import os
import time
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

G_IP_PORT = "tcp://127.0.0.1:8666"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# NPU bf16 torch.matmul is M-shape-dependent: rows not a multiple of the native
# tile (256) select a different Cube kernel that re-rounds by ~1 ULP. Padding the
# token (M) dimension of every grouped matmul to a multiple of 256 makes the
# golden bit-stable and match the triton kernel's BLOCK_M=64 tiling (see 06).
GOLDEN_MATMUL_M_TILE = 256


# ============================================================================
# 1.  Grouped matmul primitives (differentiable, M-padded to 256)
# ============================================================================

def _pad_m(slc, cnt):
    """Pad the row (M) dim of a [cnt, *] slice up to a multiple of GOLDEN_MATMUL_M_TILE."""
    pad = (-cnt) % GOLDEN_MATMUL_M_TILE
    if pad:
        slc = torch.nn.functional.pad(slc, (0, 0) * (slc.dim() - 1) + (0, pad))
    return slc


def grouped_matmul(a, weight, expert_counts, transpose=True):
    """Per-expert matmul over the token (M) dim, which is sorted/grouped by expert.

    a       : [M, K]   (rows sorted by local expert, contiguous per expert)
    weight  : [E, N, K] (transpose=True  -> out[m,n] = sum_k a[m,k] * weight[e,n,k]  => a @ weight[e].T, out [M, N])
                          (transpose=False -> out[m,n] = sum_k a[m,k] * weight[e,k,n]  => a @ weight[e],   out [M, N])
    expert_counts : [E] int

    Each expert slice is M-padded to 256 before the matmul (see GOLDEN_MATMUL_M_TILE).
    """
    out_parts = []
    start = 0
    for e in range(weight.shape[0]):
        cnt = int(expert_counts[e].item())
        if cnt > 0:
            slc = a[start:start + cnt]
            slc = _pad_m(slc, cnt)
            w = weight[e]
            res = (slc @ w.T) if transpose else (slc @ w)
            out_parts.append(res[:cnt])
            start += cnt
    if not out_parts:
        return torch.zeros(a.shape[0], weight.shape[1] if transpose else weight.shape[2],
                           dtype=a.dtype, device=a.device)
    return torch.cat(out_parts, dim=0)


def grouped_transposed_matmul(grad_out, orig_in, expert_counts):
    """Weight-grad grouped matmul: grad_weight[e] = grad_out[e]^T @ orig_in[e].

    grad_out : [M, N]   (rows sorted by expert)
    orig_in  : [M, K]   (rows sorted by expert)
    -> grad_weight [E, N, K]

    M-padded to 256 per expert (zero rows contribute nothing, but keep the same
    Cube kernel path as the input-grad matmuls for numerical consistency).
    """
    E = weight_E_from_counts(expert_counts)
    N = grad_out.shape[1]
    K = orig_in.shape[1]
    grad_weight = torch.zeros(E, N, K, dtype=grad_out.dtype, device=grad_out.device)
    start = 0
    for e in range(E):
        cnt = int(expert_counts[e].item())
        if cnt > 0:
            g = _pad_m(grad_out[start:start + cnt], cnt)      # [cnt_pad, N]
            o = _pad_m(orig_in[start:start + cnt], cnt)       # [cnt_pad, K]
            grad_weight[e] = g.T @ o                          # [N, K]
            start += cnt
    return grad_weight


def weight_E_from_counts(expert_counts):
    return int(expert_counts.shape[0])


# ============================================================================
# 2.  Metadata builder (ported from 06, BLOCK_SIZE_M=64)
# ============================================================================

BLOCK_SIZE_M = 64


def prepare_moe_metadata(expert_counts):
    """Build tile-based GroupGEMM metadata from per-expert token counts."""
    E = expert_counts.shape[0]
    expert_counts = expert_counts.to(torch.int64)
    tiles_per_expert = (expert_counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_total = int(tiles_per_expert.sum().item())
    meta_expert_ids = torch.zeros(num_total, dtype=torch.int32)
    meta_split_cum = torch.zeros(num_total, dtype=torch.int32)
    meta_tile_num = torch.zeros(num_total, dtype=torch.int32)
    meta_tile_num_cum = torch.zeros(num_total, dtype=torch.int32)
    idx = 0; tile_acc = 0; token_acc = 0
    for e in range(E):
        nt = int(tiles_per_expert[e].item())
        for t in range(nt):
            meta_expert_ids[idx] = e
            meta_split_cum[idx] = token_acc
            meta_tile_num[idx] = t
            meta_tile_num_cum[idx] = tile_acc
            idx += 1
        tile_acc += nt
        token_acc += int(expert_counts[e].item())
    num_tiles_total = torch.tensor([num_total], dtype=torch.int32)
    split_size_cum_per_expert = torch.zeros(E + 1, dtype=torch.int32)
    split_size_cum_per_expert[1:] = expert_counts.cumsum(0)
    return (split_size_cum_per_expert, meta_expert_ids, meta_split_cum,
            meta_tile_num, meta_tile_num_cum, num_tiles_total)


# ============================================================================
# 3.  Forward  (differentiable; returns output + saved dict for the hand bwd)
# ============================================================================

class AllToAll(torch.autograd.Function):
    """Autograd-aware all_to_all_single.

    dist.all_to_all_single is not autograd-aware on this torch_npu build, so we
    wrap it. The all-to-all is its own transpose: the backward of
    all_to_all(out_split=A, in_split=B) is all_to_all(out_split=B, in_split=A).
    This is exactly the combine<->dispatch direction swap used in the hand bwd.
    """

    @staticmethod
    def forward(ctx, input, output_split_sizes, input_split_sizes, group):
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        feat = input.shape[1:]
        out_rows = int(sum(output_split_sizes))
        out = input.new_empty((out_rows, *feat))
        dist.all_to_all_single(out, input, output_split_sizes=output_split_sizes,
                               input_split_sizes=input_split_sizes, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # swap split roles: grad_input = all_to_all(grad_output, out_split=B, in_split=A)
        feat = grad_output.shape[1:]
        in_rows = int(sum(ctx.input_split_sizes))
        grad_input = grad_output.new_empty((in_rows, *feat))
        dist.all_to_all_single(grad_input, grad_output,
                               output_split_sizes=ctx.input_split_sizes,
                               input_split_sizes=ctx.output_split_sizes,
                               group=ctx.group)
        return grad_input, None, None, None


def _a2a(input, output_split_sizes, input_split_sizes, group):
    return AllToAll.apply(input, output_split_sizes, input_split_sizes, group)


def moe_forward(hidden_states, routing_weights, selected_experts,
                fc1_1, fc1_2, fc2, ep_group, topk, return_saved=False):
    """EP-MoE forward. routing_weights [B,topk], selected_experts [B,topk] (global ids).

    Mirrors GPU torch_moe_fwd / 06 build_moe_fwd_inputs. No token dropping.
    When return_saved=True, also returns a dict of all intermediates needed by
    moe_backward_torch (detached under torch.no_grad() by the caller).
    """
    dtype = hidden_states.dtype
    device = hidden_states.device
    world_size = dist.get_world_size(ep_group)
    batch_size, hidden_dim = hidden_states.shape
    experts_per_rank = fc1_1.shape[0]
    num_experts = experts_per_rank * world_size
    ffn_dim = fc1_1.shape[1]
    ep_rank = dist.get_rank(ep_group)

    # ---- Dispatch: expand + sort by dest rank ----
    expanded_hidden = hidden_states.repeat_interleave(topk, dim=0)        # [B*topk, H]
    flat_weights = routing_weights.reshape(-1)                            # [B*topk]
    flat_indices = selected_experts.reshape(-1).to(torch.int64)           # [B*topk]
    expert_ranks = flat_indices // experts_per_rank
    local_experts = flat_indices % experts_per_rank

    send_counts = torch.bincount(expert_ranks, minlength=world_size).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    splits_send_list = send_counts.tolist()
    splits_recv_list = recv_counts.tolist()
    total_send = int(send_counts.sum().item())
    total_recv = int(recv_counts.sum().item())

    sort_idxs = torch.argsort(expert_ranks, stable=True)
    sorted_hidden = expanded_hidden[sort_idxs]
    sorted_weights = flat_weights[sort_idxs].to(dtype)
    sorted_local_experts = local_experts[sort_idxs]

    tokens_recv = _a2a(sorted_hidden, splits_recv_list, splits_send_list, ep_group)
    weights_recv = _a2a(sorted_weights, splits_recv_list, splits_send_list, ep_group)
    expert_ids_recv = torch.empty((total_recv,), dtype=torch.int64, device=device)
    dist.all_to_all_single(expert_ids_recv, sorted_local_experts.to(torch.int64),
                           output_split_sizes=splits_recv_list,
                           input_split_sizes=splits_send_list, group=ep_group)

    # ---- Sort received tokens by local expert ----
    local_expert_ids = expert_ids_recv % experts_per_rank
    local_sort_idxs = torch.argsort(local_expert_ids, stable=True)
    recv_hidden_sorted = tokens_recv[local_sort_idxs]
    recv_weights_sorted = weights_recv[local_sort_idxs]
    expert_counts = torch.bincount(local_expert_ids, minlength=experts_per_rank).to(torch.int32)

    meta = prepare_moe_metadata(expert_counts)
    (split_size_cum_per_expert, meta_expert_ids, meta_split_cum,
     meta_tile_num, meta_tile_num_cum, num_tiles_total) = meta

    # ---- fc1 + SwiGLU + fc2 ----
    fc1_combined = torch.cat([fc1_1, fc1_2], dim=1)                      # [E, 2*ffn, H]
    fc1_out = grouped_matmul(recv_hidden_sorted, fc1_combined, expert_counts, transpose=True)
    gate, up = fc1_out.chunk(2, dim=-1)                                  # each [M, ffn]
    swiglu_out = (torch.nn.functional.silu(gate.float()) * up.float())
    swiglu_out_weighted = (swiglu_out * recv_weights_sorted.float().unsqueeze(-1)).to(dtype)
    fc2_out = grouped_matmul(swiglu_out_weighted, fc2, expert_counts, transpose=True)  # [M, H]

    # ---- Combine: reverse-A2A + topk plain sum ----
    inv_local = torch.argsort(local_sort_idxs)
    fc2_out_unsorted = fc2_out[inv_local]                                # back to arrival order
    combined_out_flat = _a2a(fc2_out_unsorted, splits_send_list, splits_recv_list, ep_group)
    inv_sort = torch.argsort(sort_idxs)
    combined_full = combined_out_flat[inv_sort]                          # [B*topk, H]
    output = combined_full.view(batch_size, topk, hidden_dim).sum(dim=1)

    if not return_saved:
        return output

    saved = dict(
        output=output, dy=None,
        # dims / group
        batch_size=batch_size, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
        num_experts=num_experts, experts_per_rank=experts_per_rank, topk=topk,
        world_size=world_size, ep_rank=ep_rank, ep_group=ep_group,
        # dispatch / sort invariants
        sort_idxs=sort_idxs, local_sort_idxs=local_sort_idxs,
        inv_local=inv_local, inv_sort=inv_sort,
        splits_send_list=splits_send_list, splits_recv_list=splits_recv_list,
        total_send=total_send, total_recv=total_recv, M=int(expert_counts.sum().item()),
        expert_counts=expert_counts,
        split_size_cum_per_expert=split_size_cum_per_expert,
        meta_expert_ids=meta_expert_ids, meta_split_cum=meta_split_cum,
        meta_tile_num=meta_tile_num, meta_tile_num_cum=meta_tile_num_cum,
        num_tiles_total=num_tiles_total,
        # fwd activations
        recv_hidden_sorted=recv_hidden_sorted, fc1_output=fc1_out,
        gate=gate, up=up, swiglu_out_weighted=swiglu_out_weighted,
        recv_weights_sorted=recv_weights_sorted, fc2_out=fc2_out,
        # weights
        fc1_1=fc1_1, fc1_2=fc1_2, fc2=fc2, fc1_combined=fc1_combined,
    )
    return output, saved


# ============================================================================
# 4.  Hand-written backward — 5 mega-ops, each its own function
# ============================================================================
# Given dy [B, H]:
#   step1a combine_bwd_a2a : dy -> broadcast -> dispatch-A2A(home->expert) -> resort
#                            => grad_fc2_out_sorted [M, H]
#   step1b fc2_input_grad  : grad_fc2_out_sorted @ fc2[e]  => grad_swiglu [M, ffn]
#   step2  swiglu_bwd      : => grad_fc1_output [M, 2*ffn], grad_gate [M]
#   step3  fc2_weight_grad : grad_fc2_out_sorted^T @ swiglu_out_weighted => grad_fc2 [E,H,ffn]
#   step4a fc1_input_grad  : grad_fc1_output @ fc1[e] => grad_recv_hidden_sorted [M, H]
#   step4b dispatch_bwd    : reverse-A2A(expert->home) + topk sum => grad_hidden [B,H];
#                            grad_gate -> reverse-A2A + gather => grad_routing_weights [B,topk]
#   step5  fc1_weight_grad : grad_fc1_output^T @ recv_hidden_sorted => grad_fc1 -> chunk
# ============================================================================

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
    """Step 2: SwiGLU backward.
    fwd: swiglu_out = silu(gate)*up ; swiglu_out_weighted = swiglu_out * scale (scale=recv_weights_sorted)
      dGate = grad_swiglu * silu'(gate) * up * scale
      dUp   = grad_swiglu * silu(gate)  * scale
      dScale = sum(silu(gate)*up*grad_swiglu)  (per row)  => grad of recv_weights_sorted
    Returns grad_fc1_output [M,2*ffn] (=cat[dGate,dUp]), grad_gate [M]."""
    gate = saved["gate"].float()
    up = saved["up"].float()
    scale = saved["recv_weights_sorted"].float().unsqueeze(-1)
    g = grad_swiglu.float()
    sigmoid_g = torch.sigmoid(gate)
    silu_g = gate * sigmoid_g
    silu_prime = silu_g * (1 - sigmoid_g) + sigmoid_g          # d/dgate silu(gate)
    dGate = g * silu_prime * up * scale
    dUp = g * silu_g * scale
    grad_fc1_output = torch.cat([dGate, dUp], dim=-1).to(grad_swiglu.dtype)
    grad_gate = (silu_g * up * g).sum(dim=-1).to(grad_swiglu.dtype)   # dscale, [M]
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


# ============================================================================
# 5.  Autograd cross-check
# ============================================================================

def autograd_grads(hidden_states, routing_weights, selected_experts,
                   fc1_1, fc1_2, fc2, ep_group, topk, dy):
    """Run the differentiable forward and return autograd grads for all inputs."""
    hs = hidden_states.clone().requires_grad_(True)
    rw = routing_weights.clone().requires_grad_(True)
    w1 = fc1_1.clone().requires_grad_(True)
    w2 = fc1_2.clone().requires_grad_(True)
    wfc2 = fc2.clone().requires_grad_(True)
    out = moe_forward(hs, rw, selected_experts, w1, w2, wfc2, ep_group, topk, return_saved=False)
    out.backward(dy)
    return dict(
        grad_hidden=hs.grad, grad_routing_weights=rw.grad,
        grad_fc1_1=w1.grad, grad_fc1_2=w2.grad, grad_fc2=wfc2.grad,
        output=out.detach(),
    )


def _cmp(name, hand, auto, rtol=1e-3, atol=1e-3):
    hand = hand.float(); auto = auto.float()
    d = (hand - auto).abs()
    allowed = atol + rtol * auto.abs()
    n_bad = int((d > allowed).sum().item())
    max_d = float(d.max().item())
    ok = n_bad == 0
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}({n_bad},max={max_d:.2e})"
    print(f"    {name:24} shape={tuple(hand.shape)} max_abs={max_d:.3e}  {tag}", flush=True)
    return ok


def run_cross_check(ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group, seed=42):
    pe = dist.get_rank(ep_group)
    world_size = dist.get_world_size(ep_group)
    experts_per_rank = num_experts // world_size
    # fp32 for the math cross-check: isolates formula correctness from bf16
    # matmul accumulation noise (autograd's matmul backward uses a different
    # precision path than explicit torch.matmul). The bf16 triton-vs-golden
    # comparison lives in run_moe_backward.py with pad-256 + looser tolerance.
    dtype = torch.float32
    device = f"npu:{pe}"

    torch.manual_seed(seed + pe * 1000)
    hidden_states = torch.randn(ntokens, hidden_dim, dtype=dtype, device=device)
    gate_weights = torch.randn(num_experts, hidden_dim, dtype=dtype, device=device)
    fc1_1 = torch.randn(experts_per_rank, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc1_2 = torch.randn(experts_per_rank, ffn_dim, hidden_dim, dtype=dtype, device=device)
    fc2 = torch.randn(experts_per_rank, hidden_dim, ffn_dim, dtype=dtype, device=device)
    dist.broadcast(gate_weights, src=0, group=ep_group)

    # routing: softmax + topk + renorm (gives routing_weights + selected_experts)
    logits = hidden_states.float() @ gate_weights.float().T
    routing_weights = torch.softmax(logits, dim=-1).to(dtype)
    topk_w, topk_idx = torch.topk(routing_weights, topk, dim=-1)
    topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    dy = torch.randn_like(hidden_states)

    if pe == 0:
        print(f"  [cfg] tk={ntokens} h={hidden_dim} ffn={ffn_dim} E={num_experts} k={topk} "
              f"W={world_size}", flush=True)

    # ---- autograd ground truth ----
    ag = autograd_grads(hidden_states, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, dy)

    # ---- hand-written backward (no grad; detached saved) ----
    with torch.no_grad():
        _, saved = moe_forward(hidden_states, topk_w, topk_idx, fc1_1, fc1_2, fc2,
                               ep_group, topk, return_saved=True)
        hand = moe_backward_torch(saved, dy)

    # also check forward output consistency
    _cmp("output", saved["output"].float(), ag["output"].float())

    # fp32 matmul on NPU is not bit-exact across algorithms (explicit backward
    # matmul vs autograd's backward kernel), so use rtol/atol=2e-2 for the
    # cross-check. Paths with no backward matmul (output, grad_fc2) are exact;
    # grad_hidden chains two backward matmuls so it carries the most noise.
    RT, AT = 2e-2, 2e-2
    all_ok = True
    all_ok &= _cmp("grad_hidden", hand["grad_hidden"], ag["grad_hidden"], rtol=RT, atol=AT)
    all_ok &= _cmp("grad_routing_weights", hand["grad_routing_weights"], ag["grad_routing_weights"], rtol=RT, atol=AT)
    all_ok &= _cmp("grad_fc1_1", hand["grad_fc1_1"], ag["grad_fc1_1"], rtol=RT, atol=AT)
    all_ok &= _cmp("grad_fc1_2", hand["grad_fc1_2"], ag["grad_fc1_2"], rtol=RT, atol=AT)
    all_ok &= _cmp("grad_fc2", hand["grad_fc2"], ag["grad_fc2"], rtol=RT, atol=AT)
    return all_ok


def run_test_distributed():
    pe = dist.get_rank()
    ep_group = dist.group.WORLD
    num_experts = 128
    test_configs = [
        (2048,  2048, 768, 8),
        (8192,  2048, 768, 8),
    ]
    if pe == 0:
        print(f"{BOLD}[START]{RESET} MoE backward golden vs autograd cross-check", flush=True)
    results = []
    for cfg in test_configs:
        dist.barrier()
        ok = run_cross_check(*cfg, num_experts, ep_group=ep_group)
        results.append(ok)
    dist.barrier()
    if pe == 0:
        print(f"\n{BOLD}==== golden cross-check: "
              f"{'ALL PASS' if all(results) else 'SOME FAILED'} ===={RESET}", flush=True)


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print(f"[INFO] golden cross-check done", flush=True)
