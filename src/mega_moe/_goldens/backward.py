# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  src/mega_moe/_goldens/backward.py
#
#  Pure torch + hccl MoE backward golden reference.
#
#  Mirrors the GPU golden (test/nvidia/test_ep_moe_fused.py: torch_moe_fwd) and
#  the Ascend 06 tutorial forward, then adds a HAND-WRITTEN backward split into
#  the same 5 mega-ops as the GPU TritonDistFusedEpMoEFunction.backward:
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
#  The forward and grouped-matmul primitives live in
#  ``mega_moe._goldens._torch_forward_for_backward`` (the production differentiable forward reused
#  by ``MegaMoEBackwardFunction``); this module imports them so the golden and
#  the production path share one source of truth.
#
#  Usage:
#    torchrun --nproc-per-node=2 -m mega_moe._goldens.backward
# ============================================================================

import os
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from mega_moe._goldens._torch_forward_for_backward import (
    grouped_matmul,
    grouped_transposed_matmul,
    moe_forward,
)

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ============================================================================
# 1.  Hand-written backward — 5 mega-ops, each its own function
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
# 2.  Autograd cross-check
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
