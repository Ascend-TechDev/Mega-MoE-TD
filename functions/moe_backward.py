# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  functions/moe_backward.py
#
#  Integration layer for the Ascend MoE backward triton mega-kernels.
#
#  Two things live here:
#
#  1. ``moe_backward_triton(saved, dy, peer_mem)`` — the 5-op orchestrator that
#     chains the per-kernel wrappers from ``benchmark.kernel.*`` end-to-end
#     (mirrors the GPU ``TritonDistFusedEpMoeFunction.backward`` 5-op split).
#
#  2. ``MegaMoEBackwardFunction`` — a ``torch.autograd.Function`` modeled on the
#     GPU ``TritonDistFusedEpMoeFunction``: its ``forward`` runs the (torch)
#     EP-MoE forward and stashes the saved intermediates + the shared symmetric
#     buffer; its ``backward`` runs the 5 triton mega-ops. The forward reuses the
#     differentiable golden forward from ``benchmark.moe_backward_golden``; only
#     the backward is the fused triton path (the Ascend tutorial only ships a
#     triton backward — forward is the torch reference).
#
#  The 5 backward mega-ops (given dy [B,H]):
#    1. dispatch_fc2_bwd   : dispatch-A2A(home->expert) + fc2 input-grad
#    2. swiglu_bwd         : SwiGLU backward
#    3. transposed_gemm    : fc2 weight-grad  (reused for fc1)
#    4. combine_fc1_bwd    : fc1 input-grad + reverse-A2A + gate-grad
#    5. transposed_gemm    : fc1 weight-grad -> chunk(grad_fc1_1, grad_fc1_2)
# ============================================================================

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from benchmark.moe_backward_golden import moe_forward
from kernels import (
    dispatch_fc2_bwd_triton,
    swiglu_bwd_triton,
    transposed_grouped_gemm_triton,
    combine_fc1_bwd_triton,
)


# ============================================================================
# 1.  5-op orchestrator
# ============================================================================
def moe_backward_triton(saved, dy, peer_mem):
    """Run the 5 triton mega-ops end-to-end. peer_mem is ONE shared symmetric
    buffer at heap offset 0 (dl.symm_at only resolves correctly at offset 0 with
    a varying rank), reused by step 1 and step 4 (which run sequentially). The
    gate (routing-weight) grad is computed on the host. Returns a dict of grads
    matching moe_backward_torch."""
    dy = dy.to(saved["fc1_1"].dtype)
    # step 1: dispatch + fc2 input-grad
    grad_swiglu, grad_fc2_out_sorted = dispatch_fc2_bwd_triton(saved, dy, peer_mem)
    # step 2: swiglu backward
    grad_fc1_output, grad_gate = swiglu_bwd_triton(grad_swiglu, saved["fc1_output"], saved["recv_weights_sorted"])
    # step 3: fc2 weight-grad
    grad_fc2 = transposed_grouped_gemm_triton(
        grad_fc2_out_sorted, saved["swiglu_out_weighted"], saved["expert_counts"],
        saved["split_size_cum_per_expert"])
    # step 4: combine + fc1 input-grad + gate-grad
    grad_hidden, grad_routing_weights = combine_fc1_bwd_triton(
        saved, grad_fc1_output, grad_gate, peer_mem)
    # step 5: fc1 weight-grad
    grad_fc1 = transposed_grouped_gemm_triton(
        grad_fc1_output, saved["recv_hidden_sorted"], saved["expert_counts"],
        saved["split_size_cum_per_expert"])
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing_weights,
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output, grad_gate=grad_gate,
        grad_fc2_out_sorted=grad_fc2_out_sorted, grad_fc1=grad_fc1,
    )


# ============================================================================
# 2.  autograd.Function  (modeled on GPU TritonDistFusedEpMoeFunction)
# ============================================================================
class MegaMoEBackwardFunction(torch.autograd.Function):
    """Fused EP-MoE with a triton mega-kernel backward.

    forward : torch EP-MoE forward (golden, differentiable) — stashes the saved
              intermediates and the shared symmetric buffer on the ctx.
    backward: the 5 triton mega-ops (``moe_backward_triton``).

    Args (forward):
        hidden_states    [B, H]            (requires grad)
        routing_weights  [B, topk]         (requires grad)
        selected_experts [B, topk] int     (global expert ids, no grad)
        fc1_1, fc1_2     [E_local, ffn, H] (requires grad)
        fc2              [E_local, H, ffn] (requires grad)
        ep_group         dist.ProcessGroup
        topk             int
        peer_mem         shared symmetric shmem tensor at heap offset 0
    """

    @staticmethod
    def forward(ctx, hidden_states, routing_weights, selected_experts,
                fc1_1, fc1_2, fc2, ep_group, topk, peer_mem):
        with torch.no_grad():
            output, saved = moe_forward(
                hidden_states, routing_weights, selected_experts,
                fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
        # saved intermediates are freshly computed (not forward inputs) -> safe to
        # stash on ctx directly; peer_mem is a non-grad symmetric buffer.
        ctx.saved = saved
        ctx.peer_mem = peer_mem
        ctx.ep_group = ep_group
        ctx.topk = topk
        return output

    @staticmethod
    def backward(ctx, dy):
        saved = ctx.saved
        grads = moe_backward_triton(saved, dy, ctx.peer_mem)
        # match forward arg order:
        # (hidden_states, routing_weights, selected_experts, fc1_1, fc1_2, fc2,
        #  ep_group, topk, peer_mem)
        return (
            grads["grad_hidden"],          # hidden_states
            grads["grad_routing_weights"], # routing_weights
            None,                          # selected_experts (int idx, no grad)
            grads["grad_fc1_1"],           # fc1_1
            grads["grad_fc1_2"],           # fc1_2
            grads["grad_fc2"],             # fc2
            None,                          # ep_group
            None,                          # topk
            None,                          # peer_mem
        )
