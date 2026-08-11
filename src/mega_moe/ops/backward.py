# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Integration layer for the Ascend MoE backward triton mega-kernels.

Two things live here:

1. :func:`moe_backward_triton` — the 5-op orchestrator that chains the
   per-kernel wrappers from :mod:`mega_moe.kernels` end-to-end (mirrors the GPU
   ``TritonDistFusedEpMoeFunction.backward`` 5-op split).

2. :class:`MegaMoEBackwardFunction` — a ``torch.autograd.Function`` modeled on
   the GPU ``TritonDistFusedEpMoeFunction``: its ``forward`` runs the (torch)
   EP-MoE forward and stashes the saved intermediates + the shared symmetric
   buffer; its ``backward`` runs the 5 triton mega-ops. The forward reuses the
   differentiable torch forward in :mod:`mega_moe._goldens._torch_forward_for_backward`; only the
   backward is the fused triton path (the Ascend tutorial only ships a triton
   backward — forward is the torch reference).

The 5 backward mega-ops (given dy [B,H]):

1. ``dispatch_fc2_bwd``  — dispatch-A2A(home->expert) + fc2 input-grad
2. ``swiglu_bwd``        — SwiGLU backward
3. ``transposed_gemm``   — fc2 weight-grad  (reused for fc1)
4. ``combine_fc1_bwd``   — fc1 input-grad + reverse-A2A + gate-grad
5. ``transposed_gemm``   — fc1 weight-grad -> chunk(grad_fc1_1, grad_fc1_2)
"""

import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

from mega_moe._goldens._torch_forward_for_backward import moe_forward
from ..kernels import (
    dispatch_fc2_bwd_triton,
    swiglu_bwd_triton,
    transposed_grouped_gemm_triton,
    combine_fc1_bwd_triton,
)


def _grouped_wgrad_torch(grad_out, orig_in, expert_counts, ec_list=None):
    """Torch fallback for the grouped weight-grad GEMM ``grad_w[E,N,K] =
    grad_out^T @ orig_in`` (rows of grad_out/orig_in sorted by expert). Used for
    step3 (fc2 wgrad) and step5 (fc1 wgrad) when the triton
    ``transposed_grouped_gemm`` kernel is pathologically slow — currently the case
    for Kimi-K3 8-card, where the triton kernel hits a codegen pathology (0.16%
    cube peak) and torch's per-expert matmul is ~260x faster.

    Per-expert matmul, no M-padding (within bf16 tolerance vs the padded golden).
    This is the DEFAULT wgrad path for step3 (fc2) and step5 (fc1); set
    MOE_WGRAD_TRITON=1 to use the fused triton ``transposed_grouped_gemm``
    kernel instead (faster on most shapes, but pathological on Kimi-K3 8-card).
    """
    E = int(expert_counts.shape[0])
    N = grad_out.shape[1]
    K = orig_in.shape[1]
    dtype = grad_out.dtype
    dev = grad_out.device
    grad_w = torch.empty(E, N, K, dtype=dtype, device=dev)
    if ec_list is None:
        ec_list = expert_counts.cpu().tolist()  # one sync, avoid per-iter .item()
    start = 0
    for e in range(E):
        cnt = ec_list[e]
        if cnt > 0:
            g = grad_out[start:start + cnt]    # [cnt, N]
            o = orig_in[start:start + cnt]     # [cnt, K]
            grad_w[e] = (g.t() @ o).to(dtype)  # [N, K]
            start += cnt
    return grad_w


def _grouped_wgrad_npu(grad_out, orig_in, expert_counts):
    """npu grouped wgrad via torch_npu.npu_grouped_matmul(split_item=3, group_type=2).
    grad_w[e] = grad_out[e].T @ orig_in[e] = [N,K], stacked [E,N,K]. Much faster
    than the torch per-expert loop (matches bigop._grouped_wgrad). group_type=2
    requires grad_out.T as a TRANSPOSED VIEW (not .contiguous() — else loses the
    transpose marker -> EZ1001)."""
    group_list = expert_counts.to(torch.int64).to(grad_out.device)
    return torch_npu.npu_grouped_matmul(
        [grad_out.T], [orig_in], group_list=group_list,
        split_item=3, group_type=2, group_list_type=1)[0]


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
    use_triton_wgrad = os.environ.get("MOE_WGRAD_TRITON") == "1"
    use_torch_wgrad = os.environ.get("MOE_WGRAD_TORCH") == "1"  # fallback; default is npu
    _trace = bool(os.environ.get("MOE_BWD_TRACE"))
    _r = saved["ep_rank"]
    def _t(tag):
        if _trace:
            torch.npu.synchronize()
            print(f"[r{_r}] TRACE-BWD {tag}", flush=True)
    # P1 perf: run step3 (fc2 wgrad) and step5 (fc1 wgrad) on a SIDE STREAM so
    # their cube GEMMs overlap with step2 (swiglu, vector) and step4 (combine,
    # mostly vector push). Dependency: step3 needs only step1's grad_fc2_out_sorted;
    # step5 needs only step2's grad_fc1_output — neither needs the prior wgrad nor
    # step4 (verified against the golden's per-op deps). expert_counts is invariant,
    # so cache its list once to drop the per-call host sync. Roll back with
    # MOE_WGRAD_NOSTREAM=1.
    # NOTE: torch.npu.Stream is a *software* stream — it does NOT map to separate
    # cube/vector engines, so a wgrad on a side stream just contends with the main
    # stream's triton kernels for the same NPU queue. With torch's per-expert wgrad
    # loop (many small matmuls) this is a net 2.5x SLOWDOWN on small shapes (launch
    # storm + queue contention). Real cube/vector overlap has to live INSIDE one
    # kernel via al.scope(core_mode=...) — see the P0 signal/wait work. Default off;
    # opt back in with MOE_WGRAD_STREAM=1 for experimentation.
    use_side_stream = os.environ.get("MOE_WGRAD_STREAM") == "1"
    ec = saved["expert_counts"]
    ec_list = ec.cpu().tolist()  # one sync, reused by both wgrads
    wgrad_stream = torch.npu.Stream() if use_side_stream else None

    def _run_wgrad(fn, *args):
        if wgrad_stream is None:
            return fn(*args)
        ev = torch.npu.Event()
        ev.record()  # main-stream progress up to here
        with torch.npu.stream(wgrad_stream):
            wgrad_stream.wait_event(ev)  # wgrad waits until its inputs are ready
            return fn(*args)

    _t("step1-dispatch_fc2 start")
    # step 1: dispatch + fc2 input-grad
    grad_swiglu, grad_fc2_out_sorted = dispatch_fc2_bwd_triton(saved, dy, peer_mem)
    _t("step1-dispatch_fc2 done")
    # step 3: fc2 wgrad — launch on side stream NOW (depends only on step1),
    # overlaps with step2 + step4.
    if use_triton_wgrad:
        grad_fc2 = _run_wgrad(transposed_grouped_gemm_triton,
            grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec,
            saved["split_size_cum_per_expert"])
    elif use_torch_wgrad:
        grad_fc2 = _run_wgrad(_grouped_wgrad_torch,
            grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec, ec_list)
    else:
        grad_fc2 = _run_wgrad(_grouped_wgrad_npu,
            grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec)
    _t("step3-fc2_wgrad launched (side stream)")
    # step 2: swiglu backward (main stream; overlaps with step3 wgrad cube)
    grad_fc1_output, grad_gate = swiglu_bwd_triton(
        grad_swiglu, saved["fc1_output"], saved["recv_weights_sorted"])
    _t("step2-swiglu done")
    # step 5: fc1 wgrad — side stream (depends only on step2), overlaps with step4.
    if use_triton_wgrad:
        grad_fc1 = _run_wgrad(transposed_grouped_gemm_triton,
            grad_fc1_output, saved["recv_hidden_sorted"], ec,
            saved["split_size_cum_per_expert"])
    elif use_torch_wgrad:
        grad_fc1 = _run_wgrad(_grouped_wgrad_torch,
            grad_fc1_output, saved["recv_hidden_sorted"], ec, ec_list)
    else:
        grad_fc1 = _run_wgrad(_grouped_wgrad_npu,
            grad_fc1_output, saved["recv_hidden_sorted"], ec)
    _t("step5-fc1_wgrad launched (side stream)")
    # step 4: combine + fc1 input-grad + gate-grad (main stream; overlaps step5)
    grad_hidden, grad_routing_weights = combine_fc1_bwd_triton(
        saved, grad_fc1_output, grad_gate, peer_mem)
    _t("step4-combine_fc1 done")
    # ensure side-stream wgrads finished before chunk/return
    if wgrad_stream is not None:
        wgrad_stream.synchronize()
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
