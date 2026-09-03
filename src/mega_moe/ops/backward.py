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
   differentiable torch forward in :mod:`mega_moe.ops._torch_forward`; only the
   backward is the fused triton path (the Ascend tutorial only ships a triton
   backward — forward is the private differentiable Torch implementation).

3. :class:`MegaMoEFunction` — the Stage H3 merged autograd Function: its
   ``forward`` runs the *fused* operator with ``return_saved=True`` (the
   native saved dict of :mod:`mega_moe.ops._native_saved` rides along with the
   forward pass — no torch replay), and its ``backward`` runs the same 5
   mega-ops on that dict, behind the generation/owner gates and with the
   persistent tile-signal ``state`` injection (master plan §3.2.3/§3.4).

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

from ._torch_forward import moe_forward
from ..kernels import (
    dispatch_fc2_bwd_triton,
    swiglu_bwd_triton,
    transposed_grouped_gemm_triton,
    combine_fc1_bwd_triton,
)
from ..kernels.fused_swiglu_bwd_fc2_wgrad import fused_swiglu_bwd_fc2_wgrad


def _grouped_wgrad_torch(grad_out, orig_in, expert_counts, ec_list=None):
    """Torch fallback for the grouped weight-grad GEMM ``grad_w[E,N,K] =
    grad_out^T @ orig_in`` (rows of grad_out/orig_in sorted by expert). Used for
    step3 (fc2 wgrad) and step5 (fc1 wgrad) when the triton
    ``transposed_grouped_gemm`` kernel is pathologically slow — currently the case
    for Kimi-K3 8-card, where the triton kernel hits a codegen pathology (0.16%
    cube peak) and torch's per-expert matmul is ~260x faster.

    Per-expert matmul, no M-padding (within bf16 tolerance vs the padded Torch path).
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
def moe_backward_triton(saved, dy, peer_mem, grad_transport=None):
    """Run the 5 triton mega-ops end-to-end. peer_mem is ONE shared symmetric
    buffer at heap offset 0 (dl.symm_at only resolves correctly at offset 0 with
    a varying rank), reused by step 1 and step 4 (which run sequentially). The
    gate (routing-weight) grad is computed on the host. Returns a dict of grads
    matching the hand-written test baseline.

    A MoonEP ``saved_phys`` (``use_moonep=True``) runs the same 5 ops over the
    physical ``[home | replica]`` slot groups with a dual weight table per dgrad
    GEMM, forced onto the serial schedule (no dual-stream / fused / wgrad-tail
    overlap). The weight-grad ops then produce ``[epn + B, ...]`` physical
    gradients. With ``grad_transport=None`` (M2) the returned canonical keys
    carry the HOME segment ``[0, epn)`` and the replica segments ride along as
    ``_replica_grad_gate_up`` / ``_replica_grad_down`` (bf16 views) for the
    test-side owner-pull reduction oracle. Passing the transport borrowed from
    ``FusedMoEForward.lend_replica_weight_tables_for_grad`` instead sinks those
    replica segments into the forward's symmetric replica weight slots and
    reduces every copy onto its owner, so the canonical weight-grad keys are the
    final (owner-accumulated) values."""
    # Single-in-flight guard: the symmetric receive buffer is overwritten by
    # the forward's FC2 staging and by step 1 below, so a saved dict whose
    # recv_hidden_sorted still aliases it would silently read garbage.
    recv_hidden_sorted = saved.get("recv_hidden_sorted")
    if recv_hidden_sorted is None:
        raise ValueError(
            "saved is missing recv_hidden_sorted; the fused backward requires "
            "the activation section (replay saved or return_saved=True)"
        )
    if recv_hidden_sorted.data_ptr() == peer_mem.data_ptr():
        raise ValueError(
            "saved['recv_hidden_sorted'] aliases peer_mem; the receive buffer "
            "is single-in-flight and was overwritten after dispatch — the "
            "forward must clone it in the T2 window"
        )
    dy = dy.to(saved["fc1_1"].dtype)
    use_moonep = bool(saved.get("use_moonep"))
    if grad_transport is not None and not use_moonep:
        raise ValueError("grad_transport requires a MoonEP saved_phys")
    use_triton_wgrad = os.environ.get("MOE_WGRAD_TRITON") == "1"
    use_torch_wgrad = os.environ.get("MOE_WGRAD_TORCH") == "1"  # fallback; default is npu
    use_fused = (not use_moonep) and os.environ.get("MOE_FUSED_SWIGLU_WGRAD") == "1"  # step2+step3 fused (Cube/Vector concurrent)
    # MoonEP keeps the physical pipeline serial: the fused and dual-stream paths
    # assume the home-layout wgrad backends and would interleave the physical
    # group boundaries before the layout is proven.
    use_dual = (not use_moonep) and os.environ.get("MOE_BWD_DUAL_STREAM", "1") != "0"  # step3(cube) ∥ step2(vec) on two engine-pure streams (DEFAULT ON; =0 disables)
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
    # step4 (verified against the golden's per-op deps). expert_counts is
    # invariant; the torch wgrad path derives its python list lazily (only when
    # MOE_WGRAD_TORCH=1), so the default npu/triton paths pay no host sync.
    # Roll back with MOE_WGRAD_NOSTREAM=1.
    # NOTE: torch.npu.Stream is a *software* stream — it does NOT map to separate
    # cube/vector engines, so a wgrad on a side stream just contends with the main
    # stream's triton kernels for the same NPU queue. Re-tested 2026-08-13 with the
    # current npu-grouped wgrad backend on Kimi-K3 w8 t4k: MOE_WGRAD_STREAM=1 is a
    # ~12.5x SLOWDOWN (112 -> 1402 ms), even worse than the old torch per-expert
    # loop's 2.5x. Software-stream overlap is a confirmed dead end here. Real
    # cube/vector overlap has to live INSIDE one kernel via al.scope(core_mode=...)
    # — see the P0 signal/wait work. Default off; opt back in with MOE_WGRAD_STREAM=1.
    use_side_stream = (not use_moonep) and os.environ.get("MOE_WGRAD_STREAM") == "1"
    ec = saved["expert_counts"]
    wgrad_stream = torch.npu.Stream() if use_side_stream else None
    # Step-2 activation derivative: differentiate the same gated activation the
    # forward ran (SwiGLU default, SiTU-GLU for the Kimi hidden_act="situ"
    # path). Older saved dicts without the keys keep the SwiGLU derivative.
    _act_kwargs = dict(
        activation=saved.get("activation", "swiglu"),
        situ_beta=saved.get("situ_beta"),
        situ_linear_beta=saved.get("situ_linear_beta"),
    )

    def _run_wgrad(fn, *args):
        if wgrad_stream is None:
            return fn(*args)
        ev = torch.npu.Event()
        ev.record()  # main-stream progress up to here
        with torch.npu.stream(wgrad_stream):
            wgrad_stream.wait_event(ev)  # wgrad waits until its inputs are ready
            return fn(*args)

    # Per-stage NPU-event timing (default serial path only). Gated by
    # MOE_BWD_STAGE_TIMING; records 6 events -> 5 stage intervals (dispatch /
    # fc2_wgrad / swiglu / fc1_wgrad / combine), MAX-reduced across ranks and
    # appended to saved["_bwd_stage_samples"] for the bench to aggregate.
    _stage_timing = (not use_fused and not use_dual
                     and os.environ.get("MOE_BWD_STAGE_TIMING") == "1")
    if _stage_timing:
        _sev = [torch.npu.Event(enable_timing=True) for _ in range(6)]
        _sev[0].record()
    _t("step1-dispatch_fc2 start")
    # step 1: dispatch + fc2 input-grad
    grad_swiglu, grad_fc2_out_sorted = dispatch_fc2_bwd_triton(saved, dy, peer_mem)
    if _stage_timing:
        _sev[1].record()
    _t("step1-dispatch_fc2 done")
    if use_fused:
        # step2 (SwiGLU bwd, Vector) + step3 (fc2 wgrad, Cube) fused into ONE
        # launch so the two run concurrently on each AICore (vector hidden by
        # cube). Inputs all ready after step1. No side stream (B3 showed s/w
        # streams are a 12.5x regression). MOE_FUSED_WGRAD_BLOCK_M tunes cube BM.
        grad_fc1_output, grad_gate, grad_fc2 = fused_swiglu_bwd_fc2_wgrad(
            grad_swiglu, saved["fc1_output"], saved["recv_weights_sorted"],
            grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec,
            saved["split_size_cum_per_expert"], **_act_kwargs)
        _t("step2+step3 fused done")
    elif use_dual:
        # step3 (fc2 wgrad, pure Cube via npu_grouped_matmul) ∥ step2 (swiglu,
        # pure Vector triton) on TWO ENGINE-PURE streams — isolated from the
        # mixed-engine step1/step4. The B3 MOE_WGRAD_STREAM pathology (12.5x)
        # came from the wgrad stream contending with the main stream's mixed
        # cube+vector triton kernels; here the overlap window contains only
        # pure-cube (s_cube) + pure-vector (s_vec), mirroring /tmp/dual_stream.py.
        cur = torch.npu.current_stream()
        device = f"npu:{saved['ep_rank']}"
        if saved.get("_dual_cube_stream") is None:
            saved["_dual_cube_stream"] = torch.npu.Stream(device=device)
            saved["_dual_vec_stream"] = torch.npu.Stream(device=device)
        s_cube = saved["_dual_cube_stream"]
        s_vec = saved["_dual_vec_stream"]
        ev1 = torch.npu.Event(); ev1.record(cur)           # step1 main-stream progress
        ev_cube = torch.npu.Event()
        with torch.npu.stream(s_cube):
            s_cube.wait_event(ev1)
            grad_fc2 = _grouped_wgrad_npu(
                grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec)   # step3, pure Cube
            ev_cube.record(s_cube)
        ev_vec = torch.npu.Event()
        with torch.npu.stream(s_vec):
            s_vec.wait_event(ev1)
            grad_fc1_output, grad_gate = swiglu_bwd_triton(
                grad_swiglu, saved["fc1_output"], saved["recv_weights_sorted"],
                **_act_kwargs)  # step2, pure Vector
            ev_vec.record(s_vec)
        cur.wait_event(ev_cube)
        cur.wait_event(ev_vec)
        _t("step2+step3 dual-stream done")
    else:
        # step 3: fc2 wgrad — launch on side stream NOW (depends only on step1),
        # overlaps with step2 + step4.
        if use_triton_wgrad:
            grad_fc2 = _run_wgrad(transposed_grouped_gemm_triton,
                grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec,
                saved["split_size_cum_per_expert"])
        elif use_torch_wgrad:
            grad_fc2 = _run_wgrad(_grouped_wgrad_torch,
                grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec)
        else:
            grad_fc2 = _run_wgrad(_grouped_wgrad_npu,
                grad_fc2_out_sorted, saved["swiglu_out_weighted"], ec)
        if _stage_timing:
            _sev[2].record()
        _t("step3-fc2_wgrad launched (side stream)")
        # step 2: swiglu backward (main stream; overlaps with step3 wgrad cube)
        grad_fc1_output, grad_gate = swiglu_bwd_triton(
            grad_swiglu, saved["fc1_output"], saved["recv_weights_sorted"],
            **_act_kwargs)
        if _stage_timing:
            _sev[3].record()
        _t("step2-swiglu done")
    # step 5: fc1 wgrad — depends only on step2's grad_fc1_output. By default it
    # is NOT launched inline: it is deferred into step4's combine pipeline as the
    # cube-stream tail (after the last combine GEMM group), so the pure-Cube
    # wgrad overlaps step4's pure-Vector push/barrier/reduce drain instead of
    # serializing before the whole combine. Inline launch (old behavior) is
    # restored with MOE_BWD_WGRAD_TAIL=0, and stays inline for the diagnostic
    # serial/stage-timing paths and the host-syncing torch wgrad backend.
    _wgrad_tail = (
        not use_fused and not _stage_timing and not use_moonep
        and os.environ.get("MOE_BWD_COMBINE_SERIAL") != "1"
        and not use_torch_wgrad and not use_side_stream
        and os.environ.get("MOE_BWD_WGRAD_TAIL", "1") != "0"
    )

    def _fc1_wgrad():
        if use_triton_wgrad:
            return _run_wgrad(transposed_grouped_gemm_triton,
                              grad_fc1_output, saved["recv_hidden_sorted"], ec,
                              saved["split_size_cum_per_expert"])
        if use_torch_wgrad:
            return _run_wgrad(_grouped_wgrad_torch,
                              grad_fc1_output, saved["recv_hidden_sorted"], ec)
        return _run_wgrad(_grouped_wgrad_npu,
                          grad_fc1_output, saved["recv_hidden_sorted"], ec)

    if not _wgrad_tail:
        grad_fc1 = _fc1_wgrad()
        if _stage_timing:
            _sev[4].record()
        _t("step5-fc1_wgrad launched (inline)")
    # step 4: combine + fc1 input-grad + gate-grad; with the tail, step5's wgrad
    # rides the combine cube stream and is returned with the combine results.
    if _wgrad_tail:
        grad_hidden, grad_routing_weights, grad_fc1 = combine_fc1_bwd_triton(
            saved, grad_fc1_output, grad_gate, peer_mem, cube_tail=_fc1_wgrad)
    else:
        grad_hidden, grad_routing_weights = combine_fc1_bwd_triton(
            saved, grad_fc1_output, grad_gate, peer_mem)
    if _stage_timing:
        _sev[5].record()
        _sev[5].synchronize()
        _iv = [_sev[i].elapsed_time(_sev[i + 1]) for i in range(5)]
        _tmax = torch.tensor(_iv, dtype=torch.float32,
                             device=f"npu:{saved['ep_rank']}")
        dist.all_reduce(_tmax, op=dist.ReduceOp.MAX, group=saved["ep_group"])
        saved.setdefault("_bwd_stage_samples", []).append(
            [float(x) for x in _tmax.cpu().tolist()])
        if saved["ep_rank"] == 0:
            # per-backward stage wall (ms, MAX across ranks): dispatch /
            # fc2_wgrad / swiglu / fc1_wgrad / combine — surfaces directly in
            # the integrated training log, not only via the bench aggregate.
            print("[bwd-stage] " + " ".join(
                f"{v:8.1f}" for v in _tmax.cpu().tolist()), flush=True)
    _t("step4-combine_fc1 done")
    # ensure side-stream wgrads finished before chunk/return
    if wgrad_stream is not None:
        wgrad_stream.synchronize()
    if use_moonep:
        # Physical wgrad groups: the canonical keys expose the home segment
        # [0, epn) (owner semantics); the replica segments are handed to the
        # caller so the reduction oracle (and, later, the symmetric-slot
        # transport) can accumulate them onto their owners.
        home_experts = int(saved["home_experts_per_rank"])
        if grad_transport is not None:
            # Sink the replica segments into the borrowed symmetric slots and
            # owner-pull every copy back (seed = the local home segment in
            # fp32). The three transport barriers are collective, so every rank
            # reaches them through this same call.
            grad_transport.sink(grad_fc1, grad_fc2)
            grad_fc1_reduced, grad_fc2_reduced = grad_transport.reduce(
                grad_fc1, grad_fc2
            )
            grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1_reduced, 2, dim=1)
            return dict(
                grad_hidden=grad_hidden,
                grad_routing_weights=grad_routing_weights,
                grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2,
                grad_fc2=grad_fc2_reduced,
                grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
                grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
                grad_fc1=grad_fc1,
                _replica_grad_gate_up=grad_fc1[home_experts:],
                _replica_grad_down=grad_fc2[home_experts:],
                # pre-reduction seeds (views): what the owner started from
                _home_grad_fc1=grad_fc1[:home_experts],
                _home_grad_fc2=grad_fc2[:home_experts],
            )
        grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1[:home_experts], 2, dim=1)
        return dict(
            grad_hidden=grad_hidden, grad_routing_weights=grad_routing_weights,
            grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2,
            grad_fc2=grad_fc2[:home_experts],
            grad_swiglu=grad_swiglu, grad_fc1_output=grad_fc1_output,
            grad_gate=grad_gate, grad_fc2_out_sorted=grad_fc2_out_sorted,
            grad_fc1=grad_fc1,
            _replica_grad_gate_up=grad_fc1[home_experts:],
            _replica_grad_down=grad_fc2[home_experts:],
        )
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

    forward : private differentiable Torch EP-MoE forward — stashes the saved
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


# ============================================================================
# 3.  merged native-saved autograd.Function  (master plan §3.2.3, Stage H3)
# ============================================================================
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
        # The saved intermediates are freshly computed views/clones (not the
        # forward inputs), and the weight views must stay pinned until the
        # backward — stash on ctx instead of save_for_backward, mirroring
        # MegaMoEBackwardFunction.
        ctx.op = op
        ctx.saved = saved
        ctx.peer_mem = peer_mem
        ctx.state = state
        return output

    @staticmethod
    def backward(ctx, dy):
        op = ctx.op
        saved = ctx.saved
        if (
            saved.get("_owner_token") != id(op._routing_owner_token)
            or saved.get("_routing_generation") != op._routing_generation
        ):
            wrong_owner = (
                saved.get("_owner_token") != id(op._routing_owner_token)
            )
            raise RuntimeError(
                "the saved dict belongs to "
                + ("another operator instance" if wrong_owner
                   else "a routing plan this operator has already superseded")
                + "; run backward before the next forward on the operator"
            )
        # §3.4 state injection: reuse the persistent symmetric tile-signal
        # slots and keep the SET epoch monotonically advancing.  The first
        # epoch must be >= 1 — a freshly zeroed slot already reads as 0.
        if ctx.state is not None:
            if getattr(ctx.state, "signal_mem", None) is not None:
                saved["_bwd_tile_signal_mem"] = ctx.state.signal_mem
            saved["_bwd_tile_signal_epoch"] = max(
                int(getattr(ctx.state, "epoch", 0)), 1
            )
        # MoonEP physical saved (Stage N): sink the replica weight gradients
        # into the borrowed symmetric tables and owner-pull every copy — the
        # §3.3 autograd-side transport hookup.  Lending invalidates the
        # replica weight cache, so no forward may run on this operator until
        # the borrowed transport has been reduced.
        grad_transport = None
        if saved.get("use_moonep"):
            grad_transport = op.lend_replica_weight_tables_for_grad()
        grads = moe_backward_triton(saved, dy, ctx.peer_mem,
                                    grad_transport=grad_transport)
        # Write the lazily allocated signal memory and the bumped epoch back
        # so the next step reuses both.
        if ctx.state is not None:
            ctx.state.signal_mem = saved.get("_bwd_tile_signal_mem")
            ctx.state.epoch = saved.get("_bwd_tile_signal_epoch", 1)
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
