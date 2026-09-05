# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  src/mega_moe/_goldens/bigop_ref.py
#
#  Second MoE backward golden: compute core swapped to bigop's NPU fused
#  primitives (npu_grouped_matmul / npu_swiglu(_backward)); A2A on both ends
#  reused from the torch golden. See access_bigop.md (Step 2) for the design
#  rationale and the per-op mapping table.
#
#  Entry signature and return-dict keys are identical to moe_backward_torch,
#  so it drops in as a third column for three-way comparison (Phase 1) and is
#  intended to REPLACE the torch golden as the Triton comparison baseline
#  (Phase 2 = "take over" the backward golden).
#
#  Independence boundary (honest): only the compute dimension differs from the
#  torch golden — the `saved` dict, combine_bwd_a2a and dispatch_bwd are shared.
#  So the cross-check covers GEMM + SwiGLU kernels, NOT dispatch/A2A/forward-saved
#  construction (those stay covered by backward.py::run_cross_check autograd).
# ============================================================================
import os

import torch
import torch_npu  # noqa: F401  (npu_swiglu, npu_swiglu_backward)
import torch.distributed as dist

from bigop import _grouped_matmul, _grouped_wgrad
from mega_moe._goldens._torch_forward_for_backward import moe_forward
from mega_moe._goldens.backward import combine_bwd_a2a, dispatch_bwd, moe_backward_torch

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def moe_backward_bigop(saved, dy):
    """bigop-compute backward golden. Return keys are identical to ``moe_backward_torch``.

    A2A on both ends (combine_bwd_a2a / dispatch_bwd) reuse the torch golden;
    only the GEMM (input-grad + wgrad) and SwiGLU-backward cores are swapped to
    bigop's npu fused primitives — a *different kernel path* than the torch-loop
    golden, which is the whole reason this second golden exists.

    Weight layout: mega_moe stores linear weights as ``[E, out, in]``. bigop's
    ``_grouped_matmul`` wants ``[E, K=in, N=out]`` and ``_grouped_wgrad`` returns
    ``[E, in, out]`` — so input-grad steps need no adaptation (tail-two dims
    already match), while wgrad steps transpose the result back. Note the wgrad
    arg order is SWAPPED vs mega_moe ``grouped_transposed_matmul(grad_out, orig_in)``:
    bigop is ``(orig_in, grad_out)``.
    """
    dy = dy.to(saved["output"].dtype)
    counts = saved["expert_counts"]                  # int32; bigop casts to int64 internally

    # 1a combine-bwd-A2A — reuse (torch + HCCL)
    grad_fc2_out_sorted = combine_bwd_a2a(dy, saved)  # [M, H]

    # 1b fc2 input-grad — bigop: fc2[E,H,ffn] is already [E,K=H,N=ffn], no adapt
    grad_swiglu = _grouped_matmul(grad_fc2_out_sorted, saved["fc2"], counts)  # [M, ffn]

    # 2 SwiGLU bwd — bigop factorization (bwd-of-*scale + npu_swiglu_backward)
    probs = saved["recv_weights_sorted"].unsqueeze(-1)                          # [M, 1]
    grad_swiglu_for_npu = grad_swiglu * probs                                   # bwd of *scale
    grad_fc1_output = torch_npu.npu_swiglu_backward(
        grad_swiglu_for_npu, saved["fc1_output"], dim=-1)                       # [M, 2*ffn]
    swiglu_out = torch_npu.npu_swiglu(saved["fc1_output"], dim=-1)              # [M, ffn] recomputed (probs-grad needs it)
    grad_gate = (grad_swiglu * swiglu_out).sum(dim=-1)                          # [M] = dScale, feeds dispatch_bwd

    # 3 fc2 wgrad — bigop: returns [E,ffn,H], transpose back to [E,H,ffn]
    g = _grouped_wgrad(saved["swiglu_out_weighted"], grad_fc2_out_sorted, counts)  # [E, ffn, H]
    grad_fc2 = g.transpose(-1, -2).contiguous()                                # [E, H, ffn]

    # 4a fc1 input-grad — bigop: fc1_combined[E,2*ffn,H] is [E,K=2*ffn,N=H], no adapt
    grad_recv_hidden_sorted = _grouped_matmul(grad_fc1_output, saved["fc1_combined"], counts)  # [M, H]

    # 4b dispatch-bwd — reuse (torch + HCCL reverse-A2A)
    grad_hidden, grad_routing_weights = dispatch_bwd(grad_recv_hidden_sorted, grad_gate, saved)

    # 5 fc1 wgrad — bigop: returns [E,H,2*ffn], transpose -> [E,2*ffn,H], chunk
    g = _grouped_wgrad(saved["recv_hidden_sorted"], grad_fc1_output, counts)   # [E, H, 2*ffn]
    grad_fc1 = g.transpose(-1, -2).contiguous()                                # [E, 2*ffn, H]
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)                   # each [E, ffn, H]

    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing_weights,
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        # intermediates (keys identical to moe_backward_torch, for per-op diff)
        grad_fc2_out_sorted=grad_fc2_out_sorted, grad_swiglu=grad_swiglu,
        grad_fc1_output=grad_fc1_output, grad_gate=grad_gate,
        grad_recv_hidden_sorted=grad_recv_hidden_sorted, grad_fc1=grad_fc1,
    )


# ============================================================================
#  Standalone cross-check: bigop golden vs torch golden (Phase-0 smoke).
#  Needs only torch + torch_npu + HCCL — NO Triton / AscendNPU-IR, so this can
#  run before the tri-path test harness is wired up.
#
#    cd /home/z00905891/Mega-MoE-TD
#    torchrun --nproc-per-node=2 --master-port=29555 src/mega_moe/_goldens/bigop_ref.py
# ============================================================================
def _cmp(name, a, b, rtol=2e-2, atol=1e-2):
    # Global-max rule, identical to tests/_numeric.py::cmp_grad (GRAD_RTOL/ATOL).
    a = a.float(); b = b.float()
    d = (a - b).abs()
    max_d = float(d.max().item())
    gmax = float(b.abs().max().item())
    n_bad = int((d > atol + rtol * gmax).sum().item())
    rel = max_d / (gmax + 1e-9)
    ok = n_bad == 0
    tag = (f"{GREEN}PASS{RESET}" if ok
           else f"{RED}FAIL{RESET}(nbad={n_bad},max={max_d:.2e},rel={rel:.2e})")
    print(f"    {name:26} shape={tuple(a.shape)} max_abs={max_d:.3e} rel={rel:.2e}  {tag}", flush=True)
    return ok


def _run_gold_vs_big(ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group, seed=42):
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

    with torch.no_grad():
        _, saved = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
        gold = moe_backward_torch(saved, dy)
        big = moe_backward_bigop(saved, dy)

    if pe == 0:
        print(f"  [cfg] tk={ntokens} h={hidden_dim} ffn={ffn_dim} E={num_experts} k={topk} W={world_size}", flush=True)
    # full grads + the compute-path intermediates (to localize any adaptation bug)
    checks = ["grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2",
              "grad_swiglu", "grad_fc1_output", "grad_gate"]
    all_ok = True
    for n in checks:
        all_ok &= _cmp(n, gold[n], big[n])
    return all_ok


def _main():
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    ep_group = dist.group.WORLD
    dist.barrier()
    if local_pe == 0:
        print(f"{BOLD}[START]{RESET} bigop golden vs torch golden (compute-kernel cross-check, bf16)",
              flush=True)
    # small shape first (no 0-token-expert risk), then a Kimi-like shape
    configs = [
        (512, 512, 256, 4, 16),
        (2048, 2048, 768, 8, 128),
    ]
    results = []
    for cfg in configs:
        dist.barrier()
        try:
            results.append(_run_gold_vs_big(*cfg, ep_group=ep_group))
        except Exception as ex:  # noqa: BLE001
            if local_pe == 0:
                import traceback
                print(f"  [skip] {cfg}: {str(ex)[:160]}", flush=True)
                traceback.print_exc()
            results.append(False)
        dist.barrier()
    if local_pe == 0:
        verdict = "ALL PASS" if all(results) else "SOME FAILED"
        print(f"\n{BOLD}==== gold vs big: {verdict} ===={RESET}", flush=True)


if __name__ == "__main__":
    _main()
