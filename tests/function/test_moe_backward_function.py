# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  tests/function/test_moe_backward_function.py
#
#  Correctness test for ``mega_moe.MegaMoEBackwardFunction`` — the
#  autograd.Function that integrates the 5 Ascend triton backward mega-ops.
#
#  For each config:
#    1. build inputs (bf16, on NPU)
#    2. autograd path: clone inputs with requires_grad, run
#       MegaMoEBackwardFunction.apply(...).backward(dy) -> leaf .grad
#    3. golden path: moe_forward(return_saved=True) + moe_backward_torch(saved, dy)
#    4. compare the 5 grads (grad_hidden / grad_routing / grad_fc1_1/2 / grad_fc2)
#       with the bf16-appropriate metric from run_moe_backward
#
#  Usage (2 cards, AscendNPU-IR 1.2.0 bishengir):
#    PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
#    TRITON_CACHE_DIR=/tmp/triton_mb \
#    torchrun --nproc-per-node=2 tests/function/test_moe_backward_function.py
# ============================================================================

import math
import os

import torch
import torch_npu  # noqa: F401
import shmem as ash
import torch.distributed as dist

from mega_moe import MegaMoEBackwardFunction
from mega_moe.ops._legacy_backward_golden import moe_forward, moe_backward_torch

g_ash_size = 512 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _cmp(name, tri, gold, rtol=2e-2, atol=1e-2):
    tri = tri.float()
    gold = gold.float()
    d = (tri - gold).abs()
    max_d = float(d.max().item())
    gmax = float(gold.abs().max().item())
    allowed = float(atol + rtol * gmax)
    rel = max_d / (gmax + 1e-9)
    bad = ~torch.isfinite(tri) | ~torch.isfinite(gold) | ~torch.isfinite(d) | (d > allowed)
    n_bad = int(bad.sum().item())
    if not all(math.isfinite(metric) for metric in (max_d, gmax, allowed, rel)):
        n_bad = max(n_bad, 1)
    ok = n_bad == 0
    return ok, max_d, rel, n_bad


def _build_inputs(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42):
    pe = dist.get_rank(ep_group); W = dist.get_world_size(ep_group)
    epr = num_experts // W
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


def _all_ranks_pass(local_pass, device, ep_group):
    status = torch.tensor([int(local_pass)], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=ep_group)
    return bool(status.item())


def run_one(ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group):
    pe = dist.get_rank(ep_group)
    hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device = _build_inputs(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group)
    label = f"tk={ntokens:>5} h={hidden_dim} ffn={ffn_dim} k={topk}"

    # shared symmetric buffer at heap offset 0 (reused by step1 & step4)
    # size from a no_grad forward probe
    with torch.no_grad():
        _, probe = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
    peer_elems = max(probe["total_recv"], probe["total_send"]) * probe["hidden_dim"]
    peer_mem = ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=pe)

    # ---- autograd path: MegaMoEBackwardFunction ----
    hs_a = hs.clone().detach().requires_grad_(True)
    rw_a = topk_w.clone().detach().requires_grad_(True)
    w1_a = fc1_1.clone().detach().requires_grad_(True)
    w2_a = fc1_2.clone().detach().requires_grad_(True)
    wfc2_a = fc2.clone().detach().requires_grad_(True)
    output = MegaMoEBackwardFunction.apply(
        hs_a, rw_a, topk_idx, w1_a, w2_a, wfc2_a, ep_group, topk, peer_mem)
    output.backward(dy)
    tri = dict(
        grad_hidden=hs_a.grad, grad_routing_weights=rw_a.grad,
        grad_fc1_1=w1_a.grad, grad_fc1_2=w2_a.grad, grad_fc2=wfc2_a.grad,
    )

    # ---- golden path: hand torch+hccl backward ----
    with torch.no_grad():
        _, saved = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
        gold = moe_backward_torch(saved, dy)

    ash.aclshmem_free_tensor(peer_mem)

    checks = ["grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2"]
    all_ok = True
    detail = []
    for n in checks:
        ok, mx, rel, nbad = _cmp(n, tri[n], gold[n])
        all_ok &= ok
        detail.append(f"{n}:{GREEN}PASS{RESET}" if ok else f"{n}:{RED}FAIL{RESET}({nbad},mx={mx:.1e})")

    if pe == 0:
        cor = f"{GREEN}ALL PASS{RESET}" if all_ok else f"{RED}SOME FAIL{RESET}"
        print(f"  {label}  | {cor}", flush=True)
        if not all_ok:
            print("    " + "  ".join(detail), flush=True)
    return all_ok


def run_test_distributed():
    pe = dist.get_rank(); ep_group = dist.group.WORLD
    ash.set_conf_store_tls(False, "")
    attr = ash.InitAttr()
    attr.my_rank = pe; attr.n_ranks = dist.get_world_size(); attr.local_mem_size = g_ash_size
    attr.ip_port = G_IP_PORT; attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    assert ash.aclshmem_init(attr) == 0

    num_experts = 128
    test_configs = [
        (512,  512, 256, 4),
        (1024, 512, 256, 4),
        (512,  1024, 512, 8),
    ]

    if pe == 0:
        print(f"{BOLD}[START]{RESET} MegaMoEBackwardFunction (autograd) vs golden "
              f"on world_size={dist.get_world_size()}", flush=True)
    results = []
    try:
        for cfg in test_configs:
            dist.barrier()
            results.append(run_one(*cfg, num_experts, ep_group))
        dist.barrier()
        local_pass = bool(results) and all(results)
        passed = _all_ranks_pass(local_pass, f"npu:{pe}", ep_group)
        if pe == 0:
            print(f"\n{BOLD}==== MegaMoEBackwardFunction: "
                  f"{'ALL PASS' if passed else 'SOME FAILED'} ===={RESET}", flush=True)
        return passed
    finally:
        _ = ash.aclshmem_finalize()


def main():
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    passed = run_test_distributed()
    if local_pe == 0:
        print(f"[INFO] MegaMoEBackwardFunction test {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
