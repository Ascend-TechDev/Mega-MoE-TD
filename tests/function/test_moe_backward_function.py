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
#    python -m pytest tests/function/test_moe_backward_function.py::test_backward_function_2ranks \
#        -m dist -v -s
#  The legacy torchrun entry point is also kept:
#    PATH=.../AscendNPU-IR/build/bin:$PATH TRITON_CACHE_DIR=/tmp/triton_mb \
#    torchrun --nproc-per-node=2 tests/function/test_moe_backward_function.py
# ============================================================================

import os

import pytest
import torch
import torch_npu  # noqa: F401
import shmem as ash
import torch.distributed as dist

from mega_moe import MegaMoEBackwardFunction
from mega_moe.ops._torch_forward import moe_forward
from tests._goldens.backward import moe_backward_torch
from tests._moe_dist_utils import (
    BOLD,
    GREEN,
    RED,
    RESET,
    get_ash_size_bytes,
    init_aclshmem,
    make_backward_inputs,
    make_peer_mem,
)
from tests._numeric import cmp_grad
from tests._shapes import BACKWARD_FUNCTION_SHAPES

g_ash_size = get_ash_size_bytes(default_gb=1)


def run_one(ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group):
    pe = dist.get_rank(ep_group)
    hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device = make_backward_inputs(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group)
    label = f"tk={ntokens:>5} h={hidden_dim} ffn={ffn_dim} k={topk}"

    # shared symmetric buffer at heap offset 0 (reused by step1 & step4)
    # size from a no_grad forward probe
    with torch.no_grad():
        _, probe = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
    peer_mem = make_peer_mem(probe, dtype, pe)

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
        ok, mx, rel, nbad = cmp_grad(n, tri[n], gold[n])
        all_ok &= ok
        detail.append(f"{n}:{GREEN}PASS{RESET}" if ok else f"{n}:{RED}FAIL{RESET}({nbad},mx={mx:.1e})")

    if pe == 0:
        cor = f"{GREEN}ALL PASS{RESET}" if all_ok else f"{RED}SOME FAIL{RESET}"
        print(f"  {label}  | {cor}", flush=True)
        if not all_ok:
            print("    " + "  ".join(detail), flush=True)
    return all_ok


def run_test(rank, world_size):
    ep_group = dist.group.WORLD
    init_aclshmem(rank, world_size, g_ash_size)

    test_configs = BACKWARD_FUNCTION_SHAPES

    if rank == 0:
        print(f"{BOLD}[START]{RESET} MegaMoEBackwardFunction (autograd) vs golden "
              f"on world_size={world_size}", flush=True)
    results = []
    try:
        for cfg in test_configs:
            dist.barrier()
            results.append(run_one(
                cfg.tokens, cfg.hidden, cfg.ffn, cfg.topk, cfg.num_experts, ep_group))
        dist.barrier()
        if rank == 0:
            print(f"\n{BOLD}==== MegaMoEBackwardFunction: "
                  f"{'ALL PASS' if all(results) else 'SOME FAILED'} ===={RESET}", flush=True)
    finally:
        ash.aclshmem_finalize()

    # Failure must raise so pytest/CI can detect a correctness regression.
    all_ok = bool(results) and all(results)
    flag = torch.tensor([1 if all_ok else 0], dtype=torch.int32, device=f"npu:{rank}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise AssertionError("MegaMoEBackwardFunction correctness check failed.")


# ---------------------------------------------------------------------------
#  Pytest
# ---------------------------------------------------------------------------

@pytest.mark.dist
def test_backward_function_2ranks(dist_test):
    dist_test(run_test, world_size=2)


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    run_test(local_pe, dist.get_world_size())
    if local_pe == 0:
        print(f"[INFO] MegaMoEBackwardFunction test done", flush=True)
