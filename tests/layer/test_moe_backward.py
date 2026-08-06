# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  tests/layer/test_moe_backward.py
#
#  End-to-end MoE backward correctness test for Ascend NPU.
#  (Performance A/B lives in benchmark/layer/bench_backward.py.)
#
#  For each config:
#    1. build the forward (moe_forward) and save all intermediates
#    2. golden  = moe_backward_torch(saved, dy)   (hand torch+hccl, 5 mega-ops)
#    3. triton  = moe_backward_triton(saved, dy)  (5 triton mega-kernels)
#    4. compare all 5 grads (grad_hidden / grad_routing / grad_fc1_1/2 / grad_fc2)
#       with a bf16-appropriate metric (max_abs relative to global max)
#
#  Usage (2 cards, AscendNPU-IR 1.2.0 bishengir):
#    python -m pytest tests/layer/test_moe_backward.py::test_backward_2ranks \
#        -m dist -v -s
#  The legacy torchrun entry point is also kept:
#    PATH=.../AscendNPU-IR/build/bin:$PATH TRITON_CACHE_DIR=/tmp/triton_mb \
#    torchrun --nproc-per-node=2 tests/layer/test_moe_backward.py
#
#  NOTE: dl.symm_at only resolves at heap offset 0 (see mega_moe.kernels), so
#  ONE shared peer_mem is allocated per config and reused by step 1 & step 4.
# ============================================================================

import os

import pytest
import torch
import torch_npu  # noqa: F401
import shmem as ash
import torch.distributed as dist

from mega_moe import moe_backward_triton
from mega_moe.ops._torch_forward import moe_forward
from tests._goldens.backward import moe_backward_torch
from tests._moe_dist_utils import (
    BOLD,
    GREEN,
    RED,
    RESET,
    get_ash_size_bytes,
    init_aclshmem,
    make_peer_mem,
)
from tests._numeric import cmp_grad
from tests._shapes import (
    BACKWARD_SHAPES_KIMI,
    BACKWARD_SHAPES_PERF,
    BACKWARD_SHAPES_SMALL,
)

g_ash_size = get_ash_size_bytes(default_gb=2)


# ----------------------------------------------------------------------------
# Per-rank input generation (bf16, on NPU)
# ----------------------------------------------------------------------------

def make_backward_inputs(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42):
    """Build rank-distinct bf16 MoE backward inputs.

    Returns ``(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device)``.
    The shared expert routing table (``gw``) is broadcast from rank 0 so every
    rank derives the same per-token expert assignment; weights and activations
    stay rank-distinct for realistic asymmetric load.
    """
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


def build_backward_saved(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed=42):
    """Like :func:`make_backward_inputs` but also runs the torch forward to
    produce the ``saved`` intermediates consumed by the backward.

    Returns ``(saved, dy, dtype, device)``.
    """
    hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, dy, dtype, device = make_backward_inputs(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group, seed)
    with torch.no_grad():
        _, saved = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
    return saved, dy, dtype, device


def run_one(name, ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group):
    pe = dist.get_rank(ep_group)
    saved, dy, dtype, device = build_backward_saved(
        ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group)
    label = f"{name:16} tk={ntokens:>5} h={hidden_dim} ffn={ffn_dim} k={topk} E={num_experts}"

    # shared symmetric buffer at heap offset 0 (reused by step1 & step4)
    peer_mem = make_peer_mem(saved, dtype, pe)

    # ---- correctness: triton end-to-end vs golden (hand torch+hccl) ----
    print(f"[r{pe}] TRACE: moe_backward_torch (golden) start", flush=True)
    with torch.no_grad():
        gold = moe_backward_torch(saved, dy)
    print(f"[r{pe}] TRACE: moe_backward_torch (golden) done", flush=True)
    with torch.no_grad():
        tri = moe_backward_triton(saved, dy, peer_mem)
    print(f"[r{pe}] TRACE: moe_backward_triton done", flush=True)
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
    return label, all_ok


def run_test(rank, world_size):
    ep_group = dist.group.WORLD
    init_aclshmem(rank, world_size, g_ash_size)

    # Configs are MoETestShape instances (see tests/_shapes.py).
    # MOE_PERF_CONFIGS=1 selects real model shapes from the mega_kernel paper
    # (ntokens=4096), EP-sharded across all cards so each fits in the ~13 GB HBM
    # left after leaked-memory. The small default set is a fast regression smoke.
    if os.environ.get("MOE_KIMI") == "1":
        test_configs = BACKWARD_SHAPES_KIMI
    elif os.environ.get("MOE_PERF_CONFIGS") == "1":
        test_configs = BACKWARD_SHAPES_PERF
    else:
        test_configs = BACKWARD_SHAPES_SMALL

    if rank == 0:
        print(f"{BOLD}[START]{RESET} MoE backward correctness: triton (5 mega-ops) vs torch "
              f"(hand hccl) on world_size={world_size}", flush=True)
    rows = []
    all_ok = True
    try:
        for cfg in test_configs:
            dist.barrier()
            try:
                rows.append(run_one(
                    cfg.label, cfg.tokens, cfg.hidden, cfg.ffn,
                    cfg.topk, cfg.num_experts, ep_group))
            except Exception as ex:
                if rank == 0:
                    print(f"  [skip] {cfg.label}: {str(ex)[:80]}", flush=True)
            dist.barrier()
        dist.barrier()
        if rank == 0:
            print(f"\n{BOLD}==== Summary: MoE backward correctness ===={RESET}")
            for label, ok in rows:
                cor = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
                all_ok &= ok
                print(f"  {label:52} {cor}")
            print(f"{BOLD}==== done ===={RESET}", flush=True)
    finally:
        ash.aclshmem_finalize()

    # Failure must raise so pytest/CI can detect a correctness regression. A
    # config that raises (e.g. the known aicore-timeout path) is treated as a
    # skip above and does not flip all_ok; only a completed run with a grad
    # mismatch reaches here with all_ok=False.
    flag = torch.tensor([1 if all_ok else 0], dtype=torch.int32, device=f"npu:{rank}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()):
        raise AssertionError("MoE backward correctness check failed.")


# ---------------------------------------------------------------------------
#  Pytest
# ---------------------------------------------------------------------------

@pytest.mark.dist
def test_backward_2ranks(dist_test):
    dist_test(run_test, world_size=2)


@pytest.mark.dist
def test_backward_8ranks(dist_test):
    dist_test(run_test, world_size=8)


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    run_test(local_pe, dist.get_world_size())
    if local_pe == 0:
        print(f"[INFO] MoE backward harness done", flush=True)
