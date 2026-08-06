# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  benchmark/layer/bench_backward.py
#
#  MoE backward performance A/B: fused triton 5-mega-op backward vs the hand
#  torch+HCCL golden, end-to-end. Reports the slowest rank's per-call latency
#  and the triton/torch speedup, mirroring benchmark/layer/bench_full_forward.py
#  (pytest + dist_test entry; correctness lives in tests/layer/test_moe_backward.py).
#
#  Config selection reuses config/_shapes.py:
#    MOE_KIMI=1            -> BACKWARD_SHAPES_KIMI
#    MOE_PERF_CONFIGS=1    -> BACKWARD_SHAPES_PERF (all real model shapes)
#    MOE_PERF_CONFIGS=lbl  -> only perf shapes whose model label matches
#                             (comma-separated, case-insensitive; see select_perf_shapes)
#    (otherwise)           -> BACKWARD_SHAPES_SMALL (regression smoke)
#
#  The triton wgrad kernels are pathologically slow on some shapes; set
#    MOE_FC1_WGRAD_TORCH=1 MOE_FC2_WGRAD_TORCH=1
#  to fall back to torch wgrad (see src/mega_moe/ops/backward.py).
#
#  Usage:
#    python -m pytest benchmark/layer/bench_backward.py::test_bench_backward_2ranks \
#        -m dist -v -s
#  Legacy torchrun entry:
#    PATH=.../AscendNPU-IR/build/bin:$PATH TRITON_CACHE_DIR=/tmp/triton_mb \
#    torchrun --nproc-per-node=2 benchmark/layer/bench_backward.py
# ============================================================================

import json
import os
from pathlib import Path

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
    bench,
    get_ash_size_bytes,
    init_aclshmem,
    make_peer_mem,
)
from config import (
    BACKWARD_SHAPES_KIMI,
    BACKWARD_SHAPES_KIMI_SMALL,
    BACKWARD_SHAPES_SMALL,
    rank_size,
    select_perf_shapes,
)

g_ash_size = get_ash_size_bytes(default_gb=2)

BENCH_WARMUP = 5
BENCH_ITERS = 20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(os.environ.get(
    "MOE_BACKWARD_BENCH_RESULTS_DIR", str(PROJECT_ROOT / "results" / "backward")))


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


def _select_shapes():
    if os.environ.get("MOE_KIMI") == "1":
        return BACKWARD_SHAPES_KIMI
    if (perf := os.environ.get("MOE_PERF_CONFIGS")):
        return select_perf_shapes(perf)
    # When RANK_SIZE is set, default to the matching Kimi-K3 variant
    # (small at 2 cards, full at 8); otherwise keep the tiny smoke default.
    if "RANK_SIZE" in os.environ:
        return BACKWARD_SHAPES_KIMI_SMALL if rank_size() == 2 else BACKWARD_SHAPES_KIMI
    return BACKWARD_SHAPES_SMALL


def run_one_bench(shape, ep_group):
    pe = dist.get_rank(ep_group)
    saved, dy, dtype, device = build_backward_saved(
        shape.tokens, shape.hidden, shape.ffn, shape.num_experts, shape.topk, ep_group)
    label = f"{shape.label:16} tk={shape.tokens:>5} h={shape.hidden} ffn={shape.ffn} k={shape.topk} E={shape.num_experts}"

    peer_mem = make_peer_mem(saved, dtype, pe)

    def _triton():
        with torch.no_grad():
            moe_backward_triton(saved, dy, peer_mem)

    def _torch():
        with torch.no_grad():
            moe_backward_torch(saved, dy)

    tri_ms = bench(_triton, BENCH_WARMUP, BENCH_ITERS, ep_group)
    torch_ms = bench(_torch, BENCH_WARMUP, BENCH_ITERS, ep_group)
    sp = torch_ms / tri_ms if tri_ms > 0 else float("inf")

    ash.aclshmem_free_tensor(peer_mem)

    if pe == 0:
        faster = tri_ms <= torch_ms
        tag = f"{GREEN}triton faster{RESET}" if faster else f"{RED}torch faster{RESET}"
        print(f"  {label}  | torch={torch_ms:7.3f}  triton={tri_ms:7.3f}ms  "
              f"| triton/torch={sp:5.2f}x  | {tag}", flush=True)
    return label, torch_ms, tri_ms, sp


def _save_results(entries, world_size):
    """Write the per-config latency JSON to results/backward/ (rank 0 only)."""
    if dist.get_rank() != 0:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("MOE_BENCH_TAG", "default")
    out = {
        "world_size": world_size,
        "warmup": BENCH_WARMUP,
        "iters": BENCH_ITERS,
        "wgrad_torch_fallback": bool(
            os.environ.get("MOE_FC1_WGRAD_TORCH") or os.environ.get("MOE_FC2_WGRAD_TORCH")),
        "configs": [
            {"label": label, "torch_ms": t_ms, "triton_ms": r_ms, "triton_over_torch": sp}
            for label, t_ms, r_ms, sp in entries
        ],
    }
    path = RESULTS_DIR / f"bench_backward_{tag}_w{world_size}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  [saved] {path}", flush=True)


def run_benchmark(rank, world_size):
    ep_group = dist.group.WORLD
    init_aclshmem(rank, world_size, g_ash_size)
    shapes = _select_shapes()

    if rank == 0:
        print(f"{BOLD}[START]{RESET} MoE backward perf A/B: triton (5 mega-ops) vs torch "
              f"(hand hccl) on world_size={world_size}, "
              f"warmup={BENCH_WARMUP} iters={BENCH_ITERS}", flush=True)
    entries = []
    try:
        for shape in shapes:
            dist.barrier()
            try:
                entries.append(run_one_bench(shape, ep_group))
            except Exception as ex:
                if rank == 0:
                    print(f"  [skip] {shape.label}: {str(ex)[:80]}", flush=True)
            dist.barrier()
        dist.barrier()
        if rank == 0:
            print(f"\n{BOLD}==== Summary: triton vs torch (MoE backward perf) ===={RESET}")
            print(f"  {'config':52} {'torch(ms)':>9} {'triton(ms)':>10} {'triton/torch':>12}")
            sps = []
            for label, t_ms, r_ms, sp in entries:
                sps.append(sp)
                print(f"  {label:52} {t_ms:>9.3f} {r_ms:>10.3f} {sp:>10.2f}x")
            if sps:
                print(f"\n  avg triton/torch speedup: {sum(sps)/len(sps):.2f}x over {len(sps)} configs")
            print(f"{BOLD}==== done ===={RESET}", flush=True)
        _save_results(entries, world_size)
    finally:
        ash.aclshmem_finalize()


# ---------------------------------------------------------------------------
#  Pytest
# ---------------------------------------------------------------------------

@pytest.mark.dist
def test_bench_backward_2ranks(dist_test):
    dist_test(run_benchmark, world_size=2)


@pytest.mark.dist
def test_bench_backward_8ranks(dist_test):
    dist_test(run_benchmark, world_size=8)


@pytest.mark.dist
def test_bench_backward(dist_test):
    # RANK_SIZE (2 or 8, default 8) sets world_size; _select_shapes then defaults
    # to Kimi-K3-small (2 cards, 128 experts) or full Kimi-K3 (8 cards, 896).
    dist_test(run_benchmark, world_size=rank_size())


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    run_benchmark(local_pe, dist.get_world_size())
    if local_pe == 0:
        print(f"[INFO] MoE backward bench done", flush=True)
