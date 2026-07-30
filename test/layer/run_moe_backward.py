# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  test/layer/run_moe_backward.py
#
#  End-to-end MoE backward harness for Ascend NPU: correctness + performance.
#
#  For each config:
#    1. build the forward (moe_forward) and save all intermediates
#    2. golden  = moe_backward_torch(saved, dy)   (hand torch+hccl, 5 mega-ops)
#    3. triton  = moe_backward_triton(saved, dy)  (5 triton mega-kernels)
#    4. compare all 5 grads (grad_hidden / grad_routing / grad_fc1_1/2 / grad_fc2)
#       with a bf16-appropriate metric (max_abs relative to global max)
#    5. perf A/B: _bench(triton end-to-end) vs _bench(torch end-to-end), report
#       the slowest rank's latency + speedup, 06-style summary table
#
#  Usage (2 cards, AscendNPU-IR 1.2.0 bishengir):
#    PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
#    TRITON_CACHE_DIR=/tmp/triton_mb \
#    torchrun --nproc-per-node=2 test/layer/run_moe_backward.py
#
#  NOTE: dl.symm_at only resolves at heap offset 0 (see benchmark/kernel), so
#  ONE shared peer_mem is allocated per config and reused by step 1 & step 4.
# ============================================================================

import os
import sys
import time

# make the repo root importable (benchmark.*, functions.*) when launched via torchrun
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch_npu  # noqa: F401
import shmem as ash
import torch.distributed as dist

from benchmark.moe_backward_golden import moe_forward, moe_backward_torch
from functions.moe_backward import moe_backward_triton

g_ash_size = 512 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

BENCH_WARMUP = 5
BENCH_ITERS = 20


def _cmp(name, tri, gold, rtol=2e-2, atol=1e-2):
    tri = tri.float(); gold = gold.float()
    d = (tri - gold).abs()
    max_d = float(d.max().item())
    gmax = float(gold.abs().max().item())
    n_bad = int((d > atol + rtol * gmax).sum().item())
    ok = n_bad == 0
    return ok, max_d, max_d / (gmax + 1e-9), n_bad


def _bench(fn, warmup, iters, ep_group):
    """Warm up `fn` then time it over `iters` runs. Returns the slowest rank's
    average per-call latency in ms (collective → slowest rank is the truth)."""
    for _ in range(warmup):
        fn()
    dist.barrier(ep_group)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    local_ms = (t1 - t0) / iters * 1000.0
    t = torch.tensor([local_ms], device=f"npu:{dist.get_rank(ep_group)}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ep_group)
    return float(t.item())


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
    with torch.no_grad():
        _, saved = moe_forward(hs, topk_w, topk_idx, fc1_1, fc1_2, fc2, ep_group, topk, return_saved=True)
    return saved, dy, dtype, device


def run_one(ntokens, hidden_dim, ffn_dim, topk, num_experts, ep_group):
    pe = dist.get_rank(ep_group)
    saved, dy, dtype, device = _build_inputs(ntokens, hidden_dim, ffn_dim, num_experts, topk, ep_group)
    label = f"tk={ntokens:>5} h={hidden_dim} ffn={ffn_dim} k={topk}"

    # shared symmetric buffer at heap offset 0 (reused by step1 & step4)
    peer_elems = max(saved["total_recv"], saved["total_send"]) * saved["hidden_dim"]
    peer_mem = ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=pe)

    # ---- correctness: triton end-to-end vs golden (hand torch+hccl) ----
    with torch.no_grad():
        gold = moe_backward_torch(saved, dy)
        tri = moe_backward_triton(saved, dy, peer_mem)
    checks = ["grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2"]
    all_ok = True
    detail = []
    for n in checks:
        ok, mx, rel, nbad = _cmp(n, tri[n], gold[n])
        all_ok &= ok
        detail.append(f"{n}:{GREEN}PASS{RESET}" if ok else f"{n}:{RED}FAIL{RESET}({nbad},mx={mx:.1e})")

    # ---- perf A/B: triton end-to-end vs torch end-to-end ----
    def _triton():
        with torch.no_grad():
            moe_backward_triton(saved, dy, peer_mem)

    def _torch():
        with torch.no_grad():
            moe_backward_torch(saved, dy)

    tri_ms = _bench(_triton, BENCH_WARMUP, BENCH_ITERS, ep_group)
    torch_ms = _bench(_torch, BENCH_WARMUP, BENCH_ITERS, ep_group)
    sp = torch_ms / tri_ms if tri_ms > 0 else float("inf")

    ash.aclshmem_free_tensor(peer_mem)

    if pe == 0:
        cor = f"{GREEN}ALL PASS{RESET}" if all_ok else f"{RED}SOME FAIL{RESET}"
        print(f"  {label}  | torch={torch_ms:7.3f}  triton={tri_ms:7.3f}ms  "
              f"| triton/torch={sp:5.2f}x  | {cor}", flush=True)
        if not all_ok:
            print("    " + "  ".join(detail), flush=True)
    return label, torch_ms, tri_ms, sp, all_ok


def run_test_distributed():
    pe = dist.get_rank(); ep_group = dist.group.WORLD
    ash.set_conf_store_tls(False, "")
    attr = ash.InitAttr()
    attr.my_rank = pe; attr.n_ranks = dist.get_world_size(); attr.local_mem_size = g_ash_size
    attr.ip_port = G_IP_PORT; attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    assert ash.aclshmem_init(attr) == 0

    num_experts = 128
    # Small configs (always fit). Enable the Qwen3-30B-A3B perf configs below
    # once the NPU is fully free (leaked memory from crashed runs must be cleared
    # by a reboot — `npu-smi reset` fails on this box).
    if os.environ.get("MOE_PERF_CONFIGS") == "1":
        # Qwen3-30B-A3B perf configs — need a fully-free NPU.
        test_configs = [
            (2048,  2048, 768, 8),
            (8192,  2048, 768, 8),
            (16384, 2048, 768, 8),
            (32768, 2048, 768, 8),
        ]
    else:
        test_configs = [
            (512,  512, 256, 4),
            (1024, 512, 256, 4),
            (2048, 512, 256, 4),
            (512,  1024, 512, 8),
        ]

    if pe == 0:
        print(f"{BOLD}[START]{RESET} MoE backward end-to-end: triton (5 mega-ops) vs torch "
              f"(hand hccl) on world_size={dist.get_world_size()}, "
              f"warmup={BENCH_WARMUP} iters={BENCH_ITERS}", flush=True)
    rows = []
    try:
        for cfg in test_configs:
            dist.barrier()
            rows.append(run_one(*cfg, num_experts, ep_group))
        dist.barrier()
        if pe == 0:
            print(f"\n{BOLD}==== Summary: triton vs torch (MoE backward end-to-end) ===={RESET}")
            print(f"  {'config':24} {'torch(ms)':>9} {'triton(ms)':>10} {'triton/torch':>12}  correct")
            sps = []
            for label, t_ms, r_ms, sp, ok in rows:
                cor = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
                sps.append(sp)
                print(f"  {label:24} {t_ms:>9.3f} {r_ms:>10.3f} {sp:>10.2f}x  {cor}")
            if sps:
                print(f"\n  avg triton/torch speedup: {sum(sps)/len(sps):.2f}x over {len(sps)} configs")
            print(f"{BOLD}==== done ===={RESET}", flush=True)
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    print(f"[INFO] Rank {local_pe} of {dist.get_world_size()} initialised", flush=True)
    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print(f"[INFO] MoE backward harness done", flush=True)
