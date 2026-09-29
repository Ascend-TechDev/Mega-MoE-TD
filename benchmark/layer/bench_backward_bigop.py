# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Numerical gates and full backward latency against the bigop compute golden.

Run from the UDMA environment with bigop on PYTHONPATH. Uses the registered
backward saved-state builder (SwiGLU, no MoonEP); this is not a measurement of
single-kernel-forward recomputation. With --moonep, use the production
single-kernel forward and time recompute backward with UDMA gradient
transport, including lending, re-prefetch and persistent-state handling.
Both implementations redispatch inputs and recompute weighted SwiGLU.

Every timed call includes wrapper preparation, allocation, communication and
all five gradients. Samples are host-wall latency, MAX across ranks, with an
alternating implementation order. The initial correctness gate compares every
gradient to the Torch reference using the repository's global-max tolerance,
and separately records elementwise error counts and explicit finiteness.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from benchmark.layer._npu_occupancy import check_npu_occupancy
from config import resolve_case
from tests import _moe_testkit as kit
from tests._moe_baselines import build_backward_saved

GRAD_KEYS = (
    "grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2",
)
RTOL, ATOL = 2e-2, 1e-2


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


@contextmanager
def report_failure(output_dir, rank):
    # Report before ACLSHMEM/HCCL teardown, which can wait on a healthy peer
    # still inside a collective after one rank ran out of memory.
    try:
        yield
    except BaseException:
        write_json(output_dir / f"failure_rank{rank}.json", {"traceback": traceback.format_exc()})
        raise


def compare_gradient(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "error": "shape or dtype mismatch"}
    # A full Kimi weight converted to FP32 is several GB; stream comparisons
    # over experts (or token chunks), including non-contiguous FC1 views.
    maxima = torch.zeros(2, device=actual.device)
    sums = torch.zeros(4, device=actual.device)
    step = 1 if actual.ndim == 3 else 256
    for start in range(0, actual.shape[0], step):
        a = actual[start:start + step].float()
        b = expected[start:start + step].to(actual.device).float()
        delta = (a - b).abs()
        finite = torch.isfinite(a) & torch.isfinite(b)
        maxima = torch.maximum(maxima, torch.stack((delta.max(), b.abs().max())))
        sums += torch.stack((
            (~finite).float().sum(),
            ((delta > ATOL + RTOL * b.abs()) | ~finite).float().sum(),
            delta.square().sum(), b.square().sum(),
        ))
    max_abs, reference_max = maxima.cpu().tolist()
    nonfinite, elementwise_bad, squared_error, squared_ref = sums.cpu().tolist()
    return {
        "passed": nonfinite == 0 and math.isfinite(max_abs)
        and max_abs <= ATOL + RTOL * reference_max,
        "shape": list(actual.shape), "dtype": str(actual.dtype),
        "max_abs": max_abs, "reference_max": reference_max,
        "global_max_relative_error": max_abs / max(reference_max, 1e-30),
        "relative_l2": math.sqrt(squared_error / max(squared_ref, 1e-30)),
        "nonfinite": int(nonfinite), "elementwise_bad": int(elementwise_bad),
        "elements": actual.numel(),
    }


def backward_reference(saved, dy):
    """The existing Torch golden, with only its pointwise step chunked.

    The unchunked FP32 activation temporaries exceed HBM at Kimi T16K.
    Grouped GEMMs and collectives are unchanged; chunking rows changes no
    reduction boundary. This function is never used as a timed baseline.
    """
    from tests import _moe_baselines as oracle

    gco = oracle.combine_bwd_a2a(dy.to(saved["output"].dtype), saved)
    grad_swiglu = oracle.fc2_input_grad(gco, saved)
    grad_fc1_output = torch.empty_like(saved["fc1_output"])
    grad_gate = torch.empty_like(saved["recv_weights_sorted"])
    for start in range(0, grad_swiglu.shape[0], 2048):
        rows = slice(start, start + 2048)
        part = dict(saved)
        for key in ("gate", "up", "recv_weights_sorted"):
            part[key] = saved[key][rows]
        grad_fc1_output[rows], grad_gate[rows] = oracle.swiglu_bwd(grad_swiglu[rows], part)
    del grad_swiglu
    grad_fc2 = oracle.fc2_weight_grad(gco, saved)
    del gco
    grad_recv = oracle.fc1_input_grad(grad_fc1_output, saved)
    grad_hidden, grad_routing = oracle.dispatch_bwd(grad_recv, grad_gate, saved)
    del grad_recv, grad_gate
    grad_fc1_1, grad_fc1_2, _ = oracle.fc1_weight_grad(grad_fc1_output, saved)
    return dict(grad_hidden=grad_hidden, grad_routing_weights=grad_routing,
                grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2)


def prepare_call(fn):
    prepare = getattr(fn, "prepare", None)
    if prepare is not None:
        prepare()


def correctness_gate(functions, saved, dy, device, output_dir, rank,
                     reference_cpu=False, repeats=1):
    if rank == 0:
        print("[gate] building Torch reference", flush=True)
    reference = backward_reference(saved, dy)
    reference_bytes = sum(value.numel() * value.element_size() for value in reference.values())
    if reference_cpu or reference_bytes > 7 * 1024**3:
        # W4 full weights alone hold ~14 GiB of gradients. Keep the oracle
        # on the host during comparison so bigop can materialize its output
        # transpose; neither offload nor comparisons enter the timed region.
        reference = {key: value.cpu() for key, value in reference.items()}
        torch.npu.empty_cache()
    report = {}
    history = []
    for repetition in range(repeats):
        for name, fn in functions.items():
            if rank == 0:
                print(f"[gate {repetition + 1}/{repeats}] {name}", flush=True)
            prepare_call(fn)
            actual = fn()
            report[name] = {
                key: compare_gradient(actual[key], reference[key]) for key in GRAD_KEYS
            }
            del actual
            write_json(output_dir / f"correctness_rank{rank}.json", report)
            ok = all(row["passed"] for row in report[name].values())
            flag = torch.tensor([int(ok)], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            if not flag.item():
                raise AssertionError(f"{name} gradient gate failed; see correctness_rank*.json")
        history.append(dict(report))
        write_json(output_dir / f"correctness_repeats_rank{rank}.json", history)
    del reference
    gc.collect()
    torch.npu.empty_cache()


def stats(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min_ms": ordered[0], "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def worker(rank, case_id, output_dir, warmup, iterations, compare_safe, phase_samples,
           moonep=False, compare_reprefetch=False, reprefetch_profile_samples=0,
           compare_reprefetch_fused=False, compare_reprefetch_sync=False,
           correctness_repeats=1, check_saved_kernel=False, add_unused_replica=False,
           compare_reprefetch_overlap=False):
    from mega_moe import moe_backward_triton
    from mega_moe._goldens.bigop_ref import moe_backward_bigop

    case = resolve_case(case_id).validate()
    output_dir = Path(output_dir)
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=case.world_size)
    device = f"npu:{rank}"
    try:
        # Materialize HCCL before the symmetric heap, as in the forward runner.
        bootstrap = torch.zeros(case.world_size, device=device)
        dist.all_reduce(bootstrap)
        dist.all_to_all_single(torch.empty_like(bootstrap), bootstrap)
        with kit.aclshmem_session(
            rank, case.world_size, kit.get_ash_size_bytes(), enable_udma=True,
        ), report_failure(output_dir, rank), torch.no_grad():
            if rank == 0:
                print(f"[setup] {case_id}", flush=True)
            native_case = None
            if moonep:
                from benchmark.layer._backward_moonep_case import SingleKernelMoonEPCase
                native_case = SingleKernelMoonEPCase(
                    case, rank, output_dir, add_unused_replica=add_unused_replica)
                saved, dy = native_case.saved, native_case.dy
                dtype = dy.dtype
                peer_mem = native_case.peer_mem
            else:
                saved, dy, dtype, _ = build_backward_saved(
                    case.tokens, case.hidden, case.ffn, case.num_experts, case.topk,
                    dist.group.WORLD,
                )
                peer_mem = kit.make_peer_mem(saved, dtype, rank)
            counts = saved["expert_counts"].cpu().tolist()
            write_json(output_dir / f"routing_rank{rank}.json", {
                "counts": counts, "min": min(counts), "max": max(counts),
                "sum": sum(counts),
            })
            try:
                def candidate():
                    os.environ["MOE_MEGA_SAFE_WGRAD"] = "0"
                    return moe_backward_triton(saved, dy, peer_mem)

                def safe_wgrad():
                    os.environ["MOE_MEGA_SAFE_WGRAD"] = "1"
                    return moe_backward_triton(saved, dy, peer_mem)

                transport = os.environ.get("MOE_MEGA_REPREFETCH_TRANSPORT", "store")
                # Removing the two cache keys makes accidental reuse by the
                # recompute baseline fail, instead of silently biasing timing.
                bigop_saved = ({key: value for key, value in saved.items()
                                if key not in ("recv_hidden_sorted", "swiglu_out_weighted")}
                               if moonep else saved)
                functions = {"candidate": native_case.variant(transport) if moonep else candidate,
                             "bigop": lambda: moe_backward_bigop(
                                 bigop_saved, dy, recompute=moonep,
                                 hidden_states=native_case.hidden if moonep else None)}
                if compare_reprefetch_overlap:
                    functions["candidate"].trace_kernel = True
                    functions["reprefetch_preloaded"] = native_case.variant(
                        "udma_fused_pipeline",
                        fine_sync=functions["candidate"].fine_sync, preloaded=True)
                    functions["reprefetch_udma_standalone"] = native_case.variant(
                        "udma_standalone", trace_kernel=True)
                    functions["reprefetch_udma_fused_barrier"] = native_case.variant(
                        "udma_fused_barrier", trace_kernel=True)
                if compare_reprefetch_sync:
                    functions["reprefetch_quiet_b2"] = native_case.variant(
                        "udma_fused_pipeline", fine_sync=False)
                if compare_reprefetch:
                    other_transport = "udma" if transport == "store" else "store"
                    functions[f"reprefetch_{other_transport}"] = native_case.variant(other_transport)
                if compare_reprefetch_fused:
                    control = ("udma_standalone" if functions["candidate"].fused
                               else "udma_fused_barrier")
                    functions[f"reprefetch_{control}"] = native_case.variant(control)
                    pipeline_control = ("udma_fused_barrier" if functions["candidate"].pipeline
                                        else "udma_fused_pipeline")
                    functions[f"reprefetch_{pipeline_control}"] = native_case.variant(pipeline_control)
                if compare_safe:
                    functions["safe_wgrad"] = native_case.variant(transport, safe_wgrad=True) if moonep else safe_wgrad
                gate_functions = dict(functions)
                if check_saved_kernel:
                    candidate_variant = functions["candidate"]
                    gate_functions["saved_kernel"] = native_case.variant(
                        "udma_fused_pipeline", fine_sync=candidate_variant.fine_sync,
                        recompute=False)
                correctness_gate(gate_functions, saved, dy, device, output_dir, rank,
                                 reference_cpu=moonep, repeats=correctness_repeats)
                for fn in functions.values():
                    for _ in range(warmup):
                        prepare_call(fn)
                        fn()
                torch.npu.synchronize()
                samples = {name: [] for name in functions}
                local_samples = {name: [] for name in functions}
                if rank == 0:
                    print("[timing] correctness passed for every rank", flush=True)
                for iteration in range(iterations):
                    names = list(functions)
                    if iteration % 2:
                        names.reverse()
                    for name in names:
                        prepare_call(functions[name])
                        torch.npu.synchronize()
                        dist.barrier()
                        torch.npu.synchronize()
                        start = time.perf_counter_ns()
                        functions[name]()
                        torch.npu.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / 1e6
                        local_samples[name].append(elapsed)
                        value = torch.tensor([elapsed], device=device)
                        dist.all_reduce(value, op=dist.ReduceOp.MAX)
                        samples[name].append(value.item())
                write_json(output_dir / f"samples_rank{rank}.json", local_samples)
                if compare_reprefetch_overlap:
                    hashes = {name: fn.kernel_hash for name, fn in functions.items()
                              if getattr(fn, "trace_kernel", False)}
                    if functions["candidate"].fine_sync:
                        early_hashes = {key: functions[key].early_kernel_hash
                                        for key in ("candidate", "reprefetch_preloaded")}
                        write_json(output_dir / f"overlap_early_binary_rank{rank}.json", early_hashes)
                        if (not early_hashes["candidate"] or early_hashes["candidate"]
                                != early_hashes["reprefetch_preloaded"]):
                            raise AssertionError("preload must retain the same early-launch binary")
                    write_json(output_dir / f"overlap_binary_rank{rank}.json", hashes)
                    if (not hashes["candidate"] or hashes["candidate"]
                            != hashes["reprefetch_preloaded"]):
                        raise AssertionError("preloaded and candidate must share a compiled kernel")
                if rank == 0:
                    results = {name: stats(values) for name, values in samples.items()}
                    speedup = results["bigop"]["median_ms"] / results["candidate"]["median_ms"]
                    write_json(output_dir / "benchmark_result.json", {
                        "case": case_id, "scope": ("single-kernel SwiGLU MoonEP recompute backward with reprefetch and UDMA gradient transport"
                                                   if moonep else "saved SwiGLU backward without MoonEP"),
                        "protocol": {"clock": "host_wall", "rank_reduction": "MAX",
                                     "warmup": warmup, "samples": iterations,
                                     "order": "alternating", "full_wrapper": True,
                                     "injected_unused_replica": add_unused_replica,
                                     "recompute": {name: moonep for name in functions},
                                     "recompute_scope": (["dispatch_input", "weighted_swiglu"]
                                                         if moonep else []),
                                     "reprefetch_transports": {name: fn.transport for name, fn in functions.items()
                                                               if hasattr(fn, "transport")},
                                     "reprefetch_fused": {name: fn.fused for name, fn in functions.items()
                                                         if hasattr(fn, "fused")},
                                     "reprefetch_pipeline": {name: fn.pipeline for name, fn in functions.items()
                                                            if hasattr(fn, "pipeline")},
                                     "reprefetch_fine_sync": {name: fn.fine_sync for name, fn in functions.items()
                                                             if hasattr(fn, "fine_sync")},
                                     "reprefetch_preloaded": {name: fn.preloaded for name, fn in functions.items()
                                                             if hasattr(fn, "preloaded")},
                                     "safe_wgrad": {name: fn.safe_wgrad for name, fn in functions.items()
                                                    if hasattr(fn, "safe_wgrad")}},
                        "correctness": {"status": "passed_all_ranks", "rtol": RTOL,
                                        "atol": ATOL, "rule": "global_max", "finite": True,
                                        "repeats": correctness_repeats,
                                        "saved_kernel_checked": check_saved_kernel},
                        "timings": results, "speedup_vs_bigop": speedup,
                        "speedups_vs_bigop": {name: results["bigop"]["median_ms"] / row["median_ms"]
                                              for name, row in results.items() if name != "bigop"},
                        "occupancy_gate": {"status": "pending"},
                    })
                    print(json.dumps({"speedup": speedup, "median_ms": {
                        name: result["median_ms"] for name, result in results.items()
                    }}), flush=True)
                if phase_samples:
                    from benchmark.layer._backward_phase_capture import capture_phases
                    capture_phases(functions, saved, dy, device, output_dir, rank, phase_samples)
                if reprefetch_profile_samples:
                    from benchmark.layer._backward_phase_capture import capture_reprefetch
                    capture_reprefetch(functions, device, output_dir, rank, reprefetch_profile_samples)
                    from benchmark.layer._reprefetch_component_capture import capture_components
                    capture_components(functions["candidate"], output_dir, rank, reprefetch_profile_samples)
            finally:
                torch.npu.synchronize()
                if native_case is not None:
                    native_case.close()
                kit.ash.aclshmem_free_tensor(peer_mem)
    finally:
        gc.collect()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compare-safe-wgrad", action="store_true")
    parser.add_argument("--moonep", action="store_true",
                        help="same-input single-kernel MoonEP with reprefetch (W8)")
    parser.add_argument("--compare-reprefetch", action="store_true",
                        help="compare store and UDMA reprefetch under the same backward contract")
    parser.add_argument("--compare-reprefetch-fused", action="store_true",
                        help="compare pipelined, serialized fused, and standalone UDMA reprefetch")
    parser.add_argument("--compare-reprefetch-sync", action="store_true",
                        help="compare selected fine sync with conservative quiet+B2")
    parser.add_argument("--compare-reprefetch-overlap", action="store_true",
                        help="diagnostic: compare serial refresh and ideal preloaded weights using the same pipeline binary")
    parser.add_argument("--correctness-repeats", type=int, default=1,
                        help="repeat every correctness variant on reused buffers/epochs")
    parser.add_argument("--check-saved-kernel", action="store_true",
                        help="also gate the ordinary kernel with untimed prepared caches")
    parser.add_argument("--add-unused-replica", action="store_true",
                        help="diagnostic: prefetch one extra zero-token replica per rank")
    parser.add_argument("--phase-samples", type=int, default=0)
    parser.add_argument("--reprefetch-profile-samples", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    case = resolve_case(args.case).validate()
    if (case.direction != "backward" or args.warmup < 0 or args.iterations <= 0
            or args.phase_samples < 0 or args.reprefetch_profile_samples < 0
            or args.correctness_repeats < 1):
        parser.error("requires a backward case, warmup >= 0 and iterations > 0")
    if args.moonep:
        if case.world_size != 8:
            parser.error("--moonep requires W8")
        if os.environ.get("MOE_MEGA_GRAD_TRANSPORT", "udma") != "udma":
            parser.error("--moonep uses an MTE|UDMA session and requires UDMA push")
        os.environ["MOE_MEGA_GRAD_TRANSPORT"] = "udma"
        os.environ["MOE_MEGA_REPREFETCH"] = "1"
        os.environ["MOE_SAVED_RECOMPUTE"] = "1"
        os.environ["MEGAMOE_REPLICA_POOL"] = "1"
        os.environ.setdefault("MOE_MEGA_REPREFETCH_TRANSPORT", "udma")
    elif (args.compare_reprefetch or args.reprefetch_profile_samples
          or args.compare_reprefetch_fused or args.compare_reprefetch_sync
          or args.compare_reprefetch_overlap
          or args.check_saved_kernel or args.add_unused_replica):
        parser.error("reprefetch comparison/profiling requires --moonep")
    if (args.compare_reprefetch_sync or args.compare_reprefetch_overlap
            or args.check_saved_kernel) and (
        os.environ.get("MOE_MEGA_REPREFETCH_TRANSPORT") != "udma"
        or os.environ.get("MOE_MEGA_REPREFETCH_FUSED", "1") != "1"
        or os.environ.get("MOE_MEGA_REPREFETCH_PIPELINE", "0") != "1"
    ):
        parser.error("pipeline comparisons require fused UDMA and explicit PIPELINE=1")
    if args.compare_reprefetch_sync and os.environ.get("MOE_MEGA_REPREFETCH_FINE_SYNC", "0") != "1":
        parser.error("fine-sync comparison requires explicit FINE_SYNC=1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MOE_BWD_MEGA"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(40000 + os.getpid() % 20000))
    base = 20000 + (os.getpid() % 200) * 64
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", f"{base}-{base + 63}")
    os.environ.setdefault("ASH_MASTER_PORT", str(10000 + os.getpid() % 9000))
    import bigop
    import mega_moe
    import triton
    files = [Path(__file__), ROOT / "src/mega_moe/kernels/mega_bwd.py",
             ROOT / "src/mega_moe/kernels/dispatch_fc2_bwd.py",
             ROOT / "src/mega_moe/kernels/combine_fc1_bwd.py",
             ROOT / "src/mega_moe/_goldens/bigop_ref.py", Path(bigop.__file__), ROOT / "config/_shapes.py"]
    if args.phase_samples or args.reprefetch_profile_samples:
        files.append(ROOT / "benchmark/layer/_backward_phase_capture.py")
    if args.moonep:
        files.extend(ROOT / path for path in [
            "benchmark/layer/_backward_moonep_case.py",
            "benchmark/layer/bench_moe_suite.py",
            "src/mega_moe/ops/function.py",
            "src/mega_moe/ops/forward.py",
            "benchmark/layer/_reprefetch_component_capture.py",
            "src/mega_moe/ops/_single_saved_adapter.py",
            "src/mega_moe/kernels/replica_weight_prefetch.py",
            "src/mega_moe/runtime/replica_weight_prefetch.py",
            "src/mega_moe/_goldens/_torch_forward_for_backward.py",
        ])
    snapshot_dir = args.output_dir / "sources"
    snapshot_dir.mkdir()
    for path in files:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("external") / path.name
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(args.output_dir / "metadata.json", {
        "argv": sys.argv, "case": asdict(case), "python": sys.executable,
        "repo": str(ROOT), "mega_moe": mega_moe.__file__,
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "triton": triton.__version__,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "env": {k: v for k, v in os.environ.items() if k.startswith(
            ("MOE_", "TRITON_", "ASCEND_", "HCCL_", "MEGAMOE_"))},
    })
    # Observe all eight cards, even for W4, to reject concurrent device jobs.
    devices = list(range(8))
    check_npu_occupancy(args.output_dir, devices, phase="before_spawn")
    context = mp.spawn(worker, args=(case.case_id, str(args.output_dir), args.warmup,
                                    args.iterations, args.compare_safe_wgrad, args.phase_samples,
                                    args.moonep, args.compare_reprefetch, args.reprefetch_profile_samples,
                                    args.compare_reprefetch_fused, args.compare_reprefetch_sync,
                                    args.correctness_repeats, args.check_saved_kernel,
                                    args.add_unused_replica, args.compare_reprefetch_overlap),
                       nprocs=case.world_size, join=False)
    owned_pids = {process.pid for process in context.processes}
    known_host_pids = set()
    status = {"status": "unverified"}
    deadline = time.monotonic() + args.timeout
    try:
        complete = True
        while not context.join(timeout=2):
            failures = list(args.output_dir.glob("failure_rank*.json"))
            if failures:
                raise RuntimeError(f"worker failed: {failures[0].read_text()}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"workers exceeded {args.timeout}s")
            observed = check_npu_occupancy(args.output_dir, devices, owned_pids,
                                           phase="running", known_host_pids=known_host_pids)
            complete = complete and observed
        check_npu_occupancy(args.output_dir, devices, owned_pids, phase="after_join",
                            known_host_pids=known_host_pids)
        if not complete:
            raise RuntimeError("incomplete occupancy monitoring; timings rejected")
        status["status"] = "passed"
    except BaseException as error:
        status.update(status="rejected", error=str(error))
        raise
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
        write_json(args.output_dir / "occupancy_result.json", status)
        path = args.output_dir / "benchmark_result.json"
        if path.exists():
            result = json.loads(path.read_text())
            result["occupancy_gate"] = status
            write_json(path, result)


if __name__ == "__main__":
    main()
