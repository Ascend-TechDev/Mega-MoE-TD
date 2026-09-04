# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Benchmark and profile the Kimi-K3 T16K single-kernel forward.

The measured boundary starts after router/top-k generation and contains the
complete MoE forward.  It compares the experimental one-launch Mega-MoE path
with the repository's Torch-NPU grouped-GEMM + HCCL baseline under the same
inputs and weights.

The benchmark uses the repository protocol: five warmup iterations, fifty NPU
event samples, and a per-sample MAX reduction across ranks.  After the
benchmark, both implementations are captured with Level1 NPU profiling.  Raw
traces are finalized on every rank before offline analysis begins.

Usage::

    MOE_FUSED_ASH_SIZE_GB=6 env PYTHONPATH=src:. python \
        benchmark/layer/profile_single_kernel_forward.py \
        --case performance-fwd-kimi-k3-w8-t16k \
        --output-dir results/profile_forward/kimi_k3_w8_t16k
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from benchmark.layer import bench_moe_suite as bench
from benchmark.layer._grouped_forward_baseline import GroupedForwardBaseline
from config import resolve_case
from mega_moe import FusedMoEForward, MoEForwardConfig
from tests import _moe_testkit as kit


PROFILE_ACTIVE_ITERS = 2
IMPLEMENTATIONS = ("single_kernel", "torch_grouped_hccl")
_PROFILE_METRICS = (
    "none",
    "pipe",
    "arithmetic",
    "memory",
    "memory_access",
    "memory_l0",
    "memory_ub",
    "resource_conflict",
    "l2",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and profile Kimi-K3 T16K single-kernel forward "
            "against Torch-NPU grouped-GEMM + HCCL"
        )
    )
    parser.add_argument(
        "--case",
        default="performance-fwd-kimi-k3-w8-t16k",
        help="registered Kimi T16K forward performance case",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for benchmark results and profiler data",
    )
    parser.add_argument(
        "--metric",
        choices=_PROFILE_METRICS,
        default="pipe",
        help="AiCore metric collected in the Level1 profile (default: pipe)",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def _case(case_id: str):
    case = resolve_case(case_id).validate()
    if (
        case.direction != "forward"
        or "performance" not in case.tags
        or "kimi" not in case.tags
        or case.tokens != 16384
        or case.hidden != 3584
        or case.ffn != 3072
        or case.world_size not in (4, 8)
    ):
        raise ValueError(
            "--case must select a registered Kimi T16K forward performance case"
        )
    return case


def _profile_experimental_config(profiler, metric_name: str):
    metrics = {
        "none": profiler.AiCMetrics.AiCoreNone,
        "pipe": profiler.AiCMetrics.PipeUtilization,
        "arithmetic": profiler.AiCMetrics.ArithmeticUtilization,
        "memory": profiler.AiCMetrics.Memory,
        "memory_access": profiler.AiCMetrics.MemoryAccess,
        "memory_l0": profiler.AiCMetrics.MemoryL0,
        "memory_ub": profiler.AiCMetrics.MemoryUB,
        "resource_conflict": profiler.AiCMetrics.ResourceConflictRatio,
        "l2": profiler.AiCMetrics.L2Cache,
    }
    return profiler._ExperimentalConfig(
        profiler_level=profiler.ProfilerLevel.Level1,
        aic_metrics=metrics[metric_name],
        l2_cache=metric_name == "l2",
        record_op_args=False,
        op_attr=False,
        export_type=profiler.ExportType.Text,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )


def _run_profile(
    implementation,
    run_once,
    output_dir,
    metric,
    rank,
    device,
    ep_group,
):
    profiler = torch_npu.profiler
    trace_root = output_dir / f"timeline_{metric}" / implementation
    handler = profiler.tensorboard_trace_handler(
        str(trace_root),
        worker_name=f"rank{rank}",
        analyse_flag=False,
        async_mode=False,
    )
    labels = (
        f"{implementation}_profiler_warmup",
        *(f"{implementation}_active_{index + 1}" for index in range(PROFILE_ACTIVE_ITERS)),
    )
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.NPU,
        ],
        schedule=profiler.schedule(
            wait=0,
            warmup=1,
            active=PROFILE_ACTIVE_ITERS,
            repeat=1,
        ),
        on_trace_ready=handler,
        experimental_config=_profile_experimental_config(profiler, metric),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    ) as prof:
        for label in labels:
            torch.npu.synchronize(device)
            dist.barrier(group=ep_group)
            # The HCCL barrier is device-asynchronous.  Keep the profiled call
            # aligned across ranks before entering kernels with peer waits.
            torch.npu.synchronize(device)
            with torch.autograd.profiler.record_function(label):
                output = run_once()
                torch.npu.synchronize(device)
            del output
            prof.step()
    torch.npu.synchronize(device)
    dist.barrier(group=ep_group)


def _write_benchmark_result(
    output_dir,
    case,
    candidate_result,
    baseline_result,
    route_distribution,
):
    candidate_stats = candidate_result.stats
    baseline_stats = baseline_result.stats
    payload = {
        "schema_version": 1,
        "case_id": case.case_id,
        "model": case.model,
        "world_size": case.world_size,
        "tokens_per_rank": case.tokens,
        "global_tokens": case.tokens * case.world_size,
        "shape": {
            "hidden": case.hidden,
            "ffn": case.ffn,
            "topk": case.topk,
            "num_experts": case.num_experts,
        },
        "measured_boundary": (
            "post-router full forward: routing metadata, dispatch, FC1, "
            "weighted SwiGLU, FC2, and combine"
        ),
        "candidate": "Mega-MoE one physical kernel launch",
        "baseline": "Torch-NPU grouped-GEMM + HCCL",
        "protocol": kit.FORWARD_TIMING.as_dict(),
        "correctness_gate": {
            "status": "passed_before_timing",
            "cases": [
                "normal",
                "zero-receive/empty-expert",
                "negative/out-of-range all-drop",
            ],
            "rtol": 0.05,
            "atol": 0.05,
        },
        "receive_capacity_factor": case.capacity_factor,
        "symmetric_heap_size_gb": bench.G_ASH_SIZE_GB,
        "route_distribution": route_distribution,
        "metrics": {
            "single_kernel_full_e2e_ms": candidate_stats,
            "torch_grouped_hccl_full_e2e_ms": baseline_stats,
        },
        "torch_over_single_kernel_median": round(
            baseline_stats["median_ms"] / candidate_stats["median_ms"], 3
        ),
        "samples_ms": {
            "single_kernel": [round(value, 6) for value in candidate_result.samples_ms],
            "torch_grouped_hccl": [round(value, 6) for value in baseline_result.samples_ms],
        },
    }
    result_path = output_dir / "benchmark_result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"[benchmark-result] {result_path}", flush=True)


def _worker(
    rank: int,
    world_size: int,
    case_id: str,
    output_dir: str,
    metric: str,
):
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    ep_group = dist.group.WORLD
    case = _case(case_id)
    if case.world_size != world_size:
        raise ValueError(
            f"worker world_size={world_size} does not match {case.case_id}"
        )
    device = f"npu:{rank}"

    try:
        required_ash_bytes = bench._required_ash_bytes(case, world_size)
        if required_ash_bytes >= bench.G_ASH_SIZE:
            raise RuntimeError(
                f"case requires {required_ash_bytes / (1024 ** 3):.3f} GiB "
                f"of symmetric payload, but only "
                f"MOE_FUSED_ASH_SIZE_GB={bench.G_ASH_SIZE_GB} was requested"
            )

        with kit.aclshmem_session(rank, world_size, bench.G_ASH_SIZE):
            experts_per_rank = case.num_experts // world_size
            grouped_baseline = GroupedForwardBaseline(case, ep_group)
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=case.tokens,
                hidden_size=case.hidden,
                top_k=case.topk,
                num_experts=case.num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=case.capacity_factor,
                    enable_single_kernel_forward=True,
                ),
            )
            try:
                packed_w1, down_weight, torch_w2_kn = bench._make_local_weights(
                    case,
                    experts_per_rank,
                    rank,
                    device,
                )
                hidden_states, selected_experts, routing_weights = (
                    bench._prepare_inputs(case, rank, device)
                )
                route_distribution = bench._summarize_route_distribution(
                    case,
                    selected_experts,
                    world_size,
                )

                def run_single_kernel():
                    return bench.ascend_full_post_routing(
                        op,
                        hidden_states,
                        selected_experts,
                        packed_w1,
                        down_weight,
                        routing_weights,
                    )

                def run_torch_grouped():
                    return grouped_baseline.full_post_routing(
                        hidden_states,
                        selected_experts,
                        routing_weights,
                        packed_w1,
                        torch_w2_kn,
                    )

                dist.barrier(group=ep_group)
                bench._validate_case(
                    device,
                    ep_group,
                    grouped_baseline,
                    op,
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    packed_w1,
                    down_weight,
                    torch_w2_kn,
                )
                bench._validate_edge_cases(
                    device,
                    ep_group,
                    case,
                    grouped_baseline,
                    op,
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    packed_w1,
                    down_weight,
                    torch_w2_kn,
                    experts_per_rank,
                )

                runner = kit.PerformanceRunner(
                    run_single_kernel,
                    run_torch_grouped,
                    kit.FORWARD_TIMING,
                    device=device,
                    ep_group=ep_group,
                )
                candidate_result, baseline_result = runner.run()
                if rank == 0:
                    _write_benchmark_result(
                        Path(output_dir),
                        case,
                        candidate_result,
                        baseline_result,
                        route_distribution,
                    )
                dist.barrier(group=ep_group)

                for implementation, run_once in (
                    ("single_kernel", run_single_kernel),
                    ("torch_grouped_hccl", run_torch_grouped),
                ):
                    if rank == 0:
                        print(f"[profile-start] {implementation}", flush=True)
                    _run_profile(
                        implementation,
                        run_once,
                        Path(output_dir),
                        metric,
                        rank,
                        device,
                        ep_group,
                    )
                    if rank == 0:
                        print(f"[profile-finished] {implementation}", flush=True)
            finally:
                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)
                op.finalize()
                torch.npu.empty_cache()
                dist.barrier(group=ep_group)
    finally:
        dist.destroy_process_group()


def _analyse_profiles(
    output_dir: Path,
    metric: str,
    case_id: str,
    world_size: int,
):
    analyse = getattr(torch_npu.profiler, "analyse", None)
    if analyse is None:
        from torch_npu.profiler.analysis._npu_profiler import NpuProfiler

        analyse = NpuProfiler.analyse

    trace_manifest = {}
    for implementation in IMPLEMENTATIONS:
        trace_root = output_dir / f"timeline_{metric}" / implementation
        traces = sorted(trace_root.glob("rank*_ascend_pt"))
        if len(traces) != world_size:
            raise RuntimeError(
                f"{implementation}: expected {world_size} rank traces, "
                f"got {len(traces)}"
            )
        for trace in traces:
            analyse(str(trace), max_process_number=4)
        trace_manifest[implementation] = [str(path.resolve()) for path in traces]

    manifest_path = output_dir / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "world_size": world_size,
                "metric": metric,
                "profiler_level": "Level1",
                "activities": ["CPU", "NPU"],
                "active_iterations": PROFILE_ACTIVE_ITERS,
                "rank_traces": trace_manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[profile-manifest] {manifest_path}", flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_metadata(output_dir: Path, metric: str, case):
    source_paths = (
        Path(__file__).resolve(),
        _REPO_ROOT / "src/mega_moe/ops/forward.py",
        _REPO_ROOT / "src/mega_moe/kernels/fused_forward.py",
        _REPO_ROOT / "benchmark/layer/_grouped_forward_baseline.py",
    )
    metadata = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"],
            cwd=_REPO_ROOT,
            text=True,
        ).splitlines(),
        "case_id": case.case_id,
        "world_size": case.world_size,
        "candidate": "Mega-MoE one physical kernel launch",
        "baseline": "Torch-NPU grouped-GEMM + HCCL",
        "benchmark_protocol": kit.FORWARD_TIMING.as_dict(),
        "profile": {
            "profiler_level": "Level1",
            "aic_metric": metric,
            "activities": ["CPU", "NPU"],
            "schedule": {
                "wait": 0,
                "warmup": 1,
                "active": PROFILE_ACTIVE_ITERS,
                "repeat": 1,
            },
            "analysis": "offline after every distributed worker finalized",
        },
        "source_sha256": {
            str(path.relative_to(_REPO_ROOT)): _sha256(path) for path in source_paths
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = _parse_args()
    selected_case = _case(args.case)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"--output-dir must be empty: {args.output_dir}")
    (args.output_dir / f"timeline_{args.metric}").mkdir(
        parents=True,
        exist_ok=True,
    )
    _write_metadata(args.output_dir, args.metric, selected_case)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    os.environ.setdefault("ASH_MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("ASH_MASTER_PORT", "8766")
    mp.spawn(
        _worker,
        args=(
            selected_case.world_size,
            selected_case.case_id,
            str(args.output_dir),
            args.metric,
        ),
        nprocs=selected_case.world_size,
        join=True,
    )
    _analyse_profiles(
        args.output_dir,
        args.metric,
        selected_case.case_id,
        selected_case.world_size,
    )
