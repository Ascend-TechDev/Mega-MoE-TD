# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Benchmark and profile the Kimi-K3 single-kernel forward.

The measured boundary starts after router/top-k generation and contains the
complete MoE forward.  It compares the selected one-launch Mega-MoE path
with the repository's Torch-NPU grouped-GEMM + HCCL baseline under the same
inputs and weights.

The benchmark uses the repository protocol: five warmup iterations, fifty NPU
event samples, and a per-sample MAX reduction across ranks.  After the
benchmark, both implementations are captured with Level1 NPU profiling.  Raw
traces are finalized on every rank before offline analysis begins.

Usage::

    MOE_FUSED_ASH_SIZE_GB=6 env PYTHONPATH=src:. python \
        benchmark/layer/profile_single_kernel_forward.py \
        --case performance-fwd-kimi-k3-trimmed-w8-t4k \
        --output-dir results/profile_forward/kimi_k3_trimmed_w8_t4k
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import triton

from benchmark.layer import bench_moe_suite as bench
from benchmark.layer._grouped_forward_baseline import GroupedForwardBaseline
from benchmark.layer._npu_occupancy import check_npu_occupancy
from benchmark.layer._pipe_profile_summary import write_pipe_profile_summary
from config import resolve_case
from mega_moe import FusedMoEForward, MoEForwardConfig
from tests import _moe_testkit as kit


PROFILE_ACTIVE_ITERS = 2
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
            "Benchmark and profile Kimi-K3 single-kernel forward "
            "against Torch-NPU grouped-GEMM + HCCL"
        )
    )
    parser.add_argument(
        "--case",
        default="performance-fwd-kimi-k3-trimmed-w8-t4k",
        help="registered Kimi forward performance case",
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
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="run correctness gates and timing without collecting profiles",
    )
    parser.add_argument("--fc1-block", type=int, nargs=3, default=(256, 256, 128),
                        metavar=("M", "N", "K"))
    parser.add_argument("--moonep", action="store_true",
                        help="enable device planning and UDMA replica prefetch in the same launch")
    parser.add_argument("--wave-windows", type=int,
                        help="M tiles per compute wave (default: 32 with MoonEP, otherwise 16)")
    parser.add_argument("--fc2-block", type=int, nargs=3, default=(256, 256, 128),
                        metavar=("M", "N", "K"))
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def _config_overrides(args):
    overrides = {"enable_single_kernel_forward": True, "enable_moonep": args.moonep,
                 "moonep_enable_replica_cache": False,
                 "single_kernel_group_windows": args.wave_windows}
    overrides.update(zip(
        ("fc1_gemm_block_size_m", "fc1_gemm_block_size_n", "fc1_gemm_block_size_k"),
        args.fc1_block,
    ))
    overrides.update(zip(
        ("fc2_combine_block_size_m", "fc2_gemm_block_size_n", "fc2_gemm_block_size_k"),
        args.fc2_block,
    ))
    return overrides


def _case(case_id: str):
    case = resolve_case(case_id).validate()
    if (
        case.direction != "forward"
        or "performance" not in case.tags
        or "kimi" not in case.tags
        or case.tokens not in (4096, 8192, 16384)
        or case.hidden != 3584
        or case.ffn != 3072
        or case.world_size not in (4, 8)
    ):
        raise ValueError(
            "--case must select a registered Kimi forward performance case"
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
    config,
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
        "candidate": "single_kernel",
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
        "operator_config": asdict(config),
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
    benchmark_only: bool,
    config_overrides: dict,
):
    faulthandler.dump_traceback_later(120, repeat=True)
    if rank == 0:
        print("[setup] initializing NPU and HCCL", flush=True)
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
        if rank == 0:
            print("[setup] materializing HCCL before ACLSHMEM", flush=True)
        # HCCL's AICPU module must load before ACLSHMEM on the tested runtime.
        bootstrap_input = torch.zeros(world_size, dtype=torch.float32, device=device)
        bootstrap_output = torch.empty_like(bootstrap_input)
        dist.all_reduce(bootstrap_input, group=ep_group)
        dist.all_to_all_single(bootstrap_output, bootstrap_input, group=ep_group)
        torch.npu.synchronize(device)
        del bootstrap_input, bootstrap_output
        required_ash_bytes = bench._required_ash_bytes(case, world_size)
        if required_ash_bytes >= bench.G_ASH_SIZE:
            raise RuntimeError(
                f"case requires {required_ash_bytes / (1024 ** 3):.3f} GiB "
                f"of symmetric payload, but only "
                f"MOE_FUSED_ASH_SIZE_GB={bench.G_ASH_SIZE_GB} was requested"
            )

        if rank == 0:
            print("[setup] initializing ACLSHMEM", flush=True)
        with kit.aclshmem_session(rank, world_size, bench.G_ASH_SIZE,
                                 enable_udma=config_overrides.get("enable_moonep", False)):
            if rank == 0:
                print("[setup] ACLSHMEM initialized; constructing operator", flush=True)
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
                    **config_overrides,
                ),
            )
            try:
                if rank == 0:
                    print("[setup] preparing weights and routes", flush=True)
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
                def run_candidate():
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
                if rank == 0:
                    print("[correctness] normal routes", flush=True)
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
                if op.enable_moonep:
                    from mega_moe.runtime.moonep_planning import plan_moonep_b0_b3

                    raw_counts = op.context.planning_counts_mem.view(
                        world_size, op.context.planning_num_bins)[:, :case.num_experts].cpu()
                    oracle = plan_moonep_b0_b3(raw_counts)
                    actual_etc = op.context.planning_experts_to_copy.cpu()
                    if not torch.equal(actual_etc, oracle.experts_to_copy):
                        raise AssertionError("single-kernel ETC differs from the MoonEP oracle")
                    if not torch.equal(op.context.planning_alloc_cumsum.cpu(), oracle.alloc_cumsum):
                        raise AssertionError("single-kernel allocation differs from the MoonEP oracle")
                    copies = (actual_etc >= 0).sum(dim=1).tolist()
                    if "moonep-skewed" in case.tags and case.num_experts == 32:
                        if copies != [0, 0, 1, 1, 1, 1, 1, 1]:
                            raise AssertionError(f"trimmed skewed case did not exercise six replicas: {copies}")
                    route_distribution["moonep"] = {
                        "experts_to_copy": actual_etc.tolist(),
                        "copies_per_rank": copies,
                        "total_weight_bytes_per_forward": sum(copies) * case.hidden * case.ffn * 6,
                        "transport": "upstream PIPE_S UDMA",
                        "replica_refresh": "every_forward",
                    }
                    if rank == 0:
                        print(f"[moonep] copies_per_rank={copies}; GPU planner matches oracle", flush=True)
                if rank == 0:
                    print("[correctness] zero-receive and all-drop routes", flush=True)
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

                if rank == 0:
                    print("[timing] correctness passed; measuring Triton then Torch", flush=True)
                faulthandler.cancel_dump_traceback_later()
                runner = kit.PerformanceRunner(
                    run_candidate,
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
                        op.config,
                    )
                dist.barrier(group=ep_group)

                if benchmark_only:
                    return
                for implementation, run_once in (
                    ("single_kernel", run_candidate),
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
        faulthandler.cancel_dump_traceback_later()
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
    for implementation in ("single_kernel", "torch_grouped_hccl"):
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
    if metric == "pipe":
        write_pipe_profile_summary(output_dir, trace_manifest["single_kernel"],
                                   world_size, PROFILE_ACTIVE_ITERS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_rendezvous():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("ASH_MASTER_ADDR", "127.0.0.1")
    hccl_port_base = 20000 + secrets.randbelow(500) * 64
    os.environ.setdefault(
        "HCCL_NPU_SOCKET_PORT_RANGE", f"{hccl_port_base}-{hccl_port_base + 63}")
    with ExitStack() as stack:
        for port_name, address_name in (
            ("MASTER_PORT", "MASTER_ADDR"),
            ("ASH_MASTER_PORT", "ASH_MASTER_ADDR"),
        ):
            if port_name not in os.environ:
                endpoint = stack.enter_context(socket.socket())
                endpoint.bind((os.environ[address_name], 0))
                os.environ[port_name] = str(endpoint.getsockname()[1])


def _write_metadata(output_dir: Path, metric: str, case, benchmark_only: bool,
                    config_overrides: dict):
    source_paths = (
        Path(__file__).resolve(),
        _REPO_ROOT / "src/mega_moe/ops/forward.py",
        _REPO_ROOT / "src/mega_moe/kernels/fused_forward.py",
        _REPO_ROOT / "src/mega_moe/kernels/fused_moonep.py",
        _REPO_ROOT / "benchmark/layer/_kimi_routes.py",
        _REPO_ROOT / "src/mega_moe/kernels/dispatch_fc1.py",
        _REPO_ROOT / "src/mega_moe/kernels/fc2_combine.py",
        _REPO_ROOT / "src/mega_moe/config.py",
        _REPO_ROOT / "config/_shapes.py",
        _REPO_ROOT / "benchmark/layer/_grouped_forward_baseline.py",
        _REPO_ROOT / "benchmark/layer/_npu_occupancy.py",
        _REPO_ROOT / "benchmark/layer/_pipe_profile_summary.py",
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
        "candidate": "single_kernel",
        "baseline": "Torch-NPU grouped-GEMM + HCCL",
        "benchmark_protocol": kit.FORWARD_TIMING.as_dict(),
        "communication_init_order": "hccl_collectives_then_aclshmem",
        "config_overrides": config_overrides,
        "environment": {
            "hostname": platform.node(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton_runtime": triton.__version__,
            "triton_module": triton.__file__,
            "triton_distributions": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if "triton" in distribution.metadata.get("Name", "").lower()
            },
            "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
            "ascend_home_resolved": str(Path(os.environ["ASCEND_HOME_PATH"]).resolve())
            if os.environ.get("ASCEND_HOME_PATH") else None,
            "communication": {
                name: os.environ.get(name)
                for name in (
                    "MASTER_ADDR", "MASTER_PORT", "ASH_MASTER_ADDR", "ASH_MASTER_PORT",
                    "HCCL_IF_BASE_PORT", "HCCL_SOCKET_IFNAME", "HCCL_EXEC_TIMEOUT",
                    "HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE",
                    "ASCEND_LAUNCH_BLOCKING",
                )
            },
        },
        "profile": {
            "enabled": not benchmark_only,
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
    config_overrides = _config_overrides(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"--output-dir must be empty: {args.output_dir}")
    (args.output_dir / f"timeline_{args.metric}").mkdir(
        parents=True,
        exist_ok=True,
    )
    _configure_rendezvous()

    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    device_ids = (
        [int(device) for device in visible_devices.split(",")][:selected_case.world_size]
        if visible_devices else list(range(selected_case.world_size))
    )
    if len(device_ids) != selected_case.world_size:
        raise ValueError("visible NPU devices do not cover the selected world")
    check_npu_occupancy(args.output_dir, device_ids, phase="before_spawn")
    _write_metadata(args.output_dir, args.metric, selected_case, args.benchmark_only,
                    config_overrides)
    context = mp.spawn(
        _worker,
        args=(
            selected_case.world_size,
            selected_case.case_id,
            str(args.output_dir),
            args.metric,
            args.benchmark_only,
            config_overrides,
        ),
        nprocs=selected_case.world_size,
        join=False,
    )
    owned_pids = {process.pid for process in context.processes}
    known_host_pids = set()
    occupancy_status = {"status": "unverified", "log": "device_occupancy.jsonl"}
    try:
        monitoring_complete = True
        while not context.join(timeout=2):
            observed = check_npu_occupancy(
                args.output_dir, device_ids, owned_pids, phase="running",
                known_host_pids=known_host_pids)
            monitoring_complete = monitoring_complete and observed
        check_npu_occupancy(args.output_dir, device_ids, owned_pids, phase="after_join",
                            known_host_pids=known_host_pids)
        occupancy_status["status"] = "passed" if monitoring_complete else "unverified"
    except BaseException as error:
        occupancy_status.update(status="rejected", error=str(error))
        raise
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(15)
            if process.is_alive():
                process.kill()
                process.join()
        result_path = args.output_dir / "benchmark_result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            result["occupancy_gate"] = occupancy_status
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        (args.output_dir / "occupancy_result.json").write_text(
            json.dumps(occupancy_status, indent=2) + "\n")
    if occupancy_status["status"] != "passed":
        raise RuntimeError("Device monitoring was incomplete; timing is not accepted")
    if not args.benchmark_only:
        _analyse_profiles(
            args.output_dir,
            args.metric,
            selected_case.case_id,
            selected_case.world_size,
        )
