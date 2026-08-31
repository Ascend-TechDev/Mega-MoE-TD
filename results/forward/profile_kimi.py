"""Profile MoonEP UDMA replica-prefetch overlap on W8 Kimi-K3.

It uses torch_npu Level1/PipeUtilization collection, one trace per rank, and
offline ``analyse`` after all distributed workers have finalized.

The trace records three consecutive fresh replica epochs. Replica weights
change every forward, so owner staging reuse waits for the preceding epoch's
per-consumer acknowledgements. Kernel compilation and buffer allocation are
completed before profiling.

The route profile is deliberately fixed: owner-load ratios are approximately
``[1.19, 1.69, 0.69, 0.69, 0.94, 0.94, 0.94, 0.94]`` and the planner creates
52 replicas.  Provide ``--output-dir``; the default case is Kimi-K3 W8 t4k.
"""

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys

# Always profile the current worktree instead of a separately installed
# ``mega_moe`` package.  This also makes the script independent of the caller's
# PYTHONPATH, which is easy to omit in long multi-rank profiler commands.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from benchmark.layer import bench_moe_suite as bench
from config import resolve_case
from tests import _moe_testkit as kit


WORLD_SIZE = 8
ROUTE_PROFILE = "moderate-wide"
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


@dataclass(frozen=True)
class ProfileConfig:
    case_id: str
    output_dir: Path
    metric: str
    replica_gate_up_chunk_mib: int
    replica_down_chunk_mib: int


def _parse_args() -> ProfileConfig:
    parser = argparse.ArgumentParser(
        description="Profile MoonEP replica-prefetch overlap on Kimi-K3 W8"
    )
    parser.add_argument(
        "--case",
        default="performance-fwd-kimi-k3-w8-t4k",
        help="registered Kimi-K3 W8 performance case id",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for trace and profile metadata",
    )
    parser.add_argument(
        "--metric",
        choices=_PROFILE_METRICS,
        default="pipe",
        help="AiCore metric collected in this run (default: pipe)",
    )
    parser.add_argument(
        "--replica-gate-up-chunk-mib",
        type=int,
        choices=(16, 32, 64, 128, 256),
        default=64,
        help="UDMA chunk size for gate/up replica prefetch (default: 64 MiB)",
    )
    parser.add_argument(
        "--replica-down-chunk-mib",
        type=int,
        choices=(4, 16, 32, 64, 128, 256),
        default=64,
        help="UDMA chunk size for down-replica prefetch (default: 64 MiB)",
    )
    args = parser.parse_args()
    return ProfileConfig(
        case_id=args.case,
        output_dir=args.output_dir,
        metric=args.metric,
        replica_gate_up_chunk_mib=args.replica_gate_up_chunk_mib,
        replica_down_chunk_mib=args.replica_down_chunk_mib,
    )


def _profile_experimental_config(profiler, metric_name: str):
    """Build the repository's fixed Kimi profiler configuration."""
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


def _case(config: ProfileConfig):
    case = resolve_case(config.case_id).validate()
    if (
        case.direction != "forward"
        or "performance" not in case.tags
        or case.model.upper() != "KIMI-K3"
        or case.world_size != WORLD_SIZE
    ):
        raise ValueError("--case must select a Kimi-K3 W8 performance case")
    return replace(
        case,
        capacity_factor=max(
            case.capacity_factor,
            bench._moonep_receive_capacity_factor(WORLD_SIZE, ROUTE_PROFILE),
        ),
    ).validate()


def _worker(rank: int, world_size: int, config: ProfileConfig) -> None:
    """Run only the balanced warmup and the three profiler steps."""
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", rank=rank, world_size=world_size)
    ep_group = dist.group.WORLD
    case = _case(config)
    device = f"npu:{rank}"

    try:
        # Keep every rank in the same lifecycle.  In particular there is no
        # rank-local planning, JSON output, or auxiliary operator inside the
        # ACLSHMEM session.
        dist.barrier(group=ep_group)
        experts_per_rank = case.num_experts // world_size
        packed_w1, down_weight, _ = bench._make_local_weights(
            case, experts_per_rank, rank, device
        )
        hidden_states, selected_experts, routing_weights = (
            bench._prepare_moonep_inputs(
                case,
                rank,
                world_size,
                device,
                route_profile=ROUTE_PROFILE,
            )
        )
        torch.npu.synchronize(device)
        torch.npu.empty_cache()
        dist.barrier(group=ep_group)

        with kit.aclshmem_session(rank, world_size, bench.G_ASH_SIZE):
            op = bench._make_forward_op(
                case,
                ep_group,
                enable_moonep=True,
                moonep_replica_gate_up_chunk_bytes=(
                    config.replica_gate_up_chunk_mib * 1024 * 1024
                ),
                moonep_replica_down_chunk_bytes=(
                    config.replica_down_chunk_mib * 1024 * 1024
                ),
            )
            try:
                def run_once():
                    return bench.ascend_full_post_routing(
                        op,
                        hidden_states,
                        selected_experts,
                        packed_w1,
                        down_weight,
                        routing_weights,
                    )

                # Allocate lazy buffers and compile every kernel before the
                # profiler starts. Every profiled step still uses a fresh epoch.
                output = run_once()
                del output
                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)

                copied_counts = [
                    sum(expert >= 0 for expert in row)
                    for row in op._replica_prepared_experts_cpu.tolist()
                ]
                primary_copies = sum(copied_counts)
                if primary_copies != 52:
                    raise AssertionError(
                        f"{ROUTE_PROFILE} expected 52 copies, got {primary_copies}"
                    )
                bytes_per_expert = (
                    case.hidden * 3 * case.ffn * torch.bfloat16.itemsize
                )

                profiler = torch_npu.profiler
                handler = profiler.tensorboard_trace_handler(
                    str(config.output_dir / f"timeline_{config.metric}"),
                    worker_name=f"rank{rank}",
                    analyse_flag=False,
                    async_mode=False,
                )
                if rank == 0:
                    print(
                        "MoonEP profile: "
                        f"copies={primary_copies} "
                        f"payload_gib={primary_copies * bytes_per_expert / (1024**3):.3f} "
                        "steps=fresh-warmup,fresh-active-1,fresh-active-2 "
                        f"metric={config.metric}",
                        flush=True,
                    )

                labels = (
                    "moonep_udma_fresh_epoch_warmup",
                    "moonep_udma_fresh_epoch_active_1",
                    "moonep_udma_fresh_epoch_active_2",
                )
                with profiler.profile(
                    activities=[
                        profiler.ProfilerActivity.CPU,
                        profiler.ProfilerActivity.NPU,
                    ],
                    schedule=profiler.schedule(
                        wait=0, warmup=1, active=2, repeat=1
                    ),
                    on_trace_ready=handler,
                    experimental_config=_profile_experimental_config(
                        profiler, config.metric
                    ),
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                ) as prof:
                    for label in labels:
                        torch.npu.synchronize(device)
                        dist.barrier(group=ep_group)
                        # HCCL barrier completion is device-asynchronous.  A
                        # second sync prevents the next rank-local planning
                        # kernels from starting at different wall times and
                        # turning the mixed dispatch kernel into a signal wait.
                        torch.npu.synchronize(device)
                        with torch.autograd.profiler.record_function(label):
                            output = run_once()
                            torch.npu.synchronize(device)
                        del output
                        prof.step()

                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)
            finally:
                bench._finalize_forward_op(op, device, ep_group)
    finally:
        dist.destroy_process_group()


def _analyse(config: ProfileConfig) -> None:
    analyse = getattr(torch_npu.profiler, "analyse", None)
    if analyse is None:
        from torch_npu.profiler.analysis._npu_profiler import NpuProfiler

        analyse = NpuProfiler.analyse
    traces = sorted(
        (config.output_dir / f"timeline_{config.metric}").glob(
            "rank*_ascend_pt"
        )
    )
    if len(traces) != WORLD_SIZE:
        raise RuntimeError(f"expected {WORLD_SIZE} rank traces, got {len(traces)}")
    for trace in traces:
        analyse(str(trace), max_process_number=4)
    print(
        f"profile traces: {config.output_dir / f'timeline_{config.metric}'}",
        flush=True,
    )


if __name__ == "__main__":
    profile_config = _parse_args()
    # Unlike the pytest distributed fixture, a directly executed profile
    # script must provide the env:// HCCL rendezvous endpoint itself.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    if profile_config.output_dir.exists() and any(
        profile_config.output_dir.iterdir()
    ):
        raise FileExistsError(
            f"--output-dir must be empty: {profile_config.output_dir}"
        )
    trace_dir = (
        profile_config.output_dir / f"timeline_{profile_config.metric}"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (profile_config.output_dir / "profile_metadata.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(
            {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=_REPO_ROOT,
                    text=True,
                ).strip(),
                "cann_version": "9.1.0",
                "case_id": profile_config.case_id,
                "route_profile": ROUTE_PROFILE,
                "replica_gate_up_chunk_mib": (
                    profile_config.replica_gate_up_chunk_mib
                ),
                "replica_down_chunk_mib": (
                    profile_config.replica_down_chunk_mib
                ),
                "replica_transport": (
                    "dispatch-fused owner-push UDMA PIPE_S PUT_SIGNAL, "
                    "one QP per destination, gate/up before down"
                ),
                "replica_staging": "compact owner-packed local source",
                "replica_reuse": "consumer-owned consumed epoch per slot",
                "dispatch_transport": "MTE direct symmetric mapping",
                "profile_steps": [
                    "fresh epoch profiler warmup",
                    "fresh epoch active 1",
                    "fresh epoch active 2",
                ],
            },
            output_file,
            indent=2,
        )
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, profile_config),
        nprocs=WORLD_SIZE,
        join=True,
    )
    _analyse(profile_config)
