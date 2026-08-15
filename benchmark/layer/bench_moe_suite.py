# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven Ascend Mega-MoE forward/backward performance suite.

The measured model boundary deliberately excludes the router.  ``selected_experts``
and FP32 ``routing_weights`` already exist when timing starts:

    routing metadata + dispatch + packed FC1 + weighted SwiGLU + FC2 + combine

Activations and both expert weights are BF16.  Routing weights remain FP32 at
the public boundary, during transport, and in the weighted-SwiGLU multiply.
The optimized path calls the production layer interfaces directly; this file
does not carry a benchmark-local FC2/combine kernel.

FC2/combine uses one production schedule: Cube FC2 groups stage output in local
GM while striped ACLSHMEM device-put workers transport completed groups before
the local top-k reduction.

The only performance baseline is Torch-NPU grouped-GEMM + HCCL.  The primary
comparison is direct-call full E2E.  Optional event slices (preprocess,
dispatch+FC1, and the post-dispatch weighted-SwiGLU+FC2+combine segment) can be
enabled for stage diagnosis.  The Ascend post-dispatch segment uses the same
shadow activation schedule as production; the Torch segment runs weighted
SwiGLU followed by FC2+combine serially.  Event slices are diagnostics:
independently rank-reduced statistics are not additive to full E2E.

Protocol: 5 warmup iterations and 50 measured iterations for every metric.  The
forward NPU-event samples are reduced with MAX individually; backward preserves
the historical host-wall interval followed by rank-MAX reduction.
Before timing, the candidate and grouped baseline must pass normal,
zero-receive/empty-expert, and negative/out-of-range all-drop comparison gates.

Workload profiles:
    KIMI-K3: H=3584, F=3072, top-k=16, E=896,
             tokens/rank in {4096, 8192, 16384}; primary optimization target
    QWEN: H=2048, F=768, top-k=8, E=128,
          tokens/rank in {4096, 8192, 16384}
    DSV4: H=7168, F=3072, top-k=6, E=384,
          tokens/rank in {4096, 8192, 16384}; BF16 routed experts only

Usage (the pytest fixture starts workers; do not wrap in torchrun):
    source ./run.sh
    # Each node is one explicit model/world/tokens-per-rank case.
    python -m pytest --collect-only -q benchmark/layer/bench_moe_suite.py
    python -m pytest \
        'benchmark/layer/bench_moe_suite.py::test_bench_forward_case[performance-fwd-qwen-w8-t8k]' \
        -m dist -v -s

Case selection is intentionally pytest-native.  Use ``-k forward`` or
``-k backward`` for a direction and a node id (or ``-m kimi``, ``-m dsv4``,
``-m smoke``) for a subset.  ``CaseSpec.tokens`` is always the token count on
one rank; there is no environment-controlled shape or world-size selection.

The remaining environment variables are runtime knobs only:
    MOE_FULL_BENCH_BREAKDOWN=1          # opt into three-segment diagnostics
    MOE_FULL_BENCH_RESULTS_DIR=/tmp/... # optional forward result directory
    MOE_BACKWARD_BENCH_RESULTS_DIR=/tmp/... # optional backward result directory
    MOE_FUSED_ASH_SIZE_GB=6             # ACLSHMEM heap size, not shape selection
"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from collections import OrderedDict

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:  # pragma: no cover - distributed Ascend jobs require torch-npu
    torch_npu = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from mega_moe import FusedMoEForward, MoEForwardConfig
from mega_moe.kernels.fc2_combine import _fc2_device_put_worker_layout
from config import CaseSpec, select_cases
from benchmark.layer._grouped_forward_baseline import GroupedForwardBaseline
from tests import _moe_testkit as kit
from tests._moe_baselines import backward_torch_baseline, build_backward_saved


ACTIVATION_DTYPE = torch.bfloat16
ROUTING_INPUT_DTYPE = torch.float32
ROUTING_TRANSPORT_DTYPE = torch.float32
RESULT_CONTRACT = "bf16-activations-fp32-routing-device-put-v3"
_WEIGHT_INIT_CHUNK_BYTES = 128 * 1024 * 1024

# This is the published protocol.  Keep debug/short runs under a differently
# named script so their output cannot be mistaken for 5/50 evidence.
FORWARD_TIMING = kit.FORWARD_TIMING
BACKWARD_TIMING = kit.BACKWARD_TIMING
WARMUP_ITERS = FORWARD_TIMING.warmup
BENCH_ITERS = FORWARD_TIMING.iterations
# Production forward uses FC2 shadow activation. The optional three-segment
# slices exercise the same schedule, but are not part of every E2E run.
RUN_BREAKDOWN = os.environ.get("MOE_FULL_BENCH_BREAKDOWN", "0") == "1"
G_ASH_SIZE_GB = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", "6"))
RESULTS_DIR = os.environ.get(
    "MOE_FULL_BENCH_RESULTS_DIR",
    str(PROJECT_ROOT / "results" / "forward"),
)

if G_ASH_SIZE_GB <= 0:
    raise ValueError("MOE_FUSED_ASH_SIZE_GB must be a positive integer")
G_ASH_SIZE = G_ASH_SIZE_GB * 1024 * 1024 * 1024


def _benchmark_provenance():
    """Return enough immutable context to distinguish new results from old JSON."""
    with open(__file__, "rb") as source_file:
        source_sha256 = hashlib.sha256(source_file.read()).hexdigest()
    production_sources = (
        PROJECT_ROOT / "src" / "mega_moe" / "config.py",
        PROJECT_ROOT / "src" / "mega_moe" / "kernels" / "dispatch_fc1.py",
        PROJECT_ROOT / "src" / "mega_moe" / "kernels" / "fc2_combine.py",
        PROJECT_ROOT / "src" / "mega_moe" / "ops" / "forward.py",
        PROJECT_ROOT / "src" / "mega_moe" / "runtime" / "routing.py",
        PROJECT_ROOT / "src" / "mega_moe" / "runtime" / "workspace.py",
        PROJECT_ROOT / "src" / "mega_moe" / "kernels" / "weighted_swiglu.py",
    )
    forward_source_sha256 = {}
    for path in production_sources:
        with open(path, "rb") as source_file:
            forward_source_sha256[str(path.relative_to(PROJECT_ROOT))] = (
                hashlib.sha256(source_file.read()).hexdigest()
            )
    return {
        "result_contract": RESULT_CONTRACT,
        "benchmark_source": os.path.abspath(__file__),
        "benchmark_source_sha256": source_sha256,
        "forward_source_sha256": forward_source_sha256,
        "command": shlex.join(sys.argv),
    }


def _get_ash_ip_port():
    return kit.get_ash_ip_port()


def _required_ash_bytes(case: CaseSpec, world_size):
    """Conservative payload estimate; ACLSHMEM allocator metadata needs headroom."""
    experts_per_rank = case.num_experts // world_size
    max_recv_rows = int(case.tokens * case.topk * case.capacity_factor)
    token_peer_bytes = max_recv_rows * case.hidden * ACTIVATION_DTYPE.itemsize
    # FC2 device-put writes stable-send rows into a symmetric combine buffer.
    combine_bytes = case.tokens * case.topk * case.hidden * ACTIVATION_DTYPE.itemsize
    routing_peer_bytes = max_recv_rows * ROUTING_TRANSPORT_DTYPE.itemsize
    dispatch_tile_m = int(
        os.environ.get("MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M", "128")
    )
    max_source_tiles = (
        case.tokens * case.topk + dispatch_tile_m - 1
    ) // dispatch_tile_m
    signal_slots = world_size * experts_per_rank * max_source_tiles
    signal_bytes = signal_slots * 16 * torch.int32.itemsize
    metadata_bins = 1 << (case.num_experts - 1).bit_length()
    metadata_bytes = world_size * metadata_bins * torch.int32.itemsize
    return (
        token_peer_bytes
        + combine_bytes
        + routing_peer_bytes
        + signal_bytes
        + metadata_bytes
    )


def _layer_tiling_overrides():
    overrides = {}
    for env_name, parameter_name in (
        (
            "MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M",
            "dispatch_fc1_block_size_m",
        ),
        ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_M", "fc1_gemm_block_size_m"),
        ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_N", "fc1_gemm_block_size_n"),
        ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_K", "fc1_gemm_block_size_k"),
        (
            "MOE_FUSED_FC2_COMBINE_BLOCK_SIZE_M",
            "fc2_combine_block_size_m",
        ),
        ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_N", "fc2_gemm_block_size_n"),
        ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_K", "fc2_gemm_block_size_k"),
    ):
        if env_name not in os.environ:
            continue
        value = int(os.environ[env_name])
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{env_name} must be a positive power of two")
        overrides[parameter_name] = value

    return overrides


@torch.no_grad()
def _allocate_chunked_normal_weight(shape, scale, device):
    """Initialize one large BF16 expert table without a full-shape RNG workspace.

    Ascend ``torch.randn`` can request a workspace roughly twice the output
    size.  A DSV4 W2 packed W1 is already 15.75 GiB, so generating the complete
    tensor in one call exceeds 64 GiB HBM once ACLSHMEM is initialized.  Fill
    the final contiguous allocation in expert-major chunks instead.  Weight
    initialization remains outside every timed boundary.
    """
    weight = torch.empty(shape, dtype=ACTIVATION_DTYPE, device=device)
    bytes_per_expert = weight[0].numel() * weight.element_size()
    experts_per_chunk = max(1, _WEIGHT_INIT_CHUNK_BYTES // bytes_per_expert)
    for expert_start in range(0, shape[0], experts_per_chunk):
        expert_end = min(shape[0], expert_start + experts_per_chunk)
        chunk = torch.randn(
            (expert_end - expert_start, *shape[1:]),
            dtype=ACTIVATION_DTYPE,
            device=device,
        ).mul_(scale)
        weight[expert_start:expert_end].copy_(chunk)
        del chunk
    return weight


@torch.no_grad()
def _make_local_weights(case: CaseSpec, experts_per_rank, rank, device, seed=42):
    """Create BF16 weights shared by the candidate and grouped baseline."""
    torch.manual_seed(seed + rank)
    fc1_scale = (1.0 / case.hidden) ** 0.5
    fc2_scale = (1.0 / case.ffn) ** 0.5
    # Generate the production [E, K, N] model-load layout directly.  A hot-path
    # transpose or materialization would create an unnecessary multi-GiB peak.
    packed_w1 = _allocate_chunked_normal_weight(
        (experts_per_rank, case.hidden, 2 * case.ffn),
        fc1_scale,
        device,
    )
    down_weight = _allocate_chunked_normal_weight(
        (experts_per_rank, case.hidden, case.ffn),
        fc2_scale,
        device,
    )

    # Layout preparation is outside both timed paths.  W2 remains physically
    # [E,N,K] because the W4 DSV4 A/B measured it 12.6% faster than [E,K,N].
    torch_w2_kn = down_weight.transpose(1, 2)
    return packed_w1, down_weight, torch_w2_kn


@torch.no_grad()
def _prepare_inputs(case: CaseSpec, rank, device, seed=43):
    """Create post-router inputs; top-k itself is intentionally outside timing."""
    torch.manual_seed(seed + rank * 1000)
    hidden_states = torch.randn(
        (case.tokens, case.hidden), dtype=ACTIVATION_DTYPE, device=device
    ).mul_(0.5).contiguous()
    router_logits = torch.randn(
        (case.tokens, case.num_experts),
        dtype=ROUTING_INPUT_DTYPE,
        device=device,
    )
    topk_logits, selected_experts = torch.topk(
        router_logits, k=case.topk, dim=-1
    )
    routing_weights = F.softmax(topk_logits, dim=-1).to(ROUTING_INPUT_DTYPE).contiguous()
    return hidden_states, selected_experts.to(torch.int32).contiguous(), routing_weights


@torch.no_grad()
def _summarize_route_distribution(case: CaseSpec, selected_experts, world_size):
    """Collect untimed global route-distribution provenance on every rank."""
    experts_per_rank = case.num_experts // world_size
    destination_ranks = torch.div(
        selected_experts.reshape(-1),
        experts_per_rank,
        rounding_mode="floor",
    ).to(torch.int64)
    routes_received_per_rank = torch.bincount(
        destination_ranks, minlength=world_size
    ).to(torch.int64)
    dist.all_reduce(routes_received_per_rank, op=dist.ReduceOp.SUM)

    active_mask = torch.bincount(
        selected_experts.reshape(-1).to(torch.int64),
        minlength=case.num_experts,
    ).gt(0).to(torch.int32)
    dist.all_reduce(active_mask, op=dist.ReduceOp.MAX)
    return {
        "active_global_experts": int(active_mask.sum().item()),
        "routes_received_per_rank": routes_received_per_rank.cpu().tolist(),
    }


# The grouped performance implementation lives in
# ``_grouped_forward_baseline.GroupedForwardBaseline`` so it cannot be
# confused with the independent correctness references in ``tests``.

def ascend_full_post_routing(op, hidden_states, selected_experts, packed_w1, down_weight, routing_weights):
    """One direct production full-forward call; this is the primary candidate boundary."""
    return op.forward(hidden_states, selected_experts, packed_w1, down_weight, routing_weights)


def _sync_ranks_before_event(device, ep_group):
    torch.npu.synchronize(device)
    dist.barrier(group=ep_group)
    torch.npu.synchronize(device)


def _sample_rank_max(values, device, ep_group):
    elapsed = torch.tensor(values, dtype=torch.float32, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=ep_group)
    return tuple(float(value) for value in elapsed.cpu().tolist())


def _time_ascend_breakdown(
    device,
    ep_group,
    op,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
):
    events = [torch.npu.Event(enable_timing=True) for _ in range(4)]
    _sync_ranks_before_event(device, ep_group)
    stream = torch.npu.current_stream(device)
    events[0].record(stream)
    routing_plan = op.build_routing_plan(selected_experts)
    events[1].record(stream)
    dispatch_result = op.dispatch_fc1(
        hidden_states,
        selected_experts,
        routing_plan,
        packed_w1,
        routing_weights=routing_weights,
        final_barrier=False,
    )
    events[2].record(stream)
    output = op._fc2_combine_shadow_activation(down_weight, dispatch_result)
    events[3].record(stream)
    events[-1].synchronize()
    local = [events[index].elapsed_time(events[index + 1]) for index in range(3)]
    return output, _sample_rank_max(local, device, ep_group)


def _time_torch_grouped_breakdown(
    device,
    ep_group,
    grouped_baseline,
    hidden_states,
    selected_experts,
    routing_weights,
    torch_w1_kn,
    torch_w2_kn,
):
    events = [torch.npu.Event(enable_timing=True) for _ in range(4)]
    _sync_ranks_before_event(device, ep_group)
    stream = torch.npu.current_stream(device)
    events[0].record(stream)
    state = grouped_baseline.preprocess(
        hidden_states, selected_experts, routing_weights
    )
    events[1].record(stream)
    state = grouped_baseline.dispatch_fc1(state, torch_w1_kn)
    events[2].record(stream)
    weighted = grouped_baseline.weighted_swiglu(state)
    output = grouped_baseline.fc2_combine(state, weighted, torch_w2_kn)
    events[3].record(stream)
    events[-1].synchronize()
    local = [events[index].elapsed_time(events[index + 1]) for index in range(3)]
    return output, _sample_rank_max(local, device, ep_group)


def _assert_close_collective(actual, expected, device, name, ep_group):
    ok = True
    message = ""
    try:
        if actual.shape != expected.shape:
            raise AssertionError(f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}")
        if actual.dtype != ACTIVATION_DTYPE or expected.dtype != ACTIVATION_DTYPE:
            raise AssertionError(
                f"dtype mismatch: actual={actual.dtype}, expected={expected.dtype}, "
                f"required={ACTIVATION_DTYPE}"
            )
        torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
    except AssertionError as exc:
        ok = False
        message = str(exc).splitlines()[0]
    flag = torch.tensor([int(ok)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if not bool(flag.item()):
        raise AssertionError(f"{name} correctness gate failed: {message}")


def _stats(samples, device):
    values = torch.tensor(samples, dtype=torch.float32, device=device)
    return {
        "min_ms": round(float(values.min().item()), 3),
        "max_ms": round(float(values.max().item()), 3),
        "mean_ms": round(float(values.mean().item()), 3),
        "median_ms": round(float(values.median().item()), 3),
    }


def _stage_stats(samples, device):
    names = ("preprocess", "dispatch_fc1", "weighted_swiglu_fc2_combine")
    return OrderedDict(
        (f"{name}_event_ms", _stats([sample[index] for sample in samples], device))
        for index, name in enumerate(names)
    )


def _median_ratio(baseline_stats, candidate_stats):
    candidate = candidate_stats["median_ms"]
    return round(baseline_stats["median_ms"] / candidate, 3) if candidate > 0 else 0.0


def _validate_case(
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
):
    candidate = ascend_full_post_routing(
        op, hidden_states, selected_experts, packed_w1, down_weight, routing_weights
    )
    grouped = grouped_baseline.full_post_routing(
        hidden_states,
        selected_experts,
        routing_weights,
        packed_w1,
        torch_w2_kn,
    )
    _assert_close_collective(
        candidate, grouped, device, "full post-routing MoE vs grouped baseline", ep_group
    )


def _validate_edge_cases(
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
):
    """Exercise zero-receive, empty-expert, negative-id, and all-drop cases."""
    num_experts = case.num_experts
    world_size = dist.get_world_size(group=ep_group)
    # Exclude destination rank 0 while distributing routes over every other
    # rank.  This creates a zero-receive rank and many empty local experts
    # without exceeding a DSV4 capacity factor of four at W8.
    num_routes = selected_experts.numel()
    route_ordinal = torch.arange(num_routes, dtype=torch.int32, device=device)
    destination = route_ordinal.remainder(world_size - 1) + 1
    # A profile may intentionally use capacity < world_size (for example the
    # Qwen benchmark's 1.25). Drop enough edge-case routes that distributing
    # them over W-1 destinations remains within every peer buffer.
    max_valid = min(
        num_routes,
        int(
            num_routes
            * op.receive_capacity_factor
            * (world_size - 1)
            / world_size
        ),
    )
    zero_receive_flat = torch.full(
        (num_routes,), num_experts, dtype=torch.int32, device=device
    )
    valid_edge = route_ordinal < max_valid
    zero_receive_flat[valid_edge] = (
        destination[valid_edge] * experts_per_rank
    )
    zero_receive_routes = zero_receive_flat.view_as(selected_experts).contiguous()
    _validate_case(
        device,
        ep_group,
        grouped_baseline,
        op,
        hidden_states,
        zero_receive_routes,
        routing_weights,
        packed_w1,
        down_weight,
        torch_w2_kn,
    )

    all_drop_routes = torch.full_like(selected_experts, num_experts)
    all_drop_routes[0, 0] = -1
    _validate_case(
        device,
        ep_group,
        grouped_baseline,
        op,
        hidden_states,
        all_drop_routes,
        routing_weights,
        packed_w1,
        down_weight,
        torch_w2_kn,
    )


def _measure_case(
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
):
    runner = kit.PerformanceRunner(
        lambda: ascend_full_post_routing(
            op,
            hidden_states,
            selected_experts,
            packed_w1,
            down_weight,
            routing_weights,
        ),
        lambda: grouped_baseline.full_post_routing(
            hidden_states,
            selected_experts,
            routing_weights,
            packed_w1,
            torch_w2_kn,
        ),
        FORWARD_TIMING,
        device=device,
        ep_group=ep_group,
    )
    ascend_full_result, torch_grouped_full_result = runner.run()

    breakdown = None
    if RUN_BREAKDOWN:
        for _ in range(WARMUP_ITERS):
            _time_ascend_breakdown(
                device,
                ep_group,
                op,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                down_weight,
            )
            _time_torch_grouped_breakdown(
                device,
                ep_group,
                grouped_baseline,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                torch_w2_kn,
            )
        ascend_breakdown_samples = []
        torch_breakdown_samples = []
        for _ in range(BENCH_ITERS):
            _, elapsed = _time_ascend_breakdown(
                device,
                ep_group,
                op,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                down_weight,
            )
            ascend_breakdown_samples.append(elapsed)
            _, elapsed = _time_torch_grouped_breakdown(
                device,
                ep_group,
                grouped_baseline,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                torch_w2_kn,
            )
            torch_breakdown_samples.append(elapsed)
        ascend_breakdown = _stage_stats(ascend_breakdown_samples, device)
        torch_grouped_breakdown = _stage_stats(torch_breakdown_samples, device)
        breakdown_speedups = OrderedDict(
            (
                name,
                _median_ratio(
                    torch_grouped_breakdown[f"{name}_event_ms"],
                    ascend_breakdown[f"{name}_event_ms"],
                ),
            )
            for name in (
                "preprocess",
                "dispatch_fc1",
                "weighted_swiglu_fc2_combine",
            )
        )
        breakdown = OrderedDict(
            {
                "ascend_event_slices": ascend_breakdown,
                "torch_npu_grouped_hccl_event_slices": torch_grouped_breakdown,
                "torch_npu_grouped_hccl_over_ascend_median": breakdown_speedups,
                "warning": (
                    "independent complete-execution event samples reduced by rank MAX; "
                    "do not add independently summarized stage statistics into E2E"
                ),
            }
        )

    return {
        "ascend_full": ascend_full_result.stats,
        "torch_grouped_full": torch_grouped_full_result.stats,
        "breakdown": breakdown,
    }


def _make_entry(case, world_size, op, measured, route_distribution):
    """Build one schema-v1 forward result entry for one immutable case."""
    ascend_full = measured["ascend_full"]
    torch_grouped_full = measured["torch_grouped_full"]
    _, device_put_workers = _fc2_device_put_worker_layout(
        op.num_aivector_programs,
        world_size,
        op._fc2_pipeline_group_experts,
    )
    device_properties = torch_npu.npu.get_device_properties(op.rank)
    entry = OrderedDict(
        {
            "schema_version": 1,
            "direction": "forward",
            "case_id": case.case_id,
            "model": case.model,
            "world_size": world_size,
            "tokens_per_rank": case.tokens,
            "shape": {
                "hidden": case.hidden,
                "ffn": case.ffn,
                "topk": case.topk,
                "num_experts": case.num_experts,
            },
            "hardware": {
                "name": device_properties.name,
                "physical_cube_core_num": device_properties.cube_core_num,
                "physical_vector_core_num": device_properties.vector_core_num,
                "l2_cache_size_bytes": device_properties.L2_cache_size,
                "launch_program_count": op.num_aicore_programs,
            },
            "protocol": FORWARD_TIMING.as_dict(),
            "correctness_gate": {
                "status": "passed_before_timing",
                "cases": [
                    "normal",
                    "zero-receive/empty-expert",
                    "negative/out-of-range all-drop",
                ],
                "baseline": "Torch-NPU grouped-GEMM + HCCL",
            },
            "activation_dtype": "bfloat16",
            "routing_weight_input_dtype": "float32",
            "routing_weight_transport_dtype": "float32",
            "measured_boundary": "post-router full forward; router/top-k generation excluded",
            "synthetic_input_generation": {
                "hidden_states": "rank-seeded BF16 normal values scaled by 0.5",
                "selected_experts": "top-k over rank-seeded dense random logits",
                "routing_weights": "FP32 softmax over selected logits",
                **route_distribution,
            },
            "provenance": _benchmark_provenance(),
            "receive_capacity_factor": case.capacity_factor,
            "symmetric_heap_size_gb": G_ASH_SIZE_GB,
            "dispatch_fc1_schedule": op.config.dispatch_fc1_schedule,
            "weighted_vector_programs": op.num_aivector_programs,
            "fc2_combine_transport": (
                "aclshmem_device_putmem_striped_workers_stream_events"
            ),
            "fc2_pipeline_group_experts": op._fc2_pipeline_group_experts,
            "fc2_reverse_vector_workers": device_put_workers,
            "fc2_reduce_programs": op.num_aivector_programs,
            "fc2_reduce_block_n_policy": (
                "1024 if local received routes >= 1024 else 256"
            ),
            "tiles": {
                "dispatch_fc1_m": op.config.dispatch_fc1_block_size_m,
                "fc1_gemm_m": op.config.fc1_gemm_block_size_m,
                "fc1_gemm_n": op.config.fc1_gemm_block_size_n,
                "fc1_gemm_k": op.config.fc1_gemm_block_size_k,
                "fc2_combine_m": op.config.fc2_combine_block_size_m,
                "fc2_gemm_n": op.config.fc2_gemm_block_size_n,
                "fc2_gemm_k": op.config.fc2_gemm_block_size_k,
            },
            "metrics": OrderedDict(
                {
                    "ascend_full_direct_e2e_ms": ascend_full,
                    "torch_npu_grouped_hccl_full_direct_e2e_ms": torch_grouped_full,
                }
            ),
            "torch_npu_grouped_hccl_over_ascend_full_median": _median_ratio(
                torch_grouped_full, ascend_full
            ),
        }
    )
    if measured["breakdown"] is not None:
        entry["diagnostics"] = measured["breakdown"]
        entry["diagnostics"]["ascend_event_slices_scope"] = (
            "production shadow weighted+FC2+combine segment; Torch side is "
            "serial weighted+FC2+combine; stage slices remain independent "
            "diagnostics and are not additive to full E2E"
        )
    return entry


def _print_entry(entry):
    metrics = entry["metrics"]
    print(
        f"  {entry['case_id']} full-forward median: "
        f"Ascend={metrics['ascend_full_direct_e2e_ms']['median_ms']:.3f} ms  "
        f"Grouped={metrics['torch_npu_grouped_hccl_full_direct_e2e_ms']['median_ms']:.3f} ms  "
        f"speedup={entry['torch_npu_grouped_hccl_over_ascend_full_median']:.3f}x",
        flush=True,
    )


def _upsert_result(path, direction, world_size, entry, protocol):
    """Atomically merge one pytest node into a direction/world envelope."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "direction": direction,
        "world_size": world_size,
        "protocol": protocol,
        "cases": [],
    }
    if path.is_file():
        with path.open("r", encoding="utf-8") as input_file:
            existing = json.load(input_file)
        compatible = (
            isinstance(existing, dict)
            and existing.get("schema_version") == 1
            and existing.get("direction") == direction
            and existing.get("world_size") == world_size
            and existing.get("protocol") == protocol
        )
        if not compatible:
            raise ValueError(
                f"refusing to overwrite incompatible benchmark result {path}; "
                "use a fresh results directory"
            )
        existing_contracts = {
            item.get("provenance", {}).get("result_contract")
            for item in existing.get("cases", [])
            if isinstance(item, dict)
        }
        entry_contract = entry.get("provenance", {}).get("result_contract")
        if existing_contracts - {entry_contract}:
            raise ValueError(
                f"refusing to mix benchmark contracts in {path}; "
                "use a fresh results directory"
            )
        payload = existing
    cases = [
        item
        for item in payload.get("cases", [])
        if item.get("case_id") != entry["case_id"]
    ]
    cases.append(entry)
    payload["cases"] = sorted(cases, key=lambda item: item["case_id"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    os.replace(temporary, path)
    print(f"[saved] {path}", flush=True)


def run_forward_benchmark(rank: int, world_size: int, case: CaseSpec):
    """Run exactly the explicitly parameterized forward performance case."""
    case = case.validate()
    if case.direction != "forward" or "performance" not in case.tags:
        raise ValueError(f"forward runner received non-performance case {case.case_id}")
    if world_size != case.world_size:
        raise ValueError(f"worker world size does not match {case.case_id}")
    if torch_npu is None or kit.ash is None:
        raise RuntimeError("this benchmark requires torch_npu and ACLSHMEM")
    if world_size not in (2, 4, 8):
        raise ValueError(
            f"full MoE benchmark supports world sizes 2, 4, and 8; got {world_size}"
        )
    if case.num_experts % world_size:
        raise ValueError(f"{case.case_id}: expert count is not divisible by world size")

    required_ash_bytes = _required_ash_bytes(case, world_size)
    if required_ash_bytes >= G_ASH_SIZE:
        raise RuntimeError(
            f"{case.case_id} estimates {required_ash_bytes / (1024 ** 3):.3f} GiB "
            f"of symmetric payload, but MOE_FUSED_ASH_SIZE_GB={G_ASH_SIZE_GB}"
        )

    ep_group = dist.group.WORLD
    with kit.aclshmem_session(rank, world_size, G_ASH_SIZE):
        device = f"npu:{rank}"
        experts_per_rank = case.num_experts // world_size
        grouped_baseline = GroupedForwardBaseline(case, ep_group)
        config = MoEForwardConfig(
            receive_capacity_factor=case.capacity_factor,
            **_layer_tiling_overrides(),
        )
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=case.tokens,
            hidden_size=case.hidden,
            top_k=case.topk,
            num_experts=case.num_experts,
            config=config,
        )
        try:
            packed_w1, down_weight, torch_w2_kn = _make_local_weights(
                case, experts_per_rank, rank, device
            )
            hidden_states, selected_experts, routing_weights = _prepare_inputs(
                case, rank, device
            )
            route_distribution = _summarize_route_distribution(
                case, selected_experts, world_size
            )
            dist.barrier(group=ep_group)
            _validate_case(
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
            _validate_edge_cases(
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
            measured = _measure_case(
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
            entry = _make_entry(case, world_size, op, measured, route_distribution)
            if rank == 0:
                _print_entry(entry)
                _upsert_result(
                    Path(RESULTS_DIR) / f"bench_forward_suite_w{world_size}.json",
                    "forward",
                    world_size,
                    entry,
                    entry["protocol"],
                )
        finally:
            torch.npu.synchronize(device)
            dist.barrier(group=ep_group)
            op.finalize()
            torch.npu.empty_cache()
            dist.barrier(group=ep_group)

def _backward_gate(saved, dy, peer_mem):
    from mega_moe import moe_backward_triton

    with torch.no_grad():
        torch_result = backward_torch_baseline(saved, dy)
        triton_result = moe_backward_triton(saved, dy, peer_mem)
    keys = ("grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2")
    for key in keys:
        if key not in torch_result or key not in triton_result:
            raise AssertionError(f"backward result is missing {key}")
        if torch_result[key].shape != triton_result[key].shape:
            raise AssertionError(f"backward gate shape mismatch for {key}")
        if not bool(torch.isfinite(torch_result[key].float()).all()):
            raise AssertionError(f"backward baseline has non-finite {key}")
        if not bool(torch.isfinite(triton_result[key].float()).all()):
            raise AssertionError(f"backward candidate has non-finite {key}")
    return torch_result, triton_result


def run_backward_benchmark(rank: int, world_size: int, case: CaseSpec):
    """Run one explicit backward case with the published 5/50 host-wall protocol."""
    case = case.validate()
    if case.direction != "backward" or "performance" not in case.tags:
        raise ValueError(f"backward runner received non-performance case {case.case_id}")
    if world_size != case.world_size:
        raise ValueError(f"worker world size does not match {case.case_id}")
    if torch_npu is None or kit.ash is None:
        raise RuntimeError("this benchmark requires torch_npu and ACLSHMEM")

    from mega_moe import moe_backward_triton
    ep_group = dist.group.WORLD
    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(default_gb=2)):
        saved, dy, dtype, device = build_backward_saved(
            case.tokens, case.hidden, case.ffn, case.num_experts, case.topk, ep_group
        )
        peer_mem = kit.make_peer_mem(saved, dtype, rank)
        try:
            torch_gate_result, _ = _backward_gate(
                saved, dy, peer_mem
            )

            def _triton():
                with torch.no_grad():
                    moe_backward_triton(saved, dy, peer_mem)

            def _torch():
                backward_torch_baseline(saved, dy)

            triton_timing, torch_timing = kit.PerformanceRunner(
                _triton,
                _torch,
                BACKWARD_TIMING,
                device=device,
                ep_group=ep_group,
            ).run()
            triton_ms = triton_timing.stats["median_ms"]
            torch_ms = torch_timing.stats["median_ms"]
            speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")

            # Per-stage NPU-event breakdown of the default serial backward path
            # (dispatch / fc2_wgrad / swiglu / fc1_wgrad / combine). The library
            # records 6 events -> 5 intervals, MAX-reduced across ranks, appended
            # to saved["_bwd_stage_samples"]. MOE_BWD_BREAKDOWN=0 disables.
            backward_breakdown = None
            if os.environ.get("MOE_BWD_BREAKDOWN", "1") != "0":
                os.environ["MOE_BWD_STAGE_TIMING"] = "1"
                os.environ["MOE_BWD_DUAL_STREAM"] = "0"   # stage timing needs the serial step2/3 path
                saved["_bwd_stage_samples"] = []
                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)
                _bd_warmup = max(1, BACKWARD_TIMING.warmup)
                for _ in range(_bd_warmup + BACKWARD_TIMING.iterations):
                    with torch.no_grad():
                        moe_backward_triton(saved, dy, peer_mem)
                os.environ.pop("MOE_BWD_STAGE_TIMING", None)
                os.environ.pop("MOE_BWD_DUAL_STREAM", None)   # restore default (ON)
                bd_samples = saved["_bwd_stage_samples"][_bd_warmup:]
                _bwd_stage_names = ("dispatch", "fc2_wgrad", "swiglu", "fc1_wgrad", "combine")
                backward_breakdown = OrderedDict(
                    (f"{name}_event_ms", _stats([s[i] for s in bd_samples], device))
                    for i, name in enumerate(_bwd_stage_names)
                )

            # Combine 3-phase breakdown (serial combine: gemm / push+barrier /
            # reduce) — splits the combine stage into its components. The serial
            # path disables the two-stream group overlap so the phases are clean.
            combine_phase_breakdown = None
            if os.environ.get("MOE_BWD_COMBINE_PHASE", "1") != "0":
                os.environ["MOE_BWD_COMBINE_SERIAL"] = "1"
                os.environ["MOE_COMBINE_PHASE_TIMING"] = "1"
                saved["_combine_phase_samples"] = []
                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)
                _cp_warmup, _cp_iters = 2, 10
                for _ in range(_cp_warmup + _cp_iters):
                    with torch.no_grad():
                        moe_backward_triton(saved, dy, peer_mem)
                os.environ.pop("MOE_BWD_COMBINE_SERIAL", None)
                os.environ.pop("MOE_COMBINE_PHASE_TIMING", None)
                cp_samples = saved["_combine_phase_samples"][_cp_warmup:]
                _cp_names = ("combine_gemm", "combine_push_barrier", "combine_reduce")
                combine_phase_breakdown = OrderedDict(
                    (f"{name}_ms", _stats([s[i] for s in cp_samples], device))
                    for i, name in enumerate(_cp_names)
                )
            entry = {
                "schema_version": 1,
                "direction": "backward",
                "case_id": case.case_id,
                "model": case.model,
                "world_size": world_size,
                "tokens_per_rank": case.tokens,
                "shape": {
                    "hidden": case.hidden,
                    "ffn": case.ffn,
                    "topk": case.topk,
                    "num_experts": case.num_experts,
                },
                "protocol": BACKWARD_TIMING.as_dict(),
                "correctness_gate": {
                    "status": "passed_before_timing",
                    "kind": "structure/shape/dtype/finite/no-exception",
                },
                "metrics": {
                    "torch_ms": torch_ms,
                    "triton_ms": triton_ms,
                    "triton_over_torch": speedup,
                },
                "gradient_gate": {
                    "keys": sorted(torch_gate_result),
                    "comparison": "untimed structure gate",
                },
                "diagnostics": {
                    "backward_stage_slices": backward_breakdown,
                    "combine_phase_slices": combine_phase_breakdown,
                },
            }
            if rank == 0:
                _upsert_result(
                    Path(os.environ.get(
                        "MOE_BACKWARD_BENCH_RESULTS_DIR",
                        str(PROJECT_ROOT / "results" / "backward"),
                    )) / f"bench_backward_suite_w{world_size}.json",
                    "backward",
                    world_size,
                    entry,
                    entry["protocol"],
                )
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


_FORWARD_CASES = kit.make_pytest_params(
    select_cases(direction="forward", tags={"performance"})
)
_BACKWARD_CASES = kit.make_pytest_params(
    select_cases(direction="backward", tags={"performance"})
)


@pytest.mark.dist
@pytest.mark.performance
@pytest.mark.parametrize("spec", _FORWARD_CASES)
def test_bench_forward_case(dist_test, spec: CaseSpec):
    dist_test(
        run_forward_benchmark,
        world_size=spec.world_size,
        args=(spec,),
    )


@pytest.mark.dist
@pytest.mark.performance
@pytest.mark.parametrize("spec", _BACKWARD_CASES)
def test_bench_backward_case(dist_test, spec: CaseSpec):
    dist_test(
        run_backward_benchmark,
        world_size=spec.world_size,
        args=(spec,),
    )


# TODO: future work — when an explicit all-directions session is introduced,
# finish and fully clean forward before starting backward, while still keeping
# operator, peer memory, and ACLSHMEM heaps distinct.  Each parameterized case
# currently owns and finalizes its own session.
