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

The classic performance baseline is Torch-NPU grouped-GEMM + HCCL.  The
separate MoonEP Kimi runner compares balanced and unbalanced production Triton
calls against an independent logical-owner Torch/HCCL golden.  The primary
comparison is direct-call full E2E.  Optional event slices (preprocess,
dispatch+FC1, and the post-dispatch weighted-SwiGLU+FC2+combine segment) can be
enabled for stage diagnosis.  The Ascend post-dispatch segment uses the same
shadow activation schedule as production; the Torch segment runs weighted
SwiGLU followed by FC2+combine serially.  Event slices are diagnostics:
independently rank-reduced statistics are not additive to full E2E.

Protocol: 5 warmup iterations and 50 measured iterations for every metric.  The
classic forward suite uses NPU-event samples.  The MoonEP A1/B/A2 comparison
uses synchronized per-call host-wall samples so device planning, ETC D2H, and
collective stalls remain inside the measured boundary.  Samples are reduced
with rank-MAX; backward preserves its historical host-wall interval.
Before timing, the candidate and grouped baseline must pass normal,
zero-receive/empty-expert, and negative/out-of-range all-drop comparison gates.

Workload profiles:
    KIMI-K3: H=3584, F=3072, top-k=16, E=896,
             tokens/rank in {4096, 8192, 16384}; primary optimization target
    KIMI-K3-TRIMMED: H=3584, F=3072, top-k=8, E=32, EP=8,
                     tokens/rank=16384; multi-machine projection case
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
    MOE_FUSED_ASH_SIZE_GB=16 python -m pytest \
        'benchmark/layer/bench_moe_suite.py::test_bench_moonep_forward_case[performance-fwd-kimi-k3-w8-t4k-moderate-wide]' \
        -m dist -v -s

Case selection is intentionally pytest-native.  Use ``-k forward`` or
``-k backward`` for a direction and a node id (or ``-m kimi``, ``-m dsv4``,
``-m smoke``) for a subset.  ``CaseSpec.tokens`` is always the token count on
one rank; there is no environment-controlled shape or world-size selection.

The remaining environment variables are runtime knobs only:
    MOE_FULL_BENCH_BREAKDOWN=1          # opt into three-segment diagnostics
    MOE_FULL_BENCH_RESULTS_DIR=/tmp/... # optional shared forward result directory
    MOE_BACKWARD_BENCH_RESULTS_DIR=/tmp/... # optional backward result directory
    MOE_FUSED_ASH_SIZE_GB=6             # ACLSHMEM heap size, not shape selection
                                       # MoonEP Kimi cases require at least 16
    MOE_MOONEP_FUSED_BALANCED_COUNT=0   # A/B: restore scatter_add count cube
    MOE_MOONEP_FUSED_ROUTE_MAPPING=0    # A/B: restore Torch route mapping
    MOE_MOONEP_ENABLE_REPLICA_CACHE=1   # A/B: include collective cache check
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import shlex
import statistics
import sys
import time
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
from mega_moe.kernels.combine_fc1_bwd import GATE_PAD
from mega_moe.kernels.fc2_combine import _fc2_device_put_worker_layout
from config import CaseSpec, select_cases
from benchmark.layer._grouped_forward_baseline import GroupedForwardBaseline
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    backward_torch_baseline,
    build_backward_saved,
    torch_moe_fwd_golden,
)


ACTIVATION_DTYPE = torch.bfloat16
ROUTING_INPUT_DTYPE = torch.float32
ROUTING_TRANSPORT_DTYPE = torch.float32
RESULT_CONTRACT = "bf16-activations-fp32-routing-device-put-v3"
MOONEP_RESULT_CONTRACT = "bf16-activations-fp32-routing-moonep-hostwall-v7"
MOONEP_BACKWARD_RESULT_CONTRACT = "bf16-activations-fp32-routing-moonep-backward-native-saved-v2"
# Serial MoonEP backward with grad transport records 7 events -> 6 intervals;
# the last interval is the replica-grad transport (sink+reduce) stage.
_MOONEP_BWD_STAGE_NAMES = (
    "dispatch",
    "fc2_wgrad",
    "swiglu",
    "fc1_wgrad",
    "combine",
    "grad_reduce",
)
_WEIGHT_INIT_CHUNK_BYTES = 128 * 1024 * 1024
MOONEP_MIN_ASH_SIZE_GB = 16
MOONEP_CLEAR_GAIN_RATIO = 1.05
MOONEP_ROUTE_PROFILES = (
    "moderate-wide",
)
_MOONEP_MODERATE_WIDE_OWNER_COUNTS = (19, 27, 11, 11, 15, 15, 15, 15)


@dataclass(frozen=True)
class MoonEPBenchmarkSpec:
    """One explicit shape/profile node in the MoonEP benchmark matrix."""

    case: CaseSpec
    route_profile: str

    def __post_init__(self) -> None:
        _validate_moonep_route_profile(self.route_profile)
        if self.case.world_size != 8:
            raise ValueError("MoonEP performance profiles are intentionally W8-only")

    @property
    def case_id(self) -> str:
        return f"{self.case.case_id}-{self.route_profile}"

    @property
    def tags(self) -> frozenset[str]:
        return self.case.tags


def _validate_moonep_route_profile(route_profile: str) -> str:
    if route_profile not in MOONEP_ROUTE_PROFILES:
        raise ValueError(
            f"route_profile must be one of {MOONEP_ROUTE_PROFILES}, "
            f"got {route_profile!r}"
        )
    return route_profile


def _moonep_owner_period(
    world_size: int, route_profile: str
) -> tuple[int, ...]:
    """Return the selected deterministic owner-load period."""
    _validate_moonep_route_profile(route_profile)
    if world_size != 8:
        raise ValueError("MoonEP performance profiles are intentionally W8-only")
    return tuple(
        owner
        for owner, count in enumerate(_MOONEP_MODERATE_WIDE_OWNER_COUNTS)
        for _ in range(count)
    )


def _moonep_receive_capacity_factor(
    world_size: int, route_profile: str
) -> float:
    """Return the smallest buffer factor that can run the unbalanced route."""
    period = _moonep_owner_period(world_size, route_profile)
    return world_size * max(
        period.count(owner) for owner in range(world_size)
    ) / len(period)


def _moonep_active_local_experts(
    period: tuple[int, ...], experts_per_rank: int, local_shift: int
) -> list[set[int]]:
    """Derive active local ids over one owner/local-id joint period."""
    joint_period = math.lcm(len(period), experts_per_rank)
    active = [set() for _ in range(len(_MOONEP_MODERATE_WIDE_OWNER_COUNTS))]
    for route_id in range(joint_period):
        active[period[route_id % len(period)]].add(
            (route_id + local_shift) % experts_per_rank
        )
    return active


def _expected_moonep_layout(
    case: CaseSpec,
    world_size: int,
    *,
    route_profile: str,
) -> tuple[list[int], list[list[int]]]:
    """Independently derive the fixed W8 moderate-wide assignment."""
    _moonep_owner_period(world_size, route_profile)
    remote_routes = [
        0,
        0,
        5 * case.tokens,
        5 * case.tokens,
        case.tokens,
        case.tokens,
        case.tokens,
        case.tokens,
    ]
    copy_counts = [0, 0, 18, 18, 4, 4, 4, 4]
    copied_experts = [[-2] * count for count in copy_counts]
    return remote_routes, copied_experts


def _moonep_copy_layout_matches(
    actual: list[list[int]],
    expected: list[list[int]],
    experts_per_rank: int,
    *,
    route_profile: str,
) -> bool:
    """Check the moderate-wide copy counts and source-owner assignment."""
    _validate_moonep_route_profile(route_profile)
    if [len(row) for row in actual] != [len(row) for row in expected]:
        return False
    expected_owners = [None, None, 1, 1, 0, 0, 0, 1]
    return all(
        not row
        if owner is None
        else all(expert // experts_per_rank == owner for expert in row)
        for row, owner in zip(actual, expected_owners)
    )
# This is the published protocol.  Keep debug/short runs under a differently
# named script so their output cannot be mistaken for 5/50 evidence.
FORWARD_TIMING = kit.FORWARD_TIMING
BACKWARD_TIMING = kit.BACKWARD_TIMING
MOONEP_FORWARD_TIMING = kit.TimingSpec(clock="host_wall")
WARMUP_ITERS = FORWARD_TIMING.warmup
BENCH_ITERS = FORWARD_TIMING.iterations
# Production forward uses FC2 shadow activation. The optional three-segment
# slices exercise the same schedule, but are not part of every E2E run.
RUN_BREAKDOWN = os.environ.get("MOE_FULL_BENCH_BREAKDOWN", "0") == "1"
G_ASH_SIZE_GB = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", "6"))
MOONEP_FUSED_BALANCED_COUNT = (
    os.environ.get("MOE_MOONEP_FUSED_BALANCED_COUNT", "1") != "0"
)
MOONEP_FUSED_ROUTE_MAPPING = (
    os.environ.get("MOE_MOONEP_FUSED_ROUTE_MAPPING", "1") != "0"
)
MOONEP_ENABLE_REPLICA_CACHE = (
    os.environ.get("MOE_MOONEP_ENABLE_REPLICA_CACHE", "1") != "0"
)
if MOONEP_FUSED_ROUTE_MAPPING and not MOONEP_FUSED_BALANCED_COUNT:
    raise ValueError(
        "MOE_MOONEP_FUSED_ROUTE_MAPPING requires "
        "MOE_MOONEP_FUSED_BALANCED_COUNT"
    )
RESULTS_DIR = os.environ.get(
    "MOE_FULL_BENCH_RESULTS_DIR",
    str(PROJECT_ROOT / "results" / "forward"),
)

if G_ASH_SIZE_GB <= 0:
    raise ValueError("MOE_FUSED_ASH_SIZE_GB must be a positive integer")
G_ASH_SIZE = G_ASH_SIZE_GB * 1024 * 1024 * 1024


def _benchmark_provenance(result_contract=RESULT_CONTRACT):
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
    if result_contract == MOONEP_RESULT_CONTRACT:
        production_sources += (
            PROJECT_ROOT / "src" / "mega_moe" / "runtime" / "balanced_routing.py",
            PROJECT_ROOT / "src" / "mega_moe" / "runtime" / "moonep_planning.py",
            PROJECT_ROOT
            / "src"
            / "mega_moe"
            / "runtime"
            / "replica_weight_prefetch.py",
            PROJECT_ROOT
            / "src"
            / "mega_moe"
            / "kernels"
            / "replica_weight_prefetch.py",
            PROJECT_ROOT / "tests" / "_moe_baselines.py",
        )
    forward_source_sha256 = {}
    for path in production_sources:
        with open(path, "rb") as source_file:
            forward_source_sha256[str(path.relative_to(PROJECT_ROOT))] = (
                hashlib.sha256(source_file.read()).hexdigest()
            )
    return {
        "result_contract": result_contract,
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
    dispatch_tile_m = 128
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


def _required_moonep_ash_bytes(case: CaseSpec, world_size: int) -> int:
    """Estimate fixed-B MoonEP symmetric payload before allocator overhead."""
    experts_per_rank = case.num_experts // world_size
    physical_experts_per_rank = 2 * experts_per_rank
    max_recv_rows = int(case.tokens * case.topk * case.capacity_factor)
    token_peer_bytes = max_recv_rows * case.hidden * ACTIVATION_DTYPE.itemsize
    combine_bytes = (
        case.tokens * case.topk * case.hidden * ACTIVATION_DTYPE.itemsize
    )
    routing_peer_bytes = max_recv_rows * ROUTING_TRANSPORT_DTYPE.itemsize
    dispatch_tile_m = 128
    max_source_tiles = (
        case.tokens * case.topk + dispatch_tile_m - 1
    ) // dispatch_tile_m
    dispatch_signal_slots = (
        world_size * physical_experts_per_rank * max_source_tiles
    )
    replica_signal_slots = 2 * experts_per_rank
    signal_bytes = (
        (dispatch_signal_slots + replica_signal_slots)
        * 16
        * torch.int32.itemsize
    )
    num_buckets = world_size * physical_experts_per_rank
    metadata_bins = 1 << (num_buckets - 1).bit_length()
    metadata_bytes = world_size * metadata_bins * torch.int32.itemsize
    planning_bins = 1 << case.num_experts.bit_length()
    planning_bytes = world_size * planning_bins * torch.int32.itemsize
    replica_weight_bytes = (
        experts_per_rank
        * case.hidden
        * (3 * case.ffn)
        * ACTIVATION_DTYPE.itemsize
    )
    return (
        token_peer_bytes
        + combine_bytes
        + routing_peer_bytes
        + signal_bytes
        + metadata_bytes
        + planning_bytes
        + replica_weight_bytes
    )


def _required_moonep_free_hbm_bytes(case: CaseSpec, world_size: int) -> int:
    """Conservative preflight for weights plus golden/runtime temporaries."""
    experts_per_rank = case.num_experts // world_size
    local_weight_bytes = (
        experts_per_rank
        * case.hidden
        * (3 * case.ffn)
        * ACTIVATION_DTYPE.itemsize
    )
    route_rows = case.tokens * case.topk
    route_hidden_bytes = (
        route_rows * case.hidden * ACTIVATION_DTYPE.itemsize
    )
    max_recv_rows = int(route_rows * case.capacity_factor)
    recv_ffn_bytes = max_recv_rows * case.ffn * ACTIVATION_DTYPE.itemsize
    reserve = 8 * 1024**3
    golden_peak = (
        local_weight_bytes
        + 8 * route_hidden_bytes
        + 6 * recv_ffn_bytes
        + reserve
    )
    runtime_peak = (
        local_weight_bytes
        + G_ASH_SIZE
        + 4 * route_hidden_bytes
        + reserve
    )
    return max(golden_peak, runtime_peak)


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
def _prepare_moonep_inputs(
    case: CaseSpec,
    rank: int,
    world_size: int,
    device,
    *,
    route_profile: str,
):
    """Build the deterministic W8 moderate-wide Kimi route profile."""
    if world_size != 8 or case.topk != 16 or case.num_experts != 896:
        raise ValueError(
            "the MoonEP Kimi benchmark requires W8, top-k=16, E=896"
        )
    owner_period_values = _moonep_owner_period(world_size, route_profile)
    if case.tokens * case.topk % len(owner_period_values):
        raise ValueError("MoonEP token case must contain an integral owner period")

    torch.manual_seed(43 + rank * 1000)
    hidden_states = torch.randn(
        (case.tokens, case.hidden), dtype=ACTIVATION_DTYPE, device=device
    ).mul_(0.5).contiguous()

    owner_period = torch.tensor(
        owner_period_values, dtype=torch.int32, device=device
    )
    route_ids = torch.arange(
        case.tokens * case.topk, dtype=torch.int32, device=device
    )
    owners = owner_period[route_ids.remainder(len(owner_period_values))]
    experts_per_rank = case.num_experts // world_size
    global_route_ids = route_ids.to(torch.int64) + (
        rank * case.tokens * case.topk
    )
    local_experts = global_route_ids.remainder(experts_per_rank)
    selected_experts = (
        owners.to(torch.int64) * experts_per_rank + local_experts
    ).view(case.tokens, case.topk).to(torch.int32).contiguous()
    sorted_per_token = torch.sort(selected_experts, dim=1).values
    if bool((sorted_per_token[:, 1:] == sorted_per_token[:, :-1]).any().item()):
        raise AssertionError("MoonEP synthetic routes repeat an expert within a token")
    if (
        int(selected_experts.min().item()) < 0
        or int(selected_experts.max().item()) >= case.num_experts
    ):
        raise AssertionError("MoonEP synthetic routes contain an invalid expert")
    # Rank-seeded, non-uniform gates make route-slot restoration observable.
    route_logits = torch.randn(
        (case.tokens, case.topk), dtype=ROUTING_INPUT_DTYPE, device=device
    )
    routing_weights = F.softmax(route_logits, dim=-1).contiguous()
    return hidden_states, selected_experts, routing_weights


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


@torch.no_grad()
def _summarize_moonep_owner_routes(case, selected_experts, world_size):
    """Return global logical-owner route counts for the MoonEP workload."""
    experts_per_rank = case.num_experts // world_size
    owners = torch.div(
        selected_experts.reshape(-1).to(torch.int64),
        experts_per_rank,
        rounding_mode="floor",
    )
    local_counts = torch.bincount(owners, minlength=world_size).to(torch.int64)
    dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
    expected_total = case.tokens * case.topk * world_size
    if int(local_counts.sum().item()) != expected_total:
        raise AssertionError("MoonEP owner route histogram lost routes")
    return local_counts.cpu().tolist()


# The grouped performance implementation lives in
# ``_grouped_forward_baseline.GroupedForwardBaseline`` so it cannot be
# confused with the independent correctness references in ``tests``.

def ascend_full_post_routing(op, hidden_states, selected_experts, packed_w1, down_weight, routing_weights, *, return_saved=False):
    """One direct production full-forward call; this is the primary candidate boundary."""
    return op.forward(hidden_states, selected_experts, packed_w1, down_weight, routing_weights, return_saved=return_saved)


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
        actual_fp32 = actual.float()
        expected_fp32 = expected.float()
        diff = (actual_fp32 - expected_fp32).abs()
        tolerance = 5e-2 + 5e-2 * expected_fp32.abs()
        nonfinite = ~(torch.isfinite(actual_fp32) & torch.isfinite(expected_fp32))
        bad = (diff > tolerance) | nonfinite
        max_flat = int(diff.nan_to_num(posinf=float("inf")).argmax().item())
        token = max_flat // actual.shape[1]
        feature = max_flat % actual.shape[1]
        message = (
            f"{str(exc).splitlines()[0]}; bad={int(bad.sum().item())}/"
            f"{actual.numel()}, max_abs={float(diff.nan_to_num().max().item()):.6g}, "
            f"at=({token},{feature}), actual="
            f"{float(actual_fp32[token, feature].item()):.6g}, expected="
            f"{float(expected_fp32[token, feature].item()):.6g}, "
            f"actual_nonfinite={int((~torch.isfinite(actual_fp32)).sum().item())}"
        )
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


@torch.no_grad()
def _logical_torch_golden(
    case,
    ep_group,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
):
    """Run the independent logical-owner Torch/HCCL oracle once."""
    ffn_size = packed_w1.shape[2] // 2
    gate_weight = packed_w1[:, :, :ffn_size].transpose(1, 2)
    up_weight = packed_w1[:, :, ffn_size:].transpose(1, 2)
    return torch_moe_fwd_golden(
        hidden_states,
        routing_weights,
        selected_experts,
        gate_weight,
        up_weight,
        down_weight,
        case.num_experts,
        ep_group,
    )


def _make_forward_op(
    case,
    ep_group,
    *,
    enable_moonep,
    moonep_replica_gate_up_chunk_bytes=16 * 1024 * 1024,
    moonep_replica_down_chunk_bytes=4 * 1024 * 1024,
    moonep_replica_down_early_programs=8,
    moonep_replica_down_early_descriptors_per_program=2,
):
    config = MoEForwardConfig(
        receive_capacity_factor=case.capacity_factor,
        enable_moonep=enable_moonep,
        moonep_replica_gate_up_chunk_bytes=(
            moonep_replica_gate_up_chunk_bytes
        ),
        moonep_replica_down_chunk_bytes=(
            moonep_replica_down_chunk_bytes
        ),
        moonep_replica_down_early_programs=(
            moonep_replica_down_early_programs
        ),
        moonep_replica_down_early_descriptors_per_program=(
            moonep_replica_down_early_descriptors_per_program
        ),
        moonep_fused_balanced_count=MOONEP_FUSED_BALANCED_COUNT,
        moonep_fused_route_mapping=MOONEP_FUSED_ROUTE_MAPPING,
        moonep_enable_replica_cache=MOONEP_ENABLE_REPLICA_CACHE,
    )
    return FusedMoEForward(
        ep_group,
        max_tokens_per_rank=case.tokens,
        hidden_size=case.hidden,
        top_k=case.topk,
        num_experts=case.num_experts,
        config=config,
    )


def _measure_moonep_forward_e2e(
    device,
    ep_group,
    op,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
    *,
    force_replica_refill: bool = False,
):
    def run_forward():
        if force_replica_refill:
            # A production layer is invoked once, so each statistical sample
            # must start from the same non-resident replica-weight state.  Keep
            # compiled kernels and allocated buffers, but never reuse weights
            # from a preceding benchmark sample.
            op._replica_weight_cache_valid = False
        return ascend_full_post_routing(
            op,
            hidden_states,
            selected_experts,
            packed_w1,
            down_weight,
            routing_weights,
        )

    for _ in range(MOONEP_FORWARD_TIMING.warmup):
        output = run_forward()
        del output
    _sync_ranks_before_event(device, ep_group)

    local_samples = []
    for _ in range(MOONEP_FORWARD_TIMING.iterations):
        _sync_ranks_before_event(device, ep_group)
        start = time.perf_counter()
        output = run_forward()
        torch.npu.synchronize(device)
        local_samples.append((time.perf_counter() - start) * 1000.0)
        del output
    return kit.TimingResult(
        _sample_rank_max(local_samples, device, ep_group)
    )


def _assert_replica_prefetch_state(
    op,
    expected_experts,
    expected_epoch,
    device,
    ep_group,
    label,
):
    """Collectively prove that a forward consumed a concrete replica cache."""
    ok = True
    message = ""
    try:
        if not op._replica_weight_cache_valid:
            raise AssertionError("replica cache is not valid")
        if op._replica_prefetch_pending:
            raise AssertionError("replica prefetch remains pending")
        if op._replica_weight_epoch != expected_epoch:
            raise AssertionError(
                f"epoch={op._replica_weight_epoch}, expected={expected_epoch}"
            )
        if op._active_replica_weight_epoch != expected_epoch - 1:
            raise AssertionError(
                "active replica epoch does not match the published ready slots"
            )
        cached = op._replica_experts_cache
        if cached is None or not torch.equal(cached, expected_experts):
            raise AssertionError("cached experts_to_copy differs from the plan")
        active_epoch = expected_epoch - 1
        local_row = expected_experts[op.rank]
        occupied_slots = torch.where(local_row >= 0)[0].tolist()
        gate_epochs = op.context.replica_gate_ready.view(-1, 16)[:, 0].cpu()
        down_epochs = op.context.replica_down_ready.view(-1, 16)[:, 0].cpu()
        for slot in occupied_slots:
            if (
                int(gate_epochs[slot].item()) != active_epoch
                or int(down_epochs[slot].item()) != active_epoch
            ):
                raise AssertionError(
                    f"replica slot {slot} did not publish epoch {active_epoch}"
                )
    except AssertionError as exc:
        ok = False
        message = str(exc)
    flag = torch.tensor([int(ok)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if not bool(flag.item()):
        raise AssertionError(f"{label} replica prefetch state failed: {message}")


def _assert_collective_difference(
    actual, expected, device, name, ep_group
):
    """Require one poisoned destination to affect a global output."""
    finite = (
        torch.isfinite(actual).all() & torch.isfinite(expected).all()
    ).to(torch.int32).reshape(1)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=ep_group)
    if not bool(finite.item()):
        raise AssertionError(f"{name} produced a non-finite output")
    max_difference = (actual.float() - expected.float()).abs().max()
    dist.all_reduce(max_difference, op=dist.ReduceOp.MAX, group=ep_group)
    if float(max_difference.item()) <= 1e-3:
        raise AssertionError(
            f"{name} did not change the global output"
        )


def _prove_replica_weight_consumption(
    op,
    expected_experts,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
    clean_output,
    device,
    ep_group,
    *,
    weight_name,
    destination_rank,
):
    """Poison one cached table, prove execution reads it, then refill it."""
    buffers = op._replica_weight_buffers
    if buffers is None:
        raise AssertionError("replica buffers were not allocated")
    if weight_name == "gate_up":
        replica_weight = buffers.gate_up
    elif weight_name == "down":
        replica_weight = buffers.down
    else:
        raise ValueError("weight_name must be gate_up or down")

    occupied_slots = (
        torch.where(expected_experts[destination_rank] >= 0)[0].tolist()
        if op.rank == destination_rank
        else []
    )
    for slot in occupied_slots:
        replica_weight[slot].zero_()
    torch.npu.synchronize(device)
    dist.barrier(group=ep_group)

    poisoned = ascend_full_post_routing(
        op,
        hidden_states,
        selected_experts,
        packed_w1,
        down_weight,
        routing_weights,
    )
    _assert_collective_difference(
        poisoned,
        clean_output,
        device,
        f"MoonEP replica {weight_name} execution proof on rank {destination_rank}",
        ep_group,
    )
    torch.npu.synchronize(device)
    del poisoned

    refill_epoch = op._replica_weight_epoch
    op._replica_weight_cache_valid = False
    restored = ascend_full_post_routing(
        op,
        hidden_states,
        selected_experts,
        packed_w1,
        down_weight,
        routing_weights,
    )
    _assert_close_collective(
        restored,
        clean_output,
        device,
        f"MoonEP replica {weight_name} rank {destination_rank} refill",
        ep_group,
    )
    torch.npu.synchronize(device)
    del restored
    _assert_replica_prefetch_state(
        op,
        expected_experts,
        refill_epoch + 1,
        device,
        ep_group,
        f"replica {weight_name} rank {destination_rank} refill",
    )


def _summary_experts_to_copy(summary, experts_per_rank):
    rows = summary["copied_experts_by_rank"]
    table = torch.full(
        (len(rows), experts_per_rank), -1, dtype=torch.int32
    )
    for destination, experts in enumerate(rows):
        for slot, expert in enumerate(experts):
            table[destination, slot] = expert
    return table


def _finalize_forward_op(op, device, ep_group):
    """Collectively release one operator before allocating the next one."""
    torch.npu.synchronize(device)
    dist.barrier(group=ep_group)
    op.finalize()
    torch.npu.empty_cache()
    dist.barrier(group=ep_group)


def _log_moonep_phase(rank, case, phase):
    if rank == 0:
        print(f"[MoonEP][{case.case_id}] {phase}", flush=True)


def _all_finite_memory_lean(value, chunk_rows=16):
    """isfinite gate that never materializes a full fp32 copy of big grads.

    The MoonEP w8 kimi weight grads are multi-GiB bf16 tensors; a plain
    ``value.float()`` transiently doubles them and overflows the HBM budget
    once the symmetric heap and the replica mirrors are resident.
    """
    if value.dim() == 0 or value.numel() * value.element_size() <= (1 << 28):
        return bool(torch.isfinite(value.float()).all())
    for start in range(0, value.shape[0], chunk_rows):
        chunk = value[start : start + chunk_rows]
        if not bool(torch.isfinite(chunk.float()).all()):
            return False
    return True


@torch.no_grad()
def _collect_moonep_plan_summary(op, case, selected_experts, ep_group):
    """Build one untimed plan and expose its physical load/replica decisions."""
    plan = op.build_routing_plan(selected_experts)
    experts_per_rank = case.num_experts // op.world_size
    local = torch.tensor(
        [
            plan.num_received_routes,
            int(plan.received_routes_per_expert[experts_per_rank:].sum().item()),
        ],
        dtype=torch.int64,
        device=selected_experts.device,
    )
    gathered = [torch.empty_like(local) for _ in range(op.world_size)]
    dist.all_gather(gathered, local, group=ep_group)
    balanced_routes = [int(item[0].item()) for item in gathered]
    remote_routes = [int(item[1].item()) for item in gathered]
    expected_balanced = case.tokens * case.topk
    if balanced_routes != [expected_balanced] * op.world_size:
        raise AssertionError(
            "MoonEP plan did not balance every destination to routes-per-source"
        )
    if sum(remote_routes) <= 0:
        raise AssertionError("MoonEP plan did not assign any route to a replica")

    copied_experts = []
    for row in plan.experts_to_copy.cpu().tolist():
        copied_experts.append([int(expert) for expert in row if expert >= 0])
    for rank, count in enumerate(remote_routes):
        if bool(count) != bool(copied_experts[rank]):
            raise AssertionError(
                "MoonEP remote-route and copied-expert presence disagree on "
                f"rank {rank}"
            )
        for expert in copied_experts[rank]:
            if expert // experts_per_rank == rank:
                raise AssertionError(
                    f"rank {rank} copied home expert {expert}, not a remote expert"
                )
    expected_active_experts = experts_per_rank + max(
        (len(row) for row in copied_experts), default=0
    )
    if plan.active_physical_experts_per_rank != expected_active_experts:
        raise AssertionError(
            "MoonEP active physical expert high-water does not match the "
            "planned replica slots"
        )
    replica_bytes_per_expert = (
        case.hidden * 3 * case.ffn * ACTIVATION_DTYPE.itemsize
    )
    return {
        "balanced_routes_per_rank": balanced_routes,
        "remote_replica_routes_per_rank": remote_routes,
        "total_remote_replica_routes": sum(remote_routes),
        "copied_experts_by_rank": copied_experts,
        "copied_expert_count_by_rank": [len(row) for row in copied_experts],
        "active_physical_experts_per_rank": (
            plan.active_physical_experts_per_rank
        ),
        "prefetch_payload_bytes_by_rank": [
            len(row) * replica_bytes_per_expert for row in copied_experts
        ],
    }


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


def _print_moonep_entry(entry):
    metrics = entry["metrics"]
    unbalanced = metrics["triton_unbalanced_full_direct_e2e_ms"]
    balanced = metrics["triton_moonep_balanced_full_direct_e2e_ms"]
    gate = entry["performance_gate"]
    print(
        f"  {entry['case_id']} full-forward median: "
        f"unbalanced={unbalanced['median_ms']:.3f} ms  "
        f"MoonEP balanced={balanced['median_ms']:.3f} ms  "
        f"speedup={entry['triton_unbalanced_over_moonep_balanced_median']:.3f}x  "
        f"performance_gate={gate['status']}",
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
        entry_provenance = entry.get("provenance", {})
        entry_source_fingerprint = (
            entry_provenance.get("benchmark_source_sha256"),
            json.dumps(
                entry_provenance.get("forward_source_sha256", {}),
                sort_keys=True,
            ),
        )
        existing_source_fingerprints = {
            (
                item.get("provenance", {}).get("benchmark_source_sha256"),
                json.dumps(
                    item.get("provenance", {}).get(
                        "forward_source_sha256", {}
                    ),
                    sort_keys=True,
                ),
            )
            for item in existing.get("cases", [])
            if isinstance(item, dict)
        }
        if existing_source_fingerprints - {entry_source_fingerprint}:
            raise ValueError(
                f"refusing to mix benchmark source revisions in {path}; "
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


def _moonep_protocol():
    """Serialize the normal one-forward-per-sample measurement clock."""
    return OrderedDict(
        {
            "implementation": {
                "direct_bucket_scatter": True,
                "clear_gain_target_enforced": False,
            },
            "forward": {
                **MOONEP_FORWARD_TIMING.as_dict(),
                "sample_scope": "one synchronized full forward call",
                "includes": (
                    "device planning, ETC D2H, weight-cache checks, collectives, "
                    "NPU kernels, and final device synchronization"
                ),
            },
        }
    )


def _make_moonep_entry(
    case,
    world_size,
    route_profile,
    measured,
    owner_routes,
    balanced_summary,
    required_ash_bytes,
    hbm_preflight,
):
    owner_period = _moonep_owner_period(world_size, route_profile)
    experts_per_rank = case.num_experts // world_size
    active_local_experts = _moonep_active_local_experts(
        owner_period, experts_per_rank, local_shift=0
    )
    active_experts = {
        owner * experts_per_rank + local_expert
        for owner, local_ids in enumerate(active_local_experts)
        for local_expert in local_ids
    }
    local_expert_description = "global route ordinal modulo B"
    expected_mean = case.tokens * case.topk
    owner_ratios = [round(value / expected_mean, 3) for value in owner_routes]
    unbalanced_median = statistics.median(
        measured["unbalanced_samples_ms"]
    )
    balanced_median = statistics.median(measured["balanced_samples_ms"])
    raw_speedup = (
        unbalanced_median / balanced_median if balanced_median > 0 else 0.0
    )
    speedup = round(raw_speedup, 6)
    device_properties = torch_npu.npu.get_device_properties(0)
    return OrderedDict(
        {
            "schema_version": 1,
            "direction": "forward",
            "case_id": (
                f"{case.case_id}-moonep-{route_profile}-balanced"
            ),
            "base_case_id": case.case_id,
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
            },
            "protocol": _moonep_protocol(),
            "correctness_gate": {
                "status": "passed_before_and_after_timing",
                "baseline": "distributed logical Torch owner-expert golden",
                "comparisons": [
                    "unbalanced Triton vs logical golden",
                    "MoonEP balanced Triton vs logical golden",
                    "t4k: each planned replica destination poisoned changes output, then refill restores it",
                    "t4k: gate/up and down tables are both consumed by FC1/FC2",
                    "MoonEP output after timing vs logical golden",
                    "unbalanced A1 and A2 outputs after timing vs logical golden",
                ],
                "replica_execution_proof": {
                    "status": (
                        "passed_for_every_planned_destination"
                        if case.tokens == 4096
                        else "not_repeated_in_this_case"
                    ),
                    "method": (
                        "zero cached gate/up and down weights one destination "
                        "at a time; require output change; refill and restore"
                    ),
                    "suite_coverage": (
                        "the t4k case proves the shared FC1/FC2 replica path; "
                        "t8k/t16k retain per-case logical golden and replica metadata gates"
                    ),
                },
                "rtol": 5e-2,
                "atol": 5e-2,
            },
            "activation_dtype": "bfloat16",
            "routing_weight_input_dtype": "float32",
            "measured_boundary": (
                "post-router full forward; router/top-k generation excluded"
            ),
            "performance_measurement": {
                "clock": "host_wall",
                "sample_scope": "one synchronized full forward call",
                "includes": (
                    "device MoonEP planning, ETC D2H, optional weight-cache "
                    "checks, "
                    "HCCL/ACLSHMEM collectives, NPU kernels, and final device "
                    "synchronization"
                ),
                "clear_gain_target": MOONEP_CLEAR_GAIN_RATIO,
            },
            "synthetic_input_generation": {
                "route_profile": route_profile,
                "hidden_states": "rank-seeded BF16 normal values scaled by 0.5",
                "routing_weights": "rank-seeded FP32 softmax over route slots",
                "owner_period": list(owner_period),
                "owner_period_length": len(owner_period),
                "owner_routes_per_period": [
                    owner_period.count(owner) for owner in range(world_size)
                ],
                "local_expert": local_expert_description,
                "active_global_experts": len(active_experts),
            },
            "moonep": {
                "enabled": True,
                "direct_bucket_scatter": True,
                "fused_balanced_count": MOONEP_FUSED_BALANCED_COUNT,
                "fused_route_mapping": MOONEP_FUSED_ROUTE_MAPPING,
                "replica_cache_enabled": MOONEP_ENABLE_REPLICA_CACHE,
                "replica_budget_B": case.num_experts // world_size,
                "physical_experts_per_rank_P": 2 * case.num_experts // world_size,
                "owner_routes_per_rank": owner_routes,
                "owner_load_ratio_to_mean": owner_ratios,
                **balanced_summary,
            },
            "symmetric_heap_size_gb": G_ASH_SIZE_GB,
            "estimated_moonep_ash_bytes": required_ash_bytes,
            "estimated_moonep_ash_gib": round(
                required_ash_bytes / (1024**3), 3
            ),
            "hbm_preflight": hbm_preflight,
            "metrics": OrderedDict(
                {
                    "triton_unbalanced_full_direct_e2e_ms": measured[
                        "unbalanced"
                    ],
                    "triton_moonep_balanced_full_direct_e2e_ms": measured[
                        "balanced"
                    ],
                }
            ),
            "triton_unbalanced_over_moonep_balanced_median": speedup,
            "performance_gate": {
                "status": (
                    "clear_gain"
                    if raw_speedup >= MOONEP_CLEAR_GAIN_RATIO
                    else "needs_optimization"
                ),
                "target_speedup": MOONEP_CLEAR_GAIN_RATIO,
                "measured_speedup": speedup,
            },
            "diagnostics": {
                "rank_max_samples_ms": {
                    "unbalanced_before": measured[
                        "unbalanced_before_samples_ms"
                    ],
                    "balanced": measured["balanced_samples_ms"],
                    "unbalanced_after": measured[
                        "unbalanced_after_samples_ms"
                    ],
                    "unbalanced_pooled": measured[
                        "unbalanced_samples_ms"
                    ],
                },
                "measurement_order": [
                    "unbalanced_before",
                    "balanced",
                    "unbalanced_after",
                ],
                "unbalanced_bracketing": {
                    "aggregation": "pooled samples from the before/after passes",
                    "samples_per_pass": FORWARD_TIMING.iterations,
                    "before": measured["unbalanced_before"],
                    "after": measured["unbalanced_after"],
                },
                "legacy_breakdown": "disabled for the MoonEP production forward",
                "invalid_route_edge_gate": (
                    "disabled because the current MoonEP planner is dropless-only"
                ),
            },
            "provenance": _benchmark_provenance(MOONEP_RESULT_CONTRACT),
        }
    )


def run_moonep_forward_benchmark(
    rank: int,
    world_size: int,
    case: CaseSpec,
    route_profile: str,
):
    """Run the independent W4/W8 Kimi MoonEP correctness/performance case."""
    case = case.validate()
    _validate_moonep_route_profile(route_profile)
    if (
        case.direction != "forward"
        or "performance" not in case.tags
        or case.model.upper() != "KIMI-K3"
        or world_size not in (4, 8)
        or case.world_size != world_size
    ):
        raise ValueError(
            "MoonEP benchmark only accepts performance-fwd-kimi-k3 W8 cases"
        )
    case = replace(
        case,
        capacity_factor=max(
            case.capacity_factor,
            _moonep_receive_capacity_factor(world_size, route_profile),
        ),
    ).validate()
    if torch_npu is None or kit.ash is None:
        raise RuntimeError("MoonEP benchmark requires torch_npu and ACLSHMEM")
    if G_ASH_SIZE_GB < MOONEP_MIN_ASH_SIZE_GB:
        raise RuntimeError(
            "MoonEP Kimi benchmark requires "
            f"MOE_FUSED_ASH_SIZE_GB>={MOONEP_MIN_ASH_SIZE_GB}; got "
            f"{G_ASH_SIZE_GB}"
        )
    required_ash_bytes = _required_moonep_ash_bytes(case, world_size)
    if required_ash_bytes >= G_ASH_SIZE:
        raise RuntimeError(
            f"{case.case_id} MoonEP estimate is "
            f"{required_ash_bytes / (1024**3):.3f} GiB, but "
            f"MOE_FUSED_ASH_SIZE_GB={G_ASH_SIZE_GB}"
        )

    _log_moonep_phase(rank, case, "checking HBM capacity")
    required_free_hbm = _required_moonep_free_hbm_bytes(case, world_size)
    local_free_hbm, local_total_hbm = torch.npu.mem_get_info()
    hbm_info = torch.tensor(
        [local_free_hbm, local_total_hbm], dtype=torch.int64, device=f"npu:{rank}"
    )
    dist.all_reduce(hbm_info, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
    min_free_hbm, min_total_hbm = (int(value) for value in hbm_info.cpu().tolist())
    if min_free_hbm < required_free_hbm:
        raise RuntimeError(
            f"{case.case_id} requires an estimated "
            f"{required_free_hbm / (1024**3):.3f} GiB free HBM per rank, "
            f"but the least-free rank has {min_free_hbm / (1024**3):.3f} GiB"
        )
    hbm_preflight = {
        "minimum_free_hbm_bytes_before_allocation": min_free_hbm,
        "minimum_total_hbm_bytes": min_total_hbm,
        "required_free_hbm_bytes": required_free_hbm,
        "status": "passed",
    }

    ep_group = dist.group.WORLD
    device = f"npu:{rank}"
    experts_per_rank = case.num_experts // world_size
    _log_moonep_phase(rank, case, "allocating weights and deterministic hot routes")
    packed_w1, down_weight, _ = _make_local_weights(
        case, experts_per_rank, rank, device
    )
    hidden_states, selected_experts, routing_weights = _prepare_moonep_inputs(
        case, rank, world_size, device, route_profile=route_profile
    )
    owner_routes = _summarize_moonep_owner_routes(
        case, selected_experts, world_size
    )
    mean_routes = case.tokens * case.topk
    owner_period = _moonep_owner_period(world_size, route_profile)
    expected_owner_routes = [
        world_size
        * mean_routes
        * owner_period.count(owner)
        // len(owner_period)
        for owner in range(world_size)
    ]
    if owner_routes != expected_owner_routes:
        raise AssertionError(
            f"unexpected MoonEP logical-owner loads: {owner_routes}"
        )
    dist.barrier(group=ep_group)

    # Pay the independent logical-owner Torch/HCCL goldens before either
    # production operator allocates its symmetric workspaces.
    _log_moonep_phase(rank, case, "running independent logical-expert Torch golden")
    expected = _logical_torch_golden(
        case,
        ep_group,
        hidden_states,
        selected_experts,
        routing_weights,
        packed_w1,
        down_weight,
    )
    torch.npu.synchronize(device)
    dist.barrier(group=ep_group)
    # The golden's large route-major temporaries are dead; release allocator
    # cache before reserving the 16+ GiB ACLSHMEM heap.
    torch.npu.empty_cache()

    measured = {}
    unbalanced_before = None
    balanced_result = None
    balanced_summary = None
    with kit.aclshmem_session(rank, world_size, G_ASH_SIZE):
        _log_moonep_phase(rank, case, "validating and measuring unbalanced pass A1")
        unbalanced_op = _make_forward_op(
            case, ep_group, enable_moonep=False
        )
        try:
            actual = ascend_full_post_routing(
                unbalanced_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                actual,
                expected,
                device,
                "unbalanced Triton vs logical Torch golden",
                ep_group,
            )
            del actual
            unbalanced_before = _measure_moonep_forward_e2e(
                device,
                ep_group,
                unbalanced_op,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                down_weight,
            )
            unbalanced_after_a1_timing = ascend_full_post_routing(
                unbalanced_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                unbalanced_after_a1_timing,
                expected,
                device,
                "unbalanced A1 after timing vs logical Torch golden",
                ep_group,
            )
            del unbalanced_after_a1_timing
        finally:
            _finalize_forward_op(unbalanced_op, device, ep_group)

        _log_moonep_phase(rank, case, "validating MoonEP plan and replica prefetch")
        balanced_op = _make_forward_op(
            case,
            ep_group,
            enable_moonep=True,
        )
        try:
            balanced_summary = _collect_moonep_plan_summary(
                balanced_op, case, selected_experts, ep_group
            )
            (
                expected_remote_routes,
                expected_primary_copies,
            ) = _expected_moonep_layout(
                case,
                world_size,
                route_profile=route_profile,
            )
            if (
                balanced_summary["remote_replica_routes_per_rank"]
                != expected_remote_routes
                or not _moonep_copy_layout_matches(
                    balanced_summary["copied_experts_by_rank"],
                    expected_primary_copies,
                    experts_per_rank,
                    route_profile=route_profile,
                )
            ):
                raise AssertionError(
                    "primary Kimi workload did not produce the expected remote "
                    "route and replica assignment"
                )
            primary_epoch_before = balanced_op._replica_weight_epoch
            actual = ascend_full_post_routing(
                balanced_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                actual,
                expected,
                device,
                "MoonEP balanced Triton vs logical Torch golden",
                ep_group,
            )
            torch.npu.synchronize(device)
            primary_expected = _summary_experts_to_copy(
                balanced_summary, experts_per_rank
            )
            _assert_replica_prefetch_state(
                balanced_op,
                primary_expected,
                primary_epoch_before + 1,
                device,
                ep_group,
                "primary balanced forward",
            )
            replica_destinations = [
                destination
                for destination, row in enumerate(primary_expected)
                if bool((row >= 0).any())
            ]
            # The poison proof intentionally relies on the next forward using
            # the already-populated replica table.  With cache bypass enabled,
            # every forward refills that table and therefore repairs the poison
            # before FC1/FC2 consume it; the independent golden above remains
            # the correctness oracle for that one-shot mode.
            if case.tokens == 4096 and MOONEP_ENABLE_REPLICA_CACHE:
                _log_moonep_phase(
                    rank, case, "proving replica FC2 and FC1 weight consumption"
                )
                for weight_name in ("down", "gate_up"):
                    for destination_rank in replica_destinations:
                        _prove_replica_weight_consumption(
                            balanced_op,
                            primary_expected,
                            hidden_states,
                            selected_experts,
                            routing_weights,
                            packed_w1,
                            down_weight,
                            actual,
                            device,
                            ep_group,
                            weight_name=weight_name,
                            destination_rank=destination_rank,
                        )
            del actual

            sample_epoch_before = balanced_op._replica_weight_epoch
            _log_moonep_phase(rank, case, "measuring MoonEP balanced pass B")
            balanced_result = _measure_moonep_forward_e2e(
                device,
                ep_group,
                balanced_op,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                down_weight,
                force_replica_refill=True,
            )
            after_timing_actual = ascend_full_post_routing(
                balanced_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                after_timing_actual,
                expected,
                device,
                "MoonEP output after timing vs logical Torch golden",
                ep_group,
            )
            torch.npu.synchronize(device)
            del after_timing_actual
            _assert_replica_prefetch_state(
                balanced_op,
                primary_expected,
                sample_epoch_before
                + MOONEP_FORWARD_TIMING.warmup
                + MOONEP_FORWARD_TIMING.iterations
                + (0 if MOONEP_ENABLE_REPLICA_CACHE else 1),
                device,
                ep_group,
                "one-shot balanced samples",
            )
        finally:
            _finalize_forward_op(balanced_op, device, ep_group)

        _log_moonep_phase(rank, case, "measuring unbalanced bracket pass A2")
        unbalanced_after_op = _make_forward_op(
            case, ep_group, enable_moonep=False
        )
        try:
            unbalanced_after_actual = ascend_full_post_routing(
                unbalanced_after_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                unbalanced_after_actual,
                expected,
                device,
                "unbalanced bracket A2 vs logical Torch golden",
                ep_group,
            )
            del unbalanced_after_actual
            unbalanced_after = _measure_moonep_forward_e2e(
                device,
                ep_group,
                unbalanced_after_op,
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                down_weight,
            )
            unbalanced_after_a2_timing = ascend_full_post_routing(
                unbalanced_after_op,
                hidden_states,
                selected_experts,
                packed_w1,
                down_weight,
                routing_weights,
            )
            _assert_close_collective(
                unbalanced_after_a2_timing,
                expected,
                device,
                "unbalanced A2 after timing vs logical Torch golden",
                ep_group,
            )
            del unbalanced_after_a2_timing
        finally:
            _finalize_forward_op(unbalanced_after_op, device, ep_group)

    if unbalanced_before is None or balanced_result is None:
        raise RuntimeError("MoonEP benchmark did not produce every timing pass")
    measured["unbalanced_before"] = unbalanced_before.stats
    measured["unbalanced_after"] = unbalanced_after.stats
    measured["unbalanced"] = kit.TimingResult(
        unbalanced_before.samples_ms + unbalanced_after.samples_ms
    ).stats
    measured["balanced"] = balanced_result.stats
    measured["unbalanced_before_samples_ms"] = list(
        unbalanced_before.samples_ms
    )
    measured["unbalanced_after_samples_ms"] = list(
        unbalanced_after.samples_ms
    )
    measured["unbalanced_samples_ms"] = list(
        unbalanced_before.samples_ms + unbalanced_after.samples_ms
    )
    measured["balanced_samples_ms"] = list(balanced_result.samples_ms)
    raw_speedup = statistics.median(
        measured["unbalanced_samples_ms"]
    ) / statistics.median(measured["balanced_samples_ms"])
    del expected
    torch.npu.empty_cache()
    if rank == 0:
        _log_moonep_phase(rank, case, "writing correctness and performance result")
        entry = _make_moonep_entry(
            case,
            world_size,
            route_profile,
            measured,
            owner_routes,
            balanced_summary,
            required_ash_bytes,
            hbm_preflight,
        )
        _print_moonep_entry(entry)
        _upsert_result(
            Path(RESULTS_DIR)
            / f"bench_moonep_forward_suite_w{world_size}.json",
            "forward",
            world_size,
            entry,
            entry["protocol"],
        )
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


def _moonep_backward_transport_samples(
    device,
    ep_group,
    op,
    dy,
    peer_mem,
    hidden_states,
    selected_experts,
    packed_w1,
    down_weight,
    routing_weights,
    *,
    warmup: int,
    iterations: int,
    stage_samples=None,
):
    """Time the physical backward WITH the replica grad transport per sample.

    The transport is single-use and lending permanently invalidates the
    replica weight cache, so every sample pays an untimed production forward
    with native-saved capture (which re-publishes the replica weights and
    produces the fresh ``saved`` the timed backward consumes) and a fresh
    lend.  Ranks synchronize around the timed region; each sample is
    MAX-reduced across ranks, matching the published host-wall semantics.
    ``stage_samples`` (optional shared list) pre-seeds every per-sample saved
    dict's ``_bwd_stage_samples`` so the backward's stage intervals from all
    samples land in one list for the caller's breakdown.
    """
    samples_ms = []
    setup_ms = []
    for i in range(warmup + iterations):
        torch.npu.synchronize(device)
        dist.barrier(group=ep_group)
        setup_start = time.perf_counter()
        _, sample_saved = ascend_full_post_routing(
            op,
            hidden_states,
            selected_experts,
            packed_w1,
            down_weight,
            routing_weights,
            return_saved=True,
        )
        transport = op.lend_replica_weight_tables_for_grad()
        if stage_samples is not None:
            sample_saved["_bwd_stage_samples"] = stage_samples
        torch.npu.synchronize(device)
        dist.barrier(group=ep_group)
        if i >= warmup:
            setup_ms.append((time.perf_counter() - setup_start) * 1000.0)
        start = time.perf_counter()
        with torch.no_grad():
            moe_backward_triton(sample_saved, dy, peer_mem, grad_transport=transport)
        torch.npu.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        value = torch.tensor([elapsed_ms], dtype=torch.float32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=ep_group)
        if i >= warmup:
            samples_ms.append(float(value.item()))
    return samples_ms, setup_ms


def run_moonep_backward_benchmark(
    rank: int,
    world_size: int,
    case: CaseSpec,
    route_profile: str,
):
    """Run the MoonEP native-saved physical backward performance case (N5).

    The backward consumes the NATIVE saved captured by the production forward
    (``return_saved=True``) — the fused dispatch/FC1/FC2 path with RMA replica
    prefetch — with no torch replay anywhere in the measured loop.

    Timed regions, all rank-MAX: (a) the logical Torch backward baseline on a
    synthetic saved (identical to the standard backward suite protocol),
    (b) the physical MoonEP backward WITHOUT transport (nothing consumes the
    replica tables, so the plain host_wall loop applies), and (c) the same
    backward WITH the one-shot replica grad transport, per sample host_wall
    with an untimed forward-with-capture + lend setup, plus the 6-stage
    serial breakdown that isolates the grad_reduce stage.
    """
    from mega_moe import moe_backward_triton

    case = case.validate()
    _validate_moonep_route_profile(route_profile)
    if (
        case.direction != "backward"
        or "performance" not in case.tags
        or case.model.upper() != "KIMI-K3"
        or world_size != 8
        or case.world_size != world_size
    ):
        raise ValueError(
            "MoonEP backward benchmark only accepts performance-bwd-kimi-k3 W8 cases"
        )
    case = replace(
        case,
        capacity_factor=max(
            case.capacity_factor,
            _moonep_receive_capacity_factor(world_size, route_profile),
        ),
    ).validate()
    if torch_npu is None or kit.ash is None:
        raise RuntimeError("MoonEP benchmark requires torch_npu and ACLSHMEM")
    if G_ASH_SIZE_GB < MOONEP_MIN_ASH_SIZE_GB:
        raise RuntimeError(
            "MoonEP Kimi benchmark requires "
            f"MOE_FUSED_ASH_SIZE_GB>={MOONEP_MIN_ASH_SIZE_GB}; got "
            f"{G_ASH_SIZE_GB}"
        )
    peer_rows = case.tokens * case.topk
    peer_mem_bytes = peer_rows * (case.hidden + GATE_PAD) * ACTIVATION_DTYPE.itemsize
    required_ash_bytes = _required_moonep_ash_bytes(case, world_size) + peer_mem_bytes
    if required_ash_bytes >= G_ASH_SIZE:
        raise RuntimeError(
            f"{case.case_id} MoonEP backward estimate is "
            f"{required_ash_bytes / (1024**3):.3f} GiB (incl. peer_mem "
            f"{peer_mem_bytes / (1024**3):.3f} GiB), but "
            f"MOE_FUSED_ASH_SIZE_GB={G_ASH_SIZE_GB}"
        )

    _log_moonep_phase(rank, case, "checking HBM capacity")
    required_free_hbm = _required_moonep_free_hbm_bytes(case, world_size)
    local_free_hbm, local_total_hbm = torch.npu.mem_get_info()
    hbm_info = torch.tensor(
        [local_free_hbm, local_total_hbm], dtype=torch.int64, device=f"npu:{rank}"
    )
    dist.all_reduce(hbm_info, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
    min_free_hbm, min_total_hbm = (int(value) for value in hbm_info.cpu().tolist())
    if min_free_hbm < required_free_hbm:
        raise RuntimeError(
            f"{case.case_id} requires an estimated "
            f"{required_free_hbm / (1024**3):.3f} GiB free HBM per rank, "
            f"but the least-free rank has {min_free_hbm / (1024**3):.3f} GiB"
        )
    hbm_preflight = {
        "minimum_free_hbm_bytes_before_allocation": min_free_hbm,
        "minimum_total_hbm_bytes": min_total_hbm,
        "required_free_hbm_bytes": required_free_hbm,
        "status": "passed",
    }

    ep_group = dist.group.WORLD
    device = f"npu:{rank}"
    experts_per_rank = case.num_experts // world_size

    # Logical Torch baseline first, outside the symmetric session: the
    # synthetic saved never touches ACLSHMEM, and releasing it before the
    # 16+ GiB heap keeps the HBM peak close to the standard suite's.
    _log_moonep_phase(rank, case, "timing logical Torch backward baseline")
    saved_logical, dy_logical, _, _ = build_backward_saved(
        case.tokens, case.hidden, case.ffn, case.num_experts, case.topk, ep_group
    )
    torch_gate_result = backward_torch_baseline(saved_logical, dy_logical)
    for key, value in torch_gate_result.items():
        if not bool(torch.isfinite(value.float()).all()):
            raise AssertionError(f"logical Torch baseline has non-finite {key}")

    def _torch():
        backward_torch_baseline(saved_logical, dy_logical)

    _, torch_timing = kit.PerformanceRunner(
        _torch, _torch, BACKWARD_TIMING, device=device, ep_group=ep_group
    ).run()
    torch_ms = torch_timing.stats["median_ms"]
    del saved_logical, dy_logical, torch_gate_result
    torch.npu.empty_cache()
    dist.barrier(group=ep_group)

    with kit.aclshmem_session(rank, world_size, G_ASH_SIZE):
        # The backward peer_mem must sit at heap offset 0 (dl.symm_at); the
        # forward operator below allocates its own symmetric objects after it.
        peer_mem = kit.make_moonep_backward_peer_mem(
            peer_rows, peer_rows, case.hidden, ACTIVATION_DTYPE, rank, ep_group
        )
        try:
            _log_moonep_phase(
                rank, case, "allocating weights and deterministic hot routes"
            )
            packed_w1, down_weight, _ = _make_local_weights(
                case, experts_per_rank, rank, device
            )
            (
                hidden_states,
                selected_experts,
                routing_weights,
            ) = _prepare_moonep_inputs(
                case, rank, world_size, device, route_profile=route_profile
            )
            torch.manual_seed(44 + rank * 1000)
            dy = torch.randn(
                (case.tokens, case.hidden), dtype=ACTIVATION_DTYPE, device=device
            )

            _log_moonep_phase(
                rank, case,
                "capturing native saved via one production forward",
            )
            op = _make_forward_op(case, ep_group, enable_moonep=True)
            try:
                produced, native_saved = ascend_full_post_routing(
                    op,
                    hidden_states,
                    selected_experts,
                    packed_w1,
                    down_weight,
                    routing_weights,
                    return_saved=True,
                )
                if not bool(torch.isfinite(produced.float()).all()):
                    raise AssertionError(
                        "MoonEP production forward output is non-finite"
                    )
                del produced
                if native_saved.get("use_moonep") is not True:
                    raise AssertionError(
                        "the native saved capture is not flagged use_moonep"
                    )
                if (
                    int(native_saved["total_recv"]) != peer_rows
                    or int(native_saved["total_send"]) != peer_rows
                ):
                    raise AssertionError(
                        "the dropless MoonEP plan must keep total_recv == "
                        f"total_send == tokens*topk ({peer_rows}), got "
                        f"{int(native_saved['total_recv'])}/"
                        f"{int(native_saved['total_send'])}"
                    )

                _log_moonep_phase(rank, case, "running untimed structure gate")
                with torch.no_grad():
                    gate_result = moe_backward_triton(native_saved, dy, peer_mem)
                gate_keys = sorted(gate_result)
                for key, value in gate_result.items():
                    if not _all_finite_memory_lean(value):
                        raise AssertionError(
                            f"MoonEP backward gate has non-finite {key}"
                        )
                del gate_result

                _log_moonep_phase(
                    rank, case, "timing physical backward without transport"
                )

                def _local():
                    with torch.no_grad():
                        moe_backward_triton(native_saved, dy, peer_mem)

                local_timing, _ = kit.PerformanceRunner(
                    _local, _local, BACKWARD_TIMING, device=device, ep_group=ep_group
                ).run()
                local_ms = local_timing.stats["median_ms"]

                _log_moonep_phase(
                    rank, case, "timing physical backward with grad transport"
                )
                transport_samples, setup_samples = (
                    _moonep_backward_transport_samples(
                        device,
                        ep_group,
                        op,
                        dy,
                        peer_mem,
                        hidden_states,
                        selected_experts,
                        packed_w1,
                        down_weight,
                        routing_weights,
                        warmup=BACKWARD_TIMING.warmup,
                        iterations=BACKWARD_TIMING.iterations,
                    )
                )
                transport_ms = statistics.median(transport_samples)
                setup_stats = _stats(setup_samples, device)

                _log_moonep_phase(rank, case, "collecting backward stage breakdown")
                _bd_warmup = max(1, BACKWARD_TIMING.warmup)
                stage_sink = []
                os.environ["MOE_BWD_STAGE_TIMING"] = "1"
                os.environ["MOE_BWD_DUAL_STREAM"] = "0"
                try:
                    _moonep_backward_transport_samples(
                        device,
                        ep_group,
                        op,
                        dy,
                        peer_mem,
                        hidden_states,
                        selected_experts,
                        packed_w1,
                        down_weight,
                        routing_weights,
                        warmup=_bd_warmup,
                        iterations=BACKWARD_TIMING.iterations,
                        stage_samples=stage_sink,
                    )
                finally:
                    os.environ.pop("MOE_BWD_STAGE_TIMING", None)
                    os.environ.pop("MOE_BWD_DUAL_STREAM", None)
                bd_samples = stage_sink[_bd_warmup:]
                stage_widths = {len(sample) for sample in bd_samples}
                if stage_widths != {len(_MOONEP_BWD_STAGE_NAMES)}:
                    raise AssertionError(
                        "the MoonEP transport backward must record "
                        f"{len(_MOONEP_BWD_STAGE_NAMES)} stage intervals, got "
                        f"{sorted(stage_widths)}"
                    )
                backward_breakdown = OrderedDict(
                    (
                        f"{name}_event_ms",
                        _stats([sample[i] for sample in bd_samples], device),
                    )
                    for i, name in enumerate(_MOONEP_BWD_STAGE_NAMES)
                )

                entry = {
                    "schema_version": 1,
                    "direction": "backward",
                    "case_id": case.case_id,
                    "model": case.model,
                    "world_size": world_size,
                    "tokens_per_rank": case.tokens,
                    "route_profile": route_profile,
                    "effective_capacity_factor": case.capacity_factor,
                    "shape": {
                        "hidden": case.hidden,
                        "ffn": case.ffn,
                        "topk": case.topk,
                        "num_experts": case.num_experts,
                    },
                    "protocol": {
                        "torch_baseline": BACKWARD_TIMING.as_dict(),
                        "local_only": BACKWARD_TIMING.as_dict(),
                        "with_transport": {
                            "warmup": BACKWARD_TIMING.warmup,
                            "iterations": BACKWARD_TIMING.iterations,
                            "clock": "host_wall_per_sample",
                            "reduction": "rank_max",
                            "untimed_per_sample_setup": (
                                "production forward with native-saved capture "
                                "(replica refill) + fresh lend"
                            ),
                        },
                        "result_contract": MOONEP_BACKWARD_RESULT_CONTRACT,
                    },
                    "correctness_gate": {
                        "status": "passed_before_timing",
                        "kind": "structure/shape/dtype/finite/no-exception",
                        "keys": gate_keys,
                    },
                    "metrics": {
                        "torch_ms": torch_ms,
                        "local_only_ms": local_ms,
                        "with_transport_ms": transport_ms,
                        "grad_reduce_stage_ms": backward_breakdown[
                            "grad_reduce_event_ms"
                        ]["median_ms"],
                        "with_transport_over_torch": (
                            torch_ms / transport_ms if transport_ms > 0 else float("inf")
                        ),
                        "local_over_torch": (
                            torch_ms / local_ms if local_ms > 0 else float("inf")
                        ),
                    },
                    "diagnostics": {
                        "backward_stage_slices": backward_breakdown,
                        "untimed_setup_ms": setup_stats,
                        "ash_required_bytes": required_ash_bytes,
                        "ash_peer_mem_bytes": peer_mem_bytes,
                        "hbm_preflight": hbm_preflight,
                    },
                }
                if rank == 0:
                    _log_moonep_phase(
                        rank, case, "writing MoonEP backward result"
                    )
                    _upsert_result(
                        Path(
                            os.environ.get(
                                "MOE_BACKWARD_BENCH_RESULTS_DIR",
                                str(PROJECT_ROOT / "results" / "backward"),
                            )
                        )
                        / f"bench_backward_moonep_suite_w{world_size}.json",
                        "backward",
                        world_size,
                        entry,
                        entry["protocol"],
                    )
            finally:
                _finalize_forward_op(op, device, ep_group)
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)
    dist.barrier(group=ep_group)


_FORWARD_CASES = kit.make_pytest_params(
    select_cases(direction="forward", tags={"performance"})
)
_MOONEP_FORWARD_CASES = kit.make_pytest_params(
    MoonEPBenchmarkSpec(case, route_profile)
    for case in select_cases(
        direction="forward", tags={"performance", "kimi"}
    )
    if case.world_size == 8 and case.model.upper() == "KIMI-K3"
    for route_profile in MOONEP_ROUTE_PROFILES
)
_BACKWARD_CASES = kit.make_pytest_params(
    select_cases(direction="backward", tags={"performance"})
)
_MOONEP_BACKWARD_CASES = kit.make_pytest_params(
    MoonEPBenchmarkSpec(case, route_profile)
    for case in select_cases(
        direction="backward", tags={"performance", "kimi"}
    )
    if case.world_size == 8
    for route_profile in MOONEP_ROUTE_PROFILES
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
@pytest.mark.parametrize("spec", _MOONEP_FORWARD_CASES)
def test_bench_moonep_forward_case(dist_test, spec: MoonEPBenchmarkSpec):
    dist_test(
        run_moonep_forward_benchmark,
        world_size=spec.case.world_size,
        args=(spec.case, spec.route_profile),
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


@pytest.mark.dist
@pytest.mark.performance
@pytest.mark.parametrize("spec", _MOONEP_BACKWARD_CASES)
def test_bench_moonep_backward_case(dist_test, spec: MoonEPBenchmarkSpec):
    dist_test(
        run_moonep_backward_benchmark,
        world_size=spec.case.world_size,
        args=(spec.case, spec.route_profile),
    )


# TODO: future work — when an explicit all-directions session is introduced,
# finish and fully clean forward before starting backward, while still keeping
# operator, peer memory, and ACLSHMEM heaps distinct.  Each parameterized case
# currently owns and finalizes its own session.
