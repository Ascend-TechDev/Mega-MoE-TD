# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven Ascend Mega-MoE forward/backward performance suite.

The measured model boundary deliberately excludes the router.  ``selected_experts``
and FP32 ``routing_weights`` already exist when timing starts:

    routing metadata + dispatch + packed FC1 + weighted SwiGLU + FC2 + combine

Activations and both expert weights are BF16.  Routing weights remain FP32 at
the public boundary, during transport, and in the weighted-SwiGLU multiply.
The optimized path calls the production layer interfaces directly; this file
does not carry a benchmark-local FC2/combine kernel.

The only performance baseline is Torch-NPU grouped-GEMM + HCCL.  The primary
comparison is direct-call full E2E.  Four event slices (preprocess,
dispatch+FC1, weighted SwiGLU, FC2+combine) are sampled for both the candidate
and grouped baseline.  Event slices are diagnostics: independently
rank-reduced statistics are not additive.

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
    MOE_FULL_BENCH_BREAKDOWN=0          # disable four-stage event diagnostics
    MOE_FUSED_NUM_AICORE_PROGRAMS=24   # device tuning knob
    MOE_FULL_BENCH_RESULTS_DIR=/tmp/... # optional forward result directory
    MOE_BACKWARD_BENCH_RESULTS_DIR=/tmp/... # optional backward result directory
    MOE_FUSED_ASH_SIZE_GB=64            # ACLSHMEM heap size, not shape selection
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
from collections import OrderedDict

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from config import CaseSpec, select_cases
from tests import _moe_testkit as kit


# Device-only dependencies are loaded explicitly by the runner after the
# immutable checkout and environment receipt have been verified.
torch_npu = None
FusedMoEForward = None
MoEForwardConfig = None
GroupedForwardBaseline = None
backward_torch_baseline = None
build_backward_saved = None
compare_backward_gradients = None


ACTIVATION_DTYPE = torch.bfloat16
ROUTING_INPUT_DTYPE = torch.float32
ROUTING_TRANSPORT_DTYPE = torch.float32
RESULT_CONTRACT = "bf16-activations-fp32-routing-transport-v2"
BACKWARD_RESULT_CONTRACT = "backward-wgrad-explicit-paired-v1"
BACKWARD_EVIDENCE_SOURCES = (
    "src/mega_moe/ops/backward.py",
    "src/mega_moe/kernels/transposed_grouped_gemm.py",
    "src/mega_moe/kernels/common.py",
    "tests/_moe_testkit.py",
    "tests/_moe_baselines.py",
    "tests/_numeric.py",
    "config/_shapes.py",
)
BACKWARD_ENVIRONMENT_COMPONENTS = frozenset(
    {"python", "torch", "torch_npu", "triton", "cann", "aclshmem", "bigop"}
)
BACKWARD_TARGET_MODEL = "Qwen3-30B-A3B"
BACKWARD_TARGET_WORLD_SIZE = 2
BACKWARD_TARGET_SPEEDUP = 1.5
BACKWARD_RAW_ORDERS = (
    "candidate_then_baseline",
    "baseline_then_candidate",
)
BACKWARD_RAW_ARMS = ("triton_wgrad", "torch_wgrad")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_WEIGHT_INIT_CHUNK_BYTES = 128 * 1024 * 1024

# This is the published protocol.  Keep debug/short runs under a differently
# named script so their output cannot be mistaken for 5/50 evidence.
FORWARD_TIMING = kit.FORWARD_TIMING
BACKWARD_TIMING = kit.BACKWARD_TIMING
WARMUP_ITERS = FORWARD_TIMING.warmup
BENCH_ITERS = FORWARD_TIMING.iterations
RUN_BREAKDOWN = os.environ.get("MOE_FULL_BENCH_BREAKDOWN", "1") == "1"
G_ASH_SIZE_GB = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", "4"))
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


def _git_output(project_root: Path, *args: str) -> str:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    process = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if process.returncode:
        raise RuntimeError(f"git checkout identity command failed: {' '.join(args)}")
    return process.stdout.strip()


def _checkout_identity(project_root: Path = PROJECT_ROOT) -> dict:
    """Bind one clean, single-parent checkout without trusting ambient Git env."""
    project_root = Path(project_root)
    if not project_root.is_absolute():
        raise RuntimeError("benchmark checkout must be an absolute path")
    resolved_root = project_root.resolve(strict=True)
    top_level = Path(_git_output(resolved_root, "rev-parse", "--show-toplevel"))
    if top_level.resolve(strict=True) != resolved_root:
        raise RuntimeError("benchmark checkout is not the Git top-level")

    commit = _git_output(resolved_root, "rev-parse", "HEAD^{commit}")
    tree = _git_output(resolved_root, "rev-parse", f"{commit}^{{tree}}")
    ancestry = _git_output(resolved_root, "rev-list", "--parents", "-n", "1", commit)
    ancestry_fields = ancestry.split()
    if len(ancestry_fields) != 2 or ancestry_fields[0] != commit:
        raise RuntimeError("benchmark checkout must bind one sole-parent commit")
    status_before = _git_output(
        resolved_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    commit_after = _git_output(resolved_root, "rev-parse", "HEAD^{commit}")
    status_after = _git_output(
        resolved_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status_before or status_after:
        raise RuntimeError("benchmark evidence requires a clean checkout")
    if commit_after != commit:
        raise RuntimeError("benchmark checkout changed during identity verification")
    return {
        "commit": commit,
        "tree": tree,
        "sole_parent": ancestry_fields[1],
        "clean": True,
    }


def _environment_receipt_identity() -> dict:
    """Validate and bind the external environment receipt without its locator."""
    locator = os.environ.get("MOE_BENCH_ENVIRONMENT_RECEIPT")
    if not locator:
        raise RuntimeError("MOE_BENCH_ENVIRONMENT_RECEIPT is required")
    path = Path(locator)
    if not path.is_absolute() or path != path.resolve(strict=True):
        raise RuntimeError("environment receipt must be an absolute non-symlink path")
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("environment receipt must be repo-external")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError("environment receipt must be a single-link regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RuntimeError("environment receipt mode must be 0600")

    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("environment receipt must be valid JSON") from error
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError("environment receipt bytes must be canonical JSON")
    if not isinstance(payload, dict):
        raise ValueError("environment receipt must be a JSON object")
    if payload.get("schema") != "uniep.environment-receipt.v1":
        raise ValueError("environment receipt schema is invalid")
    if payload.get("status") != "VALIDATED":
        raise ValueError("environment receipt status must be VALIDATED")
    components = payload.get("components")
    if not isinstance(components, dict) or set(components) != BACKWARD_ENVIRONMENT_COMPONENTS:
        raise ValueError("environment receipt component denominator is invalid")
    for component_name in sorted(BACKWARD_ENVIRONMENT_COMPONENTS):
        component = components[component_name]
        if not isinstance(component, dict):
            raise ValueError(f"{component_name} component must be an object")
        identity = component.get("identity")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"{component_name} identity is required")
        source_sha256 = component.get("source_sha256")
        if not isinstance(source_sha256, str) or not _SHA256_PATTERN.fullmatch(
            source_sha256
        ):
            raise ValueError(f"{component_name} source_sha256 is invalid")
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "components": sorted(components),
    }


def _load_benchmark_device_runtime() -> None:
    """Import device/runtime code only after provenance verification succeeds."""
    global torch_npu
    global FusedMoEForward, MoEForwardConfig, GroupedForwardBaseline
    global backward_torch_baseline, build_backward_saved, compare_backward_gradients

    kit.load_device_runtime()
    torch_npu = kit.torch_npu
    from mega_moe import FusedMoEForward as _FusedMoEForward
    from mega_moe import MoEForwardConfig as _MoEForwardConfig
    from benchmark.layer._grouped_forward_baseline import (
        GroupedForwardBaseline as _GroupedForwardBaseline,
    )
    from tests._moe_baselines import backward_torch_baseline as _backward_torch_baseline
    from tests._moe_baselines import build_backward_saved as _build_backward_saved
    from tests._moe_baselines import (
        compare_backward_gradients as _compare_backward_gradients,
    )

    FusedMoEForward = _FusedMoEForward
    MoEForwardConfig = _MoEForwardConfig
    GroupedForwardBaseline = _GroupedForwardBaseline
    backward_torch_baseline = _backward_torch_baseline
    build_backward_saved = _build_backward_saved
    compare_backward_gradients = _compare_backward_gradients


def _backward_benchmark_provenance():
    provenance = _benchmark_provenance(BACKWARD_RESULT_CONTRACT)
    source_hashes = {}
    for relative_path in BACKWARD_EVIDENCE_SOURCES:
        with (PROJECT_ROOT / relative_path).open("rb") as source_file:
            source_hashes[relative_path] = hashlib.sha256(source_file.read()).hexdigest()
    provenance["backward_source_sha256"] = source_hashes
    provenance["checkout_identity"] = _checkout_identity()
    provenance["environment_receipt"] = _environment_receipt_identity()
    return provenance


def _get_ash_ip_port():
    return kit.get_ash_ip_port()


def _required_ash_bytes(case: CaseSpec, world_size):
    """Conservative payload estimate; ACLSHMEM allocator metadata needs headroom."""
    experts_per_rank = case.num_experts // world_size
    max_recv_rows = int(case.tokens * case.topk * case.capacity_factor)
    token_peer_bytes = max_recv_rows * case.hidden * ACTIVATION_DTYPE.itemsize
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
    return token_peer_bytes + routing_peer_bytes + signal_bytes + metadata_bytes


def _layer_tiling_overrides():
    overrides = {}
    for env_name, parameter_name in (
        (
            "MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M",
            "dispatch_fc1_block_size_m",
        ),
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
    events = [torch.npu.Event(enable_timing=True) for _ in range(5)]
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
    weighted = op.weighted_swiglu(dispatch_result)
    events[3].record(stream)
    output = op.fc2_combine(weighted, down_weight, dispatch_result)
    events[4].record(stream)
    events[-1].synchronize()
    local = [events[index].elapsed_time(events[index + 1]) for index in range(4)]
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
    events = [torch.npu.Event(enable_timing=True) for _ in range(5)]
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
    events[3].record(stream)
    output = grouped_baseline.fc2_combine(state, weighted, torch_w2_kn)
    events[4].record(stream)
    events[-1].synchronize()
    local = [events[index].elapsed_time(events[index + 1]) for index in range(4)]
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


def _assert_routing_transport_dtype_collective(routing_weight_recv, device, ep_group):
    ok = routing_weight_recv.dtype == ROUTING_TRANSPORT_DTYPE
    flag = torch.tensor([int(ok)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if not bool(flag.item()):
        raise AssertionError(
            "routing transport dtype gate failed: "
            f"got {routing_weight_recv.dtype}, required {ROUTING_TRANSPORT_DTYPE}"
        )


def _stats(samples, device):
    values = torch.tensor(samples, dtype=torch.float32, device=device)
    return {
        "min_ms": round(float(values.min().item()), 3),
        "max_ms": round(float(values.max().item()), 3),
        "mean_ms": round(float(values.mean().item()), 3),
        "median_ms": round(float(values.median().item()), 3),
    }


def _stage_stats(samples, device):
    names = ("preprocess", "dispatch_fc1", "weighted_swiglu", "fc2_combine")
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

    weighted, dispatch_result = op.dispatch_fc1_weighted_swiglu(
        hidden_states, selected_experts, routing_weights, packed_w1
    )
    _assert_routing_transport_dtype_collective(
        dispatch_result.received_routing_weights, device, ep_group
    )
    candidate_stage = op.fc2_combine(
        weighted, down_weight, dispatch_result
    )
    weighted, dispatch_result = op.dispatch_fc1_weighted_swiglu(
        hidden_states, selected_experts, routing_weights, packed_w1
    )
    grouped_stage = grouped_baseline.fc2_combine_from_dispatch(
        weighted,
        torch_w2_kn,
        dispatch_result,
        hidden_states.shape[0],
    )
    _assert_close_collective(
        candidate_stage,
        grouped_stage,
        device,
        "FC2+combine stage vs grouped baseline",
        ep_group,
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
            for name in ("preprocess", "dispatch_fc1", "weighted_swiglu", "fc2_combine")
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


def _backward_target_cases(world_size: int) -> tuple[CaseSpec, ...]:
    if world_size != BACKWARD_TARGET_WORLD_SIZE:
        return ()
    cases = tuple(
        case
        for case in select_cases(direction="backward", tags={"performance"})
        if case.model == BACKWARD_TARGET_MODEL and case.world_size == world_size
    )
    if [case.tokens for case in cases] != [4096, 8192, 16384]:
        raise RuntimeError("Qwen3 EP2 target denominator drifted from 4K/8K/16K")
    return cases


def _finite_positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return numeric


def _validate_backward_target_entry(entry: dict, expected: CaseSpec) -> None:
    if not isinstance(entry, dict):
        raise ValueError("backward target case must be an object")
    expected_identity = {
        "case_id": expected.case_id,
        "model": BACKWARD_TARGET_MODEL,
        "world_size": BACKWARD_TARGET_WORLD_SIZE,
        "tokens_per_rank": expected.tokens,
    }
    for field, value in expected_identity.items():
        if entry.get(field) != value:
            raise ValueError(f"{expected.case_id}: {field} identity drift")
    if entry.get("shape") != {
        "hidden": expected.hidden,
        "ffn": expected.ffn,
        "topk": expected.topk,
        "num_experts": expected.num_experts,
    }:
        raise ValueError(f"{expected.case_id}: shape identity drift")

    protocol = entry.get("protocol")
    expected_protocol_fields = {
        "warmup": BACKWARD_TIMING.warmup,
        "iterations": BACKWARD_TIMING.iterations,
        "clock": BACKWARD_TIMING.clock,
        "rank_reduction": "MAX",
        "paired_orders": [
            "triton_wgrad_then_torch_wgrad",
            "torch_wgrad_then_triton_wgrad",
        ],
        "samples_per_arm_per_order": BACKWARD_TIMING.iterations,
    }
    if not isinstance(protocol, dict):
        raise ValueError(f"{expected.case_id}: protocol must be an object")
    for field, value in expected_protocol_fields.items():
        if protocol.get(field) != value:
            raise ValueError(f"{expected.case_id}: protocol {field} drift")

    correctness = entry.get("correctness_gate")
    if not isinstance(correctness, dict):
        raise ValueError(f"{expected.case_id}: correctness gate is missing")
    if correctness.get("status") != "passed_before_timing" or set(
        correctness.get("arms", ())
    ) != set(BACKWARD_RAW_ARMS):
        raise ValueError(f"{expected.case_id}: correctness gate is incomplete")

    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{expected.case_id}: metrics must be an object")
    if "target_met" in metrics:
        raise ValueError(f"{expected.case_id}: target_met is finalizer-owned")
    raw_samples = metrics.get("raw_samples_ms")
    if not isinstance(raw_samples, dict) or set(raw_samples) != set(
        BACKWARD_RAW_ORDERS
    ):
        raise ValueError(f"{expected.case_id}: paired order denominator is invalid")
    for order_name in BACKWARD_RAW_ORDERS:
        arms = raw_samples[order_name]
        if not isinstance(arms, dict) or set(arms) != set(BACKWARD_RAW_ARMS):
            raise ValueError(f"{expected.case_id}: {order_name} arm denominator is invalid")
        for arm_name in BACKWARD_RAW_ARMS:
            samples = arms[arm_name]
            if not isinstance(samples, list) or len(samples) != BACKWARD_TIMING.iterations:
                raise ValueError(
                    f"{expected.case_id}: {order_name}/{arm_name} must have "
                    f"exactly {BACKWARD_TIMING.iterations} samples"
                )
            for index, sample in enumerate(samples):
                _finite_positive_number(
                    sample, f"{expected.case_id}:{order_name}/{arm_name}[{index}]"
                )

    speedups = metrics.get("speedup_by_order")
    if not isinstance(speedups, dict) or set(speedups) != set(BACKWARD_RAW_ORDERS):
        raise ValueError(f"{expected.case_id}: speedup order denominator is invalid")
    normalized_speedups = {
        order_name: _finite_positive_number(
            speedups[order_name], f"{expected.case_id}:{order_name} speedup"
        )
        for order_name in BACKWARD_RAW_ORDERS
    }
    minimum_speedup = _finite_positive_number(
        metrics.get("minimum_speedup"), f"{expected.case_id}:minimum_speedup"
    )
    if not math.isclose(
        minimum_speedup,
        min(normalized_speedups.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{expected.case_id}: minimum speedup is not the paired minimum")
    if metrics.get("target_speedup") != BACKWARD_TARGET_SPEEDUP:
        raise ValueError(f"{expected.case_id}: target speedup drift")

    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{expected.case_id}: provenance is missing")
    for field in (
        "checkout_identity",
        "environment_receipt",
        "backward_source_sha256",
    ):
        if not isinstance(provenance.get(field), dict):
            raise ValueError(f"{expected.case_id}: provenance {field} is missing")


def _finalize_backward_target_payload(payload: dict) -> None:
    expected_cases = _backward_target_cases(payload["world_size"])
    if not expected_cases:
        return
    expected_by_id = {case.case_id: case for case in expected_cases}
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("backward target cases must be a list")
    observed_ids = [item.get("case_id") for item in cases if isinstance(item, dict)]
    if len(observed_ids) != len(cases) or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("backward target case ids must be unique strings")
    extras = sorted(set(observed_ids) - set(expected_by_id))
    if extras:
        raise ValueError(f"unexpected backward target cases: {extras}")
    for item in cases:
        _validate_backward_target_entry(item, expected_by_id[item["case_id"]])

    missing = [case.case_id for case in expected_cases if case.case_id not in observed_ids]
    payload["denominator_complete"] = not missing
    payload["missing_cases"] = missing
    payload.pop("target_met", None)
    for item in cases:
        item["metrics"].pop("target_met", None)
    if missing:
        return

    checkout_identities = {
        json.dumps(item["provenance"]["checkout_identity"], sort_keys=True)
        for item in cases
    }
    environment_identities = {
        json.dumps(item["provenance"]["environment_receipt"], sort_keys=True)
        for item in cases
    }
    source_identities = {
        json.dumps(item["provenance"]["backward_source_sha256"], sort_keys=True)
        for item in cases
    }
    if any(
        len(identity_set) != 1
        for identity_set in (
            checkout_identities,
            environment_identities,
            source_identities,
        )
    ):
        raise ValueError("backward target cases do not share one evidence identity")
    for item in cases:
        item["metrics"]["target_met"] = (
            item["metrics"]["minimum_speedup"] >= BACKWARD_TARGET_SPEEDUP
        )
    payload["target_met"] = all(item["metrics"]["target_met"] for item in cases)


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
        if (
            isinstance(existing, dict)
            and existing.get("schema_version") == 1
            and existing.get("direction") == direction
            and existing.get("world_size") == world_size
            and existing.get("protocol") == protocol
        ):
            payload = existing
    cases = [item for item in payload.get("cases", []) if item.get("case_id") != entry["case_id"]]
    cases.append(entry)
    payload["cases"] = sorted(cases, key=lambda item: item["case_id"])
    if direction == "backward":
        _finalize_backward_target_payload(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    os.replace(temporary, path)
    print(f"[saved] {path}", flush=True)


def run_forward_benchmark(rank: int, world_size: int, case: CaseSpec):
    """Run exactly the explicitly parameterized forward performance case."""
    _load_benchmark_device_runtime()
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
            num_aicore_programs=int(
                os.environ.get("MOE_FUSED_NUM_AICORE_PROGRAMS", "24")
            ),
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
        candidates = {
            "torch_wgrad": moe_backward_triton(
                saved, dy, peer_mem, use_triton_wgrad=False
            ),
            "triton_wgrad": moe_backward_triton(
                saved, dy, peer_mem, use_triton_wgrad=True
            ),
        }

    details_by_arm = {}
    finite_error = None
    try:
        kit.validate_finite_backward_gradients(candidates, torch_result)
        finite_ok = True
    except AssertionError as error:
        finite_ok = False
        finite_error = str(error)
    finite_flag = torch.tensor(
        [1 if finite_ok else 0], dtype=torch.int32, device=dy.device
    )
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN, group=saved["ep_group"])
    if not bool(finite_flag.item()):
        raise AssertionError(
            "backward non-finite gradient gate failed: "
            f"{finite_error or 'non-finite gradient observed on another rank'}"
        )
    for arm_name, result in candidates.items():
        all_ok, details = compare_backward_gradients(result, torch_result)
        local_ok = torch.tensor(
            [1 if all_ok else 0], dtype=torch.int32, device=dy.device
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN, group=saved["ep_group"])
        details_by_arm[arm_name] = details
        if not bool(local_ok.item()):
            raise AssertionError(
                f"backward five-gradient gate failed for {arm_name}: {details}"
            )
    return details_by_arm


def _validate_backward_benchmark_environment():
    """Reject ambient knobs that could change or obscure the paired arms."""
    forbidden = {
        name: os.environ[name]
        for name in ("MOE_WGRAD_TRITON", "MOE_BWD_TRACE")
        if os.environ.get(name)
    }
    if forbidden:
        names = ", ".join(sorted(forbidden))
        raise RuntimeError(
            f"backward benchmark requires explicit quiet wgrad arms; unset {names}"
        )


def _backward_results_path(world_size: int) -> Path:
    locator = os.environ.get("MOE_BACKWARD_BENCH_RESULTS_DIR")
    if not locator:
        raise RuntimeError(
            "MOE_BACKWARD_BENCH_RESULTS_DIR must name a repo-external directory"
        )
    root = Path(locator)
    if not root.is_absolute():
        raise RuntimeError("backward result directory must be absolute")
    resolved_root = root.resolve(strict=False)
    try:
        resolved_root.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError("backward result directory must be repo-external")
    return resolved_root / f"bench_backward_suite_w{world_size}.json"


def run_backward_benchmark(rank: int, world_size: int, case: CaseSpec):
    """Run one explicit backward case with the published 5/50 host-wall protocol."""
    case = case.validate()
    if case.direction != "backward" or "performance" not in case.tags:
        raise ValueError(f"backward runner received non-performance case {case.case_id}")
    if world_size != case.world_size:
        raise ValueError(f"worker world size does not match {case.case_id}")
    _validate_backward_benchmark_environment()
    provenance = _backward_benchmark_provenance()
    result_path = _backward_results_path(world_size)
    _load_benchmark_device_runtime()
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
            gradient_details = _backward_gate(saved, dy, peer_mem)

            def _triton():
                with torch.no_grad():
                    moe_backward_triton(
                        saved, dy, peer_mem, use_triton_wgrad=True
                    )

            def _torch():
                with torch.no_grad():
                    moe_backward_triton(
                        saved, dy, peer_mem, use_triton_wgrad=False
                    )

            paired = kit.PerformanceRunner(
                _triton,
                _torch,
                BACKWARD_TIMING,
                device=device,
                ep_group=ep_group,
            ).run_paired()
            order_speedups = {}
            raw_samples_ms = {}
            order_stats = {}
            for order_name, results in paired.items():
                triton_result = results["candidate"]
                torch_result = results["baseline"]
                triton_ms = triton_result.median_ms
                torch_ms = torch_result.median_ms
                if triton_ms <= 0 or torch_ms <= 0:
                    raise RuntimeError(
                        f"non-positive paired timing in {order_name}: "
                        f"triton={triton_ms}, torch={torch_ms}"
                    )
                order_speedups[order_name] = torch_ms / triton_ms
                raw_samples_ms[order_name] = {
                    "triton_wgrad": list(triton_result.samples_ms),
                    "torch_wgrad": list(torch_result.samples_ms),
                }
                order_stats[order_name] = {
                    "triton_wgrad": triton_result.stats,
                    "torch_wgrad": torch_result.stats,
                }
            minimum_speedup = min(order_speedups.values())
            entry = {
                "schema_version": 2,
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
                "protocol": {
                    **BACKWARD_TIMING.as_dict(),
                    "paired_orders": [
                        "triton_wgrad_then_torch_wgrad",
                        "torch_wgrad_then_triton_wgrad",
                    ],
                    "samples_per_arm_per_order": BACKWARD_TIMING.iterations,
                    "comparison_boundary": (
                        "same five-stage backward; only step3/step5 wgrad "
                        "implementation differs"
                    ),
                },
                "correctness_gate": {
                    "status": "passed_before_timing",
                    "kind": "independent Torch oracle; five numeric gradients",
                    "arms": ["torch_wgrad", "triton_wgrad"],
                },
                "metrics": {
                    "raw_samples_ms": raw_samples_ms,
                    "order_stats": order_stats,
                    "speedup_by_order": order_speedups,
                    "minimum_speedup": minimum_speedup,
                    "target_speedup": 1.5,
                },
                "gradient_gate": {
                    "keys": [
                        "grad_hidden",
                        "grad_routing_weights",
                        "grad_fc1_1",
                        "grad_fc1_2",
                        "grad_fc2",
                    ],
                    "comparison": "untimed numeric gate for both wgrad arms",
                    "details": gradient_details,
                },
                "provenance": provenance,
            }
            if rank == 0:
                _upsert_result(
                    result_path,
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
