# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Benchmark the complete post-routing Ascend Mega-MoE forward.

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

Protocol: 5 warmup iterations and 50 measured samples for every metric.  Each
individual sample is reduced with MAX across ranks before summary statistics.
Before timing, the candidate and grouped baseline must pass normal,
zero-receive/empty-expert, and negative/out-of-range all-drop comparison gates.

Workload profiles:
    KIMI-K3: H=3584, F=3072, top-k=16, E=896,
             tokens/rank in {4096, 8192, 16384}; primary optimization target
    QWEN: H=2048, F=768, top-k=8, E=128,
          tokens/rank in {2048, 8192, 16384, 32768}
    DSV4: H=7168, F=3072, top-k=6, E=384,
          tokens/rank in {2048, 8192, 32768, 131072}; BF16 routed experts only

Usage (the pytest fixture starts workers; do not wrap in torchrun):
    source ./run.sh
    python -m pytest -p tests.conftest benchmark/layer/bench_full_forward.py -m dist -v -s

Environment:
    MOE_FULL_BENCH_CONFIG=kimi_k3_4k,kimi_k3_8k,kimi_k3_16k
                                           # comma-separated selection is accepted
    MOE_DSV4_BENCH_CONFIG=dsv4_pro_2k,dsv4_pro_4k,dsv4_pro_8k,dsv4_pro_32k,dsv4_pro_128k
                                           # comma-separated subset; unset runs all five
    MOE_FULL_BENCH_BREAKDOWN=0            # disable four-stage event diagnostics
    MOE_FULL_BENCH_ROUTE_MODE=dense_random|sparse_balanced
    MOE_FULL_BENCH_ACTIVE_EXPERTS=16       # optional sparse mode override;
                                           # must be >= top-k and divisible by W
    MOE_FUSED_NUM_AICORE_PROGRAMS=24      # 910B1 default; override for other devices
    MOE_FUSED_DISPATCH_PRODUCER_CORES=4   # static/count schedules only
    MOE_FUSED_DISPATCH_READINESS=tile|expert
    MOE_FUSED_DISPATCH_FC1_SCHEDULE=static|count|allcore|allcore_expert|allcore_expert_mn|allcore_expert_n|allcore_expert_n_tile
    MOE_FUSED_FC2_GEMM_SCHEDULE=tile_n_major|expert_n_persistent
    MOE_FUSED_FC2_COMBINE_TRANSPORT=reverse_push|direct_pull
    MOE_FUSED_FC2_REVERSE_VECTOR_WORKERS=1|2  # reverse_push only
    MOE_FUSED_FC2_REDUCE_VECTOR_WORKERS=1|2
    MOE_FULL_BENCH_RESULTS_DIR=/tmp/...   # optional experimental output directory
    MOE_FULL_BENCH_CAPACITY=1.25          # optional active-profile override;
                                           # use 4.0 or unset for default DSV4
    MOE_FUSED_ASH_SIZE_GB=64             # DSV4 128K-per-rank with capacity=4
                                           # needs about 53 GiB/rank before headroom
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from collections import OrderedDict

import pytest
import shmem as ash
import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:  # pragma: no cover - distributed Ascend jobs require torch-npu
    torch_npu = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from mega_moe import FusedMoEForward, MoEForwardConfig


ACTIVATION_DTYPE = torch.bfloat16
ROUTING_INPUT_DTYPE = torch.float32
ROUTING_TRANSPORT_DTYPE = torch.float32
RESULT_CONTRACT = "bf16-activations-fp32-routing-transport-v2"
_WEIGHT_INIT_CHUNK_BYTES = 128 * 1024 * 1024

MODEL_PROFILES = {
    "QWEN": {
        "hidden": 2048,
        "ffn_dim": 768,
        "topk": 8,
        "num_experts": 128,
        "capacity": 1.25,
        "tiling_overrides": {},
        "bench_configs": [
            ("2K", 2048),
            ("8K", 8192),
            ("16K", 16384),
            ("32K", 32768),
        ],
    },
    # Uses the CASE_SET=dsv4 model dimensions from the sibling NVIDIA benchmark
    # launcher, with token counts interpreted per rank for the Ascend workload.
    # This is the routed-expert BF16 shape only: it excludes the shared-expert
    # branch and does not model the model's FP4 expert format.
    "DSV4": {
        "hidden": 7168,
        "ffn_dim": 3072,
        "topk": 6,
        "num_experts": 384,
        "capacity": 4.0,
        "tiling_overrides": {},
        "bench_configs": [
            ("dsv4_pro_2k", 2048),
            ("dsv4_pro_4k", 4096),
            ("dsv4_pro_8k", 8192),
            ("dsv4_pro_32k", 32768),
            ("dsv4_pro_128k", 131072),
        ],
    },
    "KIMI-K3": {
        "hidden": 3584,
        "ffn_dim": 3072,
        "topk": 16,
        "num_experts": 896,
        "capacity": 1.25,
        "tiling_overrides": {},
        "bench_configs": [
            ("kimi_k3_4k", 4096),
            ("kimi_k3_8k", 8192),
            ("kimi_k3_16k", 16384),
        ],
    },
}

# This is the published protocol.  Keep debug/short runs under a differently
# named script so their output cannot be mistaken for 5/50 evidence.
WARMUP_ITERS = 5
BENCH_ITERS = 50
RUN_BREAKDOWN = os.environ.get("MOE_FULL_BENCH_BREAKDOWN", "1") == "1"
ROUTE_MODE = os.environ.get(
    "MOE_FULL_BENCH_ROUTE_MODE", "dense_random"
).lower()
if ROUTE_MODE not in {"dense_random", "sparse_balanced"}:
    raise ValueError(
        "MOE_FULL_BENCH_ROUTE_MODE must be 'dense_random' or "
        "'sparse_balanced'"
    )
G_ASH_SIZE_GB = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", "4"))
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


def _activate_model_profile(model_name):
    """Select one shape profile inside each freshly spawned benchmark worker."""
    normalized = model_name.upper()
    if normalized not in MODEL_PROFILES:
        raise ValueError(
            f"unknown model profile {model_name!r}; expected one of {sorted(MODEL_PROFILES)}"
        )
    profile = MODEL_PROFILES[normalized]
    global MODEL_NAME, HIDDEN, FFN_DIM, TOPK, NUM_EXPERTS
    global BENCH_CONFIGS, CAPACITY, PROFILE_TILING_OVERRIDES
    MODEL_NAME = normalized
    HIDDEN = profile["hidden"]
    FFN_DIM = profile["ffn_dim"]
    TOPK = profile["topk"]
    NUM_EXPERTS = profile["num_experts"]
    BENCH_CONFIGS = profile["bench_configs"]
    PROFILE_TILING_OVERRIDES = profile["tiling_overrides"]
    CAPACITY = float(
        os.environ.get("MOE_FULL_BENCH_CAPACITY", str(profile["capacity"]))
    )
    if CAPACITY < 1.0:
        raise ValueError("MOE_FULL_BENCH_CAPACITY must be at least 1.0")


_activate_model_profile(os.environ.get("MOE_FULL_BENCH_MODEL", "QWEN"))


def _selected_bench_configs():
    config_env = (
        "MOE_DSV4_BENCH_CONFIG"
        if MODEL_NAME == "DSV4"
        else "MOE_FULL_BENCH_CONFIG"
    )
    requested = os.environ.get(config_env)
    if not requested:
        selected = BENCH_CONFIGS
    else:
        labels = {
            item.strip().upper()
            for item in requested.split(",")
            if item.strip()
        }
        known = {label.upper(): label for label, _ in BENCH_CONFIGS}
        unknown = labels - known.keys()
        if unknown:
            raise ValueError(
                f"unknown {config_env} values {sorted(unknown)}; expected a "
                f"comma-separated subset of {sorted(known.values())}"
            )
        selected = [
            config for config in BENCH_CONFIGS
            if config[0].upper() in labels
        ]

    return selected


def _get_ash_ip_port():
    addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("ASH_MASTER_PORT", "8666")
    return f"tcp://{addr}:{port}"


def _required_ash_bytes(tokens_per_rank, world_size):
    """Conservative payload estimate; ACLSHMEM allocator metadata needs headroom."""
    experts_per_rank = NUM_EXPERTS // world_size
    max_recv_rows = int(tokens_per_rank * TOPK * CAPACITY)
    token_peer_bytes = max_recv_rows * HIDDEN * ACTIVATION_DTYPE.itemsize
    routing_peer_bytes = max_recv_rows * ROUTING_TRANSPORT_DTYPE.itemsize
    dispatch_tile_m = int(
        os.environ.get("MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M", "128")
    )
    max_source_tiles = (
        tokens_per_rank * TOPK + dispatch_tile_m - 1
    ) // dispatch_tile_m
    signal_slots = (
        world_size * experts_per_rank * max_source_tiles
        + experts_per_rank
    )
    signal_bytes = signal_slots * 16 * torch.int32.itemsize
    metadata_bins = 1 << (NUM_EXPERTS - 1).bit_length()
    metadata_bytes = world_size * metadata_bins * torch.int32.itemsize
    return token_peer_bytes + routing_peer_bytes + signal_bytes + metadata_bytes


def _layer_tiling_overrides():
    overrides = dict(PROFILE_TILING_OVERRIDES)
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


def _layer_schedule_overrides():
    """Read explicit A/B schedule controls; production defaults stay in config."""
    overrides = {}
    for env_name, parameter_name in (
        ("MOE_FUSED_DISPATCH_READINESS", "dispatch_readiness"),
        ("MOE_FUSED_DISPATCH_FC1_SCHEDULE", "dispatch_fc1_schedule"),
        ("MOE_FUSED_FC2_GEMM_SCHEDULE", "fc2_gemm_schedule"),
        ("MOE_FUSED_FC2_COMBINE_TRANSPORT", "fc2_combine_transport"),
    ):
        if env_name in os.environ:
            overrides[parameter_name] = os.environ[env_name]
    for env_name, parameter_name in (
        ("MOE_FUSED_DISPATCH_PRODUCER_CORES", "dispatch_producer_cores"),
        (
            "MOE_FUSED_FC2_REVERSE_VECTOR_WORKERS",
            "fc2_reverse_vector_workers",
        ),
        (
            "MOE_FUSED_FC2_REDUCE_VECTOR_WORKERS",
            "fc2_reduce_vector_workers",
        ),
    ):
        if env_name in os.environ:
            overrides[parameter_name] = int(os.environ[env_name])
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
def _make_local_weights(experts_per_rank, rank, device, seed=42):
    """Create BF16 weights shared by the candidate and grouped baseline."""
    torch.manual_seed(seed + rank)
    fc1_scale = (1.0 / HIDDEN) ** 0.5
    fc2_scale = (1.0 / FFN_DIM) ** 0.5
    # Generate the production [E, K, N] model-load layout directly.  A hot-path
    # transpose or materialization would create an unnecessary multi-GiB peak.
    packed_w1 = _allocate_chunked_normal_weight(
        (experts_per_rank, HIDDEN, 2 * FFN_DIM),
        fc1_scale,
        device,
    )
    down_weight = _allocate_chunked_normal_weight(
        (experts_per_rank, HIDDEN, FFN_DIM),
        fc2_scale,
        device,
    )

    # Layout preparation is outside both timed paths.  W2 remains physically
    # [E,N,K] because the W4 DSV4 A/B measured it 12.6% faster than [E,K,N].
    torch_w2_kn = down_weight.transpose(1, 2)
    return packed_w1, down_weight, torch_w2_kn


@torch.no_grad()
def _prepare_inputs(tokens_per_rank, rank, device, seed=43):
    """Create post-router inputs; top-k itself is intentionally outside timing."""
    torch.manual_seed(seed + rank * 1000)
    hidden_states = torch.randn(
        (tokens_per_rank, HIDDEN), dtype=ACTIVATION_DTYPE, device=device
    ).mul_(0.5).contiguous()
    if ROUTE_MODE == "dense_random":
        router_logits = torch.randn(
            (tokens_per_rank, NUM_EXPERTS),
            dtype=ROUTING_INPUT_DTYPE,
            device=device,
        )
        topk_logits, selected_experts = torch.topk(
            router_logits, k=TOPK, dim=-1
        )
    else:
        world_size = dist.get_world_size()
        experts_per_rank = NUM_EXPERTS // world_size
        default_active_experts = (
            (TOPK + world_size - 1) // world_size
        ) * world_size
        active_experts = int(
            os.environ.get(
                "MOE_FULL_BENCH_ACTIVE_EXPERTS",
                str(default_active_experts),
            )
        )
        if (
            active_experts < TOPK
            or active_experts > NUM_EXPERTS
            or active_experts % world_size
        ):
            raise ValueError(
                "MOE_FULL_BENCH_ACTIVE_EXPERTS must be in [top-k, E] "
                "and divisible by the EP world size"
            )
        active_per_rank = active_experts // world_size
        if active_per_rank > experts_per_rank:
            raise ValueError(
                "sparse active experts per rank exceed local expert count"
            )
        destination_ranks = torch.arange(
            world_size, dtype=torch.int64, device=device
        )
        local_experts = torch.arange(
            active_per_rank, dtype=torch.int64, device=device
        )
        active_global_experts = (
            destination_ranks[:, None] * experts_per_rank
            + local_experts[None, :]
        ).reshape(-1)
        active_logits = torch.randn(
            (tokens_per_rank, active_experts),
            dtype=ROUTING_INPUT_DTYPE,
            device=device,
        )
        topk_logits, active_indices = torch.topk(
            active_logits, k=TOPK, dim=-1
        )
        selected_experts = active_global_experts[active_indices]
    routing_weights = F.softmax(topk_logits, dim=-1).to(ROUTING_INPUT_DTYPE).contiguous()
    return hidden_states, selected_experts.to(torch.int32).contiguous(), routing_weights


@torch.no_grad()
def _summarize_route_distribution(selected_experts, world_size):
    """Collect untimed global route-distribution provenance on every rank."""
    experts_per_rank = NUM_EXPERTS // world_size
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
        minlength=NUM_EXPERTS,
    ).gt(0).to(torch.int32)
    dist.all_reduce(active_mask, op=dist.ReduceOp.MAX)
    return {
        "active_global_experts": int(active_mask.sum().item()),
        "routes_received_per_rank": routes_received_per_rank.cpu().tolist(),
    }


def _grouped_matmul(inputs, weight_kn, group_list):
    if inputs.shape[0] == 0:
        return torch.empty((0, weight_kn.shape[-1]), dtype=ACTIVATION_DTYPE, device=inputs.device)
    return torch_npu.npu_grouped_matmul(
        [inputs],
        [weight_kn],
        group_list=group_list,
        split_item=3,
        group_type=0,
        group_list_type=0,
        output_dtype=ACTIVATION_DTYPE,
    )[0]


@torch.no_grad()
def _torch_preprocess(hidden_states, selected_experts, routing_weights, experts_per_rank, ep_group):
    """Stable post-router filtering/sort plus HCCL count exchange."""
    tokens_per_rank = hidden_states.shape[0]
    world_size = dist.get_world_size(group=ep_group)
    flat_expert = selected_experts.reshape(-1).long()
    flat_routing = routing_weights.reshape(-1)
    valid_mask = (flat_expert >= 0) & (flat_expert < NUM_EXPERTS)
    valid_expert = flat_expert[valid_mask]
    valid_routing = flat_routing[valid_mask]
    route_token = torch.arange(tokens_per_rank, device=hidden_states.device).repeat_interleave(TOPK)
    valid_token = route_token[valid_mask]

    # A stable global-expert sort is destination-rank-major then local-expert-
    # major, exactly the order needed for dispatch and the reverse permutation.
    sort_idx = torch.argsort(valid_expert.to(torch.float32), stable=True)
    expert_send = valid_expert[sort_idx].to(torch.int32).contiguous()
    token_send = hidden_states[valid_token[sort_idx]].contiguous()
    routing_send_fp32 = valid_routing[sort_idx].contiguous()
    destination = (expert_send.long() // experts_per_rank).to(torch.int32)
    send_counts = torch.bincount(destination, minlength=world_size).to(torch.int32)
    recv_counts = torch.empty((world_size,), dtype=torch.int32, device=hidden_states.device)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)

    return {
        "tokens_per_rank": tokens_per_rank,
        "valid_mask": valid_mask,
        "sort_idx": sort_idx,
        "token_send": token_send,
        "routing_send_fp32": routing_send_fp32,
        "expert_send": expert_send,
        "send_splits": send_counts.cpu().tolist(),
        "recv_splits": recv_counts.cpu().tolist(),
        "num_send": int(send_counts.sum().item()),
        "num_recv": int(recv_counts.sum().item()),
        "experts_per_rank": experts_per_rank,
        "ep_group": ep_group,
    }


@torch.no_grad()
def _torch_dispatch_fc1(state, torch_w1_kn):
    """HCCL payload dispatch, local expert grouping, and packed FC1."""
    device = state["token_send"].device
    recv_rows = state["num_recv"]
    token_recv = torch.empty((recv_rows, HIDDEN), dtype=ACTIVATION_DTYPE, device=device)
    routing_recv = torch.empty((recv_rows,), dtype=ROUTING_TRANSPORT_DTYPE, device=device)
    expert_recv = torch.empty((recv_rows,), dtype=torch.int32, device=device)
    # The input API and the communication payload both preserve FP32 routing
    # weights.  Keep this explicit so a future dtype change cannot silently
    # invalidate the baseline/candidate comparison.
    routing_send = state["routing_send_fp32"].to(ROUTING_TRANSPORT_DTYPE)
    dist.all_to_all_single(
        token_recv,
        state["token_send"],
        output_split_sizes=state["recv_splits"],
        input_split_sizes=state["send_splits"],
        group=state["ep_group"],
    )
    dist.all_to_all_single(
        routing_recv,
        routing_send,
        output_split_sizes=state["recv_splits"],
        input_split_sizes=state["send_splits"],
        group=state["ep_group"],
    )
    dist.all_to_all_single(
        expert_recv,
        state["expert_send"],
        output_split_sizes=state["recv_splits"],
        input_split_sizes=state["send_splits"],
        group=state["ep_group"],
    )

    local_expert = expert_recv.remainder(state["experts_per_rank"])
    local_sort = torch.argsort(local_expert.to(torch.float32), stable=True)
    token_grouped = token_recv[local_sort].contiguous()
    routing_grouped = routing_recv[local_sort].contiguous()
    expert_counts = torch.bincount(local_expert.long(), minlength=state["experts_per_rank"])
    group_list = torch.cumsum(expert_counts, dim=0).to(torch.int64)
    fc1_output = _grouped_matmul(token_grouped, torch_w1_kn, group_list)
    state.update(
        {
            "local_sort": local_sort,
            "routing_grouped": routing_grouped,
            "group_list": group_list,
            "fc1_output": fc1_output,
        }
    )
    return state


@torch.no_grad()
def _torch_weighted_swiglu(state):
    gate, up = state["fc1_output"].chunk(2, dim=-1)
    # Routing transport is FP32, matching the production contract.
    return (
        F.silu(gate.float()) * up.float() * state["routing_grouped"].float().unsqueeze(-1)
    ).to(ACTIVATION_DTYPE)


def _restore_routes_and_reduce(back_sorted, sort_idx, valid_mask, tokens_per_rank):
    inverse_sort = torch.argsort(sort_idx)
    valid_rows = back_sorted[inverse_sort]
    if valid_rows.shape[0] == tokens_per_rank * TOPK:
        combined = valid_rows.view(tokens_per_rank, TOPK, HIDDEN)
    else:
        combined = torch.zeros(
            (tokens_per_rank * TOPK, HIDDEN), dtype=ACTIVATION_DTYPE, device=back_sorted.device
        )
        combined[valid_mask] = valid_rows
        combined = combined.view(tokens_per_rank, TOPK, HIDDEN)

    # Match the production kernel: fixed route-slot order, FP32 accumulation,
    # and one final BF16 cast.
    output_fp32 = torch.zeros(
        (tokens_per_rank, HIDDEN), dtype=torch.float32, device=back_sorted.device
    )
    for route_slot in range(TOPK):
        output_fp32 += combined[:, route_slot].float()
    return output_fp32.to(ACTIVATION_DTYPE)


@torch.no_grad()
def _torch_fc2_combine(state, weighted_activation, torch_w2_kn):
    """FC2, reverse HCCL A2A, and fixed-order FP32 top-k reduction."""
    fc2_grouped = _grouped_matmul(
        weighted_activation, torch_w2_kn, state["group_list"]
    )
    arrival_order = torch.argsort(state["local_sort"])
    fc2_arrival = fc2_grouped[arrival_order].contiguous()
    back_sorted = torch.empty(
        (state["num_send"], HIDDEN), dtype=ACTIVATION_DTYPE, device=weighted_activation.device
    )
    dist.all_to_all_single(
        back_sorted,
        fc2_arrival,
        output_split_sizes=state["send_splits"],
        input_split_sizes=state["recv_splits"],
        group=state["ep_group"],
    )
    return _restore_routes_and_reduce(
        back_sorted, state["sort_idx"], state["valid_mask"], state["tokens_per_rank"]
    )


@torch.no_grad()
def torch_npu_grouped_hccl_full_post_routing(
    hidden_states,
    selected_experts,
    routing_weights,
    torch_w1_kn,
    torch_w2_kn,
    experts_per_rank,
    ep_group,
):
    """Optimized Torch-NPU grouped-GEMM + HCCL full baseline."""
    state = _torch_preprocess(hidden_states, selected_experts, routing_weights, experts_per_rank, ep_group)
    state = _torch_dispatch_fc1(state, torch_w1_kn)
    weighted_activation = _torch_weighted_swiglu(state)
    return _torch_fc2_combine(state, weighted_activation, torch_w2_kn)


def _arrival_to_grouped_from_plan(routing_plan):
    """Map source-major HCCL arrival rows to expert/source grouped rows."""
    counts = routing_plan.receive_counts_by_source_expert.to(torch.int64)
    source_prefix = torch.cumsum(counts, dim=0) - counts
    starts = (
        routing_plan.received_expert_offsets[:-1].to(torch.int64).unsqueeze(0)
        + source_prefix
    )
    flat_counts = counts.reshape(-1)
    flat_starts = starts.reshape(-1)
    segment_offsets = torch.cumsum(flat_counts, dim=0) - flat_counts
    num_rows = routing_plan.num_received_routes
    repeated_starts = torch.repeat_interleave(
        flat_starts, flat_counts, output_size=num_rows
    )
    repeated_offsets = torch.repeat_interleave(
        segment_offsets, flat_counts, output_size=num_rows
    )
    within_segment = (
        torch.arange(num_rows, dtype=torch.int64, device=counts.device)
        - repeated_offsets
    )
    return repeated_starts + within_segment


@torch.no_grad()
def torch_npu_grouped_hccl_fc2_combine_from_dispatch(
    weighted_activation,
    torch_w2_kn,
    dispatch_result,
    tokens_per_rank,
    ep_group,
):
    """Grouped-GEMM + HCCL FC2+combine on the candidate stage input."""
    routing_plan = dispatch_result.routing_plan
    group_list = torch.cumsum(
        routing_plan.received_routes_per_expert.to(torch.int64), dim=0
    )
    fc2_grouped = _grouped_matmul(
        weighted_activation, torch_w2_kn, group_list
    )
    arrival_to_grouped = _arrival_to_grouped_from_plan(routing_plan)
    fc2_arrival = fc2_grouped[arrival_to_grouped].contiguous()

    recv_splits = (
        routing_plan.receive_counts_by_source_expert
        .sum(dim=1, dtype=torch.int32)
        .cpu()
        .tolist()
    )
    send_splits = (
        routing_plan.send_counts_by_rank_expert
        .sum(dim=1, dtype=torch.int32)
        .cpu()
        .tolist()
    )
    back_sorted = torch.empty(
        (routing_plan.num_sent_routes, HIDDEN),
        dtype=ACTIVATION_DTYPE,
        device=weighted_activation.device,
    )
    dist.all_to_all_single(
        back_sorted,
        fc2_arrival,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=ep_group,
    )
    return _restore_routes_and_reduce(
        back_sorted,
        routing_plan.stable_sort_indices,
        routing_plan.valid_route_mask,
        tokens_per_rank,
    )


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


def _time_call_ms(device, ep_group, fn, *args, **kwargs):
    """Time one direct call, then reduce this sample with rank MAX."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    _sync_ranks_before_event(device, ep_group)
    stream = torch.npu.current_stream(device)
    start.record(stream)
    output = fn(*args, **kwargs)
    end.record(stream)
    end.synchronize()
    (elapsed_ms,) = _sample_rank_max([start.elapsed_time(end)], device, ep_group)
    return output, elapsed_ms


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
    hidden_states,
    selected_experts,
    routing_weights,
    torch_w1_kn,
    torch_w2_kn,
    experts_per_rank,
):
    events = [torch.npu.Event(enable_timing=True) for _ in range(5)]
    _sync_ranks_before_event(device, ep_group)
    stream = torch.npu.current_stream(device)
    events[0].record(stream)
    state = _torch_preprocess(hidden_states, selected_experts, routing_weights, experts_per_rank, ep_group)
    events[1].record(stream)
    state = _torch_dispatch_fc1(state, torch_w1_kn)
    events[2].record(stream)
    weighted = _torch_weighted_swiglu(state)
    events[3].record(stream)
    output = _torch_fc2_combine(state, weighted, torch_w2_kn)
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


def _warmup_full(
    op,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
    torch_w2_kn,
    experts_per_rank,
    ep_group,
):
    for _ in range(WARMUP_ITERS):
        ascend_full_post_routing(
            op, hidden_states, selected_experts, packed_w1, down_weight, routing_weights
        )
    torch.npu.synchronize(hidden_states.device)
    dist.barrier(group=ep_group)
    for _ in range(WARMUP_ITERS):
        torch_npu_grouped_hccl_full_post_routing(
            hidden_states,
            selected_experts,
            routing_weights,
            packed_w1,
            torch_w2_kn,
            experts_per_rank,
            ep_group,
        )
    torch.npu.synchronize(hidden_states.device)
    dist.barrier(group=ep_group)


def _validate_case(
    device,
    ep_group,
    op,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
    torch_w2_kn,
    experts_per_rank,
):
    candidate = ascend_full_post_routing(
        op, hidden_states, selected_experts, packed_w1, down_weight, routing_weights
    )
    grouped = torch_npu_grouped_hccl_full_post_routing(
        hidden_states,
        selected_experts,
        routing_weights,
        packed_w1,
        torch_w2_kn,
        experts_per_rank,
        ep_group,
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
    grouped_stage = torch_npu_grouped_hccl_fc2_combine_from_dispatch(
        weighted,
        torch_w2_kn,
        dispatch_result,
        hidden_states.shape[0],
        ep_group,
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
        (num_routes,), NUM_EXPERTS, dtype=torch.int32, device=device
    )
    valid_edge = route_ordinal < max_valid
    zero_receive_flat[valid_edge] = (
        destination[valid_edge] * experts_per_rank
    )
    zero_receive_routes = zero_receive_flat.view_as(selected_experts).contiguous()
    _validate_case(
        device,
        ep_group,
        op,
        hidden_states,
        zero_receive_routes,
        routing_weights,
        packed_w1,
        down_weight,
        torch_w2_kn,
        experts_per_rank,
    )

    all_drop_routes = torch.full_like(selected_experts, NUM_EXPERTS)
    all_drop_routes[0, 0] = -1
    _validate_case(
        device,
        ep_group,
        op,
        hidden_states,
        all_drop_routes,
        routing_weights,
        packed_w1,
        down_weight,
        torch_w2_kn,
        experts_per_rank,
    )


def _measure_case(
    device,
    ep_group,
    op,
    hidden_states,
    selected_experts,
    routing_weights,
    packed_w1,
    down_weight,
    torch_w2_kn,
    experts_per_rank,
):
    _warmup_full(
        op,
        hidden_states,
        selected_experts,
        routing_weights,
        packed_w1,
        down_weight,
        torch_w2_kn,
        experts_per_rank,
        ep_group,
    )
    ascend_full_samples = []
    torch_grouped_full_samples = []
    for _ in range(BENCH_ITERS):
        _, elapsed = _time_call_ms(
            device,
            ep_group,
            ascend_full_post_routing,
            op,
            hidden_states,
            selected_experts,
            packed_w1,
            down_weight,
            routing_weights,
        )
        ascend_full_samples.append(elapsed)
        _, elapsed = _time_call_ms(
            device,
            ep_group,
            torch_npu_grouped_hccl_full_post_routing,
            hidden_states,
            selected_experts,
            routing_weights,
            packed_w1,
            torch_w2_kn,
            experts_per_rank,
            ep_group,
        )
        torch_grouped_full_samples.append(elapsed)

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
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                torch_w2_kn,
                experts_per_rank,
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
                hidden_states,
                selected_experts,
                routing_weights,
                packed_w1,
                torch_w2_kn,
                experts_per_rank,
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
        "ascend_full": _stats(ascend_full_samples, device),
        "torch_grouped_full": _stats(torch_grouped_full_samples, device),
        "breakdown": breakdown,
    }


def _make_entry(
    label,
    tokens_per_rank,
    world_size,
    op,
    measured,
    route_distribution,
):
    ascend_full = measured["ascend_full"]
    torch_grouped_full = measured["torch_grouped_full"]
    device_properties = torch_npu.npu.get_device_properties(op.rank)
    entry = OrderedDict(
        {
            "model_profile": MODEL_NAME,
            "config": label,
            "hidden": HIDDEN,
            "ffn_dim": FFN_DIM,
            "topk": TOPK,
            "num_experts": NUM_EXPERTS,
            "tokens_per_rank": tokens_per_rank,
            "global_tokens": tokens_per_rank * world_size,
            "token_count_scope": "profile specifies tokens per rank",
            "world_size": world_size,
            "hardware": {
                "name": device_properties.name,
                "physical_cube_core_num": device_properties.cube_core_num,
                "physical_vector_core_num": device_properties.vector_core_num,
                "l2_cache_size_bytes": device_properties.L2_cache_size,
                "launch_program_count": op.num_aicore_programs,
            },
            "activation_dtype": "bfloat16",
            "w1_dtype": "bfloat16",
            "w1_model_load_layout": "[E_local,K,N]",
            "w2_dtype": "bfloat16",
            "w2_model_load_layout": "[E_local,N,K]",
            "routing_weight_input_dtype": "float32",
            "routing_weight_transport_dtype": "float32",
            "routing_weight_compute_dtype": "float32",
            "routing_weight_semantics": "FP32 input and transport; FP32 weighted-SwiGLU compute",
            "synthetic_input_generation": {
                "hidden_states": "rank-seeded BF16 normal values scaled by 0.5",
                "route_mode": ROUTE_MODE,
                "selected_experts": (
                    "top-k of rank-seeded FP32 normal logits over all experts; "
                    "generated outside timing"
                    if ROUTE_MODE == "dense_random"
                    else "top-k over a rank-balanced sparse global-expert set; "
                    "generated outside timing"
                ),
                "routing_weights": "FP32 softmax over selected logits",
                "timed_normal_case_has_dropped_routes": False,
                **route_distribution,
            },
            "measured_boundary": "post-router full forward; router/top-k generation excluded",
            "weight_layout_preparation": (
                "candidate and grouped baseline share the same W1/W2 physical storage; "
                "grouped GEMM consumes untimed transpose views"
            ),
            "correctness_gates": {
                "status": "passed_before_timing",
                "cases": [
                    "normal",
                    "zero-receive/empty-expert",
                    "negative/out-of-range all-drop",
                ],
                "providers": [
                    "Ascend production",
                    "Torch-NPU grouped-GEMM + HCCL",
                ],
                "comparison_baseline": "Torch-NPU grouped-GEMM + HCCL",
                "routing_transport_dtype_asserted": True,
            },
            "provenance": _benchmark_provenance(),
            "baseline_semantics": {
                "torch_npu_grouped_hccl": (
                    "optimized BF16 npu_grouped_matmul provider; accumulator dtype "
                    "is not asserted by this benchmark"
                ),
            },
            "warmup_iters": WARMUP_ITERS,
            "benchmark_iters": BENCH_ITERS,
            "sample_rank_reduction": "MAX before statistics",
            "receive_capacity_factor": CAPACITY,
            "dispatch_readiness": op.config.dispatch_readiness,
            "dispatch_fc1_schedule": op.config.dispatch_fc1_schedule,
            "dispatch_producer_cores": (
                op.dispatch_producer_cores
                if op.config.dispatch_fc1_schedule in {"static", "count"}
                else None
            ),
            "fc2_gemm_schedule": op.config.fc2_gemm_schedule,
            "fc2_combine_transport": op.config.fc2_combine_transport,
            "fc2_reverse_vector_workers": (
                op.config.fc2_reverse_vector_workers
                if op.config.fc2_combine_transport == "reverse_push"
                else None
            ),
            "fc2_reduce_vector_workers": op.config.fc2_reduce_vector_workers,
            "tiles": {
                "dispatch_fc1_m": op.config.dispatch_fc1_block_size_m,
                "fc1_gemm_n": op.config.fc1_gemm_block_size_n,
                "fc1_gemm_k": op.config.fc1_gemm_block_size_k,
                "fc2_combine_m": op.config.fc2_combine_block_size_m,
                "fc2_gemm_n": op.config.fc2_gemm_block_size_n,
                "fc2_gemm_k": op.config.fc2_gemm_block_size_k,
                "note": "dot N/K tiles must divide the selected model dimensions",
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
    return entry


def _print_entry(entry):
    metrics = entry["metrics"]
    print(f"  {entry['config']} full-forward median:")
    print(
        f"    full E2E       Ascend={metrics['ascend_full_direct_e2e_ms']['median_ms']:.3f} ms  "
        f"Grouped={metrics['torch_npu_grouped_hccl_full_direct_e2e_ms']['median_ms']:.3f} ms  "
        f"speedup={entry['torch_npu_grouped_hccl_over_ascend_full_median']:.3f}x"
    )
    diagnostics = entry.get("diagnostics")
    if diagnostics is not None:
        print("    independent event medians (Torch-NPU grouped + HCCL / Ascend):")
        for name, speedup in diagnostics["torch_npu_grouped_hccl_over_ascend_median"].items():
            ascend_ms = diagnostics["ascend_event_slices"][f"{name}_event_ms"]["median_ms"]
            grouped_ms = diagnostics["torch_npu_grouped_hccl_event_slices"][f"{name}_event_ms"]["median_ms"]
            print(f"      {name:<16s} {ascend_ms:>8.3f} / {grouped_ms:>8.3f} ms  {speedup:>7.3f}x")
    print(flush=True)


def _save_results(entries, world_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_slug = MODEL_NAME.lower()
    path = os.path.join(
        RESULTS_DIR,
        f"bench_full_forward_{model_slug}_grouped_routefp32_w{world_size}.json",
    )
    # One authoritative file per model profile and EP world size.  Repeated
    # tuning or reruns replace the previous snapshot instead of accumulating
    # timestamped experimental files in the tutorial tree.
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)
    print(f"\n[full-moe-perf] Results saved -> {path}", flush=True)


def _print_summary(entries):
    print("\n" + "=" * 72)
    print(
        f"  {MODEL_NAME} full post-routing MoE: BF16 W1/W2/activations, "
        "FP32 routing-weight input/transport/compute"
    )
    print("=" * 72)
    print(
        f"  {'config':<16s} {'tokens/rank':>12s} {'Ascend full':>11s} "
        f"{'Grouped full':>12s} {'grp x':>8s}"
    )
    for entry in entries:
        metrics = entry["metrics"]
        print(
            f"  {entry['config']:<16s} {entry['tokens_per_rank']:>12d} "
            f"{metrics['ascend_full_direct_e2e_ms']['median_ms']:>11.3f} "
            f"{metrics['torch_npu_grouped_hccl_full_direct_e2e_ms']['median_ms']:>12.3f} "
            f"{entry['torch_npu_grouped_hccl_over_ascend_full_median']:>7.3f}x"
        )
    print("=" * 72)
    print(
        "  Full E2E and event diagnostics are separate 5/50 sample sets; stage medians are non-additive.",
    )

    breakdown_entries = [entry for entry in entries if entry.get("diagnostics") is not None]
    if breakdown_entries:
        stage_labels = OrderedDict(
            (
                ("preprocess", "preprocess"),
                ("dispatch_fc1", "dispatch+FC1"),
                ("weighted_swiglu", "weighted SwiGLU"),
                ("fc2_combine", "FC2+combine"),
            )
        )
        print("\n  Four-stage independent event medians (Torch-NPU grouped + HCCL / Ascend):")
        print("  " + "-" * 79)
        print(
            f"  {'config':<16s} {'stage':<16s} {'Ascend ms':>10s} "
            f"{'Grouped ms':>10s} {'grp x':>9s}"
        )
        for entry in breakdown_entries:
            diagnostics = entry["diagnostics"]
            for stage_name, stage_label in stage_labels.items():
                stage_key = f"{stage_name}_event_ms"
                ascend_ms = diagnostics["ascend_event_slices"][stage_key]["median_ms"]
                grouped_ms = diagnostics["torch_npu_grouped_hccl_event_slices"][stage_key]["median_ms"]
                speedup = diagnostics["torch_npu_grouped_hccl_over_ascend_median"][stage_name]
                print(
                    f"  {entry['config']:<16s} {stage_label:<16s} {ascend_ms:>10.3f} "
                    f"{grouped_ms:>10.3f} {speedup:>8.3f}x"
                )
        print("  " + "-" * 79)
        if len(breakdown_entries) != len(entries):
            print("  Some configs omitted: four-stage diagnostics were disabled for those entries.")
    else:
        print("\n  Four-stage summary omitted: MOE_FULL_BENCH_BREAKDOWN=0 disabled event diagnostics.")
    print(flush=True)


def run_benchmark(rank, world_size, model_name=None):
    _activate_model_profile(
        model_name or os.environ.get("MOE_FULL_BENCH_MODEL", "QWEN")
    )
    if world_size not in (2, 4, 8):
        raise ValueError(f"full MoE benchmark supports world sizes 2, 4, and 8; got {world_size}")
    if torch_npu is None:
        raise RuntimeError("this benchmark requires torch_npu")
    if NUM_EXPERTS % world_size != 0:
        raise ValueError(
            f"{MODEL_NAME} expert count {NUM_EXPERTS} is not divisible by world_size={world_size}"
        )
    configs = _selected_bench_configs()
    required_ash_bytes = max(_required_ash_bytes(tokens, world_size) for _, tokens in configs)
    if required_ash_bytes >= G_ASH_SIZE:
        raise RuntimeError(
            f"selected benchmark estimates {required_ash_bytes / (1024 ** 3):.3f} GiB of symmetric tensor "
            f"payload per rank, but MOE_FUSED_ASH_SIZE_GB={G_ASH_SIZE_GB}; increase the pool with headroom"
        )

    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise RuntimeError("set_conf_store_tls failed")
    attr = ash.InitAttr()
    attr.my_rank = rank
    attr.n_ranks = world_size
    attr.local_mem_size = G_ASH_SIZE
    attr.ip_port = _get_ash_ip_port()
    attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attr)
    if ret != 0:
        raise RuntimeError("aclshmem_init failed")

    device = f"npu:{rank}"
    ep_group = None
    experts_per_rank = NUM_EXPERTS // world_size
    num_aicore_programs = int(
        os.environ.get("MOE_FUSED_NUM_AICORE_PROGRAMS", "24")
    )
    tiling_overrides = _layer_tiling_overrides()
    schedule_overrides = _layer_schedule_overrides()
    entries = []

    if rank == 0:
        labels = ",".join(label for label, _ in configs)
        print(
            f"[full-moe-perf] model={MODEL_NAME} configs={labels} "
            f"H={HIDDEN} F={FFN_DIM} K={TOPK} E={NUM_EXPERTS} "
            f"W={world_size} BF16 warmup={WARMUP_ITERS} samples={BENCH_ITERS} "
            f"sample-reduction=rank-MAX",
            flush=True,
        )

    try:
        for label, tokens_per_rank in configs:
            if rank == 0:
                print(f"\n[full-moe-perf] {label}: tokens/rank={tokens_per_rank}", flush=True)
            config = MoEForwardConfig(
                num_aicore_programs=num_aicore_programs,
                receive_capacity_factor=CAPACITY,
                **tiling_overrides,
                **schedule_overrides,
            )
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens_per_rank,
                hidden_size=HIDDEN,
                top_k=TOPK,
                num_experts=NUM_EXPERTS,
                config=config,
            )
            packed_w1 = None
            down_weight = None
            torch_w2_kn = None
            hidden_states = None
            selected_experts = None
            routing_weights = None
            try:
                packed_w1, down_weight, torch_w2_kn = _make_local_weights(
                    experts_per_rank, rank, device
                )
                hidden_states, selected_experts, routing_weights = _prepare_inputs(
                    tokens_per_rank, rank, device
                )
                route_distribution = _summarize_route_distribution(
                    selected_experts, world_size
                )
                dist.barrier(group=ep_group)
                _validate_case(
                    device,
                    ep_group,
                    op,
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    packed_w1,
                    down_weight,
                    torch_w2_kn,
                    experts_per_rank,
                )
                _validate_edge_cases(
                    device,
                    ep_group,
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
                    op,
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    packed_w1,
                    down_weight,
                    torch_w2_kn,
                    experts_per_rank,
                )
                entry = _make_entry(
                    label,
                    tokens_per_rank,
                    world_size,
                    op,
                    measured,
                    route_distribution,
                )
                entries.append(entry)
                if rank == 0:
                    _print_entry(entry)
            finally:
                torch.npu.synchronize(device)
                dist.barrier(group=ep_group)
                op.finalize()
                # Python loop variables outlive an iteration.  Release the
                # previous case before allocating the next DSV4 weight pair;
                # otherwise the old 23.6-GiB W1/W2 pair overlaps the new one.
                packed_w1 = None
                down_weight = None
                torch_w2_kn = None
                hidden_states = None
                selected_experts = None
                routing_weights = None
                op = None
                gc.collect()
                torch.npu.empty_cache()
                dist.barrier(group=ep_group)
    finally:
        _ = ash.aclshmem_finialize()

    if rank == 0:
        _save_results(entries, world_size)
        _print_summary(entries)


@pytest.mark.dist
def test_bench_full_forward_2ranks(dist_test):
    dist_test(run_benchmark, world_size=2, args=("QWEN", ))


@pytest.mark.dist
def test_bench_full_forward_4ranks(dist_test):
    dist_test(run_benchmark, world_size=4, args=("QWEN", ))


@pytest.mark.dist
def test_bench_full_forward_8ranks(dist_test):
    dist_test(run_benchmark, world_size=8, args=("QWEN", ))


@pytest.mark.dist
def test_bench_full_forward_dsv4_2ranks(dist_test):
    dist_test(run_benchmark, world_size=2, args=("DSV4", ))


@pytest.mark.dist
def test_bench_full_forward_dsv4_4ranks(dist_test):
    dist_test(run_benchmark, world_size=4, args=("DSV4", ))


@pytest.mark.dist
def test_bench_full_forward_dsv4_8ranks(dist_test):
    dist_test(run_benchmark, world_size=8, args=("DSV4", ))


@pytest.mark.dist
def test_bench_full_forward_kimi_k3_4ranks(dist_test):
    dist_test(run_benchmark, world_size=4, args=("KIMI-K3", ))


@pytest.mark.dist
def test_bench_full_forward_kimi_k3_8ranks(dist_test):
    dist_test(run_benchmark, world_size=8, args=("KIMI-K3", ))


if __name__ == "__main__":
    current_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    current_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    run_benchmark(current_rank, current_world_size)
