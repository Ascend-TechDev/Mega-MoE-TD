# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven distributed forward/backward functional suites.

Every pytest node represents one concrete ``CaseSpec``.  Shape and token
selection happen at collection time; workers never inspect shape-selection
environment variables.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from triton.backends.ascend.driver import NPUUtils

from config import CaseSpec, select_cases
from mega_moe import (
    FusedMoEForward,
    MoEForwardConfig,
    moe_backward_triton,
    pack_gate_up_weights,
)
from mega_moe.ops._moonep_torch_forward import (
    build_physical_saved_from_plan,
    gather_replica_weights_via_hccl,
)
from mega_moe.ops._torch_forward import moe_forward
from mega_moe.kernels.common import all_gather_list
import mega_moe.kernels.fc2_combine as fc2_combine_module
from mega_moe.kernels.moonep_planning import launch_moonep_b0_b3
from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)
from benchmark.layer import bench_moe_suite as bench_module
from mega_moe.kernels.weighted_swiglu import (
    _BLOCK_M as _WEIGHTED_BLOCK_M,
    _BLOCK_N as _WEIGHTED_BLOCK_N,
    _weighted_activation_expert_group_kernel,
)
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    backward_torch_baseline,
    build_backward_saved,
    compare_backward_gradients,
    make_down_weights,
    make_gate_up_weights,
    make_routing_weights,
    prepare_inputs,
    run_full_one,
    run_one,
    torch_moe_fwd_golden,
)
from tests._numeric import (
    GRAD_ATOL,
    GRAD_RTOL,
    OUTPUT_ATOL,
    OUTPUT_RTOL,
    assert_close,
    diagnose,
)

# Stage H3 (master plan §3.2.3) adds the merged native-saved autograd Function
# and exports it from ``mega_moe.ops``.  Collection must stay working on a tree
# without it, so the import degrades to ``None`` and the case skips.
try:
    from mega_moe.ops import MegaMoEFunction  # noqa: E402
except ImportError:
    try:
        from mega_moe.ops.backward import MegaMoEFunction  # noqa: E402
    except ImportError:
        MegaMoEFunction = None


def _forward_config(case: CaseSpec) -> MoEForwardConfig:
    return MoEForwardConfig(
        receive_capacity_factor=case.capacity_factor,
    )


def test_device_put_workspaces_cover_send_and_expert_source_pairs(monkeypatch):
    """Size combine rows by sends and descriptors by expert/source pairs."""
    if kit.ash is None:
        pytest.skip("ACLSHMEM is unavailable")

    op = FusedMoEForward.__new__(FusedMoEForward)
    torch.nn.Module.__init__(op)
    op.max_tokens_per_rank = 8
    op.hidden_size = 4
    op.top_k = 2
    op.world_size = 8
    op.rank = 0
    op.experts_per_rank = 112
    op.activation_dtype = torch.bfloat16
    op.context = SimpleNamespace(peer_mem=torch.empty(4096, dtype=torch.bfloat16))
    op._combine_fc2_buf = None

    monkeypatch.setattr(
        kit.ash,
        "aclshmem_create_tensor",
        lambda shape, dtype, device_id: torch.empty(shape, dtype=dtype),
    )
    monkeypatch.setattr(kit.ash, "aclshmem_free_tensor", lambda tensor: None)

    op._ensure_combine_buffers()

    assert op._combine_fc2_buf.shape == (16, 4)
    assert op._route_to_send.shape == (16,)
    expected_slots = 8 * 112
    assert op._pull_tile_rank.shape == (expected_slots,)


def test_device_put_descriptor_capacity_uses_uint32_bytes():
    row_width = 3584
    max_rows = fc2_combine_module._ACLSHMEM_PUTMEM_MAX_BYTES // (
        row_width * 2
    )

    fc2_combine_module._validate_putmem_descriptor_capacity(
        max_rows,
        row_width,
    )
    with pytest.raises(ValueError, match="uint32 byte-count ABI"):
        fc2_combine_module._validate_putmem_descriptor_capacity(
            max_rows + 1,
            row_width,
        )


def test_device_put_worker_layout_requires_one_worker_per_rank():
    assert fc2_combine_module._fc2_device_put_worker_layout(8, 8, 16) == (
        1,
        8,
    )
    assert fc2_combine_module._fc2_device_put_worker_layout(64, 8, 16) == (
        8,
        64,
    )
    with pytest.raises(ValueError, match="one Vector program per rank"):
        fc2_combine_module._fc2_device_put_worker_layout(7, 8, 16)


def test_moonep_moderate_wide_capacity_covers_hottest_owner():
    assert bench_module._moonep_receive_capacity_factor(
        8, "moderate-wide"
    ) == 27 / 16


def test_pack_gate_up_weights_returns_contiguous_kn_layout():
    gate = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    up = gate + 100
    packed = pack_gate_up_weights(gate, up)

    assert packed.shape == (2, 4, 6)
    assert packed.is_contiguous()
    torch.testing.assert_close(packed[:, :, :3], gate.transpose(1, 2))
    torch.testing.assert_close(packed[:, :, 3:], up.transpose(1, 2))


def _situglu_torch_ref(fc1, routing_weights, activation, beta, linear_beta):
    """Independent FP32 reference for the target activation extension."""
    ffn_dim = fc1.shape[-1] // 2
    gate = fc1[..., :ffn_dim].float()
    up = fc1[..., ffn_dim:].float()
    if activation == "swiglu":
        activated = torch.nn.functional.silu(gate) * up
    else:
        activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        activated = activated * up
    return (activated * routing_weights.float().unsqueeze(-1)).to(fc1.dtype)


@pytest.mark.skipif(
    not torch.npu.is_available(),
    reason="grouped activation correctness requires an NPU device",
)
def test_weighted_activation_expert_groups_cover_empty_and_tail_ranges():
    device = "npu:0"
    torch.npu.set_device(0)
    torch.manual_seed(1)
    rows, ffn_dim = 29, 256
    experts_per_rank = 5
    group_experts = 2
    # Group 0 is empty, group 1 starts at row 0, and group 2 is a short tail.
    expert_offsets = torch.tensor(
        [0, 0, 0, 7, 13, rows],
        dtype=torch.int32,
        device=device,
    )
    fc1 = (
        (torch.randn(rows, 2 * ffn_dim, dtype=torch.float32) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    routing_weights = (
        torch.rand(rows, dtype=torch.float32, device=device) + 0.1
    ).contiguous()
    num_vector_programs = NPUUtils().get_aivector_core_num()
    cases = (
        ("swiglu", 0, 1.0, None),
        ("situglu", 1, 2.0, 1.0),
    )

    for activation, activation_id, beta, linear_beta in cases:
        actual = torch.empty(
            (rows, ffn_dim), dtype=torch.bfloat16, device=device
        )
        for group_id in range(3):
            _weighted_activation_expert_group_kernel[(num_vector_programs,)](
                fc1,
                routing_weights,
                actual,
                expert_offsets,
                group_id,
                ffn_dim,
                beta,
                float(linear_beta) if linear_beta is not None else 0.0,
                BLOCK_M=_WEIGHTED_BLOCK_M,
                BLOCK_N=_WEIGHTED_BLOCK_N,
                ACTIVATION=activation_id,
                HAS_LINEAR_BETA=linear_beta is not None,
                GROUP_EXPERTS=group_experts,
                EXPERTS_PER_RANK=experts_per_rank,
            )
        expected = _situglu_torch_ref(
            fc1,
            routing_weights,
            activation,
            beta,
            linear_beta,
        )
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )


def run_forward_case(rank: int, world_size: int, case: CaseSpec) -> None:
    if world_size != case.world_size:
        raise ValueError(f"worker world size does not match {case.case_id}")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("functional forward requires torch_npu and ACLSHMEM")

    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    heap_size = kit.get_ash_size_bytes(default_gb=1)
    all_passed = True

    with kit.aclshmem_session(rank, world_size, heap_size):
        op = FusedMoEForward(
            None,
            max_tokens_per_rank=case.tokens,
            hidden_size=case.hidden,
            top_k=case.topk,
            num_experts=case.num_experts,
            config=_forward_config(case),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                case.num_experts,
                case.hidden,
                case.ffn,
                world_size,
                rank,
                dtype,
                device,
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                case.num_experts,
                case.hidden,
                case.ffn,
                world_size,
                rank,
                dtype,
                device,
            )
            hs, expert_indices = prepare_inputs(
                case.tokens,
                case.hidden,
                case.num_experts,
                case.topk,
                dtype,
                device,
                seed=43 + rank * 1000,
                drop_frac=case.drop_frac,
            )
            if case.model == "S-drop":
                expert_indices[0, 0] = -1
            routing_weights = make_routing_weights(
                case.tokens, case.topk, device, seed=44 + rank * 1000
            )
            dist.barrier()

            all_passed &= run_one(
                op, hs, expert_indices, packed_w1, case.num_experts,
                f"{case.case_id}-dispatch", rank, device, dtype,
            )
            all_passed &= run_full_one(
                op, hs, expert_indices, routing_weights,
                w_gate, w_up, packed_w1, w2, case.num_experts,
                f"{case.case_id}-full", rank, device,
            )

            # Keep the asymmetric routing, all-drop and epoch-reuse coverage
            # from the former forward test, but only on the representative S.
            if case.model == "S":
                skew = torch.full_like(expert_indices, world_size)
                edge_weights = torch.zeros_like(routing_weights)
                edge_weights[:, -1] = 1.0
                all_passed &= run_one(
                    op, hs, skew, packed_w1, case.num_experts,
                    f"{case.case_id}-zero-receive", rank, device, dtype,
                )
                all_passed &= run_full_one(
                    op, hs, skew, edge_weights,
                    w_gate, w_up, packed_w1, w2, case.num_experts,
                    f"{case.case_id}-zero-receive-full", rank, device,
                )

                all_drop = torch.full_like(expert_indices, case.num_experts)
                all_passed &= run_one(
                    op, hs, all_drop, packed_w1, case.num_experts,
                    f"{case.case_id}-all-drop", rank, device, dtype,
                )
                all_passed &= run_full_one(
                    op, hs, all_drop, routing_weights,
                    w_gate, w_up, packed_w1, w2, case.num_experts,
                    f"{case.case_id}-all-drop-full", rank, device,
                )
                all_passed &= run_full_one(
                    op, hs, expert_indices, routing_weights,
                    w_gate, w_up, packed_w1, w2, case.num_experts,
                    f"{case.case_id}-full-epoch-reuse", rank, device,
                )
        finally:
            op.finalize()

        flag = torch.tensor(
            [1 if all_passed else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
        if not bool(flag.item()):
            raise AssertionError(
                f"functional forward case failed: {case.case_id}"
            )


def run_moonep_hot_expert_case(
    rank: int,
    world_size: int,
) -> None:
    """Force logical expert 0 onto a remote replica and compare full output."""
    if world_size not in (2, 4):
        raise ValueError("MoonEP hot-expert smoke requires two or four ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("MoonEP forward smoke requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        op = FusedMoEForward(
            dist.group.WORLD,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=1.0,
                enable_moonep=True,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            hs, _ = prepare_inputs(
                tokens,
                hidden,
                num_experts,
                topk,
                dtype,
                device,
                seed=701 + rank,
            )
            expert_indices = torch.zeros(
                (tokens, topk), dtype=torch.int32, device=device
            )
            routing_weights = torch.empty(
                (tokens, topk), dtype=torch.float32, device=device
            )
            routing_weights[:, 0] = 0.25
            routing_weights[:, 1] = 0.75

            invalid_expert_indices = expert_indices.clone()
            if rank == 0:
                invalid_expert_indices[0, 0] = num_experts
            rejected_invalid_routes = False
            try:
                op.build_routing_plan(invalid_expert_indices)
            except ValueError as exc:
                rejected_invalid_routes = "dropless valid routes" in str(exc)
            rejected_flag = torch.tensor(
                [int(rejected_invalid_routes)],
                dtype=torch.int32,
                device=device,
            )
            dist.all_reduce(
                rejected_flag,
                op=dist.ReduceOp.MIN,
                group=dist.group.WORLD,
            )
            if not bool(rejected_flag.item()):
                raise AssertionError(
                    "MoonEP did not collectively reject a rank-local invalid route"
                )

            if rank == 0:
                asymmetric_expert_indices = expert_indices[:-1].contiguous()
            else:
                asymmetric_expert_indices = expert_indices.clone()
                asymmetric_expert_indices[0].fill_(num_experts)
            rejected_asymmetric_routes = False
            try:
                op.build_routing_plan(asymmetric_expert_indices)
            except ValueError as exc:
                rejected_asymmetric_routes = "dropless valid routes" in str(exc)
            rejected_flag.fill_(int(rejected_asymmetric_routes))
            dist.all_reduce(
                rejected_flag,
                op=dist.ReduceOp.MIN,
                group=dist.group.WORLD,
            )
            if not bool(rejected_flag.item()):
                raise AssertionError(
                    "MoonEP did not collectively reject asymmetric input route counts"
                )

            plan = op.build_routing_plan(expert_indices)
            current_tpe_all = op.context.planning_counts_mem.view(
                world_size, op.context.planning_num_bins
            )[:, :num_experts].cpu().contiguous()
            torch_plan = plan_moonep_b0_b3(current_tpe_all)
            torch.testing.assert_close(
                op.context.planning_alloc_cumsum.cpu(),
                torch_plan.alloc_cumsum,
            )
            torch.testing.assert_close(
                op.context.planning_experts_to_copy.cpu(),
                torch_plan.experts_to_copy,
            )
            torch.testing.assert_close(
                op.context.planning_inverse_experts_to_copy.cpu(),
                build_inverse_experts_to_copy(
                    torch_plan.experts_to_copy, num_experts
                ),
            )
            if plan.active_physical_experts_per_rank != op.experts_per_rank + 1:
                raise AssertionError(
                    "single-hot-expert plan did not tighten the active replica "
                    "high-water bound to one slot"
                )
            remote_received = plan.received_routes_per_expert[
                op.experts_per_rank:
            ].sum()
            dist.all_reduce(remote_received, op=dist.ReduceOp.SUM)
            if int(remote_received.item()) <= 0:
                raise AssertionError("hot expert did not hit a remote replica slot")
            if 0 not in plan.experts_to_copy[1].cpu().tolist():
                raise AssertionError("rank 1 did not prefetch logical expert 0")

            passed = run_full_one(
                op,
                hs,
                expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-hot-expert-full",
                rank,
                device,
            )
            if not passed:
                raise AssertionError("MoonEP hot-expert forward mismatched golden")
            clean_output = bench_module.ascend_full_post_routing(
                op,
                hs,
                expert_indices,
                packed_w1,
                w2,
                routing_weights,
            )
            torch.npu.synchronize(device)
            expected_experts = op._replica_experts_cache.clone()
            replica_destinations = [
                destination
                for destination, row in enumerate(expected_experts)
                if bool((row >= 0).any())
            ]
            for weight_name in ("down", "gate_up"):
                for destination_rank in replica_destinations:
                    bench_module._prove_replica_weight_consumption(
                        op,
                        expected_experts,
                        hs,
                        expert_indices,
                        routing_weights,
                        packed_w1,
                        w2,
                        clean_output,
                        device,
                        dist.group.WORLD,
                        weight_name=weight_name,
                        destination_rank=destination_rank,
                    )
            del clean_output

            # Routing metadata is backed by a single shared workspace.  Build
            # the same plan twice through the full-forward hook contract so
            # replica preparation/layout remain valid; only the first plan's
            # generation is stale and must be rejected before a kernel launch.
            def prepare_primary_plan(experts_to_copy_cpu, experts_to_copy_device):
                op._begin_replica_prefetch(
                    experts_to_copy_cpu,
                    experts_to_copy_device,
                    packed_w1,
                    w2,
                )

            superseded_primary_plan = op.build_routing_plan(
                expert_indices,
                moonep_plan_hook=prepare_primary_plan,
            )
            current_primary_plan = op.build_routing_plan(
                expert_indices,
                moonep_plan_hook=prepare_primary_plan,
            )
            rejected_superseded_plan = False
            try:
                op.dispatch_fc1(
                    hs,
                    expert_indices,
                    superseded_primary_plan,
                    packed_w1,
                    routing_weights=routing_weights,
                    replica_fc1_weight=op._replica_weight_buffers.gate_up,
                    replica_weight_ready=op.context.replica_gate_ready,
                    replica_weight_epoch=op._active_replica_weight_epoch,
                )
            except RuntimeError as exc:
                rejected_superseded_plan = "superseded operator" in str(exc)
            rejected_flag.fill_(int(rejected_superseded_plan))
            dist.all_reduce(
                rejected_flag,
                op=dist.ReduceOp.MIN,
                group=dist.group.WORLD,
            )
            if not bool(rejected_flag.item()):
                raise AssertionError(
                    "MoonEP staged dispatch accepted a superseded workspace plan"
                )
            del current_primary_plan, superseded_primary_plan

            mutated_expert_indices = expert_indices.clone()
            mutation_plan = op.build_routing_plan(
                mutated_expert_indices,
                moonep_plan_hook=prepare_primary_plan,
            )
            mutated_expert_indices.add_(0)
            rejected_mutated_routes = False
            try:
                op.dispatch_fc1(
                    hs,
                    mutated_expert_indices,
                    mutation_plan,
                    packed_w1,
                    routing_weights=routing_weights,
                    replica_fc1_weight=op._replica_weight_buffers.gate_up,
                    replica_weight_ready=op.context.replica_gate_ready,
                    replica_weight_epoch=op._active_replica_weight_epoch,
                )
            except RuntimeError as exc:
                rejected_mutated_routes = "modified after routing plan" in str(exc)
            rejected_flag.fill_(int(rejected_mutated_routes))
            dist.all_reduce(
                rejected_flag,
                op=dist.ReduceOp.MIN,
                group=dist.group.WORLD,
            )
            if not bool(rejected_flag.item()):
                raise AssertionError(
                    "MoonEP staged dispatch accepted a mutated routing tensor"
                )
            del mutation_plan, mutated_expert_indices

            initial_tpe_all = current_tpe_all.clone()
            shifted_expert_indices = torch.ones_like(expert_indices)
            staged_shifted_plan = op.build_routing_plan(shifted_expert_indices)
            rejected_stale_replica_layout = False
            try:
                op.dispatch_fc1(
                    hs,
                    shifted_expert_indices,
                    staged_shifted_plan,
                    packed_w1,
                    routing_weights=routing_weights,
                    replica_fc1_weight=op._replica_weight_buffers.gate_up,
                    replica_weight_ready=op.context.replica_gate_ready,
                    replica_weight_epoch=op._active_replica_weight_epoch,
                )
            except RuntimeError as exc:
                rejected_stale_replica_layout = (
                    "replica preparation by full forward" in str(exc)
                    or "prepared replica layout" in str(exc)
                )
            rejected_flag.fill_(int(rejected_stale_replica_layout))
            dist.all_reduce(
                rejected_flag,
                op=dist.ReduceOp.MIN,
                group=dist.group.WORLD,
            )
            if not bool(rejected_flag.item()):
                raise AssertionError(
                    "MoonEP staged dispatch accepted a plan with stale replicas"
                )
            passed = run_full_one(
                op,
                hs,
                shifted_expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-hot-expert-plan-switch",
                rank,
                device,
            )
            if (
                not passed
                or torch.equal(
                    op.context.planning_counts_mem.view(
                        world_size, op.context.planning_num_bins
                    )[:, :num_experts].cpu(),
                    initial_tpe_all,
                )
            ):
                raise AssertionError(
                    "MoonEP device planner did not consume the new route counts"
                )
            passed = run_full_one(
                op,
                hs,
                expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-hot-expert-plan-restore",
                rank,
                device,
            )
            if (
                not passed
                or not torch.equal(
                    op.context.planning_counts_mem.view(
                        world_size, op.context.planning_num_bins
                    )[:, :num_experts].cpu(),
                    initial_tpe_all,
                )
            ):
                raise AssertionError(
                    "MoonEP device planner did not restore the original route counts"
                )
            cached_weight_key = op._replica_weight_cache_key
            cached_weight_epoch = op._replica_weight_epoch
            passed = run_full_one(
                op,
                hs,
                expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-hot-expert-cache-reuse",
                rank,
                device,
            )
            if (
                not passed
                or op._replica_weight_cache_key != cached_weight_key
                or op._replica_weight_epoch != cached_weight_epoch
                or not op._replica_weight_cache_valid
                or op._replica_prefetch_pending
            ):
                raise AssertionError("MoonEP warm replica cache reuse failed")
            epoch_before_update = op._replica_weight_epoch
            if rank == 0:
                # Change one owner's hot expert in place. Other ranks still see
                # a local cache hit, so the EP-wide MIN decision must force all
                # ranks through the same prefetch/barrier epoch.
                packed_w1[0].add_(0.125)
                w_gate[0].add_(0.125)
                w_up[0].add_(0.125)
            passed = run_full_one(
                op,
                hs,
                expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-hot-expert-single-owner-weight-update",
                rank,
                device,
            )
            if (
                not passed
                or op._replica_weight_epoch != epoch_before_update + 1
                or not op._replica_weight_cache_valid
                or op._replica_prefetch_pending
            ):
                raise AssertionError(
                    "MoonEP single-owner weight update did not refresh every rank"
                )
        finally:
            op.finalize()


def run_moonep_planning_oracle_case(rank: int, world_size: int) -> None:
    """Compare the device planner with Torch at Kimi's non-power-of-two E."""
    if world_size != 8:
        raise ValueError("the MoonEP planner oracle case requires 8 ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("MoonEP planner oracle requires NPU and ACLSHMEM")

    num_experts = 896
    experts_per_rank = num_experts // world_size
    capacity = experts_per_rank
    row = torch.zeros(num_experts, dtype=torch.int32)
    row[3 * experts_per_rank : 4 * experts_per_rank] = 1
    if int(row.sum().item()) != capacity:
        raise AssertionError("planner oracle row total must equal the capacity bin")
    tpe_all_cpu = row.repeat(world_size, 1).contiguous()
    torch_plan = plan_moonep_b0_b3(tpe_all_cpu)
    row_stride = 1 << num_experts.bit_length()
    device = f"npu:{rank}"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        tpe_all = torch.zeros(
            (world_size, row_stride), dtype=torch.int32, device=device
        )
        tpe_all[:, :num_experts].copy_(tpe_all_cpu.to(device))
        tpe_all[:, num_experts] = capacity
        expert_count = torch.empty(
            num_experts, dtype=torch.int64, device=device
        )
        transfers = torch.empty(
            (world_size, world_size), dtype=torch.int64, device=device
        )
        allocation = torch.empty(
            (world_size, num_experts), dtype=torch.int64, device=device
        )
        alloc_cumsum = torch.empty(
            (num_experts, world_size), dtype=torch.int32, device=device
        )
        experts_to_copy = torch.empty(
            (world_size, experts_per_rank), dtype=torch.int32, device=device
        )
        inverse = torch.empty(
            (world_size, num_experts), dtype=torch.int32, device=device
        )
        replica_counts = torch.empty(
            world_size, dtype=torch.int32, device=device
        )

        launch_moonep_b0_b3(
            tpe_all,
            expert_count,
            transfers,
            allocation,
            alloc_cumsum,
            experts_to_copy,
            inverse,
            replica_counts,
            world_size=world_size,
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            row_stride=row_stride,
        )
        torch.npu.synchronize(device)
        torch.testing.assert_close(
            alloc_cumsum.cpu(), torch_plan.alloc_cumsum
        )
        torch.testing.assert_close(
            experts_to_copy.cpu(), torch_plan.experts_to_copy
        )
        torch.testing.assert_close(
            inverse.cpu(),
            build_inverse_experts_to_copy(
                torch_plan.experts_to_copy, num_experts
            ),
        )
        torch.testing.assert_close(
            replica_counts.cpu(),
            (torch_plan.experts_to_copy >= 0).sum(dim=1).to(torch.int32),
        )


def run_moonep_moderate_wide_forward_case(rank: int, world_size: int) -> None:
    """Validate the W8 performance route against the logical Torch golden."""
    if world_size != 8:
        raise ValueError("the moderate-wide MoonEP forward case requires 8 ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("MoonEP forward correctness requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 64, 256, 256, 8, 112
    experts_per_rank = num_experts // world_size
    owner_counts = (19, 27, 11, 11, 15, 15, 15, 15)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    torch.manual_seed(1701 + rank)

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        op = FusedMoEForward(
            dist.group.WORLD,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=1.0,
                enable_moonep=True,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            hidden_states = torch.randn(
                (tokens, hidden), dtype=dtype, device=device
            ).mul_(0.5).contiguous()
            routing_weights = make_routing_weights(
                tokens, topk, device, seed=1702 + rank
            )

            owner_period = torch.tensor(
                [
                    owner
                    for owner, count in enumerate(owner_counts)
                    for _ in range(count)
                ],
                dtype=torch.int64,
                device=device,
            )
            route_ids = torch.arange(
                tokens * topk, dtype=torch.int64, device=device
            )
            global_route_ids = route_ids + rank * tokens * topk
            owners = owner_period[route_ids.remainder(len(owner_period))]
            expert_indices = (
                owners * experts_per_rank
                + global_route_ids.remainder(experts_per_rank)
            ).view(tokens, topk).to(torch.int32).contiguous()
            sorted_per_token = torch.sort(expert_indices, dim=1).values
            if bool(
                (sorted_per_token[:, 1:] == sorted_per_token[:, :-1])
                .any()
                .item()
            ):
                raise AssertionError(
                    "moderate-wide routes repeat an expert within a token"
                )

            owner_routes = torch.bincount(
                torch.div(
                    expert_indices.reshape(-1).to(torch.int64),
                    experts_per_rank,
                    rounding_mode="floor",
                ),
                minlength=world_size,
            ).to(torch.int64)
            dist.all_reduce(owner_routes, op=dist.ReduceOp.SUM)
            expected_owner_routes = torch.tensor(
                [
                    count * world_size * tokens * topk // len(owner_period)
                    for count in owner_counts
                ],
                dtype=torch.int64,
                device=device,
            )
            torch.testing.assert_close(owner_routes, expected_owner_routes)

            plan = op.build_routing_plan(expert_indices)
            copy_counts = (plan.experts_to_copy >= 0).sum(dim=1).cpu().tolist()
            if copy_counts != [0, 0, 3, 3, 1, 1, 1, 1]:
                raise AssertionError(
                    f"unexpected moderate-wide replica counts: {copy_counts}"
                )
            passed = run_full_one(
                op,
                hidden_states,
                expert_indices,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                "moonep-w8-moderate-wide-full",
                rank,
                device,
            )
            passed_flag = torch.tensor(
                [int(passed)], dtype=torch.int32, device=device
            )
            dist.all_reduce(
                passed_flag, op=dist.ReduceOp.MIN, group=dist.group.WORLD
            )
            if not bool(passed_flag.item()):
                raise AssertionError(
                    "W8 moderate-wide forward mismatched logical Torch golden"
                )
        finally:
            op.finalize()


def _assert_physical_saved_layout(saved_phys, tokens, topk):
    """Check the saved_phys invariants the physical backward will rely on."""
    if saved_phys["use_moonep"] is not True:
        raise AssertionError("saved_phys must be flagged use_moonep")
    expert_counts = saved_phys["expert_counts"]
    if expert_counts.dtype != torch.int32:
        raise AssertionError(f"expert_counts must be int32, got {expert_counts.dtype}")
    if tuple(expert_counts.shape) != (saved_phys["physical_experts_per_rank"],):
        raise AssertionError("expert_counts must cover every physical slot")
    if int(expert_counts.sum().item()) != tokens * topk:
        raise AssertionError("physical slot counts must conserve every route")
    if saved_phys["M"] != saved_phys["total_recv"]:
        raise AssertionError("saved M must equal total_recv")
    for key in ("sort_idxs", "inv_sort", "local_sort_idxs", "inv_local"):
        permutation = saved_phys[key]
        expected = torch.arange(
            permutation.numel(), dtype=torch.int64, device=permutation.device
        )
        if not torch.equal(torch.sort(permutation).values, expected):
            raise AssertionError(f"saved_phys[{key!r}] is not a permutation")
    if sum(saved_phys["splits_send_list"]) != tokens * topk:
        raise AssertionError("send splits must conserve every route")
    if sum(saved_phys["splits_recv_list"]) != tokens * topk:
        raise AssertionError("receive splits must conserve every route")
    if not torch.equal(
        saved_phys["plan_recv_counts_by_source_expert"].to(torch.int64).sum(dim=0),
        expert_counts.to(torch.int64),
    ):
        raise AssertionError("plan receive counts disagree with expert_counts")


def _assert_replica_tables_match_owners(
    rank,
    experts_to_copy_cpu,
    replica_gate_up,
    replica_down,
    home_gate_up,
    home_down,
    ep_group,
):
    """Compare the HCCL replica gather with an all-gather of the home tables."""
    world_size = dist.get_world_size(ep_group)
    experts_per_rank = home_gate_up.shape[0]
    gate_up_tables = [torch.empty_like(home_gate_up) for _ in range(world_size)]
    dist.all_gather(gate_up_tables, home_gate_up, group=ep_group)
    down_tables = [torch.empty_like(home_down) for _ in range(world_size)]
    dist.all_gather(down_tables, home_down, group=ep_group)
    for slot, expert in enumerate(
        experts_to_copy_cpu[rank].to(torch.int64).tolist()
    ):
        if expert < 0:
            if bool((replica_gate_up[slot] != 0).any()) or bool(
                (replica_down[slot] != 0).any()
            ):
                raise AssertionError("an empty replica slot must stay zero")
            continue
        owner = expert // experts_per_rank
        local_row = expert % experts_per_rank
        torch.testing.assert_close(
            replica_gate_up[slot], gate_up_tables[owner][local_row]
        )
        torch.testing.assert_close(
            replica_down[slot], down_tables[owner][local_row]
        )


def run_moonep_physical_forward_hot_expert_case(
    rank: int,
    world_size: int,
) -> None:
    """Replay an all-hot MoonEP plan in torch against the logical golden."""
    if world_size not in (2, 4):
        raise ValueError("the physical hot-expert case requires two or four ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("physical MoonEP forward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    experts_per_rank = num_experts // world_size
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=1.0,
                enable_moonep=True,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            hs, _ = prepare_inputs(
                tokens,
                hidden,
                num_experts,
                topk,
                dtype,
                device,
                seed=803 + rank,
            )
            expert_indices = torch.zeros(
                (tokens, topk), dtype=torch.int32, device=device
            )
            routing_weights = torch.empty(
                (tokens, topk), dtype=torch.float32, device=device
            )
            routing_weights[:, 0] = 0.25
            routing_weights[:, 1] = 0.75

            plan = op.build_routing_plan(expert_indices)
            experts_to_copy_cpu = plan.experts_to_copy_cpu
            if not bool((experts_to_copy_cpu == 0).any()):
                raise AssertionError("the all-hot plan did not replicate expert 0")
            replica_routes = plan.received_routes_per_expert[
                experts_per_rank:
            ].sum().clone()
            dist.all_reduce(replica_routes, op=dist.ReduceOp.SUM, group=ep_group)
            if int(replica_routes.item()) <= 0:
                raise AssertionError(
                    "the all-hot plan routed no traffic through replicas"
                )

            replica_gate_up, replica_down = gather_replica_weights_via_hccl(
                experts_to_copy_cpu, packed_w1, w2, ep_group
            )
            _assert_replica_tables_match_owners(
                rank,
                experts_to_copy_cpu,
                replica_gate_up,
                replica_down,
                packed_w1,
                w2,
                ep_group,
            )

            output, saved_phys = build_physical_saved_from_plan(
                plan,
                hs,
                routing_weights,
                packed_w1,
                w2,
                replica_gate_up,
                replica_down,
                ep_group=ep_group,
            )
            _assert_physical_saved_layout(saved_phys, tokens, topk)

            expected = torch_moe_fwd_golden(
                hs,
                routing_weights,
                expert_indices,
                w_gate,
                w_up,
                w2,
                num_experts,
                ep_group,
            )
            assert_close(output, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        finally:
            op.finalize()


def run_moonep_physical_forward_moderate_wide_case(
    rank: int,
    world_size: int,
) -> None:
    """Replay the W8 moderate-wide MoonEP plan in torch against the golden."""
    if world_size != 8:
        raise ValueError("the moderate-wide physical case requires 8 ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("physical MoonEP forward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 64, 256, 256, 8, 112
    experts_per_rank = num_experts // world_size
    owner_counts = (19, 27, 11, 11, 15, 15, 15, 15)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    torch.manual_seed(1801 + rank)

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=1.0,
                enable_moonep=True,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            hidden_states = torch.randn(
                (tokens, hidden), dtype=dtype, device=device
            ).mul_(0.5).contiguous()
            routing_weights = make_routing_weights(
                tokens, topk, device, seed=1802 + rank
            )

            owner_period = torch.tensor(
                [
                    owner
                    for owner, count in enumerate(owner_counts)
                    for _ in range(count)
                ],
                dtype=torch.int64,
                device=device,
            )
            route_ids = torch.arange(
                tokens * topk, dtype=torch.int64, device=device
            )
            global_route_ids = route_ids + rank * tokens * topk
            owners = owner_period[route_ids.remainder(len(owner_period))]
            expert_indices = (
                owners * experts_per_rank
                + global_route_ids.remainder(experts_per_rank)
            ).view(tokens, topk).to(torch.int32).contiguous()
            sorted_per_token = torch.sort(expert_indices, dim=1).values
            if bool(
                (sorted_per_token[:, 1:] == sorted_per_token[:, :-1])
                .any()
                .item()
            ):
                raise AssertionError(
                    "moderate-wide routes repeat an expert within a token"
                )

            plan = op.build_routing_plan(expert_indices)
            copy_counts = (plan.experts_to_copy >= 0).sum(dim=1).cpu().tolist()
            if copy_counts != [0, 0, 3, 3, 1, 1, 1, 1]:
                raise AssertionError(
                    f"unexpected moderate-wide replica counts: {copy_counts}"
                )

            replica_gate_up, replica_down = gather_replica_weights_via_hccl(
                plan.experts_to_copy_cpu, packed_w1, w2, ep_group
            )
            output, saved_phys = build_physical_saved_from_plan(
                plan,
                hidden_states,
                routing_weights,
                packed_w1,
                w2,
                replica_gate_up,
                replica_down,
                ep_group=ep_group,
            )
            _assert_physical_saved_layout(saved_phys, tokens, topk)
            if copy_counts[rank] and not int(
                saved_phys["expert_counts"][experts_per_rank:].sum().item()
            ):
                raise AssertionError(
                    "a rank holding replicas executed no replica-slot traffic"
                )

            expected = torch_moe_fwd_golden(
                hidden_states,
                routing_weights,
                expert_indices,
                w_gate,
                w_up,
                w2,
                num_experts,
                ep_group,
            )
            assert_close(output, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)
        finally:
            op.finalize()


def _reduce_moonep_replica_wgrads(rank, saved_phys, triton_result, ep_group):
    """Test-side stand-in for the M3 owner-pull replica grad reduce.

    Every rank's replica weight-grad tables are gathered over HCCL and the ones
    owned by this rank are accumulated, in (source rank, replica slot)
    lexicographic order, in fp32 onto the physical home-segment seed. A logical
    expert's routed rows partition exactly over the owner home slot plus every
    replica slot holding it, so this reduction must reproduce the logical golden
    weight grads.
    """
    home_experts = int(saved_phys["home_experts_per_rank"])
    experts_to_copy = saved_phys["experts_to_copy_cpu"].to(torch.int64).tolist()
    gate_up_tables = all_gather_list(
        triton_result["_replica_grad_gate_up"], ep_group)
    down_tables = all_gather_list(triton_result["_replica_grad_down"], ep_group)
    grad_fc1 = torch.cat(
        (triton_result["grad_fc1_1"], triton_result["grad_fc1_2"]), dim=1
    ).float()
    grad_fc2 = triton_result["grad_fc2"].float()
    accumulated = 0
    for source_rank, etc_row in enumerate(experts_to_copy):
        for slot, expert in enumerate(etc_row):
            if expert < 0 or expert // home_experts != rank:
                continue
            local_expert = expert % home_experts
            grad_fc1[local_expert] += gate_up_tables[source_rank][slot].float()
            grad_fc2[local_expert] += down_tables[source_rank][slot].float()
            accumulated += 1
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return grad_fc1_1, grad_fc1_2, grad_fc2, accumulated


def _assert_gradients_match_golden(
    rank, triton_result, torch_result, ep_group, label
):
    """Compare the five canonical keys against the logical golden, collectively."""
    all_ok, details = compare_backward_gradients(triton_result, torch_result)
    flag = torch.tensor(
        [1 if all_ok else 0], dtype=torch.int32, device=torch_result["grad_fc2"].device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if rank == 0 and not bool(flag.item()):
        print(f"{label} gradient details: {details}", flush=True)
    if not bool(flag.item()):
        raise AssertionError(
            f"{label} physical backward mismatched the logical golden"
        )


def _assert_moonep_backward_matches_logical_golden(
    rank, saved_phys, triton_result, torch_result, ep_group, label
):
    """Dual oracle for the physical 5-op backward against the logical golden.

    ``grad_hidden`` / ``grad_routing_weights`` are per-route reductions and
    compare directly, while the three weight-grad keys compare only after the
    (source rank, slot)-ordered fp32 accumulation of every replica table onto
    the home seed. A failing hidden/routing key therefore points at a dispatch
    layout or transport bug, and a failing weight-grad key at a dual-table GEMM
    or reduction bug.
    """
    grad_fc1_1, grad_fc1_2, grad_fc2, accumulated = _reduce_moonep_replica_wgrads(
        rank, saved_phys, triton_result, ep_group)
    merged = dict(triton_result)
    merged.update(
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2)
    _assert_gradients_match_golden(rank, merged, torch_result, ep_group, label)
    return accumulated


def _assert_moonep_replica_grad_shapes(
    triton_result, saved_phys, experts_per_rank, hidden, ffn
):
    """The replica segments must mirror the packed replica weight-table shapes."""
    replica_slots = (
        int(saved_phys["physical_experts_per_rank"]) - experts_per_rank
    )
    if tuple(triton_result["_replica_grad_gate_up"].shape) != (
        replica_slots, 2 * ffn, hidden
    ):
        raise AssertionError(
            "unexpected replica gate/up grad shape: "
            f"{tuple(triton_result['_replica_grad_gate_up'].shape)}"
        )
    if tuple(triton_result["_replica_grad_down"].shape) != (
        replica_slots, hidden, ffn
    ):
        raise AssertionError(
            "unexpected replica down grad shape: "
            f"{tuple(triton_result['_replica_grad_down'].shape)}"
        )


def run_moonep_backward_hot_expert_case(rank: int, world_size: int) -> None:
    """Run the 5-op backward over an all-hot physical MoonEP plan (w2/w4)."""
    if world_size not in (2, 4):
        raise ValueError("the MoonEP hot-expert backward case requires two or four ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("physical MoonEP backward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    experts_per_rank = num_experts // world_size
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must be the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below allocates its own heap
        # objects to build the routing plan.
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk, tokens * topk, hidden, dtype, rank, ep_group
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=1.0,
                    enable_moonep=True,
                ),
            )
            try:
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = make_down_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                hs, _ = prepare_inputs(
                    tokens,
                    hidden,
                    num_experts,
                    topk,
                    dtype,
                    device,
                    seed=903 + rank,
                )
                expert_indices = torch.zeros(
                    (tokens, topk), dtype=torch.int32, device=device
                )
                routing_weights = torch.empty(
                    (tokens, topk), dtype=torch.float32, device=device
                )
                routing_weights[:, 0] = 0.25
                routing_weights[:, 1] = 0.75
                torch.manual_seed(904 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                plan = op.build_routing_plan(expert_indices)
                replica_routes = plan.received_routes_per_expert[
                    experts_per_rank:
                ].sum().clone()
                dist.all_reduce(
                    replica_routes, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(replica_routes.item()) <= 0:
                    raise AssertionError(
                        "the all-hot plan routed no traffic through replicas"
                    )

                replica_gate_up, replica_down = gather_replica_weights_via_hccl(
                    plan.experts_to_copy_cpu, packed_w1, w2, ep_group
                )
                output, saved_phys = build_physical_saved_from_plan(
                    plan,
                    hs,
                    routing_weights,
                    packed_w1,
                    w2,
                    replica_gate_up,
                    replica_down,
                    ep_group=ep_group,
                )
                _assert_physical_saved_layout(saved_phys, tokens, topk)

                with torch.no_grad():
                    _, home_saved = moe_forward(
                        hs,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    torch_result = backward_torch_baseline(home_saved, dy)
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem
                    )
                _assert_moonep_replica_grad_shapes(
                    triton_result, saved_phys, experts_per_rank, hidden, ffn
                )
                accumulated = _assert_moonep_backward_matches_logical_golden(
                    rank,
                    saved_phys,
                    triton_result,
                    torch_result,
                    ep_group,
                    "moonep-hot-expert-backward",
                )
                contributions = torch.tensor(
                    [accumulated], dtype=torch.int64, device=device
                )
                dist.all_reduce(
                    contributions, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(contributions.item()) <= 0:
                    raise AssertionError(
                        "no owner accumulated a replica weight gradient"
                    )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


def run_moonep_backward_moderate_wide_case(rank: int, world_size: int) -> None:
    """Run the 5-op backward over the W8 moderate-wide physical MoonEP plan."""
    if world_size != 8:
        raise ValueError("the moderate-wide MoonEP backward case requires 8 ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("physical MoonEP backward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 64, 256, 256, 8, 112
    experts_per_rank = num_experts // world_size
    owner_counts = (19, 27, 11, 11, 15, 15, 15, 15)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    torch.manual_seed(1901 + rank)

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must be the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below allocates its own heap
        # objects to build the routing plan.
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk, tokens * topk, hidden, dtype, rank, ep_group
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=1.0,
                    enable_moonep=True,
                ),
            )
            try:
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = make_down_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                hidden_states = torch.randn(
                    (tokens, hidden), dtype=dtype, device=device
                ).mul_(0.5).contiguous()
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=1902 + rank
                )
                torch.manual_seed(1903 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                owner_period = torch.tensor(
                    [
                        owner
                        for owner, count in enumerate(owner_counts)
                        for _ in range(count)
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                route_ids = torch.arange(
                    tokens * topk, dtype=torch.int64, device=device
                )
                global_route_ids = route_ids + rank * tokens * topk
                owners = owner_period[route_ids.remainder(len(owner_period))]
                expert_indices = (
                    owners * experts_per_rank
                    + global_route_ids.remainder(experts_per_rank)
                ).view(tokens, topk).to(torch.int32).contiguous()
                sorted_per_token = torch.sort(expert_indices, dim=1).values
                if bool(
                    (sorted_per_token[:, 1:] == sorted_per_token[:, :-1])
                    .any()
                    .item()
                ):
                    raise AssertionError(
                        "moderate-wide routes repeat an expert within a token"
                    )

                plan = op.build_routing_plan(expert_indices)
                copy_counts = (plan.experts_to_copy >= 0).sum(dim=1).cpu().tolist()
                if copy_counts != [0, 0, 3, 3, 1, 1, 1, 1]:
                    raise AssertionError(
                        f"unexpected moderate-wide replica counts: {copy_counts}"
                    )

                replica_gate_up, replica_down = gather_replica_weights_via_hccl(
                    plan.experts_to_copy_cpu, packed_w1, w2, ep_group
                )
                output, saved_phys = build_physical_saved_from_plan(
                    plan,
                    hidden_states,
                    routing_weights,
                    packed_w1,
                    w2,
                    replica_gate_up,
                    replica_down,
                    ep_group=ep_group,
                )
                _assert_physical_saved_layout(saved_phys, tokens, topk)

                with torch.no_grad():
                    _, home_saved = moe_forward(
                        hidden_states,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    torch_result = backward_torch_baseline(home_saved, dy)
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem
                    )
                _assert_moonep_replica_grad_shapes(
                    triton_result, saved_phys, experts_per_rank, hidden, ffn
                )
                accumulated = _assert_moonep_backward_matches_logical_golden(
                    rank,
                    saved_phys,
                    triton_result,
                    torch_result,
                    ep_group,
                    "moonep-w8-moderate-wide-backward",
                )
                contributions = torch.tensor(
                    [accumulated], dtype=torch.int64, device=device
                )
                dist.all_reduce(
                    contributions, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(contributions.item()) <= 0:
                    raise AssertionError(
                        "no owner accumulated a replica weight gradient"
                    )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


# ----------------------------------------------------------------------------
# M3: symmetric replica slots + owner-pull reduction
# ----------------------------------------------------------------------------

def _assert_symmetric_replica_slots_match_owners(
    rank,
    experts_to_copy_cpu,
    replica_gate_up,
    replica_down,
    home_gate_up,
    home_down,
    ep_group,
):
    """Check the FILLED slots of the symmetric tables against the home rows.

    Unlike :func:`_assert_replica_tables_match_owners` the empty slots are
    skipped: ``allocate_replica_weight_buffers`` does not zero the symmetric
    heap and no kernel ever reads an empty slot, so its contents carry no
    contract.  Returns the number of filled slots checked on this rank.
    """
    world_size = dist.get_world_size(ep_group)
    experts_per_rank = home_gate_up.shape[0]
    gate_up_tables = [torch.empty_like(home_gate_up) for _ in range(world_size)]
    dist.all_gather(gate_up_tables, home_gate_up, group=ep_group)
    down_tables = [torch.empty_like(home_down) for _ in range(world_size)]
    dist.all_gather(down_tables, home_down, group=ep_group)
    checked = 0
    for slot, expert in enumerate(
        experts_to_copy_cpu[rank].to(torch.int64).tolist()
    ):
        if expert < 0:
            continue
        owner = expert // experts_per_rank
        local_row = expert % experts_per_rank
        torch.testing.assert_close(
            replica_gate_up[slot], gate_up_tables[owner][local_row]
        )
        torch.testing.assert_close(
            replica_down[slot], down_tables[owner][local_row]
        )
        checked += 1
    return checked


def _reduce_symmetric_replica_wgrads(rank, transport, triton_result, ep_group):
    """HCCL oracle for the symmetric transport: same order, independent path.

    Seeds every owner's fp32 accumulator with the *pre-reduction* home segment
    the backward reports and accumulates each rank's replica segment in
    (source rank, slot) lexicographic order — the reduction the owner-pull
    kernel has to reproduce on device.
    """
    home_experts = transport.experts_per_rank
    experts_to_copy = transport.experts_to_copy_cpu.to(torch.int64).tolist()
    gate_up_tables = all_gather_list(
        triton_result["_replica_grad_gate_up"], ep_group)
    down_tables = all_gather_list(
        triton_result["_replica_grad_down"], ep_group)
    grad_fc1 = triton_result["_home_grad_fc1"].float()
    grad_fc2 = triton_result["_home_grad_fc2"].float()
    accumulated = 0
    for source_rank, etc_row in enumerate(experts_to_copy):
        for slot, expert in enumerate(etc_row):
            if expert < 0 or expert // home_experts != rank:
                continue
            local_expert = expert % home_experts
            grad_fc1[local_expert] += gate_up_tables[source_rank][slot].float()
            grad_fc2[local_expert] += down_tables[source_rank][slot].float()
            accumulated += 1
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)
    return grad_fc1_1, grad_fc1_2, grad_fc2, accumulated


def _assert_symmetric_transport_stages(
    rank,
    transport,
    captured_slots,
    triton_result,
    ep_group,
    label,
):
    """Assert the sink / owner-reduce / zero stages of one symmetric backward.

    ``captured_slots`` holds the stream-ordered copies of the consumed slots
    taken between the sink and the reduce, so the slots are compared bit-exactly
    against the physical replica gradients before the reduction touched them.
    The canonical weight-grad keys are then compared against the independent
    HCCL oracle of :func:`_reduce_symmetric_replica_wgrads`, which accumulates
    the very same replica gradients in the very same (source, slot) order in
    fp32. Returns the number of replica contributions this owner accumulated.
    """
    consumed = transport.consumed_slots()
    failures = []
    if captured_slots["gate_up"].shape[0] != len(consumed):
        failures.append("the sink capture did not cover every consumed slot")
    for index, slot in enumerate(consumed):
        if not torch.equal(
            captured_slots["gate_up"][index],
            triton_result["_replica_grad_gate_up"][slot].t(),
        ):
            failures.append(f"gate/up slot {slot} did not hold its sunk gradient")
        if not torch.equal(
            captured_slots["down"][index],
            triton_result["_replica_grad_down"][slot],
        ):
            failures.append(f"down slot {slot} did not hold its sunk gradient")

    reduced_fc1 = torch.cat(
        (triton_result["grad_fc1_1"], triton_result["grad_fc1_2"]), dim=1
    )
    oracle_fc1_1, oracle_fc1_2, oracle_fc2, accumulated = (
        _reduce_symmetric_replica_wgrads(rank, transport, triton_result, ep_group)
    )
    try:
        assert_close(
            reduced_fc1,
            torch.cat((oracle_fc1_1, oracle_fc1_2), dim=1),
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )
    except AssertionError as error:
        failures.append(f"fc1 reduction disagreed with the oracle: {error}")
    try:
        assert_close(
            triton_result["grad_fc2"], oracle_fc2, rtol=GRAD_RTOL, atol=GRAD_ATOL
        )
    except AssertionError as error:
        failures.append(f"fc2 reduction disagreed with the oracle: {error}")

    # The zeroing is the last transport stage; drain the stream before reading.
    torch.npu.synchronize()
    for slot in consumed:
        if bool((transport.buffers.gate_up[slot] != 0).any()) or bool(
            (transport.buffers.down[slot] != 0).any()
        ):
            failures.append(f"replica slot {slot} was not zeroed after the reduce")
    # Rank-dependent failures must not strand a peer inside a collective above,
    # so the verdict is shared before anybody raises.
    flag = torch.tensor(
        [0 if failures else 1], dtype=torch.int32, device=reduced_fc1.device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if int(flag.item()) != 1:
        raise AssertionError(f"{label}: " + "; ".join(failures))
    return accumulated


def _assert_forward_after_grad_transport(
    op,
    hidden_states,
    expert_indices,
    packed_w1,
    w2,
    routing_weights,
    expected,
    epoch_before_lend,
    rank,
    label,
):
    """Prove the zeroed slots cannot leak into a later forward.

    The first post-backward forward must re-push (the lent hand-off invalidated
    the cache) and the second must be a cache hit — a slot left holding a
    gradient, or a slot zeroed after the push, only survives that second hit
    unnoticed. Both outputs are compared against the same forward golden.
    """
    epoch_after_push = epoch_before_lend + 1
    reloaded = op.forward(
        hidden_states, expert_indices, packed_w1, w2, routing_weights
    )
    if op._replica_weight_epoch != epoch_after_push:
        raise AssertionError(
            f"{label}: the post-backward forward re-pushed "
            f"{op._replica_weight_epoch - epoch_before_lend} times, expected 1"
        )
    if not op._replica_weight_cache_valid:
        raise AssertionError(
            f"{label}: the post-backward forward left the replica cache invalid"
        )
    assert_close(reloaded, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)

    cached = op.forward(
        hidden_states, expert_indices, packed_w1, w2, routing_weights
    )
    if op._replica_weight_epoch != epoch_after_push:
        raise AssertionError(
            f"{label}: the second post-backward forward was not a cache hit"
        )
    assert_close(cached, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)


def run_moonep_backward_symmetric_hot_expert_case(
    rank: int, world_size: int
) -> None:
    """Reduce the replica grads through the forward's own symmetric slots (w2/w4)."""
    if world_size not in (2, 4):
        raise ValueError(
            "the symmetric hot-expert backward case requires two or four ranks"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("symmetric MoonEP backward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    experts_per_rank = num_experts // world_size
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "moonep-symmetric-hot-expert-backward"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must be the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below allocates its own heap
        # objects to build the routing plan.
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk, tokens * topk, hidden, dtype, rank, ep_group
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=1.0,
                    enable_moonep=True,
                ),
            )
            try:
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = make_down_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                hs, _ = prepare_inputs(
                    tokens,
                    hidden,
                    num_experts,
                    topk,
                    dtype,
                    device,
                    seed=953 + rank,
                )
                expert_indices = torch.zeros(
                    (tokens, topk), dtype=torch.int32, device=device
                )
                routing_weights = torch.empty(
                    (tokens, topk), dtype=torch.float32, device=device
                )
                routing_weights[:, 0] = 0.25
                routing_weights[:, 1] = 0.75
                torch.manual_seed(954 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)
                expected = torch_moe_fwd_golden(
                    hs,
                    routing_weights,
                    expert_indices,
                    w_gate,
                    w_up,
                    w2,
                    num_experts,
                    ep_group,
                )

                # One production forward fills the symmetric replica tables via
                # the owner-push; its output doubles as a free golden check.
                produced = op.forward(
                    hs, expert_indices, packed_w1, w2, routing_weights
                )
                assert_close(
                    produced, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                )
                if (
                    op._replica_weight_buffers is None
                    or not op._replica_weight_cache_valid
                ):
                    raise AssertionError(
                        "the production forward did not publish replica weights"
                    )
                epoch_before_lend = op._replica_weight_epoch

                # The plan has to be rebuilt AFTER the forward: forward reuses
                # one planning workspace in place, so an earlier view is stale.
                plan = op.build_routing_plan(expert_indices)
                replica_routes = plan.received_routes_per_expert[
                    experts_per_rank:
                ].sum().clone()
                dist.all_reduce(
                    replica_routes, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(replica_routes.item()) <= 0:
                    raise AssertionError(
                        "the all-hot plan routed no traffic through replicas"
                    )
                checked_slots = _assert_symmetric_replica_slots_match_owners(
                    rank,
                    plan.experts_to_copy_cpu,
                    op._replica_weight_buffers.gate_up,
                    op._replica_weight_buffers.down,
                    packed_w1,
                    w2,
                    ep_group,
                )
                slot_checks = torch.tensor(
                    [checked_slots], dtype=torch.int64, device=device
                )
                dist.all_reduce(
                    slot_checks, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(slot_checks.item()) <= 0:
                    raise AssertionError(
                        "no rank filled a symmetric replica slot"
                    )

                # Replay the same plan in torch against the symmetric table
                # views (no clone): the backward reads the weights in place.
                output, saved_phys = build_physical_saved_from_plan(
                    plan,
                    hs,
                    routing_weights,
                    packed_w1,
                    w2,
                    op._replica_weight_buffers.gate_up,
                    op._replica_weight_buffers.down,
                    ep_group=ep_group,
                )
                _assert_physical_saved_layout(saved_phys, tokens, topk)
                assert_close(
                    output, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                )

                transport = op.lend_replica_weight_tables_for_grad()
                if op._replica_weight_cache_valid is not False:
                    raise AssertionError(
                        "lending the replica tables must invalidate the cache"
                    )
                captured_slots = {}

                def capture_sunk_slots(transport):
                    # Stream-ordered copies taken between the sink and the
                    # reduce, i.e. exactly what barrier #1 publishes.
                    consumed = transport.consumed_slots()
                    captured_slots["gate_up"] = transport.buffers.gate_up[
                        consumed
                    ]
                    captured_slots["down"] = transport.buffers.down[consumed]

                transport.post_sink_hook = capture_sunk_slots

                with torch.no_grad():
                    _, home_saved = moe_forward(
                        hs,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    torch_result = backward_torch_baseline(home_saved, dy)
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem, grad_transport=transport
                    )
                _assert_moonep_replica_grad_shapes(
                    triton_result, saved_phys, experts_per_rank, hidden, ffn
                )
                _assert_gradients_match_golden(
                    rank, triton_result, torch_result, ep_group, label
                )
                accumulated = _assert_symmetric_transport_stages(
                    rank,
                    transport,
                    captured_slots,
                    triton_result,
                    ep_group,
                    label,
                )
                contributions = torch.tensor(
                    [accumulated], dtype=torch.int64, device=device
                )
                dist.all_reduce(
                    contributions, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(contributions.item()) <= 0:
                    raise AssertionError(
                        "no owner pulled a replica weight gradient back"
                    )
                _assert_forward_after_grad_transport(
                    op,
                    hs,
                    expert_indices,
                    packed_w1,
                    w2,
                    routing_weights,
                    expected,
                    epoch_before_lend,
                    rank,
                    label,
                )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


def run_moonep_backward_symmetric_moderate_wide_case(
    rank: int, world_size: int
) -> None:
    """Reduce the replica grads through the symmetric slots (W8 moderate-wide)."""
    if world_size != 8:
        raise ValueError(
            "the symmetric moderate-wide backward case requires 8 ranks"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("symmetric MoonEP backward requires NPU and ACLSHMEM")

    tokens, hidden, ffn, topk, num_experts = 64, 256, 256, 8, 112
    experts_per_rank = num_experts // world_size
    owner_counts = (19, 27, 11, 11, 15, 15, 15, 15)
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "moonep-symmetric-w8-moderate-wide-backward"
    torch.manual_seed(1951 + rank)

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must be the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below allocates its own heap
        # objects to build the routing plan.
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk, tokens * topk, hidden, dtype, rank, ep_group
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=1.0,
                    enable_moonep=True,
                ),
            )
            try:
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = make_down_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                hidden_states = torch.randn(
                    (tokens, hidden), dtype=dtype, device=device
                ).mul_(0.5).contiguous()
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=1952 + rank
                )
                torch.manual_seed(1953 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                owner_period = torch.tensor(
                    [
                        owner
                        for owner, count in enumerate(owner_counts)
                        for _ in range(count)
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                route_ids = torch.arange(
                    tokens * topk, dtype=torch.int64, device=device
                )
                global_route_ids = route_ids + rank * tokens * topk
                owners = owner_period[route_ids.remainder(len(owner_period))]
                expert_indices = (
                    owners * experts_per_rank
                    + global_route_ids.remainder(experts_per_rank)
                ).view(tokens, topk).to(torch.int32).contiguous()
                sorted_per_token = torch.sort(expert_indices, dim=1).values
                if bool(
                    (sorted_per_token[:, 1:] == sorted_per_token[:, :-1])
                    .any()
                    .item()
                ):
                    raise AssertionError(
                        "moderate-wide routes repeat an expert within a token"
                    )
                expected = torch_moe_fwd_golden(
                    hidden_states,
                    routing_weights,
                    expert_indices,
                    w_gate,
                    w_up,
                    w2,
                    num_experts,
                    ep_group,
                )

                produced = op.forward(
                    hidden_states, expert_indices, packed_w1, w2, routing_weights
                )
                assert_close(
                    produced, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                )
                if (
                    op._replica_weight_buffers is None
                    or not op._replica_weight_cache_valid
                ):
                    raise AssertionError(
                        "the production forward did not publish replica weights"
                    )
                epoch_before_lend = op._replica_weight_epoch

                plan = op.build_routing_plan(expert_indices)
                copy_counts = (plan.experts_to_copy >= 0).sum(dim=1).cpu().tolist()
                if copy_counts != [0, 0, 3, 3, 1, 1, 1, 1]:
                    raise AssertionError(
                        f"unexpected moderate-wide replica counts: {copy_counts}"
                    )
                # Ranks 0/1 hold no replica and may own none either: they still
                # have to cross all three transport barriers.
                _assert_symmetric_replica_slots_match_owners(
                    rank,
                    plan.experts_to_copy_cpu,
                    op._replica_weight_buffers.gate_up,
                    op._replica_weight_buffers.down,
                    packed_w1,
                    w2,
                    ep_group,
                )

                output, saved_phys = build_physical_saved_from_plan(
                    plan,
                    hidden_states,
                    routing_weights,
                    packed_w1,
                    w2,
                    op._replica_weight_buffers.gate_up,
                    op._replica_weight_buffers.down,
                    ep_group=ep_group,
                )
                _assert_physical_saved_layout(saved_phys, tokens, topk)
                assert_close(
                    output, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                )

                transport = op.lend_replica_weight_tables_for_grad()
                if op._replica_weight_cache_valid is not False:
                    raise AssertionError(
                        "lending the replica tables must invalidate the cache"
                    )
                captured_slots = {}

                def capture_sunk_slots(transport):
                    consumed = transport.consumed_slots()
                    captured_slots["gate_up"] = transport.buffers.gate_up[
                        consumed
                    ]
                    captured_slots["down"] = transport.buffers.down[consumed]

                transport.post_sink_hook = capture_sunk_slots

                with torch.no_grad():
                    _, home_saved = moe_forward(
                        hidden_states,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    torch_result = backward_torch_baseline(home_saved, dy)
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem, grad_transport=transport
                    )
                _assert_moonep_replica_grad_shapes(
                    triton_result, saved_phys, experts_per_rank, hidden, ffn
                )
                _assert_gradients_match_golden(
                    rank, triton_result, torch_result, ep_group, label
                )
                accumulated = _assert_symmetric_transport_stages(
                    rank,
                    transport,
                    captured_slots,
                    triton_result,
                    ep_group,
                    label,
                )
                contributions = torch.tensor(
                    [accumulated], dtype=torch.int64, device=device
                )
                dist.all_reduce(
                    contributions, op=dist.ReduceOp.SUM, group=ep_group
                )
                if int(contributions.item()) <= 0:
                    raise AssertionError(
                        "no owner pulled a replica weight gradient back"
                    )
                _assert_forward_after_grad_transport(
                    op,
                    hidden_states,
                    expert_indices,
                    packed_w1,
                    w2,
                    routing_weights,
                    expected,
                    epoch_before_lend,
                    rank,
                    label,
                )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


def run_backward_case(rank: int, world_size: int, case: CaseSpec) -> None:
    if world_size != case.world_size:
        raise ValueError(f"worker world size does not match {case.case_id}")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("functional backward requires torch_npu and ACLSHMEM")

    ep_group = dist.group.WORLD
    heap_size = kit.get_ash_size_bytes(default_gb=2)
    with kit.aclshmem_session(rank, world_size, heap_size):
        saved, dy, dtype, device = build_backward_saved(
            case.tokens,
            case.hidden,
            case.ffn,
            case.num_experts,
            case.topk,
            ep_group,
        )
        peer_mem = kit.make_peer_mem(saved, dtype, rank)
        try:
            torch_result = backward_torch_baseline(saved, dy)
            with torch.no_grad():
                triton_result = moe_backward_triton(saved, dy, peer_mem)
            all_ok, details = compare_backward_gradients(triton_result, torch_result)
            flag = torch.tensor([1 if all_ok else 0], dtype=torch.int32, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
            if rank == 0 and not bool(flag.item()):
                print(f"backward gradient details for {case.case_id}: {details}", flush=True)
            if not bool(flag.item()):
                raise AssertionError(f"functional backward case failed: {case.case_id}")
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


# ----------------------------------------------------------------------------
# Stage H (master plan §3.1/§3.2): home-layout native-saved acceptance cases.
#
# The fused forward grows ``return_saved=True`` so one pass produces the
# backward's saved values directly; the torch replay
# (``mega_moe.ops._torch_forward.moe_forward``, 39-key contract at
# ``_torch_forward.py:283-306``) is demoted to the test oracle.  H1 covers the
# metadata / permutation / scalar keys, H2 (same driver, below) extends the
# comparison to the activation and weight-view keys, and H3 adds the autograd
# Function case.
# ----------------------------------------------------------------------------

# Layout-convention-independent keys: the received route multiset
# (``expert_counts`` comes from ``plan.received_routes_per_expert`` — the
# MoERoutingPlan has no ``expert_counts`` field), the grouped-GEMM metadata
# derived from it, and the input routes themselves.  These must match the
# replay bit-for-bit.
NATIVE_SAVED_BITWISE_KEYS = (
    "expert_counts",
    "split_size_cum_per_expert",
    "meta_expert_ids",
    "meta_split_cum",
    "meta_tile_num",
    "meta_tile_num_cum",
    "num_tiles_total",
    "selected_experts",
)

# Send/receive permutations.  The replay sorts sends by destination rank only
# (``argsort(expert_ranks, stable=True)``) while the fused dispatch groups them
# by (destination, local expert) bucket (``plan.send_route_indices``, built by
# the float32 stable argsort at ``runtime/routing.py:645-668``); the arrival
# buffers therefore differ (source-major/flat vs source-major/expert) and the
# raw permutation values are only bitwise comparable when the native saved
# adopts the replay convention.  Both conventions are valid for
# ``moe_backward_triton`` — each is the exact inverse of its own dispatch order
# — so the check accepts either, reports which one matched, and rejects a mix.
NATIVE_SAVED_PERMUTATION_KEYS = (
    ("sort_idxs", "inv_sort"),
    ("local_sort_idxs", "inv_local"),
)

# Plain ints and per-rank split lists (§3.1 E; the native path builds the two
# [world] D2H transfers itself).
NATIVE_SAVED_SCALAR_KEYS = (
    "batch_size",
    "hidden_dim",
    "ffn_dim",
    "topk",
    "world_size",
    "ep_rank",
    "experts_per_rank",
    "M",
    "total_send",
    "total_recv",
    "splits_send_list",
    "splits_recv_list",
)

# ---- Stage H2 (§3.1 D/F): activation keys ---------------------------------
#
# ``recv_hidden_sorted`` is the dispatched-token copy; both paths land in the
# same (local expert, source, flat) receive row order (coordinator-verified
# against the multi-source simulation), so it stays bit-for-bit.  A mismatch
# here is a major signal — the message tags it as such and the assertion is
# never relaxed.
NATIVE_SAVED_H2_BITWISE_KEYS = ("recv_hidden_sorted",)

# Computed activations.  These go through different GEMM kernels, so the plan
# grants the single 2e-2 figure; the comparison runs in fp32 through
# ``assert_close`` like every other numeric gate in this suite.  The native
# ``recv_weights_sorted`` is the bf16 cast of the received routing weights,
# matching the replay's ``flat_weights[...].to(dtype)`` bookkeeping.
NATIVE_SAVED_H2_FLOAT_KEYS = (
    "fc1_output",
    "swiglu_out_weighted",
    "recv_weights_sorted",
)
NATIVE_SAVED_H2_RTOL = 2e-2
NATIVE_SAVED_H2_ATOL = 2e-2

# Zero-copy weight references (§3.1 F).  The native side saves views of the
# forward inputs — ``fc1_combined`` is ``gate_up_weight.transpose(1, 2)`` as a
# stride view, ``fc1_1``/``fc1_2`` are slice-transpose views, ``fc2`` is the
# down weight itself — while the replay side rebuilds its twins from the
# unpacked weights (its ``fc1_combined`` is the contiguous ``cat`` product).
# ``pack_gate_up_weights`` is the exact inverse of those views and both sides
# of this harness are built from the same ``w_gate``/``w_up``/``w2`` tensors,
# so values, dtype, and shape must agree bit-for-bit.  Aliasing the forward
# input's storage is a hard gate (§3.1B zero-copy contract, coordinator
# ruling: a silently materialized fc1_combined alone costs ~4.6 GiB of the
# §3.5 memory budget); contiguity is only reported, because stride views are
# legal.
NATIVE_SAVED_H2_VIEW_KEYS = (
    "fc1_1",
    "fc1_2",
    "fc1_combined",
    "fc2",
)

# §3.1 F deliberately omits ``output``/``dy``/``fc2_out``/``gate``/``up``/
# ``num_experts`` from the native saved; they stay replay-only and are not
# asserted here.


def _inverse_permutation(permutation):
    """Exact inverse of a permutation tensor (scatter, no argsort tie rules)."""
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), dtype=permutation.dtype
    )
    return inverse


def _stable_expert_major_send_order(selected_experts):
    """Independent oracle for the fused dispatch send order (dropless input).

    ``build_routing_plan`` stable-sorts the valid routes by global expert id in
    float32 (``runtime/routing.py:645-668``); this recomputes that order from
    the raw router output without touching any production helper.  The stable
    tie rule (original flat order inside one expert bucket) is what keeps the
    receive-side row order comparable with the replay.
    """
    flat = selected_experts.reshape(-1).to(torch.int64).cpu()
    return torch.argsort(flat.to(torch.float32), stable=True)


def _arrival_slot_receive_order(recv_counts_re, expert_counts_cpu):
    """Independent oracle for the fused dispatch receive permutation.

    The dispatch arrival buffer groups rows by source rank and, inside one
    source, by local expert bucket; the grouped GEMMs need (local expert,
    source) order.  Both layouts move whole (source, expert) blocks, so the
    permutation is a block reshuffle; it is rebuilt here from the independently
    gathered receive-count cube rather than from the production
    ``_moonep_torch_forward._arrival_to_slot_permutation``.  Convention matches
    the replay's ``local_sort_idxs``: ``sorted[i] == arrival[permutation[i]]``.
    """
    counts = recv_counts_re.to(torch.int64).cpu()
    slot_starts = torch.zeros(counts.shape[1] + 1, dtype=torch.int64)
    slot_starts[1:] = expert_counts_cpu.to(torch.int64).cumsum(0)
    source_totals = counts.sum(dim=1)
    source_starts = source_totals.cumsum(dim=0) - source_totals
    within_source = counts.cumsum(dim=1) - counts
    within_slot = counts.cumsum(dim=0) - counts
    arrival_start = (source_starts.unsqueeze(1) + within_source).reshape(-1)
    sorted_start = (slot_starts[:-1].unsqueeze(0) + within_slot).reshape(-1)
    sizes = counts.reshape(-1)
    total = int(sizes.sum())
    permutation = torch.empty(total, dtype=torch.int64)
    if total:
        group_ids = torch.repeat_interleave(torch.arange(sizes.numel()), sizes)
        group_base = sizes.cumsum(dim=0) - sizes
        lanes = torch.arange(total) - group_base[group_ids]
        permutation[sorted_start[group_ids] + lanes] = (
            arrival_start[group_ids] + lanes
        )
    return permutation


def _check_native_saved_tensor(native_saved, replay_saved, key, failures):
    """One integer/index key: present, same dtype/shape, bit-for-bit values."""
    native_value = native_saved.get(key)
    if native_value is None:
        failures.append(f"{key}: missing from the native saved")
        return
    replay_value = replay_saved[key]
    if not isinstance(native_value, torch.Tensor):
        failures.append(
            f"{key}: native value is {type(native_value).__name__}, "
            "expected a tensor"
        )
        return
    if native_value.dtype != replay_value.dtype:
        failures.append(
            f"{key}: dtype native={native_value.dtype} "
            f"replay={replay_value.dtype}"
        )
    if tuple(native_value.shape) != tuple(replay_value.shape):
        failures.append(
            f"{key}: shape native={tuple(native_value.shape)} "
            f"replay={tuple(replay_value.shape)}"
        )
        return
    # The replay keeps the meta_* family on the host while the native path
    # keeps device tensors; compare both on the CPU.
    native_cpu = native_value.detach().cpu()
    replay_cpu = replay_value.detach().cpu()
    if not torch.equal(native_cpu, replay_cpu):
        mismatch = native_cpu != replay_cpu
        failures.append(
            f"{key}: {int(mismatch.sum().item())}/{mismatch.numel()} "
            "elements differ"
        )


def _check_native_saved_float_tensor(
    native_saved, replay_saved, key, rtol, atol, failures
):
    """One float activation key: present, same dtype/shape, within tolerance."""
    native_value = native_saved.get(key)
    if native_value is None:
        failures.append(
            f"{key}: missing from the native saved (H2 activation capture)"
        )
        return
    replay_value = replay_saved[key]
    if not isinstance(native_value, torch.Tensor):
        failures.append(
            f"{key}: native value is {type(native_value).__name__}, "
            "expected a tensor"
        )
        return
    if native_value.dtype != replay_value.dtype:
        failures.append(
            f"{key}: dtype native={native_value.dtype} "
            f"replay={replay_value.dtype}"
        )
    if tuple(native_value.shape) != tuple(replay_value.shape):
        failures.append(
            f"{key}: shape native={tuple(native_value.shape)} "
            f"replay={tuple(replay_value.shape)}"
        )
        return
    try:
        assert_close(native_value, replay_value, rtol=rtol, atol=atol)
    except AssertionError:
        failures.append(
            f"{key}: {diagnose(native_value, replay_value)} exceeds "
            f"rtol={rtol} atol={atol}"
        )


def _check_native_saved_weight_view(
    native_saved, replay_saved, key, forward_input, input_name, failures, notes
):
    """One zero-copy weight-reference key: values, dtype, shape, and aliasing.

    The native key is a view of the forward input (stride views are legal), so
    contiguity is never asserted and only reported.  Storage aliasing of the
    forward input is a hard gate (§3.1B zero-copy contract; §3.5's memory
    saving is a core promise of the merge — a silently materialized
    ``fc1_combined`` alone costs ~4.6 GiB).
    """
    native_value = native_saved.get(key)
    if native_value is None:
        failures.append(
            f"{key}: missing from the native saved (H2 weight reference)"
        )
        return
    replay_value = replay_saved[key]
    if not isinstance(native_value, torch.Tensor):
        failures.append(
            f"{key}: native value is {type(native_value).__name__}, "
            "expected a tensor"
        )
        return
    if native_value.dtype != replay_value.dtype:
        failures.append(
            f"{key}: dtype native={native_value.dtype} "
            f"replay={replay_value.dtype}"
        )
    if tuple(native_value.shape) != tuple(replay_value.shape):
        failures.append(
            f"{key}: shape native={tuple(native_value.shape)} "
            f"replay={tuple(replay_value.shape)}"
        )
        return
    native_cpu = native_value.detach().cpu()
    replay_cpu = replay_value.detach().cpu()
    if not torch.equal(native_cpu, replay_cpu):
        mismatch = native_cpu != replay_cpu
        failures.append(
            f"{key}: weight reference differs from the replay weights "
            f"({int(mismatch.sum().item())}/{mismatch.numel()} elements)"
        )
        return
    if forward_input is None:
        notes.append(
            f"{key}: no forward input passed in; storage aliasing unchecked"
        )
        return
    if (
        native_value.untyped_storage().data_ptr()
        != forward_input.untyped_storage().data_ptr()
    ):
        failures.append(
            f"{key}: materialized weight reference violates the zero-copy "
            f"contract (§3.1B) — it must alias the {input_name} forward "
            "input storage"
        )
        return
    notes.append(
        f"{key}: zero-copy view of {input_name} "
        f"(contiguous={native_value.is_contiguous()})"
    )


def _check_native_saved_scalar(native_saved, replay_saved, key, failures):
    """One scalar / split-list key: present and exactly equal."""
    native_value = native_saved.get(key)
    if native_value is None:
        failures.append(f"{key}: missing from the native saved")
        return
    replay_value = replay_saved[key]
    if isinstance(replay_value, list) and isinstance(native_value, torch.Tensor):
        native_value = native_value.detach().cpu().tolist()
    elif isinstance(native_value, torch.Tensor):
        if native_value.numel() != 1:
            failures.append(
                f"{key}: native scalar is a {tuple(native_value.shape)} tensor"
            )
            return
        native_value = native_value.item()
    if native_value != replay_value:
        failures.append(
            f"{key}: native={native_value!r} replay={replay_value!r}"
        )


def _check_native_saved_permutation_pair(
    native_saved,
    replay_saved,
    key,
    inverse_key,
    expected_fused,
    failures,
    conventions,
):
    """One permutation pair against both legal dispatch conventions.

    ``sort_idxs``/``local_sort_idxs`` must be a valid permutation equal to
    either the replay layout (rank-major send, flat arrival) or the fused
    dispatch layout ((destination, expert) send, bucket arrival), and the
    matching ``inv_*`` key must be its exact inverse.  A value matching
    neither layout, or the two pairs disagreeing on the layout, is a real
    backward-breaking mismatch.
    """
    native_value = native_saved.get(key)
    native_inverse = native_saved.get(inverse_key)
    if native_value is None:
        failures.append(f"{key}: missing from the native saved")
        return
    if native_inverse is None:
        failures.append(f"{inverse_key}: missing from the native saved")
        return
    if not isinstance(native_value, torch.Tensor) or not isinstance(
        native_inverse, torch.Tensor
    ):
        failures.append(f"{key}/{inverse_key}: expected tensors")
        return
    if native_value.dtype != replay_saved[key].dtype:
        failures.append(
            f"{key}: dtype native={native_value.dtype} "
            f"replay={replay_saved[key].dtype}"
        )
    if tuple(native_value.shape) != tuple(replay_saved[key].shape):
        failures.append(
            f"{key}: shape native={tuple(native_value.shape)} "
            f"replay={tuple(replay_saved[key].shape)}"
        )
        return
    native_cpu = native_value.detach().to(torch.int64).cpu()
    native_inverse_cpu = native_inverse.detach().to(torch.int64).cpu()
    if not torch.equal(
        torch.sort(native_cpu).values,
        torch.arange(native_cpu.numel(), dtype=torch.int64),
    ):
        failures.append(
            f"{key}: not a permutation of range({native_cpu.numel()})"
        )
        return
    replay_cpu = replay_saved[key].detach().to(torch.int64).cpu()
    expected_cpu = expected_fused.to(torch.int64).cpu()
    if torch.equal(native_cpu, replay_cpu):
        conventions[key] = "replay"
        reference_inverse = (
            replay_saved[inverse_key].detach().to(torch.int64).cpu()
        )
        convention_note = "matches the replay layout bit-for-bit"
    elif torch.equal(native_cpu, expected_cpu):
        conventions[key] = "fused"
        reference_inverse = _inverse_permutation(expected_cpu)
        convention_note = (
            "matches the fused (destination, expert) dispatch layout "
            "bit-for-bit; it legitimately differs from the replay rank-major "
            "order"
        )
    else:
        failures.append(
            f"{key}: matches neither the replay rank-major permutation nor "
            "the fused (destination, expert) send order ("
            f"{int((native_cpu != expected_cpu).sum())} of "
            f"{native_cpu.numel()} positions differ from the latter)"
        )
        return
    if not torch.equal(native_inverse_cpu, reference_inverse):
        failures.append(
            f"{inverse_key}: is not the inverse of the {key} order "
            f"({convention_note})"
        )
    return convention_note


def _check_native_saved_plan_snapshots(
    native_saved, expert_counts_cpu, failures, notes
):
    """Soft-check the §3.1 C plan snapshots (native-only, no replay twin).

    The key spelling is not fixed by §3.1 C (MoERoutingPlan field names vs the
    moonep ``plan_*`` precedent in ``_moonep_torch_forward``), so whichever
    spelling is present gets the conservation invariants the backward relies
    on; absence is only reported, because the home backward re-derives these
    tables from ``selected_experts``.
    """
    receive_cube = next(
        (
            native_saved[name]
            for name in (
                "receive_counts_by_source_expert",
                "plan_recv_counts_by_source_expert",
            )
            if name in native_saved
        ),
        None,
    )
    if receive_cube is not None:
        received = receive_cube.detach().to(torch.int64).cpu().sum(dim=0)
        if not torch.equal(received, expert_counts_cpu):
            failures.append(
                "plan receive counts disagree with expert_counts (checked as "
                "receive_counts_by_source_expert / "
                "plan_recv_counts_by_source_expert)"
            )
    else:
        notes.append("no receive-count plan snapshot key found")

    expert_offsets = next(
        (
            native_saved[name]
            for name in (
                "received_expert_offsets",
                "plan_received_expert_offsets",
            )
            if name in native_saved
        ),
        None,
    )
    if expert_offsets is not None:
        offsets = expert_offsets.detach().to(torch.int64).cpu()
        expected_offsets = torch.zeros_like(offsets)
        expected_offsets[1:] = expert_counts_cpu.cumsum(0)
        if not torch.equal(offsets, expected_offsets):
            failures.append(
                "received_expert_offsets is not the cumsum of expert_counts"
            )
    else:
        notes.append("no received-expert-offsets plan snapshot key found")

    send_cube = next(
        (
            native_saved[name]
            for name in (
                "send_counts_by_rank_expert",
                "plan_send_counts_by_rank_expert",
            )
            if name in native_saved
        ),
        None,
    )
    if send_cube is not None:
        sent = send_cube.detach().to(torch.int64).cpu().sum(dim=1)
        expected_sent = torch.tensor(
            native_saved.get("splits_send_list", []), dtype=torch.int64
        )
        if tuple(sent.shape) != tuple(expected_sent.shape) or not torch.equal(
            sent, expected_sent
        ):
            failures.append(
                "send_counts_by_rank_expert per-destination totals disagree "
                "with splits_send_list"
            )
    else:
        notes.append("no send-count plan snapshot key found")


def _assert_native_saved_metadata(
    native_saved,
    replay_saved,
    expert_indices,
    recv_counts_re,
    rank,
    world_size,
    label,
    gate_up_weight=None,
    down_weight=None,
):
    """H1+H2 verdict: every native-saved key against the replay, key by key."""
    failures = []
    notes = []
    for key in NATIVE_SAVED_BITWISE_KEYS:
        _check_native_saved_tensor(native_saved, replay_saved, key, failures)
    for key in NATIVE_SAVED_SCALAR_KEYS:
        _check_native_saved_scalar(native_saved, replay_saved, key, failures)

    # ---- H2 activations.  ``recv_hidden_sorted`` is a pure dispatch copy in
    # the same receive row order on both paths, so any deviation is a major
    # signal (the tag marks it; the assertion itself is never relaxed).
    for key in NATIVE_SAVED_H2_BITWISE_KEYS:
        h2_bitwise_failures = []
        _check_native_saved_tensor(
            native_saved, replay_saved, key, h2_bitwise_failures
        )
        for message in h2_bitwise_failures:
            failures.append(
                f"{message}  [MAJOR: {key} is a pure dispatch copy and must "
                "match the replay bit-for-bit — report to the orchestrator, "
                "do not relax the assertion]"
            )
    for key in NATIVE_SAVED_H2_FLOAT_KEYS:
        _check_native_saved_float_tensor(
            native_saved,
            replay_saved,
            key,
            NATIVE_SAVED_H2_RTOL,
            NATIVE_SAVED_H2_ATOL,
            failures,
        )

    # ---- H2 zero-copy weight references.  fc1_1/fc1_2/fc1_combined alias the
    # packed gate/up input, fc2 the down weight; both sides of this harness
    # are built from the same weight tensors, so the values must be identical.
    h2_view_inputs = {
        "fc1_1": (gate_up_weight, "gate_up_weight"),
        "fc1_2": (gate_up_weight, "gate_up_weight"),
        "fc1_combined": (gate_up_weight, "gate_up_weight"),
        "fc2": (down_weight, "down_weight"),
    }
    for key in NATIVE_SAVED_H2_VIEW_KEYS:
        forward_input, input_name = h2_view_inputs[key]
        _check_native_saved_weight_view(
            native_saved,
            replay_saved,
            key,
            forward_input,
            input_name,
            failures,
            notes,
        )

    native_group = native_saved.get("ep_group")
    if native_group is None:
        failures.append("ep_group: missing from the native saved")
    else:
        try:
            group_rank = dist.get_rank(native_group)
            group_world = dist.get_world_size(native_group)
        except (RuntimeError, ValueError) as exc:
            failures.append(f"ep_group: unusable process group ({exc})")
        else:
            if group_rank != rank or group_world != world_size:
                failures.append(
                    "ep_group: native rank/world "
                    f"({group_rank}/{group_world}) != worker "
                    f"({rank}/{world_size})"
                )

    # Route conservation on the native scalars (mirrors the saved_phys
    # invariants in _assert_physical_saved_layout).
    if native_saved.get("M") != native_saved.get("total_recv"):
        failures.append(
            f"M={native_saved.get('M')!r} must equal "
            f"total_recv={native_saved.get('total_recv')!r}"
        )
    expected_routes = replay_saved["batch_size"] * replay_saved["topk"]
    for split_name, total_name in (
        ("splits_send_list", "total_send"),
        ("splits_recv_list", "total_recv"),
    ):
        splits = native_saved.get(split_name)
        if isinstance(splits, list) and sum(splits) != native_saved.get(
            total_name
        ):
            failures.append(
                f"sum({split_name})={sum(splits)} must equal "
                f"{total_name}={native_saved.get(total_name)!r}"
            )
    if native_saved.get("total_send") not in (None, expected_routes):
        failures.append(
            f"total_send={native_saved.get('total_send')!r} must conserve all "
            f"{expected_routes} dropless routes"
        )

    expert_counts_cpu = replay_saved["expert_counts"].to(
        torch.int64
    ).cpu()
    expected_fused = {
        "sort_idxs": _stable_expert_major_send_order(expert_indices),
        "local_sort_idxs": _arrival_slot_receive_order(
            recv_counts_re, expert_counts_cpu
        ),
    }
    conventions = {}
    for key, inverse_key in NATIVE_SAVED_PERMUTATION_KEYS:
        convention_note = _check_native_saved_permutation_pair(
            native_saved,
            replay_saved,
            key,
            inverse_key,
            expected_fused[key],
            failures,
            conventions,
        )
        if convention_note is not None:
            notes.append(f"{key}: {convention_note}")
    if len(set(conventions.values())) > 1:
        failures.append(
            "sort_idxs and local_sort_idxs disagree on the dispatch layout "
            f"({conventions}); the backward requires one layout"
        )

    _check_native_saved_plan_snapshots(
        native_saved, expert_counts_cpu, failures, notes
    )

    ok = not failures
    if not ok:
        print(f"[rank {rank}] {label}: {'; '.join(failures)}", flush=True)
    if rank == 0:
        suffix = "" if ok else "  |  " + "; ".join(failures)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}", flush=True)
        for note in notes:
            print(f"[NOTE] {label}: {note}", flush=True)
    return ok


def run_megamoe_native_saved_metadata_case(
    rank: int, world_size: int
) -> None:
    """H1+H2: home native saved vs the torch replay, key by key (w2/w4).

    The same inputs drive the fused forward with ``return_saved=True`` and the
    ``_torch_forward.moe_forward`` replay; every native-saved key of the
    39-key replay contract must agree — bit-for-bit wherever the dispatch
    layout permits it (counts, grouped-GEMM metadata, scalars, splits,
    ``recv_hidden_sorted``, and the weight references), the documented
    convention check for the send/receive permutations, and the plan's 2e-2
    tolerance for the computed activations.
    """
    if world_size not in (2, 4):
        raise ValueError(
            "the native saved metadata case requires two or four ranks"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("native saved metadata requires NPU and ACLSHMEM")

    # Shape family of the registry case functional-bwd-small-h512-f256-k4
    # (H=512, F=256, K=4, E=128) at the two smallest worlds; the receive
    # capacity mirrors the functional forward smoke cases.
    tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 128
    experts_per_rank = num_experts // world_size
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"megamoe-native-saved-metadata-w{world_size}"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=1)
    ):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=float(world_size),
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device
            )
            hs, expert_indices = prepare_inputs(
                tokens,
                hidden,
                num_experts,
                topk,
                dtype,
                device,
                seed=2003 + rank * 1000,
            )
            routing_weights = make_routing_weights(
                tokens, topk, device, seed=2004 + rank * 1000
            )
            dist.barrier()

            with torch.no_grad():
                native_output, native_saved = op.forward(
                    hs,
                    expert_indices,
                    packed_w1,
                    w2,
                    routing_weights,
                    return_saved=True,
                )
            dist.barrier()
            with torch.no_grad():
                _, replay_saved = moe_forward(
                    hs,
                    routing_weights,
                    expert_indices,
                    w_gate,
                    w_up,
                    w2,
                    ep_group,
                    topk,
                    return_saved=True,
                )

            # return_saved must be capture-only: the fused output is still the
            # replay output.  Folded into the collective verdict below rather
            # than raised here so a single-rank mismatch cannot hang the
            # following collective.
            output_matches = True
            try:
                assert_close(
                    native_output,
                    replay_saved["output"],
                    rtol=OUTPUT_RTOL,
                    atol=OUTPUT_ATOL,
                )
            except AssertionError:
                output_matches = False

            # Independent receive-count cube for the fused-arrival oracle: my
            # per-(destination, local expert) send counts gathered from every
            # source, reindexed as [source, local expert] on this rank.
            send_counts_re = torch.bincount(
                expert_indices.reshape(-1).to(torch.int64),
                minlength=world_size * experts_per_rank,
            ).to(torch.int32)
            all_send = torch.stack(
                all_gather_list(send_counts_re, ep_group)
            ).reshape(world_size, world_size, experts_per_rank)
            recv_counts_re = all_send[:, rank, :].to(torch.int64).cpu()
            if int(recv_counts_re.sum().item()) != replay_saved["total_recv"]:
                raise AssertionError(
                    "the independent receive-count cube does not conserve "
                    f"total_recv={replay_saved['total_recv']}"
                )

            all_passed = _assert_native_saved_metadata(
                native_saved,
                replay_saved,
                expert_indices,
                recv_counts_re,
                rank,
                world_size,
                label,
                gate_up_weight=packed_w1,
                down_weight=w2,
            )
            if not output_matches:
                print(
                    f"[rank {rank}] {label}: return_saved changed the forward "
                    "output",
                    flush=True,
                )
            all_passed = all_passed and output_matches
            flag = torch.tensor(
                [1 if all_passed else 0], dtype=torch.int32, device=device
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
            if not bool(flag.item()):
                raise AssertionError(
                    f"native saved metadata mismatched the replay: {label}"
                )
        finally:
            op.finalize()


def _megamoe_function_grads(leaves, ffn_dim):
    """Map the Function's autograd grads onto the five canonical check keys.

    ``grad_gate_up`` arrives in the Kimi ``[E, H, 2F]`` layout
    (``cat(g1, g2, dim=1).transpose(1, 2)``); the canonical keys expect the
    replay's ``[E, F, H]`` halves.
    """
    hidden_leaf, routing_leaf, gate_up_leaf, down_leaf = leaves
    grad_gate_up = gate_up_leaf.grad
    return dict(
        grad_hidden=hidden_leaf.grad,
        grad_routing_weights=routing_leaf.grad,
        grad_fc1_1=grad_gate_up[:, :, :ffn_dim].transpose(1, 2),
        grad_fc1_2=grad_gate_up[:, :, ffn_dim:].transpose(1, 2),
        grad_fc2=down_leaf.grad,
    )


def run_megamoe_native_autograd_case(rank: int, world_size: int) -> None:
    """H3: the merged native-saved autograd Function against the eager golden.

    ``MegaMoEFunction`` (§3.2.3) runs the fused forward with
    ``return_saved=True`` and the 5-op triton backward inside one
    ``torch.autograd.Function``:

    * forward ``(ctx, op, hidden_states, routing_weights, selected_experts,
      gate_up_weight, down_weight, peer_mem, state)`` -> ``output``;
    * backward returns the 8-tuple in forward-argument order with
      ``grad_gate_up`` already merged back to the Kimi ``[E, H, 2F]`` layout.

    Verified: the four ``.grad`` tensors against the hand-written eager
    backward golden; two consecutive backwards through one persistent
    ``state`` reproduce every gradient bit-for-bit (signal_mem/epoch reuse);
    and a plain forward after the backwards still matches the independent
    forward golden.
    """
    if world_size != 2:
        raise ValueError("the native autograd case requires exactly two ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("native autograd requires NPU and ACLSHMEM")
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")

    tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 128
    device = f"npu:{rank}"
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "megamoe-native-autograd-w2"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must stay the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below claims its own heap
        # objects for planning and dispatch.  Rows must cover the worst-case
        # per-rank receive: with receive_capacity_factor == world_size the
        # fused plan is dropless, so a rank may receive up to every route in
        # the world (tokens*topk*world_size), not just its own send share.
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk * world_size, tokens * topk, hidden, dtype, rank,
            ep_group,
        )
        try:
            op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=float(world_size),
                ),
            )
            try:
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = make_down_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                hs, expert_indices = prepare_inputs(
                    tokens,
                    hidden,
                    num_experts,
                    topk,
                    dtype,
                    device,
                    seed=2103 + rank,
                )
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=2104 + rank
                )
                torch.manual_seed(2105 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                # Eager golden: the replay saved plus the hand-written 5-op
                # backward baseline, on the plain (non-autograd) tensors.
                with torch.no_grad():
                    _, golden_saved = moe_forward(
                        hs,
                        routing_weights,
                        expert_indices,
                        w_gate,
                        w_up,
                        w2,
                        ep_group,
                        topk,
                        return_saved=True,
                    )
                    golden = backward_torch_baseline(golden_saved, dy)

                # §3.2.3/§3.4: signal_mem/epoch live in a caller-owned state
                # object, injected at backward entry and written back for
                # cross-step reuse.
                state = SimpleNamespace(signal_mem=None, epoch=0)

                def run_function_step():
                    hidden_leaf = hs.clone().requires_grad_(True)
                    routing_leaf = routing_weights.clone().requires_grad_(True)
                    gate_up_leaf = packed_w1.clone().requires_grad_(True)
                    down_leaf = w2.clone().requires_grad_(True)
                    output = MegaMoEFunction.apply(
                        op,
                        hidden_leaf,
                        routing_leaf,
                        expert_indices,
                        gate_up_leaf,
                        down_leaf,
                        peer_mem,
                        state,
                    )
                    output.backward(dy)
                    return hidden_leaf, routing_leaf, gate_up_leaf, down_leaf

                dist.barrier()
                first_grads = _megamoe_function_grads(
                    run_function_step(), ffn
                )
                for name, value in first_grads.items():
                    if value is None or tuple(value.shape) != tuple(
                        golden[name].shape
                    ):
                        raise AssertionError(
                            f"{label}: grad {name} shape "
                            f"{None if value is None else tuple(value.shape)} "
                            f"!= golden {tuple(golden[name].shape)}"
                        )
                all_ok, details = compare_backward_gradients(
                    first_grads, golden
                )
                epoch_after_first = state.epoch
                if state.signal_mem is None:
                    raise AssertionError(
                        f"{label}: the Function did not persist signal_mem in "
                        "the caller state"
                    )

                # Epoch reuse: the second backward through the same persistent
                # state must reproduce every gradient bit-for-bit.
                second_grads = _megamoe_function_grads(
                    run_function_step(), ffn
                )
                for name, again in second_grads.items():
                    if not torch.equal(again, first_grads[name]):
                        raise AssertionError(
                            f"{label}: epoch reuse changed grad {name}"
                        )
                if state.epoch < epoch_after_first or state.epoch < 1:
                    raise AssertionError(
                        f"{label}: the Function did not advance/write back the "
                        f"epoch (after first={epoch_after_first}, "
                        f"after second={state.epoch})"
                    )

                # Backward then forward: the plain forward must still match the
                # independent golden (mirrors _assert_forward_after_grad_transport).
                expected = torch_moe_fwd_golden(
                    hs,
                    routing_weights,
                    expert_indices,
                    w_gate,
                    w_up,
                    w2,
                    num_experts,
                    ep_group,
                )
                assert_close(
                    op.forward(
                        hs, expert_indices, packed_w1, w2, routing_weights
                    ),
                    expected,
                    rtol=OUTPUT_RTOL,
                    atol=OUTPUT_ATOL,
                )

                flag = torch.tensor(
                    [1 if all_ok else 0], dtype=torch.int32, device=device
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
                if rank == 0 and not bool(flag.item()):
                    print(f"{label} gradient details: {details}", flush=True)
                if not bool(flag.item()):
                    raise AssertionError(
                        f"{label}: Function grads mismatched the eager golden"
                    )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


FUNCTIONAL_FORWARD_CASES = kit.make_pytest_params(
    select_cases(direction="forward", tags={"functional", "smoke"})
)
FUNCTIONAL_BACKWARD_CASES = kit.make_pytest_params(
    select_cases(direction="backward", tags={"functional", "smoke"})
)


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("case", FUNCTIONAL_FORWARD_CASES)
def test_forward_suite(dist_test, case: CaseSpec):
    dist_test(run_forward_case, world_size=case.world_size, args=(case,))


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_hot_expert_forward(dist_test, world_size):
    dist_test(
        run_moonep_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
def test_moonep_planning_matches_torch_at_e896_w8(dist_test):
    dist_test(run_moonep_planning_oracle_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_moonep_moderate_wide_forward_w8(dist_test):
    dist_test(run_moonep_moderate_wide_forward_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_physical_forward_hot_expert(dist_test, world_size):
    dist_test(
        run_moonep_physical_forward_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
def test_moonep_physical_forward_moderate_wide_w8(dist_test):
    dist_test(
        run_moonep_physical_forward_moderate_wide_case,
        world_size=8,
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_backward_hot_expert(dist_test, world_size):
    dist_test(
        run_moonep_backward_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
def test_moonep_backward_moderate_wide_w8(dist_test):
    dist_test(
        run_moonep_backward_moderate_wide_case,
        world_size=8,
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_backward_symmetric_hot_expert(dist_test, world_size):
    dist_test(
        run_moonep_backward_symmetric_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
def test_moonep_backward_symmetric_moderate_wide_w8(dist_test):
    dist_test(
        run_moonep_backward_symmetric_moderate_wide_case,
        world_size=8,
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("case", FUNCTIONAL_BACKWARD_CASES)
def test_backward_suite(dist_test, case: CaseSpec):
    dist_test(run_backward_case, world_size=case.world_size, args=(case,))


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_megamoe_native_saved_metadata(dist_test, world_size):
    dist_test(
        run_megamoe_native_saved_metadata_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
def test_megamoe_native_autograd(dist_test):
    if MegaMoEFunction is None:
        pytest.skip(
            "H3 API pending: mega_moe does not export MegaMoEFunction yet"
        )
    dist_test(run_megamoe_native_autograd_case, world_size=2)


# TODO: future work — when an all-directions session is introduced, finish and
# fully clean forward before starting backward; do not reuse operator, peer
# memory, or ACLSHMEM heap.  Each parameterized node currently owns an isolated
# lifecycle.
