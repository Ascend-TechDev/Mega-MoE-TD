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
import mega_moe.kernels.fc2_combine as fc2_combine_module
from mega_moe.kernels.moonep_planning import launch_moonep_b0_b3
from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)
from mega_moe.runtime.replica_weight_prefetch import (
    replica_weight_push_geometry,
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
)
from tests._numeric import OUTPUT_ATOL, OUTPUT_RTOL


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


def test_replica_weight_udma_geometry_uses_one_kimi_request_per_panel():
    gate_up_panel_elements = 3584 * 3072
    down_panel_elements = (3584 // 2) * 3072
    chunk_bytes = 64 * 1024 * 1024

    assert replica_weight_push_geometry(
        gate_up_panel_elements, chunk_bytes=chunk_bytes
    )[1] == 1
    assert replica_weight_push_geometry(
        down_panel_elements, chunk_bytes=chunk_bytes
    )[1] == 1
    with pytest.raises(ValueError, match="UDMA 256 MiB"):
        replica_weight_push_geometry(
            gate_up_panel_elements,
            chunk_bytes=256 * 1024 * 1024 + 2,
        )


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

            # Exercise asymmetric routing, all-drop, and epoch reuse on the
            # representative small case.
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

            # Routing metadata is backed by a single shared workspace.  Build
            # the same plan twice; the first plan's generation is stale and
            # must be rejected before a kernel launch.
            superseded_primary_plan = op.build_routing_plan(expert_indices)
            current_primary_plan = op.build_routing_plan(expert_indices)
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
            mutation_plan = op.build_routing_plan(mutated_expert_indices)
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
            repeat_epoch = op._replica_weight_epoch
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
                "moonep-hot-expert-repeat-refill",
                rank,
                device,
            )
            if (
                not passed
                or op._replica_weight_epoch != repeat_epoch + 1
                or op._replica_prefetch_pending
            ):
                raise AssertionError("MoonEP repeated replica refill failed")
            epoch_before_update = op._replica_weight_epoch
            if rank == 0:
                # Change one owner's hot expert in place. Every rank must still
                # publish and consume the next prefetch epoch.
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

    tokens, hidden, ffn, topk, num_experts = 64, 256, 256, 16, 896
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
            if copy_counts != [0, 0, 18, 18, 4, 4, 4, 4]:
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
@pytest.mark.parametrize("case", FUNCTIONAL_BACKWARD_CASES)
def test_backward_suite(dist_test, case: CaseSpec):
    dist_test(run_backward_case, world_size=case.world_size, args=(case,))
