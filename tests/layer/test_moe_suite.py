# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven distributed forward/backward functional suites.

Every pytest node represents one concrete ``CaseSpec``.  Shape and token
selection happen at collection time; workers never inspect shape-selection
environment variables.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from triton.backends.ascend.driver import NPUUtils

from config import CaseSpec, select_cases
from mega_moe import FusedMoEForward, MoEForwardConfig, moe_backward_triton, pack_gate_up_weights
import mega_moe.kernels.fc2_combine as fc2_combine_module
from mega_moe.kernels.weighted_swiglu import weighted_swiglu_forward
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
    run_weighted_one,
)
from tests._numeric import OUTPUT_ATOL, OUTPUT_RTOL


def _tiling_overrides() -> dict[str, int]:
    """Read only runtime tuning knobs; shape selection is case-driven."""
    names = (
        ("MOE_FUSED_DISPATCH_FC1_BLOCK_SIZE_M", "dispatch_fc1_block_size_m"),
        ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_N", "fc1_gemm_block_size_n"),
        ("MOE_FUSED_FC1_GEMM_BLOCK_SIZE_K", "fc1_gemm_block_size_k"),
        ("MOE_FUSED_FC2_COMBINE_BLOCK_SIZE_M", "fc2_combine_block_size_m"),
        ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_N", "fc2_gemm_block_size_n"),
        ("MOE_FUSED_FC2_GEMM_BLOCK_SIZE_K", "fc2_gemm_block_size_k"),
    )
    return {
        parameter: int(os.environ[name])
        for name, parameter in names
        if name in os.environ
    }


def _forward_config(case: CaseSpec) -> MoEForwardConfig:
    return MoEForwardConfig(
        receive_capacity_factor=case.capacity_factor,
        **_tiling_overrides(),
    )


def test_remote_store_workspaces_cover_send_and_receive_capacity(monkeypatch):
    """Size combine rows by sends and descriptors by receive capacity."""
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
    op._fc2_transport_block_m = fc2_combine_module._FC2_TRANSPORT_BLOCK_M
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
    expected_slots = 4 + 8 * 112
    assert op._max_pull_tile_slots == expected_slots
    assert op._pull_tile_rank.shape == (expected_slots,)


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
    reason="SiTU-GLU kernel correctness requires an NPU device",
)
def test_weighted_swiglu_kernel_supports_swiglu_and_situglu():
    device = "npu:0"
    torch.npu.set_device(0)
    torch.manual_seed(0)
    rows, ffn_dim = 64, 256
    fc1 = (
        (torch.randn(rows, 2 * ffn_dim, dtype=torch.float32) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    routing_weights = (
        torch.rand(rows, dtype=torch.float32, device=device) + 0.1
    ).contiguous()
    cases = (
        ("swiglu", 1.0, None),
        ("situglu", 1.0, None),
        ("situglu", 1.5, None),
        ("situglu", 2.0, 1.0),
    )
    num_vector_programs = NPUUtils().get_aivector_core_num()
    for activation, beta, linear_beta in cases:
        actual = weighted_swiglu_forward(
            fc1,
            routing_weights,
            num_vector_programs,
            activation=activation,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        expected = _situglu_torch_ref(
            fc1, routing_weights, activation, beta, linear_beta
        )
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )

    empty = weighted_swiglu_forward(
        fc1[:0],
        routing_weights[:0],
        num_vector_programs,
        activation="situglu",
        situ_beta=1.0,
    )
    assert empty.shape == (0, ffn_dim)
    assert empty.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="activation must be 'swiglu' or 'situglu'"):
        weighted_swiglu_forward(
            fc1, routing_weights, num_vector_programs, activation="relu"
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
            all_passed &= run_weighted_one(
                op, hs, expert_indices, routing_weights,
                w_gate, w_up, packed_w1, case.num_experts,
                f"{case.case_id}-weighted", rank, device, dtype,
            )

            if case.model in {"S", "S-drop"}:
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
                all_passed &= run_weighted_one(
                    op, hs, skew, edge_weights,
                    w_gate, w_up, packed_w1, case.num_experts,
                    f"{case.case_id}-zero-receive-weighted", rank, device, dtype,
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
                all_passed &= run_weighted_one(
                    op, hs, all_drop, routing_weights,
                    w_gate, w_up, packed_w1, case.num_experts,
                    f"{case.case_id}-all-drop-weighted", rank, device, dtype,
                )
                all_passed &= run_full_one(
                    op, hs, all_drop, routing_weights,
                    w_gate, w_up, packed_w1, w2, case.num_experts,
                    f"{case.case_id}-all-drop-full", rank, device,
                )
                all_passed &= run_one(
                    op, hs, expert_indices, packed_w1, case.num_experts,
                    f"{case.case_id}-epoch-reuse", rank, device, dtype,
                )
        finally:
            op.finalize()

        flag = torch.tensor([1 if all_passed else 0], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
        if not bool(flag.item()):
            raise AssertionError(f"functional forward case failed: {case.case_id}")


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
@pytest.mark.parametrize("case", FUNCTIONAL_BACKWARD_CASES)
def test_backward_suite(dist_test, case: CaseSpec):
    dist_test(run_backward_case, world_size=case.world_size, args=(case,))


# TODO: future work — when an all-directions session is introduced, finish and
# fully clean forward before starting backward; do not reuse operator, peer
# memory, or ACLSHMEM heap.  Each parameterized node currently owns an isolated
# lifecycle.
