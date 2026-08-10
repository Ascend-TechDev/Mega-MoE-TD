# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven distributed forward/backward functional suites.

Every pytest node represents one concrete ``CaseSpec``.  Shape and token
selection happen at collection time; workers never inspect shape-selection
environment variables.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from config import CaseSpec, select_cases
from mega_moe import FusedMoEForward, MoEForwardConfig, moe_backward_triton, pack_gate_up_weights
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
        num_aicore_programs=int(os.environ.get("MOE_FUSED_NUM_AICORE_PROGRAMS", "24")),
        receive_capacity_factor=case.capacity_factor,
        **_tiling_overrides(),
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
