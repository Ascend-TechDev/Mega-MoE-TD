# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Data-driven distributed forward/backward functional suites.

Every pytest node represents one concrete ``CaseSpec``.  Shape and token
selection happen at collection time; workers never inspect shape-selection
environment variables.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
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
from mega_moe.runtime.device import device_str, resolve_local_device
import mega_moe.kernels.fc2_combine as fc2_combine_module
import mega_moe.kernels.fused_forward as fused_forward_module
from benchmark.layer import _fwd_phase_timing as fwd_timing_table
from mega_moe.kernels.moonep_planning import launch_moonep_b0_b3
from mega_moe.runtime.moonep_planning import (
    build_inverse_experts_to_copy,
    plan_moonep_b0_b3,
)
from mega_moe.runtime.replica_weight_prefetch import replica_pool_snapshot
from benchmark.layer import bench_moe_suite as bench_module
from mega_moe.kernels.weighted_swiglu import (
    _BLOCK_M as _WEIGHTED_BLOCK_M,
    _BLOCK_N as _WEIGHTED_BLOCK_N,
    _weighted_activation_expert_group_kernel,
)
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    _compare_full_output,
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
    cmp_grad,
    diagnose,
)

# Stage H3 (master plan §3.2.3) adds the merged native-saved autograd Function
# and exports it from ``mega_moe.ops``.  Collection must stay working on a tree
# without it, so the import degrades to ``None`` and the case skips.
try:
    from mega_moe.ops import MegaMoEFunction  # noqa: E402
except ImportError:
    try:
        from mega_moe.ops.function import MegaMoEFunction  # noqa: E402
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

    device = device_str(resolve_local_device(rank))
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


def run_single_kernel_forward_case(
    rank: int, world_size: int, fc1_block_m: int = 256, tokens: int = 64,
    num_experts: int = 8, save_fc1_dtype: str = "bf16",
) -> None:
    """Exercise the one-launch home-expert path against the Torch oracle.

    ``save_fc1_dtype`` selects the saved-FC1 format asserted in the
    return_saved sub-case (the "bf16" raw default, "fp8" E4M3 + scales, or
    the "fp16" plain-copy A/B branch); the fp8 opt-in is additionally
    exercised with a halved M tile through its own operator below."""
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("single-kernel forward requires NPU and ACLSHMEM")

    hidden, ffn, topk = 256, 512, 2
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    all_passed = True

    with kit.aclshmem_session(
        rank,
        world_size,
        kit.get_ash_size_bytes(default_gb=1),
    ):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=float(world_size),
                enable_single_kernel_forward=True,
                save_fc1_dtype=save_fc1_dtype,
                fc1_gemm_block_size_m=fc1_block_m,
                fc2_combine_block_size_m=fc1_block_m,
            ),
        )
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts,
                hidden,
                ffn,
                world_size,
                rank,
                dtype,
                device,
            )
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts,
                hidden,
                ffn,
                world_size,
                rank,
                dtype,
                device,
            )
            hs, expert_indices = prepare_inputs(
                tokens,
                hidden,
                num_experts,
                topk,
                dtype,
                device,
                seed=9100 + rank,
                drop_frac=0.25,
            )
            expert_indices[0, 0] = -1
            routing_weights = make_routing_weights(
                tokens,
                topk,
                device,
                seed=9200 + rank,
            )
            dist.barrier(group=ep_group)

            for iteration in range(2):
                all_passed &= run_full_one(
                    op,
                    hs,
                    expert_indices,
                    routing_weights,
                    w_gate,
                    w_up,
                    packed_w1,
                    w2,
                    num_experts,
                    f"single-kernel-forward-w{world_size}-iter{iteration}",
                    rank,
                    device,
                )

            all_to_rank_zero = torch.zeros_like(expert_indices)
            all_passed &= run_full_one(
                op,
                hs,
                all_to_rank_zero,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                f"single-kernel-forward-w{world_size}-zero-receive",
                rank,
                device,
            )

            all_dropped = torch.full_like(expert_indices, num_experts)
            all_passed &= run_full_one(
                op,
                hs,
                all_dropped,
                routing_weights,
                w_gate,
                w_up,
                packed_w1,
                w2,
                num_experts,
                f"single-kernel-forward-w{world_size}-all-dropped",
                rank,
                device,
            )

            # return_saved=True drives the same single launch and returns the
            # redesigned backward contract: the raw FC1 GEMM result plus the
            # dispatch in pre-dispatch form (input references + receive
            # layout tables, no materialized receive clone).  The replay
            # oracle cannot replay dropped routes, so this sub-case uses its
            # own dropless inputs.
            saved_hs, saved_experts = prepare_inputs(
                tokens,
                hidden,
                num_experts,
                topk,
                dtype,
                device,
                seed=7300 + rank,
            )
            saved_weights = make_routing_weights(
                tokens, topk, device, seed=7400 + rank,
            )
            dist.barrier(group=ep_group)
            with torch.no_grad():
                saved_output, saved = op.forward(
                    saved_hs,
                    saved_experts,
                    packed_w1,
                    w2,
                    saved_weights,
                    return_saved=True,
                )
            saved_golden = torch_moe_fwd_golden(
                saved_hs,
                saved_weights,
                saved_experts,
                w_gate,
                w_up,
                w2,
                num_experts,
                ep_group,
            )
            all_passed &= _compare_full_output(
                saved_output,
                saved_golden,
                f"single-kernel-forward-w{world_size}-saved-output",
                rank,
                device,
                ep_group,
            )
            with torch.no_grad():
                _, replay_saved = moe_forward(
                    saved_hs,
                    saved_weights,
                    saved_experts,
                    w_gate,
                    w_up,
                    w2,
                    ep_group,
                    topk,
                    return_saved=True,
                )
            saved_keys = {
                "fc1_output",
                # megaknl_zjg's mega backward recompute consumes the per-row
                # routing weight (ops/backward.py scale_ptr contract), so the
                # single-kernel saved dict carries one key past origin/main.
                "recv_weights_sorted",
                "hidden_states",
                "selected_experts",
                "routing_weights",
                "recv_expert_offsets",
                "recv_counts_by_source_expert",
                "send_token_indices",
                "send_route_indices",
                "total_send",
                "total_recv",
                "M",
                "_routing_generation",
                "_owner_token",
            }
            if save_fc1_dtype == "fp8":
                saved_keys |= {"fc1_output_scale", "fc1_scale_group_size"}
            experts_per_rank = num_experts // world_size
            group_n = op.config.fc1_gemm_block_size_n // 2
            num_groups = (2 * ffn) // group_n
            send_tables_ok = (
                saved["send_token_indices"].shape
                == saved["send_route_indices"].shape
                and int(saved["send_token_indices"].numel())
                == int(saved["total_send"])
                and int(saved["total_send"])
                == int(replay_saved["total_send"])
                and bool(
                    (saved["send_token_indices"] >= 0).all()
                    and (saved["send_token_indices"] < tokens).all()
                )
            )
            recv_layout_ok = (
                tuple(saved["recv_counts_by_source_expert"].shape)
                == (world_size, experts_per_rank)
                and int(saved["recv_counts_by_source_expert"].sum().item())
                == int(saved["total_recv"])
                and int(saved["recv_expert_offsets"].numel())
                == experts_per_rank + 1
                and int(saved["recv_expert_offsets"][-1].item())
                == int(saved["total_recv"])
            )
            saved_ok = (
                set(saved) == saved_keys
                and saved.get("_owner_token") == id(op._routing_owner_token)
                and saved.get("_routing_generation")
                == op._routing_generation
                # Pre-dispatch references are zero-copy views of the inputs.
                and saved["hidden_states"] is saved_hs
                and saved["selected_experts"] is saved_experts
                and saved["routing_weights"] is saved_weights
                and int(saved["total_recv"]) == int(replay_saved["total_recv"])
                and int(saved["M"]) == int(saved["total_recv"])
                and send_tables_ok
                and recv_layout_ok
                and tuple(saved["fc1_output"].shape)
                == tuple(replay_saved["fc1_output"].shape)
            )
            if save_fc1_dtype == "fp8":
                saved_ok &= (
                    # FP8 E4M3 payload + per-row/per-group FP32 scales.
                    saved["fc1_output"].dtype == torch.float8_e4m3fn
                    and saved["fc1_output_scale"].dtype == torch.float32
                    and int(saved["fc1_scale_group_size"]) == group_n
                    and tuple(saved["fc1_output_scale"].shape)
                    == (int(saved["total_recv"]), num_groups)
                )
            elif save_fc1_dtype == "fp16":
                # Plain FP16 copy of the raw gate/up values.
                saved_ok &= saved["fc1_output"].dtype == torch.float16
            else:
                # Default format: raw BF16 gate/up tiles, no scale keys.
                saved_ok &= saved["fc1_output"].dtype == torch.bfloat16
            if saved_ok:
                try:
                    ref = replay_saved["fc1_output"].float()
                    if save_fc1_dtype == "fp8":
                        # Grouped error bound for the E4M3 quantized save:
                        # dequantize with the saved scales and compare
                        # against the replay oracle per column group.  E4M3
                        # round-to-nearest is bounded by 1/16 of the value
                        # (a truncating backend doubles that to 1/8), and
                        # the two GEMM implementations themselves differ by
                        # ~2e-2, so a 0.15 x group-amax bound covers both
                        # with margin.
                        deq = (
                            saved["fc1_output"].float()
                            .view(int(saved["total_recv"]), num_groups,
                                  group_n)
                            * saved["fc1_output_scale"][:, :, None]
                        ).view(ref.shape)
                        group_amax = (
                            ref.view(ref.shape[0], num_groups, group_n)
                            .abs()
                            .amax(dim=-1)
                        )
                        err = (deq - ref).abs().view(
                            ref.shape[0], num_groups, group_n
                        )
                        if not bool((err <= 0.15 * group_amax + 1e-3).all()):
                            raise AssertionError(
                                "dequantized fc1 exceeds the grouped error "
                                f"bound (max {float(err.max().item()):.4f})"
                            )
                        # The scale must be the group amax / 448 (E4M3 max);
                        # near-zero groups keep the kernel's 1/448 sentinel,
                        # and the relative GEMM difference between the two
                        # sides inflates small amaxes, so the bound stays
                        # loose — its job is catching a misaligned scale
                        # table, which would explode the error bound above,
                        # not re-measuring fp8.
                        expected_scale = torch.where(
                            group_amax > 1e-3, group_amax,
                            torch.ones_like(group_amax),
                        ) / 448.0
                        assert_close(
                            saved["fc1_output_scale"],
                            expected_scale,
                            rtol=1e-1,
                            atol=1e-3,
                        )
                    elif save_fc1_dtype == "fp16":
                        # The FP16 copy keeps the raw GEMM values: only the
                        # two implementations' ~2e-2 difference plus fp16
                        # rounding show up.
                        err = (saved["fc1_output"].float() - ref).abs()
                        if not bool(
                            (err <= 0.15 * ref.abs() + 1e-3).all()
                        ):
                            raise AssertionError(
                                "fp16 fc1 exceeds the grouped error bound "
                                f"(max {float(err.max().item()):.4f})"
                            )
                    else:
                        # Default BF16: the raw values — only the two GEMM
                        # implementations' ~2e-2 difference shows up.
                        assert_close(
                            saved["fc1_output"].float(), ref,
                            rtol=NATIVE_SAVED_H2_RTOL,
                            atol=NATIVE_SAVED_H2_ATOL,
                        )
                except AssertionError:
                    saved_ok = False
            if not saved_ok:
                print(
                    f"[rank {rank}] single-kernel-forward-w{world_size}: "
                    "saved contract (fc1 + pre-dispatch tables) mismatched "
                    "the replay oracle",
                    flush=True,
                )
            all_passed &= saved_ok

            # save_fc1_dtype="fp8" opt-in (!59 contract): FP8 E4M3 payload
            # plus per-row/per-group FP32 scales.  The in-kernel quantize leg
            # raises the fused launch's Unified Buffer demand, so this op
            # halves the M tile to stay under budget at this suite shape.
            fp8_op = FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=float(world_size),
                    enable_single_kernel_forward=True,
                    fc1_gemm_block_size_m=min(fc1_block_m, 128),
                    fc2_combine_block_size_m=min(fc1_block_m, 128),
                    save_fc1_dtype="fp8",
                ),
            )
            try:
                dist.barrier(group=ep_group)
                with torch.no_grad():
                    _, fp8_saved = fp8_op.forward(
                        saved_hs,
                        saved_experts,
                        packed_w1,
                        w2,
                        saved_weights,
                        return_saved=True,
                    )
                fp8_keys = saved_keys | {
                    "fc1_output_scale",
                    "fc1_scale_group_size",
                }
                fp8_group_n = fp8_op.config.fc1_gemm_block_size_n // 2
                fp8_num_groups = (2 * ffn) // fp8_group_n
                fp8_ok = (
                    set(fp8_saved) == fp8_keys
                    and fp8_saved["fc1_output"].dtype == torch.float8_e4m3fn
                    and fp8_saved["fc1_output_scale"].dtype == torch.float32
                    and int(fp8_saved["fc1_scale_group_size"]) == fp8_group_n
                    and tuple(fp8_saved["fc1_output_scale"].shape)
                    == (int(fp8_saved["total_recv"]), fp8_num_groups)
                    and int(fp8_saved["total_recv"])
                    == int(replay_saved["total_recv"])
                )
                if fp8_ok:
                    try:
                        # Grouped error bound for the E4M3 quantized save:
                        # dequantize with the saved scales and compare against
                        # the replay oracle per column group.  E4M3 round-to-
                        # nearest is bounded by 1/16 of the value (a
                        # truncating backend doubles that to 1/8), and the two
                        # GEMM implementations themselves differ by ~2e-2, so
                        # a 0.15 x group-amax bound covers both with margin.
                        ref = replay_saved["fc1_output"].float()
                        deq = (
                            fp8_saved["fc1_output"].float()
                            .view(
                                int(fp8_saved["total_recv"]),
                                fp8_num_groups,
                                fp8_group_n,
                            )
                            * fp8_saved["fc1_output_scale"][:, :, None]
                        ).view(ref.shape)
                        group_amax = (
                            ref.view(ref.shape[0], fp8_num_groups, fp8_group_n)
                            .abs()
                            .amax(dim=-1, keepdim=True)
                        )
                        err = (deq - ref).abs().view(
                            ref.shape[0], fp8_num_groups, fp8_group_n
                        )
                        if not bool((err <= 0.15 * group_amax + 1e-3).all()):
                            raise AssertionError(
                                "dequantized fc1 exceeds the grouped error "
                                "bound "
                                f"(max {float(err.max().item()):.4f})"
                            )
                        # The scale must be the group amax / 448 (E4M3 max);
                        # near-zero groups keep the kernel's 1/448 sentinel,
                        # and the relative GEMM difference between the two
                        # sides inflates small amaxes, so the bound stays
                        # loose — its job is catching a misaligned scale
                        # table, which would explode the error bound above,
                        # not re-measuring fp8.
                        expected_scale = torch.where(
                            group_amax > 1e-3, group_amax,
                            torch.ones_like(group_amax),
                        ).squeeze(-1) / 448.0
                        assert_close(
                            fp8_saved["fc1_output_scale"],
                            expected_scale,
                            rtol=1e-1,
                            atol=1e-3,
                        )
                    except AssertionError:
                        fp8_ok = False
                if not fp8_ok:
                    print(
                        f"[rank {rank}] single-kernel-forward-w{world_size}: "
                        "FP8 saved contract mismatched the replay oracle",
                        flush=True,
                    )
                all_passed &= fp8_ok
            finally:
                fp8_op.finalize()

            # MOE_FWD_TIMING=1 smoke: the same launch carries the SYS_CNT
            # phase stamps.  The unsaved run must leave the ring's save
            # column zero (SAVE_FC1=0 compiles the block out); the saved
            # run stamps every checkpoint on a monotone per-program clock.
            previous_timing_env = os.environ.get("MOE_FWD_TIMING")
            os.environ["MOE_FWD_TIMING"] = "1"
            try:
                with torch.no_grad():
                    op.forward(
                        hs, expert_indices, packed_w1, w2, routing_weights)
                torch.npu.synchronize(device)
                # Snapshot immediately: read_last_forward_phase_timing hands
                # out the LIVE shared buffers, and the saved launch below
                # zeroes and rewrites them — a bare reference would let the
                # saved run's save column leak into the plain assertions.
                plain_timing = {
                    key: value.clone()
                    for key, value in
                    op.read_last_forward_phase_timing().items()
                }
                with torch.no_grad():
                    _, timing_saved = op.forward(
                        saved_hs, saved_experts, packed_w1, w2, saved_weights,
                        return_saved=True,
                    )
                torch.npu.synchronize(device)
                saved_timing = {
                    key: value.clone()
                    for key, value in
                    op.read_last_forward_phase_timing().items()
                }

                timing_ok = (
                    fused_forward_module.FWD_TS_SLOTS
                    == fwd_timing_table.FWD_TS_SLOTS
                    and fused_forward_module.FWD_ACC_SLOTS
                    == fwd_timing_table.FWD_ACC_SLOTS
                    and int(fused_forward_module.FWD_RING_COLS)
                    == fwd_timing_table.FWD_RING_COLS
                    and fused_forward_module.fwd_ring_slots(5, 16)
                    == fwd_timing_table.fwd_ring_slots(5, 16)
                )
                for products in (plain_timing, saved_timing):
                    ts = products["ts"]
                    timing_ok &= tuple(ts.shape) == (
                        op.num_aicore_programs, fwd_timing_table.FWD_TS_SLOTS)
                    timing_ok &= bool((ts > 0).all())
                    timing_ok &= bool((ts[:, 1:] >= ts[:, :-1]).all())
                    timing_ok &= tuple(products["acc"].shape) == (
                        op.num_aicore_programs * 3,
                        fwd_timing_table.FWD_ACC_SLOTS)
                    timing_ok &= bool((products["acc"] >= 0).all())
                    timing_ok &= tuple(products["ring"].shape) == (
                        op.num_aicore_programs * 3, op._fwd_ring_slots,
                        fwd_timing_table.FWD_RING_COLS,
                    )
                save_column = int(fused_forward_module.FWD_RING_SAVE)
                timing_ok &= int(
                    plain_timing["ring"][:, :, save_column].abs().sum().item()
                ) == 0
                if int(timing_saved["total_recv"]) > 0:
                    timing_ok &= int(
                        saved_timing["ring"][:, :, save_column].sum().item()
                    ) > 0
                if not timing_ok:
                    print(
                        f"[rank {rank}] single-kernel-forward-w{world_size}: "
                        "MOE_FWD_TIMING products violated the stamp contract",
                        flush=True,
                    )
                all_passed &= timing_ok
            finally:
                if previous_timing_env is None:
                    os.environ.pop("MOE_FWD_TIMING", None)
                else:
                    os.environ["MOE_FWD_TIMING"] = previous_timing_env
        finally:
            op.finalize()

        overflow_op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=tokens,
            hidden_size=hidden,
            top_k=topk,
            num_experts=num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=1.0,
                enable_single_kernel_forward=True,
                fc1_gemm_block_size_m=fc1_block_m,
                fc2_combine_block_size_m=fc1_block_m,
            ),
        )
        try:
            rejected_overflow = False
            try:
                overflow_op.forward(
                    hs,
                    torch.zeros_like(expert_indices),
                    packed_w1,
                    w2,
                    routing_weights,
                )
            except ValueError as exc:
                rejected_overflow = "required receive size" in str(exc)
            all_passed &= rejected_overflow
        finally:
            overflow_op.finalize()

        flag = torch.tensor(
            [1 if all_passed else 0],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
        if not bool(flag.item()):
            raise AssertionError("single-kernel forward mismatched the oracle")


def run_single_kernel_kimi_k3_case(rank: int, world_size: int) -> None:
    """Validate the fused launch at the trimmed Kimi-K3 W8/T4K shape."""
    if world_size != 8:
        raise ValueError("the Kimi-K3 single-kernel case requires eight ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("Kimi-K3 single-kernel forward requires NPU and ACLSHMEM")

    base_case = next(
        case
        for case in select_cases(
            direction="forward",
            tags={"performance", "kimi", "trimmed"},
        )
        if case.world_size == world_size and case.tokens == 4096
    )
    heap_size = kit.get_ash_size_bytes(default_gb=2)
    required_heap = bench_module._required_ash_bytes(base_case, world_size)
    if required_heap >= heap_size:
        raise RuntimeError(
            "Kimi-K3 single-kernel test needs more ACLSHMEM heap: "
            f"required={required_heap}, configured={heap_size}"
        )

    device = device_str(resolve_local_device(rank))
    ep_group = dist.group.WORLD
    with kit.aclshmem_session(rank, world_size, heap_size):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=base_case.tokens,
            hidden_size=base_case.hidden,
            top_k=base_case.topk,
            num_experts=base_case.num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=base_case.capacity_factor,
                enable_single_kernel_forward=True,
                fc1_gemm_block_size_k=256,
                fc2_gemm_block_size_k=256,
            ),
        )
        try:
            experts_per_rank = base_case.num_experts // world_size
            packed_w1, down_weight, _ = bench_module._make_local_weights(
                base_case,
                experts_per_rank,
                rank,
                device,
            )
            for token_count in (64, base_case.tokens):
                case = (
                    base_case
                    if token_count == base_case.tokens
                    else replace(
                        base_case,
                        case_id="performance-fwd-kimi-k3-trimmed-w8-t64",
                        tokens=token_count,
                    ).validate()
                )
                hidden_states, selected_experts, routing_weights = (
                    bench_module._prepare_inputs(case, rank, device)
                )
                dist.barrier(group=ep_group)
                actual = op.forward(
                    hidden_states,
                    selected_experts,
                    packed_w1,
                    down_weight,
                    routing_weights,
                )
                expected = bench_module._logical_torch_golden(
                    case,
                    ep_group,
                    hidden_states,
                    selected_experts,
                    routing_weights,
                    packed_w1,
                    down_weight,
                )
                bench_module._assert_close_collective(
                    actual,
                    expected,
                    device,
                    f"single-kernel-{case.case_id}",
                    ep_group,
                )
                if rank == 0:
                    print(f"[PASS] single-kernel-{case.case_id}", flush=True)
                del actual, expected
                del hidden_states, selected_experts, routing_weights
                torch.npu.empty_cache()
                dist.barrier(group=ep_group)
        finally:
            op.finalize()


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
    device = device_str(resolve_local_device(rank))
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
    device = device_str(resolve_local_device(rank))

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
    device = device_str(resolve_local_device(rank))
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
    device = device_str(resolve_local_device(rank))
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
    device = device_str(resolve_local_device(rank))
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
    # research instrumentation (2026-09-18): the collective MIN reduce means
    # the failing rank's details never print when rank0 is fine — always
    # print on the locally-failing rank.
    if not all_ok:
        print(f"{label} rank{rank} LOCAL-FAIL gradient details: {details}",
              flush=True)
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
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must be the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below allocates its own heap
        # objects to build the routing plan.
        peer_mem = kit.make_moonep_backward_peer_mem(
            # recv side budgets worst-case imbalance: per-rank recv is
            # data-dependent, up to tokens*topk*world_size dropless; send is
            # bounded by the local tokens*topk exactly.
            tokens * topk * world_size, tokens * topk,
            hidden, dtype, rank, ep_group,
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
    device = device_str(resolve_local_device(rank))
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
            # recv side budgets worst-case imbalance: per-rank recv is
            # data-dependent, up to tokens*topk*world_size dropless; send is
            # bounded by the local tokens*topk exactly.
            tokens * topk * world_size, tokens * topk,
            hidden, dtype, rank, ep_group,
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
    # MOE_BWD_MEGA=1 sinks INSIDE the launch (P6a), so no host hook can fire
    # between sink and reduce — `captured_slots` arrives empty and the
    # bit-exact slot compare is skipped; the HCCL-oracle reduction compare
    # and the post-zero check below still cover the whole chain end-to-end.
    if captured_slots:
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

    Pooled-tables mode (MEGAMOE_REPLICA_POOL=1, 2026-09-17): the replica cache
    is inert by design (same-shape layers share the tables, so a weight-key
    "hit" says nothing about the content) and EVERY forward pushes; with
    MOE_MEGA_REPREFETCH=1 the backward's in-kernel re-push also mints from the
    same table-level monotonic counter.  The epoch arithmetic below absorbs
    those mints; the output checks are unchanged and stay the point of the
    helper.
    """
    pooled = os.environ.get("MEGAMOE_REPLICA_POOL") == "1"
    # the backward's in-kernel re-push minted from the shared counter iff the
    # mega path ran with the knob on (default on since 2026-09-22 — mirrors
    # mega_bwd's gate; every caller of this helper has replica traffic, i.e.
    # active_e > home_e, so the wrapper's reprefetch was live)
    bwd_mints = 1 if (
        os.environ.get("MOE_BWD_MEGA") == "1"
        and os.environ.get("MOE_MEGA_REPREFETCH", "1") == "1"
        and getattr(op, "enable_moonep", False)
    ) else 0
    epoch_after_push = epoch_before_lend + 1 + bwd_mints
    reloaded = op.forward(
        hidden_states, expert_indices, packed_w1, w2, routing_weights
    )
    if op._replica_weight_epoch != epoch_after_push:
        raise AssertionError(
            f"{label}: the post-backward forward re-pushed "
            f"{op._replica_weight_epoch - epoch_before_lend - bwd_mints} times, expected 1"
        )
    if not pooled and not op._replica_weight_cache_valid:
        raise AssertionError(
            f"{label}: the post-backward forward left the replica cache invalid"
        )
    assert_close(reloaded, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL)

    cached = op.forward(
        hidden_states, expert_indices, packed_w1, w2, routing_weights
    )
    if not pooled and op._replica_weight_epoch != epoch_after_push:
        raise AssertionError(
            f"{label}: the second post-backward forward was not a cache hit"
        )
    if pooled and op._replica_weight_epoch != epoch_after_push + 1:
        raise AssertionError(
            f"{label}: the pooled second post-backward forward did not re-push"
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
    device = device_str(resolve_local_device(rank))
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
            # recv side budgets worst-case imbalance: per-rank recv is
            # data-dependent, up to tokens*topk*world_size dropless; send is
            # bounded by the local tokens*topk exactly.
            tokens * topk * world_size, tokens * topk,
            hidden, dtype, rank, ep_group,
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
                if os.environ.get("MOE_BWD_MEGA") != "1":
                    # host-sink capture; under MOE_BWD_MEGA=1 the sink rides
                    # the launch (P6a) and no host hook can fire
                    def capture_sunk_slots(transport):
                        # Stream-ordered copies taken between the sink and
                        # the reduce, i.e. exactly what barrier #1 publishes.
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
                    # hidden_states kwarg: consumed only under
                    # MOE_SAVED_RECOMPUTE=1 (backward-side recompute).
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem, grad_transport=transport,
                        hidden_states=hs,
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
    device = device_str(resolve_local_device(rank))
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
            # recv side budgets worst-case imbalance: per-rank recv is
            # data-dependent, up to tokens*topk*world_size dropless; send is
            # bounded by the local tokens*topk exactly.
            tokens * topk * world_size, tokens * topk,
            hidden, dtype, rank, ep_group,
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
                # Pooling-era contract keys (MOE_MEGA_REPREFETCH=1): the
                # oracle-built saved predates pooling, so mirror the three
                # entries _native_saved._attach_moonep_plan_sections adds.
                saved_phys.update(
                    replica_buffers=op._replica_weight_buffers,
                    replica_gate_ready=op.context.replica_gate_ready,
                    replica_down_ready=op.context.replica_down_ready,
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
                if os.environ.get("MOE_BWD_MEGA") != "1":
                    # host-sink capture; under MOE_BWD_MEGA=1 the sink rides
                    # the launch (P6a) and no host hook can fire
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
                    # hidden_states kwarg: consumed only under
                    # MOE_SAVED_RECOMPUTE=1 (backward-side recompute).
                    triton_result = moe_backward_triton(
                        saved_phys, dy, peer_mem, grad_transport=transport,
                        hidden_states=hidden_states,
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

# Zero-copy weight references (§3.1 F).  Both sides save ``fc1_combined`` as
# ``gate_up_weight.transpose(1, 2)`` stride views of the packed table (the
# replay side matches the production layout since the MOE_MEGA_REPREFETCH
# flat RMA push reads its natural storage order), ``fc1_1``/``fc1_2`` are
# slice-transpose views, ``fc2`` is the down weight itself.
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
    device = device_str(resolve_local_device(rank))
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


# ----------------------------------------------------------------------------
# Stage N1/N2 (master plan §N): MoonEP-physical native-saved acceptance case.
#
# The fused forward with ``enable_moonep`` + ``return_saved=True`` must produce
# the ``saved_phys`` contract of
# ``_moonep_torch_forward.build_physical_saved_from_plan`` directly out of the
# production path (fused dispatch + RMA replica prefetch + shadow FC2), with no
# torch replay.  Unlike the home-layout replay (destination-rank-stable send
# order) the MoonEP oracle replays *the plan's own* (destination, physical
# slot) bucket-stable send order, so every permutation is bitwise comparable
# with the native capture — no convention negotiation needed.
# ----------------------------------------------------------------------------

MOONEP_NATIVE_SAVED_BITWISE_KEYS = (
    # plan metadata over the physical [home | replica] slot range
    "expert_counts",
    "split_size_cum_per_expert",
    "meta_expert_ids",
    "meta_split_cum",
    "meta_tile_num",
    "meta_tile_num_cum",
    "num_tiles_total",
    # raw plan snapshots
    "plan_send_counts_by_rank_expert",
    "plan_send_bucket_starts",
    "plan_send_bucket_dst_starts",
    "plan_recv_counts_by_source_expert",
    "plan_received_expert_offsets",
    # permutations (same plan send order on both sides)
    "sort_idxs",
    "inv_sort",
    "local_sort_idxs",
    "inv_local",
    # MoonEP plan sections
    "experts_to_copy",
    "experts_to_copy_cpu",
    # input routes
    "selected_experts",
)

MOONEP_NATIVE_SAVED_SCALAR_KEYS = NATIVE_SAVED_SCALAR_KEYS + (
    "num_experts",
    "home_experts_per_rank",
    "physical_experts_per_rank",
    "active_physical_experts_per_rank",
)

# Computed activations: fused kernels vs torch grouped matmul -> plan's 2e-2.
MOONEP_NATIVE_SAVED_FLOAT_KEYS = (
    "fc1_output",
    "swiglu_out_weighted",
    "recv_weights_sorted",
)

# Weight tables: the oracle's copies (HCCL p2p gather / contiguous twins) must
# be bit-identical to the native references — replica tables filled by the
# production RMA prefetch, home views of the forward inputs.
MOONEP_NATIVE_SAVED_WEIGHT_KEYS = (
    "replica_gate_up",
    "replica_down",
    "fc1_1",
    "fc1_2",
    "fc1_combined",
    "fc2",
)


def _check_moonep_native_saved_weight(
    native_saved, oracle_saved, key, failures
):
    """One weight-table key: present, same dtype/shape, bit-identical values."""
    native_value = native_saved.get(key)
    oracle_value = oracle_saved.get(key)
    if not isinstance(native_value, torch.Tensor):
        failures.append(f"{key}: missing from the native saved")
        return
    if not isinstance(oracle_value, torch.Tensor):
        failures.append(f"{key}: missing from the oracle saved_phys")
        return
    if native_value.dtype != oracle_value.dtype:
        failures.append(
            f"{key}: dtype {native_value.dtype} != oracle {oracle_value.dtype}"
        )
        return
    if tuple(native_value.shape) != tuple(oracle_value.shape):
        failures.append(
            f"{key}: shape {tuple(native_value.shape)} != oracle "
            f"{tuple(oracle_value.shape)}"
        )
        return
    if not torch.equal(
        native_value.contiguous(), oracle_value.contiguous()
    ):
        failures.append(
            f"{key}: values differ from the oracle copy of the same weights"
        )


def _assert_moonep_native_saved(
    native_saved,
    native_output,
    oracle_saved,
    oracle_output,
    op,
    rank,
    label,
):
    """N1+N2 verdict: native MoonEP saved against the saved_phys oracle."""
    failures = []
    for key in MOONEP_NATIVE_SAVED_BITWISE_KEYS:
        if key == "recv_hidden_sorted":
            continue
        _check_native_saved_tensor(native_saved, oracle_saved, key, failures)

    # Pure dispatch copy in the same physical (slot, source) row order — any
    # deviation is a major signal (row-order mismatch between the fused
    # dispatch and the oracle regroup), never relaxed.
    h2_failures = []
    _check_native_saved_tensor(
        native_saved, oracle_saved, "recv_hidden_sorted", h2_failures
    )
    for message in h2_failures:
        failures.append(
            f"{message}  [MAJOR: recv_hidden_sorted is a pure dispatch copy in "
            "the physical slot row order and must match the oracle "
            "bit-for-bit — report to the orchestrator, do not relax]"
        )

    for key in MOONEP_NATIVE_SAVED_SCALAR_KEYS:
        _check_native_saved_scalar(native_saved, oracle_saved, key, failures)
    if native_saved.get("use_moonep") is not True:
        failures.append("use_moonep: native saved is not flagged True")
    for key in MOONEP_NATIVE_SAVED_FLOAT_KEYS:
        _check_native_saved_float_tensor(
            native_saved,
            oracle_saved,
            key,
            NATIVE_SAVED_H2_RTOL,
            NATIVE_SAVED_H2_ATOL,
            failures,
        )
    for key in MOONEP_NATIVE_SAVED_WEIGHT_KEYS:
        _check_moonep_native_saved_weight(
            native_saved, oracle_saved, key, failures
        )

    # The replica tables must alias the operator's live symmetric buffers
    # (zero-copy contract): a silently materialized copy would pin ~168 MiB per
    # MoE layer per step on the kimi-k3 shapes.
    if op._replica_weight_buffers is None:
        failures.append("replica weight buffers are not allocated")
    else:
        for key, live_view in (
            ("replica_gate_up", op._replica_weight_buffers.gate_up),
            ("replica_down", op._replica_weight_buffers.down),
        ):
            native_value = native_saved.get(key)
            if (
                not isinstance(native_value, torch.Tensor)
                or native_value.data_ptr() != live_view.data_ptr()
            ):
                failures.append(
                    f"{key}: saved table does not alias the live symmetric "
                    "replica buffer"
                )

    output_matches = True
    try:
        assert_close(
            native_output,
            oracle_output,
            rtol=OUTPUT_RTOL,
            atol=OUTPUT_ATOL,
        )
    except AssertionError:
        output_matches = False
        failures.append("output: return_saved output differs from the oracle")

    ok = not failures
    if not ok:
        print(f"[rank {rank}] {label}: {'; '.join(failures)}", flush=True)
    if rank == 0:
        suffix = "" if ok else "  |  " + "; ".join(failures)
        print(f"[{'PASS' if ok else 'FAIL'}] {label}{suffix}", flush=True)
    return ok


def run_moonep_native_saved_hot_expert_case(
    rank: int,
    world_size: int,
) -> None:
    """N1+N2: MoonEP native saved (return_saved=True) vs the saved_phys oracle.

    One production ``op.forward(..., return_saved=True)`` on the all-hot
    MoonEP plan, then the independent oracle: a metadata-only plan rebuild
    (planning is deterministic for the same routes) plus the torch+HCCL
    replay.  The native replica tables come from the production RMA prefetch,
    the oracle's from ``gather_replica_weights_via_hccl`` — both must carry
    the owners' weights bit-for-bit.
    """
    if world_size not in (2, 4):
        raise ValueError(
            "the MoonEP native saved case requires two or four ranks"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError(
            "the MoonEP native saved case requires NPU and ACLSHMEM"
        )

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"moonep-native-saved-hot-expert-w{world_size}"

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
                seed=2103 + rank,
            )
            expert_indices = torch.zeros(
                (tokens, topk), dtype=torch.int32, device=device
            )
            routing_weights = torch.empty(
                (tokens, topk), dtype=torch.float32, device=device
            )
            routing_weights[:, 0] = 0.25
            routing_weights[:, 1] = 0.75

            with torch.no_grad():
                native_output, native_saved = op.forward(
                    hs,
                    expert_indices,
                    packed_w1,
                    w2,
                    routing_weights,
                    return_saved=True,
                )
            _assert_physical_saved_layout(native_saved, tokens, topk)

            # Independent oracle from a fresh (deterministic) plan build.
            plan = op.build_routing_plan(expert_indices)
            replica_routes = plan.received_routes_per_expert[
                num_experts // world_size:
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
            oracle_output, oracle_saved = build_physical_saved_from_plan(
                plan,
                hs,
                routing_weights,
                packed_w1,
                w2,
                replica_gate_up,
                replica_down,
                ep_group=ep_group,
            )
            _assert_physical_saved_layout(oracle_saved, tokens, topk)

            all_passed = _assert_moonep_native_saved(
                native_saved,
                native_output,
                oracle_saved,
                oracle_output,
                op,
                rank,
                label,
            )
            flag = torch.tensor(
                [1 if all_passed else 0], dtype=torch.int32, device=device
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
            if not bool(flag.item()):
                raise AssertionError(
                    f"MoonEP native saved mismatched the oracle: {label}"
                )
        finally:
            op.finalize()


def run_moonep_native_backward_symmetric_hot_expert_case(
    rank: int,
    world_size: int,
) -> None:
    """N3+N4: fused forward -> native saved -> 5-op backward + grad transport.

    The full production closure with no torch replay in the hot path: one
    ``op.forward(..., return_saved=True)`` (which also fills the symmetric
    replica tables via the owner-push), the 5-op triton backward consuming the
    *native* saved, and the symmetric-slot grad transport sinking the replica
    weight gradients onto their owners.  Gradients are checked against the
    hand-written logical golden; the transport stages, the published sunk
    slots, and a plain forward after the reduce are checked like the replay
    symmetric case.
    """
    if world_size not in (2, 4):
        raise ValueError(
            "the native symmetric backward case requires two or four ranks"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError(
            "the native symmetric backward case requires NPU and ACLSHMEM"
        )

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    experts_per_rank = num_experts // world_size
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "moonep-native-backward-symmetric-hot-expert"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # peer_mem must stay the session's FIRST symmetric allocation
        # (dl.symm_at offset-0); the operator below claims its own heap
        # objects for planning and dispatch.
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
                    seed=2153 + rank,
                )
                expert_indices = torch.zeros(
                    (tokens, topk), dtype=torch.int32, device=device
                )
                routing_weights = torch.empty(
                    (tokens, topk), dtype=torch.float32, device=device
                )
                routing_weights[:, 0] = 0.25
                routing_weights[:, 1] = 0.75
                torch.manual_seed(2154 + rank)
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

                # One production forward doubles as the native-saved capture;
                # its output is a free golden check on the fused path.
                with torch.no_grad():
                    produced, native_saved = op.forward(
                        hs,
                        expert_indices,
                        packed_w1,
                        w2,
                        routing_weights,
                        return_saved=True,
                    )
                assert_close(
                    produced, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                )
                _assert_physical_saved_layout(native_saved, tokens, topk)
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

                transport = op.lend_replica_weight_tables_for_grad()
                if op._replica_weight_cache_valid is not False:
                    raise AssertionError(
                        "lending the replica tables must invalidate the cache"
                    )
                captured_slots = {}
                if os.environ.get("MOE_BWD_MEGA") != "1":
                    # host-sink capture; under MOE_BWD_MEGA=1 the sink rides
                    # the launch (P6a) and no host hook can fire
                    def capture_sunk_slots(transport):
                        # Stream-ordered copies taken between the sink and
                        # the reduce, i.e. exactly what barrier #1 publishes.
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
                    # N3: the 5-op backward consumes the NATIVE saved dict.
                    # hidden_states kwarg: consumed only under
                    # MOE_SAVED_RECOMPUTE=1 (backward-side recompute).
                    triton_result = moe_backward_triton(
                        native_saved, dy, peer_mem, grad_transport=transport,
                        hidden_states=hs,
                    )
                _assert_moonep_replica_grad_shapes(
                    triton_result, native_saved, experts_per_rank, hidden, ffn
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


def _perturb_weights_layer_local(
    w_local, layer, rank, world_size, num_experts, amp=0.05,
):
    """Add layer-only, rank-sliced full-table ±amp noise to local weights.

    The session pool hands every same-shape layer the same symmetric replica
    table, and the stock weight makers use fixed seeds — every layer would
    own identical weights and a stale-table read would be invisible.  The
    noise comes from a FULL ``[num_experts, ...]`` table seeded only by the
    layer, sliced to this rank's experts, so every rank agrees on every
    expert's true weight while each layer differs measurably.
    """
    epr = num_experts // world_size
    g = torch.Generator(device="cpu").manual_seed(9100 + 13 * layer)
    full = (
        torch.rand((num_experts,) + tuple(w_local.shape[1:]), generator=g) * 2
        - 1
    ) * amp
    return (
        w_local
        + full[rank * epr:(rank + 1) * epr].to(
            dtype=w_local.dtype, device=w_local.device
        )
    ).contiguous()


def run_moonep_multilayer_pool_epoch_case(rank: int, world_size: int) -> None:
    """Cross-layer staleness regression for the pooled replica tables.

    L same-shape operators (one per MoE layer) share the session-level
    replica-table pool.  Every layer runs forward -> native backward +
    symmetric grad transport in order, twice: step 2 reseeds the inputs, so a
    layer consuming a stale pooled table — another layer's weights, its own
    step-1 weights, or gradients/zeroes left by a previous transport — fails
    the gradient oracle.  The pooled epochs must also advance strictly across
    the whole (layer, step) sequence, all layers must resolve to one buffers
    object with pool refcount L, and finalize must drain the pool.
    """
    if world_size not in (2, 4):
        raise ValueError(
            "the multi-layer pool epoch case requires two or four ranks"
        )
    if os.environ.get("MEGAMOE_REPLICA_POOL") != "1":
        raise RuntimeError(
            "the multi-layer pool epoch case requires MEGAMOE_REPLICA_POOL=1"
        )
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError(
            "the multi-layer pool epoch case requires NPU and ACLSHMEM"
        )

    tokens, hidden, ffn, topk, num_experts = 32, 256, 512, 2, 8
    num_layers = 3
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "moonep-multilayer-pool-epoch"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # Backward scratch, reused layer by layer (the session's first
        # symmetric allocation keeps the dl.symm_at offset-0 contract).
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk, tokens * topk, hidden, dtype, rank, ep_group
        )
        ops = []
        layer_weights = []
        try:
            for layer in range(num_layers):
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
                w_gate, w_up = make_gate_up_weights(
                    num_experts, hidden, ffn, world_size, rank, dtype, device
                )
                w_gate = _perturb_weights_layer_local(
                    w_gate, layer, rank, world_size, num_experts
                )
                w_up = _perturb_weights_layer_local(
                    w_up, layer, rank, world_size, num_experts
                )
                packed_w1 = pack_gate_up_weights(w_gate, w_up)
                w2 = _perturb_weights_layer_local(
                    make_down_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype, device
                    ),
                    layer, rank, world_size, num_experts,
                )
                ops.append(op)
                layer_weights.append((packed_w1, w2, w_gate, w_up))

            if os.environ.get("MOONEP_TEST_BALANCED") == "1":
                # M1 discriminator (2026-09-20): near-uniform round-robin
                # routing — the documented concentrated-routing whole-kernel
                # miscompile (mega_bwd.py:147-164) must not fire here.
                expert_indices = (
                    torch.arange(tokens * topk, dtype=torch.int32)
                    .view(tokens, topk) % num_experts
                ).to(device)
            else:
                expert_indices = torch.zeros(
                    (tokens, topk), dtype=torch.int32, device=device
                )
            routing_weights = torch.empty(
                (tokens, topk), dtype=torch.float32, device=device
            )
            routing_weights[:, 0] = 0.25
            routing_weights[:, 1] = 0.75

            epoch_sequence = []
            for step in (0, 1):
                for layer, (op, (packed_w1, w2, w_gate, w_up)) in enumerate(
                    zip(ops, layer_weights)
                ):
                    hs, _ = prepare_inputs(
                        tokens,
                        hidden,
                        num_experts,
                        topk,
                        dtype,
                        device,
                        seed=2153 + rank + 77 * step,
                    )
                    torch.manual_seed(2154 + rank + 77 * step)
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
                    with torch.no_grad():
                        produced, native_saved = op.forward(
                            hs,
                            expert_indices,
                            packed_w1,
                            w2,
                            routing_weights,
                            return_saved=True,
                        )
                    assert_close(
                        produced, expected, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL
                    )
                    epoch_sequence.append(op._replica_weight_epoch)
                    transport = op.lend_replica_weight_tables_for_grad()
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
                            native_saved,
                            dy,
                            peer_mem,
                            grad_transport=transport,
                            hidden_states=hs,
                        )
                    _assert_gradients_match_golden(
                        rank,
                        triton_result,
                        torch_result,
                        ep_group,
                        f"{label}-L{layer}-s{step}",
                    )
                    accumulated = _assert_symmetric_transport_stages(
                        rank,
                        transport,
                        {},
                        triton_result,
                        ep_group,
                        f"{label}-L{layer}-s{step}",
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

            if any(
                later <= earlier
                for earlier, later in zip(epoch_sequence, epoch_sequence[1:])
            ):
                raise AssertionError(
                    f"pooled replica epochs must strictly increase across "
                    f"layers/steps, got {epoch_sequence}"
                )
            snapshot = replica_pool_snapshot()
            if len(snapshot) != 1 or list(snapshot.values()) != [num_layers]:
                raise AssertionError(
                    "all same-shape layers must pool into one entry with "
                    f"refcount {num_layers}, got {snapshot}"
                )
            if len({id(op._replica_weight_buffers) for op in ops}) != 1:
                raise AssertionError(
                    "pooled layers did not resolve to one buffers object"
                )
        finally:
            for op in ops:
                op.finalize()
        snapshot = replica_pool_snapshot()
        if snapshot:
            raise AssertionError(
                f"finalize must drain the replica pool, got {snapshot}"
            )
        kit.ash.aclshmem_free_tensor(peer_mem)


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
    device = device_str(resolve_local_device(rank))
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


def run_megamoe_situglu_autograd_case(rank: int, world_size: int) -> None:
    """SiTU-GLU backward parity: the H3 recipe with ``activation="situglu``.

    Kimi-K3 runs ``hidden_act="situ"`` (β=4.0, lβ=25.0), so the fused
    backward's step-2 derivative must be the SiTU one, not silu'.  Same
    lifecycle and verdict discipline as ``run_megamoe_native_autograd_case``;
    the eager golden is made SiTU-consistent by re-deriving the replay's
    weighted activation from the SAME pre-activation halves (fp32 SiTU, fp32
    route scale, bf16 store — the operator's numeric path) and telling the
    baseline which activation to differentiate.  A silu-derived backward
    cannot pass this golden: the SiTU derivative differs at every element.
    """
    if world_size != 2:
        raise ValueError("the situglu autograd case requires exactly two ranks")
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("situglu autograd requires NPU and ACLSHMEM")
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")

    tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 128
    situ_beta, situ_linear_beta = 4.0, 25.0
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = "megamoe-situglu-autograd-w2"

    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2)
    ):
        # See the native autograd case: first symmetric allocation, rows sized
        # for the worst-case dropless receive (tokens*topk*world_size).
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
                    activation="situglu",
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
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
                    tokens, hidden, num_experts, topk, dtype, device,
                    seed=2203 + rank,
                )
                routing_weights = make_routing_weights(
                    tokens, topk, device, seed=2204 + rank
                )
                torch.manual_seed(2205 + rank)
                dy = torch.randn(tokens, hidden, dtype=dtype, device=device)

                # Eager golden: the replay saved (silu) converted to SiTU — the
                # replay's dispatch metadata, halves and weights are
                # activation-independent; only swiglu_out_weighted and the
                # differentiated activation change.
                with torch.no_grad():
                    _, golden_saved = moe_forward(
                        hs, routing_weights, expert_indices, w_gate, w_up, w2,
                        ep_group, topk, return_saved=True,
                    )
                    gate = golden_saved["gate"].float()
                    up = golden_saved["up"].float()
                    situ_a = (
                        situ_beta
                        * torch.tanh(gate / situ_beta)
                        * torch.sigmoid(gate)
                    )
                    up_v = situ_linear_beta * torch.tanh(
                        up / situ_linear_beta
                    )
                    golden_saved["swiglu_out_weighted"] = (
                        situ_a
                        * up_v
                        * golden_saved["recv_weights_sorted"]
                        .float()
                        .unsqueeze(-1)
                    ).to(dtype)
                    golden_saved["activation"] = "situglu"
                    golden_saved["situ_beta"] = situ_beta
                    golden_saved["situ_linear_beta"] = situ_linear_beta
                    golden = backward_torch_baseline(golden_saved, dy)

                state = SimpleNamespace(signal_mem=None, epoch=0)

                def run_function_step():
                    hidden_leaf = hs.clone().requires_grad_(True)
                    routing_leaf = routing_weights.clone().requires_grad_(True)
                    gate_up_leaf = packed_w1.clone().requires_grad_(True)
                    down_leaf = w2.clone().requires_grad_(True)
                    output = MegaMoEFunction.apply(
                        op, hidden_leaf, routing_leaf, expert_indices,
                        gate_up_leaf, down_leaf, peer_mem, state,
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

                # Epoch reuse must stay bitwise on the SiTU derivative too.
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

                flag = torch.tensor(
                    [1 if all_ok else 0], dtype=torch.int32, device=device
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
                if rank == 0 and not bool(flag.item()):
                    print(f"{label} gradient details: {details}", flush=True)
                if not bool(flag.item()):
                    raise AssertionError(
                        f"{label}: situglu Function grads mismatched the "
                        "SiTU eager golden"
                    )
            finally:
                op.finalize()
        finally:
            kit.ash.aclshmem_free_tensor(peer_mem)


def run_single_kernel_situglu_autograd_case(
    rank: int, world_size: int, fc1_offload: bool = False,
    down_direct: bool = False,
) -> None:
    """End-to-end gate for the single-kernel forward + mega recompute backward.

    ``enable_single_kernel_forward`` returns the MINIMAL saved contract; ops/
    function.py routes it through ``_single_saved_adapter`` (closed-form plan
    tables + the two layout tripwires) and the backward runs the one-launch
    mega kernel under ``MOE_BWD_MEGA=1`` + ``MOE_SAVED_RECOMPUTE=1``, which
    recomputes the big activations in-launch instead of consuming them.  This
    case pins that whole chain to the same SiTU eager golden as
    ``run_megamoe_situglu_autograd_case`` — any layout drift in the adapter
    (fc1_output / recv_weights_sorted pass-through) or the re-dispatch
    recompute shows up as a gradient mismatch.

    ``fc1_offload=True`` additionally sets ``MEGAMOE_FC1_OFFLOAD=1``: the
    forward D2Hs ``fc1_output`` to a pooled pinned host buffer on a side
    stream and the backward entry H2Ds it back (the SwapTensor idiom from
    the framework's async_offload.py) — the grads must stay identical.

    ``down_direct=True`` additionally sets ``MOE_DOWN_DIRECT=1`` and hands
    the op the FRAMEWORK'S down-projection view — a transposed stride view
    of an ``[E, F, H]`` table (values identical, strides ``(F*H, 1, H)``)
    — instead of a contiguous ``[E, H, F]`` table: the single-kernel
    forward and the mega backward must address it through their stride
    parameters with no staging (no ``_fc2_ws``, no backward
    ``.contiguous()``).
    """
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("single-kernel autograd requires NPU and ACLSHMEM")
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")

    # The single-kernel contract is only consumable by the recompute mega
    # backward; set the gates for this case and restore them after (the suite
    # may run other backward variants in the same process).
    _bwd_env = {"MOE_BWD_MEGA": "1", "MOE_SAVED_RECOMPUTE": "1"}
    if fc1_offload:
        _bwd_env["MEGAMOE_FC1_OFFLOAD"] = "1"
    if down_direct:
        _bwd_env["MOE_DOWN_DIRECT"] = "1"
    _env_before = {k: os.environ.get(k) for k in _bwd_env}
    os.environ.update(_bwd_env)

    # E must stay <= 32: the kernel scatter's multi-bin-block path corrupts
    # the send tables above 32 bins (adapter rejects it; see
    # _single_saved_adapter).  At w8 use the exact Kimi-K3 integration shape
    # (S=1024, H=7168, F=3072, topk=8, E=32, EPR=4); w2 is the small smoke.
    if world_size == 8:
        tokens, hidden, ffn, topk, num_experts = 1024, 7168, 3072, 8, 32
    else:
        tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 32
    situ_beta, situ_linear_beta = 4.0, 25.0
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"single-kernel-situglu-autograd-w{world_size}"

    try:
        with kit.aclshmem_session(
            rank, world_size, kit.get_ash_size_bytes(default_gb=2)
        ):
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
                        activation="situglu",
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                        enable_single_kernel_forward=True,
                        # single-kernel constraint: the two block sizes must
                        # match (defaults are 256/256; set explicitly).
                        fc1_gemm_block_size_m=256,
                        fc2_combine_block_size_m=256,
                    ),
                )
                try:
                    w_gate, w_up = make_gate_up_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )
                    packed_w1 = pack_gate_up_weights(w_gate, w_up)
                    w2 = make_down_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )
                    if down_direct:
                        # framework hand-off pattern: a transposed stride
                        # view of the host's [E, F, H] table (identical
                        # values, strides (F*H, 1, H), non-contiguous)
                        w2_view = (
                            w2.transpose(1, 2).contiguous().transpose(1, 2)
                        )
                        if w2_view.is_contiguous() or not torch.equal(
                            w2_view, w2
                        ):
                            raise AssertionError(
                                "down_direct setup failed to build a "
                                "value-identical strided view"
                            )
                    # Droless inputs (no drop_frac): the adapter rejects
                    # dropped routes — capacity_factor=world_size keeps them.
                    hs, expert_indices = prepare_inputs(
                        tokens, hidden, num_experts, topk, dtype, device,
                        seed=2303 + rank,
                    )
                    routing_weights = make_routing_weights(
                        tokens, topk, device, seed=2304 + rank
                    )
                    torch.manual_seed(2305 + rank)
                    dy = torch.randn(
                        tokens, hidden, dtype=dtype, device=device
                    )

                    # Same SiTU eager golden as the situglu autograd case.
                    with torch.no_grad():
                        _, golden_saved = moe_forward(
                            hs, routing_weights, expert_indices, w_gate, w_up,
                            w2, ep_group, topk, return_saved=True,
                        )
                        gate = golden_saved["gate"].float()
                        up = golden_saved["up"].float()
                        situ_a = (
                            situ_beta
                            * torch.tanh(gate / situ_beta)
                            * torch.sigmoid(gate)
                        )
                        up_v = situ_linear_beta * torch.tanh(
                            up / situ_linear_beta
                        )
                        golden_saved["swiglu_out_weighted"] = (
                            situ_a
                            * up_v
                            * golden_saved["recv_weights_sorted"]
                            .float()
                            .unsqueeze(-1)
                        ).to(dtype)
                        golden_saved["activation"] = "situglu"
                        golden_saved["situ_beta"] = situ_beta
                        golden_saved["situ_linear_beta"] = situ_linear_beta
                        golden = backward_torch_baseline(golden_saved, dy)

                    state = SimpleNamespace(
                        signal_mem=None, epoch=0, mega_persistent={}
                    )

                    def run_function_step():
                        hidden_leaf = hs.clone().requires_grad_(True)
                        routing_leaf = (
                            routing_weights.clone().requires_grad_(True)
                        )
                        gate_up_leaf = packed_w1.clone().requires_grad_(True)
                        if down_direct:
                            # the strided view itself is the leaf — cloning
                            # would materialize a contiguous table and
                            # defeat the point
                            down_leaf = w2_view.requires_grad_(True)
                        else:
                            down_leaf = w2.clone().requires_grad_(True)
                        output = MegaMoEFunction.apply(
                            op, hidden_leaf, routing_leaf, expert_indices,
                            gate_up_leaf, down_leaf, peer_mem, state,
                        )
                        output.backward(dy)
                        return (
                            hidden_leaf, routing_leaf, gate_up_leaf, down_leaf
                        )

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
                                f"{None if value is None else tuple(value.shape)}"
                                f" != golden {tuple(golden[name].shape)}"
                            )
                    all_ok, details = compare_backward_gradients(
                        first_grads, golden
                    )
                    epoch_after_first = state.epoch
                    if state.signal_mem is None:
                        raise AssertionError(
                            f"{label}: the Function did not persist signal_mem "
                            "in the caller state"
                        )
                    if not getattr(state, "mega_persistent", {}):
                        raise AssertionError(
                            f"{label}: the mega backward did not persist its "
                            "slabs/epochs in state.mega_persistent"
                        )
                    if down_direct and op._fc2_ws is not None:
                        raise AssertionError(
                            f"{label}: MOE_DOWN_DIRECT staged the down weight "
                            "anyway (op._fc2_ws is allocated)"
                        )

                    # Slab/epoch reuse across steps must stay bitwise.
                    second_grads = _megamoe_function_grads(
                        run_function_step(), ffn
                    )
                    for name, again in second_grads.items():
                        if not torch.equal(again, first_grads[name]):
                            raise AssertionError(
                                f"{label}: slab reuse changed grad {name}"
                            )
                    if state.epoch < epoch_after_first or state.epoch < 1:
                        raise AssertionError(
                            f"{label}: the Function did not advance/write back "
                            f"the epoch (after first={epoch_after_first}, "
                            f"after second={state.epoch})"
                        )

                    if fc1_offload:
                        # The swap must have actually run on BOTH steps —
                        # one D2H per forward, one H2D per backward — and
                        # the device-side round trip is stream-ordered
                        # behind the grad checks above, so the counters are
                        # final here.
                        from mega_moe.ops._fc1_host_offload import (
                            fc1_offload_stats,
                        )
                        d2h_n, h2d_n, d2h_bytes = fc1_offload_stats()
                        if d2h_n < 2 or h2d_n < 2 or d2h_bytes <= 0:
                            raise AssertionError(
                                f"{label}: fc1 host offload did not run "
                                f"(d2h={d2h_n} h2d={h2d_n} "
                                f"bytes={d2h_bytes})"
                            )

                    flag = torch.tensor(
                        [1 if all_ok else 0], dtype=torch.int32, device=device
                    )
                    dist.all_reduce(
                        flag, op=dist.ReduceOp.MIN, group=ep_group
                    )
                    if rank == 0 and not bool(flag.item()):
                        print(
                            f"{label} gradient details: {details}", flush=True
                        )
                    if not bool(flag.item()):
                        raise AssertionError(
                            f"{label}: single-kernel forward + mega recompute "
                            "backward grads mismatched the SiTU eager golden"
                        )
                finally:
                    op.finalize()
            finally:
                kit.ash.aclshmem_free_tensor(peer_mem)
    finally:
        for key, value in _env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _quantize_fc1_half(x, group_n: int):
    """Golden-side twin of the kernel's fp8 save leg (fused_forward's
    FC1_FP8 block): per-(row, group) E4M3 quantize-dequantize with scale =
    max(amax, 1e-3-sentinel)/448.  Applying it to the EAGER golden's gate/up
    halves makes the gradient comparison measure the MoonEP path's handling
    of the quantized save, not fp8 rounding noise (both sides then sit at
    the same quantized point; the residual is the two implementations'
    usual ~2e-2 plus e4m3 cast-boundary ulps)."""
    rows, ffn_dim = x.shape
    if ffn_dim % group_n:
        raise ValueError(
            f"quantize groups must tile the half: ffn={ffn_dim} "
            f"group_n={group_n}")
    g = x.float().view(rows, ffn_dim // group_n, group_n)
    amax = g.abs().amax(dim=-1, keepdim=True)
    safe = torch.where(amax > 1e-3, amax, torch.ones_like(amax))
    q = (g * (448.0 / safe)).to(torch.float8_e4m3fn)
    deq = (q.float() * (safe / 448.0)).view(rows, ffn_dim)
    return deq.to(x.dtype)


def _compare_grads_relaxed(grads, golden, *, rtol, atol):
    """cmp_grad over the five canonical keys with explicit tolerances —
    compare_backward_gradients' fixed GRAD_* bound does not fit a
    quantized-save gate."""
    rows = []
    all_ok = True
    for name in ("grad_hidden", "grad_routing_weights", "grad_fc1_1",
                 "grad_fc1_2", "grad_fc2"):
        ok, max_abs, rel, nbad = cmp_grad(
            name, grads[name], golden[name], rtol=rtol, atol=atol)
        rows.append({"name": name, "ok": bool(ok), "max_abs": max_abs,
                     "relative": rel, "mismatches": nbad})
        all_ok = all_ok and ok
    return bool(all_ok), rows


def run_single_kernel_moonep_autograd_case(
    rank: int, world_size: int, save_fc1_dtype: str = "bf16",
) -> None:
    """Single-kernel MoonEP forward + mega recompute backward + reprefetch.

    ``enable_single_kernel_forward`` + ``enable_moonep``: the minimal
    contract's layout tables are PHYSICAL-slot (home | replica); the
    adapter's MoonEP branch expands them into the physical backward contract
    (plan tables snapshotted from the device planner's metadata, live
    replica weight tables, auto-lend staged) and the backward runs the
    one-launch mega kernel.  Gradients pin to the SAME SiTU eager golden as
    the non-MoonEP case — MoonEP only moves rows between slots, so every
    gradient must be numerically identical.

    The step interleave is the megamoe_shared_op whole-net scenario in
    miniature on ONE operator: fwd(step1) -> fwd(step2) -> bwd(step2) ->
    bwd(step1).  Step 2 uses SCALED weights, so at bwd(step1) time both the
    operator's replica tables and its staged ETC cache hold step 2's state —
    the grads can only match the golden if (a) MOE_MEGA_REPREFETCH (default
    on) really re-pushed step 1's replica weights in-launch, and (b) the
    grad transport pulled over step 1's own ETC (the saved-dict override),
    not the operator cache's.  Skewed Kimi routes guarantee replica traffic
    in both steps (asserted against the host planning oracle up front).
    """
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("single-kernel autograd requires NPU and ACLSHMEM")
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")

    # MOE_MEGA_REPREFETCH stays UNSET: default-on is part of the contract
    # under test (pooled/rewritten tables are only correct with the
    # in-launch re-push; =0 is valid only with per-layer, per-step tables).
    _bwd_env = {
        "MOE_BWD_MEGA": "1",
        "MOE_SAVED_RECOMPUTE": "1",
        "MEGAMOE_REPLICA_POOL": "1",
    }
    if save_fc1_dtype == "fp8":
        # production pairing (!59): the fp8 save rides the FC1 host-offload
        # leg in the 整网 config — exercise both together under moonep
        _bwd_env["MEGAMOE_FC1_OFFLOAD"] = "1"
    _env_before = {k: os.environ.get(k) for k in _bwd_env}
    os.environ.update(_bwd_env)

    # Same shapes as the non-MoonEP single-kernel case: w2 small smoke
    # (E=32, epn=16), w8 the exact Kimi-K3 integration shape (E=32, EPR=4,
    # needs the 8GB symmetric heap like its non-MoonEP twin).
    if world_size == 8:
        tokens, hidden, ffn, topk, num_experts = 1024, 7168, 3072, 8, 32
    else:
        tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 32
    situ_beta, situ_linear_beta = 4.0, 25.0
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"single-kernel-moonep-autograd-w{world_size}"
    if save_fc1_dtype == "fp8":
        label += "-fp8"

    from benchmark.layer._kimi_routes import kimi_skewed_routes

    def cpu_step_routes(shift, r):
        """Replica-engaging routes per (step shift, source rank), CPU int64.

        w8/Kimi shape: the recorded router-collapse profile (the helper
        requires W8 + topk 8/16).  w2 smoke: one hot expert per step
        (alternating home ranks) plus a per-rank random spread — the hot
        expert overflows its home budget onto replicas while the rest keeps
        every expert's gradient exercised.
        """
        if world_size == 8:
            return (
                (kimi_skewed_routes(tokens, num_experts, topk, r) + shift)
                % num_experts
            )
        torch.manual_seed(9000 + r * 10 + shift)
        routes = torch.randint(
            0, num_experts, (tokens, topk), dtype=torch.int64
        )
        # step1 hammers rank0's first home expert, step2 rank1's — each
        # step's overflow lands on the OTHER rank's replica slots.
        routes[:, 0] = 0 if shift == 0 else num_experts // world_size
        return routes

    def skewed_step_routes(shift):
        return cpu_step_routes(shift, rank).to(
            device=device, dtype=torch.int32
        ).contiguous()

    # The skewed routes must actually engage replicas in every step — check
    # against the host planning oracle BEFORE running, so a routing change
    # that silently drops replica traffic fails loudly here, not as a
    # vacuous pass downstream.
    for shift in (0, 4):
        counts = torch.stack(
            [
                torch.bincount(
                    cpu_step_routes(shift, r).flatten().long(),
                    minlength=num_experts,
                )
                for r in range(world_size)
            ]
        )
        oracle = plan_moonep_b0_b3(counts)
        if not bool((oracle.experts_to_copy >= 0).any()):
            raise AssertionError(
                f"{label}: skewed routes (shift={shift}) produced no "
                "replicas; the case cannot exercise the replica paths"
            )

    try:
        with kit.aclshmem_session(
            rank, world_size, kit.get_ash_size_bytes(default_gb=2),
            enable_udma=True,
        ):
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
                        activation="situglu",
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                        enable_single_kernel_forward=True,
                        enable_moonep=True,
                        save_fc1_dtype=save_fc1_dtype,
                        # fp8 halves the fc1 payload UB demand (e4m3 pair
                        # buffer rides the block), so the quantized config
                        # halves the M blocks — mirrors the !29 quantized-run
                        # geometry
                        fc1_gemm_block_size_m=(
                            128 if save_fc1_dtype == "fp8" else 256
                        ),
                        fc2_combine_block_size_m=(
                            128 if save_fc1_dtype == "fp8" else 256
                        ),
                    ),
                )
                try:
                    w_gate, w_up = make_gate_up_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )
                    w2 = make_down_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )

                    def make_golden(hs, rw, ei, dy, gate_w, up_w, down_w):
                        # same SiTU eager recipe as the situglu autograd case
                        with torch.no_grad():
                            _, gs = moe_forward(
                                hs, rw, ei, gate_w, up_w, down_w, ep_group,
                                topk, return_saved=True,
                            )
                            gate = gs["gate"].float()
                            up = gs["up"].float()
                            if save_fc1_dtype == "fp8":
                                # the mega backward reads the QUANTIZED save;
                                # quantize the golden to the same point so the
                                # compare measures the moonep path's handling
                                # of the fp8 save, not e4m3 noise.  Kernel
                                # group size: fc1_scale_group_size =
                                # fc1_gemm_block_size_n // 2
                                group_n = op.config.fc1_gemm_block_size_n // 2
                                gate = _quantize_fc1_half(gate, group_n)
                                up = _quantize_fc1_half(up, group_n)
                            situ_a = (
                                situ_beta
                                * torch.tanh(gate / situ_beta)
                                * torch.sigmoid(gate)
                            )
                            up_v = situ_linear_beta * torch.tanh(
                                up / situ_linear_beta
                            )
                            gs["swiglu_out_weighted"] = (
                                situ_a
                                * up_v
                                * gs["recv_weights_sorted"]
                                .float()
                                .unsqueeze(-1)
                            ).to(dtype)
                            gs["activation"] = "situglu"
                            gs["situ_beta"] = situ_beta
                            gs["situ_linear_beta"] = situ_linear_beta
                            return backward_torch_baseline(gs, dy)

                    # Two steps: different routes AND different weight values
                    # (step 2 scaled), so the pooled replica content at
                    # bwd(step1) time is step 2's — reading it un-repushed
                    # would produce wrong grads, not just a stale-but-equal
                    # pass.
                    steps = []
                    for idx, (shift, g_scale, d_scale) in enumerate(
                        ((0, 1.0, 1.0), (4, 0.75, 0.5))
                    ):
                        hs, _ = prepare_inputs(
                            tokens, hidden, num_experts, topk, dtype, device,
                            seed=2303 + rank + idx,
                        )
                        rw = make_routing_weights(
                            tokens, topk, device, seed=2304 + rank + idx
                        )
                        torch.manual_seed(2305 + rank + idx)
                        dy = torch.randn(
                            tokens, hidden, dtype=dtype, device=device
                        )
                        ei = skewed_step_routes(shift)
                        gate_w = w_gate * g_scale
                        up_w = w_up * g_scale
                        down_w = w2 * d_scale
                        steps.append(
                            dict(
                                hs=hs, rw=rw, ei=ei, dy=dy,
                                gate_w=gate_w, up_w=up_w, down_w=down_w,
                                packed=pack_gate_up_weights(gate_w, up_w),
                                golden=make_golden(
                                    hs, rw, ei, dy, gate_w, up_w, down_w
                                ),
                            )
                        )

                    state = SimpleNamespace(
                        signal_mem=None, epoch=0, mega_persistent={}
                    )

                    def forward_step(step):
                        hidden_leaf = step["hs"].clone().requires_grad_(True)
                        routing_leaf = (
                            step["rw"].clone().requires_grad_(True)
                        )
                        gate_up_leaf = (
                            step["packed"].clone().requires_grad_(True)
                        )
                        down_leaf = step["down_w"].clone().requires_grad_(True)
                        output = MegaMoEFunction.apply(
                            op, hidden_leaf, routing_leaf, step["ei"],
                            gate_up_leaf, down_leaf, peer_mem, state,
                        )
                        return output, (
                            hidden_leaf, routing_leaf, gate_up_leaf, down_leaf
                        )

                    dist.barrier()
                    out1, leaves1 = forward_step(steps[0])
                    if op._replica_experts_cache is None:
                        raise AssertionError(
                            f"{label}: the adapter did not stage "
                            "op._replica_experts_cache for the auto-lend"
                        )
                    # fwd(step2) rewrites the shared planning workspaces,
                    # the replica tables AND the operator's ETC cache.
                    out2, leaves2 = forward_step(steps[1])

                    # G2 r42b: forward-output row-level probe.  The grads
                    # compare never looks at out1/out2, so "forward green"
                    # was an assumption; this decides serve-vs-backward by
                    # direct row compare against the same SiTU eager recipe
                    # make_golden trusts.  Hot row = one of the token's
                    # routes hits the step's hammered expert — exactly the
                    # rows the OTHER rank's replica slots serve.
                    if os.environ.get("MOE_MEGA_FWD_DUMP", "0") == "1":
                        for name, out, step in (
                            ("step1", out1, steps[0]),
                            ("step2", out2, steps[1]),
                        ):
                            hot = 0 if name == "step1" else (
                                num_experts // world_size
                            )
                            with torch.no_grad():
                                # return_saved=False yields the bare output
                                # tensor (no saved-state tuple to unpack).
                                ref = moe_forward(
                                    step["hs"], step["rw"], step["ei"],
                                    step["gate_w"], step["up_w"],
                                    step["down_w"], ep_group, topk,
                                    return_saved=False,
                                    activation="situglu",
                                    situ_beta=situ_beta,
                                    situ_linear_beta=situ_linear_beta,
                                )
                            diff = (out.float() - ref.float()).abs()
                            rowbad = (diff > 5e-2).any(dim=-1)
                            has_hot = (step["ei"] == hot).any(dim=-1)
                            print(
                                f"[fwd-dump r{rank}] {name} "
                                f"rows_bad={int(rowbad.sum())}/{tokens} "
                                f"hot_bad={int((rowbad & has_hot).sum())}"
                                f"/{int(has_hot.sum())} "
                                f"nonhot_bad="
                                f"{int((rowbad & ~has_hot).sum())} "
                                f"max_abs={float(diff.max()):.6f}",
                                flush=True,
                            )

                    # Reverse-order backwards (framework autograd order).
                    out2.backward(steps[1]["dy"])
                    grads2 = _megamoe_function_grads(leaves2, ffn)
                    out1.backward(steps[0]["dy"])
                    grads1 = _megamoe_function_grads(leaves1, ffn)

                    for step_name, grads, step in (
                        ("step2", grads2, steps[1]),
                        ("step1", grads1, steps[0]),
                    ):
                        if save_fc1_dtype == "fp8":
                            # e4m3 cast boundaries move the activation a
                            # quantization step; the fixed GRAD_* bound of
                            # compare_backward_gradients does not fit a
                            # quantized-save gate
                            all_ok, details = _compare_grads_relaxed(
                                grads, step["golden"], rtol=1e-1, atol=5e-2
                            )
                        else:
                            all_ok, details = compare_backward_gradients(
                                grads, step["golden"]
                            )
                        if not all_ok:
                            print(
                                f"{label} {step_name} rank{rank} LOCAL-FAIL "
                                f"gradient details: {details}",
                                flush=True,
                            )
                        flag = torch.tensor(
                            [1 if all_ok else 0],
                            dtype=torch.int32,
                            device=device,
                        )
                        dist.all_reduce(
                            flag, op=dist.ReduceOp.MIN, group=ep_group
                        )
                        if not bool(flag.item()):
                            raise AssertionError(
                                f"{label}: {step_name} single-kernel MoonEP "
                                "forward + mega recompute backward grads "
                                "mismatched the SiTU eager golden"
                            )

                    if save_fc1_dtype == "fp8":
                        # the offload swap ran on BOTH steps — one D2H per
                        # forward, one H2D per backward (mirrors the
                        # fc1offload twin's assertion); counters are final
                        # behind the grad checks above
                        from mega_moe.ops._fc1_host_offload import (
                            fc1_offload_stats,
                        )
                        d2h_n, h2d_n, d2h_bytes = fc1_offload_stats()
                        if d2h_n < 2 or h2d_n < 2 or d2h_bytes <= 0:
                            raise AssertionError(
                                f"{label}: fc1 host offload did not run "
                                f"(d2h={d2h_n} h2d={h2d_n} "
                                f"bytes={d2h_bytes})"
                            )

                    # Post-lend recovery: the backwards invalidated the
                    # replica weight cache; a plain forward must re-push and
                    # still match the independent forward golden.
                    expected1 = torch_moe_fwd_golden(
                        steps[0]["hs"], steps[0]["rw"], steps[0]["ei"],
                        steps[0]["gate_w"], steps[0]["up_w"],
                        steps[0]["down_w"], num_experts, ep_group,
                    )
                    with torch.no_grad():
                        produced1, _ = op.forward(
                            steps[0]["hs"], steps[0]["ei"],
                            steps[0]["packed"], steps[0]["down_w"],
                            steps[0]["rw"], return_saved=True,
                        )
                    assert_close(
                        produced1, expected1,
                        rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL,
                    )
                finally:
                    op.finalize()
            finally:
                kit.ash.aclshmem_free_tensor(peer_mem)
    finally:
        for key, value in _env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_single_kernel_shared_op_interleave_case(
    rank: int, world_size: int
) -> None:
    """Shared-operator interleave: all forwards first, backwards in reverse.

    With ``ep_plan.megamoe_shared_op: true`` a framework host runs every
    same-shape MoE layer on ONE operator (one set of planning/mirror
    workspaces, one state): layer1 fwd, layer2 fwd, ..., layerN bwd, ...,
    layer1 bwd.  The second forward bumps the operator's routing
    generation BEFORE layer1's backward runs — the guard in
    MegaMoEFunction.backward must let that pass for the single-kernel
    snapshot (``_single_kernel_snapshot``), while the backward of each
    layer must still reproduce its own SiTU eager golden (the snapshot
    must not alias anything the later forward rewrote).  Two steps on one
    operator with DIFFERENT inputs mirror the two-layer interleaving; the
    shared state also carries the mega slabs/epoch across both backwards.
    """
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("single-kernel autograd requires NPU and ACLSHMEM")
    if MegaMoEFunction is None:
        raise RuntimeError("MegaMoEFunction is unavailable")

    _bwd_env = {"MOE_BWD_MEGA": "1", "MOE_SAVED_RECOMPUTE": "1"}
    _env_before = {k: os.environ.get(k) for k in _bwd_env}
    os.environ.update(_bwd_env)

    # small smoke shape (the interleave property is shape-independent; E<=32
    # per the adapter's scatter guard)
    tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 32
    situ_beta, situ_linear_beta = 4.0, 25.0
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD
    label = f"single-kernel-shared-op-interleave-w{world_size}"

    try:
        with kit.aclshmem_session(
            rank, world_size, kit.get_ash_size_bytes(default_gb=2)
        ):
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
                        activation="situglu",
                        situ_beta=situ_beta,
                        situ_linear_beta=situ_linear_beta,
                        enable_single_kernel_forward=True,
                        fc1_gemm_block_size_m=256,
                        fc2_combine_block_size_m=256,
                    ),
                )
                try:
                    w_gate, w_up = make_gate_up_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )
                    packed_w1 = pack_gate_up_weights(w_gate, w_up)
                    w2 = make_down_weights(
                        num_experts, hidden, ffn, world_size, rank, dtype,
                        device,
                    )

                    def make_step_inputs(seed):
                        hs, expert_indices = prepare_inputs(
                            tokens, hidden, num_experts, topk, dtype, device,
                            seed=seed + rank,
                        )
                        routing_weights = make_routing_weights(
                            tokens, topk, device, seed=seed + 100 + rank
                        )
                        torch.manual_seed(seed + 200 + rank)
                        dy = torch.randn(
                            tokens, hidden, dtype=dtype, device=device
                        )
                        return hs, routing_weights, expert_indices, dy

                    def make_golden(hs, routing_weights, expert_indices, dy):
                        # same SiTU eager recipe as the situglu autograd case
                        with torch.no_grad():
                            _, golden_saved = moe_forward(
                                hs, routing_weights, expert_indices, w_gate,
                                w_up, w2, ep_group, topk, return_saved=True,
                            )
                            gate = golden_saved["gate"].float()
                            up = golden_saved["up"].float()
                            situ_a = (
                                situ_beta
                                * torch.tanh(gate / situ_beta)
                                * torch.sigmoid(gate)
                            )
                            up_v = situ_linear_beta * torch.tanh(
                                up / situ_linear_beta
                            )
                            golden_saved["swiglu_out_weighted"] = (
                                situ_a
                                * up_v
                                * golden_saved["recv_weights_sorted"]
                                .float()
                                .unsqueeze(-1)
                            ).to(dtype)
                            golden_saved["activation"] = "situglu"
                            golden_saved["situ_beta"] = situ_beta
                            golden_saved["situ_linear_beta"] = situ_linear_beta
                            return backward_torch_baseline(golden_saved, dy)

                    steps = [
                        make_step_inputs(2401),
                        make_step_inputs(2402),
                    ]
                    goldens = [
                        make_golden(hs, rw, ei, dy)
                        for hs, rw, ei, dy in steps
                    ]

                    def apply_step(hs, rw, ei):
                        hidden_leaf = hs.clone().requires_grad_(True)
                        routing_leaf = rw.clone().requires_grad_(True)
                        gate_up_leaf = (
                            packed_w1.clone().requires_grad_(True)
                        )
                        down_leaf = w2.clone().requires_grad_(True)
                        output = MegaMoEFunction.apply(
                            op, hidden_leaf, routing_leaf, ei,
                            gate_up_leaf, down_leaf, peer_mem, state,
                        )
                        return output, (
                            hidden_leaf, routing_leaf, gate_up_leaf,
                            down_leaf,
                        )

                    state = SimpleNamespace(
                        signal_mem=None, epoch=0, mega_persistent={}
                    )

                    # all forwards first (the second bumps the routing
                    # generation before the first backward), backwards in
                    # reverse layer order afterwards
                    dist.barrier()
                    outs = [
                        apply_step(hs, rw, ei)
                        for hs, rw, ei, _dy in steps
                    ]
                    all_ok = True
                    details = ""
                    for idx in reversed(range(len(steps))):
                        out, leaves = outs[idx]
                        out.backward(steps[idx][3])
                        grads = _megamoe_function_grads(leaves, ffn)
                        ok, details = compare_backward_gradients(
                            grads, goldens[idx]
                        )
                        all_ok = all_ok and ok
                    if state.epoch < len(steps) or state.epoch < 1:
                        raise AssertionError(
                            f"{label}: shared state epoch did not advance "
                            f"through both backwards (epoch={state.epoch})"
                        )
                    if not getattr(state, "mega_persistent", {}):
                        raise AssertionError(
                            f"{label}: the mega backward did not persist its "
                            "slabs/epochs in the shared state"
                        )

                    flag = torch.tensor(
                        [1 if all_ok else 0], dtype=torch.int32, device=device
                    )
                    dist.all_reduce(
                        flag, op=dist.ReduceOp.MIN, group=ep_group
                    )
                    if rank == 0 and not bool(flag.item()):
                        print(
                            f"{label} gradient details: {details}", flush=True
                        )
                    if not bool(flag.item()):
                        raise AssertionError(
                            f"{label}: shared-operator interleaved backwards "
                            "mismatched their SiTU eager goldens"
                        )
                finally:
                    op.finalize()
            finally:
                kit.ash.aclshmem_free_tensor(peer_mem)
    finally:
        for key, value in _env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def test_single_kernel_forward_w2(dist_test):
    dist_test(run_single_kernel_forward_case, world_size=2)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_forward_w8(dist_test):
    dist_test(run_single_kernel_forward_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_forward_fp16_saved_w2(dist_test):
    dist_test(
        run_single_kernel_forward_case, world_size=2, args=(256, 64, 8, "fp16")
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize(
    "tokens,block_m", [(3, 256), (4097, 256), (4097, 128)],
    ids=("tiny-tail", "m256-multi-wave", "m128-multi-wave"),
)
def test_single_kernel_dynamic_waves_w8(dist_test, tokens, block_m):
    dist_test(
        run_single_kernel_forward_case, world_size=8,
        args=(block_m, tokens, 32),
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.slow
@pytest.mark.kimi
def test_single_kernel_kimi_k3_t4k_w8(dist_test):
    dist_test(run_single_kernel_kimi_k3_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_situglu_autograd_w2(dist_test):
    dist_test(run_single_kernel_situglu_autograd_case, world_size=2)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_situglu_autograd_fc1offload_w2(dist_test):
    dist_test(
        run_single_kernel_situglu_autograd_case, world_size=2, args=(True,)
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.kimi
def test_single_kernel_situglu_autograd_fc1offload_w8(dist_test):
    dist_test(
        run_single_kernel_situglu_autograd_case, world_size=8, args=(True,)
    )


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_situglu_autograd_w8(dist_test):
    dist_test(run_single_kernel_situglu_autograd_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_moonep_autograd_w2(dist_test):
    dist_test(run_single_kernel_moonep_autograd_case, world_size=2)


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.kimi
def test_single_kernel_moonep_autograd_w8(dist_test):
    dist_test(run_single_kernel_moonep_autograd_case, world_size=8)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_moonep_autograd_fp8_w2(dist_test):
    """fp8 saved-FC1 + FC1 host offload under the single-kernel moonep path.

    The production quantized config (!29/!59) pairs the e4m3 save with the
    host offload; this exercises both through the moonep adapter (FC1_FP8
    and MOONEP are independent constexprs in the forward launch, the scale
    save is moonep-agnostic, the mega backward dequantizes at load sites).
    The golden quantizes its gate/up to the same e4m3 point, so the relaxed
    compare isolates the moonep handling from quantization noise.
    """
    dist_test(
        run_single_kernel_moonep_autograd_case, world_size=2, args=("fp8",)
    )


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_shared_op_interleave_w2(dist_test):
    dist_test(run_single_kernel_shared_op_interleave_case, world_size=2)


@pytest.mark.dist
@pytest.mark.functional
def test_single_kernel_situglu_autograd_downdirect_w2(dist_test):
    dist_test(
        run_single_kernel_situglu_autograd_case, world_size=2,
        args=(False, True),
    )


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
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_native_saved_hot_expert(dist_test, world_size):
    dist_test(
        run_moonep_native_saved_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_native_backward_symmetric_hot_expert(dist_test, world_size):
    dist_test(
        run_moonep_native_backward_symmetric_hot_expert_case,
        world_size=world_size,
    )


@pytest.mark.dist
@pytest.mark.functional
@pytest.mark.parametrize("world_size", (2, 4))
def test_moonep_multilayer_pool_epoch(dist_test, world_size):
    if os.environ.get("MEGAMOE_REPLICA_POOL") != "1":
        pytest.skip("the pooled-table epoch case requires MEGAMOE_REPLICA_POOL=1")
    dist_test(
        run_moonep_multilayer_pool_epoch_case,
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


@pytest.mark.dist
@pytest.mark.functional
def test_megamoe_situglu_autograd(dist_test):
    if MegaMoEFunction is None:
        pytest.skip(
            "H3 API pending: mega_moe does not export MegaMoEFunction yet"
        )
    dist_test(run_megamoe_situglu_autograd_case, world_size=2)


# TODO: future work — when an all-directions session is introduced, finish and
# fully clean forward before starting backward; do not reuse operator, peer
# memory, or ACLSHMEM heap.  Each parameterized node currently owns an isolated
# lifecycle.
