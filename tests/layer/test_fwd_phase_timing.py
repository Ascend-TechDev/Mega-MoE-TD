# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MOE_FWD_TIMING single-kernel phase stamps, one variant per pytest node.

Each node collects the SYS_CNT phase products (ts segments, busy parts, FC1
ring) for exactly ONE launch configuration — case x saved/unsaved x save
dtype — and writes ``phase_timing_<case>_<dtype>_<variant>.json`` under
``MOE_FWD_TIMING_OUT_DIR`` (default ``results/fwd_phase_timing``).  The
variants are separate nodes on purpose: comparisons are made across runs,
never by auto-running a bundle inside one node.

Every node also gates its configuration on accuracy BEFORE the timed loop:
the same independent logical-owner Torch/HCCL golden the benchmark suite
uses, plus the saved contract's route-table tripwires for the saved
variants.  A timing number is only meaningful for a configuration that
computes the right answer, and the recorded ``correctness`` block says
whether the gate ran.  ``MOE_FWD_TIMING_SKIP_CORRECTNESS=1`` skips the
comparison (recorded as ``"skipped"``) for numbers-only runs.

``MOE_FWD_TIMING`` itself stays an environment gate (the MOE_MEGA_TIMING
precedent); the worker sets and restores it internally, so the runner never
exports it by hand.  Block sizes are the config defaults (FC1/FC2 M=256,
N=256, K=128) — the tiling the UB accounting was measured against.

Usage examples::

    # one node: fp16 save, saved launch, trimmed w8 t4k
    pytest tests/layer/test_fwd_phase_timing.py \\
        -k "phase_timing and fp16 and saved and trimmed-w8-t4k"

    # every unsaved node at the small token counts
    pytest tests/layer/test_fwd_phase_timing.py -k "unsaved and t4k"

    MOE_FUSED_ASH_SIZE_GB=6 MOE_FWD_TIMING_OUT_DIR=/tmp/stamps \\
        pytest tests/layer/test_fwd_phase_timing.py -k "w4-t16k and fp8"
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from benchmark.layer import bench_moe_suite as bench_module
from benchmark.layer._fwd_phase_timing import (
    FWD_ACC_SLOTS,
    FWD_TS_SLOTS,
    _stats,
    annotate_us,
    calibrate_ticks_per_us,
    summarize_acc,
    summarize_fc2_waves,
    summarize_ring,
    summarize_ts_stamps,
)
from config import CaseSpec, select_cases
from mega_moe import FusedMoEForward, MoEForwardConfig
from mega_moe.kernels import fused_forward as fused_forward_module
from tests import _moe_testkit as kit

_REPO_ROOT = Path(__file__).resolve().parents[2]
KIMI_FORWARD_CASES = kit.make_pytest_params(
    select_cases(direction="forward", tags={"performance", "kimi"})
)


def _assert_phase_timing_tables():
    """The kernel and the host reduction must agree on the slot tables."""
    from benchmark.layer import _fwd_phase_timing as table

    if fused_forward_module.FWD_TS_SLOTS != table.FWD_TS_SLOTS:
        raise AssertionError("FWD_TS_SLOTS drifted between kernel and host table")
    if fused_forward_module.FWD_ACC_SLOTS != table.FWD_ACC_SLOTS:
        raise AssertionError("FWD_ACC_SLOTS drifted between kernel and host table")
    if int(fused_forward_module.FWD_RING_COLS) != table.FWD_RING_COLS:
        raise AssertionError("FWD_RING_COLS drifted between kernel and host table")
    for groups, experts in ((1, 1), (5, 16), (64, 8)):
        if (fused_forward_module.fwd_ring_slots(groups, experts)
                != table.fwd_ring_slots(groups, experts)):
            raise AssertionError(
                "ring sizing formula drifted between kernel and host table")


def _route_table_errors(op, saved, selected_experts, num_experts, topk):
    """Saved-contract tripwires for the route tables the backward consumes.

    These are the invariants ``ops/_single_saved_adapter`` checks before it
    hands the tables to the mega backward: a true permutation of
    ``0..total_send-1``, bucket-segmented (flat expert ids non-decreasing),
    ``send_token_indices == send_route_indices // topk``, ``route_to_send``
    the exact inverse, and the receive sentinel equal to ``total_recv``.  A
    scatter regression shows up here even when the numeric output stays
    inside the comparison tolerance.
    """
    errors = []
    flat = selected_experts.reshape(-1).to(torch.int64)
    total_send = int(saved["total_send"])
    send_route = saved["send_route_indices"].to(torch.int64)
    if int(send_route.numel()) < total_send:
        return [f"send_route_indices has {int(send_route.numel())} entries "
                f"for total_send={total_send}"]
    send_route = send_route[:total_send]
    if not torch.equal(
        torch.sort(send_route).values,
        torch.arange(total_send, dtype=torch.int64, device=send_route.device),
    ):
        return ["send_route_indices is not a permutation of 0..total_send-1"]
    expert_seq = flat[send_route]
    if total_send > 1 and bool((expert_seq[1:] < expert_seq[:-1]).any()):
        errors.append("send_route_indices is not bucket-segmented "
                      "(flat expert ids decrease along the table)")
    tokens = saved["send_token_indices"][:total_send].to(torch.int64)
    if not torch.equal(tokens, send_route // topk):
        errors.append("send_token_indices != send_route_indices // topk")
    inverse = op._route_to_send[:flat.numel()].to(torch.int64)
    expected_inverse = torch.full_like(flat, -1)
    expected_inverse[send_route] = torch.arange(
        total_send, dtype=torch.int64, device=flat.device)
    if not torch.equal(inverse, expected_inverse):
        errors.append("route_to_send is not the inverse of send_route_indices")
    valid = (flat >= 0) & (flat < num_experts)
    if int(valid.sum()) != total_send:
        errors.append(f"route table holds {total_send} of "
                      f"{int(valid.sum())} valid routes")
    elif not torch.equal(
        send_route,
        torch.argsort(flat.to(torch.float32), stable=True),
    ):
        errors.append("send_route_indices is not the stable expert-major order")
    if int(saved["recv_expert_offsets"][-1].item()) != int(saved["total_recv"]):
        errors.append("recv_expert_offsets[-1] != total_recv")
    return errors


def _error_metrics(actual, expected, rtol, atol, rows=1024):
    """max|diff| and tolerance violations, without a full-size fp32 copy."""
    max_abs = 0.0
    violations = 0
    for start in range(0, int(actual.shape[0]), rows):
        block = actual[start:start + rows].float()
        reference = expected[start:start + rows].float()
        diff = (block - reference).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        violations += int(
            (diff > (atol + rtol * reference.abs())).sum().item())
        del block, reference, diff
    return {"max_abs_diff": max_abs, "tolerance_violations": violations}


def _accuracy_gate(rank, case, op, ep_group, device, saved, hidden_states,
                   selected_experts, routing_weights, packed_w1, down_weight,
                   golden, label):
    """Independent golden + saved tripwires for one launch configuration.

    ``golden`` is computed once by the caller and shared by both gates (the
    production TIMING=0 build and the measured TIMING=1 build); this function
    never frees it.  The saved variant additionally checks the route-table
    tripwires the backward contract depends on.
    """
    with torch.no_grad():
        produced = op.forward(
            hidden_states, selected_experts, packed_w1, down_weight,
            routing_weights, return_saved=saved)
    actual, saved_dict = produced if saved else (produced, None)

    # Capacity first: an over-capacity launch computes nothing (the in-kernel
    # capacity flag zeroes the whole pipeline), so its timings would look
    # great and mean nothing.  Report that instead of a numeric mismatch.
    max_received_routes = math.ceil(
        case.tokens * case.topk * case.capacity_factor)
    required = int(op.context.metadata_stats[1].item())
    if required > max_received_routes:
        raise AssertionError(
            f"[rank {rank}] {case.case_id} needs {required} receive rows but "
            f"the capacity is {max_received_routes}: the single-kernel launch "
            "skipped its whole wave pipeline, so both its output and its "
            "phase timings are meaningless.  Raise receive_capacity_factor."
        )

    metrics = _error_metrics(actual, golden, rtol=5e-2, atol=5e-2)
    bench_module._assert_close_collective(
        actual, golden, device, f"phase-timing-{case.case_id}-{label}",
        ep_group)

    table_errors = []
    total_recv = required
    if saved:
        table_errors = _route_table_errors(
            op, saved_dict, selected_experts, case.num_experts, case.topk)
        if table_errors:
            shown = "; ".join(table_errors[:4])
            raise AssertionError(
                f"[rank {rank}] {case.case_id} ({label}) saved route tables "
                f"failed the contract ({len(table_errors)} errors): {shown}")
        total_recv = int(saved_dict["total_recv"])

    del produced, actual, saved_dict
    return {
        "status": "passed",
        "build": label,
        "baseline": "distributed logical Torch owner-expert golden",
        "rtol": 5e-2,
        "atol": 5e-2,
        "max_abs_diff": metrics["max_abs_diff"],
        "tolerance_violations": metrics["tolerance_violations"],
        "route_table_errors": table_errors,
        "max_receive_rows": required,
        "receive_capacity_rows": max_received_routes,
        "total_recv": total_recv,
    }


def run_fwd_phase_timing_case(
    rank: int,
    world_size: int,
    case: CaseSpec,
    saved: bool,
    save_fc1_dtype: str,
    warmup: int,
    samples: int,
) -> None:
    """Collect the SYS_CNT phase products for one launch configuration."""
    if kit.ash is None or kit.torch_npu is None:
        raise RuntimeError("phase timing requires NPU and ACLSHMEM")
    if case.world_size != world_size:
        raise ValueError(
            f"case {case.case_id} needs world_size={case.world_size}, "
            f"got {world_size}"
        )
    _assert_phase_timing_tables()

    heap_size = kit.get_ash_size_bytes(default_gb=2)
    required_heap = bench_module._required_ash_bytes(case, world_size)
    if required_heap >= heap_size:
        raise RuntimeError(
            f"phase timing for {case.case_id} needs more ACLSHMEM heap: "
            f"required={required_heap}, configured={heap_size} "
            "(raise MOE_FUSED_ASH_SIZE_GB)"
        )

    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    with kit.aclshmem_session(rank, world_size, heap_size):
        op = FusedMoEForward(
            ep_group,
            max_tokens_per_rank=case.tokens,
            hidden_size=case.hidden,
            top_k=case.topk,
            num_experts=case.num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=case.capacity_factor,
                enable_single_kernel_forward=True,
                save_fc1_dtype=save_fc1_dtype,
            ),
        )
        try:
            experts_per_rank = case.num_experts // world_size
            packed_w1, down_weight, _ = bench_module._make_local_weights(
                case, experts_per_rank, rank, device)
            hidden_states, selected_experts, routing_weights = (
                bench_module._prepare_inputs(case, rank, device)
            )
            dist.barrier(group=ep_group)

            # Accuracy gates.  Gate A checks the production build (TIMING=0);
            # gate B, inside the env block below, re-checks the exact build the
            # numbers are read from — TIMING=1 has its own UB budget and its
            # own history of measurement-only defects (README 2026-09-17), so
            # the two builds are gated separately.  The golden is computed
            # once and shared.
            skip_correctness = (
                os.environ.get("MOE_FWD_TIMING_SKIP_CORRECTNESS", "0") == "1")
            golden = None
            correctness = {
                "status": "skipped",
                "reason": "MOE_FWD_TIMING_SKIP_CORRECTNESS=1",
            }
            if not skip_correctness:
                golden = bench_module._logical_torch_golden(
                    case, ep_group, hidden_states, selected_experts,
                    routing_weights, packed_w1, down_weight)
                correctness = _accuracy_gate(
                    rank, case, op, ep_group, device, saved, hidden_states,
                    selected_experts, routing_weights, packed_w1, down_weight,
                    golden, label="production")
                dist.barrier(group=ep_group)

            # The env gate is set for this worker only, and restored even on
            # failure — the runner never exports MOE_FWD_TIMING by hand.
            previous_timing_env = os.environ.get("MOE_FWD_TIMING")
            os.environ["MOE_FWD_TIMING"] = "1"
            try:
                if golden is not None:
                    correctness_timed = _accuracy_gate(
                        rank, case, op, ep_group, device, saved,
                        hidden_states, selected_experts, routing_weights,
                        packed_w1, down_weight, golden, label="timing")
                    dist.barrier(group=ep_group)
                else:
                    correctness_timed = correctness
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                stream = torch.npu.current_stream(device)
                products = {"ts": [], "acc": [], "ring": [], "fc2w": []}
                elapsed_ms = []
                for iteration in range(warmup + samples):
                    torch.npu.synchronize(device)
                    start.record(stream)
                    output = op.forward(
                        hidden_states,
                        selected_experts,
                        packed_w1,
                        down_weight,
                        routing_weights,
                        return_saved=saved,
                    )
                    end.record(stream)
                    torch.npu.synchronize(device)
                    del output
                    if iteration >= warmup:
                        elapsed_ms.append(start.elapsed_time(end))
                        last = op.read_last_forward_phase_timing()
                        for name, tensor in last.items():
                            products[name].append(
                                tensor.cpu().numpy().copy())
            finally:
                if previous_timing_env is None:
                    os.environ.pop("MOE_FWD_TIMING", None)
                else:
                    os.environ["MOE_FWD_TIMING"] = previous_timing_env
                # Release the shared golden before the reduction/JSON work so
                # its route-major temporaries never outlive their last use.
                del golden
                torch.npu.empty_cache()

            # Stamp contract, asserted on every rank.  Monotonicity is
            # PER CORE (each program's own clock): core-to-core SYS_CNT
            # offsets are unrelated and only enter the host reduction as
            # per-column max/min.  A per-core regression is a real
            # finding, so the failure names the core and slot.
            cores = op.num_aicore_programs
            contract_errors = []
            for sample_index, (ts, acc, ring) in enumerate(zip(
                products["ts"], products["acc"], products["ring"]
            )):
                if tuple(ts.shape) != (cores, FWD_TS_SLOTS):
                    contract_errors.append(
                        f"sample {sample_index}: ts shape {ts.shape}")
                    continue
                if not bool((ts > 0).all()):
                    contract_errors.append(
                        f"sample {sample_index}: zero ts stamp")
                for core_id in range(cores):
                    row = ts[core_id]
                    for slot in range(1, FWD_TS_SLOTS):
                        if row[slot] < row[slot - 1]:
                            contract_errors.append(
                                f"sample {sample_index}: core {core_id} "
                                f"stamp {slot} regressed "
                                f"({row[slot]} < {row[slot - 1]})")
                if tuple(acc.shape) != (cores * 3, FWD_ACC_SLOTS):
                    contract_errors.append(
                        f"sample {sample_index}: acc shape {acc.shape}")
                elif not bool((acc >= 0).all()):
                    contract_errors.append(
                        f"sample {sample_index}: negative acc ticks")
                if tuple(ring.shape) != (
                    cores * 3, op._fwd_ring_slots, int(
                        fused_forward_module.FWD_RING_COLS)
                ):
                    contract_errors.append(
                        f"sample {sample_index}: ring shape {ring.shape}")
                elif not bool((ring >= 0).all()):
                    contract_errors.append(
                        f"sample {sample_index}: negative ring ticks")
                fc2w = products["fc2w"][sample_index]
                if tuple(fc2w.shape) != (cores, op._single_pipeline_max_groups):
                    contract_errors.append(
                        f"sample {sample_index}: fc2w shape {fc2w.shape}")
                elif not bool((fc2w >= 0).all()):
                    contract_errors.append(
                        f"sample {sample_index}: negative fc2w ticks")
            save_column = int(fused_forward_module.FWD_RING_SAVE)
            save_sum = int(
                products["ring"][-1][:, :, save_column].sum().item()
            ) if products["ring"] else 0
            if saved:
                total_recv = int(op.context.metadata_stats[0].item())
                if total_recv > 0 and save_sum <= 0:
                    contract_errors.append(
                        f"saved run wrote no ring save ticks ({save_sum})")
            elif save_sum != 0:
                contract_errors.append(
                    f"unsaved run has ring save ticks ({save_sum})")
            if contract_errors:
                shown = "; ".join(contract_errors[:8])
                if len(contract_errors) > 8:
                    shown += f"; ... ({len(contract_errors) - 8} more)"
                raise AssertionError(
                    f"[rank {rank}] MOE_FWD_TIMING stamp contract violated "
                    f"({len(contract_errors)} errors): {shown}"
                )

            if rank == 0:
                calibration_pairs = [
                    (max(row[-1] for row in ts) - min(row[0] for row in ts),
                     event_ms * 1000.0)
                    for ts, event_ms in zip(products["ts"], elapsed_ms)
                ]
                ticks_per_us = calibrate_ticks_per_us(calibration_pairs)
                summary = summarize_ts_stamps(products["ts"])
                fc2_waves = summarize_fc2_waves(products["fc2w"])
                result = {
                    "schema_version": 1,
                    "case_id": case.case_id,
                    "world_size": world_size,
                    "variant": "saved" if saved else "unsaved",
                    "save_fc1_dtype": save_fc1_dtype,
                    "warmup": warmup,
                    "samples": samples,
                    "calibration": {
                        "sys_cnt_ticks_per_us": ticks_per_us,
                        "pairs": len(calibration_pairs),
                    },
                    "segments": annotate_us(summary["segments"], ticks_per_us),
                    "correctness": {
                        "production_build": correctness,
                        "timed_build": correctness_timed,
                    },
                    "per_core_total": summary["per_core_total"],
                    "busy_parts": annotate_us(
                        summarize_acc(products["acc"]), ticks_per_us),
                    "fc1_ring": annotate_us(
                        summarize_ring(products["ring"]), ticks_per_us),
                    "fc2_waves": {
                        "waves": fc2_waves["waves"],
                        "wall_p50_us": (
                            fc2_waves["wall"]["p50"] / ticks_per_us),
                        "residual_p50_us": (
                            fc2_waves["residual"]["p50"] / ticks_per_us),
                        "per_wave_wall_us": [
                            value / ticks_per_us
                            for value in fc2_waves["per_wave_wall"]
                        ],
                        "per_wave_residual_us": [
                            value / ticks_per_us
                            for value in fc2_waves["per_wave_residual"]
                        ],
                    },
                    "e2e_event_ms": _stats(elapsed_ms),
                }
                out_dir = Path(os.environ.get(
                    "MOE_FWD_TIMING_OUT_DIR",
                    str(_REPO_ROOT / "results" / "fwd_phase_timing"),
                ))
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = (
                    out_dir
                    / f"phase_timing_{case.case_id}_{save_fc1_dtype}"
                      f"_{'saved' if saved else 'unsaved'}.json"
                )
                out_path.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(f"[phase-timing-result] {out_path}", flush=True)
                print(
                    f"[phase-timing] {case.case_id} {save_fc1_dtype} "
                    f"{'saved' if saved else 'unsaved'}: "
                    f"e2e p50 {result['e2e_event_ms']['p50']:.3f} ms, "
                    f"wave_pipeline p50 "
                    f"{result['segments']['wave_pipeline']['us']['p50']:.2f} us, "
                    f"fc2_wave_wall p50 "
                    f"{result['fc2_waves']['wall_p50_us']:.2f} us "
                    f"({result['fc2_waves']['waves']} waves)",
                    flush=True,
                )
            dist.barrier(group=ep_group)
        finally:
            torch.npu.synchronize(device)
            dist.barrier(group=ep_group)
            op.finalize()
            torch.npu.empty_cache()
            dist.barrier(group=ep_group)


@pytest.mark.dist
@pytest.mark.performance
@pytest.mark.parametrize("warmup,samples", ((5, 20),), ids=("w5s20",))
@pytest.mark.parametrize("save_fc1_dtype", ("fp8", "fp16"))
@pytest.mark.parametrize("saved", (False, True), ids=("unsaved", "saved"))
@pytest.mark.parametrize("case", KIMI_FORWARD_CASES)
def test_single_kernel_phase_timing(
    dist_test, case: CaseSpec, saved: bool, save_fc1_dtype: str,
    warmup: int, samples: int,
):
    dist_test(
        run_fwd_phase_timing_case,
        world_size=case.world_size,
        args=(case, saved, save_fc1_dtype, warmup, samples),
    )
