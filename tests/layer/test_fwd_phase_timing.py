# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MOE_FWD_TIMING single-kernel phase stamps, one variant per pytest node.

Each node collects the SYS_CNT phase products (ts segments, busy parts, FC1
ring) for exactly ONE launch configuration — case x saved/unsaved x save
dtype — and writes ``phase_timing_<case>_<dtype>_<variant>.json`` under
``MOE_FWD_TIMING_OUT_DIR`` (default ``results/fwd_phase_timing``).  The
variants are separate nodes on purpose: comparisons are made across runs,
never by auto-running a bundle inside one node.

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

            # The env gate is set for this worker only, and restored even on
            # failure — the runner never exports MOE_FWD_TIMING by hand.
            previous_timing_env = os.environ.get("MOE_FWD_TIMING")
            os.environ["MOE_FWD_TIMING"] = "1"
            try:
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
