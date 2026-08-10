#!/usr/bin/env python3
"""Authoritative four-arm current-human baseline runner.

Host dry-run is dependency-free. Device modules are imported only after an
identity-bound external environment receipt has passed validation.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from baseline_contract import (
    ARM_IDS,
    ARM_OPERATIONS,
    BACKWARD_GRADIENTS,
    BIGOP_COMMIT,
    CONTRACT_VERSION,
    ContractError,
    EP_WORLD_SIZE,
    FIXTURE_SEED,
    MODEL,
    REPOSITORY_COMMIT,
    REPOSITORY_TREE,
    REPOSITORY_URL,
    SAMPLES,
    TOKENS_PER_RANK,
    WARMUP,
    build_plan,
    envelope,
    read_json,
    validate_dry_run_envelope,
    validate_environment,
    validate_execution_envelope,
    validate_plan,
    write_envelope,
)
from providers import current_main, grouped_hccl


AUTHORITATIVE_BASELINE_RUNNER = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HARNESS_PATHS = {
    "docs/design/HARNESS_DESIGN_PHILOSOPHY.md",
    "benchmark/contracts/current_human_baseline_v1.schema.json",
    "benchmark/baseline_contract.py",
    "benchmark/current_human_baseline.py",
    "benchmark/providers/current_main.py",
    "benchmark/providers/grouped_hccl.py",
    "scripts/architecture_lint.py",
    "tests/host/test_architecture_contract.py",
    "tests/host/test_baseline_contract.py",
    "tests/host/test_baseline_runner.py",
}
DEVICE_MODULE_ROOTS = ("torch", "torch_npu", "triton", "shmem")


def _provider_descriptions() -> list[dict[str, Any]]:
    return [grouped_hccl.describe(), current_main.describe()]


def _plan() -> dict[str, Any]:
    return build_plan(_provider_descriptions())


def _validate_receipt_dir(receipt_dir: Path) -> Path:
    resolved = receipt_dir.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return resolved
    raise ContractError("receipt directory must be outside the source checkout")


def _device_modules_loaded() -> list[str]:
    loaded = []
    for name in DEVICE_MODULE_ROOTS:
        if name in sys.modules or any(module.startswith(name + ".") for module in sys.modules):
            loaded.append(name)
    return loaded


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(f"git identity command failed: {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_source_checkout() -> dict[str, Any]:
    _require_equal(_git("rev-parse", f"{REPOSITORY_COMMIT}^{{tree}}"), REPOSITORY_TREE, "source tree")
    _require_equal(_git("remote", "get-url", "origin"), REPOSITORY_URL, "origin URL")
    changed = {
        line
        for line in _git("diff", "--name-only", REPOSITORY_COMMIT, "HEAD").splitlines()
        if line
    }
    if changed != ALLOWED_HARNESS_PATHS:
        raise ContractError(
            "checkout differs from the fixed product commit outside the exact harness scope: "
            f"{sorted(changed ^ ALLOWED_HARNESS_PATHS)}"
        )
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ContractError("execution checkout is dirty")
    gitlink = _git("rev-parse", f"{REPOSITORY_COMMIT}:3rdparty/bigop")
    _require_equal(gitlink, BIGOP_COMMIT, "bigop gitlink")
    return {
        "harness_commit": _git("rev-parse", "HEAD"),
        "harness_tree": _git("rev-parse", "HEAD^{tree}"),
        "product_commit": REPOSITORY_COMMIT,
        "product_tree": REPOSITORY_TREE,
        "changed_paths": sorted(changed),
    }


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _collect_rank_max_samples(
    function: Callable[[], Any], *, torch, dist, ep_group
) -> list[float]:
    for _ in range(WARMUP):
        dist.barrier(group=ep_group)
        function()
        torch.npu.synchronize()
    samples: list[float] = []
    rank = dist.get_rank(group=ep_group)
    for _ in range(SAMPLES):
        dist.barrier(group=ep_group)
        torch.npu.synchronize()
        started = time.perf_counter()
        function()
        torch.npu.synchronize()
        local_ms = (time.perf_counter() - started) * 1000.0
        reduced = torch.tensor([local_ms], dtype=torch.float64, device=f"npu:{rank}")
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX, group=ep_group)
        samples.append(float(reduced.item()))
    return samples


def _collective_assert_close(actual, expected, *, torch, dist, ep_group, label: str) -> None:
    local_ok = True
    message = ""
    try:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=5e-2, atol=5e-2)
    except AssertionError as error:
        local_ok = False
        message = str(error).splitlines()[0]
    rank = dist.get_rank(group=ep_group)
    flag = torch.tensor([int(local_ok)], dtype=torch.int32, device=f"npu:{rank}")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep_group)
    if not bool(flag.item()):
        raise ContractError(f"{label} correctness failed: {message}")


def _forward_fixture(tokens: int, rank: int, device: str, ep_group, ff, mega_moe):
    ff._activate_model_profile("QWEN")
    expected = (ff.HIDDEN, ff.FFN_DIM, ff.TOPK, ff.NUM_EXPERTS)
    _require_equal(expected, (MODEL["hidden"], MODEL["ffn"], MODEL["topk"], MODEL["num_experts"]), "forward model")
    experts_per_rank = MODEL["num_experts"] // EP_WORLD_SIZE
    packed_w1, down_weight, torch_w2_kn = ff._make_local_weights(
        experts_per_rank, rank, device, seed=FIXTURE_SEED
    )
    hidden_states, selected_experts, routing_weights = ff._prepare_inputs(
        tokens, rank, device, seed=FIXTURE_SEED
    )
    config = mega_moe.MoEForwardConfig(receive_capacity_factor=1.25)
    op = mega_moe.FusedMoEForward(
        ep_group,
        max_tokens_per_rank=tokens,
        hidden_size=MODEL["hidden"],
        top_k=MODEL["topk"],
        num_experts=MODEL["num_experts"],
        config=config,
    )
    return {
        "op": op,
        "hidden_states": hidden_states,
        "selected_experts": selected_experts,
        "routing_weights": routing_weights,
        "packed_w1": packed_w1,
        "down_weight": down_weight,
        "torch_w2_kn": torch_w2_kn,
        "experts_per_rank": experts_per_rank,
        "ep_group": ep_group,
    }


def _backward_fixture(tokens: int, rank: int, ep_group, bb, utils):
    saved, dy, dtype, _device = bb.build_backward_saved(
        tokens,
        MODEL["hidden"],
        MODEL["ffn"],
        MODEL["num_experts"],
        MODEL["topk"],
        ep_group,
        seed=FIXTURE_SEED,
    )
    return {
        "saved": saved,
        "dy": dy,
        "peer_mem": utils.make_peer_mem(saved, dtype, rank),
    }


def _execute(environment: Mapping[str, Any], receipt_dir: Path) -> Path | None:
    validate_environment(environment)
    receipt_dir = _validate_receipt_dir(receipt_dir)
    harness_identity = _verify_source_checkout()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    importlib.import_module("torch_npu")
    ash = importlib.import_module("shmem")
    mega_moe = importlib.import_module("mega_moe")
    ff = importlib.import_module("benchmark.layer.bench_full_forward")
    bb = importlib.import_module("benchmark.layer.bench_backward")
    utils = importlib.import_module("tests._moe_dist_utils")
    golden_module = importlib.import_module("mega_moe._goldens.backward")

    created_process_group = not dist.is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.npu.set_device(local_rank)
    if created_process_group:
        dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _require_equal(world_size, EP_WORLD_SIZE, "expert-parallel world size")
    ep_group = dist.group.WORLD
    utils.init_aclshmem(rank, world_size, utils.get_ash_size_bytes(default_gb=4))

    results = {
        arm_id: {"arm_id": arm_id, "operation": ARM_OPERATIONS[arm_id], "shapes": []}
        for arm_id in ARM_IDS
    }
    try:
        for tokens in TOKENS_PER_RANK:
            forward = _forward_fixture(tokens, rank, f"npu:{local_rank}", ep_group, ff, mega_moe)
            try:
                grouped_output = grouped_hccl.execute_forward(forward)
                fused_output = current_main.execute_forward(forward)
                _collective_assert_close(
                    fused_output,
                    grouped_output,
                    torch=torch,
                    dist=dist,
                    ep_group=ep_group,
                    label=f"forward tokens={tokens}",
                )
                grouped_samples = _collect_rank_max_samples(
                    lambda: grouped_hccl.execute_forward(forward),
                    torch=torch,
                    dist=dist,
                    ep_group=ep_group,
                )
                fused_samples = _collect_rank_max_samples(
                    lambda: current_main.execute_forward(forward),
                    torch=torch,
                    dist=dist,
                    ep_group=ep_group,
                )
                results[ARM_IDS[0]]["shapes"].append(
                    {
                        "tokens_per_rank": tokens,
                        "correctness": {"forward_output": "PASS"},
                        "samples_ms": grouped_samples,
                    }
                )
                results[ARM_IDS[1]]["shapes"].append(
                    {
                        "tokens_per_rank": tokens,
                        "correctness": {"forward_output": "PASS"},
                        "samples_ms": fused_samples,
                    }
                )
            finally:
                forward["op"].finalize()

            backward = _backward_fixture(tokens, rank, ep_group, bb, utils)
            try:
                golden = golden_module.moe_backward_torch(backward["saved"], backward["dy"])
                default_output = current_main.execute_backward(backward, use_triton_wgrad=False)
                triton_output = current_main.execute_backward(backward, use_triton_wgrad=True)
                for gradient in BACKWARD_GRADIENTS:
                    _collective_assert_close(
                        default_output[gradient],
                        golden[gradient],
                        torch=torch,
                        dist=dist,
                        ep_group=ep_group,
                        label=f"default {gradient} tokens={tokens}",
                    )
                    _collective_assert_close(
                        triton_output[gradient],
                        golden[gradient],
                        torch=torch,
                        dist=dist,
                        ep_group=ep_group,
                        label=f"triton {gradient} tokens={tokens}",
                    )
                default_samples = _collect_rank_max_samples(
                    lambda: current_main.execute_backward(backward, use_triton_wgrad=False),
                    torch=torch,
                    dist=dist,
                    ep_group=ep_group,
                )
                triton_samples = _collect_rank_max_samples(
                    lambda: current_main.execute_backward(backward, use_triton_wgrad=True),
                    torch=torch,
                    dist=dist,
                    ep_group=ep_group,
                )
                correctness = {name: "PASS" for name in BACKWARD_GRADIENTS}
                results[ARM_IDS[2]]["shapes"].append(
                    {
                        "tokens_per_rank": tokens,
                        "correctness": dict(correctness),
                        "samples_ms": default_samples,
                    }
                )
                results[ARM_IDS[3]]["shapes"].append(
                    {
                        "tokens_per_rank": tokens,
                        "correctness": dict(correctness),
                        "samples_ms": triton_samples,
                    }
                )
            finally:
                ash.aclshmem_free_tensor(backward["peer_mem"])
    finally:
        ash.aclshmem_finalize()
        if created_process_group:
            dist.destroy_process_group()

    if rank != 0:
        return None
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "COMPLETE",
        "plan": _plan(),
        "environment": dict(environment),
        "harness_identity": harness_identity,
        "arms": [results[arm_id] for arm_id in ARM_IDS],
    }
    value = envelope(payload)
    validate_execution_envelope(value)
    write_envelope(receipt_dir / "current_human_baseline_execution.json", payload)
    return receipt_dir / "current_human_baseline_execution.json"


def _dry_run(receipt_dir: Path) -> Path:
    receipt_dir = _validate_receipt_dir(receipt_dir)
    plan = _plan()
    validate_plan(plan)
    loaded = _device_modules_loaded()
    if loaded:
        raise ContractError(f"host dry-run imported device modules: {loaded}")
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "DRY_RUN_VALIDATED",
        "plan": plan,
        "device_modules_loaded": loaded,
    }
    path = receipt_dir / "current_human_baseline_dry_run.json"
    validate_dry_run_envelope(envelope(payload))
    write_envelope(path, payload)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            path = _dry_run(args.receipt_dir)
        else:
            if args.environment_receipt is None:
                raise ContractError("--environment-receipt is required for --execute")
            environment = read_json(args.environment_receipt)
            path = _execute(environment, args.receipt_dir)
        if path is not None:
            print(path)
        return 0
    except ContractError as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
