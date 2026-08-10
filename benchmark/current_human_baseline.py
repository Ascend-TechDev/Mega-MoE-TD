#!/usr/bin/env python3
"""Authoritative four-arm current-human baseline runner.

Host dry-run is dependency-free. Device modules are imported only after an
identity-bound external environment receipt has passed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
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
    ROUTING_ENVIRONMENT,
    SAMPLES,
    SIDECAR_BINDING,
    TOKENS_PER_RANK,
    WARMUP,
    build_plan,
    compare_environment,
    envelope,
    read_json,
    read_verified_envelope,
    recompute_trusted_checkout,
    validate_dry_run_envelope,
    validate_environment,
    validate_execution_envelope,
    validate_plan,
    write_envelope,
)
from providers import current_main, grouped_hccl


AUTHORITATIVE_BASELINE_RUNNER = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_MODULE_ROOTS = ("torch", "torch_npu", "triton", "shmem")
_CANN_VERSION_FILE_NAMES = {"version.info", "ascend_toolkit_install.info"}


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


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"runtime identity file unreadable: {path}: {error}") from error
    return digest.hexdigest()


def _runtime_file_identity(path: Path, version: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError(f"runtime identity source is unavailable: {path}: {error}") from error
    if not resolved.is_file():
        raise ContractError(f"runtime identity source is not a file: {resolved}")
    return {
        "version": version,
        "source": {"kind": "runtime_file", "locator": str(resolved)},
        "sha256": _sha256_file(resolved),
    }


def _module_version(module: Any, distribution: str) -> str:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        source = getattr(module, "__file__", None)
        if not source:
            raise ContractError(f"runtime component {distribution} has no version or source")
        return "sha256:" + _sha256_file(Path(source).resolve())


def _module_identity(module: Any, distribution: str) -> dict[str, Any]:
    source = getattr(module, "__file__", None)
    if not source:
        raise ContractError(f"runtime component {distribution} has no source file")
    return _runtime_file_identity(Path(source), _module_version(module, distribution))


def _import_required_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise ContractError(f"runtime component import failed: {name}: {error}") from error


def _cann_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    source = expected.get("source")
    if not isinstance(source, Mapping) or source.get("kind") != "runtime_file":
        raise ContractError("CANN recoverable source must be a runtime version file")
    try:
        locator = Path(str(source.get("locator", ""))).resolve(strict=True)
        roots = [
            Path(value).resolve(strict=True)
            for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME")
            if (value := os.environ.get(variable))
        ]
    except OSError as error:
        raise ContractError(f"CANN runtime identity path is unavailable: {error}") from error
    if not roots:
        raise ContractError("CANN runtime root identity is unavailable")
    if locator.name not in _CANN_VERSION_FILE_NAMES or not any(
        locator.is_relative_to(root) for root in roots
    ):
        raise ContractError("CANN version source is outside the active runtime root")
    try:
        text = locator.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"CANN version source is unreadable: {error}") from error
    matches = re.findall(r"[0-9]+(?:\.[0-9A-Za-z_-]+)+", text)
    if not matches:
        raise ContractError("CANN version source has no recoverable version")
    return _runtime_file_identity(locator, matches[0])


def _capture_runtime_environment(
    expected: Mapping[str, Any], *, torch: Any, torch_npu: Any, triton: Any, ash: Any
) -> dict[str, Any]:
    # The product-base gitlink is immutable and separately bound by the plan.
    gitlink = BIGOP_COMMIT
    components = {
        "python": _runtime_file_identity(Path(sys.executable), platform.python_version()),
        "torch": _module_identity(torch, "torch"),
        "torch_npu": _module_identity(torch_npu, "torch-npu"),
        "triton": _module_identity(triton, "triton"),
        "cann": _cann_identity(expected["components"]["cann"]),
        "aclshmem": _module_identity(ash, "aclshmem"),
        "bigop": {
            "version": gitlink,
            "source": {"kind": "gitlink", "locator": "3rdparty/bigop"},
            "sha256": hashlib.sha256(gitlink.encode("ascii")).hexdigest(),
        },
    }
    return {
        "repository": dict(expected["repository"]),
        "variables": dict(ROUTING_ENVIRONMENT),
        "components": components,
    }


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
    ambient_routing = {name: os.environ.get(name) for name in ROUTING_ENVIRONMENT}
    if ambient_routing != ROUTING_ENVIRONMENT:
        raise ContractError(
            f"routing environment mismatch: expected {ROUTING_ENVIRONMENT!r}, "
            f"got {ambient_routing!r}"
        )
    receipt_dir = _validate_receipt_dir(receipt_dir)
    harness_identity = recompute_trusted_checkout(PROJECT_ROOT)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    torch = _import_required_module("torch")
    dist = _import_required_module("torch.distributed")
    torch_npu = _import_required_module("torch_npu")
    triton = _import_required_module("triton")
    ash = _import_required_module("shmem")
    actual_environment = _capture_runtime_environment(
        environment, torch=torch, torch_npu=torch_npu, triton=triton, ash=ash
    )
    compare_environment(environment, actual_environment)
    mega_moe = _import_required_module("mega_moe")
    ff = _import_required_module("benchmark.layer.bench_full_forward")
    bb = _import_required_module("benchmark.layer.bench_backward")
    utils = _import_required_module("tests._moe_dist_utils")
    golden_module = _import_required_module("mega_moe._goldens.backward")

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
        "environment": actual_environment,
        "harness_identity": harness_identity,
        "sidecar_binding": dict(SIDECAR_BINDING),
        "arms": [results[arm_id] for arm_id in ARM_IDS],
    }
    value = envelope(payload)
    validate_execution_envelope(value, trusted_checkout=harness_identity)
    path = receipt_dir / "current_human_baseline_execution.json"
    write_envelope(path, payload)
    read_verified_envelope(path, require_complete=True, trusted_checkout=harness_identity)
    return path


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
