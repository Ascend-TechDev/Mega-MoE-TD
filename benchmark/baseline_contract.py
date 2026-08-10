"""Fail-closed contract for the current-human four-arm baseline.

This module is deliberately standard-library only so architecture checks and
provider dry-runs never import an NPU runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


CONTRACT_VERSION = "current-human-baseline/v1"
REPOSITORY_URL = "https://gitcode.com/jzhoujg/Mega-MoE-TD.git"
REPOSITORY_COMMIT = "2887022970f6ae5c7ab5be096680809591100c30"
REPOSITORY_TREE = "2b5c447da8ff9ff788125aae5eec644579df105d"
BIGOP_COMMIT = "68ca4cb68dca41ca8960e6aedc38bd224b08a2af"
MODEL = {
    "name": "Qwen3-30B-A3B",
    "activation_dtype": "bfloat16",
    "hidden": 2048,
    "ffn": 768,
    "topk": 8,
    "num_experts": 128,
}
EP_WORLD_SIZE = 2
TOKENS_PER_RANK = (4096, 8192, 16384)
FIXTURE_SEED = 42
WARMUP = 5
SAMPLES = 50

ARM_IDS = (
    "unfused_grouped_hccl_forward",
    "fused_current_main_forward",
    "backward_default_torch_wgrad",
    "backward_optin_triton_wgrad",
)
ARM_OPERATIONS = {
    ARM_IDS[0]: "forward",
    ARM_IDS[1]: "forward",
    ARM_IDS[2]: "backward",
    ARM_IDS[3]: "backward",
}
LEGAL_COMPARISONS = (
    {
        "comparison_id": "forward_grouped_hccl_vs_fused",
        "baseline_arm": ARM_IDS[0],
        "candidate_arm": ARM_IDS[1],
        "operation": "forward",
    },
    {
        "comparison_id": "backward_default_vs_triton_wgrad",
        "baseline_arm": ARM_IDS[2],
        "candidate_arm": ARM_IDS[3],
        "operation": "backward",
    },
)
BACKWARD_GRADIENTS = (
    "grad_hidden",
    "grad_routing_weights",
    "grad_fc1_1",
    "grad_fc1_2",
    "grad_fc2",
)
REQUIRED_ENVIRONMENT_COMPONENTS = (
    "python",
    "torch",
    "torch_npu",
    "triton",
    "cann",
    "aclshmem",
    "bigop",
)
FUSION_SWITCHES = {
    ARM_IDS[0]: {
        "forward": "unfused_torch_npu_grouped_gemm_plus_hccl",
        "wgrad": "not_applicable",
    },
    ARM_IDS[1]: {
        "forward": "current_main_dispatch_fc1_weighted_swiglu_fc2_combine",
        "wgrad": "not_applicable",
    },
    ARM_IDS[2]: {
        "forward": "not_timed",
        "wgrad": "torch_per_expert_default",
    },
    ARM_IDS[3]: {
        "forward": "not_timed",
        "wgrad": "triton_transposed_grouped_gemm_optin",
    },
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ContractError(ValueError):
    """Raised whenever evidence cannot satisfy the fixed contract."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(payload)
    return {"payload": materialized, "payload_sha256": canonical_sha256(materialized)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _exact_mapping(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _require(dict(actual) == dict(expected), f"{label} mismatch")


def _provider_arms(providers: Iterable[Mapping[str, Any]]) -> list[str]:
    return [arm for provider in providers for arm in provider.get("arms", [])]


def build_plan(provider_descriptions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    providers = [dict(description) for description in provider_descriptions]
    plan = {
        "contract_version": CONTRACT_VERSION,
        "repository": {
            "url": REPOSITORY_URL,
            "commit": REPOSITORY_COMMIT,
            "tree": REPOSITORY_TREE,
        },
        "submodules": [{"path": "3rdparty/bigop", "commit": BIGOP_COMMIT}],
        "model": dict(MODEL),
        "parallel": {"expert_parallel_world_size": EP_WORLD_SIZE},
        "tokens_per_rank": list(TOKENS_PER_RANK),
        "fixture_seed": FIXTURE_SEED,
        "timing": {
            "warmup": WARMUP,
            "samples": SAMPLES,
            "boundary": "synchronized full provider call; host wall-clock; per-sample rank MAX",
        },
        "arms": list(ARM_IDS),
        "arm_operations": dict(ARM_OPERATIONS),
        "fusion_switches": {key: dict(value) for key, value in FUSION_SWITCHES.items()},
        "legal_comparisons": [dict(value) for value in LEGAL_COMPARISONS],
        "precision": {
            "forward": ["full output"],
            "backward": list(BACKWARD_GRADIENTS),
            "timing_requires_precision_pass": True,
        },
        "providers": providers,
        "raw_receipt": {
            "canonical_json": True,
            "full_sample_arrays": True,
            "payload_sha256": True,
            "file_sha256_sidecar": True,
        },
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    _require(plan.get("contract_version") == CONTRACT_VERSION, "contract version mismatch")
    _exact_mapping(
        plan.get("repository", {}),
        {"url": REPOSITORY_URL, "commit": REPOSITORY_COMMIT, "tree": REPOSITORY_TREE},
        "repository identity",
    )
    _require(
        plan.get("submodules") == [{"path": "3rdparty/bigop", "commit": BIGOP_COMMIT}],
        "submodule identity mismatch",
    )
    _exact_mapping(plan.get("model", {}), MODEL, "model identity")
    _require(
        plan.get("parallel") == {"expert_parallel_world_size": EP_WORLD_SIZE},
        "parallel identity mismatch",
    )
    _require(tuple(plan.get("tokens_per_rank", ())) == TOKENS_PER_RANK, "shape set mismatch")
    _require(plan.get("fixture_seed") == FIXTURE_SEED, "fixture seed mismatch")
    timing = plan.get("timing", {})
    _require(
        timing.get("warmup") == WARMUP
        and timing.get("samples") == SAMPLES
        and timing.get("boundary")
        == "synchronized full provider call; host wall-clock; per-sample rank MAX",
        "timing contract mismatch",
    )
    _require(tuple(plan.get("arms", ())) == ARM_IDS, "four-arm set mismatch")
    _exact_mapping(plan.get("arm_operations", {}), ARM_OPERATIONS, "arm operation map")
    _exact_mapping(plan.get("fusion_switches", {}), FUSION_SWITCHES, "fusion switch map")
    _require(plan.get("legal_comparisons") == list(LEGAL_COMPARISONS), "legal comparison set mismatch")

    providers = plan.get("providers")
    _require(isinstance(providers, list) and providers, "provider source identity missing")
    seen_ids: set[str] = set()
    for provider in providers:
        _require(isinstance(provider, Mapping), "provider source identity malformed")
        provider_id = provider.get("provider_id")
        _require(isinstance(provider_id, str) and provider_id and provider_id not in seen_ids, "provider id mismatch")
        seen_ids.add(provider_id)
        _require(provider.get("test_only") is False, "test-only provider is not executable evidence")
        source = provider.get("source")
        _require(isinstance(source, Mapping), "provider source identity missing")
        _require(
            source.get("repository_url") == REPOSITORY_URL
            and source.get("commit") == REPOSITORY_COMMIT
            and source.get("tree") == REPOSITORY_TREE,
            "provider source identity mismatch",
        )
        paths = source.get("paths")
        _require(isinstance(paths, list) and paths and all(isinstance(path, str) and path for path in paths), "provider source paths missing")
    provider_arms = _provider_arms(providers)
    _require(
        tuple(provider_arms) == ARM_IDS and len(set(provider_arms)) == len(ARM_IDS),
        "provider four-arm set mismatch",
    )

    precision = plan.get("precision", {})
    _require(
        precision.get("forward") == ["full output"]
        and tuple(precision.get("backward", ())) == BACKWARD_GRADIENTS
        and precision.get("timing_requires_precision_pass") is True,
        "precision gate mismatch",
    )
    raw = plan.get("raw_receipt", {})
    _require(raw and all(raw.values()), "raw receipt identity contract missing")


def validate_environment(environment: Mapping[str, Any]) -> None:
    _exact_mapping(
        environment.get("repository", {}),
        {"url": REPOSITORY_URL, "commit": REPOSITORY_COMMIT, "tree": REPOSITORY_TREE},
        "environment repository identity",
    )
    components = environment.get("components")
    _require(isinstance(components, Mapping), "environment component set missing")
    _require(set(components) == set(REQUIRED_ENVIRONMENT_COMPONENTS), "environment component set mismatch")
    for name in REQUIRED_ENVIRONMENT_COMPONENTS:
        identity = components[name]
        _require(isinstance(identity, Mapping), f"environment component {name} malformed")
        _require(isinstance(identity.get("version"), str) and identity["version"], f"environment component {name} version missing")
        _require(isinstance(identity.get("source"), str) and identity["source"], f"environment component {name} source missing")
        _require(bool(_SHA256.fullmatch(str(identity.get("sha256", "")))), f"environment component {name} sha256 invalid")


def _verify_envelope(value: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = value.get("payload")
    _require(isinstance(payload, Mapping), "receipt payload missing")
    digest = value.get("payload_sha256")
    _require(bool(_SHA256.fullmatch(str(digest or ""))), "payload hash invalid")
    _require(digest == canonical_sha256(payload), "payload hash mismatch")
    return payload


def validate_dry_run_envelope(value: Mapping[str, Any]) -> None:
    payload = _verify_envelope(value)
    _require(payload.get("contract_version") == CONTRACT_VERSION, "contract version mismatch")
    _require(payload.get("status") == "DRY_RUN_VALIDATED", "dry-run status mismatch")
    validate_plan(payload.get("plan", {}))
    _require(payload.get("device_modules_loaded") == [], "host dry-run imported a device module")


def _validate_correctness(operation: str, correctness: Mapping[str, Any]) -> None:
    if operation == "forward":
        _require(correctness == {"forward_output": "PASS"}, "forward correctness gate missing")
    else:
        _require(
            tuple(correctness.keys()) == BACKWARD_GRADIENTS
            and all(correctness[name] == "PASS" for name in BACKWARD_GRADIENTS),
            "five-gradient correctness gate missing",
        )


def validate_execution_envelope(value: Mapping[str, Any]) -> None:
    payload = _verify_envelope(value)
    _require(payload.get("contract_version") == CONTRACT_VERSION, "contract version mismatch")
    _require(payload.get("status") == "COMPLETE", "execution status is not COMPLETE")
    validate_plan(payload.get("plan", {}))
    validate_environment(payload.get("environment", {}))
    arms = payload.get("arms")
    _require(isinstance(arms, list) and tuple(arm.get("arm_id") for arm in arms) == ARM_IDS, "execution four-arm set mismatch")
    for arm in arms:
        arm_id = arm["arm_id"]
        operation = ARM_OPERATIONS[arm_id]
        _require(arm.get("operation") == operation, "execution arm operation mismatch")
        shapes = arm.get("shapes")
        _require(
            isinstance(shapes, list)
            and tuple(shape.get("tokens_per_rank") for shape in shapes) == TOKENS_PER_RANK,
            "execution shape set mismatch",
        )
        for shape in shapes:
            correctness = shape.get("correctness")
            _require(isinstance(correctness, Mapping), "correctness gate missing")
            _validate_correctness(operation, correctness)
            samples = shape.get("samples_ms")
            _require(isinstance(samples, list) and len(samples) == SAMPLES, "full sample array missing")
            _require(
                all(isinstance(sample, (int, float)) and math.isfinite(sample) and sample > 0 for sample in samples),
                "sample array contains invalid latency",
            )


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON receipt {path}: {error}") from error


def write_envelope(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = envelope(payload)
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)
    file_digest = hashlib.sha256(raw).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(file_digest + "\n", encoding="utf-8")
    os.replace(sidecar_temporary, sidecar)
    return value
