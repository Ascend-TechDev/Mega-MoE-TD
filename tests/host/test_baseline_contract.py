import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import baseline_contract as contract  # noqa: E402
from providers import current_main, grouped_hccl  # noqa: E402


def _descriptions():
    return [grouped_hccl.describe(), current_main.describe()]


def _source(name):
    return {
        "kind": "runtime_file",
        "locator": f"/opt/current-human-baseline/{name}/identity.bin",
    }


def _environment():
    return {
        "repository": {
            "url": contract.REPOSITORY_URL,
            "commit": contract.REPOSITORY_COMMIT,
            "tree": contract.REPOSITORY_TREE,
        },
        "variables": dict(contract.ROUTING_ENVIRONMENT),
        "components": {
            name: {
                "version": "1.2.3",
                "source": _source(name),
                "sha256": "a" * 64,
            }
            for name in contract.REQUIRED_ENVIRONMENT_COMPONENTS
        },
    }


def _trusted_checkout():
    return {
        "repository_url": contract.REPOSITORY_URL,
        "commit": "b" * 40,
        "tree": "c" * 40,
        "parents": ["d" * 40],
        "product_base_commit": contract.REPOSITORY_COMMIT,
        "product_base_tree": contract.REPOSITORY_TREE,
        "changed_paths": list(contract.EXACT_HARNESS_PATHS),
        "clean": True,
    }


def _complete_payload():
    plan = contract.build_plan(_descriptions())
    arms = []
    for arm_id in contract.ARM_IDS:
        operation = contract.ARM_OPERATIONS[arm_id]
        arms.append(
            {
                "arm_id": arm_id,
                "operation": operation,
                "shapes": [
                    {
                        "tokens_per_rank": tokens,
                        "correctness": (
                            {"forward_output": "PASS"}
                            if operation == "forward"
                            else {name: "PASS" for name in contract.BACKWARD_GRADIENTS}
                        ),
                        "samples_ms": [float(index + 1) for index in range(contract.SAMPLES)],
                    }
                    for tokens in contract.TOKENS_PER_RANK
                ],
            }
        )
    return {
        "contract_version": contract.CONTRACT_VERSION,
        "status": "COMPLETE",
        "plan": plan,
        "environment": _environment(),
        "harness_identity": _trusted_checkout(),
        "sidecar_binding": dict(contract.SIDECAR_BINDING),
        "arms": arms,
    }


def test_fixed_plan_and_complete_receipt_validate():
    plan = contract.build_plan(_descriptions())
    contract.validate_plan(plan)
    envelope = contract.envelope(_complete_payload())
    contract.validate_execution_envelope(envelope, trusted_checkout=_trusted_checkout())


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda plan: plan["repository"].update(commit="0" * 40), "repository identity"),
        (lambda plan: plan["arms"].pop(), "four-arm set"),
        (lambda plan: plan.update(tokens_per_rank=[4096, 8192]), "shape set"),
        (lambda plan: plan["providers"][0].pop("source"), "provider source identity"),
    ],
)
def test_plan_identity_and_scope_fail_closed(mutation, expected):
    plan = contract.build_plan(_descriptions())
    mutation(plan)
    with pytest.raises(contract.ContractError, match=expected):
        contract.validate_plan(plan)


def test_execution_rejects_missing_gradient_and_truncated_samples():
    payload = _complete_payload()
    backward = next(arm for arm in payload["arms"] if arm["operation"] == "backward")
    backward["shapes"][0]["correctness"].pop(contract.BACKWARD_GRADIENTS[-1])
    backward["shapes"][1]["samples_ms"].pop()
    with pytest.raises(contract.ContractError, match="five-gradient correctness"):
        contract.validate_execution_envelope(
            contract.envelope(payload), trusted_checkout=_trusted_checkout()
        )

    payload = _complete_payload()
    payload["arms"][0]["shapes"][0]["samples_ms"].pop()
    with pytest.raises(contract.ContractError, match="full sample array"):
        contract.validate_execution_envelope(
            contract.envelope(payload), trusted_checkout=_trusted_checkout()
        )


def test_environment_receipt_requires_every_bound_component():
    environment = _environment()
    environment["components"].pop("aclshmem")
    with pytest.raises(contract.ContractError, match="environment component set"):
        contract.validate_environment(environment)


def test_environment_component_order_is_not_identity():
    environment = _environment()
    environment["components"] = dict(reversed(list(environment["components"].items())))
    contract.validate_environment(environment)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda env: env["components"]["torch"].update(version="identity-bound"), "placeholder"),
        (lambda env: env["components"]["torch"].update(source="receipt://torch"), "recoverable source"),
        (lambda env: env["components"]["torch"].update(sha256="arbitrary"), "sha256"),
        (lambda env: env["variables"].pop("MOE_FULL_BENCH_ROUTE_MODE"), "routing environment"),
        (lambda env: env["variables"].update(MOE_FULL_BENCH_ACTIVE_EXPERTS="16"), "routing environment"),
    ],
)
def test_environment_placeholder_and_route_drift_fail_closed(mutation, expected):
    environment = _environment()
    mutation(environment)
    with pytest.raises(contract.ContractError, match=expected):
        contract.validate_environment(environment)


def test_live_environment_comparison_rejects_nonmatching_identity():
    expected = _environment()
    actual = copy.deepcopy(expected)
    actual["components"]["torch"]["sha256"] = "b" * 64
    with pytest.raises(contract.ContractError, match="runtime environment mismatch"):
        contract.compare_environment(expected, actual)
    contract.compare_environment(expected, copy.deepcopy(expected))


def test_plan_binds_exact_route_mode_and_active_experts():
    plan = contract.build_plan(_descriptions())
    assert plan["routing_environment"] == contract.ROUTING_ENVIRONMENT
    for name in contract.ROUTING_ENVIRONMENT:
        mutated = copy.deepcopy(plan)
        mutated["routing_environment"].pop(name)
        with pytest.raises(contract.ContractError, match="routing environment"):
            contract.validate_plan(mutated)


def test_complete_requires_exact_trusted_harness_identity():
    payload = _complete_payload()
    trusted = _trusted_checkout()
    contract.validate_execution_envelope(contract.envelope(payload), trusted_checkout=trusted)

    missing = copy.deepcopy(payload)
    missing.pop("harness_identity")
    with pytest.raises(contract.ContractError, match="harness identity"):
        contract.validate_execution_envelope(contract.envelope(missing), trusted_checkout=trusted)

    tampered = copy.deepcopy(payload)
    tampered["harness_identity"]["tree"] = "e" * 40
    with pytest.raises(contract.ContractError, match="trusted checkout"):
        contract.validate_execution_envelope(contract.envelope(tampered), trusted_checkout=trusted)

    ambient_moved = copy.deepcopy(trusted)
    ambient_moved["commit"] = "f" * 40
    with pytest.raises(contract.ContractError, match="trusted checkout"):
        contract.validate_execution_envelope(
            contract.envelope(payload), trusted_checkout=ambient_moved
        )


def _mock_checkout(monkeypatch, *, changed_paths=None, status="", head="b" * 40):
    changed_paths = changed_paths or contract.EXACT_HARNESS_PATHS

    def fake_git(_root, *arguments):
        lookup = {
            ("remote", "get-url", "origin"): contract.REPOSITORY_URL,
            ("rev-parse", f"{contract.REPOSITORY_COMMIT}^{{tree}}"): contract.REPOSITORY_TREE,
            ("rev-parse", f"{contract.REPOSITORY_COMMIT}:3rdparty/bigop"): contract.BIGOP_COMMIT,
            ("diff", "--name-only", contract.REPOSITORY_COMMIT, "HEAD"): "\n".join(changed_paths),
            ("status", "--porcelain=v1", "--untracked-files=all"): status,
            ("show", "-s", "--format=%P", "HEAD"): "d" * 40,
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "HEAD^{tree}"): "c" * 40,
        }
        return lookup[arguments]

    monkeypatch.setattr(contract, "_git", fake_git)
    monkeypatch.setattr(
        contract.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )


def test_trusted_checkout_recomputation_rejects_deleted_path_and_dirty_mutation(
    tmp_path, monkeypatch
):
    for relative in contract.EXACT_HARNESS_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("identity\n", encoding="utf-8")
    _mock_checkout(monkeypatch)
    trusted = contract.recompute_trusted_checkout(tmp_path)
    assert trusted == _trusted_checkout()

    deleted = tmp_path / contract.EXACT_HARNESS_PATHS[-1]
    deleted.unlink()
    with pytest.raises(contract.ContractError, match="harness path missing"):
        contract.recompute_trusted_checkout(tmp_path)
    deleted.write_text("identity\n", encoding="utf-8")

    _mock_checkout(monkeypatch, status=" M benchmark/baseline_contract.py")
    with pytest.raises(contract.ContractError, match="dirty"):
        contract.recompute_trusted_checkout(tmp_path)


def test_envelope_hash_detects_mutation():
    envelope = contract.envelope(_complete_payload())
    envelope["payload"]["status"] = "MUTATED"
    with pytest.raises(contract.ContractError, match="payload hash"):
        contract.validate_execution_envelope(envelope, trusted_checkout=_trusted_checkout())


def test_verified_sidecar_api_rejects_missing_and_tampered_sidecar(tmp_path):
    payload = _complete_payload()
    path = tmp_path / "receipt.json"
    contract.write_envelope(path, payload)
    contract.read_verified_envelope(
        path, require_complete=True, trusted_checkout=_trusted_checkout()
    )

    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.unlink()
    with pytest.raises(contract.ContractError, match="sidecar"):
        contract.read_verified_envelope(
            path, require_complete=True, trusted_checkout=_trusted_checkout()
        )

    contract.write_envelope(path, payload)
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="sidecar"):
        contract.read_verified_envelope(
            path, require_complete=True, trusted_checkout=_trusted_checkout()
        )
