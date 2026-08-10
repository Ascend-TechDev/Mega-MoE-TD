import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import baseline_contract as contract  # noqa: E402
from providers import current_main, grouped_hccl  # noqa: E402

REAL_LIVE_READER = contract._read_live_branch
LOCAL_HEAD = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    text=True,
    capture_output=True,
    check=True,
).stdout.strip()


@pytest.fixture(autouse=True)
def _stable_live_authority(monkeypatch):
    monkeypatch.setattr(
        contract,
        "_read_live_branch",
        lambda: [(LOCAL_HEAD, contract.AUTHORIZED_LIVE_REF)],
    )


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


def _fake_checkout():
    return {
        "repository_url": contract.REPOSITORY_URL,
        "checkout_locator": "/opt/authorized/Mega-MoE-TD",
        "branch": "codex02/uniep-current-main-recovery-20260809",
        "live_ref": "refs/heads/codex02/uniep-current-main-recovery-20260809",
        "live_commit": "b" * 40,
        "commit": "b" * 40,
        "tree": "c" * 40,
        "parents": ["d" * 40],
        "product_base_commit": contract.REPOSITORY_COMMIT,
        "product_base_tree": contract.REPOSITORY_TREE,
        "changed_paths": list(contract.EXACT_HARNESS_PATHS),
        "clean": True,
    }


def _authorized_checkout(tmp_path, name="authorized"):
    target = tmp_path / name
    branch = "codex02/uniep-current-main-recovery-20260809"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--branch", branch, str(ROOT), str(target)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "remote", "set-url", "origin", contract.REPOSITORY_URL],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "update-ref",
            f"refs/remotes/origin/{branch}",
            "HEAD",
        ],
        check=True,
    )
    return target


def _complete_payload(harness_identity=None):
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
        "harness_identity": harness_identity or _fake_checkout(),
        "sidecar_binding": dict(contract.SIDECAR_BINDING),
        "arms": arms,
    }


def test_fixed_plan_and_complete_receipt_validate(tmp_path):
    plan = contract.build_plan(_descriptions())
    contract.validate_plan(plan)
    authorized = _authorized_checkout(tmp_path)
    identity = contract.recompute_trusted_checkout(authorized)
    value = contract.envelope(_complete_payload(identity))
    contract.validate_execution_envelope(value, authorized_checkout=authorized)


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


def test_execution_rejects_missing_gradient_and_truncated_samples(tmp_path):
    authorized = _authorized_checkout(tmp_path)
    identity = contract.recompute_trusted_checkout(authorized)
    payload = _complete_payload(identity)
    backward = next(arm for arm in payload["arms"] if arm["operation"] == "backward")
    backward["shapes"][0]["correctness"].pop(contract.BACKWARD_GRADIENTS[-1])
    backward["shapes"][1]["samples_ms"].pop()
    with pytest.raises(contract.ContractError, match="five-gradient correctness"):
        contract.validate_execution_envelope(contract.envelope(payload), authorized_checkout=authorized)

    payload = _complete_payload(identity)
    payload["arms"][0]["shapes"][0]["samples_ms"].pop()
    with pytest.raises(contract.ContractError, match="full sample array"):
        contract.validate_execution_envelope(contract.envelope(payload), authorized_checkout=authorized)


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


def test_plan_binds_exact_route_and_backward_trace_environment():
    plan = contract.build_plan(_descriptions())
    assert plan["routing_environment"] == contract.ROUTING_ENVIRONMENT
    assert contract.ROUTING_ENVIRONMENT["MOE_BWD_TRACE"] == ""
    for name in contract.ROUTING_ENVIRONMENT:
        mutated = copy.deepcopy(plan)
        mutated["routing_environment"].pop(name)
        with pytest.raises(contract.ContractError, match="routing environment"):
            contract.validate_plan(mutated)


def test_complete_recomputes_authorized_checkout_and_rejects_self_signed_mapping(tmp_path):
    authorized = _authorized_checkout(tmp_path)
    trusted = contract.recompute_trusted_checkout(authorized)
    payload = _complete_payload(trusted)
    contract.validate_execution_envelope(
        contract.envelope(payload), authorized_checkout=authorized
    )

    missing = copy.deepcopy(payload)
    missing.pop("harness_identity")
    with pytest.raises(contract.ContractError, match="harness identity"):
        contract.validate_execution_envelope(
            contract.envelope(missing), authorized_checkout=authorized
        )

    self_signed = _complete_payload(_fake_checkout())
    with pytest.raises(contract.ContractError, match="authorized checkout"):
        contract.validate_execution_envelope(
            contract.envelope(self_signed), authorized_checkout=authorized
        )

    empty_plan = copy.deepcopy(payload)
    empty_plan["plan"] = {}
    with pytest.raises(contract.ContractError, match="contract version"):
        contract.validate_execution_envelope(
            contract.envelope(empty_plan), authorized_checkout=authorized
        )


def test_authorized_checkout_recomputation_rejects_deletion_and_dirty(tmp_path):
    deleted_checkout = _authorized_checkout(tmp_path, "deleted")
    contract.recompute_trusted_checkout(deleted_checkout)
    deleted = deleted_checkout / contract.EXACT_HARNESS_PATHS[-1]
    deleted.unlink()
    with pytest.raises(contract.ContractError, match="harness path missing"):
        contract.recompute_trusted_checkout(deleted_checkout)

    dirty_checkout = _authorized_checkout(tmp_path, "dirty")
    (dirty_checkout / "benchmark" / "baseline_contract.py").write_text(
        "dirty\n", encoding="utf-8"
    )
    with pytest.raises(contract.ContractError, match="dirty"):
        contract.recompute_trusted_checkout(dirty_checkout)

def test_live_remote_authority_cannot_be_replaced_by_local_tracking(
    tmp_path, monkeypatch
):
    authorized = _authorized_checkout(tmp_path)
    subprocess.run(
        [
            "git",
            "-C",
            str(authorized),
            "update-ref",
            f"refs/remotes/origin/{contract.AUTHORIZED_HARNESS_BRANCH}",
            "HEAD",
        ],
        check=True,
    )
    monkeypatch.setattr(contract, "_read_live_branch", lambda: [])
    with pytest.raises(contract.ContractError, match="live remote authority"):
        contract.recompute_trusted_checkout(authorized)


def test_live_remote_reader_uses_fixed_ref_and_sanitized_environment(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{LOCAL_HEAD}\t{contract.AUTHORIZED_LIVE_REF}\n",
            stderr="",
        )

    monkeypatch.setattr(contract, "_read_live_branch", REAL_LIVE_READER)
    monkeypatch.setattr(contract.subprocess, "run", fake_run)
    assert contract._read_live_branch() == [(LOCAL_HEAD, contract.AUTHORIZED_LIVE_REF)]
    assert captured["command"] == [
        "git",
        "ls-remote",
        "--heads",
        contract.REPOSITORY_URL,
        contract.AUTHORIZED_LIVE_REF,
    ]
    assert "HOME" not in captured["environment"]
    assert "GIT_DIR" not in captured["environment"]
    assert captured["environment"]["GIT_CONFIG_GLOBAL"] == contract.os.devnull


def test_authorized_checkout_rejects_alias_detached_fake_and_origin_substitution(tmp_path):
    authorized = _authorized_checkout(tmp_path, "physical")
    with pytest.raises(contract.ContractError, match="physical checkout locator"):
        contract.recompute_trusted_checkout(str(authorized) + "/.")

    alias = tmp_path / "alias"
    alias.symlink_to(authorized, target_is_directory=True)
    with pytest.raises(contract.ContractError, match="physical checkout locator"):
        contract.recompute_trusted_checkout(alias)

    detached = _authorized_checkout(tmp_path, "detached")
    subprocess.run(
        ["git", "-C", str(detached), "switch", "--quiet", "--detach", "HEAD"], check=True
    )
    with pytest.raises(contract.ContractError, match="branch"):
        contract.recompute_trusted_checkout(detached)

    substituted = _authorized_checkout(tmp_path, "substituted")
    subprocess.run(
        [
            "git",
            "-C",
            str(substituted),
            "remote",
            "set-url",
            "origin",
            "https://example.invalid/fake.git",
        ],
        check=True,
    )
    with pytest.raises(contract.ContractError, match="repository locator"):
        contract.recompute_trusted_checkout(substituted)

    fake = _authorized_checkout(tmp_path, "fake")
    subprocess.run(["git", "-C", str(fake), "config", "user.name", "host-test"], check=True)
    subprocess.run(
        ["git", "-C", str(fake), "config", "user.email", "host-test@example.invalid"],
        check=True,
    )
    design = fake / contract.EXACT_HARNESS_PATHS[0]
    design.write_text(design.read_text(encoding="utf-8") + "\nlocal fake\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(fake), "add", str(design)], check=True)
    subprocess.run(["git", "-C", str(fake), "commit", "--quiet", "-m", "local fake"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(fake),
            "update-ref",
            f"refs/remotes/origin/{contract.AUTHORIZED_HARNESS_BRANCH}",
            "HEAD",
        ],
        check=True,
    )
    with pytest.raises(contract.ContractError, match="live remote authority"):
        contract.recompute_trusted_checkout(fake)


def test_envelope_hash_detects_mutation(tmp_path):
    authorized = _authorized_checkout(tmp_path)
    identity = contract.recompute_trusted_checkout(authorized)
    envelope = contract.envelope(_complete_payload(identity))
    envelope["payload"]["status"] = "MUTATED"
    with pytest.raises(contract.ContractError, match="payload hash"):
        contract.validate_execution_envelope(envelope, authorized_checkout=authorized)


def test_verified_sidecar_api_rejects_missing_and_tampered_sidecar(tmp_path):
    authorized = _authorized_checkout(tmp_path)
    identity = contract.recompute_trusted_checkout(authorized)
    payload = _complete_payload(identity)
    path = tmp_path / "receipt.json"
    contract.write_envelope(path, payload)
    contract.read_verified_envelope(path, require_complete=True, authorized_checkout=authorized)

    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.unlink()
    with pytest.raises(contract.ContractError, match="sidecar"):
        contract.read_verified_envelope(path, require_complete=True, authorized_checkout=authorized)

    contract.write_envelope(path, payload)
    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="sidecar"):
        contract.read_verified_envelope(path, require_complete=True, authorized_checkout=authorized)


def test_verified_reader_rejects_noncanonical_raw_bytes_with_recomputed_sidecar(tmp_path):
    authorized = _authorized_checkout(tmp_path)
    identity = contract.recompute_trusted_checkout(authorized)
    path = tmp_path / "pretty.json"
    contract.write_envelope(path, _complete_payload(identity))
    parsed = json.loads(path.read_bytes())
    pretty = (json.dumps(parsed, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(pretty)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(hashlib.sha256(pretty).hexdigest() + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="canonical"):
        contract.read_verified_envelope(
            path, require_complete=True, authorized_checkout=authorized
        )
