from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import baseline_contract as contract  # noqa: E402
from providers import current_main, grouped_hccl  # noqa: E402


def _descriptions():
    return [grouped_hccl.describe(), current_main.describe()]


def _environment():
    return {
        "repository": {
            "url": contract.REPOSITORY_URL,
            "commit": contract.REPOSITORY_COMMIT,
            "tree": contract.REPOSITORY_TREE,
        },
        "components": {
            name: {
                "version": "identity-bound",
                "source": f"receipt://{name}",
                "sha256": "a" * 64,
            }
            for name in contract.REQUIRED_ENVIRONMENT_COMPONENTS
        },
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
        "arms": arms,
    }


def test_fixed_plan_and_complete_receipt_validate():
    plan = contract.build_plan(_descriptions())
    contract.validate_plan(plan)
    envelope = contract.envelope(_complete_payload())
    contract.validate_execution_envelope(envelope)


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
        contract.validate_execution_envelope(contract.envelope(payload))

    payload = _complete_payload()
    payload["arms"][0]["shapes"][0]["samples_ms"].pop()
    with pytest.raises(contract.ContractError, match="full sample array"):
        contract.validate_execution_envelope(contract.envelope(payload))


def test_environment_receipt_requires_every_bound_component():
    environment = _environment()
    environment["components"].pop("aclshmem")
    with pytest.raises(contract.ContractError, match="environment component set"):
        contract.validate_environment(environment)


def test_environment_component_order_is_not_identity():
    environment = _environment()
    environment["components"] = dict(reversed(list(environment["components"].items())))
    contract.validate_environment(environment)


def test_envelope_hash_detects_mutation():
    envelope = contract.envelope(_complete_payload())
    envelope["payload"]["status"] = "MUTATED"
    with pytest.raises(contract.ContractError, match="payload hash"):
        contract.validate_execution_envelope(envelope)
