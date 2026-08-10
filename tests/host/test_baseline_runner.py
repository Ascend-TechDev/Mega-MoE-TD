import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark" / "current_human_baseline.py"
sys.path.insert(0, str(ROOT / "benchmark"))

import baseline_contract as contract  # noqa: E402
import current_human_baseline as runner  # noqa: E402


def test_host_dry_run_emits_identity_bound_canonical_receipt(tmp_path):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--receipt-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = tmp_path / "current_human_baseline_dry_run.json"
    sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
    assert receipt.is_file()
    assert sidecar.is_file()
    raw = receipt.read_bytes()
    assert sidecar.read_text(encoding="utf-8").strip() == hashlib.sha256(raw).hexdigest()
    envelope = contract.read_verified_envelope(receipt)
    contract.validate_dry_run_envelope(envelope)
    payload = envelope["payload"]
    assert payload["device_modules_loaded"] == []
    assert tuple(payload["plan"]["arms"]) == contract.ARM_IDS
    assert tuple(payload["plan"]["tokens_per_rank"]) == contract.TOKENS_PER_RANK


def test_execute_without_environment_receipt_fails_before_device_import(tmp_path):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--execute", "--receipt-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--environment-receipt" in result.stderr
    assert not list(tmp_path.glob("*.json"))


def test_receipts_cannot_be_written_into_the_source_checkout():
    prohibited = ROOT / ".current-human-baseline-receipt"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--dry-run", "--receipt-dir", str(prohibited)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "outside the source checkout" in result.stderr
    assert not prohibited.exists()


def test_provider_descriptions_are_host_safe_and_real():
    loaded_before = set(sys.modules)
    from providers import current_main, grouped_hccl

    descriptions = [grouped_hccl.describe(), current_main.describe()]
    loaded_after = set(sys.modules) - loaded_before
    assert not ({"torch", "torch_npu", "triton", "shmem"} & loaded_after)
    assert descriptions[0]["provider_id"] == "current-main-grouped-hccl"
    assert descriptions[1]["provider_id"] == "current-main-production"
    assert descriptions[0]["test_only"] is False
    assert descriptions[1]["test_only"] is False
    assert set(descriptions[0]["arms"] + descriptions[1]["arms"]) == set(contract.ARM_IDS)


def test_execute_route_environment_is_explicit_not_defaulted(tmp_path):
    environment = {
        "repository": {
            "url": contract.REPOSITORY_URL,
            "commit": contract.REPOSITORY_COMMIT,
            "tree": contract.REPOSITORY_TREE,
        },
        "variables": dict(contract.ROUTING_ENVIRONMENT),
        "components": {
            name: {
                "version": "1.2.3",
                "source": {
                    "kind": "runtime_file",
                    "locator": f"/opt/current-human-baseline/{name}/identity.bin",
                },
                "sha256": "a" * 64,
            }
            for name in contract.REQUIRED_ENVIRONMENT_COMPONENTS
        },
    }
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    process_environment = dict(os.environ)
    process_environment.pop("MOE_FULL_BENCH_ROUTE_MODE", None)
    process_environment.pop("MOE_FULL_BENCH_ACTIVE_EXPERTS", None)
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--execute",
            "--receipt-dir",
            str(tmp_path / "receipt"),
            "--environment-receipt",
            str(environment_path),
        ],
        cwd=ROOT,
        env=process_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "routing environment" in result.stderr
    assert "torch" not in result.stderr.lower()


def test_runtime_environment_is_recomputed_from_real_sources_without_device_import(
    tmp_path, monkeypatch
):
    loaded_before = set(sys.modules)

    def fake_module(name, version):
        source = tmp_path / f"{name}.bin"
        source.write_bytes(name.encode("ascii"))
        return SimpleNamespace(__file__=str(source), __version__=version)

    modules = {
        "torch": fake_module("torch", "2.7.0"),
        "torch_npu": fake_module("torch_npu", "2.7.0"),
        "triton": fake_module("triton", "3.2.0"),
        "aclshmem": fake_module("aclshmem", "1.0.0"),
    }
    cann_root = tmp_path / "cann"
    cann_root.mkdir()
    cann_version = cann_root / "version.info"
    cann_version.write_text("Version=9.1.0\n", encoding="utf-8")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_root))

    expected = {
        "repository": {
            "url": contract.REPOSITORY_URL,
            "commit": contract.REPOSITORY_COMMIT,
            "tree": contract.REPOSITORY_TREE,
        },
        "variables": dict(contract.ROUTING_ENVIRONMENT),
        "components": {
            "python": runner._runtime_file_identity(
                Path(sys.executable), runner.platform.python_version()
            ),
            "torch": runner._module_identity(modules["torch"], "torch"),
            "torch_npu": runner._module_identity(modules["torch_npu"], "torch-npu"),
            "triton": runner._module_identity(modules["triton"], "triton"),
            "cann": runner._runtime_file_identity(cann_version, "9.1.0"),
            "aclshmem": runner._module_identity(modules["aclshmem"], "aclshmem"),
            "bigop": {
                "version": contract.BIGOP_COMMIT,
                "source": {"kind": "gitlink", "locator": "3rdparty/bigop"},
                "sha256": hashlib.sha256(contract.BIGOP_COMMIT.encode("ascii")).hexdigest(),
            },
        },
    }
    actual = runner._capture_runtime_environment(
        expected,
        torch=modules["torch"],
        torch_npu=modules["torch_npu"],
        triton=modules["triton"],
        ash=modules["aclshmem"],
    )
    contract.compare_environment(expected, actual)
    loaded_after = set(sys.modules) - loaded_before
    assert not ({"torch", "torch_npu", "triton", "shmem"} & loaded_after)
