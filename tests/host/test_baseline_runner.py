import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark" / "current_human_baseline.py"
sys.path.insert(0, str(ROOT / "benchmark"))

import baseline_contract as contract  # noqa: E402


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
    envelope = json.loads(raw)
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
