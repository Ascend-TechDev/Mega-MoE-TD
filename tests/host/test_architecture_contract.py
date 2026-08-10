import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LINT = ROOT / "scripts" / "architecture_lint.py"


def _run_lint(root: Path):
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "architecture_lint.py"), "--strict", "--root", str(root)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_strict_architecture_lint_is_green():
    result = _run_lint(ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "architecture: OK" in result.stdout


def test_lint_causally_rejects_missing_evidence_principles(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    architecture = candidate / "docs" / "design" / "HARNESS_DESIGN_PHILOSOPHY.md"
    architecture.write_text(
        architecture.read_text(encoding="utf-8").replace(
            "## 4. Evidence principles", "## removed evidence section", 1
        ),
        encoding="utf-8",
    )
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "required section" in result.stdout


def test_lint_causally_rejects_arm_set_drift(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    schema_path = candidate / "benchmark" / "contracts" / "current_human_baseline_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["$defs"]["arm_id"]["enum"].pop()
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "four-arm set" in result.stdout


def test_lint_rejects_duplicate_canonical_design(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    duplicate = candidate / "docs" / "design" / "DUPLICATE.md"
    duplicate.write_text(
        "<!-- CANONICAL_ARCHITECTURE_SOURCE: current-human-baseline-v1 -->\n",
        encoding="utf-8",
    )
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "canonical architecture" in result.stdout


def test_lint_rejects_duplicate_authoritative_runner(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    duplicate = candidate / "benchmark" / "duplicate_runner.py"
    duplicate.write_text("AUTHORITATIVE_BASELINE_RUNNER = True\n", encoding="utf-8")
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "authoritative runner" in result.stdout


def test_lint_rejects_eleventh_provider_path(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    extra = candidate / "benchmark" / "providers" / "extra.py"
    extra.write_text("def describe(): return {}\n", encoding="utf-8")
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "provider path contract" in result.stdout


def test_complete_schema_requires_harness_and_sidecar_binding():
    schema_path = ROOT / "benchmark" / "contracts" / "current_human_baseline_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    complete = schema["$defs"]["payload"]["allOf"][0]["then"]["required"]
    assert "harness_identity" in complete
    assert "sidecar_binding" in complete


def test_complete_schema_binds_full_plan_authorized_git_and_backward_trace():
    schema_path = ROOT / "benchmark" / "contracts" / "current_human_baseline_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["$defs"]["payload"]["properties"]["plan"] == {
        "$ref": "#/$defs/plan"
    }
    required_plan = set(schema["$defs"]["plan"]["required"])
    assert required_plan == {
        "contract_version",
        "repository",
        "submodules",
        "model",
        "parallel",
        "tokens_per_rank",
        "fixture_seed",
        "routing_environment",
        "timing",
        "arms",
        "arm_operations",
        "fusion_switches",
        "legal_comparisons",
        "precision",
        "providers",
        "raw_receipt",
    }
    variables = schema["$defs"]["environment"]["properties"]["variables"]["const"]
    assert variables["MOE_BWD_TRACE"] == ""
    harness_required = set(schema["$defs"]["harness_identity"]["required"])
    assert {"checkout_locator", "branch", "live_ref", "live_commit"} <= harness_required
    payload_required = set(schema["$defs"]["payload"]["required"])
    assert {"harness_identity", "sidecar_binding"} <= payload_required


def test_lint_rejects_incomplete_portable_plan_contract(tmp_path):
    candidate = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        candidate,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    schema_path = candidate / "benchmark" / "contracts" / "current_human_baseline_v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["$defs"]["plan"]["required"].remove("providers")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    result = _run_lint(candidate)
    assert result.returncode == 1
    assert "full plan contract" in result.stdout
