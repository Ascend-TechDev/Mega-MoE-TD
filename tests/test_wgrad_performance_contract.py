# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only guards for the Torch-wgrad versus Triton-wgrad experiment."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest import mock

import pytest
import torch

from config import select_cases
from tests import _moe_testkit as kit


ROOT = Path(__file__).resolve().parents[1]
BACKWARD = ROOT / "src" / "mega_moe" / "ops" / "backward.py"
BENCHMARK = ROOT / "benchmark" / "layer" / "bench_moe_suite.py"
FUNCTIONAL = ROOT / "tests" / "layer" / "test_moe_suite.py"

_BACKWARD_ORDERS = (
    "candidate_then_baseline",
    "baseline_then_candidate",
)
_BACKWARD_ARMS = ("triton_wgrad", "torch_wgrad")
_QWEN_EP2_CASES = (
    ("performance-bwd-qwen3-30b-a3b-w2-t4k", 4096),
    ("performance-bwd-qwen3-30b-a3b-w2-t8k", 8192),
    ("performance-bwd-qwen3-30b-a3b-w2-t16k", 16384),
)


def _load_benchmark_suite():
    """Load the benchmark contract without importing unavailable NPU kernels."""
    mega_moe = types.ModuleType("mega_moe")
    mega_moe.FusedMoEForward = type("FusedMoEForward", (), {})
    mega_moe.MoEForwardConfig = type("MoEForwardConfig", (), {})

    grouped = types.ModuleType("benchmark.layer._grouped_forward_baseline")
    grouped.GroupedForwardBaseline = type("GroupedForwardBaseline", (), {})

    baselines = types.ModuleType("tests._moe_baselines")
    baselines.backward_torch_baseline = lambda *args, **kwargs: None
    baselines.build_backward_saved = lambda *args, **kwargs: None
    baselines.compare_backward_gradients = lambda *args, **kwargs: None

    module_name = "_host_bench_moe_suite_contract"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "mega_moe": mega_moe,
            "benchmark.layer._grouped_forward_baseline": grouped,
            "tests._moe_baselines": baselines,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name} in {path}")


def test_backward_api_has_explicit_wgrad_selector():
    node = _function(BACKWARD, "moe_backward_triton")
    kwonly = [argument.arg for argument in node.args.kwonlyargs]
    assert "use_triton_wgrad" in kwonly
    source = ast.unparse(node)
    assert "use_triton_wgrad is None" in source
    assert "MOE_WGRAD_TRITON" in source


def test_torch_wgrad_zero_token_experts_are_deterministic_zero():
    source = ast.unparse(_function(BACKWARD, "_grouped_wgrad_torch"))
    assert "torch.zeros" in source
    assert "torch.empty" not in source


def test_functional_gate_checks_both_explicit_wgrad_modes_numerically():
    source = ast.unparse(_function(FUNCTIONAL, "run_backward_case"))
    assert "('torch-wgrad', False)" in source
    assert "('triton-wgrad', True)" in source
    assert "use_triton_wgrad=use_triton_wgrad" in source
    assert "compare_backward_gradients" in source


def test_backward_benchmark_has_two_numeric_arms_and_honest_target():
    source = BENCHMARK.read_text(encoding="utf-8")
    gate = ast.unparse(_function(BENCHMARK, "_backward_gate"))
    runner = ast.unparse(_function(BENCHMARK, "run_backward_benchmark"))
    assert "compare_backward_gradients" in gate
    assert "use_triton_wgrad=False" in gate
    assert "use_triton_wgrad=True" in gate
    assert "use_triton_wgrad=False" in runner
    assert "use_triton_wgrad=True" in runner
    assert "run_paired" in runner
    assert '"target_speedup": 1.5' in source
    assert '"target_met"' in source
    assert '"raw_samples_ms"' in source
    assert "_validate_backward_benchmark_environment()" in runner
    environment_gate = ast.unparse(
        _function(BENCHMARK, "_validate_backward_benchmark_environment")
    )
    assert "MOE_WGRAD_TRITON" in environment_gate
    assert "MOE_BWD_TRACE" in environment_gate


def test_qwen_ep2_target_denominator_is_exact_three_shape_sweep():
    cases = tuple(
        case
        for case in select_cases(direction="backward", tags={"performance"})
        if case.model == "Qwen3-30B-A3B" and case.world_size == 2
    )
    assert [case.tokens for case in cases] == [4096, 8192, 16384]
    assert {
        (case.hidden, case.ffn, case.topk, case.num_experts)
        for case in cases
    } == {(2048, 768, 8, 128)}


def test_paired_runner_executes_both_orders_and_keeps_raw_samples():
    calls: list[str] = []

    def candidate():
        calls.append("candidate")

    def baseline():
        calls.append("baseline")

    runner = object.__new__(kit.PerformanceRunner)
    runner.candidate = candidate
    runner.baseline = baseline
    runner.timing = kit.TimingSpec(warmup=1, iterations=2, clock="host_wall")
    runner._measure_single = lambda fn: (fn(), float(len(calls)))[1]

    paired = runner.run_paired()

    assert list(paired) == ["candidate_then_baseline", "baseline_then_candidate"]
    assert paired["candidate_then_baseline"]["candidate"].samples_ms == (3.0, 5.0)
    assert paired["candidate_then_baseline"]["baseline"].samples_ms == (4.0, 6.0)
    assert paired["baseline_then_candidate"]["baseline"].samples_ms == (9.0, 11.0)
    assert paired["baseline_then_candidate"]["candidate"].samples_ms == (10.0, 12.0)
    assert paired["candidate_then_baseline"]["candidate"].median_ms == 4.0
    assert calls == [
        "candidate", "baseline",
        "candidate", "baseline", "candidate", "baseline",
        "baseline", "candidate",
        "baseline", "candidate", "baseline", "candidate",
    ]


def _finite_gradient_result():
    return {
        name: torch.zeros(2, dtype=torch.float32)
        for name in (
            "grad_hidden",
            "grad_routing_weights",
            "grad_fc1_1",
            "grad_fc1_2",
            "grad_fc2",
        )
    }


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("bad_side", ["oracle", "torch_wgrad", "triton_wgrad"])
def test_backward_finite_gate_rejects_nonfinite_in_any_arm_or_oracle(
    bad_value, bad_side
):
    oracle = _finite_gradient_result()
    candidates = {
        "torch_wgrad": _finite_gradient_result(),
        "triton_wgrad": _finite_gradient_result(),
    }
    target = oracle if bad_side == "oracle" else candidates[bad_side]
    target["grad_hidden"][0] = bad_value

    with pytest.raises(AssertionError, match="non-finite.*grad_hidden"):
        kit.validate_finite_backward_gradients(candidates, oracle)


def _valid_backward_entry(case_id: str, tokens: int) -> dict:
    raw = {
        order: {arm: [1.0] * 50 for arm in _BACKWARD_ARMS}
        for order in _BACKWARD_ORDERS
    }
    stats = {
        order: {
            arm: {"min_ms": 1.0, "max_ms": 1.0, "mean_ms": 1.0, "median_ms": 1.0}
            for arm in _BACKWARD_ARMS
        }
        for order in _BACKWARD_ORDERS
    }
    return {
        "schema_version": 2,
        "direction": "backward",
        "case_id": case_id,
        "model": "Qwen3-30B-A3B",
        "world_size": 2,
        "tokens_per_rank": tokens,
        "shape": {"hidden": 2048, "ffn": 768, "topk": 8, "num_experts": 128},
        "protocol": {
            **kit.BACKWARD_TIMING.as_dict(),
            "paired_orders": [
                "triton_wgrad_then_torch_wgrad",
                "torch_wgrad_then_triton_wgrad",
            ],
            "samples_per_arm_per_order": 50,
            "comparison_boundary": (
                "same five-stage backward; only step3/step5 wgrad implementation differs"
            ),
        },
        "correctness_gate": {
            "status": "passed_before_timing",
            "kind": "independent Torch oracle; five numeric gradients",
            "arms": ["torch_wgrad", "triton_wgrad"],
        },
        "metrics": {
            "raw_samples_ms": raw,
            "order_stats": stats,
            "speedup_by_order": {order: 2.0 for order in _BACKWARD_ORDERS},
            "minimum_speedup": 2.0,
            "target_speedup": 1.5,
        },
        "gradient_gate": {
            "keys": list(_finite_gradient_result()),
            "comparison": "untimed numeric gate for both wgrad arms",
            "details": {},
        },
        "provenance": {
            "checkout_identity": {
                "commit": "a" * 40,
                "tree": "b" * 40,
                "sole_parent": "c" * 40,
                "clean": True,
            },
            "environment_receipt": {
                "schema": "uniep.environment-receipt.v1",
                "status": "VALIDATED",
                "sha256": "d" * 64,
                "size": 1,
                "components": [
                    "aclshmem",
                    "bigop",
                    "cann",
                    "python",
                    "torch",
                    "torch_npu",
                    "triton",
                ],
            },
            "backward_source_sha256": {
                relative_path: "e" * 64
                for relative_path in (
                    "src/mega_moe/ops/backward.py",
                    "src/mega_moe/kernels/transposed_grouped_gemm.py",
                    "src/mega_moe/kernels/common.py",
                    "tests/_moe_testkit.py",
                    "tests/_moe_baselines.py",
                    "tests/_numeric.py",
                    "config/_shapes.py",
                )
            },
        },
    }


def test_backward_writer_marks_missing_cases_and_withholds_target(tmp_path):
    benchmark_suite = _load_benchmark_suite()
    path = tmp_path / "backward.json"
    case_id, tokens = _QWEN_EP2_CASES[0]
    entry = _valid_backward_entry(case_id, tokens)

    benchmark_suite._upsert_result(
        path, "backward", 2, entry, entry["protocol"]
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["denominator_complete"] is False
    assert payload["missing_cases"] == [item[0] for item in _QWEN_EP2_CASES[1:]]
    assert "target_met" not in payload
    assert all("target_met" not in item["metrics"] for item in payload["cases"])


def test_backward_finalizer_requires_exact_three_complete_cases(tmp_path):
    benchmark_suite = _load_benchmark_suite()
    path = tmp_path / "backward.json"
    for case_id, tokens in _QWEN_EP2_CASES:
        entry = _valid_backward_entry(case_id, tokens)
        benchmark_suite._upsert_result(
            path, "backward", 2, entry, entry["protocol"]
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["denominator_complete"] is True
    assert payload["missing_cases"] == []
    assert payload["target_met"] is True
    assert all(item["metrics"]["target_met"] is True for item in payload["cases"])

    bad = _valid_backward_entry(*_QWEN_EP2_CASES[0])
    bad["metrics"]["raw_samples_ms"][_BACKWARD_ORDERS[0]]["triton_wgrad"].pop()
    with pytest.raises(ValueError, match="50 samples"):
        benchmark_suite._upsert_result(
            tmp_path / "bad.json", "backward", 2, bad, bad["protocol"]
        )


def test_backward_provenance_binds_full_evidence_chain(monkeypatch, tmp_path):
    benchmark_suite = _load_benchmark_suite()
    expected_sources = {
        "src/mega_moe/ops/backward.py",
        "src/mega_moe/kernels/transposed_grouped_gemm.py",
        "src/mega_moe/kernels/common.py",
        "tests/_moe_testkit.py",
        "tests/_moe_baselines.py",
        "tests/_numeric.py",
        "config/_shapes.py",
    }
    assert set(benchmark_suite.BACKWARD_EVIDENCE_SOURCES) == expected_sources

    checkout_root = tmp_path / "checkout"
    checkout_root.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout_root)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout_root), "config", "user.name", "Host Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout_root), "config", "user.email", "host@test.invalid"],
        check=True,
    )
    tracked = checkout_root / "tracked.txt"
    tracked.write_text("parent\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout_root), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout_root), "commit", "-qm", "parent"], check=True
    )
    parent = subprocess.check_output(
        ["git", "-C", str(checkout_root), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked.write_text("child\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout_root), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout_root), "commit", "-qm", "child"], check=True
    )
    checkout_identity = benchmark_suite._checkout_identity(checkout_root)
    assert checkout_identity["sole_parent"] == parent
    assert checkout_identity["clean"] is True
    (checkout_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean checkout"):
        benchmark_suite._checkout_identity(checkout_root)

    receipt_path = tmp_path / "environment.json"
    receipt = {
        "schema": "uniep.environment-receipt.v1",
        "status": "VALIDATED",
        "components": {
            name: {"identity": f"{name}-identity", "source_sha256": "e" * 64}
            for name in (
                "python", "torch", "torch_npu", "triton", "cann", "aclshmem", "bigop"
            )
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o600)
    monkeypatch.setenv("MOE_BENCH_ENVIRONMENT_RECEIPT", str(receipt_path))
    environment_identity = benchmark_suite._environment_receipt_identity()
    assert environment_identity["sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert "locator" not in json.dumps(environment_identity).lower()

    receipt["components"]["torch"]["identity"] = ""
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="torch.*identity"):
        benchmark_suite._environment_receipt_identity()

    checkout = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "sole_parent": "c" * 40,
        "clean": True,
    }
    environment = {
        "schema": "uniep.environment-receipt.v1",
        "sha256": "d" * 64,
        "size": 123,
        "status": "VALIDATED",
        "components": [
            "aclshmem", "bigop", "cann", "python", "torch", "torch_npu", "triton"
        ],
    }
    monkeypatch.setattr(benchmark_suite, "_checkout_identity", lambda: checkout)
    monkeypatch.setattr(
        benchmark_suite, "_environment_receipt_identity", lambda: environment
    )

    provenance = benchmark_suite._backward_benchmark_provenance()
    assert provenance["checkout_identity"] == checkout
    assert provenance["environment_receipt"] == environment
    assert set(provenance["backward_source_sha256"]) == expected_sources
    for relative_path, digest in provenance["backward_source_sha256"].items():
        expected = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert digest == expected
    assert "locator" not in json.dumps(provenance["environment_receipt"]).lower()


def test_backward_provenance_is_frozen_before_device_session():
    runner = ast.unparse(_function(BENCHMARK, "run_backward_benchmark"))
    assert runner.index("_backward_benchmark_provenance()") < runner.index(
        "_load_benchmark_device_runtime()"
    )
    assert runner.index("_backward_benchmark_provenance()") < runner.index(
        "aclshmem_session"
    )
    assert "provenance = _backward_benchmark_provenance()" in runner
    assert "'provenance': provenance" in runner
