# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only guards for the Torch-wgrad versus Triton-wgrad experiment."""

from __future__ import annotations

import ast
from pathlib import Path

from config import select_cases
from tests import _moe_testkit as kit


ROOT = Path(__file__).resolve().parents[1]
BACKWARD = ROOT / "src" / "mega_moe" / "ops" / "backward.py"
BENCHMARK = ROOT / "benchmark" / "layer" / "bench_moe_suite.py"
FUNCTIONAL = ROOT / "tests" / "layer" / "test_moe_suite.py"


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
