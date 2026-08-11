# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only guards for the Torch-wgrad versus Triton-wgrad experiment."""

from __future__ import annotations

import ast
import base64
import contextlib
import functools
import hashlib
import http.server
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
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
    expected_sources = (
        "benchmark/layer/bench_moe_suite.py",
        "conftest.py",
        "src/mega_moe/__init__.py",
        "src/mega_moe/ops/backward.py",
        "src/mega_moe/kernels/__init__.py",
        "src/mega_moe/kernels/transposed_grouped_gemm.py",
        "src/mega_moe/kernels/common.py",
        "tests/_moe_testkit.py",
        "tests/_moe_baselines.py",
        "tests/_numeric.py",
        "config/_shapes.py",
        "tests/layer/test_moe_suite.py",
    )
    assert benchmark_suite.BACKWARD_EVIDENCE_SOURCES == expected_sources

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
    assert tuple(provenance["backward_source_sha256"]) == expected_sources
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


_TASK2_SOURCE_PATHS = (
    "benchmark/layer/bench_moe_suite.py",
    "conftest.py",
    "src/mega_moe/__init__.py",
    "src/mega_moe/ops/backward.py",
    "src/mega_moe/kernels/__init__.py",
    "src/mega_moe/kernels/transposed_grouped_gemm.py",
    "src/mega_moe/kernels/common.py",
    "tests/_moe_testkit.py",
    "tests/_moe_baselines.py",
    "tests/_numeric.py",
    "config/_shapes.py",
    "tests/layer/test_moe_suite.py",
)
_TASK2_COMPONENTS = (
    "aclshmem",
    "bigop",
    "cann",
    "python",
    "torch",
    "torch_npu",
    "triton",
)


def _task2_git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=cwd,
        input=input_bytes,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _task2_commit(repo: Path, message: str) -> str:
    _task2_git(repo, "add", "-A")
    _task2_git(repo, "commit", "-qm", message)
    return _task2_git(repo, "rev-parse", "HEAD").decode().strip()


def _task2_product_fixture(tmp_path: Path) -> types.SimpleNamespace:
    work = tmp_path / "product-work"
    work.mkdir()
    _task2_git(work, "init", "-q", "-b", "main")
    _task2_git(work, "config", "user.name", "Task2 Fixture")
    _task2_git(work, "config", "user.email", "task2@example.invalid")
    for index, relative in enumerate(_TASK2_SOURCE_PATHS):
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"anchor-{index}\n", encoding="utf-8")
    main_commit = _task2_commit(work, "main")
    runner = work / _TASK2_SOURCE_PATHS[0]
    runner.write_text("anchor-0\nfeature\n", encoding="utf-8")
    feature_commit = _task2_commit(work, "feature")
    tree = _task2_git(work, "rev-parse", f"{feature_commit}^{{tree}}").decode().strip()

    remote = tmp_path / "product.git"
    _task2_git(tmp_path, "clone", "-q", "--bare", str(work), str(remote))
    _task2_git(remote, "update-ref", "refs/heads/main", main_commit)
    _task2_git(
        remote,
        "update-ref",
        "refs/heads/codex02/uniep-triton-wgrad-1p5x-20260810",
        feature_commit,
    )

    sources = []
    for relative in _TASK2_SOURCE_PATHS:
        fields = _task2_git(
            work, "ls-tree", feature_commit, "--", relative
        ).decode().strip().split()
        blob_oid = fields[2]
        blob = _task2_git(work, "cat-file", "blob", blob_oid)
        sources.append(
            {
                "blob_oid": blob_oid,
                "path": relative,
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    product = {
        "authority_path": (
            "authorities/uniep/wgrad/"
            f"{feature_commit}/environment-authority.json"
        ),
        "commit": feature_commit,
        "feature_ref": "refs/heads/codex02/uniep-triton-wgrad-1p5x-20260810",
        "main_is_ancestor": True,
        "main_ref": "refs/heads/main",
        "observed_main": main_commit,
        "remote": str(remote),
        "sole_parent": main_commit,
        "sources": sources,
        "tree": tree,
    }
    return types.SimpleNamespace(
        remote=remote,
        work=work,
        product=product,
        feature_commit=feature_commit,
        main_commit=main_commit,
        tree=tree,
    )


def _task2_environment() -> dict:
    environment = {}
    for name in _TASK2_COMPONENTS:
        member_bytes = f"{name}-member\n".encode()
        members = [
            {
                "elf_build_id": None,
                "kind": "file",
                "mode": 0o100644,
                "name": f"{name}.identity",
                "sha256": hashlib.sha256(member_bytes).hexdigest(),
                "size": len(member_bytes),
            }
        ]
        encoded = json.dumps(
            members, sort_keys=True, separators=(",", ":")
        ).encode() + b"\n"
        environment[name] = {
            "identity": f"{name}==fixture",
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "member_count": 1,
            "members": members,
            "resolver_id": f"uniep-{name}-resolver-v1",
            "total_bytes": len(member_bytes),
        }
    return environment


def _task2_canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _task2_authority_fixture(
    tmp_path: Path, mutation: str | None = None, *, authority_missing: bool = False
) -> types.SimpleNamespace:
    product_fixture = _task2_product_fixture(tmp_path)
    work = tmp_path / "authority-work"
    work.mkdir()
    _task2_git(work, "init", "-q", "-b", "main")
    _task2_git(work, "config", "user.name", "Task2 C01 Fixture")
    _task2_git(work, "config", "user.email", "c01@example.invalid")
    tool_path = work / "tools" / "uniep_environment_authority.py"
    tool_path.parent.mkdir(parents=True)
    tool_path.write_text("# reviewed producer\n", encoding="utf-8")
    producer_commit = _task2_commit(work, "producer tool")
    producer_tree = _task2_git(
        work, "rev-parse", f"{producer_commit}^{{tree}}"
    ).decode().strip()
    tool_blob_oid = _task2_git(
        work, "rev-parse", f"{producer_commit}:tools/uniep_environment_authority.py"
    ).decode().strip()
    producer = {
        "commit": producer_commit,
        "identity": "autoport-codex01",
        "policy": "uniep-wgrad-environment-authority-v1",
        "review_verdict_id": "1001",
        "tool_blob_oid": tool_blob_oid,
        "tool_path": "tools/uniep_environment_authority.py",
        "tree": producer_tree,
    }
    if mutation == "wrong_tool":
        producer["tool_blob_oid"] = "0" * 40
    product = json.loads(json.dumps(product_fixture.product))
    if mutation == "alternate_product":
        product["commit"] = "9" * 40
    authority = {
        "environment": _task2_environment(),
        "producer": producer,
        "product": product,
        "schema": "uniep.environment-authority.v1",
        "status": "AUTHORIZED",
    }
    envelope = {
        "authority": authority,
        "authority_payload_sha256": hashlib.sha256(
            _task2_canonical_json(authority)
        ).hexdigest(),
        "schema": "uniep.environment-authority-envelope.v1",
    }
    if mutation == "payload_digest":
        envelope["authority_payload_sha256"] = "0" * 64
    if mutation == "self_hash":
        envelope["full_sha256"] = "1" * 64
    raw = _task2_canonical_json(envelope)
    if mutation == "wrong_blob":
        raw = b"not-json\n"
    elif mutation == "noncanonical":
        raw = json.dumps(envelope, indent=2).encode() + b"\n"

    relative = product_fixture.product["authority_path"]
    if mutation == "wrong_path":
        relative = f"wrong/{Path(relative).name}"
    if not authority_missing:
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    else:
        marker = work / "authorities" / "uniep" / "wgrad" / "README.md"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("authority missing\n", encoding="utf-8")
    authority_commit = _task2_commit(work, "authority")
    marker = work / "moving-marker"
    marker.write_text("moved\n", encoding="utf-8")
    moving_commit = _task2_commit(work, "moving authority ref")
    remote = tmp_path / "authority.git"
    _task2_git(tmp_path, "clone", "-q", "--bare", str(work), str(remote))
    _task2_git(remote, "update-ref", "refs/heads/main", authority_commit)
    _task2_git(remote, "update-server-info")
    return types.SimpleNamespace(
        authority_commit=authority_commit,
        authority_remote=remote,
        moving_commit=moving_commit,
        product=product_fixture,
        raw=raw,
    )


def _task2_configure_remotes(monkeypatch, suite, fixture) -> None:
    monkeypatch.setattr(suite, "AUTHORITY_REMOTE", str(fixture.authority_remote))
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))


class _Task2BasicAuthHandler(http.server.SimpleHTTPRequestHandler):
    expected_authorization = ""

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") == self.expected_authorization:
            return True
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="uniep-task2"')
        self.end_headers()
        return False

    def do_GET(self):
        if self._authorized():
            super().do_GET()

    def do_HEAD(self):
        if self._authorized():
            super().do_HEAD()

    def log_message(self, _format, *_args):
        return


@contextlib.contextmanager
def _task2_authenticated_remote(remote: Path, username: str, password: str):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    handler = functools.partial(
        _Task2BasicAuthHandler, directory=str(remote.parent)
    )
    _Task2BasicAuthHandler.expected_authorization = f"Basic {token}"
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{remote.name}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _task2_credential(suite, username: str, password: str):
    fd = os.memfd_create(
        "uniep-c01-credential", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    )
    payload = _task2_canonical_json({"password": password, "username": username})
    os.write(fd, payload)
    import fcntl

    seals = (
        fcntl.F_SEAL_SEAL
        | fcntl.F_SEAL_SHRINK
        | fcntl.F_SEAL_GROW
        | fcntl.F_SEAL_WRITE
    )
    fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
    os.set_inheritable(fd, True)
    return suite._credential_capability_from_c01_launcher(fd)


def test_task2_authority_valid_local_bare_remotes(monkeypatch, tmp_path):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    _task2_configure_remotes(monkeypatch, suite, fixture)

    anchor, envelope = suite._read_authority_object(
        fixture.product.feature_commit, tmp_path / "scratch", None
    )

    assert anchor.commit == fixture.authority_commit
    assert anchor.path == fixture.product.product["authority_path"]
    assert envelope.raw == fixture.raw
    assert envelope.product == fixture.product.product
    assert tuple(envelope.environment) == _TASK2_COMPONENTS


def test_task2_missing_real_authority_is_exact_and_pre_loader(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path, authority_missing=True)
    _task2_configure_remotes(monkeypatch, suite, fixture)
    loader_events = []
    monkeypatch.setattr(
        suite, "_load_benchmark_device_runtime", lambda: loader_events.append("loader")
    )

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._read_authority_object(
            fixture.product.feature_commit, tmp_path / "scratch", None
        )
    assert captured.value.code == "AUTHORITY_MISSING"
    assert loader_events == []


@pytest.mark.parametrize(
    "mutation",
    (
        "moving_ref",
        "wrong_path",
        "wrong_blob",
        "wrong_tool",
        "self_hash",
        "noncanonical",
        "payload_digest",
        "caller_path",
        "caller_digest",
        "local_config",
        "url_rewrite",
        "alternate_product",
    ),
)
def test_task2_authority_and_canonical_substitutions_fail_closed(
    monkeypatch, tmp_path, mutation
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(
        tmp_path,
        mutation if mutation in {
            "wrong_path",
            "wrong_blob",
            "wrong_tool",
            "self_hash",
            "noncanonical",
            "payload_digest",
            "alternate_product",
        } else None,
    )
    _task2_configure_remotes(monkeypatch, suite, fixture)
    scratch = tmp_path / "scratch"
    if mutation == "caller_path":
        monkeypatch.setenv("MOE_UNIEP_AUTHORITY_PATH", "/caller/authority.json")
    elif mutation == "caller_digest":
        monkeypatch.setenv("MOE_UNIEP_AUTHORITY_SHA256", "a" * 64)
    elif mutation == "url_rewrite":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.file:///caller/.insteadOf")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://")
    elif mutation == "local_config":
        scratch.mkdir()
        _task2_git(scratch, "init", "-q")
    elif mutation == "moving_ref":
        original = suite._run_git
        moved = False

        def move_after_first_authority_read(cwd, args, **kwargs):
            nonlocal moved
            completed = original(cwd, args, **kwargs)
            if (
                not moved
                and "ls-remote" in args
                and str(fixture.authority_remote) in args
            ):
                _task2_git(
                    fixture.authority_remote,
                    "update-ref",
                    "refs/heads/main",
                    fixture.moving_commit,
                )
                moved = True
            return completed

        monkeypatch.setattr(suite, "_run_git", move_after_first_authority_read)

    with pytest.raises(suite.AuthorityPreflightError):
        suite._read_authority_object(
            fixture.product.feature_commit, scratch, None
        )


def test_task2_authority_source_denominator_is_ordered_and_exact(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    _task2_configure_remotes(monkeypatch, suite, fixture)
    _, envelope = suite._read_authority_object(
        fixture.product.feature_commit, tmp_path / "scratch", None
    )
    assert tuple(item["path"] for item in envelope.product["sources"]) == (
        _TASK2_SOURCE_PATHS
    )
    assert suite.BACKWARD_EVIDENCE_SOURCES == _TASK2_SOURCE_PATHS


def test_task2_remote_auth_missing_capability_is_distinct_and_secret_free(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path, authority_missing=True)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    with _task2_authenticated_remote(fixture.authority_remote, "c01", "secret") as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._read_authority_object(
                fixture.product.feature_commit, tmp_path / "scratch", None
            )
    assert captured.value.code == "AUTHORITY_REMOTE_AUTH_FAILED"
    assert b"secret" not in captured.value.rendered_bytes


@pytest.mark.parametrize(
    "mutation",
    (
        "ambient_helper",
        "url_rewrite",
        "caller_credential",
        "missing_capability",
        "invalid_capability",
        "secret_output",
        "helper_identity_drift",
    ),
)
def test_task2_remote_auth_and_credential_mutations_fail_closed(
    monkeypatch, tmp_path, mutation
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path, authority_missing=True)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    username = "c01"
    password = "never-render-this-secret"
    server_password = password if mutation != "secret_output" else "different"
    side_effect = tmp_path / "ambient-helper-ran"
    if mutation in {"ambient_helper", "caller_credential"}:
        helper = tmp_path / "ambient-askpass"
        helper.write_text(
            "#!/bin/sh\nprintf ran > '" + str(side_effect) + "'\nprintf caller\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
        monkeypatch.setenv("GIT_ASKPASS", str(helper))
    if mutation == "url_rewrite":
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.file:///caller/.insteadOf")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "http://")

    capability = None
    if mutation not in {
        "ambient_helper",
        "url_rewrite",
        "caller_credential",
        "missing_capability",
    }:
        capability = _task2_credential(suite, username, password)
    if mutation == "invalid_capability":
        sealed_fd = capability.fd
        invalid_fd = os.memfd_create("unsealed-caller")
        capability = suite.CredentialCapability(
            fd=invalid_fd,
            helper_blob_oid=capability.helper_blob_oid,
            helper_sha256=capability.helper_sha256,
            policy=capability.policy,
        )
        os.close(sealed_fd)
    elif mutation == "helper_identity_drift":
        object.__setattr__(capability, "helper_sha256", "0" * 64)

    with _task2_authenticated_remote(
        fixture.authority_remote, username, server_password
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._read_authority_object(
                fixture.product.feature_commit,
                tmp_path / "scratch",
                capability,
            )
    assert captured.value.code == "AUTHORITY_REMOTE_AUTH_FAILED"
    assert password.encode() not in captured.value.rendered_bytes
    assert not side_effect.exists()
    if capability is not None:
        with pytest.raises(OSError):
            os.fstat(capability.fd)
    scratch = tmp_path / "scratch"
    if scratch.exists():
        assert list(scratch.glob(".uniep-askpass-*")) == []


def test_task2_authenticated_fixed_ref_absent_product_path_is_authority_missing(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path, authority_missing=True)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    password = "bounded-c01-secret"
    capability = _task2_credential(suite, "c01", password)
    receipt_identity = capability.receipt_identity()
    assert "fd" not in receipt_identity and "path" not in json.dumps(receipt_identity)
    with _task2_authenticated_remote(
        fixture.authority_remote, "c01", password
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._read_authority_object(
                fixture.product.feature_commit,
                tmp_path / "scratch",
                capability,
            )
    assert captured.value.code == "AUTHORITY_MISSING"
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert list((tmp_path / "scratch").glob(".uniep-askpass-*")) == []
