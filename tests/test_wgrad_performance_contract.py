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
import shutil
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


def _task2_sealed_credential_fd(username: str, password: str) -> int:
    fd = os.memfd_create(
        "uniep-test-only-credential", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
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
    return fd


def _task2_test_only_credential(suite, username: str, password: str):
    fd = _task2_sealed_credential_fd(username, password)
    try:
        return suite._credential_capability_from_test_only_launcher(fd)
    except Exception:
        os.close(fd)
        raise


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


def test_task2_local_bare_config_cannot_rewrite_fixed_remote(tmp_path):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    caller_bare = tmp_path / "caller-controlled.git"
    _task2_git(tmp_path, "init", "-q", "--bare", str(caller_bare))
    fixed_remote = "https://fixed.invalid/authority.git"
    _task2_git(
        caller_bare,
        "config",
        f"url.file://{fixture.authority_remote}/.insteadOf",
        fixed_remote,
    )

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._fetch_refs(
            tmp_path,
            caller_bare,
            fixed_remote,
            (("refs/heads/main", "refs/uniep/authority", fixture.authority_commit),),
            None,
        )
    assert captured.value.code == "AUTHORITY_INVALID"


@pytest.mark.parametrize("component_name", _TASK2_COMPONENTS)
def test_task2_component_resolver_id_is_exact(component_name):
    suite = _load_benchmark_suite()
    component = _task2_environment()[component_name]
    component["resolver_id"] = f"caller-selected-{component_name}"

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._validate_component(component_name, component)

    assert captured.value.code == "AUTHORITY_INVALID"


def test_task2_helper_identity_failure_removes_materialized_askpass(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    capability = _task2_test_only_credential(
        suite, "fixture", "helper-cleanup-secret"
    )
    original_read_bytes = Path.read_bytes

    def drift_materialized_helper(path):
        if path.name.startswith(".uniep-askpass-"):
            return b"identity drift\n"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", drift_materialized_helper)
    try:
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._write_askpass_helper(tmp_path, capability)
        assert captured.value.code == "AUTHORITY_REMOTE_AUTH_FAILED"
        assert list(tmp_path.glob(".uniep-askpass-*")) == []
    finally:
        os.close(capability.fd)


def test_task2_remote_auth_missing_capability_is_distinct_and_secret_free(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path, authority_missing=True)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    with _task2_authenticated_remote(
        fixture.authority_remote, "fixture", "secret"
    ) as url:
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
    username = "fixture"
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
        capability = _task2_test_only_credential(suite, username, password)
    if mutation == "invalid_capability":
        sealed_fd = capability.fd
        invalid_fd = os.memfd_create("unsealed-caller")
        capability = suite.CredentialCapability(
            fd=invalid_fd,
            helper_blob_oid=capability.helper_blob_oid,
            helper_sha256=capability.helper_sha256,
            policy=capability.policy,
            provenance=capability.provenance,
        )
        os.close(sealed_fd)
    elif mutation == "helper_identity_drift":
        object.__setattr__(capability, "helper_sha256", "0" * 64)

    with _task2_authenticated_remote(
        fixture.authority_remote, username, server_password
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            if capability is None:
                suite._read_authority_object(
                    fixture.product.feature_commit,
                    tmp_path / "scratch",
                    capability,
                )
            else:
                suite._read_test_only_authority_object(
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
    password = "synthetic-loopback-secret"
    capability = _task2_test_only_credential(suite, "fixture", password)
    receipt_identity = capability.receipt_identity()
    assert "fd" not in receipt_identity and "path" not in json.dumps(receipt_identity)
    assert receipt_identity["status"] == "TEST_ONLY"
    with _task2_authenticated_remote(
        fixture.authority_remote, "fixture", password
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._read_test_only_authority_object(
                fixture.product.feature_commit,
                tmp_path / "scratch",
                capability,
            )
    assert captured.value.code == "AUTHORITY_MISSING"
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert list((tmp_path / "scratch").glob(".uniep-askpass-*")) == []


def test_task2_test_only_authenticated_host_closed_loop_has_explicit_receipt(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    password = "synthetic-loopback-only"
    capability = _task2_test_only_credential(suite, "fixture", password)
    device_events = []
    monkeypatch.setattr(
        suite, "_load_benchmark_device_runtime", lambda: device_events.append("loader")
    )

    with _task2_authenticated_remote(
        fixture.authority_remote, "fixture", password
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        anchor, envelope, receipt = suite._read_test_only_authority_object(
            fixture.product.feature_commit,
            tmp_path / "scratch",
            capability,
        )

    assert anchor.commit == fixture.authority_commit
    assert envelope.product == fixture.product.product
    assert receipt["schema"] == "uniep.test-only-authority-read-receipt.v1"
    assert receipt["status"] == "TEST_ONLY"
    assert receipt["production_authority_status"] == "NOT_ESTABLISHED"
    assert receipt["credential"]["status"] == "TEST_ONLY"
    assert receipt["credential"]["provenance"] == "TEST_ONLY_INJECTED_LAUNCHER"
    assert receipt["product_commit"] == fixture.product.feature_commit
    assert receipt["authority_anchor"]["blob_oid"] == anchor.blob_oid
    assert receipt["source_count"] == len(_TASK2_SOURCE_PATHS)
    assert tuple(receipt["environment_components"]) == _TASK2_COMPONENTS
    assert receipt["device_actions"] == 0
    rendered = json.dumps(receipt, sort_keys=True)
    assert password not in rendered
    assert str(tmp_path) not in rendered
    assert "fd" not in receipt["credential"]
    assert device_events == []
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert list((tmp_path / "scratch").glob(".uniep-askpass-*")) == []


def test_task2_test_only_injection_refuses_production_remotes_before_git(tmp_path):
    suite = _load_benchmark_suite()
    capability = _task2_test_only_credential(suite, "fixture", "local-only")

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._read_test_only_authority_object(
            "1" * 40,
            tmp_path / "scratch",
            capability,
        )

    assert captured.value.code == "TEST_ONLY_REQUIRED"
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert not (tmp_path / "scratch").exists()


def test_task2_product_minted_c01_capability_never_claims_production_pass():
    suite = _load_benchmark_suite()
    fd = _task2_sealed_credential_fd("caller", "self-minted")
    capability = suite._credential_capability_from_c01_launcher(fd)
    try:
        receipt = capability.receipt_identity()
        assert receipt["status"] == "PRODUCTION_AUTHORITY_NOT_ESTABLISHED"
        assert receipt["provenance"] == "UNVERIFIED_C01_ISSUER"
        assert receipt["status"] != "TEST_ONLY"
    finally:
        os.close(capability.fd)


def test_task2_test_only_capability_requires_explicit_test_only_reader(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    _task2_configure_remotes(monkeypatch, suite, fixture)
    capability = _task2_test_only_credential(suite, "fixture", "local-only")

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._read_authority_object(
            fixture.product.feature_commit,
            tmp_path / "scratch",
            capability,
        )

    assert captured.value.code == "TEST_ONLY_REQUIRED"
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert not (tmp_path / "scratch").exists()


def test_task2_test_only_reader_rejects_unverified_c01_claim(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    _task2_configure_remotes(monkeypatch, suite, fixture)
    fd = _task2_sealed_credential_fd("caller", "self-minted")
    capability = suite._credential_capability_from_c01_launcher(fd)

    with pytest.raises(suite.AuthorityPreflightError) as captured:
        suite._read_test_only_authority_object(
            fixture.product.feature_commit,
            tmp_path / "scratch",
            capability,
        )

    assert captured.value.code == "TEST_ONLY_REQUIRED"
    with pytest.raises(OSError):
        os.fstat(capability.fd)
    assert not (tmp_path / "scratch").exists()


def test_task2_unknown_credential_policy_with_none_provenance_fails_closed():
    suite = _load_benchmark_suite()
    fd = _task2_sealed_credential_fd("fixture", "unknown-policy")
    capability = suite.CredentialCapability(
        fd=fd,
        helper_blob_oid=suite._ASKPASS_HELPER_BLOB_OID,
        helper_sha256=suite._ASKPASS_HELPER_SHA256,
        policy="unknown-policy",
        provenance=None,
    )
    try:
        with pytest.raises(suite.AuthorityPreflightError) as captured:
            suite._validate_credential_capability(capability)
        assert captured.value.code == "AUTHORITY_REMOTE_AUTH_FAILED"
    finally:
        os.close(fd)


def test_task2_test_only_closed_loop_closes_capability_exactly_once(
    monkeypatch, tmp_path
):
    suite = _load_benchmark_suite()
    fixture = _task2_authority_fixture(tmp_path)
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(fixture.product.remote))
    capability = _task2_test_only_credential(suite, "fixture", "close-once")
    close_calls = []
    original_close = suite._close_credential

    def counted_close(candidate):
        if candidate is not None:
            close_calls.append(candidate.fd)
        original_close(candidate)

    monkeypatch.setattr(suite, "_close_credential", counted_close)
    with _task2_authenticated_remote(
        fixture.authority_remote, "fixture", "close-once"
    ) as url:
        monkeypatch.setattr(suite, "AUTHORITY_REMOTE", url)
        _, _, receipt = suite._read_test_only_authority_object(
            fixture.product.feature_commit,
            tmp_path / "scratch",
            capability,
        )

    assert receipt["status"] == "TEST_ONLY"
    assert close_calls == [capability.fd]
    with pytest.raises(OSError):
        os.fstat(capability.fd)


def _task3_blob(repo: Path, commit: str, relative: str) -> tuple[str, bytes, int]:
    record = _task2_git(repo, "ls-tree", commit, "--", relative).decode().strip()
    mode, kind, oid, observed = record.split(maxsplit=3)
    assert kind == "blob" and observed == relative
    return oid, _task2_git(repo, "cat-file", "blob", oid), int(mode, 8)


def _task3_fixture(monkeypatch, tmp_path: Path):
    bigop_work = tmp_path / "bigop-work"
    bigop_work.mkdir()
    _task2_git(bigop_work, "init", "-q", "-b", "main")
    _task2_git(bigop_work, "config", "user.name", "Task3 Bigop Fixture")
    _task2_git(bigop_work, "config", "user.email", "task3@example.invalid")
    (bigop_work / "include").mkdir()
    (bigop_work / "include" / "bigop.h").write_text("#define BIGOP 1\n")
    executable = bigop_work / "probe.sh"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    bigop_commit = _task2_commit(bigop_work, "bigop fixture")
    bigop_remote = tmp_path / "bigop.git"
    _task2_git(tmp_path, "clone", "-q", "--bare", str(bigop_work), str(bigop_remote))

    product_work = tmp_path / "task3-product-work"
    product_work.mkdir()
    _task2_git(product_work, "init", "-q", "-b", "main")
    _task2_git(product_work, "config", "user.name", "Task3 Product Fixture")
    _task2_git(product_work, "config", "user.email", "task3@example.invalid")
    for index, relative in enumerate(_TASK2_SOURCE_PATHS):
        target = product_work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"anchor-{index}\n", encoding="utf-8")
    modules = (
        '[submodule "3rdparty/bigop"]\n'
        "\tpath = 3rdparty/bigop\n"
        f"\turl = {bigop_remote}\n"
    )
    (product_work / ".gitmodules").write_text(modules, encoding="utf-8")
    _task2_git(product_work, "add", "-A")
    _task2_git(
        product_work,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{bigop_commit},3rdparty/bigop",
    )
    _task2_git(product_work, "commit", "-qm", "task3 main")
    main_commit = _task2_git(product_work, "rev-parse", "HEAD").decode().strip()
    runner = product_work / _TASK2_SOURCE_PATHS[0]
    runner.write_text("anchor-0\nfeature\n", encoding="utf-8")
    _task2_git(product_work, "add", _TASK2_SOURCE_PATHS[0])
    _task2_git(product_work, "commit", "-qm", "task3 feature")
    feature_commit = _task2_git(product_work, "rev-parse", "HEAD").decode().strip()
    tree = _task2_git(
        product_work, "rev-parse", f"{feature_commit}^{{tree}}"
    ).decode().strip()
    product_remote = tmp_path / "task3-product.git"
    _task2_git(
        tmp_path, "clone", "-q", "--bare", str(product_work), str(product_remote)
    )
    _task2_git(product_remote, "update-ref", "refs/heads/main", main_commit)
    _task2_git(
        product_remote,
        "update-ref",
        "refs/heads/codex02/uniep-triton-wgrad-1p5x-20260810",
        feature_commit,
    )

    sources = []
    for relative in _TASK2_SOURCE_PATHS:
        oid, raw, _mode = _task3_blob(product_work, feature_commit, relative)
        sources.append(
            {"blob_oid": oid, "path": relative, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    product = {
        "authority_path": (
            f"authorities/uniep/wgrad/{feature_commit}/environment-authority.json"
        ),
        "commit": feature_commit,
        "feature_ref": "refs/heads/codex02/uniep-triton-wgrad-1p5x-20260810",
        "main_is_ancestor": True,
        "main_ref": "refs/heads/main",
        "observed_main": main_commit,
        "remote": str(product_remote),
        "sole_parent": main_commit,
        "sources": sources,
        "tree": tree,
    }
    environment = _task2_environment()
    bigop_members = []
    for relative in ("include/bigop.h", "probe.sh"):
        oid, raw, mode = _task3_blob(bigop_work, bigop_commit, relative)
        del oid
        bigop_members.append(
            {
                "elf_build_id": None,
                "kind": "file",
                "mode": mode,
                "name": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
    encoded_members = _task2_canonical_json(bigop_members)
    environment["bigop"] = {
        "identity": f"gitlink:{bigop_commit}",
        "manifest_sha256": hashlib.sha256(encoded_members).hexdigest(),
        "member_count": len(bigop_members),
        "members": bigop_members,
        "resolver_id": "uniep-bigop-resolver-v1",
        "total_bytes": sum(item["size"] for item in bigop_members),
    }
    suite = _load_benchmark_suite()
    monkeypatch.setattr(suite, "PRODUCT_REMOTE", str(product_remote))
    monkeypatch.setattr(suite, "BIGOP_REMOTE", str(bigop_remote))
    modules_oid, _modules_raw, _modules_mode = _task3_blob(
        product_work, feature_commit, ".gitmodules"
    )
    monkeypatch.setattr(
        suite, "BIGOP_GITMODULES_BLOB_OID", modules_oid, raising=False
    )
    monkeypatch.setattr(
        suite, "BIGOP_GITLINK_COMMIT", bigop_commit, raising=False
    )
    envelope = suite.AuthorityEnvelope(
        raw=b"{}\n",
        payload_sha256="a" * 64,
        product=product,
        environment=environment,
        producer={},
    )
    return types.SimpleNamespace(
        bigop_commit=bigop_commit,
        envelope=envelope,
        product_work=product_work,
        suite=suite,
    )


def test_task3_snapshot_is_git_derived_content_addressed_and_reusable(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "benchmark.py").write_text("malicious\n", encoding="utf-8")

    first = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    fixture.suite._verify_product_snapshot(first, fixture.envelope)
    second = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )

    assert second == first
    assert first.tree == fixture.envelope.product["tree"]
    assert first.root.name == f"{first.tree}-{first.manifest_sha256}"
    assert first.root_device == first.root.stat().st_dev
    assert first.root_inode == first.root.stat().st_ino
    assert any(
        item.kind == "gitlink"
        and item.path == "3rdparty/bigop"
        and item.commit_oid == fixture.bigop_commit
        for item in first.entries
    )
    assert (first.root / "3rdparty" / "bigop" / "include" / "bigop.h").is_file()
    assert str(ambient) not in first.canonical_manifest.decode()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "symlink",
        "hardlink",
        "content",
        "mode",
        "bigop_content",
        "root_replacement",
        "gitlink_identity",
        "gitmodules_url",
    ),
)
def test_task3_snapshot_and_bigop_mutations_fail_closed(
    monkeypatch, tmp_path, mutation
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    snapshot = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    target = snapshot.root / "benchmark" / "layer" / "bench_moe_suite.py"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (snapshot.root / "extra").write_text("extra\n", encoding="utf-8")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(snapshot.root / "conftest.py")
    elif mutation == "hardlink":
        foreign = tmp_path / "foreign"
        foreign.write_bytes(target.read_bytes())
        target.unlink()
        os.link(foreign, target)
    elif mutation == "content":
        target.write_text("mutated\n", encoding="utf-8")
    elif mutation == "mode":
        target.chmod(0o755)
    elif mutation == "bigop_content":
        (snapshot.root / "3rdparty" / "bigop" / "probe.sh").write_text(
            "#!/bin/sh\nexit 7\n", encoding="utf-8"
        )
    elif mutation == "root_replacement":
        held = tmp_path / "held-root"
        snapshot.root.rename(held)
        shutil.copytree(held, snapshot.root)
    elif mutation == "gitlink_identity":
        fixture.envelope.environment["bigop"]["identity"] = "gitlink:" + "0" * 40
    elif mutation == "gitmodules_url":
        monkeypatch.setattr(fixture.suite, "BIGOP_REMOTE", str(tmp_path / "wrong.git"))

    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._verify_product_snapshot(snapshot, fixture.envelope)


def test_task3_snapshot_never_imports_or_calls_device(monkeypatch, tmp_path):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    device_events = []
    monkeypatch.setattr(
        fixture.suite,
        "_load_benchmark_device_runtime",
        lambda: device_events.append("loader"),
    )

    snapshot = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    fixture.suite._verify_product_snapshot(snapshot, fixture.envelope)

    assert device_events == []
    assert "torch_npu" not in sys.modules


def test_task3_enumeration_to_open_same_bytes_swap_is_rejected(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    snapshot = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    target = snapshot.root / "benchmark" / "layer" / "bench_moe_suite.py"
    held = tmp_path / "enumerated-leaf"
    original_open = fixture.suite.os.open
    replaced = False

    def swap_before_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if (
            not replaced
            and path == "bench_moe_suite.py"
            and flags & os.O_DIRECTORY == 0
            and kwargs.get("dir_fd") is not None
        ):
            target.rename(held)
            shutil.copy2(held, target)
            replaced = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fixture.suite.os, "open", swap_before_open)
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._verify_product_snapshot(snapshot, fixture.envelope)
    assert replaced is True


def test_task3_partial_content_addressed_generation_is_not_reused(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    seed = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "seed-snapshots"
    )
    root = tmp_path / "snapshots"
    root.mkdir(mode=0o700)
    partial = root / seed.root.name
    partial.mkdir(mode=0o700)
    (partial / "partial").write_text("not a snapshot\n", encoding="utf-8")

    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._materialize_product_snapshot(fixture.envelope, root)

    assert (partial / "partial").read_text(encoding="utf-8") == "not a snapshot\n"


def test_task3_late_extra_after_directory_enumeration_is_rejected(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    snapshot = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    original_listdir = fixture.suite.os.listdir
    injected = False

    def inject_after_enumeration(path):
        nonlocal injected
        names = original_listdir(path)
        if not injected and isinstance(path, int):
            (snapshot.root / "late-extra").write_text("late\n", encoding="utf-8")
            injected = True
        return names

    monkeypatch.setattr(fixture.suite.os, "listdir", inject_after_enumeration)
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._verify_product_snapshot(snapshot, fixture.envelope)
    assert injected is True


def test_task3_late_descendant_during_later_sibling_walk_is_rejected(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    snapshot = fixture.suite._materialize_product_snapshot(
        fixture.envelope, tmp_path / "snapshots"
    )
    original_listdir = fixture.suite.os.listdir
    injected = False

    def inject_while_walking_later_sibling(path):
        nonlocal injected
        names = original_listdir(path)
        if isinstance(path, int) and not injected:
            resolved = os.readlink(f"/proc/self/fd/{path}")
            if resolved.endswith("/benchmark"):
                (snapshot.root / "3rdparty" / "bigop" / "late-extra").write_text(
                    "late\n", encoding="utf-8"
                )
                injected = True
        return names

    monkeypatch.setattr(
        fixture.suite.os, "listdir", inject_while_walking_later_sibling
    )
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._verify_product_snapshot(snapshot, fixture.envelope)
    assert injected is True


def test_task3_snapshot_parent_replacement_during_plan_is_rejected(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    root = tmp_path / "snapshots"
    fixture.suite._materialize_product_snapshot(fixture.envelope, root)
    held = tmp_path / "held-snapshot-parent"
    original_derive = fixture.suite._derive_snapshot_plan
    replaced = False

    def replace_parent_after_plan(*args, **kwargs):
        nonlocal replaced
        plan = original_derive(*args, **kwargs)
        if not replaced:
            root.rename(held)
            shutil.copytree(held, root)
            replaced = True
        return plan

    monkeypatch.setattr(
        fixture.suite, "_derive_snapshot_plan", replace_parent_after_plan
    )
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._materialize_product_snapshot(fixture.envelope, root)
    assert replaced is True


def test_task3_matching_envelope_cannot_override_required_bigop_gitlink(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(fixture.suite, "BIGOP_GITLINK_COMMIT", "0" * 40)

    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._materialize_product_snapshot(
            fixture.envelope, tmp_path / "snapshots"
        )


@pytest.mark.parametrize("failure_stage", ("create", "publish"))
def test_task3_failed_materialization_reclaims_owned_staging(
    monkeypatch, tmp_path, failure_stage
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    root = tmp_path / "snapshots"
    target = (
        "_create_snapshot_tree" if failure_stage == "create" else "_publish_snapshot"
    )

    def fail(*_args, **_kwargs):
        raise fixture.suite.AuthorityPreflightError(
            "SNAPSHOT_PUBLISH_FAILED", failure_stage
        )

    monkeypatch.setattr(fixture.suite, target, fail)
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._materialize_product_snapshot(fixture.envelope, root)

    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_task3_staging_open_failure_reclaims_created_directory(
    monkeypatch, tmp_path
):
    fixture = _task3_fixture(monkeypatch, tmp_path)
    root = tmp_path / "snapshots"
    original_open = fixture.suite.os.open
    failed = False

    def fail_first_staging_open(path, flags, *args, **kwargs):
        nonlocal failed
        if (
            not failed
            and isinstance(path, str)
            and ".staging-" in path
            and flags & os.O_DIRECTORY
        ):
            failed = True
            raise OSError(11, "injected staging open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fixture.suite.os, "open", fail_first_staging_open)
    with pytest.raises(fixture.suite.AuthorityPreflightError):
        fixture.suite._materialize_product_snapshot(fixture.envelope, root)

    assert failed is True
    assert root.is_dir()
    assert list(root.iterdir()) == []
