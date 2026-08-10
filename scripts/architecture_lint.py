#!/usr/bin/env python3
"""Strict anti-rot lint for the current-human baseline architecture."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys


CANONICAL_MARKER = "CANONICAL_ARCHITECTURE_SOURCE: current-human-baseline-v1"
REQUIRED_PATHS = (
    "docs/design/HARNESS_DESIGN_PHILOSOPHY.md",
    "benchmark/contracts/current_human_baseline_v1.schema.json",
    "benchmark/baseline_contract.py",
    "benchmark/current_human_baseline.py",
    "benchmark/providers/current_main.py",
    "benchmark/providers/grouped_hccl.py",
    "scripts/architecture_lint.py",
    "tests/host/test_architecture_contract.py",
    "tests/host/test_baseline_contract.py",
    "tests/host/test_baseline_runner.py",
)
EXPECTED_ARMS = (
    "unfused_grouped_hccl_forward",
    "fused_current_main_forward",
    "backward_default_torch_wgrad",
    "backward_optin_triton_wgrad",
)
EXPECTED_SHAPES = (4096, 8192, 16384)
DEVICE_IMPORTS = {"torch", "torch_npu", "triton", "shmem"}


def _load_contract(root: Path):
    path = root / "benchmark" / "baseline_contract.py"
    spec = importlib.util.spec_from_file_location("_architecture_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load baseline contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _top_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def lint(root: Path) -> list[str]:
    findings: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            findings.append(f"required path missing: {relative}")

    architecture = root / REQUIRED_PATHS[0]
    if architecture.is_file():
        text = architecture.read_text(encoding="utf-8")
        if text.count(CANONICAL_MARKER) != 1:
            findings.append("canonical architecture marker must occur exactly once")
        for heading in ("## 2. Component boundaries", "## 4. Evidence principles"):
            if heading not in text:
                findings.append(f"required section missing: {heading}")

    schema_path = root / REQUIRED_PATHS[1]
    if schema_path.is_file():
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            arms = tuple(schema["$defs"]["arm_id"]["enum"])
            shapes = tuple(schema["$defs"]["shape_result"]["properties"]["tokens_per_rank"]["enum"])
            if arms != EXPECTED_ARMS:
                findings.append("schema four-arm set drift")
            if shapes != EXPECTED_SHAPES:
                findings.append("schema shape set drift")
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            findings.append(f"schema is not inspectable: {error}")

    contract_path = root / REQUIRED_PATHS[2]
    if contract_path.is_file():
        try:
            contract = _load_contract(root)
            if tuple(contract.ARM_IDS) != EXPECTED_ARMS:
                findings.append("contract four-arm set drift")
            if tuple(contract.TOKENS_PER_RANK) != EXPECTED_SHAPES:
                findings.append("contract shape set drift")
        except (ImportError, AttributeError, RuntimeError, SyntaxError) as error:
            findings.append(f"contract import failed: {error}")

    runner = root / REQUIRED_PATHS[3]
    if runner.is_file():
        text = runner.read_text(encoding="utf-8")
        if "AUTHORITATIVE_BASELINE_RUNNER = True" not in text:
            findings.append("single authoritative runner marker missing")
        if "README.md" in text:
            findings.append("authoritative runner contains README fallback")
        if "except Exception" in text:
            findings.append("authoritative runner contains broad exception handling")
        imported = _top_level_import_roots(runner)
        if imported & DEVICE_IMPORTS:
            findings.append("authoritative runner imports device modules before identity validation")

    for relative in (REQUIRED_PATHS[4], REQUIRED_PATHS[5]):
        provider = root / relative
        if provider.is_file():
            imported = _top_level_import_roots(provider)
            if imported & DEVICE_IMPORTS:
                findings.append(f"provider imports device modules at module import: {relative}")
            text = provider.read_text(encoding="utf-8")
            if '"test_only": False' not in text:
                findings.append(f"provider real-source marker missing: {relative}")

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="return nonzero for any finding")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    findings = lint(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"architecture: ERROR: {finding}")
        return 1 if args.strict else 0
    print("architecture: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
