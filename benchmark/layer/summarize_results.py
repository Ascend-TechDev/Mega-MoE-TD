#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Extract classic, MoonEP, and event-slice tables from benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "forward"
DEFAULT_PATTERNS = (
    "bench_forward_suite_w*.json",
    "bench_moonep_forward_suite_w*.json",
    "bench_full_forward_*_grouped_routefp32_w*.json",
)
LEGACY_STAGES = (
    ("preprocess", "preprocess"),
    ("dispatch_fc1", "dispatch+FC1"),
    ("weighted_swiglu", "weighted SwiGLU"),
    ("fc2_combine", "FC2+combine"),
)
CURRENT_STAGES = (
    ("preprocess", "preprocess"),
    ("dispatch_fc1", "dispatch+FC1"),
    ("weighted_swiglu_fc2_combine", "weighted SwiGLU+FC2+combine"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Ascend Mega-MoE benchmark JSON. When no file is given, "
            "all stable result files in results/forward are used."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="result JSON files or directories (default: all stable result JSON)",
    )
    parser.add_argument(
        "--section",
        choices=("all", "full", "stages"),
        default="all",
        help="table section to print (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "tsv"),
        default="markdown",
        help="output format (default: markdown)",
    )
    parser.add_argument(
        "--stat",
        choices=("median", "mean", "min", "max"),
        default="median",
        help="latency statistic used in tables and speedups (default: median)",
    )
    return parser.parse_args()


def _expand_paths(requested: list[Path]) -> list[Path]:
    if not requested:
        paths = sorted(
            path
            for pattern in DEFAULT_PATTERNS
            for path in RESULTS_DIR.glob(pattern)
        )
    else:
        paths = []
        for path in requested:
            if path.is_dir():
                paths.extend(
                    sorted(
                        candidate
                        for pattern in DEFAULT_PATTERNS
                        for candidate in path.glob(pattern)
                    )
                )
            else:
                paths.append(path)

    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise ValueError("no benchmark result JSON files were found")
    missing = [str(path) for path in unique_paths if not path.is_file()]
    if missing:
        raise ValueError("result file does not exist: " + ", ".join(missing))
    return unique_paths


def _load_entries(paths: list[Path]) -> list[dict]:
    entries = []
    for path in paths:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, list):
            file_entries = payload
        elif (
            isinstance(payload, dict)
            and payload.get("schema_version") == 1
            and isinstance(payload.get("cases"), list)
        ):
            file_entries = payload["cases"]
        else:
            raise ValueError(f"{path}: expected a legacy array or schema-v1 envelope")
        for index, entry in enumerate(file_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: entry {index} is not a JSON object")
            correctness = entry.get("correctness_gate") or entry.get(
                "correctness_gates", {}
            )
            status = correctness.get("status")
            if status not in {
                "passed_before_timing",
                "passed_before_and_after_timing",
            }:
                print(
                    f"warning: {path.name} entry {index} has correctness status {status!r}",
                    file=sys.stderr,
                )
            entries.append(entry)

    entries.sort(
        key=lambda entry: (
            str(entry.get("model", entry.get("model_profile", ""))),
            int(entry.get("world_size", 0)),
            int(entry.get("tokens_per_rank", 0)),
            str(entry.get("case_id", entry.get("config", ""))),
        )
    )
    return entries


def _stat_value(stats: dict, stat: str) -> float:
    key = f"{stat}_ms"
    if key not in stats:
        raise ValueError(f"result entry does not contain {key!r}")
    return float(stats[key])


def _speedup(grouped_ms: float, ascend_ms: float) -> float:
    if ascend_ms == 0:
        return float("inf")
    return grouped_ms / ascend_ms


def _format_ms(value: float) -> str:
    return f"{value:.3f}"


def _format_speedup(value: float) -> str:
    return f"{value:.3f}x"


def _full_rows(entries: list[dict], stat: str) -> list[list[str]]:
    rows = []
    for entry in entries:
        metrics = entry["metrics"]
        if any(key.startswith("triton_moonep_balanced_") for key in metrics):
            continue
        ascend_full = _stat_value(metrics["ascend_full_direct_e2e_ms"], stat)
        grouped_full = _stat_value(metrics["torch_npu_grouped_hccl_full_direct_e2e_ms"], stat)
        rows.append(
            [
                str(entry.get("model", entry.get("model_profile", ""))),
                str(entry["world_size"]),
                str(entry.get("case_id", entry.get("config", ""))),
                str(entry["tokens_per_rank"]),
                _format_ms(ascend_full),
                _format_ms(grouped_full),
                _format_speedup(_speedup(grouped_full, ascend_full)),
            ]
        )
    return rows


def _moonep_rows(entries: list[dict], stat: str) -> list[list[str]]:
    rows = []
    for entry in entries:
        metrics = entry["metrics"]
        balanced_key = "triton_moonep_balanced_full_direct_e2e_ms"
        unbalanced_key = "triton_unbalanced_full_direct_e2e_ms"
        if balanced_key not in metrics:
            continue
        balanced_ms = _stat_value(metrics[balanced_key], stat)
        unbalanced_ms = _stat_value(metrics[unbalanced_key], stat)
        rows.append(
            [
                str(entry.get("model", entry.get("model_profile", ""))),
                str(entry["world_size"]),
                str(entry.get("case_id", entry.get("config", ""))),
                str(entry["tokens_per_rank"]),
                _format_ms(balanced_ms),
                _format_ms(unbalanced_ms),
                _format_speedup(_speedup(unbalanced_ms, balanced_ms)),
            ]
        )
    return rows


def _stage_rows(entries: list[dict], stat: str) -> tuple[list[list[str]], list[str]]:
    rows = []
    skipped = []
    for entry in entries:
        diagnostics = entry.get("diagnostics")
        if any(
            key.startswith("triton_moonep_balanced_")
            for key in entry.get("metrics", {})
        ):
            continue
        if not diagnostics:
            skipped.append(
                f"{entry.get('model', entry.get('model_profile'))}/"
                f"W{entry.get('world_size')}/"
                f"{entry.get('case_id', entry.get('config'))}"
            )
            continue
        ascend_slices = diagnostics["ascend_event_slices"]
        grouped_slices = diagnostics["torch_npu_grouped_hccl_event_slices"]
        stages = (
            CURRENT_STAGES
            if "weighted_swiglu_fc2_combine_event_ms" in ascend_slices
            else LEGACY_STAGES
        )
        for stage_key, stage_label in stages:
            stats_key = f"{stage_key}_event_ms"
            ascend_ms = _stat_value(ascend_slices[stats_key], stat)
            grouped_ms = _stat_value(grouped_slices[stats_key], stat)
            rows.append(
                [
                    str(entry.get("model", entry.get("model_profile", ""))),
                    str(entry["world_size"]),
                    str(entry.get("case_id", entry.get("config", ""))),
                    stage_label,
                    _format_ms(ascend_ms),
                    _format_ms(grouped_ms),
                    _format_speedup(_speedup(grouped_ms, ascend_ms)),
                ]
            )
    return rows, skipped


def _print_markdown(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"## {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    print()


def _print_tsv(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print(f"# {title}")
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    print()


def main() -> int:
    args = _parse_args()
    try:
        paths = _expand_paths(args.files)
        entries = _load_entries(paths)
        full_rows = _full_rows(entries, args.stat)
        moonep_rows = _moonep_rows(entries, args.stat)
        stage_rows, skipped_stages = _stage_rows(entries, args.stat)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    printer = _print_markdown if args.format == "markdown" else _print_tsv
    if args.section in ("all", "full"):
        if full_rows:
            printer(
                f"Full forward ({args.stat})",
                [
                    "Model",
                    "W",
                    "Case",
                    "tokens/rank",
                    "Ascend full/ms",
                    "Grouped full/ms",
                    "full speedup",
                ],
                full_rows,
            )
        if moonep_rows:
            printer(
                f"MoonEP balanced forward ({args.stat})",
                [
                    "Model",
                    "W",
                    "Case",
                    "tokens/rank",
                    "Balanced one-shot/ms",
                    "Unbalanced/ms",
                    "balanced speedup",
                ],
                moonep_rows,
            )
    if args.section in ("all", "stages"):
        printer(
            f"Independent event slices ({args.stat})",
            ["Model", "W", "Case", "Stage", "Ascend/ms", "Grouped/ms", "speedup"],
            stage_rows,
        )

    for case in skipped_stages:
        print(f"warning: no four-stage diagnostics recorded for {case}", file=sys.stderr)
    if args.format == "markdown":
        print("> speedup = Grouped latency / Ascend latency; values greater than 1 mean Ascend is faster.")
        if moonep_rows:
            print(
                "> MoonEP balanced speedup = unbalanced Triton latency / "
                "balanced one-shot Triton latency; every balanced sample refills "
                "replica weights and represents one layer forward."
            )
        print("> Event slices are sampled independently and must not be added to reconstruct full E2E.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
