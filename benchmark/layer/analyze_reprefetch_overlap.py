"""Summarize actual replica traffic, serial refresh, and ideal-residency gap.

The ideal control moves real UDMA refresh+publication before timing. It must
pass the same gradient oracle and use the same compiled mega binary. Its gap
includes launch-side work and bandwidth contention, not just ready spinning.
"""

import argparse
import json
import random
import statistics
from pathlib import Path


def quantile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def describe(values):
    return {"n": len(values), "median_ms": statistics.median(values),
            "mean_ms": statistics.mean(values), "p95_ms": quantile(values, .95),
            "min_ms": min(values), "max_ms": max(values)}


def analyze(run):
    result = json.loads((run / "benchmark_result.json").read_text())
    assert result["occupancy_gate"]["status"] == "passed"
    assert result["correctness"]["status"] == "passed_all_ranks"
    ranks = [json.loads((run / f"moonep_rank{rank}.json").read_text())
             for rank in range(8)]
    table = ranks[0]["experts_to_copy"]
    assert all(row["experts_to_copy"] == table for row in ranks)
    epn, world = len(table[0]), len(table)
    metadata = json.loads((run / "metadata.json").read_text())
    shape = metadata["case"]
    per_copy = 3 * shape["hidden"] * shape["ffn"] * 2
    sends = [[0] * world for _ in range(world)]
    ids = set()
    for dst, row in enumerate(table):
        for expert in row:
            if expert >= 0 and expert // epn != dst:
                sends[expert // epn][dst] += 1
                ids.add(expert)
    totals = list(map(sum, sends))
    copies = sum(totals)
    assert copies * per_copy == ranks[0]["global_reprefetch_payload_bytes"]
    traffic = {"sent_expert_copies": copies, "unique_experts": len(ids),
               "bytes_per_expert_copy": per_copy, "total_bytes": copies * per_copy,
               "down_bytes": copies * per_copy // 3,
               "gate_up_bytes": copies * per_copy * 2 // 3,
               "largest_owner_peer_bytes": max(map(max, sends)) * per_copy,
               "owner_peer_matrix": sends, "sent_by_rank": totals,
               "received_by_rank": [sum(row[d] for row in sends) for d in range(world)]}
    hashes = [json.loads((run / f"overlap_binary_rank{rank}.json").read_text())
              for rank in range(world)]
    assert all(h["candidate"] and h["candidate"] == h["reprefetch_preloaded"]
               for h in hashes)
    early_paths = [run / f"overlap_early_binary_rank{rank}.json"
                   for rank in range(world)]
    if any(path.exists() for path in early_paths):
        assert all(path.exists() for path in early_paths)
        early_hashes = [json.loads(path.read_text()) for path in early_paths]
        assert all(h["candidate"] and h["candidate"] == h["reprefetch_preloaded"]
                   for h in early_hashes)
    timings = result["timings"]
    normal = timings["candidate"]["samples_ms"]
    resident = timings["reprefetch_preloaded"]["samples_ms"]
    diffs = [x - y for x, y in zip(normal, resident)]
    rng = random.Random(2909)
    bootstrap = [statistics.median(rng.choices(diffs, k=len(diffs)))
                 for _ in range(10000)]
    gap = {"paired_difference": describe(diffs),
           "paired_median_bootstrap_95pct_ms": [quantile(bootstrap, .025), quantile(bootstrap, .975)],
           "median_difference_ms": statistics.median(normal) - statistics.median(resident)}
    serial = []
    launch_name = "reprefetch_udma_standalone"
    for rank in range(world):
        path = run / f"reprefetch_phases_rank{rank}.json"
        if not path.exists():
            break
        captures = json.loads(path.read_text())["captures"][launch_name]["samples"]
        # Sum on one rank/current stream FIRST, then take rank MAX.
        serial.append([sum(x["ms"] for x in sample["launches"]
                           if x["name"] in ("reprefetch_udma", "reprefetch_publish_barrier"))
                       for sample in captures])
    report = {"case": result["case"], "protocol": result["protocol"],
              "correctness": result["correctness"], "traffic": traffic,
              "same_pipeline_binary_all_ranks": True,
              "matched_empty_early_launch": all(path.exists() for path in early_paths),
              "wrapper_timings": {k: {a: b for a, b in row.items() if a != "samples_ms"}
                                  for k, row in timings.items()}, "residual": gap,
              "notes": ["Preloaded weights are a diagnostic ideal, not a valid pooled production mode.",
                        "Residual includes scheduling and contention, not only explicit ready waits.",
                        "Bootstrap interval describes this alternating run, not between-run variation."]}
    if len(serial) == world:
        joined = [max(values) for values in zip(*serial)]
        duration = statistics.median(joined)
        exposed = statistics.median(diffs)
        report["serial_udma_plus_publish_event"] = describe(joined)
        report["approx_hidden_fraction"] = 1 - max(0, exposed) / duration
        report["remaining_ideal_gain_percent"] = 100 * exposed / statistics.median(normal)
        report["remaining_ideal_speedup"] = statistics.median(normal) / statistics.median(resident)
        report["serial_plus_ideal_pipeline_ms"] = duration + statistics.median(resident)
        report["notes"].append("Event diagnostic and wrapper clocks differ; hidden fraction is a model estimate.")
    component_paths = [run / f"reprefetch_components_rank{rank}.json"
                       for rank in range(world)]
    if all(path.exists() for path in component_paths):
        component_ranks = [json.loads(path.read_text())["samples"]
                           for path in component_paths]
        assert len(set(map(len, component_ranks))) == 1
        report["serial_components_event"] = {
            key: describe([max(rank_samples[i][key] for rank_samples in component_ranks)
                           for i in range(len(component_ranks[0]))])
            for key in ("down_ms", "gate_up_ms", "publish_ms", "total_ms")}
        report["notes"].append(
            "Component runs serialize down+quiet, gate/up+quiet and publication. "
            "The total is measured on each rank before MAX; component MAX values must not be added.")
    launch_paths = [run / f"reprefetch_phases_rank{rank}.json" for rank in range(world)]
    if all(path.exists() for path in launch_paths):
        launch_ranks = [json.loads(path.read_text())["captures"] for path in launch_paths]
        launches = {}
        for name, capture in launch_ranks[0].items():
            by_label = {}
            for i in range(len(capture["samples"])):
                sample = {}
                for rank in launch_ranks:
                    # Same-named launches are summed per rank FIRST. Early
                    # submissions may be on a side stream; never add them
                    # to the mega event as if they were a serial critical path.
                    local = {}
                    for row in rank[name]["samples"][i]["launches"]:
                        local[row["name"]] = local.get(row["name"], 0) + row["ms"]
                    for key, value in local.items():
                        sample[key] = max(sample.get(key, 0), value)
                for key, value in sample.items():
                    by_label.setdefault(key, []).append(value)
            launches[name] = {key: describe(values) for key, values in by_label.items()}
        report["launch_event_rank_max"] = launches
    phase_paths = [run / "phases" / f"phases_rank{rank}.json" for rank in range(world)]
    if all(path.exists() for path in phase_paths):
        phase_ranks = [json.loads(path.read_text()) for path in phase_paths]
        windows = {"entry_to_b1": (0, 3), "b1_to_b2": (3, 5),
                   "b2_to_b4": (5, 8), "b4_to_b5": (8, 10),
                   "b5_to_p6_end": (10, 12), "entry_to_p6_end": (0, 12)}
        phases = {}
        for name, captures in phase_ranks[0]["captures"].items():
            phases[name] = {}
            for label, (start, end) in windows.items():
                values = [max(
                    (core[end] - core[start]) / rank["clock"]["ticks_per_us"] / 1000
                    for rank in phase_ranks
                    for core in rank["captures"][name][i]["stamps"])
                    for i in range(len(captures))]
                phases[name][label] = describe(values)
        report["phase_windows_rank_core_max"] = phases
        report["notes"].append(
            "Phase windows are separately gated instrumented binaries, per-core deltas then rank/core MAX. "
            "Do not sum different phase maxima or interpret issue stamps as completion.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    (args.run / "reprefetch_overlap_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
