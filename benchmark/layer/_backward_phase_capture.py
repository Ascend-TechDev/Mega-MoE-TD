# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Capture existing backward SYS_CNT probes, with an independent clock probe."""

import os
import statistics
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from mega_moe.kernels.common import _sys_cnt_tick


@triton.jit
def clock_probe(out, ticks):
    lane = tl.arange(0, 1)
    start = _sys_cnt_tick(lane)
    now = start
    while tl.sum(now - start, 0) < ticks:
        now = _sys_cnt_tick(now)
    tl.store(out + lane, start)
    tl.store(out + 1 + lane, now)


def calibrate_clock(device):
    out = torch.empty(2, dtype=torch.int64, device=device)
    clock_probe[(1,)](out, 20_000_000)
    torch.npu.synchronize()
    pairs = []
    for _ in range(5):
        start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
        start.record()
        clock_probe[(1,)](out, 20_000_000)
        end.record()
        end.synchronize()
        ticks = out.cpu().tolist()
        pairs.append({"ticks": ticks[1] - ticks[0], "event_us": start.elapsed_time(end) * 1000})
    return {"ticks_per_us": statistics.median(p["ticks"] / p["event_us"] for p in pairs),
            "pairs": pairs}


class TimedLaunch:
    def __init__(self, kernel):
        self.kernel = kernel
        self.events = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def measured(*args, **kwargs):
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            result = launch(*args, **kwargs)
            end.record()
            self.events.append((start, end))
            return result

        return measured


def capture_reprefetch(functions, device, output_dir, rank, samples):
    """Event brackets around actual launches; no timing code inside kernels.

    Sum only calls on this rank/current stream. Rank maxima of individual
    stages do not form an end-to-end critical path and must not be added.
    """
    from benchmark.layer.bench_backward_bigop import prepare_call, write_json
    from mega_moe.kernels import mega_bwd

    records = []

    def bracket(name, fn):
        def measured(*args, **kwargs):
            start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            result = fn(*args, **kwargs)
            end.record()
            records.append((name, start, end))
            return result
        return measured

    class Launch:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            return bracket(self.name, self.kernel[grid])

    names = {
        "_kernel_compact_local_replica_descriptors": "replica_descriptors",
        "_kernel_replica_repush_store": "reprefetch_store",
        "_kernel_replica_repush_udma": "reprefetch_udma",
        "kernel_moe_backward_mega_recompute": "mega_recompute",
    }
    originals = {name: getattr(mega_bwd, name) for name in names}
    originals["launch_replica_grad_barrier"] = mega_bwd.launch_replica_grad_barrier
    captures = {}
    try:
        for name, label in names.items():
            setattr(mega_bwd, name, Launch(label, originals[name]))
        def replica_barrier(*args, **kwargs):
            after_mega = any(name == "mega_recompute" for name, _, _ in records)
            label = "gradient_cleanup_barrier" if after_mega else "reprefetch_publish_barrier"
            return bracket(label, originals["launch_replica_grad_barrier"])(*args, **kwargs)

        mega_bwd.launch_replica_grad_barrier = replica_barrier
        for name, fn in functions.items():
            if not hasattr(fn, "transport"):
                continue
            rows = []
            for _ in range(samples):
                prepare_call(fn)
                torch.npu.synchronize()
                dist.barrier()
                records.clear()
                begin = time.perf_counter_ns()
                result = fn()
                torch.npu.synchronize()
                wall_ms = (time.perf_counter_ns() - begin) / 1e6
                rows.append({
                    "host_wall_ms": wall_ms,
                    "launches": [{"name": label, "ms": start.elapsed_time(end)}
                                 for label, start, end in records],
                })
                del result
            captures[name] = {"transport": fn.transport, "samples": rows}
        write_json(output_dir / f"reprefetch_phases_rank{rank}.json", {
            "boundary": "actual backward launches, NPU events on current stream",
            "notes": "diagnostic only; event insertion adds overhead; forward and table overwrite excluded; no P6 separation inside mega",
            "captures": captures,
        })
    finally:
        for name, fn in originals.items():
            setattr(mega_bwd, name, fn)


def capture_phases(functions, saved, dy, device, output_dir, rank, samples):
    from benchmark.layer.bench_backward_bigop import correctness_gate, prepare_call, write_json
    from mega_moe.kernels import mega_bwd

    output_dir = output_dir / "phases"
    output_dir.mkdir(exist_ok=True)
    previous = os.environ.get("MOE_MEGA_TIMING")
    os.environ["MOE_MEGA_TIMING"] = "1"
    names = ("kernel_moe_backward_mega", "kernel_moe_backward_mega_recompute")
    originals = {name: getattr(mega_bwd, name) for name in names}
    timed = {name: TimedLaunch(kernel) for name, kernel in originals.items()}
    moonep = any(hasattr(fn, "owner") for fn in functions.values())
    try:
        # Instrumentation changes code generation; gate that binary separately.
        correctness_gate(functions, saved, dy, device, output_dir, rank,
                         reference_cpu=moonep)
        calibration = calibrate_clock(device)
        for name, kernel in timed.items():
            setattr(mega_bwd, name, kernel)
        result = {}
        for name, fn in functions.items():
            if name == "bigop":
                continue
            captures = []
            for _ in range(samples):
                prepare_call(fn)
                torch.npu.synchronize()
                dist.barrier()
                for kernel in timed.values():
                    kernel.events.clear()
                fn()
                torch.npu.synchronize()
                active_saved = fn.owner.native_saved if hasattr(fn, "owner") else saved
                ts, waits = active_saved["_mega_timing_last"]
                events = [event for kernel in timed.values() for event in kernel.events]
                if len(events) != 1:
                    raise RuntimeError(f"expected one mega launch, got {len(events)}")
                start, end = events[0]
                captures.append({"stamps": ts.cpu().tolist(), "waits": waits.cpu().tolist(),
                                 "kernel_event_ms": start.elapsed_time(end)})
            result[name] = captures
        write_json(output_dir / f"phases_rank{rank}.json", {
            "clock": calibration,
            "stamp_names": ["entry", "p1_vector_issue", "p1_cube_issue", "after_b1",
                            "p23_issue", "after_b2", "p4ab_issue", "p5a_issue",
                            "after_b4", "p5b_issue_pre_transport", "after_b5",
                            "after_b6_down_reduce_gate_seed", "after_p6_gate_reduce"],
            "boundary_notes": "Only barrier checkpoints denote engine drain; other stamps denote issue completion. B5/B6/P6 brackets include waits for other ranks.",
            "captures": result,
        })
    finally:
        for name, kernel in originals.items():
            setattr(mega_bwd, name, kernel)
        if previous is None:
            os.environ.pop("MOE_MEGA_TIMING", None)
        else:
            os.environ["MOE_MEGA_TIMING"] = previous
