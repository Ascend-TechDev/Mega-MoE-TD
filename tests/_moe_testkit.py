# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared helpers for the multi-rank Mega-MoE tests and benchmarks.

Consolidates the ACLSHMEM lifecycle, per-rank input generation, peer-memory
allocation, benchmark timing, and the bf16 grad-comparison metric that were
duplicated across the forward/backward layer and function tests.

These helpers are test/benchmark-only: they live under ``tests/`` and are not
shipped by the ``mega_moe`` package.
"""

import contextlib
import os
import statistics
import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist

from mega_moe.kernels.combine_fc1_bwd import GATE_PAD

try:  # Keep registry/collection checks usable on a CPU-only Python install.
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - exercised only outside Ascend.
    torch_npu = None

try:
    import shmem as ash
except ImportError:  # pragma: no cover - exercised only outside Ascend.
    ash = None


__all__ = [
    "TimingSpec",
    "TimingResult",
    "PerformanceRunner",
    "FORWARD_TIMING",
    "BACKWARD_TIMING",
    "CaseContext",
    "aclshmem_session",
    "collective_case_guard",
    "get_ash_size_bytes",
    "get_ash_ip_port",
    "init_aclshmem",
    "make_peer_mem",
    "make_moonep_backward_peer_mem",
    "make_pytest_params",
    "validate_timing_spec",
    "ash",
]

# ----------------------------------------------------------------------------
# ANSI colors (disabled under NO_COLOR for CI logs)
# ----------------------------------------------------------------------------

if os.environ.get("NO_COLOR"):
    GREEN = RED = RESET = BOLD = ""
else:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# ----------------------------------------------------------------------------
# ACLSHMEM bootstrap / lifecycle
# ----------------------------------------------------------------------------

def get_ash_size_bytes(default_gb=2):
    """Symmetric-heap size in bytes, driven by ``MOE_FUSED_ASH_SIZE_GB``."""
    gb = int(os.environ.get("MOE_FUSED_ASH_SIZE_GB", str(default_gb)))
    if gb <= 0:
        raise ValueError("MOE_FUSED_ASH_SIZE_GB must be a positive integer")
    return gb * 1024 * 1024 * 1024


def get_ash_ip_port():
    """ACLSHMEM bootstrap endpoint, overridable via ``ASH_MASTER_ADDR/PORT``."""
    addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("ASH_MASTER_PORT", "8666")
    return f"tcp://{addr}:{port}"


def init_aclshmem(
    rank,
    world_size,
    size_bytes,
    ip_port=None,
):
    """Initialize the ACLSHMEM symmetric heap for this rank.

    Caller is responsible for ``aclshmem_finalize()`` in a ``finally`` block.
    """
    if ash is None:
        raise RuntimeError("ACLSHMEM support is unavailable in this Python environment")
    ash.set_conf_store_tls(False, "")
    attr = ash.InitAttr()
    attr.my_rank = rank
    attr.n_ranks = world_size
    attr.local_mem_size = size_bytes
    attr.ip_port = ip_port if ip_port is not None else get_ash_ip_port()
    attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    if ash.aclshmem_init(attr) != 0:
        raise RuntimeError("aclshmem_init failed")


def make_peer_mem(saved, dtype, rank):
    """Allocate the shared symmetric buffer sized to the GLOBAL max of send/recv.

    ``dl.symm_at`` only resolves at heap offset 0 (see mega_moe.kernels), so one
    peer_mem is allocated per config and reused by backward step 1 and step 4.
    Using the GLOBAL max (all_reduce MAX) — not per-rank max — keeps peer_mem the
    SAME size on every rank, so any subsequent symmetric allocation (e.g. the
    backward signal_mem) lands at the same heap offset on every rank. Without this,
    signal_mem's offset differs per rank and signal_op RMA writes land at the wrong
    dst offset -> dl.wait never resolves -> deadlock.
    """
    if ash is None:
        raise RuntimeError("ACLSHMEM support is unavailable in this Python environment")
    local_elems = max(saved["total_recv"], saved["total_send"]) * (saved["hidden_dim"] + GATE_PAD)
    t = torch.tensor([local_elems], dtype=torch.int64, device=f"npu:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=saved["ep_group"])
    peer_elems = int(t.item())
    return ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=rank)


def make_moonep_backward_peer_mem(
    total_recv,
    total_send,
    hidden_dim,
    dtype,
    rank,
    ep_group,
):
    """Allocate the MoonEP backward peer_mem BEFORE any other symmetric buffer.

    The MoonEP backward case must also build a ``FusedMoEForward`` (whose context
    allocates its own symmetric heap objects) to produce the routing plan, so the
    backward peer_mem has to be reserved first to sit at heap offset 0 for
    ``dl.symm_at``. Sizing is passed explicitly because the physical
    ``saved_phys`` (and its total_recv) only exists after the plan is built; a
    dropless plan keeps ``total_recv == total_send == tokens * topk``. The
    all_reduce MAX keeps the same size — and therefore the same subsequent heap
    offsets (e.g. step1's signal_mem) — on every rank.
    """
    if ash is None:
        raise RuntimeError("ACLSHMEM support is unavailable in this Python environment")
    local_elems = max(int(total_recv), int(total_send)) * (hidden_dim + GATE_PAD)
    t = torch.tensor([local_elems], dtype=torch.int64, device=f"npu:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ep_group)
    return ash.aclshmem_create_tensor(
        [int(t.item())], dtype=dtype, device_id=rank
    )


@dataclass(frozen=True)
class TimingSpec:
    """Declarative timing protocol used by the performance runner.

    ``clock`` is either ``"npu_event"`` for forward or ``"host_wall"`` for
    backward.  The implementation keeps the protocol data-driven so the
    non-NPU unit tests can validate it without touching a device.
    """

    warmup: int = 5
    iterations: int = 50
    clock: str = "host_wall"
    reduction: str = "rank_max"

    def __post_init__(self):
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.clock not in {"host_wall", "npu_event"}:
            raise ValueError("clock must be 'host_wall' or 'npu_event'")
        if self.reduction != "rank_max":
            raise ValueError("the MoE protocol requires rank_max reduction")

    def as_dict(self) -> dict[str, object]:
        """Serialize the protocol without exposing implementation objects."""
        return {
            "warmup": self.warmup,
            "iterations": self.iterations,
            "clock": self.clock,
            "rank_reduction": "MAX",
        }


# Keep the two clocks explicit while sharing the same sample counts/reduction.
# These values are safe to import in CPU-only registry tests; timing functions
# are only invoked inside the distributed runners.
FORWARD_TIMING = TimingSpec(clock="npu_event")
BACKWARD_TIMING = TimingSpec(clock="host_wall")


@dataclass(frozen=True)
class TimingResult:
    """Raw samples plus a stable statistics envelope."""

    samples_ms: tuple[float, ...]

    def __post_init__(self):
        if not self.samples_ms:
            raise ValueError("a timing result needs at least one sample")

    @property
    def stats(self) -> dict[str, float]:
        values = tuple(float(value) for value in self.samples_ms)
        return {
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
        }


class PerformanceRunner:
    """Run two already-bound callables under one explicit timing protocol.

    Shape/case selection deliberately does not belong here.  The runner only
    owns the repeated-call protocol and the rank-MAX reduction, which lets the
    forward NPU-event benchmark and backward host-wall benchmark share one
    implementation without sharing operators or buffers.
    """

    def __init__(self, candidate, baseline, timing: TimingSpec, *, device, ep_group):
        self.candidate = candidate
        self.baseline = baseline
        self.timing = validate_timing_spec(timing)
        self.device = device
        self.ep_group = ep_group

    def run(self) -> tuple[TimingResult, TimingResult]:
        return self.measure(self.candidate), self.measure(self.baseline)

    def measure(self, fn) -> TimingResult:
        if self.timing.clock == "host_wall":
            return self._measure_host_wall(fn)
        return self._measure_npu_events(fn)

    def _synchronize_ranks(self):
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.ep_group)
        torch.npu.synchronize(self.device)

    def _rank_max(self, value_ms: float) -> float:
        value = torch.tensor([value_ms], dtype=torch.float32, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.ep_group)
        return float(value.item())

    def _measure_host_wall(self, fn) -> TimingResult:
        for _ in range(self.timing.warmup):
            fn()
        self._synchronize_ranks()
        start = time.perf_counter()
        for _ in range(self.timing.iterations):
            fn()
        torch.npu.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / self.timing.iterations
        return TimingResult((self._rank_max(elapsed_ms),))

    def _measure_npu_events(self, fn) -> TimingResult:
        samples = []
        for _ in range(self.timing.warmup):
            fn()
        torch.npu.synchronize(self.device)
        dist.barrier(group=self.ep_group)
        for _ in range(self.timing.iterations):
            self._synchronize_ranks()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            stream = torch.npu.current_stream(self.device)
            start.record(stream)
            fn()
            end.record(stream)
            end.synchronize()
            samples.append(self._rank_max(start.elapsed_time(end)))
        return TimingResult(tuple(samples))


@dataclass(frozen=True)
class CaseContext:
    """Small immutable description passed to worker-side builders."""

    rank: int
    world_size: int
    device: str
    ep_group: object


@contextlib.contextmanager
def aclshmem_session(
    rank,
    world_size,
    size_bytes,
    ip_port=None,
) -> Iterator[None]:
    """Initialize and finalize one isolated ACLSHMEM session."""
    init_aclshmem(
        rank,
        world_size,
        size_bytes,
        ip_port=ip_port,
    )
    try:
        yield
    finally:
        if ash is not None:
            ash.aclshmem_finalize()


@contextlib.contextmanager
def collective_case_guard(ep_group, label: str) -> Iterator[None]:
    """Provide a single place for future collective failure coordination.

    A worker exception is re-raised locally; callers still use barriers around
    case boundaries.  The TODO is intentionally retained until a direction-
    level session can coordinate failures across all parameterized cases.
    """
    try:
        yield
    except Exception:
        # Do not attempt an extra collective while a rank may already be
        # unwinding.  The pytest worker wrapper reports the original exception.
        raise


def validate_timing_spec(spec: TimingSpec) -> TimingSpec:
    """Return ``spec`` after validation; useful for pure Python tests."""
    if not isinstance(spec, TimingSpec):
        raise TypeError("spec must be a TimingSpec")
    return spec


def make_pytest_params(cases):
    """Turn immutable cases into visible, marker-bearing pytest parameters."""
    import pytest

    params = []
    for case in cases:
        marks = [getattr(pytest.mark, tag) for tag in sorted(case.tags)]
        params.append(pytest.param(case, id=case.case_id, marks=marks))
    return params
