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
from mega_moe.runtime.device import device_str, multi_node_enabled, resolve_local_device

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
    """ACLSHMEM bootstrap endpoint, overridable via ``ASH_MASTER_ADDR/PORT``.

    Multi-node NOTE (2026-09-22 dual-node adaptation): the ip_port bootstrap is
    the ONLY python-reachable path that can carry an engine mask —
    ``aclshmem_init_using_unique_id(rank, npes, size, uid)`` has no attributes
    parameter and always initializes with the MTE default, which cannot cross
    nodes.  Dual-node therefore REQUIRES an explicit ``ASH_MASTER_ADDR``
    pointing at node 0 (this mirrors upstream PR#184's cross-node fix); a
    loopback default under ``MEGAMOE_MULTI_NODE=1`` is rejected loudly instead
    of deadlocking later inside the first cross-node kernel.
    """
    addr = os.environ.get("ASH_MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("ASH_MASTER_PORT", "8666")
    if multi_node_enabled() and addr == "127.0.0.1":
        raise RuntimeError(
            "MEGAMOE_MULTI_NODE=1 requires ASH_MASTER_ADDR=<node0-IP> (and a "
            "node0-fixed ASH_MASTER_PORT on both nodes): the ip_port bootstrap "
            "is the only one that can set the cross-node MTE|UDMA engine mask"
        )
    return f"tcp://{addr}:{port}"


def init_aclshmem(
    rank,
    world_size,
    size_bytes,
    ip_port=None,
    enable_udma=False,
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
    # Data-op engine for the symmetric heap.  MTE is the proven default for
    # the signal/wait + symm_at kernels.  Engine tiers:
    #  - MOE_ASH_ENGINE=udma (mega-kernel experiments): pure UDMA — the
    #    08-ascend-transpose-all2all notes warn getmem/putmem_signal corrupt
    #    on this box, so signal_op paths need the correctness gates re-run
    #    under UDMA before trusting any number.
    #  - MOE_ASH_ENGINE=combo, enable_udma, or MEGAMOE_MULTI_NODE=1:
    #    MTE|UDMA combo — MTE physically cannot cross nodes
    #    (shmem_device_mte.h: "does not support cross-PCIe"), so multi-node
    #    defaults to the combo mask (intra-node MTE, inter-node UDMA, both
    #    udma transports MOE_MEGA_{GRAD,REPREFETCH}_TRANSPORT=udma included).
    #  - MOE_ASH_ENGINE=mte or unset: pure MTE (bit-identical single-node
    #    default; mte also lets G1 isolate the device-split variable).
    if os.environ.get("MOE_ASH_ENGINE") == "udma":
        attr.option_attr.data_op_engine_type = ash.OpEngineType.UDMA
    elif os.environ.get("MOE_ASH_ENGINE") == "mte":
        attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    elif os.environ.get("MOE_ASH_ENGINE") == "roce":
        # Cross-node leg over the RoCE NICs (G2 r22-R3): UDMA/combo route
        # cross-server traffic through the CLOS plane and the local-view
        # /etc rootinfo mis-resolves the peer (r20 topo_reader log) — the
        # fabric-standard cross-server path is the RoCE engine.
        attr.option_attr.data_op_engine_type = ash.OpEngineType.ROCE
    elif os.environ.get("MOE_ASH_ENGINE") == "combo_roce":
        attr.option_attr.data_op_engine_type = ash.OpEngineType(
            ash.OpEngineType.MTE.value | ash.OpEngineType.UDMA.value
            | ash.OpEngineType.ROCE.value)
    elif enable_udma or os.environ.get("MOE_ASH_ENGINE") == "combo" or multi_node_enabled():
        attr.option_attr.data_op_engine_type = ash.OpEngineType(
            ash.OpEngineType.MTE.value | ash.OpEngineType.UDMA.value)
    else:
        attr.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    # G2 r23 two-phase /etc: HCCL parsed the file at comm init (the conftest
    # barrier) long before this point; the shmem topo reader parses it HERE.
    # Swapping a world-scoped rootinfo in between gives the data plane a
    # cross-server view without breaking HCCL's local-only validation
    # (hand-merged remote entries make hcclCommInitRootInfoConfig fail with
    # code 4 — proven locally, r22-R2).  The runner's trap restores the
    # per-machine resting file at exit.
    swap = os.environ.get("MOE_ROOTINFO_SWAP")
    if swap == "REMOVE":
        # G2 r25: libshmem falls back to GENERATING a rootinfo view from the
        # driver topo json (atlas_950_1.json, full 64-peer supernode) when
        # /etc/hccl_rootinfo.json is absent.  Hand-merged files are exhausted
        # (r22-R2/r23/r24: rank_list[world_rank] target lookup vs first-entry
        # self-check cannot both hold on node1), so let the library build its
        # own view.  The runner trap still restores the resting file at exit.
        try:
            os.remove("/etc/hccl_rootinfo.json")
        except FileNotFoundError:
            pass
    elif swap:
        import shutil
        shutil.copy(swap, "/etc/hccl_rootinfo.json")
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
    t = torch.tensor([local_elems], dtype=torch.int64, device=device_str(resolve_local_device(rank)))
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=saved["ep_group"])
    peer_elems = int(t.item())
    return ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=resolve_local_device(rank))


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
    ``saved_phys`` (and its total_recv) only exists after the plan is built.
    The recv argument must budget for WORST-CASE routing imbalance: per-rank
    ``total_recv`` is data-dependent (it counts every global token-slot whose
    expert lands on this rank), NOT ``tokens * topk`` — under random imbalance
    a dropless plan can drive it up to ``tokens * topk * world_size``, and an
    undersized buffer first overflows silently in the FORWARD (raw-pointer
    writes, no bounds check) and then raises on a single rank in the backward
    ``.view()`` — which leaves the peer rank spinning in its kernel (the
    "one rank vanishes, no traceback" hang form). The all_reduce MAX keeps
    the same size — and therefore the same subsequent heap offsets (e.g.
    step1's signal_mem) — on every rank.
    """
    if ash is None:
        raise RuntimeError("ACLSHMEM support is unavailable in this Python environment")
    local_elems = max(int(total_recv), int(total_send)) * (hidden_dim + GATE_PAD)
    t = torch.tensor([local_elems], dtype=torch.int64, device=device_str(resolve_local_device(rank)))
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=ep_group)
    return ash.aclshmem_create_tensor(
        [int(t.item())], dtype=dtype, device_id=resolve_local_device(rank)
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
# Env-overridable sample counts (default protocol unchanged): hang-bisect
# runs set MOE_BENCH_BWD_WARMUP/ITERS small to cut the per-sample cost of the
# transport loop (2026-09-21, E896 w8 aicore-timeout bisect).
BACKWARD_TIMING = TimingSpec(
    warmup=int(os.environ.get("MOE_BENCH_BWD_WARMUP", "5")),
    iterations=int(os.environ.get("MOE_BENCH_BWD_ITERS", "50")),
    clock="host_wall",
)


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
    enable_udma=False,
) -> Iterator[None]:
    """Initialize and finalize one isolated ACLSHMEM session."""
    init_aclshmem(
        rank,
        world_size,
        size_bytes,
        ip_port=ip_port,
        enable_udma=enable_udma,
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
