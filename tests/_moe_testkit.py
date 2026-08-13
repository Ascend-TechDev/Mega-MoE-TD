# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared helpers for the multi-rank Mega-MoE tests and benchmarks.

Consolidates the ACLSHMEM lifecycle, per-rank input generation, peer-memory
allocation, benchmark timing, and the bf16 grad-comparison metric that were
duplicated across the forward/backward layer and function tests.

These helpers are test/benchmark-only: they live under ``tests/`` and are not
shipped by the ``mega_moe`` package.
"""

import contextlib
import ast
import hashlib
import importlib
import os
from pathlib import Path
import statistics
import stat
import sys
import time
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist

# Device modules are loaded explicitly by a device runner only after its
# checkout/environment provenance gate.  Merely importing this host contract
# must not initialize or even probe an NPU runtime.
torch_npu = None
ash = None
_ACTIVE_RUNTIME_SEAL = None


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
    "load_device_runtime",
    "LoadedModuleRecord",
    "AuthorizedRuntimeSeal",
    "capture_loaded_product_closure",
    "issue_authorized_runtime_seal",
    "verify_loaded_product_closure",
    "verify_benchmark_package_anchors",
    "activate_authorized_runtime",
    "require_authorized_runtime",
    "make_peer_mem",
    "make_pytest_params",
    "validate_timing_spec",
    "validate_finite_backward_gradients",
    "ash",
]

_BACKWARD_GRADIENT_KEYS = (
    "grad_hidden",
    "grad_routing_weights",
    "grad_fc1_1",
    "grad_fc1_2",
    "grad_fc2",
)

_TERMINAL_VERIFICATION_TRACE = (
    "stdlib_no_site",
    "authority_first",
    "snapshot_first",
    "runtime_first",
    "modules_loaded",
    "authority_final",
    "snapshot_final",
    "runtime_final",
    "modules_final",
)


@dataclass(frozen=True, order=True)
class LoadedModuleRecord:
    module_name: str
    relative_path: str
    loaded_sha256: str
    loader_kind: str
    manifest_blob_oid: str
    source_mode: int
    source_device: int
    source_inode: int


@dataclass(frozen=True)
class AuthorizedRuntimeSeal:
    authority_sha256: str
    snapshot_sha256: str
    runtime_sha256: str
    loaded_product_modules: tuple[LoadedModuleRecord, ...]
    explicit_plugins: tuple[LoadedModuleRecord, ...]
    verification_trace: tuple[str, ...]


def _sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _module_source_bytes(
    module_name: str, module, snapshot_root: Path
) -> tuple[Path, bytes, os.stat_result]:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not isinstance(origin, str):
        raise RuntimeError(f"loaded product closure has no origin: {module_name}")
    path = Path(origin)
    if path.suffix in {".pyc", ".pyo"}:
        raise RuntimeError(f"loaded product closure uses cached bytecode: {module_name}")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
        resolved.relative_to(snapshot_root)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"loaded product closure origin is outside snapshot: {module_name}"
        ) from error
    if (
        path.absolute() != resolved
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RuntimeError(f"loaded product closure origin is invalid: {module_name}")
    try:
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
    except OSError as error:
        raise RuntimeError(
            f"loaded product closure origin is unreadable: {module_name}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
        ):
            raise RuntimeError(
                f"loaded product closure origin changed: {module_name}"
            )
        chunks = []
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1024 * 1024, offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        content = b"".join(chunks)
        after = resolved.lstat()
        if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError(
                f"loaded product closure origin replaced: {module_name}"
            )
    finally:
        os.close(descriptor)
    return resolved, content, opened


def capture_loaded_product_closure(
    snapshot_root,
    namespace_prefixes,
    required_anchor_modules,
    snapshot_manifest=None,
) -> tuple[LoadedModuleRecord, ...]:
    """Capture every loaded source module in the verified product namespaces."""
    root = Path(snapshot_root).resolve(strict=True)
    prefixes = tuple(namespace_prefixes)
    if not prefixes or not all(isinstance(item, str) and item for item in prefixes):
        raise RuntimeError("loaded product closure namespace denominator is invalid")
    records = []
    for module_name, module in sorted(sys.modules.items()):
        if module is None or not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            continue
        path, content, metadata = _module_source_bytes(module_name, module, root)
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(content).hexdigest()
        blob_oid = hashlib.sha1(
            f"blob {len(content)}\0".encode("ascii") + content
        ).hexdigest()
        if snapshot_manifest is not None:
            expected = snapshot_manifest.get(relative_path)
            if (
                expected is None
                or getattr(expected, "kind", None) != "file"
                or getattr(expected, "sha256", None) != digest
                or getattr(expected, "blob_oid", None) != blob_oid
                or getattr(expected, "size", None) != len(content)
                or getattr(expected, "mode", None)
                != (stat.S_IFREG | stat.S_IMODE(metadata.st_mode))
            ):
                raise RuntimeError(
                    f"loaded product closure misses snapshot manifest: {module_name}"
                )
        records.append(
            LoadedModuleRecord(
                module_name=module_name,
                relative_path=relative_path,
                loaded_sha256=digest,
                loader_kind=type(getattr(module, "__loader__", None)).__name__,
                manifest_blob_oid=blob_oid,
                source_mode=stat.S_IFREG | stat.S_IMODE(metadata.st_mode),
                source_device=metadata.st_dev,
                source_inode=metadata.st_ino,
            )
        )
    names = {record.module_name for record in records}
    required = tuple(required_anchor_modules)
    if not required or len(set(required)) != len(required) or not set(required) <= names:
        raise RuntimeError("loaded product closure misses a required anchor")
    if len(names) != len(records):
        raise RuntimeError("loaded product closure contains duplicate module names")
    return tuple(records)


def issue_authorized_runtime_seal(
    *,
    authority_first_sha256,
    authority_final_sha256,
    snapshot_first_sha256,
    snapshot_final_sha256,
    runtime_first_sha256,
    runtime_final_sha256,
    loaded_product_modules,
    explicit_plugins,
    verification_trace,
) -> AuthorizedRuntimeSeal:
    """Issue a seal only after exact first/final identities and order agree."""
    pairs = (
        (authority_first_sha256, authority_final_sha256),
        (snapshot_first_sha256, snapshot_final_sha256),
        (runtime_first_sha256, runtime_final_sha256),
    )
    records = tuple(loaded_product_modules)
    plugins = tuple(explicit_plugins)
    if (
        not all(_sha256_hex(first) and first == final for first, final in pairs)
        or not records
        or not all(isinstance(record, LoadedModuleRecord) for record in records)
        or not all(isinstance(record, LoadedModuleRecord) for record in plugins)
        or len({record.module_name for record in records}) != len(records)
        or tuple(verification_trace) != _TERMINAL_VERIFICATION_TRACE
    ):
        raise RuntimeError("runtime seal terminal verification is invalid")
    seal = AuthorizedRuntimeSeal(
        authority_sha256=authority_first_sha256,
        snapshot_sha256=snapshot_first_sha256,
        runtime_sha256=runtime_first_sha256,
        loaded_product_modules=records,
        explicit_plugins=plugins,
        verification_trace=tuple(verification_trace),
    )
    verify_loaded_product_closure(seal)
    return seal


def verify_loaded_product_closure(seal: AuthorizedRuntimeSeal) -> None:
    if not isinstance(seal, AuthorizedRuntimeSeal):
        raise RuntimeError("authorized runtime seal is required")
    seen = set()
    for record in (*seal.loaded_product_modules, *seal.explicit_plugins):
        module = sys.modules.get(record.module_name)
        if module is None or record.module_name in seen:
            raise RuntimeError("loaded product closure module is missing or duplicated")
        seen.add(record.module_name)
        spec = getattr(module, "__spec__", None)
        origin_value = getattr(spec, "origin", None) or getattr(
            module, "__file__", None
        )
        if not isinstance(origin_value, str):
            raise RuntimeError("loaded product closure origin is missing")
        origin_path = Path(origin_value).resolve(strict=True)
        snapshot_root = origin_path.parents[len(Path(record.relative_path).parts) - 1]
        origin, content, metadata = _module_source_bytes(
            record.module_name, module, snapshot_root
        )
        blob_oid = hashlib.sha1(
            f"blob {len(content)}\0".encode("ascii") + content
        ).hexdigest()
        if (
            origin.as_posix().endswith("/" + record.relative_path) is False
            or hashlib.sha256(content).hexdigest() != record.loaded_sha256
            or blob_oid != record.manifest_blob_oid
            or (stat.S_IFREG | stat.S_IMODE(metadata.st_mode)) != record.source_mode
            or metadata.st_dev != record.source_device
            or metadata.st_ino != record.source_inode
        ):
            raise RuntimeError("loaded product closure bytes changed")


def verify_benchmark_package_anchors(
    snapshot_root, records: tuple[LoadedModuleRecord, ...]
) -> None:
    """Bind both benchmark package parents to regular inert source leaves."""
    root = Path(snapshot_root).resolve(strict=True)
    by_name = {record.module_name: record for record in records}
    expected = {
        "benchmark": "benchmark/__init__.py",
        "benchmark.layer": "benchmark/layer/__init__.py",
    }
    for module_name, relative_path in expected.items():
        record = by_name.get(module_name)
        path = root / relative_path
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RuntimeError("benchmark package anchor is missing") from error
        if (
            record is None
            or record.relative_path != relative_path
            or not path.is_file()
            or path.is_symlink()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o644
        ):
            raise RuntimeError("benchmark package anchor is not a regular 0644 leaf")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record.loaded_sha256:
            raise RuntimeError("benchmark package anchor bytes changed")
        try:
            tree = ast.parse(content.decode("utf-8"))
        except (UnicodeDecodeError, SyntaxError) as error:
            raise RuntimeError("benchmark package anchor source is invalid") from error
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            raise RuntimeError("benchmark package anchor has an import side effect")


def activate_authorized_runtime(seal: AuthorizedRuntimeSeal) -> AuthorizedRuntimeSeal:
    global _ACTIVE_RUNTIME_SEAL
    verify_loaded_product_closure(seal)
    _ACTIVE_RUNTIME_SEAL = seal
    return seal


def require_authorized_runtime(
    seal: AuthorizedRuntimeSeal | None = None,
) -> AuthorizedRuntimeSeal:
    verified = seal if seal is not None else _ACTIVE_RUNTIME_SEAL
    if not isinstance(verified, AuthorizedRuntimeSeal):
        raise RuntimeError("authorized runtime seal is required before device entry")
    verify_loaded_product_closure(verified)
    return verified


def load_device_runtime(seal: AuthorizedRuntimeSeal | None = None) -> None:
    """Load NPU-only modules after the caller's provenance gate has passed."""
    require_authorized_runtime(seal)
    global torch_npu, ash
    if torch_npu is None:
        torch_npu = importlib.import_module("torch_npu")
    if ash is None:
        ash = importlib.import_module("shmem")


def validate_finite_backward_gradients(candidates, oracle) -> None:
    """Reject NaN/Inf in either candidate arm or the independent oracle."""
    if not isinstance(candidates, dict) or not candidates:
        raise AssertionError("backward candidates must be a non-empty mapping")
    sources = (("oracle", oracle), *tuple(candidates.items()))
    for source_name, gradients in sources:
        if not isinstance(gradients, dict):
            raise AssertionError(f"{source_name} gradients must be a mapping")
        for gradient_name in _BACKWARD_GRADIENT_KEYS:
            tensor = gradients.get(gradient_name)
            if not isinstance(tensor, torch.Tensor):
                raise AssertionError(
                    f"{source_name} gradient {gradient_name} must be a tensor"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise AssertionError(
                    f"non-finite {source_name} gradient {gradient_name}"
                )

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


def init_aclshmem(rank, world_size, size_bytes, ip_port=None):
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
    """Allocate the shared symmetric buffer sized to the larger of send/recv.

    ``dl.symm_at`` only resolves at heap offset 0 (see mega_moe.kernels), so one
    peer_mem is allocated per config and reused by backward step 1 and step 4.
    """
    if ash is None:
        raise RuntimeError("ACLSHMEM support is unavailable in this Python environment")
    peer_elems = max(saved["total_recv"], saved["total_send"]) * saved["hidden_dim"]
    return ash.aclshmem_create_tensor([peer_elems], dtype=dtype, device_id=rank)


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
    def median_ms(self) -> float:
        """Unrounded median used for threshold decisions."""
        return float(statistics.median(self.samples_ms))

    @property
    def stats(self) -> dict[str, float]:
        values = tuple(float(value) for value in self.samples_ms)
        return {
            "min_ms": round(min(values), 3),
            "max_ms": round(max(values), 3),
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(self.median_ms, 3),
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

    def run_paired(self) -> dict[str, dict[str, TimingResult]]:
        """Measure both arms in both orders and retain every rank-MAX sample.

        A fixed candidate-then-baseline order can turn thermal or allocator
        drift into an apparent speedup.  The wgrad decision therefore uses two
        explicit orders in the same process/session.  Each order receives its
        own warmup, and every measured invocation is synchronized and reduced
        across ranks before the next arm starts.
        """
        orders = (
            ("candidate_then_baseline", (
                ("candidate", self.candidate),
                ("baseline", self.baseline),
            )),
            ("baseline_then_candidate", (
                ("baseline", self.baseline),
                ("candidate", self.candidate),
            )),
        )
        results = {}
        for order_name, arms in orders:
            for _ in range(self.timing.warmup):
                for _, fn in arms:
                    fn()
            samples = {label: [] for label, _ in arms}
            for _ in range(self.timing.iterations):
                for label, fn in arms:
                    samples[label].append(self._measure_single(fn))
            results[order_name] = {
                label: TimingResult(tuple(values))
                for label, values in samples.items()
            }
        return results

    def measure(self, fn) -> TimingResult:
        if self.timing.clock == "host_wall":
            return self._measure_host_wall(fn)
        return self._measure_npu_events(fn)

    def _measure_single(self, fn) -> float:
        """Measure one invocation with the configured clock and rank MAX."""
        self._synchronize_ranks()
        if self.timing.clock == "host_wall":
            start = time.perf_counter()
            fn()
            torch.npu.synchronize(self.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return self._rank_max(elapsed_ms)

        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        stream = torch.npu.current_stream(self.device)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        return self._rank_max(start.elapsed_time(end))

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
def aclshmem_session(rank, world_size, size_bytes, ip_port=None) -> Iterator[None]:
    """Initialize and finalize one isolated ACLSHMEM session."""
    init_aclshmem(rank, world_size, size_bytes, ip_port=ip_port)
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
