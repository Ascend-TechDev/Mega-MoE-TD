# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration for the standalone Ascend Mega-MoE forward."""

from dataclasses import dataclass, field
from typing import Optional


# The mixed FC1 kernel has been validated through a 256-row GEMM window on
# the current Ascend backend. Larger FP32 accumulators can fail in codegen.
_MAX_FC1_GEMM_BLOCK_SIZE_M = 256
_MAX_FC1_GEMM_ACCUMULATOR_ELEMENTS = 256 * 256

# FC2 uses the same FP32 Cube accumulator limit; this bound applies only to the
# local expert GEMM and is independent from device-put transport.
_MAX_FC2_GEMM_BLOCK_SIZE_M = 256
_MAX_FC2_GEMM_ACCUMULATOR_ELEMENTS = 256 * 256

# Supported post-FC1 gated activations.  swiglu is silu(gate) * up;
# situglu is beta * tanh(gate / beta) * sigmoid(gate) * up (with an
# optional linear_beta * tanh(up / linear_beta) transform on up).
_ACTIVATIONS = (
    "swiglu",
    "situglu",
)

_MAX_REPLICA_PREFETCH_CHUNK_BYTES = (1 << 32) - 1


def _detect_physical_aicore_count() -> Optional[int]:
    """Return the active device's physical AICore count when available.

    The Triton Ascend driver owns the platform table, so using it here keeps
    the forward launch grid aligned with the installed 950PR/950DT variant.
    Import and driver failures are deliberately converted to ``None``: this
    module is also imported by host-only unit tests and documentation tools.
    """
    try:
        from triton.backends.ascend.driver import NPUUtils

        count = NPUUtils().get_aicore_num()
    except Exception:
        return None
    return count if type(count) is int and count > 0 else None


def _default_num_aicore_programs() -> int:
    """Resolve the launch grid from CANN's active-device report."""
    count = _detect_physical_aicore_count()
    if count is None:
        raise RuntimeError(
            "cannot determine the NPU AICore count; initialize CANN before "
            "creating MoEForwardConfig"
        )
    return count


def _detect_physical_aivector_core_count() -> Optional[int]:
    """Return the active device's physical Vector-core count when available."""
    try:
        from triton.backends.ascend.driver import NPUUtils

        count = NPUUtils().get_aivector_core_num()
    except Exception:
        return None
    return count if type(count) is int and count > 0 else None


def _detect_fc_gemm_accumulator_budget() -> Optional[int]:
    """Return the active device's FP32 Cube-accumulator element budget.

    The 910B family carries a 128KB L0C, so one M*N FP32 accumulator tile
    must stay within 32768 elements there; a 256x256 tile needs 256KB and
    the backend rejects it with a Cc-overflow compile error.  ``None``
    keeps the 256x256-validated defaults on every other arch.
    """
    try:
        from triton.backends.ascend.driver import NPUUtils

        arch = NPUUtils().get_arch()
    except Exception:
        return None
    if isinstance(arch, str) and arch.startswith("Ascend910B"):
        return 32768
    return None


@dataclass(frozen=True)
class MoEForwardConfig:
    """Stage-specific launch and tiling parameters.

    FC1 uses the fixed all-core expert/N-tile pipeline.  FC1 and FC2 tiles
    remain independent because they have different shapes and data-movement
    paths.

    ``dispatch_fc1_block_size_m`` controls dispatch readiness slots.
    ``fc1_gemm_block_size_{m,n,k}`` independently control the FC1 dot axes.
    Likewise, ``fc2_combine_block_size_m`` controls the FC2 GEMM row tile and
    ``fc2_gemm_block_size_{n,k}`` control the remaining FC2 dot axes.  FC2
    stages each expert group in local GM, then uses striped ACLSHMEM device-put
    workers for reverse transport.  All detected Vector cores participate in
    reduction.  The post-FC1 activation remains selectable for compatibility
    with the target repository SiTU-GLU path.

    activation selects SwiGLU or SiTU-GLU; situ_beta and situ_linear_beta
    configure the latter and are ignored for SwiGLU.

    The launch grid queries CANN's ``NPUUtils().get_aicore_num()`` and uses
    every physical AICore; it is device state rather than a user setting.
    """

    num_aicore_programs: int = field(init=False)
    num_aivector_programs: int = field(init=False)
    receive_capacity_factor: Optional[float] = None
    dispatch_fc1_block_size_m: int = 128
    fc1_gemm_block_size_n: int = 256
    fc1_gemm_block_size_k: int = 128
    fc2_combine_block_size_m: int = 256
    fc2_gemm_block_size_n: int = 256
    fc2_gemm_block_size_k: int = 128

    # Post-FC1 gated activation.  swiglu (default) preserves the original
    # silu(gate) * up path; situglu selects SiTU-GLU.
    activation: str = "swiglu"
    situ_beta: float = 1.0
    situ_linear_beta: Optional[float] = None
    # Appended to preserve positional construction of the older config fields.
    fc1_gemm_block_size_m: int = 256
    # MoonEP load balancing is opt-in while the weight-prefetch path is being
    # validated.  Its replica budget is fixed to experts_per_rank, so enabling
    # it doubles the destination-local physical expert slots.
    enable_moonep: bool = False
    # Gate/up is pushed by the otherwise-idle second Vector subcores inside
    # the mixed dispatch/FC1 kernel.  Keep its chunk independently tunable:
    # larger chunks favor bandwidth, while smaller chunks yield the shared MTE
    # path to activation dispatch more frequently.
    moonep_replica_gate_up_chunk_bytes: int = 16 * 1024 * 1024
    # Keep down-weight MTE requests short enough that activation dispatch can
    # make forward progress on the shared transport.
    moonep_replica_down_chunk_bytes: int = 4 * 1024 * 1024
    # During activation dispatch, only this many second Vector subcores may
    # prefetch down replicas.  Each handles at most the configured descriptor
    # count; first Vector subcores take the remainder after their own dispatch
    # work finishes.  This static split avoids cross-subcore atomics in the
    # latency-critical mixed kernel.
    moonep_replica_down_early_programs: int = 8
    moonep_replica_down_early_descriptors_per_program: int = 2
    # Independent experiment switches keep the correctness-reference path
    # available for A/B runs.  The route-map fusion writes the count cube and
    # all route metadata directly into the reusable context workspaces.
    moonep_fused_balanced_count: bool = True
    moonep_fused_route_mapping: bool = True
    # Repeated stable routes can reuse replica tables.  One-shot/cache-miss
    # benchmarks may disable the collective hit check without deleting the
    # useful production cache path.
    moonep_enable_replica_cache: bool = True

    def __post_init__(self):
        object.__setattr__(
            self, "num_aicore_programs", _default_num_aicore_programs()
        )
        vector_count = _detect_physical_aivector_core_count()
        if vector_count is None:
            raise RuntimeError(
                "cannot determine the NPU Vector-core count; initialize CANN "
                "before creating MoEForwardConfig"
            )
        object.__setattr__(self, "num_aivector_programs", vector_count)
        if self.num_aicore_programs < 2:
            raise ValueError("num_aicore_programs must be at least 2")

        for name, value in (
            ("dispatch_fc1_block_size_m", self.dispatch_fc1_block_size_m),
            ("fc1_gemm_block_size_m", self.fc1_gemm_block_size_m),
            ("fc1_gemm_block_size_n", self.fc1_gemm_block_size_n),
            ("fc1_gemm_block_size_k", self.fc1_gemm_block_size_k),
            ("fc2_combine_block_size_m", self.fc2_combine_block_size_m),
            ("fc2_gemm_block_size_n", self.fc2_gemm_block_size_n),
            ("fc2_gemm_block_size_k", self.fc2_gemm_block_size_k),
        ):
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two no smaller than 16")

        if self.fc1_gemm_block_size_m > _MAX_FC1_GEMM_BLOCK_SIZE_M:
            raise ValueError(
                "fc1_gemm_block_size_m must be no larger than "
                f"{_MAX_FC1_GEMM_BLOCK_SIZE_M} on the current Ascend backend"
            )
        if (
            self.fc1_gemm_block_size_m * self.fc1_gemm_block_size_n
            > _MAX_FC1_GEMM_ACCUMULATOR_ELEMENTS
        ):
            raise ValueError(
                "fc1 GEMM M*N tile must be no larger than "
                f"{_MAX_FC1_GEMM_ACCUMULATOR_ELEMENTS} elements on the current "
                "Ascend backend"
            )

        if self.fc2_combine_block_size_m > _MAX_FC2_GEMM_BLOCK_SIZE_M:
            raise ValueError(
                "fc2_combine_block_size_m must be no larger than "
                f"{_MAX_FC2_GEMM_BLOCK_SIZE_M} on the current Ascend backend"
            )
        if (
            self.fc2_combine_block_size_m * self.fc2_gemm_block_size_n
            > _MAX_FC2_GEMM_ACCUMULATOR_ELEMENTS
        ):
            raise ValueError(
                "FC2 GEMM M*N tile must be no larger than "
                f"{_MAX_FC2_GEMM_ACCUMULATOR_ELEMENTS} elements on the current "
                "Ascend backend"
            )

        budget = _detect_fc_gemm_accumulator_budget()
        if budget is not None:
            # 910B parts expose half the L0C the 256x256 defaults were sized
            # for; shrink the N tiles (keeping the validated 256-row M
            # windows) until each FP32 accumulator fits the smaller Cube
            # cache instead of failing in codegen.
            for m_name, n_name in (
                ("fc1_gemm_block_size_m", "fc1_gemm_block_size_n"),
                ("fc2_combine_block_size_m", "fc2_gemm_block_size_n"),
            ):
                n_tile = getattr(self, n_name)
                m_tile = getattr(self, m_name)
                while m_tile * n_tile > budget and n_tile > 16:
                    n_tile //= 2
                if n_tile != getattr(self, n_name):
                    object.__setattr__(self, n_name, n_tile)

        if self.receive_capacity_factor is not None and self.receive_capacity_factor < 1.0:
            raise ValueError("receive_capacity_factor must be at least 1")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(
                "activation must be one of "
                + ", ".join(repr(value) for value in _ACTIVATIONS)
            )
        if not (type(self.situ_beta) is float or type(self.situ_beta) is int):
            raise TypeError("situ_beta must be a float")
        if float(self.situ_beta) <= 0.0:
            raise ValueError("situ_beta must be positive")
        if self.situ_linear_beta is not None:
            if not (
                type(self.situ_linear_beta) is float
                or type(self.situ_linear_beta) is int
            ):
                raise TypeError("situ_linear_beta must be a float or None")
            if float(self.situ_linear_beta) <= 0.0:
                raise ValueError("situ_linear_beta must be positive when set")
        if type(self.enable_moonep) is not bool:
            raise TypeError("enable_moonep must be a bool")
        if type(self.moonep_fused_balanced_count) is not bool:
            raise TypeError("moonep_fused_balanced_count must be a bool")
        if type(self.moonep_fused_route_mapping) is not bool:
            raise TypeError("moonep_fused_route_mapping must be a bool")
        if type(self.moonep_enable_replica_cache) is not bool:
            raise TypeError("moonep_enable_replica_cache must be a bool")
        if (
            self.moonep_fused_route_mapping
            and not self.moonep_fused_balanced_count
        ):
            raise ValueError(
                "moonep_fused_route_mapping requires the fused balanced count "
                "cube"
            )
        if (
            type(self.moonep_replica_gate_up_chunk_bytes) is not int
            or self.moonep_replica_gate_up_chunk_bytes <= 0
            or self.moonep_replica_gate_up_chunk_bytes
            > _MAX_REPLICA_PREFETCH_CHUNK_BYTES
            or self.moonep_replica_gate_up_chunk_bytes % 2
        ):
            raise ValueError(
                "moonep_replica_gate_up_chunk_bytes must be a positive even "
                "integer no larger than the ACLSHMEM uint32 byte-count ABI"
            )
        if (
            type(self.moonep_replica_down_chunk_bytes) is not int
            or self.moonep_replica_down_chunk_bytes <= 0
            or self.moonep_replica_down_chunk_bytes
            > _MAX_REPLICA_PREFETCH_CHUNK_BYTES
            or self.moonep_replica_down_chunk_bytes % 2
        ):
            raise ValueError(
                "moonep_replica_down_chunk_bytes must be a positive even "
                "integer no larger than the ACLSHMEM uint32 byte-count ABI"
            )
        if (
            type(self.moonep_replica_down_early_programs) is not int
            or not 0 <= self.moonep_replica_down_early_programs
            <= self.num_aicore_programs
        ):
            raise ValueError(
                "moonep_replica_down_early_programs must be an integer between "
                "zero and num_aicore_programs"
            )
        if (
            type(self.moonep_replica_down_early_descriptors_per_program)
            is not int
            or self.moonep_replica_down_early_descriptors_per_program < 0
        ):
            raise ValueError(
                "moonep_replica_down_early_descriptors_per_program must be a "
                "non-negative integer"
            )

    def resolved_receive_capacity_factor(self, world_size: int) -> float:
        """Return the configured capacity, or the worst-case-safe default."""
        if self.receive_capacity_factor is None:
            return float(world_size)
        return float(self.receive_capacity_factor)


__all__ = ["MoEForwardConfig"]
