# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration for the standalone Ascend Mega-MoE forward."""

from dataclasses import dataclass, field
from typing import Optional


_DISPATCH_FC1_SCHEDULES = (
    "allcore_expert_n_tile",
)

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
    dispatch_fc1_schedule: str = "allcore_expert_n_tile"

    # Post-FC1 gated activation.  swiglu (default) preserves the original
    # silu(gate) * up path; situglu selects SiTU-GLU.
    activation: str = "swiglu"
    situ_beta: float = 1.0
    situ_linear_beta: Optional[float] = None
    # Appended to preserve positional construction of the older config fields.
    fc1_gemm_block_size_m: int = 256

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
        if self.dispatch_fc1_schedule not in _DISPATCH_FC1_SCHEDULES:
            raise ValueError(
                "dispatch_fc1_schedule must be one of "
                + ", ".join(repr(value) for value in _DISPATCH_FC1_SCHEDULES)
            )

    def resolved_receive_capacity_factor(self, world_size: int) -> float:
        """Return the configured capacity, or the worst-case-safe default."""
        if self.receive_capacity_factor is None:
            return float(world_size)
        return float(self.receive_capacity_factor)


__all__ = ["MoEForwardConfig"]
