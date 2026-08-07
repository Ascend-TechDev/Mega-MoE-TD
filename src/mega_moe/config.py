# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration for the standalone Ascend Mega-MoE forward."""

from dataclasses import dataclass
from typing import Optional


_DISPATCH_FC1_SCHEDULES = (
    "allcore_expert_n_tile",
)


# Supported post-FC1 gated activations.  swiglu is silu(gate) * up;
# situglu is beta * tanh(gate / beta) * sigmoid(gate) * up (with an
# optional linear_beta * tanh(up / linear_beta) transform on up).
_ACTIVATIONS = (
    "swiglu",
    "situglu",
)


@dataclass(frozen=True)
class MoEForwardConfig:
    """Stage-specific launch and tiling parameters.

    FC1 uses the fixed all-core expert/N-tile pipeline.  FC1 and FC2 tiles
    remain independent because they have different shapes and data-movement
    paths.

    ``dispatch_fc1_block_size_m`` controls both dispatch readiness slots and
    FC1 dot rows; ``fc1_gemm_block_size_{n,k}`` control the other FC1 dot axes.
    Likewise, ``fc2_combine_block_size_m`` controls FC2/direct-pull row tiles,
    while ``fc2_gemm_block_size_{n,k}`` control only FC2 dot tiles.  FC2 uses
    the measured persistent expert-N GEMM, direct-pull transport, and both
    Vector sub-cores for reduction; dominated A/B controls are not public
    configuration fields.  The post-FC1 activation remains selectable for
    compatibility with the target repository SiTU-GLU path.

    activation selects SwiGLU or SiTU-GLU; situ_beta and situ_linear_beta
    configure the latter and are ignored for SwiGLU.
    """

    num_aicore_programs: int = 24
    receive_capacity_factor: Optional[float] = None
    dispatch_fc1_block_size_m: int = 128
    fc1_gemm_block_size_n: int = 256
    fc1_gemm_block_size_k: int = 128
    fc2_combine_block_size_m: int = 128
    fc2_gemm_block_size_n: int = 256
    fc2_gemm_block_size_k: int = 128
    dispatch_fc1_schedule: str = "allcore_expert_n_tile"

    # Post-FC1 gated activation.  swiglu (default) preserves the original
    # silu(gate) * up path; situglu selects SiTU-GLU.
    activation: str = "swiglu"
    situ_beta: float = 1.0
    situ_linear_beta: Optional[float] = None

    def __post_init__(self):
        if self.num_aicore_programs < 2:
            raise ValueError("num_aicore_programs must be at least 2")

        for name, value in (
            ("dispatch_fc1_block_size_m", self.dispatch_fc1_block_size_m),
            ("fc1_gemm_block_size_n", self.fc1_gemm_block_size_n),
            ("fc1_gemm_block_size_k", self.fc1_gemm_block_size_k),
            ("fc2_combine_block_size_m", self.fc2_combine_block_size_m),
            ("fc2_gemm_block_size_n", self.fc2_gemm_block_size_n),
            ("fc2_gemm_block_size_k", self.fc2_gemm_block_size_k),
        ):
            if value < 16 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two no smaller than 16")

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
