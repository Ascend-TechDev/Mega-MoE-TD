# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Configuration for the standalone Ascend Mega-MoE forward."""

from dataclasses import dataclass
from typing import Optional


_DISPATCH_FC1_SCHEDULES = (
    "static",
    "count",
    "allcore",
    "allcore_expert",
    "allcore_expert_mn",
    "allcore_expert_n",
    "allcore_expert_n_tile",
)


@dataclass(frozen=True)
class MoEForwardConfig:
    """Stage-specific launch and tiling parameters.

    The defaults preserve the previously tuned implementation.  FC1 and FC2
    tiles are deliberately independent because they have different shapes and
    data-movement paths.

    ``dispatch_fc1_block_size_m`` controls both dispatch readiness slots and
    FC1 dot rows; ``fc1_gemm_block_size_{n,k}`` control the other FC1 dot axes.
    Likewise, ``fc2_combine_block_size_m`` controls FC2/reverse-A2A row tiles, while
    ``fc2_gemm_block_size_{n,k}`` control only FC2 dot tiles.

    ``dispatch_producer_cores`` is used by the split-role ``static`` and
    ``count`` schedules.  All-core schedules still receive the resolved value
    as a compile-time argument, but do not assign fixed producer-only cores.
    """

    num_aicore_programs: int = 24
    receive_capacity_factor: Optional[float] = None
    dispatch_fc1_block_size_m: int = 128
    fc1_gemm_block_size_n: int = 256
    fc1_gemm_block_size_k: int = 128
    fc2_combine_block_size_m: int = 64
    fc2_gemm_block_size_n: int = 128
    fc2_gemm_block_size_k: int = 128
    dispatch_producer_cores: Optional[int] = None
    dispatch_readiness: str = "tile"
    dispatch_fc1_schedule: str = "allcore_expert_n_tile"

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
        if self.dispatch_readiness not in ("expert", "tile"):
            raise ValueError("dispatch_readiness must be 'expert' or 'tile'")
        if self.dispatch_fc1_schedule not in _DISPATCH_FC1_SCHEDULES:
            raise ValueError(
                "dispatch_fc1_schedule must be one of "
                + ", ".join(repr(value) for value in _DISPATCH_FC1_SCHEDULES)
            )
        expert_schedules = {
            "allcore_expert",
            "allcore_expert_mn",
            "allcore_expert_n",
        }
        if self.dispatch_fc1_schedule in expert_schedules and self.dispatch_readiness != "expert":
            raise ValueError(
                f"{self.dispatch_fc1_schedule} requires dispatch_readiness='expert'"
            )
        tile_schedules = {"count", "allcore", "allcore_expert_n_tile"}
        if self.dispatch_fc1_schedule in tile_schedules and self.dispatch_readiness != "tile":
            raise ValueError(
                f"{self.dispatch_fc1_schedule} requires dispatch_readiness='tile'"
            )
        producer_cores = self.dispatch_producer_cores
        if producer_cores is not None and not 0 < producer_cores < self.num_aicore_programs:
            raise ValueError(
                "dispatch_producer_cores must be in [1, num_aicore_programs)"
            )

    def resolved_receive_capacity_factor(self, world_size: int) -> float:
        """Return the configured capacity, or the worst-case-safe default."""
        if self.receive_capacity_factor is None:
            return float(world_size)
        return float(self.receive_capacity_factor)

    def resolved_dispatch_producer_cores(self) -> int:
        """Return the producer-core count used by split-role schedules."""
        if self.dispatch_producer_cores is not None:
            return self.dispatch_producer_cores
        return max(1, min(self.num_aicore_programs - 1, self.num_aicore_programs // 5))


__all__ = ["MoEForwardConfig"]
