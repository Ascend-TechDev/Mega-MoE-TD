# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Central model / shape configuration package.

Re-exports the public config surface from :mod:`config._shapes` so callers can
write ``from config import MoETestShape, FORWARD_SHAPES, MODEL_PROFILES, ...``.
"""

from config._shapes import (
    BACKWARD_SHAPES_KIMI,
    BACKWARD_SHAPES_PERF,
    BACKWARD_SHAPES_SMALL,
    FORWARD_SHAPES,
    FORWARD_SHAPES_KIMI,
    MODEL_PROFILES,
    MoETestShape,
    select_perf_shapes,
)

__all__ = [
    "MoETestShape",
    "FORWARD_SHAPES",
    "FORWARD_SHAPES_KIMI",
    "BACKWARD_SHAPES_SMALL",
    "BACKWARD_SHAPES_KIMI",
    "BACKWARD_SHAPES_PERF",
    "MODEL_PROFILES",
    "select_perf_shapes",
]
