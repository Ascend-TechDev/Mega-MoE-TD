# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Operator package: public entry points for the fused MoE autograd Functions.

``mega_moe.ops.MegaMoEFunction`` is the Stage H3 merged native Function (plan
§3.2.3) — fused forward with ``return_saved=True`` + the fused 5-op backward,
exported here as the primary import path for the training-side adapter.
"""

from .backward import (
    MegaMoEBackwardFunction,
    MegaMoEFunction,
    moe_backward_triton,
)

__all__ = [
    "MegaMoEFunction",
    "MegaMoEBackwardFunction",
    "moe_backward_triton",
]
