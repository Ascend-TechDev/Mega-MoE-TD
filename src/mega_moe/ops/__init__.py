# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Operator layer of the Ascend Mega-MoE tutorial.

Stage H3 (master plan §3.2.3) exports the merged native-saved autograd
Function here so training code can ``from mega_moe.ops import MegaMoEFunction``
without reaching into the implementation modules.
"""

from .backward import (
    MegaMoEBackwardFunction,
    MegaMoEFunction,
    moe_backward_triton,
)
from .forward import DispatchFC1Result, FusedMoEForward, pack_gate_up_weights

__all__ = [
    "DispatchFC1Result",
    "FusedMoEForward",
    "MegaMoEBackwardFunction",
    "MegaMoEFunction",
    "moe_backward_triton",
    "pack_gate_up_weights",
]
