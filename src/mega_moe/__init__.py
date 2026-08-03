"""Public API for Ascend Mega-MoE forward and backward operators."""

from .config import MoEForwardConfig
from .ops.backward import MegaMoEBackwardFunction, moe_backward_triton
from .ops.forward import DispatchFC1Result, FusedMoEForward, pack_gate_up_weights
from .runtime.routing import MoERoutingPlan

__all__ = [
    "DispatchFC1Result",
    "FusedMoEForward",
    "MegaMoEBackwardFunction",
    "MoEForwardConfig",
    "MoERoutingPlan",
    "moe_backward_triton",
    "pack_gate_up_weights",
]
