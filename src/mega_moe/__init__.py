"""Public API for Ascend Mega-MoE forward and backward operators."""

# Forward.
from .ops.forward import DispatchFC1Result, FusedMoEForward, pack_gate_up_weights
# Backward.
from .ops.backward import MegaMoEBackwardFunction, moe_backward_triton
# Configuration & routing plan.
from .config import MoEForwardConfig
from .runtime.routing import MoERoutingPlan

__all__ = [
    # Forward
    "FusedMoEForward",
    "DispatchFC1Result",
    "pack_gate_up_weights",
    # Backward
    "MegaMoEBackwardFunction",
    "moe_backward_triton",
    # Configuration & routing plan
    "MoEForwardConfig",
    "MoERoutingPlan",
]
