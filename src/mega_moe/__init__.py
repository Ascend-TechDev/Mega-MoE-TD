"""Public API for Ascend Mega-MoE forward and backward operators.

Exports are deliberately lazy: the verified host bootstrap may import the
package and inspect its origin without importing a device driver, Triton
kernel, or production operator before the terminal runtime seal.
"""

import importlib


_LAZY_EXPORTS = {
    "FusedMoEForward": (".ops.forward", "FusedMoEForward"),
    "DispatchFC1Result": (".ops.forward", "DispatchFC1Result"),
    "pack_gate_up_weights": (".ops.forward", "pack_gate_up_weights"),
    "MegaMoEBackwardFunction": (".ops.backward", "MegaMoEBackwardFunction"),
    "moe_backward_triton": (".ops.backward", "moe_backward_triton"),
    "MoEForwardConfig": (".config", "MoEForwardConfig"),
    "MoERoutingPlan": (".runtime.routing", "MoERoutingPlan"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
