# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Single source of truth for the local NPU device number.

Kernel parameter ``LOCAL_RANK`` is the ACLSHMEM *global PE* (``ep_group.rank()``)
and has nothing to do with torchrun's ``$LOCAL_RANK``.  On a single node the two
numbers coincide, which is why historical call sites pass the PE where a
*device* is required (``aclshmem_create_tensor(device_id=...)``, ``f"npu:{...}"``).
Multi-node breaks the coincidence — node 1's PEs are 8..15 while only devices
0..7 exist — so every host-side device choice must resolve through this module.

Resolution order (``resolve_local_device``):

1. ``MEGAMOE_LOCAL_DEVICE`` env — explicit override (debugging, odd launchers);
2. ``MEGAMOE_MULTI_NODE=1`` → ``torch.npu.current_device()``;
3. otherwise the PE itself — bit-identical to the historical single-node
   behavior, which is the G0 regression contract.
"""

import os
from typing import Optional


def multi_node_enabled() -> bool:
    """Whether multi-node device resolution is active (default off)."""
    return os.environ.get("MEGAMOE_MULTI_NODE", "0") == "1"


def resolve_local_device(pe_rank: int) -> int:
    """Return the local NPU ordinal for ``aclshmem_create_tensor``/``f"npu:"``."""
    override = os.environ.get("MEGAMOE_LOCAL_DEVICE")
    if override is not None:
        return int(override)
    if multi_node_enabled():
        import torch
        import torch_npu  # noqa: F401  # Register the torch.npu backend.

        return int(torch.npu.current_device())
    return pe_rank


def saved_device_id(saved: Optional[dict]) -> int:
    """Local device for a backward ``saved`` dict; old dicts fall back to the PE."""
    if saved is not None and saved.get("local_device") is not None:
        return int(saved["local_device"])
    if saved is None:
        raise ValueError("saved contract dict is required")
    return resolve_local_device(int(saved["ep_rank"]))


def device_str(dev: int) -> str:
    """Uniform ``f"npu:{dev}"`` spelling for the ~30 historical call sites."""
    return f"npu:{dev}"


__all__ = [
    "multi_node_enabled",
    "resolve_local_device",
    "saved_device_id",
    "device_str",
]
