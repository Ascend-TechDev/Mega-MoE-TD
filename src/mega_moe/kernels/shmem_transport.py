# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Engine-selective ACLSHMEM helpers for Ascend Triton kernels.

ACLSHMEM's high-level ``putmem`` selects the highest-priority initialized
transport.  With the MoonEP MTE|UDMA configuration that means UDMA, which is
desirable for large replica weights but not for the fine-grained dispatch and
metadata traffic.  The latter uses the MTE-visible symmetric address directly
so its transport does not depend on the high-level priority order.
"""

import triton
import triton.language as tl
from triton_dist.language.extra import libshmem_device


@triton.jit
def mte_put(
    destination,
    source,
    num_elements,
    destination_rank,
    BLOCK_ELEMENTS: tl.constexpr,
):
    """Copy typed elements to a peer through its MTE/P2P GM mapping.

    ``destination`` must be in the ACLSHMEM symmetric heap.  Translating it
    with ``aclshmem_ptr`` bypasses generic RMA transport selection; Triton GM
    loads/stores then lower to the normal MTE data path.  Callers retain their
    existing ``fence``/signal or collective completion point.
    """
    remote_destination = libshmem_device.remote_ptr(
        destination,
        destination_rank,
    )
    copy_start = 0
    while copy_start < num_elements:
        offsets = copy_start + tl.arange(0, BLOCK_ELEMENTS)
        mask = offsets < num_elements
        values = tl.load(source + offsets, mask=mask, other=0)
        tl.store(remote_destination + offsets, values, mask=mask)
        copy_start += BLOCK_ELEMENTS


__all__ = ["mte_put"]
