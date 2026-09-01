# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Lifecycle kernels for MoonEP replica-weight panel PUTs."""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton_dist.language.extra import libshmem_device
from .shmem_udma import udma_quiet


@triton.jit(
    do_not_specialize=["signal_epoch", "replica_slot_count"]
)
def _kernel_publish_replica_consumed(
    experts_to_copy_ptr,
    gate_ready_ptr,
    down_ready_ptr,
    consumed_epoch_ptr,
    signal_epoch,
    replica_slot_count,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Release active and empty-token slots after all panels arrive."""
    pid = tl.program_id(0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if pid == 0:
            replica_slot = 0
            while replica_slot < replica_slot_count:
                descriptor = LOCAL_RANK * EXPERTS_PER_RANK + replica_slot
                global_expert = tl.load(experts_to_copy_ptr + descriptor)
                if global_expert >= 0:
                    libshmem_device.signal_wait_until(
                        gate_ready_ptr + replica_slot * 16,
                        libshmem_device.ACLSHMEM_CMP_EQ,
                        signal_epoch,
                    )
                    libshmem_device.signal_wait_until(
                        down_ready_ptr + replica_slot * 2 * 16,
                        libshmem_device.ACLSHMEM_CMP_EQ,
                        signal_epoch,
                    )
                    libshmem_device.signal_wait_until(
                        down_ready_ptr + (replica_slot * 2 + 1) * 16,
                        libshmem_device.ACLSHMEM_CMP_EQ,
                        signal_epoch,
                    )
                # Inactive padding slots are also advanced.  A later plan may
                # assign them to a different owner, which can then use the same
                # uniform previous-epoch wait without an inactive-slot bitmap.
                tl.store(
                    consumed_epoch_ptr + replica_slot * 16,
                    signal_epoch,
                )
                replica_slot += 1
            libshmem_device.fence()


@triton.jit
def _kernel_quiet_replica_panel_pushes(
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """Drain owner PUT QPs only before resizing or freeing symmetric memory."""
    pid = tl.program_id(0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (pid < WORLD_SIZE) & (pid != LOCAL_RANK):
            udma_quiet(pid)
        libshmem_device.barrier_all_vec()

__all__ = [
    "_kernel_publish_replica_consumed",
    "_kernel_quiet_replica_panel_pushes",
]
