# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Owner packing and lifecycle kernels for MoonEP replica-weight panel PUTs.

Owners pack only the unique experts requested by the current routing plan into
ordinary compact staging.  The mixed dispatch/FC1 kernel pushes each panel to
its symmetric consumer destination with hardware readiness notification.  A
consumer-owned epoch protects both staging and destination-slot reuse without a
rank-wide fence in the normal forward path.
"""

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton_dist.language.extra import libshmem_device
from .shmem_udma import udma_quiet


@triton.jit(do_not_specialize=["previous_epoch", "replica_slot_count"])
def _kernel_wait_previous_replica_consumed(
    previous_experts_to_copy_ptr,
    consumed_epoch_ptr,
    previous_epoch,
    replica_slot_count,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Protect local staging reuse with consumer-owned completion epochs."""
    pid = tl.program_id(0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if (pid == 0) & (previous_epoch > 0):
            for destination_rank in tl.static_range(0, WORLD_SIZE):
                if destination_rank != LOCAL_RANK:
                    replica_slot = 0
                    while replica_slot < replica_slot_count:
                        descriptor = (
                            destination_rank * EXPERTS_PER_RANK + replica_slot
                        )
                        global_expert = tl.load(
                            previous_experts_to_copy_ptr + descriptor
                        )
                        owned = (
                            (global_expert >= 0)
                            & (
                                global_expert // EXPERTS_PER_RANK
                                == LOCAL_RANK
                            )
                        )
                        if owned:
                            remote_consumed = libshmem_device.remote_ptr(
                                consumed_epoch_ptr + replica_slot * 16,
                                destination_rank,
                            )
                            libshmem_device.signal_wait_until(
                                remote_consumed,
                                libshmem_device.ACLSHMEM_CMP_EQ,
                                previous_epoch,
                            )
                        replica_slot += 1


@triton.jit(do_not_specialize=["signal_epoch"])
def _kernel_pack_requested_replica_weight_panels(
    gate_up_weight_ptr,
    down_weight_ptr,
    source_gate_up_ptr,
    source_down_ptr,
    source_slot_by_global_expert_ptr,
    source_ready_ptr,
    signal_epoch,
    NUM_PROGRAMS: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    FFN_SIZE: tl.constexpr,
    COPY_BLOCK_ELEMENTS: tl.constexpr,
    GATE_BLOCK_K: tl.constexpr,
    GATE_BLOCK_N: tl.constexpr,
):
    """Pack one expert panel per program and publish it immediately."""
    pid = tl.program_id(0)
    gate_panel_elements: tl.constexpr = HIDDEN_SIZE * FFN_SIZE
    down_panel_elements: tl.constexpr = (HIDDEN_SIZE // 2) * FFN_SIZE
    num_tasks: tl.constexpr = EXPERTS_PER_RANK * 4

    with al.scope(core_mode="vector", disable_auto_sync=True):
        task_id = pid
        while task_id < num_tasks:
            owner_local_expert = task_id // 4
            panel_kind = task_id % 4
            global_expert = (
                LOCAL_RANK * EXPERTS_PER_RANK + owner_local_expert
            )
            source_slot = tl.load(
                source_slot_by_global_expert_ptr + global_expert
            )
            if source_slot >= 0:
                if panel_kind < 2:
                    panel = panel_kind
                    k_inner = tl.arange(0, GATE_BLOCK_K)
                    n_inner = tl.arange(0, GATE_BLOCK_N)
                    for k_start in range(0, HIDDEN_SIZE, GATE_BLOCK_K):
                        k_offsets = k_start + k_inner
                        k_mask = k_offsets < HIDDEN_SIZE
                        for n_start in tl.static_range(
                            0, FFN_SIZE, GATE_BLOCK_N
                        ):
                            n_offsets = n_start + n_inner
                            n_mask = n_offsets < FFN_SIZE
                            mask = k_mask[:, None] & n_mask[None, :]
                            source_offsets = (
                                owner_local_expert * (2 * gate_panel_elements)
                                + k_offsets[:, None] * (2 * FFN_SIZE)
                                + panel * FFN_SIZE
                                + n_offsets[None, :]
                            )
                            destination_offsets = (
                                source_slot * (2 * gate_panel_elements)
                                + panel * gate_panel_elements
                                + k_offsets[:, None] * FFN_SIZE
                                + n_offsets[None, :]
                            )
                            values = tl.load(
                                gate_up_weight_ptr + source_offsets,
                                mask=mask,
                                other=0.0,
                            )
                            tl.store(
                                source_gate_up_ptr + destination_offsets,
                                values,
                                mask=mask,
                            )
                else:
                    panel = panel_kind - 2
                    copy_start = 0
                    while copy_start < down_panel_elements:
                        panel_offsets = (
                            copy_start + tl.arange(0, COPY_BLOCK_ELEMENTS)
                        )
                        mask = panel_offsets < down_panel_elements
                        source_offsets = (
                            owner_local_expert * (2 * down_panel_elements)
                            + panel * down_panel_elements
                            + panel_offsets
                        )
                        destination_offsets = (
                            source_slot * (2 * down_panel_elements)
                            + panel * down_panel_elements
                            + panel_offsets
                        )
                        values = tl.load(
                            down_weight_ptr + source_offsets,
                            mask=mask,
                            other=0.0,
                        )
                        tl.store(
                            source_down_ptr + destination_offsets,
                            values,
                            mask=mask,
                        )
                        copy_start += COPY_BLOCK_ELEMENTS
                libshmem_device.fence()
                tl.store(
                    source_ready_ptr
                    + (source_slot * 4 + panel_kind) * 16,
                    signal_epoch,
                )
            task_id += NUM_PROGRAMS


@triton.jit(
    do_not_specialize=["signal_epoch", "replica_slot_count"]
)
def _kernel_publish_replica_consumed(
    experts_to_copy_ptr,
    down_ready_ptr,
    consumed_epoch_ptr,
    signal_epoch,
    replica_slot_count,
    LOCAL_RANK: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
):
    """Release active and empty-token slots after their last panel arrives."""
    pid = tl.program_id(0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if pid == 0:
            replica_slot = 0
            while replica_slot < replica_slot_count:
                descriptor = LOCAL_RANK * EXPERTS_PER_RANK + replica_slot
                global_expert = tl.load(experts_to_copy_ptr + descriptor)
                if global_expert >= 0:
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
    "_kernel_pack_requested_replica_weight_panels",
    "_kernel_publish_replica_consumed",
    "_kernel_quiet_replica_panel_pushes",
    "_kernel_wait_previous_replica_consumed",
]
