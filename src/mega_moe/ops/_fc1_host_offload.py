# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host offload of ``saved["fc1_output"]`` between forward and backward.

The saved-contract backward keeps one big activation on device across the
forward/backward boundary: ``fc1_output`` ([total_recv, 2F] bf16, ~100MB at
the kimi mock shape, up to ~805MB at capacity routing).  With
``MEGAMOE_FC1_OFFLOAD=1`` the MegaMoEFunction layer copies it to a pinned
host buffer on a side NPU stream right after the saved dict is finalized
(post enrichment — what lands on host is the forward's own fc1_output
snapshot; since the plan-B adapter no longer reorder-copies it) and
releases the device block immediately; the backward entry copies it back
(H2D) before anything reads it.  Everything else stays on device.

This is the operator-side counterpart of the framework's activation
offload (mindspeed_mm/fsdp/features/memory/async_offload.py): the native
saved dict does NOT flow through ``saved_tensors_hooks``, so the framework
mechanism cannot see ``fc1_output`` — the swap must live where the dict
lives, on MegaMoEFunction.

Discipline (SwapTensor idiom, stream-ordered end to end — no host sync):

  D2H (forward)   fwd_ev on current stream -> swap_stream waits -> pinned
                  copy -> d2h_ev on swap_stream -> ``fc1.record_stream``
                  so the caching allocator defers block reuse until the
                  copy retires, then the Python ref is dropped.
  H2D (backward)  current stream waits d2h_ev -> fresh device tensor +
                  non_blocking copy from pinned -> h2d_ev; the mega launch
                  on the same stream is ordered behind the copy for free.

Pinned buffers are pooled per device and reused across steps (a fresh
100MB pin_memory allocation per step would cost more than the copy).  A
pool entry's last H2D event gates its next D2H write — release/acquire is
stream-ordered too.
"""

import os

import torch

_ENV = "MEGAMOE_FC1_OFFLOAD"


def fc1_offload_enabled():
    return os.environ.get(_ENV, "0") == "1"


class _PinnedEntry:
    """One pooled pinned host buffer (flat uint8) + its ordering event."""

    __slots__ = ("buf", "cap", "event", "in_use")

    def __init__(self, cap):
        self.buf = torch.empty(cap, dtype=torch.uint8, pin_memory=True,
                               device="cpu")
        self.cap = cap
        self.event = None          # last H2D read of this buffer, if any
        self.in_use = False


class _Fc1OffloadManager:
    """Per-device side stream + pinned pool + counters (for the gates)."""

    def __init__(self):
        self._per_device = {}
        # observable stats so tests can assert the swap actually ran
        self.d2h_count = 0
        self.h2d_count = 0
        self.d2h_bytes = 0

    def state(self, device):
        st = self._per_device.get(device)
        if st is None:
            st = {
                "swap_stream": torch.npu.Stream(device=device),
                "pool": [],
            }
            self._per_device[device] = st
        return st

    def acquire(self, nbytes, device, swap_stream):
        """Best-fit pooled pinned buffer of at least ``nbytes`` bytes."""
        pool = self.state(device)["pool"]
        best = None
        for entry in pool:
            if entry.in_use or entry.cap < nbytes:
                continue
            if best is None or entry.cap < best.cap:
                best = entry
        if best is None:
            best = _PinnedEntry(nbytes)
            pool.append(best)
        else:
            # the previous H2D read of this buffer must retire before the
            # swap stream overwrites it
            if best.event is not None:
                swap_stream.wait_event(best.event)
                best.event = None
        best.in_use = True
        return best

    def release(self, entry, event):
        entry.event = event
        entry.in_use = False


_MANAGER = _Fc1OffloadManager()


def fc1_offload_stats():
    """``(d2h_count, h2d_count, d2h_bytes)`` — gate assertions read this."""
    return (_MANAGER.d2h_count, _MANAGER.h2d_count, _MANAGER.d2h_bytes)


class _Fc1HostSwap:
    """The holder parked at ``saved["_fc1_host_swap"]`` while offloaded."""

    __slots__ = ("entry", "shape", "dtype", "d2h_event")

    def __init__(self, entry, shape, dtype, d2h_event):
        self.entry = entry
        self.shape = shape
        self.dtype = dtype
        self.d2h_event = d2h_event


def maybe_offload_fc1(saved):
    """Forward tail: swap ``saved["fc1_output"]`` to pinned host memory.

    No-op unless ``MEGAMOE_FC1_OFFLOAD=1`` and the dict carries a device
    ``fc1_output``.  The key is popped — anything reading the saved dict
    before the backward entry restore fails loudly instead of silently
    reading a holder object.
    """
    fc1 = saved.get("fc1_output")
    if (not fc1_offload_enabled() or not isinstance(fc1, torch.Tensor)
            or fc1.device.type != "npu" or fc1.numel() == 0):
        return
    device = fc1.device
    flat_bytes = fc1.reshape(-1).view(torch.uint8)
    nbytes = flat_bytes.numel()
    st = _MANAGER.state(device)
    swap_stream = st["swap_stream"]
    entry = _MANAGER.acquire(nbytes, device, swap_stream)
    host_view = entry.buf[:nbytes]

    fwd_event = torch.npu.Event()
    fwd_event.record()                     # current stream: fc1 finalized
    swap_stream.wait_event(fwd_event)
    with torch.npu.stream(swap_stream):
        host_view.copy_(flat_bytes, non_blocking=True)
        d2h_event = torch.npu.Event()
        d2h_event.record(swap_stream)
    # the allocator may only hand this block out once the copy retires
    fc1.record_stream(swap_stream)

    saved.pop("fc1_output")
    saved["_fc1_host_swap"] = _Fc1HostSwap(
        entry, tuple(fc1.shape), fc1.dtype, d2h_event,
    )
    _MANAGER.d2h_count += 1
    _MANAGER.d2h_bytes += nbytes


def maybe_reload_fc1(saved):
    """Backward entry: H2D the swapped ``fc1_output`` back into the dict.

    Restores ``saved["fc1_output"]`` as a fresh device tensor whose copy is
    stream-ordered ahead of every consumer (the one-launch mega kernel and
    the orchestrator both launch on the current stream, after this).
    """
    swap = saved.pop("_fc1_host_swap", None)
    if swap is None:
        return
    device = torch.device("npu", torch.npu.current_device())
    cur = torch.npu.current_stream(device)
    cur.wait_event(swap.d2h_event)        # pinned buffer fully written
    fc1 = torch.empty(swap.shape, dtype=swap.dtype, device=device)
    nbytes = fc1.numel() * fc1.element_size()
    with torch.npu.stream(cur):
        fc1.reshape(-1).view(torch.uint8).copy_(
            swap.entry.buf[:nbytes], non_blocking=True)
        h2d_event = torch.npu.Event()
        h2d_event.record(cur)
    saved["fc1_output"] = fc1
    _MANAGER.release(swap.entry, h2d_event)
    _MANAGER.h2d_count += 1


__all__ = [
    "fc1_offload_enabled", "fc1_offload_stats",
    "maybe_offload_fc1", "maybe_reload_fc1",
]
