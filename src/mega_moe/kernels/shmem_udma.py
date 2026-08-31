# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Ascend ACLSHMEM low-level UDMA bindings missing from Triton-Dist 1.6.

The installed Triton binding exposes only generic RMA.  When UDMA is enabled,
generic RMA uses the high-level PIPE_MTE3 path and requires a fixed UB WQE
scratch.  These narrow externs select SHMEM's PIPE_S implementation
instead: scalar instructions publish the small WQE directly to the SQ, while
UDMA still transports the weight payload itself.
"""

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call


_BF16_PTR = core.pointer_type(core.dtype("bf16"))
_INT32_PTR = core.pointer_type(core.dtype("int32"))


@core.extern
def udma_bfloat16_put_nbi(
    destination,
    source,
    num_elements,
    destination_rank,
    _builder=None,
):
    """Submit one BF16 PIPE_S UDMA put without using a UB staging buffer."""
    return extern_call(
        "libshmem_device",
        "",
        [
            destination,
            source,
            tl.cast(num_elements, tl.uint32, _builder=_builder),
            tl.cast(destination_rank, tl.int32, _builder=_builder),
        ],
        {
            (_BF16_PTR, _BF16_PTR, tl.uint32, tl.int32): (
                "aclshmemi_udma_put_nbi",
                (),
            ),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def udma_bfloat16_put_signal_nbi(
    destination,
    source,
    num_elements,
    signal_address,
    signal,
    destination_rank,
    _builder=None,
):
    """Submit one BF16 PIPE_S WRITE_WITH_NOTIFY UDMA request.

    The remote int32 epoch occupies the low word of the aligned uint64 notify
    slot reserved by the replica readiness table.
    """
    return extern_call(
        "libshmem_device",
        "",
        [
            destination,
            source,
            tl.cast(num_elements, tl.uint32, _builder=_builder),
            signal_address,
            tl.cast(signal, tl.int32, _builder=_builder),
            tl.cast(destination_rank, tl.int32, _builder=_builder),
        ],
        {
            (
                _BF16_PTR,
                _BF16_PTR,
                tl.uint32,
                _INT32_PTR,
                tl.int32,
                tl.int32,
            ): ("aclshmemx_udma_put_signal_nbi", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def udma_quiet(destination_rank, _builder=None):
    """Wait for all UDMA work previously submitted to one destination PE."""
    return extern_call(
        "libshmem_device",
        "",
        [tl.cast(destination_rank, tl.int32, _builder=_builder)],
        {
            (tl.int32,): ("aclshmemx_udma_quiet", ()),
        },
        is_pure=False,
        _builder=_builder,
    )


__all__ = [
    "udma_bfloat16_put_nbi",
    "udma_bfloat16_put_signal_nbi",
    "udma_quiet",
]
