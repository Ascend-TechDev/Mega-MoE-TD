"""P4 acquire regression: immutable ready weights, transposed GEMM B operand.

The failure must be detectable without UDMA or gradient-slot reuse. Exercise
partial tiles, nonzero expert bases, repeated epochs and changed weight data.
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from mega_moe.kernels.common import ncore
from mega_moe.kernels.mega_bwd import _mega_combine_gemm


@triton.jit
def _p4_ready_probe(x, w, out, experts, row0, rows, ready, epoch,
                    N: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr,
                    NCORES: tl.constexpr, WAIT: tl.constexpr):
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _mega_combine_gemm(
            tl.program_id(0), NCORES, x, K, 1, w, N * K, 1, K, 112,
            out, experts, row0, rows, N, K, triton.cdiv(N, 256), TILES,
            0, TILES, ready, epoch, 256, 256, 128, 2,
            SIGNAL_ON=False, LOCAL_RANK=0, GROUP_M=8,
            replica_weight_ready_ptr=ready, replica_weight_epoch=epoch,
            WAIT_REPLICA_WEIGHTS=WAIT)


def run_p4_ready_probe(rank, world_size):
    device = f"npu:{rank}"
    torch.manual_seed(411 + rank)
    n, k = 3584, 6144
    slot_ids = [0, 2, 1, 3, 2, 3]
    counts = [256, 33, 256, 129, 1, 256]
    rows = torch.tensor(counts, device=device, dtype=torch.int32)
    row0 = rows.cumsum(0).to(torch.int32) - rows
    experts = torch.tensor(slot_ids, device=device, dtype=torch.int32) + 112
    x = torch.randn((sum(counts), k), device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn((4, n, k), device=device, dtype=torch.bfloat16) * 0.1
    out = torch.empty((sum(counts), n), device=device, dtype=torch.bfloat16)
    ready = torch.zeros(4 * 16, device=device, dtype=torch.int32)
    for epoch in (7, 19):
        w.add_(0.03125)
        expected = torch.cat([
            part @ w[slot].T for part, slot in zip(x.split(counts), slot_ids)
        ])
        ready.fill_(epoch)
        for wait in (False, True):
            _p4_ready_probe[(ncore(),)](
                x, w, out, experts, row0, rows, ready, epoch, n, k,
                len(counts), ncore(), wait, num_warps=8,
                limit_auto_multi_buffer_of_local_buffer="no-l0c")
            torch.npu.synchronize()
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.dist
def test_p4_acquire_transposed_weight(dist_test):
    dist_test(run_p4_ready_probe, world_size=8)
