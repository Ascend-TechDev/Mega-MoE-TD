# coding=utf-8
"""MoonEP VM padding 行清零（每步 planning 之后、dispatch 之前独立 launch）。

为什么必须独立 kernel：段长随 plan 变化，VM 是脏内存；混合 all-core
dispatch kernel 在 ``disable_auto_sync=True`` 下 Cube 只等信号——padding
行没有任何源 tile 覆盖，必须在 dispatch 之前由**先序 launch** 清零
（流序保证可见）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["launch_moonep_zero_fill", "moonep_zero_fill"]


@triton.jit
def moonep_zero_fill(
    vm_ptr,                 # bf16 [rows_pad, H]（展平）
    w_ptr,                  # fp32 [rows_pad]
    zfr_ptr,                # int32 [E+B, 2]（[start, count)）
    n_ranges,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,  # 2 的幂，≥ H 向上取整
):
    for i in range(0, n_ranges):
        start = tl.load(zfr_ptr + i * 2)
        count = tl.load(zfr_ptr + i * 2 + 1)
        for r in range(0, count):
            for h0 in range(0, H, BLOCK_H):
                offs = h0 + tl.arange(0, BLOCK_H)
                m = offs < H
                tl.store(vm_ptr + (start + r) * H + offs,
                         tl.zeros((BLOCK_H,), dtype=tl.bfloat16), mask=m)
            tl.store(w_ptr + start + r, 0.0)


def launch_moonep_zero_fill(vm: torch.Tensor, w_recv: torch.Tensor,
                            zero_fill_ranges: torch.Tensor, H: int):
    n = zero_fill_ranges.shape[0]
    block_h = max(16, 1 << (H - 1).bit_length())
    moonep_zero_fill[(1, 1, 1)](
        vm, w_recv, zero_fill_ranges.contiguous(), n, H=H, BLOCK_H=block_h)
