# coding=utf-8
"""MoonEP combine：FC2 grouped GEMM + 按 src_info 推回 + top-k 归约。

v1 结构（dispatch 已含全组 barrier，数据就绪，无需信号协议）：

1. ``moonep_fc2_gemm``：Seg 段分组 GEMM（act [rows_pad,F] × down →
   fc2_out [rows_pad,H]）。**复用** dispatch_fc1 的
   ``_triton_grouped_gemm_one_mn_tile_tail``（通用 tile 函数）。
   权重视图注意（b[k,n] 步长约定，classic 同款）：
   - FC1：b[k,n]=phys[k,n] → 传 ``gate_up.transpose(-1,-2)``；
   - FC2：b[f,n]=phys[n,f] → 传 ``down`` 原样（[Seg,H,F] 即逻辑 [Seg,N=H,K=F]）。
2. ``moonep_combine_push``：逐 VM 行按 ``src_info[r] = sr·NvS + offv``
   putmem 推回源 rank 的 combine_buf 行（每条目一行，dup 条目 v1 有自己的
   payload 行）。尾部 barrier_all。
3. ``moonep_topk_reduce``：out[t] = Σ_k combine_buf[t·K+k]——**纯求和**，
   路由权重已在 weighted_swiglu 里乘过（classic 语义）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .dispatch_fc1 import _triton_grouped_gemm_one_mn_tile_tail

__all__ = ["launch_moonep_combine", "moonep_combine_push",
           "moonep_fc2_gemm", "moonep_topk_reduce"]


@triton.jit
def moonep_fc2_gemm(
    act_ptr, weight_ptr, output_ptr,
    seg_counts_ptr, seg_offsets_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    NUM_PROGRAM_CORES: tl.constexpr,
    SEG: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dtype = tl.bfloat16
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_tasks = SEG * num_n_tiles
    for task_id in range(pid, num_tasks, NUM_PROGRAM_CORES):
        seg = task_id // num_n_tiles
        n_tile = task_id % num_n_tiles
        seg_size = tl.load(seg_counts_ptr + seg)
        seg_off = tl.load(seg_offsets_ptr + seg)
        if seg_size > 0:
            num_m_windows = tl.cdiv(seg_size, BLOCK_SIZE_M)
            for m_window in range(0, num_m_windows):
                w_start = m_window * BLOCK_SIZE_M
                w_size = tl.minimum(BLOCK_SIZE_M, seg_size - w_start)
                _triton_grouped_gemm_one_mn_tile_tail(
                    act_ptr, weight_ptr, output_ptr,
                    seg, seg_off + w_start, w_size, n_tile, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


@triton.jit
def moonep_combine_push(
    fc2_ptr,                # bf16 [rows_pad, H]
    src_info_ptr,           # int32 [NvS]：sr·NvS + offv（-1 空槽）
    combine_buf_ptr,        # bf16 [N, H] 对称
    num_rows,
    NvS: tl.constexpr,
    H: tl.constexpr,
    NUM_PROGRAM_CORES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            for r in range(pid, num_rows, NUM_PROGRAM_CORES):
                info = tl.load(src_info_ptr + r)
                if info >= 0:
                    sr = info // NvS
                    offv = info - sr * NvS
                    libshmem_device.putmem(
                        combine_buf_ptr + offv * H,
                        fc2_ptr + r * H,
                        H * 2, sr)
    libshmem_device.barrier_all()


@triton.jit
def moonep_topk_reduce(
    buf_ptr,                # bf16 [N=S·K, H]
    out_ptr,                # bf16 [S, H]
    N: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    S: tl.constexpr = N // K
    offs_h = tl.arange(0, BLOCK_H)
    m_h = offs_h < H
    for t0 in range(0, S, BLOCK_T):
        tok = t0 + tl.arange(0, BLOCK_T)
        m_t = tok < S
        acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)
        for k in range(K):
            row = tok * K + k
            v = tl.load(buf_ptr + row[:, None] * H + offs_h[None, :],
                        mask=m_t[:, None] & m_h[None, :], other=0.0)
            acc += v.to(tl.float32)      # 权重已在 weighted_swiglu 乘过
        tl.store(out_ptr + tok[:, None] * H + offs_h[None, :],
                 acc.to(tl.bfloat16), mask=m_t[:, None] & m_h[None, :])


def launch_moonep_combine(
    ws_combine_buf: torch.Tensor,
    activation: torch.Tensor,          # bf16 [rows_pad, F]（weighted_swiglu 出）
    down: torch.Tensor,                # bf16 [Seg, H, F]（原样传）
    src_info: torch.Tensor,            # int32 [NvS]
    outs: dict,
    *,
    rank: int,
    epn: int,
    E: int,
    B: int,
    NvS: int,
    K: int,
    H: int,
    F: int,
    num_cores: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """FC2 + 推回 + 归约，返回 [S, H] bf16。"""
    from mega_moe.runtime.moonep_routing import build_moonep_segment_meta

    dev = activation.device
    Seg = epn + B
    cu = outs["cu_seqlens"].cpu()
    seg_counts, seg_offsets, _ = build_moonep_segment_meta(
        cu, outs["experts_to_copy"].cpu()[rank], rank, epn, E, B)
    rows_pad = int(seg_offsets[-1])
    assert activation.shape[0] == rows_pad

    fc2_out = torch.empty((rows_pad, H), dtype=torch.bfloat16, device=dev)
    moonep_fc2_gemm[(num_cores, 1, 1)](
        activation, down, fc2_out,
        seg_counts.to(dev), seg_offsets.to(dev),
        H, F,
        activation.stride(0), activation.stride(1),
        down.stride(0), down.stride(1), down.stride(2),
        fc2_out.stride(0), fc2_out.stride(1),
        NUM_PROGRAM_CORES=num_cores, SEG=Seg,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k)

    used = rows_pad                      # src_info 有效域 = [0, rows_pad)
    moonep_combine_push[(num_cores, 1, 1)](
        fc2_out, src_info, ws_combine_buf, used,
        NvS=NvS, H=H, NUM_PROGRAM_CORES=num_cores)

    N = outs["dst"].numel()
    S = N // K
    if output is None:
        output = torch.empty((S, H), dtype=torch.bfloat16, device=dev)
    block_h = max(16, 1 << (H - 1).bit_length())
    moonep_topk_reduce[(1, 1, 1)](
        ws_combine_buf, output, N=N, K=K, H=H,
        BLOCK_T=8, BLOCK_H=block_h)
    return output
