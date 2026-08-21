# coding=utf-8
"""MoonEP dispatch：dst 表驱动的 payload/权重散布 + FC1 grouped GEMM（融合）。

结构镜像 ``kernels/dispatch_fc1.py`` 的 all-core 流水：
- Vector 半边（sub_vec 0）：按 (dst, seg) 连续 run 走条目，逐行 putmem
  payload（H bf16）与路由权重（4B），每 BLOCK_M tile 一次
  ``fence + signal_op(SET, epoch)``——信号槽键 ``(src·Seg + seg)·
  MAX_SOURCE_TILES + tile``，与消费侧一致；
- Cube 半边：**原样调用** dispatch_fc1 的
  ``_triton_grouped_gemm_expert_n_merged_tiles_wait``，仅把
  ``EXPERTS_PER_RANK`` 换成 ``Seg = epn + B``——压缩段 id 与 gate_up 表
  行号恒等（见 runtime/moonep_routing.py），kernel 体零改动。

v1 语义：dup 条目（负 dst）解码后**照常发 payload**（数值等价，多耗
带宽；抑制 + 接收端扇出留 v2）。段内 loff 升序 = 段内 source-major
就绪前提（MoonEP gidx 构造天然满足）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .dispatch_fc1 import (
    _triton_grouped_gemm_expert_n_merged_tiles_wait,
    _triton_grouped_gemm_one_mn_tile_tail,
)

__all__ = ["MoonepDispatchState", "launch_moonep_dispatch_fc1",
           "moonep_dispatch_fc1"]


class MoonepDispatchState:
    """跨步持有的 dispatch 状态（信号 epoch 单调递增，slot 不清零）。"""

    def __init__(self):
        self.epoch = 1


@triton.jit
def _moonep_push_runs(
    pid, ncore: tl.constexpr,
    input_ptr, vm_ptr, routing_weight_ptr, rw_recv_ptr, signal_mem_ptr,
    send_src_idx_ptr, send_offv_ptr, send_loff_ptr,
    run_dst_ptr, run_seg_ptr, run_start_ptr, run_count_ptr, num_runs,
    signal_epoch, hidden, stride_input_m,
    LOCAL_RANK: tl.constexpr, SEG: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
):
    for run in range(pid, num_runs, ncore):
        dst = tl.load(run_dst_ptr + run)
        seg = tl.load(run_seg_ptr + run)
        st = tl.load(run_start_ptr + run)
        cnt = tl.load(run_count_ptr + run)
        num_tiles = tl.cdiv(cnt, BLOCK_M)
        for t in range(0, num_tiles):
            j0 = t * BLOCK_M
            j1 = tl.minimum(j0 + BLOCK_M, cnt)
            for j in range(j0, j1):
                idx = st + j
                src_tok = tl.load(send_src_idx_ptr + idx)
                offv = tl.load(send_offv_ptr + idx)
                loff = tl.load(send_loff_ptr + idx)
                # 长度参数为字节数（hidden*2 / 4），dispatch_fc1.py 同款
                libshmem_device.putmem(
                    vm_ptr + loff * hidden,
                    input_ptr + src_tok * stride_input_m,
                    hidden * 2, dst)
                libshmem_device.putmem(
                    rw_recv_ptr + loff,
                    routing_weight_ptr + offv,
                    4, dst)
            libshmem_device.fence()
            signal_slot = (LOCAL_RANK * SEG + seg) * MAX_SOURCE_TILES + t
            libshmem_device.signal_op(
                signal_mem_ptr + signal_slot * 16,
                signal_epoch,
                libshmem_device.ACLSHMEM_SIGNAL_SET,
                dst,
            )


@triton.jit
def moonep_dispatch_push(
    input_ptr, vm_ptr, routing_weight_ptr, rw_recv_ptr,
    send_src_idx_ptr, send_offv_ptr, send_loff_ptr,
    run_dst_ptr, run_seg_ptr, run_start_ptr, run_count_ptr, num_runs,
    hidden, stride_input_m,
    NUM_PROGRAM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PUSH_RW: tl.constexpr,
):
    """两段式第一步：全部 run 的 payload/权重 putmem + fence + 全组 barrier。

    CASE-15：融合形态（同 kernel 内 push∥GEMM+信号等待）对 **dst==self 的
    行**出现非确定性脏读（坏行稳定落在自源块；fence+signal 对自 putmem 的
    排序在 cube 侧读取时偶发失效——classic 同构却生产无恙，根因未明）。
    v1 采用与 moonep_combine 同款「push→barrier→无信号 GEMM」两段式，
    多轮验证零翻车；融合流水（信号协议）留 M5 复原。

    ``PUSH_RW=0``：跳过权重推送（反向 B-1 的 dy 散布复用本 kernel——
    dy 逐行推到 dy_recv，布局与 dispatch payload 完全同构）。
    """
    pid = tl.program_id(axis=0)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            for run in range(pid, num_runs, NUM_PROGRAM_CORES):
                dst = tl.load(run_dst_ptr + run)
                st = tl.load(run_start_ptr + run)
                cnt = tl.load(run_count_ptr + run)
                for j in range(0, cnt):
                    idx = st + j
                    src_tok = tl.load(send_src_idx_ptr + idx)
                    offv = tl.load(send_offv_ptr + idx)
                    loff = tl.load(send_loff_ptr + idx)
                    libshmem_device.putmem(
                        vm_ptr + loff * hidden,
                        input_ptr + src_tok * stride_input_m,
                        hidden * 2, dst)
                    if PUSH_RW:
                        libshmem_device.putmem(
                            rw_recv_ptr + loff,
                            routing_weight_ptr + offv,
                            4, dst)
            libshmem_device.fence()
    libshmem_device.barrier_all()


@triton.jit
def moonep_fc1_gemm(
    vm_ptr, weight_ptr, output_ptr,
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
    """两段式第二步：无信号 FC1 段 GEMM（数据经 barrier 全就绪）。"""
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
                    vm_ptr, weight_ptr, output_ptr,
                    seg, seg_off + w_start, w_size, n_tile, N, K,
                    stride_input_m, stride_input_k,
                    stride_weight_0, stride_weight_1, stride_weight_2,
                    stride_output_m, stride_output_n,
                    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, dtype)


@triton.jit
def moonep_dispatch_fc1(
    # 输入 / 对称缓冲
    input_ptr, vm_ptr, routing_weight_ptr, rw_recv_ptr, signal_mem_ptr,
    weight_ptr,            # gate_up.transpose(-1,-2) 视图（逻辑 [Seg,2F,H]）
    output_ptr,            # bf16 [rows_pad, 2F]
    # 发送表 / 段表
    send_src_idx_ptr, send_offv_ptr, send_loff_ptr,
    run_dst_ptr, run_seg_ptr, run_start_ptr, run_count_ptr, num_runs,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    signal_epoch,
    # 维度
    hidden: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # 步长
    stride_input_m, stride_input_k,
    stride_weight_0, stride_weight_1, stride_weight_2,
    stride_output_m, stride_output_n,
    # 元参
    NUM_PROGRAM_CORES: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SEG: tl.constexpr,
    MAX_SOURCE_TILES: tl.constexpr,
    FINAL_BARRIER: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    dtype = tl.bfloat16
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _moonep_push_runs(
                pid, NUM_PROGRAM_CORES,
                input_ptr, vm_ptr, routing_weight_ptr, rw_recv_ptr,
                signal_mem_ptr,
                send_src_idx_ptr, send_offv_ptr, send_loff_ptr,
                run_dst_ptr, run_seg_ptr, run_start_ptr, run_count_ptr,
                num_runs, signal_epoch, hidden, stride_input_m,
                LOCAL_RANK, SEG, MAX_SOURCE_TILES, BLOCK_SIZE_M)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _triton_grouped_gemm_expert_n_merged_tiles_wait(
            pid, NUM_PROGRAM_CORES,
            vm_ptr, signal_mem_ptr, weight_ptr, output_ptr,
            recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
            signal_epoch,
            N, K, stride_input_m, stride_input_k,
            stride_weight_0, stride_weight_1, stride_weight_2,
            stride_output_m, stride_output_n,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            WORLD_SIZE, SEG, MAX_SOURCE_TILES, dtype)
    if FINAL_BARRIER:
        libshmem_device.barrier_all()


def launch_moonep_dispatch_fc1(
    vm: torch.Tensor,
    rw_recv: torch.Tensor,
    signal_mem: torch.Tensor,
    gate_up: torch.Tensor,
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    outs: dict,
    dst_all: torch.Tensor,        # int64/cpu [R,N] 全组 dst（宿主已收集）
    *,
    rank: int,
    epn: int,
    E: int,
    B: int,
    NvS: int,
    K: int,
    H: int,
    state: MoonepDispatchState,
    num_cores: int,
    block_m: int = 16,
    block_n: int = 16,
    block_k: int = 16,
    fc1_output: torch.Tensor | None = None,
    final_barrier: bool = True,
):
    """返回 (fc1_output, rows_pad, used_epoch)。原地消费 planning outs。

    前置：remote_stats[0] ≤ B（副本槽全覆盖——压缩段 id 与权重行恒等的
    前提；违反即 RuntimeError，报出实际需要的槽数）。
    """
    from mega_moe.runtime.moonep_routing import (
        build_moonep_send_meta,
        build_moonep_segment_meta,
        build_recv_counts,
    )

    remote_n = int(outs["remote_stats"][0].item())
    if remote_n > B:
        raise RuntimeError(
            f"moonep dispatch: rank{rank} 远程专家数 {remote_n} > 副本槽数 "
            f"B={B}——未进槽的远程专家无本地权重。请调大 MoonepConfig 的 "
            f"num_slots（当前路由至少需要 {remote_n}）。")

    R = dst_all.shape[0]
    Seg = epn + B
    N = dst_all.shape[1]
    dev = vm.device
    tbl = outs["_tbl"]                      # planning 捎带的全组宿主表
    cu_all = tbl["cu_all"].to(torch.int64)
    cu = cu_all[rank]
    seg_counts, seg_offsets, _seg_expert = build_moonep_segment_meta(
        cu, outs["experts_to_copy"].cpu()[rank], rank, epn, E, B)
    send = build_moonep_send_meta(outs["dst"].cpu(), cu_all, K, epn, E, B,
                                  NvS)
    recv_counts = build_recv_counts(dst_all, cu, rank, epn, E, B, NvS)
    rows_pad = int(seg_offsets[-1])

    num_runs = send["run_dst"].numel()
    weight_for_gemm = gate_up.transpose(-1, -2)      # 逻辑 [Seg, 2F, H] 视图
    _, out_size, red_size = weight_for_gemm.shape
    assert out_size == gate_up.shape[2] and red_size == H, \
            f"gate_up 形状 {tuple(gate_up.shape)} 与 H={H} 不符"
    if fc1_output is None:
        fc1_output = torch.empty((rows_pad, out_size), dtype=torch.bfloat16,
                                 device=dev)

    d = lambda t: t.to(dev)
    max_src_tiles = (N + block_m - 1) // block_m
    epoch = state.epoch
    # ---- 两段式（CASE-15）：push+barrier → 无信号 GEMM ----
    moonep_dispatch_push[(num_cores, 1, 1)](
        hidden_states, vm,
        routing_weights.reshape(-1).contiguous(), rw_recv,
        d(send["send_src_idx"]), d(send["send_offv"]), d(send["send_loff"]),
        d(send["run_dst"]), d(send["run_seg"]), d(send["run_start"]),
        d(send["run_count"]), num_runs,
        H, hidden_states.stride(0),
        NUM_PROGRAM_CORES=num_cores, BLOCK_M=block_m, PUSH_RW=1,
    )
    moonep_fc1_gemm[(num_cores, 1, 1)](
        vm, weight_for_gemm, fc1_output,
        d(seg_counts), d(seg_offsets),
        out_size, red_size,
        H, 1,
        weight_for_gemm.stride(0), weight_for_gemm.stride(1),
        weight_for_gemm.stride(2),
        fc1_output.stride(0), fc1_output.stride(1),
        NUM_PROGRAM_CORES=num_cores, SEG=Seg,
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
    )
    state.epoch += 1
    return fc1_output, rows_pad, epoch
