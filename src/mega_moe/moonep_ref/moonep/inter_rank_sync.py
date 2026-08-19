# coding=utf-8
"""moonep 跨 rank 同步原语（对齐 GPU 源码 moonep/inter_rank_sync.py 的绑定契约）。

对应设计契约 docs/design.md §3 inter_rank_sync.py：

- ``cross_rank_barrier(arena)``        全组屏障（GPU 版 3 槽自复位协议 →
                                       参考实现退化为全组屏障）；
- ``inter_rank_sync(arena)``           通信前的对齐点（与 cross_rank_barrier
                                       同语义，保留独立封装位）；
- ``launch_inter_rank_sync(ctx)``      签名对齐 GPU 源码 inter_rank_sync.py:119
                                       ``launch_inter_rank_sync(ctx: dict) -> None``；
                                       内部调 ``cross_rank_barrier(ctx['arena'])``。

GPU 版背景：这两个同步点由对称内存 meta_buf BARRIER 区上的 **3 槽自复位协议**
（3-slot self-resetting handshake，仿 DeepGEMM ``nvlink_barrier``：每 rank 复用
3 个 int32 槽位，``+0/+1`` 为两个相位信号、``+2`` 为本地相位/符号状态，单个原子
操作交替相位后自复位，全零初态即正确）实现，用以避免每次同步都走重量级屏障；
``launch_inter_rank_sync`` 对应单 CTA 内核 ``InterRankSyncKernel``（planning
启动前对齐各 EP rank，削弱 CPU/上游 stream 偏斜对计时的影响）。

本 torch 语义级参考实现不模拟槽位协议，三者均退化为全组屏障
（``arena.barrier()``）：语义等价于"全组到齐后，此前发起的 push_all/pull_all
数据移动对本 rank 可见"，只是不具备 GPU 版的低开销特性。后续 AscendC 化时
可在同一调用点换回轻量协议。
"""

import torch

from .buffer import SymmetricArena

__all__ = ["cross_rank_barrier", "inter_rank_sync", "launch_inter_rank_sync"]


def cross_rank_barrier(arena: SymmetricArena) -> None:
    """全组屏障（transport barrier 的别名封装）。

    语义：调用返回后，全组所有 rank 都已到达本同步点，且它们在此之前发起的
    push_all/pull_all 数据移动对本 rank 可见。
    GPU 版对应 meta_buf BARRIER 区 3 槽自复位协议的一次全组握手；
    参考实现退化为全组屏障。
    """
    arena.barrier()


def inter_rank_sync(arena: SymmetricArena) -> None:
    """通信前的对齐点（同 cross_rank_barrier 语义，独立封装位）。

    用于 dispatch/combine 等集合操作之前，保证全组都已进入同一 API 阶段
    （例如 pull 远端数据前，先确认全组的 push 均已完成）。
    GPU 版对应 inter_rank_sync.py 的单 CTA 同步内核（内部同样走 3 槽
    自复位协议）；参考实现退化为全组屏障。
    与 cross_rank_barrier 拆成两个函数，仅为保留 GPU 版的调用点语义位，
    便于后续 AscendC 化时分别替换实现。
    """
    arena.barrier()


def launch_inter_rank_sync(ctx: dict) -> None:
    """planning 启动前的跨 rank 对齐（签名对齐 GPU 源码 inter_rank_sync.py:119）。

    GPU 版：编译并启动单 CTA 的 ``InterRankSyncKernel``，在 meta_buf BARRIER 区
    上完成全组握手；用到 ``ctx['R']`` / ``ctx['meta_chunk_padded']`` /
    ``ctx['grid_sync_bar']`` / ``ctx['BARRIER_OFF']`` / ``ctx['rank']`` 与当前
    CUDA stream。

    参考实现：不模拟槽位协议与 kernel 启动，保留 GPU 版的入口断言
    （meta_buf 为连续 int32）后，内部调 ``cross_rank_barrier(ctx['arena'])``
    （退化为全组屏障）；上述 GPU 专属 ctx 键保留不使用。
    """
    assert ctx['meta_buf'].dtype == torch.int32 and ctx['meta_buf'].is_contiguous()
    cross_rank_barrier(ctx['arena'])
