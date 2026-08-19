# coding=utf-8
"""MoonEP 权重槽预取（push 式，owner 侧发起）。

与参考实现（mega_moe/moonep_ref/moonep/prefetch.py，pull 式 pull_into）
语义等价：把属主 rank 的专家权重行整块搬进 consumer rank 的本地预取槽行
（``gate_up/down`` 对称表的行 [epn, epn+B)）。实现取 push 方向——repo 只有
putmem 惯例（无 getmem 用例）；etc 表全组同表（planning 产物），每个
owner 自行筛出"哪些 consumer 的槽选了我的专家"。

bit-exact 语义：槽行 b 内容 == 属主的 home 行（e − owner·epn）逐位相等
（bf16 位壳搬运）；空槽（etc == −1）不拉不写，本地槽行保持原值。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id

__all__ = ["launch_moonep_prefetch", "moonep_prefetch_push"]


@triton.jit
def moonep_prefetch_push(
    gate_up_ptr,             # bf16 [Seg, H, 2F]（对称表）
    down_ptr,                # bf16 [Seg, H, F]（对称表）
    elist_ptr,               # int32 [M] 属主本地专家行号 e_local
    blist_ptr,               # int32 [M] consumer 槽号 b
    pelist_ptr,              # int32 [M] 目标 rank（consumer）
    M,                       # runtime：本 rank 的推送条数
    EPN: tl.constexpr,
    H: tl.constexpr,
    F: tl.constexpr,
):
    gu_row = H * (2 * F)     # gate_up 行元素数
    dn_row = H * F
    if sub_vec_id() == 0:
        for i in range(0, M):
            e = tl.load(elist_ptr + i)
            b = tl.load(blist_ptr + i)
            pe = tl.load(pelist_ptr + i)
            # 同 offset 语义：dst 用"目标偏移的本地指针表示"；长度为字节数
            libshmem_device.putmem(
                gate_up_ptr + (EPN + b) * gu_row,
                gate_up_ptr + e * gu_row,
                gu_row * 2, pe)
            libshmem_device.putmem(
                down_ptr + (EPN + b) * dn_row,
                down_ptr + e * dn_row,
                dn_row * 2, pe)
    libshmem_device.barrier_all_vec()   # 槽行对全组可见


def launch_moonep_prefetch(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    experts_to_copy: torch.Tensor,      # int32 [R, B] 全组槽表（planning 产物）
    *,
    rank: int,
    world_size: int,
    epn: int,
    H: int,
    F: int,
    skip_if_same_as: torch.Tensor | None = None,
) -> bool:
    """推送本 rank 属主权重到各 consumer 槽行。返回是否实际执行。

    ``skip_if_same_as``：上一轮 etc（同 device 张量）；与当前一致时跳过
    （推理/路由稳定场景省重复搬运）。**契约**：kernel 尾部 barrier 是全组
    集合——跳过决策必须在所有 rank 上一致（etc 全组同表、上一轮 etc 也
    全组同表即天然满足），否则部分 rank 跳过会导致 barrier 悬挂。
    """
    if skip_if_same_as is not None and \
            skip_if_same_as.shape == experts_to_copy.shape and \
            torch.equal(skip_if_same_as, experts_to_copy):
        # 内容未变也要保证全组集合形态一致：所有 rank 一致跳过即可
        return False

    etc_cpu = experts_to_copy.detach().cpu()
    B = etc_cpu.shape[1]
    e_l, b_l, pe_l = [], [], []
    for c in range(world_size):
        for b in range(B):
            e = int(etc_cpu[c, b])
            if e >= 0 and e // epn == rank:
                e_l.append(e - rank * epn)
                b_l.append(b)
                pe_l.append(c)
    dev = gate_up.device
    elist = torch.tensor(e_l, dtype=torch.int32, device=dev)
    blist = torch.tensor(b_l, dtype=torch.int32, device=dev)
    pelist = torch.tensor(pe_l, dtype=torch.int32, device=dev)
    moonep_prefetch_push[(1, 1, 1)](
        gate_up, down, elist, blist, pelist, elist.numel(),
        EPN=epn, H=H, F=F,
    )
    return True
