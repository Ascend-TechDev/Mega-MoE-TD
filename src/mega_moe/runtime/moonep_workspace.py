# coding=utf-8
"""MoonEP 前向的对称内存工作区（MoonepWorkspace）。

分配纪律（沿用仓库硬约束，见 tests/_moe_testkit.py make_peer_mem docstring）：
- **全 rank 按相同顺序、相同形状分配**（形状只由拓扑推出，不看运行期
  路由数据）——保证各 rank 的对称堆偏移逐一对齐；
- ``vm`` 永远是第 1 个分配（``dl.symm_at`` 仅在堆偏移 0 可靠——v1 未用
  symm_at，仍按此纪律留保险）；
- ``finalize`` 按分配逆序释放。

形状公式（MoonEP 语义，planning.py/api.py）：
    NvS = S·K + (token_padding−1)·2·epn        （接收区槽数上界）
    rows_pad_max = ceil(NvS / tp)·tp           （VM 行数，向上对齐）
    Seg = epn + B                              （权重表行数 = 段数）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_COPY_CHUNK_ELEMS = 32 * 1024 * 1024   # register_weights 分块拷贝（bf16 元素数）


@dataclass(frozen=True)
class MoonepTopology:
    """拓扑常量（全部为构造期定值，不看运行期数据）。"""

    S: int                  # 每 rank token 数（恒定，v1 约束）
    K: int                  # top-k
    E: int                  # 全组专家数（2 的幂）
    R: int                  # EP 世界大小
    B: int                  # 副本槽数
    H: int                  # hidden
    F: int                  # ffn 中间维
    token_padding: int = 2
    dispatch_block_m: int = 128

    @property
    def epn(self) -> int:
        assert self.E % self.R == 0
        return self.E // self.R

    @property
    def N(self) -> int:
        return self.S * self.K

    @property
    def NvS(self) -> int:
        return self.N + (self.token_padding - 1) * 2 * self.epn

    @property
    def seg(self) -> int:
        return self.epn + self.B

    @property
    def rows_pad_max(self) -> int:
        tp = self.token_padding
        return math.ceil(self.NvS / tp) * tp

    @property
    def max_src_tiles(self) -> int:
        return math.ceil(self.N / self.dispatch_block_m)


class MoonepWorkspace:
    """按固定顺序持有全部对称张量；宿主侧生命周期管理。

    分配序（1→7，全部 rank 一致）：
        1 vm                 bf16 [rows_pad_max, H]   接收区（VM）
        2 routing_weight_recv fp32 [rows_pad_max]
        3 gate_up            bf16 [Seg, H, 2F]        行 [0,epn)=home，槽在后
        4 down               bf16 [Seg, H, F]         物理 [e, N=H, K=F]
        5 combine_buf        bf16 [N, H]              每 (token,slot) 一行
        6 signal_mem         int32 [R·Seg·max_src_tiles·16]
        7 order0             int32 [N]                planning：rank1→rank0
    """

    def __init__(self, topo: MoonepTopology, rank: int, device):
        import shmem as ash

        self.topo = topo
        self.rank = rank
        self.device = device
        did = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        t = topo
        self._allocs = []                      # (tensor, name) 按序

        def _alloc(shape, dtype, name):
            ten = ash.aclshmem_create_tensor(list(shape), dtype, device_id=did)
            self._allocs.append((ten, name))
            return ten

        self.vm = _alloc((t.rows_pad_max, t.H), torch.bfloat16, "vm")
        self.routing_weight_recv = _alloc((t.rows_pad_max,), torch.float32,
                                          "routing_weight_recv")
        self.gate_up = _alloc((t.seg, t.H, 2 * t.F), torch.bfloat16, "gate_up")
        self.down = _alloc((t.seg, t.H, t.F), torch.bfloat16, "down")
        self.combine_buf = _alloc((t.N, t.H), torch.bfloat16, "combine_buf")
        self.signal_mem = _alloc(
            (t.R * t.seg * t.max_src_tiles * 16,), torch.int32, "signal_mem")
        self.order0 = _alloc((t.N,), torch.int32, "order0")

    # ------------------------------------------------------------------
    @staticmethod
    def required_bytes(topo: MoonepTopology) -> int:
        """估算所需对称堆字节（供 aclshmem_session sizing + benchmark）。"""
        t = topo
        elems_bf16 = (t.rows_pad_max * t.H
                      + t.seg * t.H * 2 * t.F + t.seg * t.H * t.F
                      + t.N * t.H)
        elems_other = t.rows_pad_max * 4 + \
            t.R * t.seg * t.max_src_tiles * 16 * 4 + t.N * 4
        return elems_bf16 * 2 + elems_other

    # ------------------------------------------------------------------
    def register_weights(self, gate_up_local: torch.Tensor,
                         down_local: torch.Tensor):
        """把本 rank 属主权重搬入对称表行 [0, epn)，槽行清零。

        返回对称表视图作为新的权重张量；调用方须弃用原张量（显存峰值
        不可避免 ~1× 单表，拷完 ``empty_cache`` 回收）。
        """
        t = self.topo
        epn = t.epn
        assert tuple(gate_up_local.shape) == (epn, t.H, 2 * t.F), \
            f"gate_up_local 形状应为 ({epn},{t.H},{2 * t.F})"
        assert tuple(down_local.shape) == (epn, t.H, t.F), \
            f"down_local 形状应为 ({epn},{t.H},{t.F})"

        def _copy_chunks(dst, src):
            flat_dst = dst.reshape(-1)
            flat_src = src.reshape(-1)
            n = flat_src.numel()
            for s in range(0, n, _COPY_CHUNK_ELEMS):
                e = min(s + _COPY_CHUNK_ELEMS, n)
                flat_dst[s:e].copy_(flat_src[s:e])

        _copy_chunks(self.gate_up[:epn], gate_up_local)
        _copy_chunks(self.down[:epn], down_local)
        self.gate_up[epn:].zero_()
        self.down[epn:].zero_()
        torch.npu.synchronize(self.device)
        return self.gate_up, self.down

    # ------------------------------------------------------------------
    def finalize(self):
        import shmem as ash

        torch.npu.synchronize(self.device)
        for ten, _name in reversed(self._allocs):
            ash.aclshmem_free_tensor(ten)
        self._allocs.clear()
