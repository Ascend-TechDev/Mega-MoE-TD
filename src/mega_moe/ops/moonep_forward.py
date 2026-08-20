# coding=utf-8
"""MoonEP 前向算子（负载均衡 dispatch + 副本槽 prefetch + 经典 GEMM 复用）。

组装 M1-M4 的全部部件，接口镜像经典 ``FusedMoEForward`` 的生命周期
（构造 → register_weights → forward* N 次 → finalize）：

    op = MoonepForward(ep_group, max_tokens_per_rank=S, hidden_size=H,
                       ffn_dim=F, top_k=K, num_experts=E, num_slots=B)
    gu, dn = op.register_weights(gate_up_local, down_local)   # 弃用原张量
    out = op.forward(hidden_states, selected_experts, routing_weights)
    op.finalize()

执行序列（每步）：
    planning（宿主 B 表 + C.1/C.2/dedup kernel）→ prefetch（push 式槽预取）
    → zero_fill → dispatch_fc1（融合：putmem 散布 + FC1 GEMM）
    → weighted_swiglu（复用经典）→ FC2 GEMM + src_info 推回 + top-k 归约

v1 约束（见 kernels/moonep_planning.py 头注释）：dropless 有效路由、每 rank
恒定 S token、E 为 2 的幂、GEMM block 下限 128（CASE-12）、R ≤ 64。
"""

from __future__ import annotations

import torch

from mega_moe.kernels.moonep_combine import launch_moonep_combine
from mega_moe.kernels.moonep_dispatch import (
    MoonepDispatchState,
    launch_moonep_dispatch_fc1,
)
from mega_moe.kernels.moonep_planning import (
    MoonepPlanBuffers,
    launch_moonep_planning,
)
from mega_moe.kernels.moonep_prefetch import launch_moonep_prefetch
from mega_moe.kernels.moonep_zero_fill import launch_moonep_zero_fill
from mega_moe.kernels.weighted_swiglu import weighted_swiglu_forward
from mega_moe.runtime.moonep_workspace import (
    MoonepTopology,
    MoonepWorkspace,
)

__all__ = ["MoonepForward"]


class MoonepForward:
    """MoonEP 前向（单 in-flight；plan 输出张量原地复用）。"""

    def __init__(
        self,
        ep_group,
        *,
        max_tokens_per_rank: int,
        hidden_size: int,
        ffn_dim: int,
        top_k: int,
        num_experts: int,
        num_slots: int | None = None,
        token_padding: int = 2,
        num_cores: int = 8,
        block_size: int = 128,
    ):
        import torch.distributed as dist

        self.ep_group = ep_group
        self.rank = dist.get_rank(ep_group)
        self.world_size = dist.get_world_size(ep_group)
        self.num_cores = num_cores
        self.block = block_size
        self._num_slots_resolved = num_slots
        epn = num_experts // self.world_size
        b = num_slots if num_slots is not None else min(epn, 16)
        self.topo = MoonepTopology(
            S=max_tokens_per_rank, K=top_k, E=num_experts,
            R=self.world_size, B=b, H=hidden_size, F=ffn_dim,
            token_padding=token_padding, dispatch_block_m=block_size,
        )
        self._device = f"npu:{self.rank}"
        self.ws = MoonepWorkspace(self.topo, self.rank, self._device)
        self.pbufs = MoonepPlanBuffers(self.topo.N, self._device,
                                       order0=self.ws.order0)
        self.dstate = MoonepDispatchState()
        self._last_etc = None
        t = self.topo
        self._outs = {
            "dst": torch.empty(t.N, dtype=torch.int32, device=self._device),
            "cu_seqlens": torch.empty(t.E + t.B, dtype=torch.int32,
                                      device=self._device),
            "experts_to_copy": torch.empty(t.R, t.B, dtype=torch.int32,
                                           device=self._device),
            "zero_fill_ranges": torch.empty(t.E + t.B, 2, dtype=torch.int32,
                                            device=self._device),
            "remote_stats": torch.empty(2, dtype=torch.int32,
                                        device=self._device),
            "src_info": torch.empty(t.NvS, dtype=torch.int32,
                                    device=self._device),
        }

    # ------------------------------------------------------------------
    def register_weights(self, gate_up_local: torch.Tensor,
                         down_local: torch.Tensor):
        """权重搬入对称表并返回视图（调用方弃用原张量）。"""
        return self.ws.register_weights(gate_up_local, down_local)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,      # bf16 [S, H]
        selected_experts: torch.Tensor,   # int32 [S, K]（有效 id ∈ [0,E)）
        routing_weights: torch.Tensor,    # fp32 [S, K]
    ) -> torch.Tensor:
        t = self.topo
        dev = self._device
        hs = hidden_states.contiguous()
        topk = selected_experts.reshape(-1).to(torch.int32).contiguous()
        rw = routing_weights.reshape(-1).to(torch.float32).contiguous()

        # 1) planning
        launch_moonep_planning(
            self.pbufs, self._outs, topk,
            torch.bincount(topk.to(torch.int64), minlength=t.E)
            .to(torch.int32).to(dev),
            rank=self.rank, world_size=t.R, ep_group=self.ep_group,
            S=t.S, K=t.K, E=t.E, B=t.B, NvS=t.NvS,
            token_padding=t.token_padding)

        # 2) prefetch（首步 _last_etc=None 必执行；etc 未变则全组一致跳过）
        launch_moonep_prefetch(
            self.ws.gate_up, self.ws.down, self._outs["experts_to_copy"],
            rank=self.rank, world_size=t.R, epn=t.epn, H=t.H, F=t.F,
            skip_if_same_as=self._last_etc)
        self._last_etc = self._outs["experts_to_copy"].clone()

        # 3) zero_fill + dispatch_fc1（融合 FC1 GEMM）
        launch_moonep_zero_fill(self.ws.vm, self.ws.routing_weight_recv,
                                self._outs["zero_fill_ranges"], t.H)
        fc1_out, rows_pad, _ep = launch_moonep_dispatch_fc1(
            self.ws.vm, self.ws.routing_weight_recv, self.ws.signal_mem,
            self.ws.gate_up, hs, rw, self._outs, self._outs["_dst_all"],
            rank=self.rank, epn=t.epn, E=t.E, B=t.B, NvS=t.NvS, K=t.K, H=t.H,
            state=self.dstate, num_cores=self.num_cores,
            block_m=self.block, block_n=self.block, block_k=self.block)

        # 4) weighted swiglu（权重在此乘入）
        act = weighted_swiglu_forward(
            fc1_out, self.ws.routing_weight_recv[:rows_pad].contiguous(),
            self.num_cores, activation="swiglu")

        # 5) FC2 + 推回 + 归约
        return launch_moonep_combine(
            self.ws.combine_buf, act, self.ws.down, self._outs["src_info"],
            self._outs,
            rank=self.rank, epn=t.epn, E=t.E, B=t.B, NvS=t.NvS, K=t.K,
            H=t.H, F=t.F, num_cores=self.num_cores,
            block_m=self.block, block_n=self.block, block_k=self.block)

    # ------------------------------------------------------------------
    def sync(self):
        torch.npu.synchronize(self._device)

    def finalize(self):
        torch.npu.synchronize(self._device)
        self.pbufs.finalize()
        self.ws.finalize()
