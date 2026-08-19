# coding=utf-8
"""MoonEP DispatchEpilogueKernel + launch_dispatch_epilogue 参考实现（torch 语义级）。

语义基准：GPU 版 MoonEP（source_code/MoonEP/moonep/dispatch_epilogue.py）的
DispatchEpilogueKernel 与 launch_dispatch_epilogue。按绑定级设计契约
（docs/design.md §4）：类名、``__init__`` 常量配置、``__call__`` 入参/出参、
以及 ``launch_dispatch_epilogue`` 宿主启动函数签名与 GPU 源码逐字一致，仅把
kernel 体（G2S producer / S2G consumer 双 warp TMA 流水）替换为纯 torch 实现
（CPU 可跑）；``launch_dispatch_epilogue`` 内部构造本包 DispatchEpilogueKernel
并调用其 torch 体。性能优化（AscendC kernel 化）是后续工作。

语义（对齐 GPU kernel，dispatch_epilogue.py:168-305）：dispatch 完成后在本
rank NVL shard 上原地做 dup 扇出——对每个 dup 组，把 primary 行（dispatch 唯一
真正跨 rank 发送的副本）复制到该组全部 dup 行（dup_loffs 紧凑表，组内 loff
升序（GPU ctz 发射序）；primary 行只读一次，无论扇出多少份）。dup 行与 primary 行互不重叠、各组
之间亦无重叠（planning 保证），逐组处理无先后依赖。

纯本地操作、无跨 rank 通信：须在 DispatchKernel（含其出口 arena.barrier()，
发布远端 NVL 写）之后调用；其输出由本地 GEMM 消费、或由 combine 的入口屏障
发布，故本 kernel 自身不带屏障。padding 行已由 DispatchKernel 的 zero 路径
清零，per-topk 权重已由其权重路径散布，均不在本 kernel 职责内。

契约 §0 的签名扩展与留参（GPU 专属形参保留不使用）：

- ``__init__`` 追加 ``arena`` 关键字参数（全包 kernel 统一的注入点；本 kernel
  纯本地，arena 留存不消费）。
- num_sms/smem_budget/pdl_launch：GPU 的 grid 规模、smem 流水几何与 PDL 启动
  开关，无语义作用；GPU 的 _pick_geometry/_smem_bytes 推导（含 stages==0 的
  报错与 B<=32 的 warp 约束）不复制。
- stream：cuda.CUstream 形参保留（接受 None，不使用）。

launch_dispatch_epilogue（签名逐字对齐 GPU dispatch_epilogue.py:367）与 GPU
host 侧的偏差（契约 §0 许可）：CUDA 专属断言（is_cuda / H%8 对齐 /
device_index）不保留——参考实现 CPU 可跑；grid 规模取
ctx['num_sms_dedup']（GPU 键），缺省回退 ctx['num_sms']（torch 体不消费，
仅作 kernel 常量透传）；arena 取自 ctx['arena']（契约 §2 扩展键）。

dtype 契约：payload bf16；dedup 表 int32。语义精确优先，不做性能技巧。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .planning import MoonEPCommPlan

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["DispatchEpilogueKernel", "launch_dispatch_epilogue"]

# 参考实现的 smem 预算取值（与 api.py / prefetch.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 kernel 构造契约而透传。
_SMEM_BUDGET_BYTES = 231424


class DispatchEpilogueKernel:
    """本 rank NVL shard 上的原地 dup 组扇出（类名/配置/签名与 GPU 源码逐字一致）。

    由 plan 持有的 dedup 表驱动全部工作：dup_groups 给出
    (primary_loff, dup_start, dup_count) 组头，dup_loffs 给出紧凑平坦的 dup
    槽位，dup_counts[0] 给出 dup_groups 的有效前缀长度。
    """

    def __init__(
        self,
        H: int,
        NvS: int,
        num_sms: int,
        smem_budget: int,
        pdl_launch: bool,
        *,
        arena: "SymmetricArena",
    ):
        # 常量配置与 GPU 源码 dispatch_epilogue.py:73 起逐字一致；arena 为契约
        # §0 的唯一签名扩展点（本 kernel 纯本地，不消费）。
        self.H = H
        self.NvS = NvS
        # GPU 专属配置：num_sms/smem_budget/pdl_launch 为 grid 规模、smem 流水
        # 几何（B/stages 推导）与 PDL 启动开关；参考实现无对应概念，留存参不消费。
        self.num_sms = num_sms
        self.smem_budget = smem_budget
        self.pdl_launch = pdl_launch
        self.arena = arena

    # ------------------------------------------------------------------ call

    def __call__(
        self,
        hidden_buf_local_ptr: torch.Tensor,   # bf16 [NvS, H] 本 rank NVL shard
        dup_groups_ptr: torch.Tensor,         # int32 [NvS, 3]
        dup_loffs_ptr: torch.Tensor,          # int32 [NvS,]
        dup_counts_ptr: torch.Tensor,         # int32 [2,]
        stream,                               # 保留不使用：cuda.CUstream 形参位（接受 None）
    ):
        """执行一次 dup 扇出（同步语义，纯本地）。

        Args:
            hidden_buf_local_ptr: [NvS, H] bf16 本 rank shard（原地读写）。
            dup_groups_ptr: [NvS, 3] int32，前 dup_counts[0] 行有效，每行
                (primary_loff, dup_start, dup_count)。
            dup_loffs_ptr: [NvS] int32，前 dup_counts[1] 项有效，组内 kidx 升序。
            dup_counts_ptr: [2] int32，[n_groups, n_dup_loffs]。
            stream: GPU 专属形参，保留不使用（接受 None）。

        Returns:
            None（全部结果经 hidden_buf_local_ptr 原地写出）。
        """
        H, NvS = self.H, self.NvS
        assert hidden_buf_local_ptr.dtype == torch.bfloat16 and \
            hidden_buf_local_ptr.is_contiguous(), \
            "hidden_buf_local_ptr 必须是连续 bf16"
        assert tuple(hidden_buf_local_ptr.shape) == (NvS, H), \
            f"hidden_buf_local_ptr 形状应为 ({NvS}, {H})，" \
            f"got {tuple(hidden_buf_local_ptr.shape)}"
        assert dup_groups_ptr.dtype == torch.int32 and \
            tuple(dup_groups_ptr.shape) == (NvS, 3), \
            f"dup_groups_ptr 必须是 ({NvS}, 3) int32"
        assert dup_loffs_ptr.dtype == torch.int32 and \
            tuple(dup_loffs_ptr.shape) == (NvS,), \
            f"dup_loffs_ptr 必须是 ({NvS},) int32"
        assert dup_counts_ptr.dtype == torch.int32 and \
            tuple(dup_counts_ptr.shape) == (2,), \
            "dup_counts_ptr 必须是 (2,) int32"

        # dup_groups 有效前缀长度（GPU 在 device 侧读；参考实现同步取值）
        n_groups = int(dup_counts_ptr[0])
        for g in range(n_groups):
            primary_loff = int(dup_groups_ptr[g, 0])
            dup_start = int(dup_groups_ptr[g, 1])
            dup_count = int(dup_groups_ptr[g, 2])
            if dup_count <= 0:
                continue
            # 组内 dup 槽（kidx 升序，建表时已钉死）
            dups = dup_loffs_ptr[dup_start:dup_start + dup_count].to(torch.int64)
            # primary 行只读一次；clone 杜绝写回别名后再扇出到全部 dup 行
            src = hidden_buf_local_ptr[primary_loff].clone()
            hidden_buf_local_ptr[dups] = src.unsqueeze(0).expand(dup_count, -1)


# ============================================================================
# Host launcher（torch 语义级参考实现；签名与 GPU dispatch_epilogue.py:367 逐字一致）
# ============================================================================

def _check_epilogue_plan(ctx: dict, plan: MoonEPCommPlan) -> None:
    """plan 与 ctx 的一致性校验（对齐 GPU dispatch_epilogue.py:317-336 的语义子集）。

    plan 各张量字段的形状/dtype/连续性已由 MoonEPCommPlan.__post_init__ 钉死；
    此处复核 plan.NvS 与 ctx 一致。GPU 版的 CUDA 设备断言不保留——参考实现
    CPU 可跑。
    """
    NvS = int(ctx["NvS"])
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"


def launch_dispatch_epilogue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_launch: bool = False,
):
    """Expand duplicate rows in place on the local NVL shard（torch 语义级参考实现）。

    签名与 GPU 源码 dispatch_epilogue.py:367 逐字一致；内部按 ctx 常量构造本包
    DispatchEpilogueKernel 并调用其 torch 体（契约 §4：launch_* 内部调用本包
    torch kernel 实现）。须在 launch_dispatch（含其出口 arena.barrier()，发布
    远端 NVL 写）之后调用；消费 fresh dispatch builder 物化的 plan 持有 dedup
    三表（reuse 路径原样传入已保存的表）。纯本地——无跨 rank 通信。

    Args:
        ctx: 通信上下文（契约 §2 键集）；本函数取用 H / NvS / num_sms_dedup
            （缺省回退 num_sms）/ hidden_buf_local（缺省回退 hidden_buf）/
            arena。
        plan: 通信规划（MoonEPCommPlan），携带 dedup 三表。
        pdl_launch: GPU PDL 开关；签名保留，参考实现无语义作用（仅作 kernel
            常量透传）。

    Returns:
        None（全部结果经 hidden_buf_local 原地写出）。
    """
    assert isinstance(plan, MoonEPCommPlan)
    H = int(ctx["H"])
    NvS = int(ctx["NvS"])
    # GPU 取 ctx['num_sms_dedup']（dedup 专用 grid 规模）；torch 体不消费
    # num_sms，仅为保持 kernel 构造契约而透传，键缺省时回退 ctx['num_sms']。
    num_sms = int(ctx.get("num_sms_dedup", ctx["num_sms"]))

    hidden_buf_local = ctx.get("hidden_buf_local")
    if hidden_buf_local is None:
        # 回退现行 api.py 键：参考实现的 hidden_buf 即本 rank 分片
        hidden_buf_local = ctx.get("hidden_buf")
    assert isinstance(hidden_buf_local, torch.Tensor), (
        "ctx 缺少 'hidden_buf_local'（本 rank [NvS,H] bf16 对称分片，契约 §2 键）"
    )
    assert hidden_buf_local.dtype == torch.bfloat16 and \
        hidden_buf_local.is_contiguous(), \
        "hidden_buf_local must be contiguous bf16"
    assert tuple(hidden_buf_local.shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(hidden_buf_local.shape)}"
    # 注：GPU 的 is_cuda / H%8 对齐 / device_index 断言为 CUDA 专属，不保留。
    _check_epilogue_plan(ctx, plan)

    arena = ctx.get("arena")
    assert arena is not None, "ctx 缺少 'arena' 键（契约 §2 参考实现扩展键）"

    kernel = DispatchEpilogueKernel(
        H=H,
        NvS=NvS,
        num_sms=num_sms,
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        pdl_launch=bool(pdl_launch),
        arena=arena,
    )
    kernel(
        hidden_buf_local,                     # hidden_buf_local_ptr
        plan.dup_groups,                      # dup_groups_ptr
        plan.dup_loffs,                       # dup_loffs_ptr
        plan.dup_counts,                      # dup_counts_ptr
        None,                                 # stream 留参（参考实现同步执行）
    )
