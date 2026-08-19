# coding=utf-8
"""MoonEP CombinePrologueKernel 参考实现（torch 语义级）。

语义基准：GPU 版 MoonEP（source_code/MoonEP/moonep/combine_prologue.py）的
CombinePrologueKernel 与 launch_combine_prologue。按绑定级设计契约
（docs/design.md §4）：类名、``__init__`` 常量配置、``__call__`` 入参/出参、
以及 ``launch_combine_prologue`` 宿主启动函数签名与 GPU 源码逐字一致，仅把
kernel 体（G2S producer TMA 流水 + fp32 ACC consumers）替换为纯 torch 实现
（CPU 可跑）。性能优化（AscendC kernel 化）是后续工作。

语义（对齐 GPU kernel，combine_prologue.py:358-482）：combine 之前在本 rank
NVL shard 上原地做 **dup 归并**——对每个 dup 组，把 primary 行与其全部 dup 行
以 fp32 无权求和（primary 先行，dup 行按 dup_loffs 表序，组内 kidx 升序（GPU ctz 发射序），
求和顺序确定可复现），结果以 bf16 写回 primary 行。dispatch_epilogue 把
primary 行扇出到 dup 行，本 kernel 是 combine 方向上的反向归约。
**不读 WEIGHTS 区、不加权**——dup 归并与路由权重无关。

纯本地操作、无跨 rank 通信与屏障：须在 combine 输入（专家 FFN 输出）staging
进本 shard 之后、CombineKernel 之前调用；其写回由 CombineKernel 入口的
arena.barrier() 向全组发布。

契约 §0 的签名扩展与留参（GPU 专属形参保留不使用）：

- ``__init__`` 追加 ``arena`` 关键字参数（全包 kernel 统一的注入点；本 kernel
  纯本地，arena 留存不消费）。
- num_sms/smem_budget/pdl_trigger：GPU 的 grid 规模、smem 流水几何与 PDL 触发
  开关，无语义作用；GPU 的 _pick_geometry/_smem_bytes 推导（含 stages_acc==0
  的报错与 H % ACC_THREADS 的 warp 布局断言）不复制。
- stream：cuda.CUstream 形参保留（接受 None，不使用）。
- launch 侧 GPU 专属断言不保留：H % ACC_THREADS == 0 与 16B bulk-copy 对齐、
  CUDA device、plan.K <= 32（lane shuffle 宽度）均为 GPU kernel 实现约束，
  torch 体无对应概念（与 prefetch.py 的 launch_prefetch 先例一致）。

dtype 契约：payload bf16；dedup 表 int32。语义精确优先，不做性能技巧。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .planning import MoonEPCommPlan

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["CombinePrologueKernel", "launch_combine_prologue"]

# 参考实现的 smem 预算取值（与 api.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 __init__ 的构造契约。
_SMEM_BUDGET_BYTES = 231424


class CombinePrologueKernel:
    """本 rank NVL shard 上的 dup 组 fp32 无权归并（类名/配置/签名与 GPU 源码
    逐字一致）。

    由 plan 持有的 dedup 表驱动全部工作：dup_groups 给出
    (primary_loff, dup_start, dup_count) 组头，dup_loffs 给出紧凑平坦的 dup
    槽位，dup_counts[0] 给出 dup_groups 的有效前缀长度。
    """

    def __init__(
        self,
        H: int,
        R: int,
        NvS: int,
        num_sms: int,
        smem_budget: int,
        pdl_trigger: bool,
        *,
        arena: "SymmetricArena",
    ):
        # 常量配置与 GPU 源码 combine_prologue.py:98 起逐字一致；arena 为契约
        # §0 的唯一签名扩展点（本 kernel 纯本地，不消费）。
        self.H = H
        self.R = R
        self.NvS = NvS
        # GPU 专属配置：num_sms/smem_budget/pdl_trigger 为 grid 规模、smem 流水
        # 几何（B/stages_acc/VEC 推导）与 PDL 触发开关；参考实现无对应概念，
        # 留存参不消费。
        self.num_sms = num_sms
        self.smem_budget = smem_budget
        self.pdl_trigger = pdl_trigger
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
        """执行一次 dup 归并（同步语义，纯本地）。

        每组：``acc = fp32(primary 行)``，再按 dup_loffs 表序（组内 kidx 升序）
        逐个累加 ``fp32(dup 行)``，最后以 bf16 写回 primary 行（shard 为 bf16，
        与 GPU epilogue 的 fp32→bf16 存储一致）。``.to(fp32)`` 产出新张量，
        ``add_`` 不会改动 shard 上的 dup 原行；primary 行的写回发生在本组全部
        行读完之后，dup 行只读不写，无读写别名。

        Args:
            hidden_buf_local_ptr: [NvS, H] bf16 本 rank shard（原地读写）。
            dup_groups_ptr: [NvS, 3] int32，前 dup_counts[0] 行有效，每行
                (primary_loff, dup_start, dup_count)。
            dup_loffs_ptr: [NvS] int32，紧凑平坦的 dup 槽表（组内 kidx 升序）。
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
            # fp32 寄存器式累加：primary 行先行入 acc，再按 kidx 升序（物化表序）加 dup 行
            acc = hidden_buf_local_ptr[primary_loff].to(torch.float32)
            for j in range(dup_count):
                loff = int(dup_loffs_ptr[dup_start + j])
                acc.add_(hidden_buf_local_ptr[loff].to(torch.float32))
            # fp32 → bf16 写回 primary 行
            hidden_buf_local_ptr[primary_loff].copy_(acc.to(torch.bfloat16))


# ============================================================================
# 宿主启动函数（签名与 GPU 源码 combine_prologue.py:545 逐字一致）
# ============================================================================

def launch_combine_prologue(
    ctx: dict,
    plan: MoonEPCommPlan,
    *,
    pdl_trigger: bool = False,
):
    """Accumulate duplicate rows into their primary row on the NVL shard.

    Must run after the combine input has been staged into
    ``ctx['hidden_buf_local']`` and before ``launch_combine`` on the same
    stream. Consumes the plan-owned ``dup_groups`` / ``dup_loffs`` /
    ``dup_counts`` from a fresh dispatch builder (reuse paths pass the saved
    tensors unchanged).

    参考实现说明：内部取 ``ctx['arena']`` 构造本包 CombinePrologueKernel 并
    调用其 torch 体（契约 §4）；GPU 侧的每 (H,R,NvS,num_sms,pdl_trigger)
    编译缓存（_get_compiled）在 torch 体下无编译概念，kernel 实例仅为常量
    配置持有者，每次调用新建即可。GPU 专属断言（H % ACC_THREADS、CUDA
    device、plan.K <= 32 的 lane shuffle 宽度）为 kernel 实现约束，torch 体
    无对应概念，不保留。
    """
    H = int(ctx['H'])
    R = int(ctx['R'])
    NvS = int(ctx['NvS'])
    # GPU 版取 ctx['num_sms_dedup']；参考实现 ctx 不区分 dedup 专用 grid 规模，
    # 优先取该键、缺省回退 ctx['num_sms']（GPU 专属配置，torch 体不消费）。
    num_sms = int(ctx.get('num_sms_dedup', ctx['num_sms']))
    # ctx 键名兼容：契约 §2 记 hidden_buf_local，api.py ctx 记 hidden_buf
    hidden_buf_local = ctx.get('hidden_buf_local', ctx.get('hidden_buf'))
    arena = ctx['arena']

    assert isinstance(plan, MoonEPCommPlan), "plan must be a MoonEPCommPlan"
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"
    assert hidden_buf_local is not None, \
        "ctx 缺少 hidden_buf_local/hidden_buf（本 rank NVL shard）"
    assert hidden_buf_local.dtype == torch.bfloat16 and \
        hidden_buf_local.is_contiguous(), \
        "hidden_buf_local must be contiguous bf16"
    assert tuple(hidden_buf_local.shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(hidden_buf_local.shape)}"
    assert arena is not None, "ctx['arena'] 缺失（契约 §0 的跨 rank 通道注入点）"

    def _check_tensor(t: torch.Tensor, name: str, shape: tuple) -> None:
        assert t.dtype == torch.int32 and t.is_contiguous(), \
            f"{name} must be contiguous int32"
        assert tuple(t.shape) == shape, \
            f"{name} must be shape {shape}, got {tuple(t.shape)}"

    _check_tensor(plan.dup_groups, "dup_groups", (NvS, 3))
    _check_tensor(plan.dup_loffs, "dup_loffs", (NvS,))
    _check_tensor(plan.dup_counts, "dup_counts", (2,))

    kernel = CombinePrologueKernel(
        H=H, R=R, NvS=NvS, num_sms=num_sms,
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        pdl_trigger=bool(pdl_trigger),
        arena=arena,
    )
    kernel(
        hidden_buf_local,   # hidden_buf_local_ptr
        plan.dup_groups,    # dup_groups_ptr
        plan.dup_loffs,     # dup_loffs_ptr
        plan.dup_counts,    # dup_counts_ptr
        None,               # stream 留参（参考实现同步执行）
    )
