# coding=utf-8
"""MoonEP CombineKernel 参考实现（torch 语义级）。

语义基准：GPU 版 MoonEP（source_code/MoonEP/moonep/combine.py）的
CombineKernel 与 launch_combine。按绑定级设计契约（docs/design.md §4）：
类名、``__init__`` 常量配置、``__call__`` 入参/出参、以及 ``launch_combine``
宿主启动函数签名与 GPU 源码逐字一致，仅把 kernel 体（3 级 warp 专业化的
G2S / fp32 ACC / S2G TMA 流水 + 权重 gather warp）替换为纯 torch 实现
（CPU 可跑）。性能优化（AscendC kernel 化）是后续工作。

语义（对齐 GPU kernel，combine.py:210-511）：把本 rank 每个 token 的 K 份
专家输出从全组对称分片拉回并归约——

1. **入口屏障**：对应 GPU kernel 入口的 cross_rank_barrier
   （combine.py:237-242），参考实现以 ``arena.barrier()`` 承担——发布全组已
   staging + 归并（CombinePrologueKernel）的分片后，才允许读取其它 rank 的行；
2. **hidden 路径**（combine.py:305-475）：逐 token 的 K 个**非负** dst，
   ``dr = v // NvS``、``loff = v % NvS``，经 ``arena.pull_all`` 从属主 rank
   dr 的 hidden 分片第 loff 块（块 = H 元素）拉回，fp32 累加（每 token 按
   k 升序，与 GPU ACC warps 的累加顺序逐位一致），最后一次性 cast bf16 写
   ``output_ptr [S, H]``；**负 dst 整项跳过**——其贡献已由
   CombinePrologueKernel 在属主 rank 归并进 primary 行；
3. **权重 gather**（GPU warp6，combine.py:485-511）：对全部 N=S·K 个槽位
   （含负 dst 的 ``raw = -v - 1`` 解码——路由权重按 topk 逐份散布、从不
   去重），经 ``arena.pull_all`` 从属主 rank dr 的 meta 分片 WEIGHTS 区
   （块偏移 ``weights_off + loff``，块 = 1 元素）取回 int32 位壳，原样写回
   ``output_sk [S, K]``——4 字节位壳搬运、无 fp32 运算，与 GPU
   ``meta[drank*meta_stride + weights_off + loff]`` 逐位一致（dispatch 权重
   散射的目的 rank 即此处的属主 rank dr）。

契约 §0 的替代设计（相对 GPU 的偏差以契约为准）：

- TMA cp.async.bulk + mbarrier 流水 → ``arena.pull_all``（块粒度、一次性
  集合调用）；PDL 无语义作用，不体现；
- GPU 的 ``hidden_ptr`` / ``meta_ptr`` 是覆盖全组 R 个分片的平坦对称视图
  （[R*NvS_padded, H] / [R*meta_chunk_padded]）；本参考实现传入**本 rank 已
  注册的分片张量**（[NvS, H] bf16 块 = H 元素 / [meta_chunk_padded] int32
  块 = 1 元素），跨 rank 寻址 ``drank*NvS_padded + loff`` 与
  ``drank*meta_stride + weights_off + loff`` 以 arena **展平块号**表达
  （flat = dr*chunk_blocks + off，两个注册张量的 chunk 步长即
  ``NvS_padded`` / ``meta_stride``，torch 体真实消费，与 GPU 源码逐字同形）。

签名保留不使用的 GPU 专属形参（docstring 逐一注明）：``bar_ptr``（grid
屏障计数器）、``rank``（GPU 版仅用于 cross_rank_barrier；arena 自带 rank
身份）、``barrier_off``（GPU 屏障槽区偏移；屏障由 arena 承担）、``stream``
（cuda.CUstream 形参位，接受 None）。

dtype 契约：payload bf16、meta int32、route weight fp32（位壳 int32）。
语义精确优先，不做性能技巧。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["CombineKernel", "launch_combine"]

# 参考实现的 smem 预算取值（与 api.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 __init__ 的构造契约。
_SMEM_BUDGET_BYTES = 231424


class CombineKernel:
    """跨 rank 拉取 + fp32 归约的 combine（类名/配置/签名与 GPU 源码逐字一致）。

    dedup 契约（combine.py:317-321 / 499-511）：非负 dst 拉行累加；负 dst
    （``-raw-1``，重复 topk 项）hidden 路径整项跳过；权重 gather 对负 dst
    照常 raw 解码读取（路由权重按 topk 逐份散布、从不去重）。
    """

    def __init__(
        self,
        H: int,
        R: int,
        S: int,
        K: int,
        NvS: int,
        NvS_padded: int,
        meta_stride: int,
        num_sms: int,
        with_weights: bool,
        smem_budget: int,
        pdl_launch: bool,
        *,
        arena: "SymmetricArena",
    ):
        # 常量配置与 GPU 源码 combine.py:66 起逐字一致；arena 为契约 §0 的
        # 唯一签名扩展点（torch 体的跨 rank 通道）。
        self.H = H
        self.R = R
        self.S = S
        self.K = K
        self.NvS = NvS
        # NvS_padded/meta_stride 为两个注册张量的 chunk 步长（展平块号寻址，
        # torch 体真实消费）；num_sms/smem_budget/pdl_launch 为 grid 规模、
        # smem 流水几何（stages_l 推导）与 PDL 启动开关；参考实现无对应概念，
        # 留存参不消费。
        self.NvS_padded = NvS_padded
        self.meta_stride = meta_stride
        self.num_sms = num_sms
        self.with_weights = with_weights
        self.smem_budget = smem_budget
        self.pdl_launch = pdl_launch
        self.arena = arena

    # ------------------------------------------------------------------ call

    def __call__(
        self,
        output_ptr: torch.Tensor,             # bf16 [S, H]
        output_sk_ptr: torch.Tensor,          # fp32 [S, K] 的 int32 位壳视图（或占位张量）
        hidden_ptr: torch.Tensor,             # bf16 [NvS, H] 本 rank 对称分片
        meta_ptr: torch.Tensor,               # int32 [meta_stride] 本 rank chunk
        dst_ptr: torch.Tensor,                # int32 [N=S*K]
        bar_ptr: torch.Tensor,                # 保留不使用：GPU grid 屏障计数器
        rank: int,                            # 保留不使用：GPU 版仅用于 cross_rank_barrier
        weights_off: int,
        barrier_off: int,                     # 保留不使用：GPU cross_rank_barrier 槽偏移
        stream,                               # 保留不使用：cuda.CUstream 形参位（接受 None）
    ):
        """执行一次 combine（同步语义）。

        Args:
            output_ptr: [S, H] bf16 输出缓冲（原地写）。
            output_sk_ptr: [S, K] fp32 路由权重的 int32 位壳视图（直接给 fp32
                张量亦可，kernel 内按 int32 位壳写回）；``with_weights=False``
                时为占位张量，不解引用（与 GPU host 传 dst 占位的约定一致）。
            hidden_ptr: [NvS, H] bf16 本 rank 对称分片（arena 注册，块 = H）。
            meta_ptr: [meta_stride] int32 本 rank meta chunk（arena 注册，
                块 = 1）；WEIGHTS 区位于 [weights_off, weights_off+NvS)，
                仅权重 gather 时读取。
            dst_ptr: [N=S*K] int32 路由偏移（须与散布进分片的那次 dispatch
                的 dst 一致）；负值 = -raw-1（hidden 跳过、权重照常 gather）。
            bar_ptr / rank / barrier_off / stream: GPU 专属形参，保留不使用
                （入口屏障由 arena.barrier() 承担，arena 自带 rank 身份）。
            weights_off: meta chunk 内 WEIGHTS 区偏移（int32 元素）。

        Returns:
            None（与 GPU kernel 一致，全部结果经 output_ptr / output_sk_ptr
            原地写出）。
        """
        H, R, S, K, NvS = self.H, self.R, self.S, self.K, self.NvS
        N = S * K
        arena = self.arena
        dev = dst_ptr.device

        # ---- 入参校验（dtype 契约：payload bf16 / meta int32 / 权重 fp32 位壳）----
        assert dst_ptr.dtype == torch.int32 and dst_ptr.is_contiguous() and \
            dst_ptr.numel() == N, \
            f"dst_ptr 必须是连续 int32 [N={N}]，got {tuple(dst_ptr.shape)}"
        assert output_ptr.dtype == torch.bfloat16 and \
            output_ptr.is_contiguous() and \
            tuple(output_ptr.shape) == (S, H), \
            f"output_ptr 形状应为 ({S}, {H}) bf16，got {tuple(output_ptr.shape)}"
        assert hidden_ptr.dtype == torch.bfloat16 and \
            hidden_ptr.is_contiguous() and \
            tuple(hidden_ptr.shape) == (NvS, H), \
            f"hidden_ptr 形状应为 ({NvS}, {H}) bf16，got {tuple(hidden_ptr.shape)}"

        # ---- 1) 入口屏障（对应 GPU kernel 入口 cross_rank_barrier）：发布全组
        #        已 staging + prologue 归并的分片，之后才读取其它 rank 的行 ----
        arena.barrier()

        # ---- dst 编解码（语义对齐 GPU combine.py:317-326 / 499-507）----
        # v >= 0 → raw = v（hidden 路径拉行累加）；v < 0 → raw = -v - 1
        # （hidden 路径跳过；权重 gather 照常解码读取）。内部统一 int64：
        # R*NvS 可逼近 int32 上限，且保证设备无关的确定性。
        v = dst_ptr.to(torch.int64).reshape(-1)                    # [N]
        nonneg = v >= 0
        raw = torch.where(nonneg, v, -v - 1)
        dr = torch.div(raw, NvS, rounding_mode="floor")            # 属主 rank
        loff = raw - dr * NvS                                      # 属主分片内槽位
        nonneg_sk = nonneg.view(S, K)

        # ---- 2) hidden 路径：非负 dst 槽位按展平块号一次 pull_all 集合拉取
        #        （对齐 GPU combine.py:327 的 srow = drank*NvS_padded + loff；
        #        请求序 = (s,k) 行主序，应答序 == 请求序）。集合调用：无请求
        #        也以 n=0 参与，保证全组调用点一致。本 rank 源由 transport
        #        走本地取数。----
        flat_h = dr * int(self.NvS_padded) + loff                # [N] 全槽
        pulled_h = arena.pull_all(hidden_ptr, flat_h[nonneg])    # [n, H] bf16

        # fp32 累加：k 外层循环保证每 token 严格按 k 升序累加（与 GPU ACC
        # warps 一致）；pos 记录每个非负槽位在请求/应答里的序号（行主序掩码
        # 还原 k 升序）。K 项全负的 token 累加结果为 0（与 GPU 相同）。
        pos = torch.full((N,), -1, dtype=torch.int64, device=dev)
        n_sel = int(nonneg.sum())
        pos[nonneg] = torch.arange(n_sel, dtype=torch.int64, device=dev)
        pos_sk = pos.view(S, K)
        acc32 = torch.zeros(S, H, dtype=torch.float32, device=dev)
        for k in range(K):
            sel = nonneg_sk[:, k]                              # [S]
            if bool(sel.any()):
                acc32[sel] += pulled_h[pos_sk[sel, k]].to(torch.float32)
        # fp32 → bf16 写回输出（GPU epilogue 的转换点，combine.py:426-429）
        output_ptr.copy_(acc32.to(torch.bfloat16))

        # ---- 3) 权重 gather（GPU warp6，combine.py:485-511）：全槽 raw 解码，
        #        从属主 rank 的 meta WEIGHTS 区取回 int32 位壳 ----
        if self.with_weights:
            assert meta_ptr.dtype == torch.int32 and meta_ptr.dim() == 1, \
                "meta_ptr 必须是一维 int32"
            assert meta_ptr.numel() >= weights_off + NvS, \
                "meta_ptr 长度不足以覆盖 WEIGHTS 区"
            assert output_sk_ptr.is_contiguous() and output_sk_ptr.numel() == N, \
                f"output_sk_ptr 必须是连续且含 N={N} 个元素的 [S, K]"
            # fp32 位壳（与 GPU api 层的 .view(torch.int32) 一致）
            sk_i32 = output_sk_ptr
            if sk_i32.dtype == torch.float32:
                sk_i32 = sk_i32.view(torch.int32)
            assert sk_i32.dtype == torch.int32, \
                f"output_sk_ptr 必须是 fp32（或其 int32 位壳视图），got {output_sk_ptr.dtype}"
            sk_flat = sk_i32.reshape(-1)                     # [N] int32 视图

            # 展平坐标全槽一次请求（对齐 GPU combine.py:508-511 的
            # meta[drank*meta_stride + weights_off + loff]）；dr/loff 由
            # dst_ptr.reshape(-1) 推出，请求序即 (s,k) 行主序，应答序 ==
            # 请求序——int32 位壳直接按序写回（4 字节 gather，无 fp32 运算）。
            flat_w = dr * int(self.meta_stride) + int(weights_off) + loff
            pulled_w = arena.pull_all(meta_ptr, flat_w)      # [N, 1] int32
            sk_flat.copy_(pulled_w.reshape(-1))


# ============================================================================
# 宿主启动函数（签名与 GPU 源码 combine.py:565 逐字一致）
# ============================================================================

def launch_combine(
    ctx: dict,
    output_sh,
    dst,
    output_sk=None,
    *,
    pdl_launch: bool = False,
):
    """Launch the combine kernel.

    Args:
        output_sh: [S, H] bf16 output buffer to receive the per-token
            accumulated result.
        dst: [N=S*K] int32 routing offsets (must match the dispatch that
            populated hidden_buf / weights_buf). Non-negative entries encode
            ``dest_rank * NvS + local_offset`` and are pulled and accumulated.
            Negative entries encode the same raw destination as
            ``-raw_dst - 1`` (duplicate top-k entries, pre-reduced into the
            primary slot by the combine prologue): the hidden path skips
            them, the weights gather decodes and reads them as usual.
        output_sk: [S, K] fp32 buffer to receive gathered route weights, or
            None to skip the weights gather (placeholder tensor is passed to
            satisfy the non-null pointer constraint; kernel ignores it when
            with_weights=False).

    参考实现说明：内部取 ``ctx['arena']`` 构造本包 CombineKernel 并调用其
    torch 体（契约 §4）；GPU 侧的每 (H,R,S,K,NvS,...,with_weights,pdl_launch)
    编译缓存（_get_compiled）在 torch 体下无编译概念，kernel 实例仅为常量
    配置持有者，每次调用新建即可。GPU 专属断言（H % ACC_THREADS、CUDA
    device、smem 预算）为 kernel 实现约束，torch 体无对应概念，不保留
    （与 prefetch.py 的 launch_prefetch 先例一致）。
    """
    with_weights = output_sk is not None
    if not with_weights:
        # 占位；kernel 不解引用（与 GPU host 传 dst 占位的约定一致）
        output_sk = dst

    assert output_sh.dtype == torch.bfloat16 and output_sh.is_contiguous(), \
        "output_sh must be contiguous bf16"
    assert dst.dtype == torch.int32 and dst.is_contiguous(), \
        "dst must be contiguous int32"
    # ctx 键名兼容：契约 §2 记 hidden_buf_local（api.py 实际写入键），回退 hidden_buf
    hidden_buf = ctx.get('hidden_buf_local', ctx.get('hidden_buf'))
    meta_buf = ctx['meta_buf']
    assert hidden_buf is not None, \
        "ctx 缺少 hidden_buf/hidden_buf_local（本 rank NVL shard）"
    assert hidden_buf.dtype == torch.bfloat16 and hidden_buf.is_contiguous()
    assert meta_buf.dtype == torch.int32 and meta_buf.is_contiguous()
    if with_weights:
        assert output_sk.dtype == torch.float32 and output_sk.is_contiguous(), \
            "output_sk must be contiguous fp32"

    H = int(ctx['H'])
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    NvS = int(ctx['NvS'])
    NvS_padded = int(ctx['NvS_padded'])
    meta_stride = int(ctx['meta_chunk_padded'])
    num_sms = int(ctx['num_sms'])
    arena = ctx['arena']
    assert arena is not None, "ctx['arena'] 缺失（契约 §0 的跨 rank 通道注入点）"

    kernel = CombineKernel(
        H=H, R=R, S=S, K=K, NvS=NvS, NvS_padded=NvS_padded,
        meta_stride=meta_stride, num_sms=num_sms,
        with_weights=with_weights,
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        pdl_launch=bool(pdl_launch),
        arena=arena,
    )

    # int32 位壳视图（对齐 GPU host 侧：kernel 始终看到 int32 指针，4 字节
    # gather、无 fp32 运算）；无权重路径 output_sk 已被 dst 占位
    sk_int = output_sk.view(torch.int32) if with_weights else output_sk

    kernel(
        output_sh,                      # output_ptr [S, H] bf16
        sk_int,                         # output_sk_ptr（int32 位壳或占位）
        hidden_buf,                     # hidden_ptr 本 rank 分片
        meta_buf,                       # meta_ptr 本 rank chunk
        dst,                            # dst_ptr [N] int32
        ctx.get('grid_sync_bar'),       # bar_ptr 占位（保留不使用；契约 §2 允许 None）
        int(ctx['rank']),               # rank（保留不使用）
        int(ctx['WEIGHTS_OFF']),        # weights_off
        int(ctx['BARRIER_OFF']),        # barrier_off（保留不使用）
        None,                           # stream 留参（参考实现同步执行）
    )
