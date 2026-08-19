# coding=utf-8
"""MoonEP DispatchKernel + launch_dispatch 参考实现（torch 语义级）。

语义基准：GPU 版 MoonEP（source_code/MoonEP/moonep/dispatch.py）的 DispatchKernel
与 launch_dispatch。按绑定级设计契约（docs/design.md §4）：类名、``__init__``
常量配置、``__call__`` 入参/出参、以及 ``launch_dispatch`` 宿主启动函数签名与
GPU 源码逐字一致，仅把 kernel 体（warp 专用化 G2S/S2G TMA 流水、zero warp、
dedup builder warps）替换为纯 torch 实现（CPU 可跑）；``launch_dispatch`` 内部
构造本包 DispatchKernel 并调用其 torch 体。性能优化（AscendC kernel 化）是后续
工作。

与 GPU kernel 体的逐条语义对应（行号为 GPU 源码 dispatch.py）：

- consumer warp 的 dst 编解码与写路径（407-433）：对 offv∈[0,N)，v=dst[offv]；
  v>=0 → raw=v，payload 行 hidden_sh[offv//K] 与权重（fp32 位壳 int32）都写目的
  rank dr=raw//NvS 的 loff 槽；v<0 → raw=-v-1 同形解码，**只写权重**（dup 槽的
  payload 由 DispatchEpilogueKernel 在目的 shard 内从 primary 行原地扇出补齐）。
- zero warp（467-497）：按 zero_fill_ranges 把本 shard 各 VM 段的 padding 行
  清零；with_weights 时对应权重槽一并清零（fp32 0.0 与 int32 0 同为全零位型，
  int32 0 即 fp32 位壳 0.0）。reuse 路径同样执行。
- dedup builder warps（502-681，仅 build_dedup_map=True）：GPU 用
  primary_packed/kmask/kidx_to_loff scratch 做 atom_min 选举；参考实现改为读
  meta 的 SRC_INFO 本 rank 切片（本地 kidx 规则，与 GPU builder 逐位一致；同一
  (src_rank, token) 组内 dst>=0 的唯一槽为 primary），物化
  dup_groups/dup_loffs/dup_counts；组按 primary_loff 升序、组内 dup 按 loff
  升序（契约 §4 钉死的确定性顺序，保证下游 fp32 求和顺序可复现）。注意等价
  范围仅限 primary/dup 集合：GPU pass 2b 以 ctz 遍历 kmask、组内 dup 按 kidx
  升序发射，且组间顺序为 atomicAdd 到达序（运行间不稳定）；当同组 dup 的
  loff 与 kidx 非单调（不同 expert 段基址乱序，常见）时两边 dup_loffs 表序
  不同。对 dispatch_epilogue 无影响（各 dup 槽写入内容相同），但
  combine_prologue 按表序做 fp32 无权累加（primary 先行、dup 按表序），fp32
  加法不可结合——写回 primary 行的 bf16 结果与 GPU 可能存在末位 ulp 差异，
  combine 输出随之与 GPU 产生末位偏差。dedup 三表与 combine 输出不可与 GPU
  逐位对比，精度验证须对 naive oracle 用 bf16 容差（契约 §6）。
- 出口 cross_rank_barrier（687-692）→ arena.barrier()（契约 §3 全组屏障）。

契约 §0 的签名扩展与留参（GPU 专属形参保留不使用）：

- ``__init__`` 追加 ``arena``（与 ``ctx`` 可选）关键字参数——全包唯一的签名
  扩展点；kernel ctx 的 ``dst_all`` [R,N] int32 为 PlanningKernel 冗余重算副
  产物，build_dedup_map=True 时取用。
- smem_budget/num_sms/pdl_trigger：smem 流水几何与 PDL 触发，无语义作用；
  GPU 的 _pick_stages/_smem_bytes 推导（含 stages==0 的报错）不复制。
- NvS_padded / meta_stride：展平块号的 chunk 步长（每 rank 的块数），torch
  体真实消费——展平坐标 flat = dr*NvS_padded + loff / dr*meta_stride + off
  与 GPU 源码逐字同形（NvS_padded 取值当前退化为 NvS，见契约 §0）。
- bar_ptr/builder_bar_ptr：grid/builder 屏障计数器 scratch，保留不使用。
- primary_packed_ptr/kmask_ptr/kidx_to_loff_ptr：builder 选举 scratch，
  保留不使用（参考实现的选举与 GPU 同为本地 kidx 规则）。
- barrier_off：GPU cross_rank_barrier 的 meta 槽偏移，保留不使用
  （出口屏障由 arena.barrier() 承担）。
- stream：cuda.CUstream 形参保留（接受 None，不使用）。

寻址映射（契约 §0/§3）：GPU 的 gmem_dst[dr*NvS_padded+loff] 与
meta[dr*meta_stride+...] 为全组平坦寻址；参考实现以 arena 的**展平块号**
表达同一坐标：flat = dr*chunk_blocks + off（hidden_buf chunk 步长 =
NvS_padded，meta chunk 步长 = meta_stride），经单次 push_all 写出——
pe==rank 的项由 transport 本地拷贝（GPU 语义：所有 chunk RW 映射、本地
VA 直写），不再保留本 rank 直写分支。push_all 为集合调用，全组须在同一
API 阶段内调用——本 kernel 每次调用固定发起 1 次 hidden 的 push_all
（with_weights 时再加 1 次 meta 的 push_all），无写也以 n=0 参与。

launch_dispatch（签名逐字对齐 GPU dispatch.py:839）与 GPU host 侧的偏差
（契约 §0 许可）：

- CUDA 专属断言（is_cuda / H%8 对齐 / device_index）不保留——参考实现 CPU
  可跑；
- GPU scratch 实参（grid_sync_bar / primary_packed / kmask / kidx_to_loff /
  builder_bar）从 ctx 取用（契约 §2：这些键可省略或以 None 占位），缺省时以
  plan.dst 占位（kernel 体不解引用，与 GPU reuse 路径的占位约定一致）；
- 本 rank hidden 分片优先取 ctx['hidden_buf_local']（契约 §2 键），回退
  ctx['hidden_buf']（参考实现的 hidden_buf 即本 rank 分片，无 GPU 的全组平坦
  视图）；
- build_dedup_map=True 时 dup 三表由 kernel 读 meta 的 SRC_INFO 本 rank 切片
  以本地 kidx 规则物化（与 GPU builder 一致），不依赖任何 rank 的 dst。

dtype 契约：payload bf16、route weight fp32（线上以 fp32 位壳 int32 传输）、
meta/dedup 表 int32；块索引 int64。语义精确优先，不做性能技巧。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .planning import KIDX_BITS, MoonEPCommPlan

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["DispatchKernel", "launch_dispatch"]

# 参考实现的 smem 预算取值（与 api.py / prefetch.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 kernel 构造契约而透传。
_SMEM_BUDGET_BYTES = 231424

_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1


class DispatchKernel:
    """Dispatch 的 torch 语义级参考实现（类名/配置/签名与 GPU 源码逐字一致）。

    语义概览（与 GPU kernel 的 warp 分工对应，详见模块 docstring 行号映射）：

    - 数据路径：按 dst 编码把本 rank [S,H] token 的 payload 行散射到目的 rank
      的 hidden_buf 第 loff 块（仅 v>=0 槽位），并把全部槽位（含 v<0 的 raw
      解码槽）的 route weight 以 fp32 位壳写目的 rank meta 的 WEIGHTS 区；
    - zero 路径：按 zero_fill_ranges 清零本 shard 的 padding 行（及权重槽）；
    - builder 路径（仅 build_dedup_map=True）：由 meta SRC_INFO 切片以本地
      kidx 规则物化本 rank 的 dup_groups/dup_loffs/dup_counts（与 GPU 一致）；
    - 出口：arena.barrier()，向全组发布上述 NVL 写。
    """

    def __init__(
        self,
        H: int,
        R: int,
        S: int,
        K: int,
        zero_groups: int,       # zero_fill_ranges entries = E + B
        NvS: int,
        NvS_padded: int,
        SRC_INFO_OFF: int,
        meta_stride: int,
        num_sms: int,
        with_weights: bool,
        build_dedup_map: bool,
        smem_budget: int,
        pdl_trigger: bool,
        *,
        arena: "SymmetricArena",
        ctx=None,
    ):
        # 常量配置与 GPU 源码 dispatch.py:83-99 逐字一致；arena/ctx 为契约 §0
        # 的唯一签名扩展点。
        self.H = H
        self.R = R
        self.S = S
        self.K = K
        self.zero_groups = zero_groups
        self.NvS = NvS
        self.NvS_padded = NvS_padded
        self.meta_stride = meta_stride
        self.SRC_INFO_OFF = SRC_INFO_OFF
        self.num_sms = num_sms
        self.with_weights = with_weights
        self.pdl_trigger = pdl_trigger
        self.build_dedup_map = build_dedup_map
        # GPU 专属配置：smem_budget 为 TMA 流水 smem 预算（GPU 据此推导流水
        # 深度 stages）；参考实现无 smem 概念，留存参不消费。
        self.smem_budget = smem_budget
        self.arena = arena
        self.ctx = ctx

    # ------------------------------------------------------------------ call

    def __call__(
        self,
        hidden_sh_ptr: torch.Tensor,          # bf16 [S, H]
        hidden_buf_ptr: torch.Tensor,         # bf16 [NvS, H] 本 rank 对称分片
        weights_ptr: torch.Tensor,            # fp32 位壳 int32 视图 [S, K]（或占位张量）
        dst_ptr: torch.Tensor,                # int32 [N=S*K]
        meta_ptr: torch.Tensor,               # int32 [meta_stride] 本 rank chunk
        zero_fill_ranges_ptr: torch.Tensor,   # int32 [E+B, 2]（col0=pad_start, col1=n_pad）
        bar_ptr: torch.Tensor,                # 保留不使用：GPU grid 屏障计数器
        primary_packed_ptr: torch.Tensor,     # 保留不使用：GPU builder 选举 scratch
        kmask_ptr: torch.Tensor,              # 保留不使用：GPU builder 选举 scratch
        kidx_to_loff_ptr: torch.Tensor,       # 保留不使用：GPU builder 选举 scratch
        dup_groups_ptr: torch.Tensor,         # int32 [NvS, 3]（plan 持有，build_dedup_map 时物化）
        dup_loffs_ptr: torch.Tensor,          # int32 [NvS]（同左）
        dup_counts_ptr: torch.Tensor,         # int32 [2]（同左）
        builder_bar_ptr: torch.Tensor,        # 保留不使用：GPU builder 屏障
        rank: int,
        weights_off: int,
        barrier_off: int,                     # 保留不使用：GPU cross_rank_barrier 槽偏移
        stream,                               # 保留不使用：cuda.CUstream 形参位（接受 None）
    ):
        """执行一次 dispatch（同步语义；写路径见模块 docstring 的行号映射）。

        Args:
            hidden_sh_ptr: [S, H] bf16 源 token。
            hidden_buf_ptr: [NvS, H] bf16 本 rank 对称分片（arena 注册，块=H）。
            weights_ptr: fp32 [S, K] 路由权重的 int32 位壳视图（直接给 fp32 张量
                亦可，kernel 内按 int32 位壳取用）；with_weights=False 时为占位
                张量，不解引用。
            dst_ptr: [N] int32 目的编码；负值 = -raw-1（只写权重）。
            meta_ptr: [meta_stride] int32 本 rank meta chunk（arena 注册，块=1）；
                WEIGHTS 区位于 [weights_off, weights_off+NvS)，SRC_INFO 区位于
                [SRC_INFO_OFF, SRC_INFO_OFF+NvS)。
            zero_fill_ranges_ptr: [zero_groups, 2] int32 各 VM 段 padding
                [pad_start, n_pad)。
            bar_ptr / primary_packed_ptr / kmask_ptr / kidx_to_loff_ptr /
                builder_bar_ptr / barrier_off / stream: GPU 专属形参，保留不使用。
            dup_groups_ptr / dup_loffs_ptr / dup_counts_ptr: plan 持有的 dedup 表；
                仅 build_dedup_map=True 时物化，reuse 路径保持不动。
            rank: 本 rank 序号。
            weights_off: meta chunk 内 WEIGHTS 区偏移（int32 元素）。

        Returns:
            None（与 GPU kernel 一致，全部结果经对称内存原地写出）。
        """
        H, S, K, R, NvS = self.H, self.S, self.K, self.R, self.NvS
        N = S * K
        rank = int(rank)

        # ---- 入参校验（dtype 契约：payload bf16 / meta int32 / 权重 fp32 位壳）----
        assert hidden_sh_ptr.dtype == torch.bfloat16 and hidden_sh_ptr.is_contiguous(), \
            "hidden_sh_ptr 必须是连续 bf16"
        assert tuple(hidden_sh_ptr.shape) == (S, H), \
            f"hidden_sh_ptr 形状应为 ({S}, {H})，got {tuple(hidden_sh_ptr.shape)}"
        assert hidden_buf_ptr.dtype == torch.bfloat16 and hidden_buf_ptr.is_contiguous(), \
            "hidden_buf_ptr 必须是连续 bf16"
        assert tuple(hidden_buf_ptr.shape) == (NvS, H), \
            f"hidden_buf_ptr 形状应为 ({NvS}, {H})，got {tuple(hidden_buf_ptr.shape)}"
        assert dst_ptr.dtype == torch.int32 and dst_ptr.numel() == N, \
            f"dst_ptr 必须是 [N={N}] int32"
        assert meta_ptr.dtype == torch.int32 and meta_ptr.dim() == 1, \
            "meta_ptr 必须是一维 int32"
        assert meta_ptr.numel() >= self.SRC_INFO_OFF + NvS and \
            meta_ptr.numel() >= weights_off + NvS, \
            "meta_ptr 长度不足以覆盖 WEIGHTS/SRC_INFO 区"
        assert zero_fill_ranges_ptr.dtype == torch.int32 and \
            tuple(zero_fill_ranges_ptr.shape) == (self.zero_groups, 2), \
            f"zero_fill_ranges_ptr 必须是 ({self.zero_groups}, 2) int32"

        # ---- dst 编解码（语义对齐 GPU dispatch.py:407-433）----
        # v >= 0 → raw = v（payload 与权重都写）；v < 0 → raw = -v - 1（只写权重）。
        # 内部统一 int64：R*NvS 可逼近 int32 上限，且保证设备无关的确定性。
        v = dst_ptr.to(torch.int64).reshape(-1)                    # [N]
        nonneg = v >= 0
        raw = torch.where(nonneg, v, -v - 1)
        dr = torch.div(raw, NvS, rounding_mode="floor")            # 目的 rank
        loff = raw - dr * NvS                                      # 目的槽位（块内偏移）
        offv = torch.arange(N, dtype=torch.int64, device=v.device)
        tok = torch.div(offv, K, rounding_mode="floor")            # payload 源行 = offv//K

        # ---- payload 散射：仅 v>=0 槽位（每个目的槽恰被一个 offv 写，planning
        # 保证）。展平块号寻址（对齐 GPU dispatch.py:418 的
        # drow = drank*NvS_padded + loff）：flat = dr*NvS_padded + loff，
        # pe==rank 的项由 transport 本地拷贝（GPU 语义：所有 chunk RW 映射、
        # 本地 VA 直写），单次 push_all 完成本地+远端全部写。
        flat_h = dr[nonneg] * int(self.NvS_padded) + loff[nonneg]   # offv 升序
        self.arena.push_all(hidden_buf_ptr, flat_h, hidden_sh_ptr[tok[nonneg]])

        # ---- 权重散射（对应 GPU 的 with_weights 路径）：全部槽位（含 v<0 的
        # raw 解码槽；路由权重按 topk 逐份散布、从不去重）。写 meta 的 WEIGHTS
        # 区（块=1 元素）。展平坐标（对齐 GPU dispatch.py:431 的
        # meta[drank*meta_stride + weights_off + loff]）。
        if self.with_weights:
            w = weights_ptr
            if w.dtype == torch.float32:
                # fp32 位壳（与 GPU api 层的 .view(torch.int32) 一致）
                w = w.view(torch.int32)
            assert w.dtype == torch.int32 and w.numel() == N, \
                f"weights_ptr 必须是 fp32/int32 位壳且含 N={N} 个元素"
            w_flat = w.reshape(-1)                                 # [N] int32
            flat_w = dr * int(self.meta_stride) + int(weights_off) + loff
            self.arena.push_all(meta_ptr, flat_w, w_flat.unsqueeze(1))

        # ---- padding 清零（对应 GPU zero warp；纯本地，reuse 路径同样执行）----
        zfr = zero_fill_ranges_ptr.to(torch.int64)
        for g in range(self.zero_groups):
            pad_start = int(zfr[g, 0])
            n_pad = int(zfr[g, 1])
            if n_pad > 0:
                hidden_buf_ptr[pad_start:pad_start + n_pad].zero_()
                if self.with_weights:
                    # fp32 0.0 与 int32 0 同为全零位型，int32 0 即 fp32 位壳 0.0
                    w_lo = weights_off + pad_start
                    meta_ptr[w_lo:w_lo + n_pad].zero_()

        # ---- dedup 建表（对应 GPU builder warps；仅 fresh planning 路径）----
        if self.build_dedup_map:
            self._build_dedup_map(
                meta_ptr, dup_groups_ptr, dup_loffs_ptr, dup_counts_ptr, rank
            )

        # ---- 出口屏障（对应 GPU cross_rank_barrier）：向全组发布本 rank 的
        # hidden/meta 写，之后对端 rank 才能消费这些行 ----
        self.arena.barrier()

    # ------------------------------------------------------- dedup 建表（内部）

    def _build_dedup_map(
        self,
        meta_ptr: torch.Tensor,
        dup_groups_ptr: torch.Tensor,
        dup_loffs_ptr: torch.Tensor,
        dup_counts_ptr: torch.Tensor,
        rank: int,
    ) -> None:
        """物化本 rank 的 dup_groups/dup_loffs/dup_counts（plan 持有，原地写）。

        与 GPU builder（dispatch.py:502-681）逐条对齐的**纯本地**算法——只需
        meta 的 SRC_INFO 本 rank 切片，不需要任何 rank 的 dst：

        - pass 1（557-575）：槽位 loff 的出处 info=(src_rank, offv)，组键
          key=src_rank*S+token，packed=(kidx<<NvS_BITS)|loff 按 key 取 min
          选 primary（GPU 用 atom_min），kmask 按 key 累积 1<<kidx
          （GPU 用 atom_or），kidx_to_loff[key*K+kidx]=loff；
        - primary 即组内 min kidx 槽——恰为 planning Phase D 按 k 升序首个
          保持非负 dst 的槽（同一 (src_rank, token) 组在本 rank 的槽位集合
          就是该 token 发往本 rank 的全部条目，Phase D 的 k 序扫描保证首个
          非负），两边选举结果逐位一致；
        - pass 2（592-679）：loff==primary_loff 且 dup_count=popc(kmask)-1>0
          的槽成组；组内 dup 按 **kidx 升序**（GPU ctz(kmask) 发射序）经
          kidx_to_loff 映射发出；组间顺序 GPU 为 atomicAdd 到达序（运行间
          不稳定），本实现取确定性的 primary_loff 升序——消费者
          （epilogue/prologue）按下标迭代，语义兼容。
        """
        S, K, NvS = self.S, self.K, self.NvS
        assert dup_groups_ptr.dtype == torch.int32 and \
            tuple(dup_groups_ptr.shape) == (NvS, 3)
        assert dup_loffs_ptr.dtype == torch.int32 and \
            tuple(dup_loffs_ptr.shape) == (NvS,)
        assert dup_counts_ptr.dtype == torch.int32 and \
            tuple(dup_counts_ptr.shape) == (2,)
        dev = meta_ptr.device

        NvS_BITS = 32 - 1 - KIDX_BITS
        NvS_MASK = (1 << NvS_BITS) - 1

        # SRC_INFO 本 rank 切片（GPU：meta[rank*meta_stride + SRC_INFO_OFF + loff]）
        src_info = meta_ptr[self.SRC_INFO_OFF:self.SRC_INFO_OFF + NvS].to(torch.int64)
        valid = src_info >= 0
        loffs = valid.nonzero(as_tuple=False).squeeze(1)           # 非空槽，loff 升序

        dup_groups_ptr.zero_()
        dup_loffs_ptr.zero_()
        dup_counts_ptr.zero_()

        if loffs.numel() == 0:
            return

        info = src_info[loffs]
        src_rank = torch.div(info, NvS, rounding_mode="floor")
        offv = info - src_rank * NvS
        token = torch.div(offv, K, rounding_mode="floor")
        kidx = offv - token * K
        key = src_rank * S + token                                 # (源 rank, token) 组键
        packed = (kidx << NvS_BITS) | loffs                        # 选举编码

        # ---- pass 1：atom_min 选举 + atom_or 累积 kmask（按 key 分组）----
        uniq, inverse = torch.unique(key, return_inverse=True)     # key 升序
        G = uniq.numel()
        prim_packed = torch.full((G,), _INT64_MAX, dtype=torch.int64, device=dev)
        prim_packed.scatter_reduce_(0, inverse, packed, reduce="amin")
        kmask = torch.zeros(G, dtype=torch.int64, device=dev)
        kmask.scatter_add_(0, inverse, 1 << kidx)                  # 同 key 内 kidx 唯一
        kidx_to_loff = torch.zeros(G, K, dtype=torch.int64, device=dev)
        kidx_to_loff[inverse, kidx] = loffs

        # ---- pass 2：分类并物化 ----
        prim_packed_slot = prim_packed[inverse]                    # 每槽其组的选举结果
        primary_loff = prim_packed_slot & NvS_MASK
        primary_kidx = prim_packed_slot >> NvS_BITS
        kc = kmask[inverse]
        dup_count = torch.zeros_like(kc)
        _bits = kc
        while bool((_bits > 0).any()):
            dup_count += _bits & 1
            _bits = _bits >> 1
        dup_count -= 1                                             # popc(kmask)-1

        is_primary = loffs == primary_loff
        has_dups = is_primary & (dup_count > 0)

        # 组间按 primary_loff 升序（loffs 本已升序，has_dups 选择保持该序）
        grp_loffs = loffs[has_dups]
        grp_counts = dup_count[has_dups]
        grp_keys = key[has_dups]
        n_groups = int(grp_loffs.numel())

        dup_loff_list = []
        for gi in range(n_groups):
            g = int((uniq == grp_keys[gi]).nonzero(as_tuple=False).item())
            pk = int(primary_kidx[(inverse == g).nonzero(as_tuple=False)[0, 0]])
            for kk in range(K):                                  # ctz 序 = kidx 升序
                if kk != pk and bool((kmask[g] >> kk) & 1):
                    dup_loff_list.append(int(kidx_to_loff[g, kk]))

        n_dups = len(dup_loff_list)
        if n_groups > 0:
            starts = torch.zeros(n_groups, dtype=torch.int64, device=dev)
            if n_groups > 1:
                starts[1:] = grp_counts.cumsum(dim=0)[:-1]
            dup_groups_ptr[:n_groups, 0] = grp_loffs.to(torch.int32)
            dup_groups_ptr[:n_groups, 1] = starts.to(torch.int32)
            dup_groups_ptr[:n_groups, 2] = grp_counts.to(torch.int32)
        if n_dups > 0:
            dup_loffs_ptr[:n_dups] = torch.tensor(
                dup_loff_list, dtype=torch.int32, device=dev
            )
        dup_counts_ptr[0] = n_groups
        dup_counts_ptr[1] = n_dups


# ============================================================================
# Host launcher（torch 语义级参考实现；签名与 GPU dispatch.py:839 逐字一致）
# ============================================================================

def _check_dispatch_plan(ctx: dict, hidden_sh: torch.Tensor, plan: MoonEPCommPlan) -> None:
    """plan 与 ctx 的一致性校验（对齐 GPU dispatch.py:787-814 的语义子集）。

    plan 各张量字段的形状/dtype/连续性已由 MoonEPCommPlan.__post_init__ 钉死；
    此处复核 plan 整数字段与 ctx 常量一致、plan 张量与 hidden_sh 同设备。
    GPU 版的 CUDA 设备断言（t.device == cuda dev）不保留——参考实现 CPU 可跑。
    """
    S = int(ctx["S"])
    K = int(ctx["K"])
    N = S * K
    R = int(ctx["R"])
    NvS = int(ctx["NvS"])

    assert plan.N == N, f"plan.N must be S*K={N}, got {plan.N}"
    assert plan.R == R, f"plan.R must match ctx R={R}, got {plan.R}"
    assert plan.K == K, f"plan.K must match ctx K={K}, got {plan.K}"
    assert plan.NvS == NvS, f"plan.NvS must match ctx NvS={NvS}, got {plan.NvS}"
    assert plan.dst.device == hidden_sh.device, (
        f"plan tensors must be on {hidden_sh.device}, got {plan.dst.device}"
    )


def _resolve_hidden_buf_local(ctx: dict) -> torch.Tensor:
    """取本 rank hidden 对称分片 [NvS, H] bf16。

    优先 ctx['hidden_buf_local']（契约 §2 键）；回退 ctx['hidden_buf']——参考
    实现的 hidden_buf 即本 rank 分片（每 rank 一块的 arena 对称张量），不存在
    GPU 的全组平坦视图 [R*NvS_padded, H]。
    """
    t = ctx.get("hidden_buf_local")
    if t is None:
        t = ctx.get("hidden_buf")
    assert isinstance(t, torch.Tensor), (
        "ctx 缺少 'hidden_buf_local'（本 rank [NvS,H] bf16 对称分片，契约 §2 键）"
    )
    return t


def _resolve_scratch(ctx: dict, name: str, fallback: torch.Tensor) -> torch.Tensor:
    """取 GPU scratch 占位实参：ctx 键可缺省或为 None（契约 §2），缺省用 fallback。

    kernel 体对这些形参一律不解引用（保留不使用的 GPU 专属形参），fallback 取
    plan.dst——与 GPU reuse 路径把 builder scratch 指到 plan.dst 的占位约定一致。
    """
    t = ctx.get(name)
    return t if isinstance(t, torch.Tensor) else fallback


def launch_dispatch(
    ctx: dict,
    hidden_sh,
    route_weights_sk,
    plan,
    *,
    build_dedup_map: bool = True,
    pdl_trigger: bool = False,
):
    """Launch the dispatch kernel（torch 语义级参考实现）。

    签名与 GPU 源码 dispatch.py:839 逐字一致；内部按 ctx 常量构造本包
    DispatchKernel 并调用其 torch 体（契约 §4：launch_* 内部调用本包 torch
    kernel 实现）。

    Args:
        hidden_sh: [S, H] bf16 source hidden states。
        route_weights_sk: [S, K] fp32 route weights；None 时跳过权重散射（按
            GPU 惯例以 plan.dst 作占位实参，kernel 不解引用）。
        plan: 通信规划（MoonEPCommPlan），携带 ``dst`` / ``zero_fill_ranges``
            与 plan 持有的 dedup 三表。非负 ``dst`` = ``dest_rank*NvS +
            local_offset``（搬 payload）；负值 = ``-raw_dst - 1``（只散布权重）。
        build_dedup_map: 仅新鲜规划后为 True；复用与反向路径传 False，已保存的
            dedup 三表（``dup_groups`` / ``dup_loffs`` / ``dup_counts``）保持
            不动。zero 路径两条路径都执行。
        pdl_trigger: GPU PDL 开关；签名保留，参考实现无语义作用（仅作 kernel
            常量透传）。

    Returns:
        None（全部结果经对称内存与 plan 持有的 dedup 表原地写出）。
    """
    assert isinstance(plan, MoonEPCommPlan)
    with_weights = route_weights_sk is not None
    if not with_weights:
        # 占位；kernel 不解引用（与 GPU host 侧同约定）
        route_weights_sk = plan.dst

    H = int(ctx["H"])
    R = int(ctx["R"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    E = int(ctx["E"])
    B = int(ctx.get("B", 0))
    NvS = int(ctx["NvS"])
    # 参考实现无 VMM 粒度对齐：NvS_padded 退化为 NvS（缺省键亦按 NvS 处理）
    NvS_padded = int(ctx.get("NvS_padded", NvS))
    meta_stride = int(ctx["meta_chunk_padded"])
    SRC_INFO_OFF = int(ctx["SRC_INFO_OFF"])
    num_sms = int(ctx["num_sms"])

    assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), \
        "hidden_sh must be contiguous bf16"
    assert tuple(hidden_sh.shape) == (S, H), \
        f"hidden_sh must be shape [S={S}, H={H}], got {tuple(hidden_sh.shape)}"
    _check_dispatch_plan(ctx, hidden_sh, plan)
    hidden_buf_local = _resolve_hidden_buf_local(ctx)
    meta_buf = ctx["meta_buf"]
    assert hidden_buf_local.dtype == torch.bfloat16 and \
        hidden_buf_local.is_contiguous(), \
        "hidden_buf_local must be contiguous bf16"
    assert tuple(hidden_buf_local.shape) == (NvS, H), \
        f"hidden_buf_local must be shape ({NvS}, {H}), " \
        f"got {tuple(hidden_buf_local.shape)}"
    assert meta_buf.dtype == torch.int32 and meta_buf.is_contiguous() and \
        meta_buf.dim() == 1, "meta_buf must be contiguous 1-D int32"
    assert meta_buf.numel() >= SRC_INFO_OFF + NvS, \
        "meta_buf 长度不足以覆盖 SRC_INFO 区"
    if with_weights:
        assert route_weights_sk.dtype == torch.float32 and \
            route_weights_sk.is_contiguous(), \
            "route_weights_sk must be contiguous fp32"
        assert tuple(route_weights_sk.shape) == (S, K), \
            f"route_weights_sk must be shape [S={S}, K={K}], " \
            f"got {tuple(route_weights_sk.shape)}"
    # 注：GPU 的 is_cuda / H%8 对齐 / device_index 断言为 CUDA 专属，不保留。
    if build_dedup_map:
        # 编码不变式（GPU _check_dedup_builder_bounds 的语义子集）：
        # dst/src_info 线性编码的上界约束。primary_packed/kmask 的位域约束
        # （KIDX_BITS 等）与 _check_dedup_builder_tensors 的 scratch 形状校验为
        # GPU builder 选举编码专属——参考实现的建表不消费这些 scratch，不保留。
        N = S * K
        assert N <= NvS, (
            f"src_info NvS-stride encoding requires S*K <= NvS, got S*K={N}, NvS={NvS}"
        )
        assert R * NvS <= _INT32_MAX, (
            "src_info linear encoding requires R*NvS <= int32_max: "
            f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={_INT32_MAX}"
        )

    arena = ctx.get("arena")
    assert arena is not None, "ctx 缺少 'arena' 键（契约 §2 参考实现扩展键）"

    # GPU 版 builder 的 scratch 经 kernel 的 ctx 形参传入；本实现的建表只读
    # meta SRC_INFO 切片（本地 kidx 规则，见 _build_dedup_map），无需副产物
    # 通道，ctx 形参置 None。
    kernel_ctx = None

    # GPU scratch 占位（kernel 体不解引用；ctx 键缺省/为 None 时以 plan.dst 占位）
    grid_sync_bar = _resolve_scratch(ctx, "grid_sync_bar", plan.dst)
    primary_packed = _resolve_scratch(ctx, "primary_packed", plan.dst)
    kmask = _resolve_scratch(ctx, "kmask", plan.dst)
    kidx_to_loff = _resolve_scratch(ctx, "kidx_to_loff", plan.dst)
    builder_bar = _resolve_scratch(ctx, "builder_bar", plan.dst)

    kernel = DispatchKernel(
        H=H, R=R, S=S, K=K,
        zero_groups=E + B,
        NvS=NvS, NvS_padded=NvS_padded,
        SRC_INFO_OFF=SRC_INFO_OFF, meta_stride=meta_stride,
        num_sms=num_sms,
        with_weights=with_weights,
        build_dedup_map=bool(build_dedup_map),
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        pdl_trigger=bool(pdl_trigger),
        arena=arena,
        ctx=kernel_ctx,
    )

    # fp32 权重以 int32 位壳传入（与 GPU host 侧 .view(torch.int32) 一致）；
    # 无权重路径 route_weights_sk 已是 plan.dst 占位。
    w_int = route_weights_sk.view(torch.int32) if with_weights else route_weights_sk

    kernel(
        hidden_sh,                            # hidden_sh_ptr
        hidden_buf_local,                     # hidden_buf_ptr（本 rank 分片）
        w_int,                                # weights_ptr（int32 位壳或占位）
        plan.dst,                             # dst_ptr
        meta_buf,                             # meta_ptr（本 rank chunk）
        plan.zero_fill_ranges,                # zero_fill_ranges_ptr
        grid_sync_bar,                        # bar_ptr 占位（保留不使用）
        primary_packed,                       # primary_packed_ptr 占位（保留不使用）
        kmask,                                # kmask_ptr 占位（保留不使用）
        kidx_to_loff,                         # kidx_to_loff_ptr 占位（保留不使用）
        plan.dup_groups,                      # dup_groups_ptr
        plan.dup_loffs,                       # dup_loffs_ptr
        plan.dup_counts,                      # dup_counts_ptr
        builder_bar,                          # builder_bar_ptr 占位（保留不使用）
        int(ctx["rank"]),                     # rank
        int(ctx["WEIGHTS_OFF"]),              # weights_off
        int(ctx["BARRIER_OFF"]),              # barrier_off（保留不使用）
        None,                                 # stream 留参（参考实现同步执行）
    )
