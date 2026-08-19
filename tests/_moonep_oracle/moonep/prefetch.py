"""远程专家权重预取（PrefetchKernel + launch_prefetch 的 torch 语义级参考实现）。

语义基准：MoonEP ``moonep/prefetch.py`` 的 PrefetchKernel（persistent、warp 专用
2D TMA 流水：warp0 GMEM->SMEM TMA load，warp1 SMEM->GMEM TMA store，把选中的
远程专家权重从 ``remote_expert[E, H, H']`` 拷进 ``prefetch_buffers[B, H, H']``）。
绑定级设计契约见 docs/design.md §4：类名、``__init__`` 常量配置、``__call__``
入参/出参、以及 ``launch_prefetch`` 宿主启动函数签名与 GPU 源码逐字一致，
仅把 kernel 体替换为 torch 实现。

GPU 版借助 NVL 对称堆直接读属主 rank 的 ``remote_expert[E, H, H']``：远端行寻址
``remote row = expert_id * H + h``（即"读属主 rank 的第 e 个专家行"），再经 2D TMA
流水写入本地 ``prefetch_buffers[B, H, H']``。本参考实现以 arena 的集合词汇表达
同一语义（契约 §0：TMA cp.async.bulk + mbarrier 流水不模拟，代之以块粒度一次性
``pull_into`` 直达拷贝；契约 §3：pull_into 按本地张量 data_ptr 解析注册身份与
对端镜像）：

- 框架事先把 ``full_weight[epn+B, H, H']`` bf16 登记为对称张量（块 = H*H'
  元素，即一个专家行一整块，chunk_blocks = epn+B）：行 [0, epn) 是本 rank
  属主专家权重，行 [epn, epn+B) 是预取槽（契约 §5：
  ``remote_expert = full_weight``（注册张量本身）、
  ``prefetch_buffers = full_weight[epn:]``（其后缀视图）；每 rank 物理仅
  epn+B 行，对比 GPU 全局表 [E+B] 省 (R-1)*epn 行镜像死空间）。
- 对槽 b：``e = experts_ptr[b]``；``e < 0`` 为空槽，跳过（不拉取、不写入，
  本地槽行保持原值，与 GPU 版 launch_prefetch 的空槽约定一致）；否则属主
  rank ``= e // epn``，经 ``arena.pull_into`` 把属主 ``remote_expert`` 第 e
  行整块 [H,H'] **直接拉进本地预取槽行** ``prefetch_buf_ptr[b]``（对称堆
  直达，不经过普通内存落盘张量——对齐 GPU 版 TMA 直写 prefetch_buffers
  的单跳语义）。

epn 的来源说明：GPU 版 PrefetchKernel 的 ``__init__`` 没有 R（NVL 对称堆按行
直接寻址，无需属主 rank 概念）；参考实现求属主 rank 需要 epn = E // R，R 只能
取自 ``arena.world``（唯一可行来源，特此注明）。

规划语义保证槽中只放远程专家（planning 在 top-B argmax 前已把本地专家计数清零），
即 ``e // epn != rank`` 恒成立；本模块不对此额外假设，本地属主理论上也可经
pull_into 堆内自拷贝正确工作。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["PrefetchKernel", "launch_prefetch"]

# 参考实现的 smem 预算取值（与 api.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 __init__ 的构造契约（stages 选择与预算不足时的
# RuntimeError 行为）；取值须保证 _pick_stages 至少选中 2 级流水。
_SMEM_BUDGET_BYTES = 231424


class PrefetchKernel:
    """Persistent 2D TMA remote-expert prefetch（torch 语义级参考实现）。

    类常量与 ``__init__`` 常量配置照抄 GPU 源码（moonep/prefetch.py:32-58）；
    其中 NUM_THREADS / PRODUCER_WARP / CONSUMER_WARP / M_BLOCK / N_BLOCK /
    num_sms / smem_budget / stages 均为 GPU 专属配置，torch 体不消费，仅为
    保持构造契约（含 smem 预算不足时的 RuntimeError 行为）而原样保留。
    ``arena`` 为唯一签名扩展点（契约 §0：torch 体的跨 rank 数据移动须经 arena）。
    """

    NUM_THREADS = 64
    PRODUCER_WARP = 0
    CONSUMER_WARP = 1
    M_BLOCK = 128
    N_BLOCK = 128

    def __init__(
        self,
        E: int,
        H: int,
        Hp: int,
        B: int,
        num_sms: int,
        smem_budget: int,
        *,
        arena: "SymmetricArena",
    ):
        # ---- 常量配置与 GPU 源码 prefetch.py:38-58 逐字一致 ----
        self.E = E
        self.H = H
        self.Hp = Hp
        self.B = B
        self.num_sms = num_sms
        self.stages = self._pick_stages(smem_budget)
        if self.stages == 0:
            raise RuntimeError(
                "prefetch: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} bf16 tile under budget "
                f"{smem_budget} B"
            )
        # ---- 参考实现扩展（唯一签名扩展点）：对称内存注册表 ----
        self.arena = arena

    def _smem_bytes(self, stages: int) -> int:
        # 与 GPU 源码逐字一致（torch 体不消费 smem，仅为保持 __init__ 语义）
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * 2
        return (
            _round_up(stages * tile_bytes, 128)
            + _round_up(stages * 2 * 8, 16)
            + _round_up(2 * self.B * 4, 16)  # expert/slot compaction tables
            + 256
        )

    def _pick_stages(self, smem_budget: int) -> int:
        # 与 GPU 源码逐字一致
        for stages in (6, 5, 4, 3, 2):
            if self._smem_bytes(stages) <= smem_budget:
                return stages
        return 0

    def __call__(
        self,
        remote_expert_ptr: torch.Tensor,    # bf16 [epn+B, H, H']（本 rank 局部权重表，注册张量）
        prefetch_buf_ptr: torch.Tensor,     # bf16 [B, H, H']（remote_expert_ptr[epn:] 后缀视图）
        experts_ptr: torch.Tensor,          # int32 [B]
        stream,                             # 保留不使用：cuda.CUstream 形参位（接受 None）
    ):
        """torch 体：逐槽把属主 rank 的专家行整块**直接拉进**本地预取槽。

        对槽 b：``e = experts_ptr[b]``；``e < 0`` 跳过（空槽不拉取、不写入，
        本地槽行保持原值）；否则属主 rank ``= e // epn``，经 ``arena.pull_into``
        把属主 ``remote_expert`` 第 e 行整块 [H,H']（块 = H*H' 元素）直接写入
        本地预取槽行 ``prefetch_buf_ptr[b]``（对称堆直达，单跳 bf16 逐位拷贝，
        无普通内存落盘中转）。

        ``remote_expert_ptr`` 必须为已在 arena 注册的对称张量本身（data_ptr
        即 chunk 基址），``prefetch_buf_ptr`` 必须为其 ``[epn:]`` 后缀视图
        （同一段对称堆），pull_into 按注册张量的 data_ptr 解析注册身份、以
        ``dst_flat = epn + b`` 寻址目标槽行。
        ``stream`` 为 GPU 专属形参（cuda.CUstream），签名保留、参考实现不使用。
        """
        E, H, Hp, B = self.E, self.H, self.Hp, self.B
        # epn = E // R：GPU __init__ 无 R（NVL 按行直寻址），R 取自 arena.world
        world = self.arena.world
        assert E % world == 0, \
            f"E ({E}) 必须整除 arena.world ({world})"
        epn = E // world

        # dtype 契约：payload/bf16、meta/int32。
        # remote_expert_ptr 为本 rank 的**局部权重表** [epn+B, H, H']（已注册
        # 对称张量）：行 [0,epn) 为本 rank 属主专家（prefetch 远端读源），
        # 行 [epn,epn+B) 为预取槽——每 rank 物理仅 epn+B 行（对比全局表
        # [E+B] 的 R 倍冗余）。
        assert remote_expert_ptr.dtype == torch.bfloat16 \
            and tuple(remote_expert_ptr.shape) == (epn + B, H, Hp), \
            f"remote_expert_ptr 必须为 bf16 [epn+B, H, H']=[{epn + B}, {H}, {Hp}]，" \
            f"got {remote_expert_ptr.dtype} {tuple(remote_expert_ptr.shape)}"
        assert prefetch_buf_ptr.dtype == torch.bfloat16 \
            and tuple(prefetch_buf_ptr.shape) == (B, H, Hp), \
            f"prefetch_buf_ptr 必须为 bf16 [B, H, H']=[{B}, {H}, {Hp}]，" \
            f"got {prefetch_buf_ptr.dtype} {tuple(prefetch_buf_ptr.shape)}"
        assert experts_ptr.dtype == torch.int32 \
            and tuple(experts_ptr.shape) == (B,), \
            f"experts_ptr 必须为 int32 [B]=[{B}]，" \
            f"got {experts_ptr.dtype} {tuple(experts_ptr.shape)}"
        # 契约：prefetch_buf_ptr 必须为 remote_expert_ptr[epn:] 的后缀视图
        # （同一段对称堆）；pull_into 经注册张量寻址，本断言保证槽行视图与
        # dst_flat = epn + b 指向同一物理块。
        assert prefetch_buf_ptr.data_ptr() == \
            remote_expert_ptr[epn:].data_ptr(), (
                f"prefetch_buf_ptr 必须为 remote_expert_ptr[{epn}:] 的后缀视图"
                f"（data_ptr 应为 "
                f"{remote_expert_ptr[epn:].data_ptr()}，got "
                f"{prefetch_buf_ptr.data_ptr()}）"
            )

        # 展平块号（属主局部寻址）：src flat = owner*chunk_blocks + local_e，
        # dst flat = epn + b（本 rank 预取槽行），其中 chunk_blocks = epn + B
        # （每 rank 物理块数），local_e = e - owner*epn 为专家在属主 rank 内
        # 的局部行号。按 b 升序建表，请求顺序确定可复现。
        chunk_blocks = epn + B
        src_flats = []   # 源展平块号（b 升序）
        dst_flats = []   # 本地目标槽块号（与 src_flats 一一对应）
        for b in range(B):
            e = int(experts_ptr[b])
            if e < 0:
                continue  # 空槽：跳过，本地槽行保持原值
            assert e < E, f"experts_ptr[{b}]={e} 越界 [0, {E})"
            owner = e // epn
            src_flats.append(owner * chunk_blocks + (e - owner * epn))
            dst_flats.append(epn + b)

        # 集合调用：全组须在同一 API 阶段内各调一次 pull_into；
        # 即使本 rank 无活动槽（n=0）也必须参与，保证集合时序对齐。
        # 远端专家行一跳直达本地预取槽（对称堆内直写，无落盘中转）。
        self.arena.pull_into(
            remote_expert_ptr,
            torch.tensor(src_flats, dtype=torch.int64,
                         device=remote_expert_ptr.device),
            torch.tensor(dst_flats, dtype=torch.int64,
                         device=remote_expert_ptr.device),
        )


def launch_prefetch(
    remote_expert: torch.Tensor,
    prefetch_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    num_sms: int,
):
    """Launch remote expert prefetch（torch 语义级参考实现）。

    签名与 GPU 源码 moonep/prefetch.py:317 逐字一致（无 ctx）；arena 经全局
    注册表 ``resolve_arena_for(remote_expert)`` 获取（契约 §3：无 ctx 的
    launch_* 取 arena 的通道），内部构造本包 PrefetchKernel 并调用其 torch 体。

    Args:
        remote_expert: contiguous bf16 tensor shaped [epn+B, H, H']
            （本 rank 局部权重表，epn = E//R）。须为已在 arena 注册的对称
            张量本身（data_ptr 即 chunk 基址）。
        prefetch_buffers: contiguous bf16 tensor shaped [B, H, H']。 Must be
            the ``[epn:]`` suffix view of ``remote_expert``（同一段对称堆，
            kernel 体经 data_ptr 断言校验）。
        experts_to_copy: contiguous int32 tensor shaped [B].  Entries are
            expert ids in [0, E), or -1 for unused slots.  Unused slots are
            not written by this kernel.
        num_sms: number of persistent CTAs to launch。GPU 专属配置，torch 体
            不消费，仅为保持构造契约而透传。
    """
    if prefetch_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    assert remote_expert.dtype == torch.bfloat16 and remote_expert.is_contiguous(), \
        "remote_expert must be contiguous bf16 [epn+B, H, H']（本 rank 局部权重表）"
    assert prefetch_buffers.dtype == torch.bfloat16 and prefetch_buffers.is_contiguous(), \
        "prefetch_buffers must be contiguous bf16 [B, H, H']"
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [B]"
    assert remote_expert.ndim == 3, \
        f"remote_expert must have rank 3, got shape={tuple(remote_expert.shape)}"
    assert prefetch_buffers.ndim == 3, \
        f"prefetch_buffers must have rank 3, got shape={tuple(prefetch_buffers.shape)}"

    rows, H, Hp = (int(x) for x in remote_expert.shape)
    B, out_H, out_Hp = (int(x) for x in prefetch_buffers.shape)
    assert out_H == H and out_Hp == Hp, \
        f"prefetch_buffers shape {tuple(prefetch_buffers.shape)} incompatible with remote_expert {tuple(remote_expert.shape)}"
    assert experts_to_copy.numel() == B, \
        f"experts_to_copy length {experts_to_copy.numel()} must equal B={B}"
    # 注：GPU 版要求 H/Hp 为 128 的倍数（2D TMA tile 约束），参考实现的 torch 体
    # 无此约束（支持任意 H/Hp，测试规格 H=32/H'=16），故不保留该断言；
    # CUDA device 断言同属 GPU 专属（参考实现纯 torch、CPU 可跑），一并略去。
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"

    # 无 ctx：按 remote_expert 的 data_ptr 在全局注册表中解析所属 arena
    # （惰性导入：resolve_arena_for 由 buffer.py 提供，避免模块加载期的硬依赖）
    from .buffer import resolve_arena_for

    arena = resolve_arena_for(remote_expert)

    # remote_expert 为本 rank 局部表 [epn+B, H, H']（属主局部寻址，见 kernel
    # 注释）；由 rows = epn+B 与 arena.world 反解全局 E = epn*world。
    world = arena.world
    epn = rows - B
    assert epn > 0 and (epn * world) % world == 0, \
        f"remote_expert 行数 {rows} 与 B={B} 不满足 epn=rows-B>0"
    E = epn * world

    kernel = PrefetchKernel(
        E=E,
        H=H,
        Hp=Hp,
        B=B,
        num_sms=int(num_sms),
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        arena=arena,
    )
    kernel(remote_expert, prefetch_buffers, experts_to_copy, None)
