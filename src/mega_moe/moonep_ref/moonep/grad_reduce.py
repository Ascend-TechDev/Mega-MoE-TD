"""远程专家梯度归约（GradReduceKernel + launch_grad_reduce 的 torch 语义级参考实现）。

语义基准：MoonEP ``moonep/grad_reduce.py`` 的 GradReduceKernel（persistent
warp 专用 tile 归约：1 个 load warp 流水拉远程 reduce tile，4 个 fp32 ACC warp
以本地梯度为种子累加并写回；prescan 协同压缩活跃专家表；phase 2 跨 rank 屏障后
各 rank 仅本地清零自己被消费的槽）。绑定级设计契约见 docs/design.md §4：类名、
``__init__`` 常量配置、``__call__`` 入参/出参、以及 ``launch_grad_reduce``
宿主启动函数签名与 GPU 源码逐字一致，仅把 kernel 体替换为 torch 实现。

本参考实现以 arena 集合词汇表达同一语义（契约 §4），分三步：

1. 预扫描 + 集合拉取：对本 rank 每个属主专家
   ``e ∈ [rank*epn, (rank+1)*epn)``，在全组 ``experts_ptr [R,B]`` 中找所有
   ``(r,b)`` 使 ``experts_ptr[r,b] == e``（GPU prescan 仅按专家 id 区间匹配，
   **不排除 r == rank 的自发槽**；真实规划中槽里只放远程专家，自发槽不会
   出现，但合成 plan（如 fan_in 自发自收）语义须与 GPU 逐位一致）。按 rb
   升序（即 (r,b) 字典序，
   与 GPU prescan 的 slist 顺序逐位一致——slist 由 match_any 保序，逐位等价
   于串行 rb 扫描）整理清单，一次 ``pull_all`` 拉取全部相关 reduce 槽
   （fp32，块 = H*H' 元素；pull_all 按 ``reduce_buf_ptr`` 的 data_ptr 解析
   注册身份与对端镜像，契约 §3）。
2. 累加（对应 GPU phase 1）：以 ``expert_grad_ptr[e]``（本地 wgrad，已在其中）
   为种子，按 (r,b) 字典序 fp32 累加各槽——fp32 加法不可结合，固定顺序保证
   确定可复现。无远程槽的属主专家零开销（与 GPU 版 active-list 行为一致）。
   与 GPU 版一致，累加从不写 reduce buffer。
3. 屏障 + 清零（对应 GPU phase 2）：``arena.barrier()`` 确保全组都读完本 rank
   的槽之后，各 rank 把本 rank reduce buffer 中被消费的槽
   （``experts_ptr[rank,b] >= 0`` 的 b）本地清零。清零严格发生在 barrier 之后
   （"全组读完再清零"的调用契约）。GPU 版 cross_rank_barrier 的槽位协议
   （meta_ptr 的 BARRIER 区 + bar_ptr 网格屏障计数，barrier_off 为区内偏移）
   其语义由 ``arena.barrier()`` 承担，故这三个形参保留不使用。

reduce_buf 形状说明：GPU 版 ``reduce_buf_ptr`` 为 NVL 对称堆上的
``[R, B, H, H']`` 全组视图（远端槽按行直接寻址）；参考实现中对称张量按
"每 rank 一个物理分片"注册，故本模块消费的是本 rank 的 ``[B, H, H']``
物理分片（块 = H*H' 元素），对端镜像由 arena 解析。契约 §5：``reduce_grad``
逐投影把已 register 的 reduce buffer 作为 ``remote_reduce_buffers`` 传入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffer import SymmetricArena

__all__ = ["GradReduceKernel", "launch_grad_reduce"]

# 参考实现的 smem 预算取值（与 api.py 的 _SMEM_BUDGET_BYTES 一致）。
# torch 体不消费 smem，仅为保持 __init__ 的构造契约（预算不足时的
# RuntimeError 行为）；取值须大于 _smem_bytes() 的需求。
_SMEM_BUDGET_BYTES = 231424


class GradReduceKernel:
    """Persistent fp32 tile reducer for remote expert grads（torch 语义级参考实现）。

    类常量与 ``__init__`` 常量配置照抄 GPU 源码（moonep/grad_reduce.py:49-79）；
    其中 M_BLOCK / N_BLOCK / STAGES / ACC_THREADS / NUM_THREADS / num_sms /
    smem_budget / meta_stride 均为 GPU 专属配置，torch 体不消费，仅为保持
    构造契约（含 smem 预算不足时的 RuntimeError 行为）而原样保留。
    ``arena`` 为唯一签名扩展点（契约 §0：torch 体的跨 rank 数据移动须经 arena）。
    """

    M_BLOCK = 128
    N_BLOCK = 128
    STAGES = 3                 # load-pipeline depth
    ACC_THREADS = 128          # warps 1..4
    NUM_THREADS = 160          # 5 warps: 1 load + 4 fp32 acc (acc also stores)

    def __init__(
        self,
        E: int,
        H: int,
        Hp: int,
        R: int,
        B: int,
        meta_stride: int,
        num_sms: int,
        smem_budget: int,
        *,
        arena: "SymmetricArena",
    ):
        # ---- 常量配置与 GPU 源码 grad_reduce.py:55-79 逐字一致 ----
        self.E = E
        self.H = H
        self.Hp = Hp
        self.R = R
        self.B = B
        self.meta_stride = meta_stride
        self.num_sms = num_sms
        need = self._smem_bytes()
        if need > smem_budget:
            raise RuntimeError(
                "grad_reduce: not enough per-block shared memory for one "
                f"{self.M_BLOCK}x{self.N_BLOCK} fp32 acc tile: need {need} B, "
                f"budget {smem_budget} B"
            )
        # ---- 参考实现扩展（唯一签名扩展点）：对称内存注册表 ----
        self.arena = arena

    def _smem_bytes(self) -> int:
        # 与 GPU 源码逐字一致（torch 体不消费 smem，仅为保持 __init__ 语义）
        def _round_up(n: int, a: int) -> int:
            return (n + a - 1) // a * a

        tile_bytes = self.M_BLOCK * self.N_BLOCK * 4
        stage = _round_up(self.STAGES * tile_bytes, 128)
        mbar = _round_up(self.STAGES * 2 * 8, 16)
        # off[EPN+1] + alist[EPN] + acnt[1] + slist[R*B] + sexp[R*B]
        # + cur[EPN+1] + clist[B] + ccnt[1]
        epn = self.E // self.R
        scan = _round_up(
            (3 * epn + 4 + 2 * self.R * self.B + self.B) * 4, 128)
        return stage + mbar + scan + 256

    def __call__(
        self,
        expert_grad_ptr: torch.Tensor,    # fp32 [epn, H, H']（本 rank 属主局部梯度表）
        reduce_buf_ptr: torch.Tensor,     # fp32 [R, B, H, H']（参考实现为本 rank 的 [B,H,H'] 物理分片）
        experts_ptr: torch.Tensor,        # int32 [R, B]
        meta_ptr: torch.Tensor,           # int32 [R*meta_stride] (barrier)：GPU 专属形参，保留不使用
        bar_ptr: torch.Tensor,            # int32 [1] grid barrier counter：GPU 专属形参，保留不使用
        rank: int,
        barrier_off: int,
        stream,                           # 保留不使用：cuda.CUstream 形参位（接受 None）
    ):
        """torch 体：把全组预取槽中的 fp32 梯度归并回本 rank 属主专家梯度行。

        对本地属主专家 ``e ∈ [rank*epn, (rank+1)*epn)``（epn = E//R），找全组
        ``(r,b)`` 使 ``experts_ptr[r,b] == e``（与 GPU prescan 一致：仅按专家
        id 区间匹配，不排除 r == rank 的自发槽），经
        ``arena.pull_all`` 拉回其 reduce 槽 fp32 块，按 (r,b) 字典序累加进
        ``expert_grad_ptr[e]``；随后 ``arena.barrier()``；最后清零本 rank
        reduce_buf 中被消费的槽（``experts_ptr[rank, b] >= 0`` 的 b）。

        ``reduce_buf_ptr`` 必须为已在 arena 注册的本 rank 对称张量
        （[B,H,H'] fp32，块 = H*H' 元素），pull_all 按 data_ptr 解析注册身份。
        ``meta_ptr`` / ``bar_ptr`` / ``barrier_off`` / ``stream`` 为 GPU 专属
        形参，签名保留、参考实现不使用——bar/meta 的 BARRIER 区语义由
        ``arena.barrier()`` 承担。
        """
        E, H, Hp, R, B = self.E, self.H, self.Hp, self.R, self.B
        epn = E // R
        # dtype 契约：梯度/fp32、meta/int32。
        # expert_grad_ptr 为本 rank 的**属主局部梯度表** [epn, H, H']（仅本 rank
        # 属主专家行，按局部行号 le = e - rank*epn 索引）——对比全局表 [E, ...]
        # 每 rank 省 (R-1)*epn 行死空间。
        assert expert_grad_ptr.dtype == torch.float32 \
            and tuple(expert_grad_ptr.shape) == (epn, H, Hp), \
            f"expert_grad_ptr 必须为 fp32 [epn, H, H']=[{epn}, {H}, {Hp}]，" \
            f"got {expert_grad_ptr.dtype} {tuple(expert_grad_ptr.shape)}"
        assert reduce_buf_ptr.dtype == torch.float32 \
            and tuple(reduce_buf_ptr.shape) == (B, H, Hp), \
            f"reduce_buf_ptr 必须为 fp32 [B, H, H']=[{B}, {H}, {Hp}]（本 rank 物理分片），" \
            f"got {reduce_buf_ptr.dtype} {tuple(reduce_buf_ptr.shape)}"
        assert experts_ptr.dtype == torch.int32 \
            and tuple(experts_ptr.shape) == (R, B), \
            f"experts_ptr 必须为 int32 [R, B]=[{R}, {B}]，" \
            f"got {experts_ptr.dtype} {tuple(experts_ptr.shape)}"
        assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
        assert isinstance(rank, int) and 0 <= rank < R, \
            f"rank must be in [0, {R}), got {rank}"
        assert self.arena.world == R, \
            f"arena.world ({self.arena.world}) 与 R ({R}) 不一致"

        lo, hi = rank * epn, (rank + 1) * epn  # 本 rank 属主专家区间 [lo, hi)

        # ---- 预扫描（prescan）：rb 升序逐槽匹配本地属主专家 ----
        # rb = r*B+b，故 rb 升序即 (r,b) 字典序，与 GPU prescan 的 slist 顺序
        # 逐位一致。展平块号即 rb（chunk 步长 = B 块/rank，对应 GPU 的
        # 地址 base + r*(B*H*Hp) + b*(H*Hp)）。
        acc_lists = [[] for _ in range(epn)]  # le -> [rb, ...]，rb 升序
        flats = []                            # 展平块号（rb 升序）
        pos_in_pull = {}                      # rb -> 该槽在应答里的序号
        for rb in range(R * B):
            r, b = divmod(rb, B)
            # 与 GPU prescan 逐位一致：仅按专家 id 区间匹配，不排除自发槽
            e = int(experts_ptr[r, b])
            if lo <= e < hi:
                acc_lists[e - lo].append(rb)
                pos_in_pull[rb] = len(flats)
                flats.append(rb)

        # ---- 集合拉取：一次 pull_all 取回全部相关槽 ----
        # 集合调用：全组须在同一 API 阶段内各调一次；无匹配槽的 rank 也必须参与。
        pulled = self.arena.pull_all(
            reduce_buf_ptr,
            torch.tensor(flats, dtype=torch.int64,
                         device=reduce_buf_ptr.device),
        )

        # ---- 累加：属主专家升序、(r,b) 字典序 fp32 累加（本地 wgrad 为种子）----
        # 应答序 == 请求序（与 flats 一一对应）；目标行为属主局部行号 le。
        for le in range(epn):
            if not acc_lists[le]:
                continue  # 无远程槽的属主专家零开销
            for rb in acc_lists[le]:
                block = pulled[pos_in_pull[rb]]  # [H*Hp] fp32
                expert_grad_ptr[le].add_(block.view(H, Hp))

        # ---- 跨 rank 屏障：全组读完本 rank 的槽之后才允许清零 ----
        # （GPU 版 cross_rank_barrier 的 meta/bar 槽位协议由 arena.barrier() 承担）
        self.arena.barrier()

        # ---- 本地清零被消费的槽（experts_ptr[rank, b] >= 0 的 b）----
        for b in range(B):
            if int(experts_ptr[rank, b]) >= 0:
                reduce_buf_ptr[b].zero_()


def launch_grad_reduce(
    remote_expert_grads: torch.Tensor,
    remote_reduce_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
) -> None:
    """Launch remote expert grad reduction（torch 语义级参考实现）。

    签名与 GPU 源码 moonep/grad_reduce.py:437 逐字一致（无 ctx）；arena 经全局
    注册表 ``resolve_arena_for(remote_reduce_buffers)`` 获取（契约 §3：无 ctx
    的 launch_* 取 arena 的通道），内部构造本包 GradReduceKernel 并调用其
    torch 体。

    Args:
        remote_expert_grads: contiguous fp32 tensor shaped [epn, H, H']
            （本 rank 属主局部梯度表，epn = E//R）。 The current rank owns
            expert ids ``rank * (E // R) : (rank + 1) * (E // R)``; only that
            range is updated（按局部行号 ``e - rank*epn`` 索引）。
        remote_reduce_buffers: GPU 版为 NVL 对称堆上的 contiguous fp32
            [R, B, H, H'] 全组视图；参考实现为本 rank 已注册的 contiguous
            fp32 [B, H, H'] 物理分片（块 = H*H' 元素）。Slots whose
            ``experts_to_copy[r, b]`` belongs to this rank's owner range are
            accumulated into ``remote_expert_grads``; afterwards a cross-rank
            barrier fences peers and each rank clears its own consumed slots
            locally.
        experts_to_copy: contiguous int32 tensor shaped [R, B]。
        rank: current EP rank。
        num_sms: number of persistent CTAs to launch。GPU 专属配置，torch 体
            不消费，仅为保持构造契约而透传。
        meta_buf: int32 NVL-distributed meta buffer holding the barrier slots。
            GPU 专属形参，签名保留、参考实现不使用（屏障语义由
            ``arena.barrier()`` 承担）。
        meta_stride: per-rank stride (meta_chunk_padded) into ``meta_buf``。
            仅作为 kernel 构造配置透传，torch 体不消费。
        barrier_off: offset of the barrier slots within each rank's chunk。
            GPU 专属形参，签名保留、参考实现不使用。
        grid_sync_bar: int32 [1] grid-barrier counter。GPU 专属形参，签名
            保留、参考实现不使用。

    The caller must ensure all ranks have finished writing
    ``remote_reduce_buffers`` before launching this kernel on any rank.
    """
    if remote_reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    assert remote_expert_grads.dtype == torch.float32 and remote_expert_grads.is_contiguous(), \
        "remote_expert_grads must be contiguous fp32 [epn, H, H']（本 rank 局部梯度表）"
    assert remote_reduce_buffers.dtype == torch.float32 and remote_reduce_buffers.is_contiguous(), \
        "remote_reduce_buffers must be contiguous fp32 [B, H, H']（本 rank 物理分片）"
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous(), \
        "experts_to_copy must be contiguous int32 [R, B]"
    assert remote_expert_grads.ndim == 3, \
        f"remote_expert_grads must have rank 3, got shape={tuple(remote_expert_grads.shape)}"
    assert remote_reduce_buffers.ndim == 3, \
        f"remote_reduce_buffers must have rank 3 (本 rank 物理分片), got shape={tuple(remote_reduce_buffers.shape)}"
    assert experts_to_copy.ndim == 2, \
        f"experts_to_copy must have rank 2, got shape={tuple(experts_to_copy.shape)}"

    rows, H, Hp = (int(x) for x in remote_expert_grads.shape)
    R, B = (int(x) for x in experts_to_copy.shape)
    buf_B, buf_H, buf_Hp = (int(x) for x in remote_reduce_buffers.shape)
    assert buf_B == B and buf_H == H and buf_Hp == Hp, (
        f"remote_reduce_buffers shape {tuple(remote_reduce_buffers.shape)} "
        f"incompatible with remote_expert_grads {tuple(remote_expert_grads.shape)} "
        f"and experts_to_copy {tuple(experts_to_copy.shape)}"
    )
    # remote_expert_grads 为本 rank 属主局部表 [epn, H, H']（局部寻址），
    # 由 R 反解全局 E = epn*R。
    epn = rows
    E = epn * R
    assert E % R == 0, f"E ({E}) must be divisible by R ({R})"
    assert 0 <= int(rank) < R, f"rank must be in [0, {R}), got {rank}"
    # 注：GPU 版要求 H/Hp 为 128 的倍数（2D TMA tile 约束），参考实现的 torch 体
    # 无此约束（支持任意 H/Hp，测试规格 H=32/H'=16），故不保留该断言；
    # CUDA device 断言同属 GPU 专属（参考实现纯 torch、CPU 可跑），一并略去。
    assert isinstance(num_sms, int) and num_sms > 0, \
        f"num_sms must be a positive int, got {num_sms}"

    # 无 ctx：按 remote_reduce_buffers 的 data_ptr 在全局注册表中解析所属 arena
    # （惰性导入：resolve_arena_for 由 buffer.py 提供，避免模块加载期的硬依赖）
    from .buffer import resolve_arena_for

    arena = resolve_arena_for(remote_reduce_buffers)

    kernel = GradReduceKernel(
        E=E,
        H=H,
        Hp=Hp,
        R=R,
        B=B,
        meta_stride=int(meta_stride),
        num_sms=int(num_sms),
        smem_budget=_SMEM_BUDGET_BYTES,  # torch 体不消费 smem，仅保持构造契约
        arena=arena,
    )
    # __call__ 的 GPU 专属形参 meta_buf / grid_sync_bar / barrier_off / stream
    # 按签名位次原样透传（torch 体不使用）
    kernel(
        remote_expert_grads,    # expert_grad_ptr
        remote_reduce_buffers,  # reduce_buf_ptr（本 rank [B,H,H'] 物理分片）
        experts_to_copy,        # experts_ptr
        meta_buf,               # meta_ptr（保留不使用）
        grid_sync_bar,          # bar_ptr（保留不使用）
        int(rank),
        int(barrier_off),
        None,                   # stream（保留不使用）
    )
