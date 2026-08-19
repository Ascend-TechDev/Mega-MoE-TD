# coding=utf-8
"""MoonEP PlanningKernel 的 torch 语义参考实现（昇腾路径，rank 差异化结构）。

语义基准：`source_code/MoonEP/moonep/planning.py` 的 Phase A/B/C/D（下文逐条
标注行号）。本实现**保留源码的 rank 差异化结构**——rank0 集中规划、rank1
代算 rank0 的 C1、其他 rank 走普通路径；跨 rank 数据流动全部经对称 meta
内存（arena.push_all/pull_all），不使用 allgather 类集合通信。

与 GPU 版的唯一结构偏差（契约 §0）：multimem.st 多播广播在昇腾无等价物，
Phase B.5 的规划表分发改为 **rank0 经 ``arena.push_all`` 逐 peer 推送**
（per-peer put 循环，内容等价）；Phase D 各 rank 从 rank0 chunk 拉取
CU/ZFR/ETC/STATS 切片的语义与源码一致（1030-1047）。

执行结构（三次 arena.barrier() 对应源码的三处 cross_rank_barrier）：

- Phase A（601-609）：全组把本地 tpe[E] 推入 **rank0 chunk** 的 TPE 区第
  rank 行；R>1 时 rank0 额外把自己的 topk/tpe 推入 **rank1 chunk** 的
  TOPK0/TPE 区（C1 外包输入）。→ barrier #1；
- Phase B（610-960，**仅 rank0**）：从自己 chunk 的 TPE 汇聚区读出 [R,E]，
  串行计算 tpe_cumsum → 单源填充 z → alloc/alloc_cumsum → top-B → VM 布局，
  把 3ER 广播段（ALLOC/TPE/EOFF）+ CU/ZFR/ETC/STATS 写回自己 chunk 的 PLAN
  区，再把 3ER 表 push 到每个 peer 的 PLAN 区（B.5 替代）；
- Phase C.1（398-517，**rank≠0**）：各 rank 本地计数排序 topk→order 写入
  自己 chunk 的 ORDER 区；**rank1 额外**从 TOPK0 区取 rank0 的 topk 代算
  C1，结果写入自己的 ORDER0 区后推送回 rank0 的 ORDER 区（961-970 语义）。
  rank0 不跑 C1（忙于 Phase B）。R=1 时 rank0 自跑 C1。→ barrier #2；
- Phase C.2（1048-1077，全组）：读自己 chunk 的 ORDER 区与 3ER 表，逐排序
  位计算 dst，并把槽位出处**远端写**进目的 rank 的 SRC_INFO 区。→ barrier #3；
- Phase D（1030-1047 + 1079-1113，全组）：从 rank0 chunk 拉取本 rank 的
  CU/ZFR/STATS 切片与全组 ETC 表（写输出张量），再逐 token 扫 K 个 dst 做
  dedup 负编码（同一目的 rank 第 2 次及以后出现 → dst=-raw-1）。

绑定级 I/O 契约（契约 §4）：

- ``PlanningKernel.__init__`` 形参与源码 planning.py:353-356 逐字一致，仅追加
  关键字形参 ``arena``（唯一签名扩展点）；
- ``PlanningKernel.__call__`` 形参与源码 planning.py:366-368 逐字一致
  （``cute.Pointer`` → 同名 torch.Tensor，``rank: Int32`` → int，``stream``
  形参保留、接受 None、不使用）；
- ``mc``/``alloc``/``group_tokens``/``z``/``local_hist``/``bar`` 为 GPU 专属
  形参，签名保留、参考实现不使用（详见 ``__call__`` docstring）；
- ``launch_planning(ctx, topk_experts_flat, tokens_per_expert, cu_seqlens, plan)``
  签名与源码 planning.py:1294-1300 逐字一致；内部取 ``ctx['arena']`` 构造
  PlanningKernel 并调用，返回前把 ``src_info`` 并入 plan 扩展字段；
- ``allocate_planning_outputs`` / ``physical_tokens_per_expert`` 为契约 §4
  钉死的宿主侧辅助函数。

逐条对齐的源码语义（planning.py 行号）：

- Phase B.0（610-670 行）：tpe_cumsum[R,E] 为 tpe 沿 rank 维的**inclusive**
  前缀和（C2 处按 ``tpe_cumsum[rank-1]`` 取用，效果即"不含本行"）；
  group_tokens[h] 为 h 组专家的全组 token 总数。
- Phase B.1（671-702 行）：单源填充迁移矩阵 z[R,R]。balance=group_tokens-CAP
  （CAP=NvS_capacity=S*K，api.py:268）；循环取盈余 rank（argmax，平局取小下
  标）与亏空 rank（argmin，平局取小下标），move=-deficit 一次性把亏空方补
  到 CAP；z[s,d]=s 迁给 d 的 token 数。性质：每个目的 rank 的远程 token 只
  来自单一 home 组。
- Phase B.2（703-798 行）：按属主 rank 贪心切分专家 token 配额得 alloc[d,e]
  （每目的 rank 行和不超过 CAP）：每轮取配额最大的目的 rank（argmax 平局取
  小下标）与剩余 token 最多的本地专家（argmax 平局取小下标），
  take=min(quota, remaining)。alloc_cumsum[e,r] 为 alloc 沿 rank 的
  inclusive 前缀（C2 二分用）。
- Phase B.3（834-885 行）：对每个目的 rank，将其远程来源专家（非本地且
  alloc>0）做 B 轮 argmax 选出 top-B 写入 experts_to_copy——**平局取大下
  标**（reg_scan_argmax_max_idx 的 >= 语义），选中条目清零再选下一轮，不
  足 B 个写 -1。remote_stats[d]=[d 的远程非零专家数, 全组选中 d 所属专家的
  副本槽数]。
- Phase B.4（887-953 行）：VM 布局。段序 = 组号 0..E+B-1：0..E-1 为按全局
  专家号的常驻段（被选中为副本槽的专家其常驻段为空，token 落到槽段），
  E..E+B-1 为副本槽段（槽序 0..B-1）；非空段按 token_padding 向上对齐，空
  段不占行（expert_offsets 保持 0，cu_seqlens 记当前 offset，zero_fill 记
  (0,0)）。得 expert_offsets[R,E] / cu_seqlens[R,E+B] /
  zero_fill_ranges[R,E+B,2]。
- Phase C.1（398-517 行）：计数排序 order[N]——topk 按专家 id 稳定排序
  （同专家内保持原 offv 升序；计数排序天然稳定，torch.stable sort 等价）。
- Phase C.2（1048-1077 行）：对排序位 idx：offv=order[idx]，e=topk[offv]，
  prev=tpe_cumsum[rank-1,e]（rank=0 时 0），gidx=prev+(idx-expoff[e])
  （expoff=本地 tpe 的 exclusive 前缀）；在 alloc_cumsum[e,:] 上取第一个
  cumsum>gidx 的目的 rank lo（等价内核定步长二分），
  pc=alloc_cumsum[e,lo-1]（lo=0 时 0），loff=expert_offsets[lo,e]+(gidx-pc)；
  dst[offv]=lo*NvS+loff，并在目的 rank 的 loff 槽位记
  src_info=rank*NvS+offv（-1 为空槽哨兵）。
- Phase D（1079-1113 行）：逐 token 扫 K 个 dst，同一目的 rank 第 2 次及以
  后出现 → dst=-raw-1（该条目只散布 route weight，不搬 payload）。

边界说明（与源码逐位一致）：R=1 时 z=0、无远程专家（experts_to_copy 全
-1、槽段全空、remote_stats=[0,0]），rank0 自跑 C1；Phase D 去重编码**仍然
生效**——K>1 时同一 token 的 K 个条目全部落在 rank 0，第 2 次起 dst 取负。
这与 GPU 内核行为一致（Phase D 不做 R>1 分支）。

dtype 契约：全部对外张量为 int32 连续；内部统一 int64 计算以保证确定性并避
免中间溢出。
"""

from dataclasses import dataclass

import torch

__all__ = [
    "MoonEPCommPlan",
    "PlanningKernel",
    "allocate_planning_outputs",
    "launch_planning",
    "physical_tokens_per_expert",
]

_INT32_MAX = 2**31 - 1

# meta 的 BARRIER 区占 3 槽（源码 api.py: cross_rank_barrier 的 2 相信号 +
# 1 相/符号计数），SRC_INFO 区紧随其后（源码 planning.py:544-545）。
_BARRIER_SLOTS = 3

# primary_packed/kmask 编码位宽（逐字对齐 GPU moonep/constants.py；
# 参考实现不建 constants 模块，仅 _check_dedup_encoding_bounds 使用）。
KIDX_BITS = 7


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    """一次 dispatch/combine 往返的通信规划（契约 §2，字段序与之一致）。

    除整数字段外全部为 int32 连续张量：

    - dst [N]：本 rank 每个 topk 条目的目的编码 lo*NvS+loff；负值 = -raw-1，
      表示该条目为 dup（只散布 weight，不搬 payload）。
    - experts_to_copy [R,B]：全组各 rank 选中的副本专家（-1 空槽）。
    - zero_fill_ranges [E+B,2]：本 rank 各 VM 段 padding 区间 [start,count)。
    - remote_stats [2]：本 rank 的 [远程非零专家数, 本 rank 专家被选为副本的
      槽数]。
    - dup_groups [NvS,3]：(primary_loff, dup_start, dup_count)，仅前
      dup_counts[0] 行有效；由 DispatchKernel(build_dedup_map=True) 物化，
      组内 dup 按 kidx 升序（与 GPU 的 ctz(kmask) 发射序一致）。
    - dup_loffs [NvS]：dup 槽位紧凑表，仅前 dup_counts[1] 项有效。
    - dup_counts [2]：[n_groups, n_dup_loffs]。
    - src_info [NvS]：本 rank 各槽位的出处 src_rank*NvS+offv（-1 空槽/填充
      行；参考实现扩展字段，launch_planning 返回前从 meta SRC_INFO 区拷入）。

    注：cu_seqlens 不是 plan 字段——allocate_planning_outputs 以
    (plan, cu_seqlens) 二元组返回（对齐源码同名函数）。
    """

    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int
    # —— 参考实现扩展 ——
    src_info: torch.Tensor

    def __post_init__(self) -> None:
        N, R, E, B, NvS = (
            int(self.N),
            int(self.R),
            int(self.E),
            int(self.B),
            int(self.NvS),
        )

        def _chk(t: torch.Tensor, shape: tuple, name: str) -> None:
            assert t.dtype == torch.int32 and t.is_contiguous(), (
                f"plan.{name} 须为 int32 连续张量，got dtype={t.dtype}"
            )
            assert tuple(t.shape) == shape, (
                f"plan.{name} 形状应为 {shape}，got {tuple(t.shape)}"
            )

        _chk(self.dst, (N,), "dst")
        _chk(self.experts_to_copy, (R, B), "experts_to_copy")
        _chk(self.zero_fill_ranges, (E + B, 2), "zero_fill_ranges")
        _chk(self.remote_stats, (2,), "remote_stats")
        _chk(self.dup_groups, (NvS, 3), "dup_groups")
        _chk(self.dup_loffs, (NvS,), "dup_loffs")
        _chk(self.dup_counts, (2,), "dup_counts")
        _chk(self.src_info, (NvS,), "src_info")

    def clone(self) -> "MoonEPCommPlan":
        """深拷贝全部张量字段（plan 复用路径的防御性拷贝用）。"""
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N,
            R=self.R,
            E=self.E,
            B=self.B,
            NvS=self.NvS,
            K=self.K,
            src_info=self.src_info.clone(),
        )


# ============================================================
# 分阶段纯函数（内部统一 int64；同一输入下结果逐位确定）
# ============================================================
def _phase_b_tables(tpe_all: torch.Tensor,   # [R,E] int64（rank0 chunk 的 TPE 汇聚区）
                    *, R: int, E: int, B: int, NvS: int, NvS_capacity: int,
                    token_padding: int) -> dict:
    """Phase B.0-B.4（610-953）：仅 rank0 执行的规划表计算。

    输入为全组 tpe 汇聚表 [R,E]（rank0 从自己 chunk 的 TPE 区读回）。
    返回 3ER 广播段（alloc_cumsum/tpe_cumsum/expert_offsets）与
    CU/ZFR/ETC/STATS 全组表，全部 int64。
    """
    assert tuple(tpe_all.shape) == (R, E), (
        f"tpe_all 形状应为 ({R},{E})，got {tuple(tpe_all.shape)}"
    )
    assert E % R == 0, f"E({E}) 必须整除 R({R})"
    assert B > 0, f"B 必须为正，got {B}"
    assert token_padding > 0, f"token_padding 必须为正，got {token_padding}"
    assert R * NvS <= _INT32_MAX, (
        f"dst/src_info 线性编码要求 R*NvS <= int32_max，got R={R} NvS={NvS}"
    )

    epn = E // R
    CAP = NvS_capacity           # 源码 CAP = NvS_capacity（= S*K，api.py:268）
    tp = token_padding
    dev = tpe_all.device
    tpe_i64 = tpe_all.to(torch.int64)

    # ---------- Phase B.0（610-670）：tpe_cumsum / expert_count / group_tokens ----------
    tpe_cumsum = tpe_i64.cumsum(dim=0)                     # [R,E] inclusive 前缀
    expert_count = tpe_cumsum[R - 1]                       # [E] 各专家全组 token 总数
    group_tokens = expert_count.view(R, epn).sum(dim=1)    # [R] 各 home 组 token 总数

    # ---------- Phase B.1（671-702）：单源填充 z[R,R] ----------
    # 盈余取 argmax（平局小下标），亏空取 argmin（平局小下标）；
    # 终止条件与内核一致：surplus<=0 或 deficit>=0；move 一次性补到 CAP。
    bal = [int(group_tokens[r]) - CAP for r in range(R)]
    z = [[0] * R for _ in range(R)]
    while True:
        surplus = max(bal)
        deficit = min(bal)
        if surplus <= 0 or deficit >= 0:
            break
        s_rank = bal.index(surplus)            # 第一个最大值（平局取小下标）
        d_rank = bal.index(deficit)            # 第一个最小值（平局取小下标）
        move = -deficit                        # 一次性把亏空方补回 CAP
        z[s_rank][d_rank] = move
        bal[s_rank] -= move
        bal[d_rank] = 0

    # ---------- Phase B.2（703-798）：alloc[R,E] 与 alloc_cumsum[E,R] ----------
    alloc_l = [[0] * E for _ in range(R)]      # alloc[d,e]
    for h in range(R):
        e0 = h * epn
        quotas = z[h][:]                       # quotas[d] = h 迁给 d 的配额
        remaining = [int(expert_count[e0 + le]) for le in range(epn)]
        acc = [[0] * epn for _ in range(R)]
        for le in range(epn):
            acc[h][le] = remaining[le]         # 属主初始持有全部 token
        while True:
            max_quota = max(quotas)
            if max_quota <= 0:
                break
            d = quotas.index(max_quota)        # argmax 平局取小 rank
            max_rem = max(remaining)
            if max_rem <= 0:
                break
            le = remaining.index(max_rem)      # argmax 平局取小本地专家号
            take = max_rem if max_rem < max_quota else max_quota
            quotas[d] = max_quota - take
            remaining[le] = max_rem - take
            acc[d][le] += take
            acc[h][le] = max_rem - take
        for d in range(R):
            alloc_l[d][e0:e0 + epn] = acc[d]
    alloc = torch.tensor(alloc_l, dtype=torch.int64, device=dev)       # [R,E]

    # 守恒自检：每个专家的 token 总量守恒；每个目的 rank 的配额不超过 CAP。
    assert bool((alloc.sum(dim=0) == expert_count).all()), "逐专家 token 守恒被破坏"
    assert bool((alloc.sum(dim=1) <= CAP).all()), "目的 rank 配额超过 CAP"

    alloc_cumsum = alloc.cumsum(dim=0).t().contiguous()                # [E,R] inclusive

    # ---------- Phase B.3（834-885）：experts_to_copy[R,B] 与 remote_stats[R,2] ----------
    # 对目的 rank 的远程非零专家做 B 轮 argmax，**平局取大下标**
    # （reg_scan_argmax_max_idx 的 >= 语义）；全 0 时该槽写 -1。
    etc_l = [[-1] * B for _ in range(R)]
    stats_l = [[0, 0] for _ in range(R)]
    sel_mask_l = [[False] * E for _ in range(R)]   # 被选中为副本槽的专家
    for d in range(R):
        lo_e, hi_e = d * epn, (d + 1) * epn
        rc = [0 if lo_e <= e < hi_e else int(alloc[d, e]) for e in range(E)]
        stats_l[d][0] = sum(1 for v in rc if v > 0)
        for b in range(B):
            best_cnt = max(rc)
            if best_cnt <= 0:
                break                             # 其余槽位保持 -1
            best_idx = max(e for e in range(E) if rc[e] == best_cnt)  # 平局取大下标
            etc_l[d][b] = best_idx
            sel_mask_l[d][best_idx] = True
            stats_l[best_idx // epn][1] += 1
            rc[best_idx] = 0
    experts_to_copy = torch.tensor(etc_l, dtype=torch.int64, device=dev)      # [R,B]

    # ---------- Phase B.4（887-953）：VM 布局 ----------
    eoff_l = [[0] * E for _ in range(R)]
    cu_l = [[0] * (E + B) for _ in range(R)]
    zfr_l = [[[0, 0] for _ in range(E + B)] for _ in range(R)]
    for d in range(R):
        start = 0
        for g in range(E + B):
            cnt = 0
            eid = -1
            if g < E:
                # 常驻段：未被选为副本槽的专家按全局专家号占位
                if not sel_mask_l[d][g]:
                    cnt = int(alloc[d, g])
                    eid = g
            else:
                # 副本槽段：槽内专家的 token 整体落到槽段
                se = etc_l[d][g - E]
                if se >= 0:
                    eid = se
                    cnt = int(alloc[d, se])
            padded = ((cnt + tp - 1) // tp) * tp if cnt > 0 else 0
            cu_l[d][g] = start + padded
            if cnt > 0:
                eoff_l[d][eid] = start
                if padded > cnt:
                    zfr_l[d][g][0] = start + cnt      # [pad_start, pad_count)
                    zfr_l[d][g][1] = padded - cnt
            start += padded
        assert start <= NvS, f"rank {d} 的 padded 布局超过 NvS：{start} > {NvS}"
    return {
        "tpe_cumsum": tpe_cumsum,          # [R,E]
        "alloc_cumsum": alloc_cumsum,      # [E,R]
        "expert_offsets": torch.tensor(eoff_l, dtype=torch.int64, device=dev),  # [R,E]
        "cu_all": torch.tensor(cu_l, dtype=torch.int64, device=dev),            # [R,E+B]
        "zfr_all": torch.tensor(zfr_l, dtype=torch.int64, device=dev),          # [R,E+B,2]
        "etc_all": experts_to_copy,        # [R,B]
        "stats_all": torch.tensor(stats_l, dtype=torch.int64, device=dev),      # [R,2]
    }


def _phase_c1_order(topk_flat: torch.Tensor) -> torch.Tensor:
    """Phase C.1（398-517）：计数排序——topk 按专家 id 稳定排序，返回 order [N]。

    计数排序天然稳定（同专家内保持原 offv 升序），torch 稳定排序等价。
    """
    return torch.sort(topk_flat.to(torch.int64), stable=True).indices


def _phase_c2_rank(order: torch.Tensor,          # [N] int64 本 rank 排序结果
                   topk_flat: torch.Tensor,      # [N] 本 rank topk
                   tpe_local: torch.Tensor,      # [E] 本 rank tpe
                   tpe_cumsum: torch.Tensor,     # [R,E] int64（3ER 表，来自 meta）
                   alloc_cumsum: torch.Tensor,   # [E,R] int64
                   expert_offsets: torch.Tensor, # [R,E] int64
                   rank: int, *, R: int, E: int, S: int, K: int, NvS: int):
    """Phase C.2（1048-1077）：本 rank 的 dst 与 src_info 远端发布。

    返回 (dst_row int64 [N], src_writes {dest_rank: (loffs int64, vals int64)})；
    本阶段 dst 全部非负（dedup 负编码在 Phase D）。
    """
    N = S * K
    dev = order.device
    topk_i64 = topk_flat.to(torch.int64)
    tpe_i64 = tpe_local.to(torch.int64)
    arange_n = torch.arange(N, dtype=torch.int64, device=dev)

    e_sorted = topk_i64[order]                                             # 排序位上的专家
    expoff = torch.zeros(E, dtype=torch.int64, device=dev)
    expoff[1:] = tpe_i64.cumsum(dim=0)[:-1]                                # 本地 exclusive 前缀
    zero_e = torch.zeros_like(expoff)
    prev_row = tpe_cumsum[rank - 1] if rank > 0 else zero_e                # 低 rank 同专家 token 数
    gidx = prev_row[e_sorted] + (arange_n - expoff[e_sorted])              # 全组专家序
    rows = alloc_cumsum[e_sorted]                                          # [N,R]
    lo = torch.searchsorted(rows, gidx.unsqueeze(1), right=True).squeeze(1)
    assert bool((lo < R).all()), "alloc_cumsum 二分未找到目的 rank"
    pc = torch.zeros(N, dtype=torch.int64, device=dev)
    pos = lo > 0
    pc[pos] = rows[pos].gather(1, (lo[pos] - 1).unsqueeze(1)).squeeze(1)
    loff = expert_offsets[lo, e_sorted] + (gidx - pc)                      # 段基址 + 段内偏移

    dst_row = torch.zeros(N, dtype=torch.int64, device=dev)
    dst_row[order] = lo * NvS + loff
    # src_info 远端发布：槽 (lo, loff) ← rank*NvS+offv，按目的 rank 分组
    src_writes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    vals = rank * NvS + order
    for dr in range(R):
        m = lo == dr
        if bool(m.any()):
            src_writes[dr] = (loff[m].contiguous(), vals[m].contiguous())
    return dst_row, src_writes


def _phase_d_dedup(dst_row: torch.Tensor, *, S: int, K: int, R: int, NvS: int) -> torch.Tensor:
    """Phase D（1079-1113）：逐 token 扫 K 个 dst，同一目的 rank 第 2 次及以后
    出现编码为 -raw-1。R=1 时同样生效（全部条目落在 rank 0）。"""
    raw = dst_row.to(torch.int64).view(S, K).clone()
    dest = torch.div(raw, NvS, rounding_mode="floor")   # 此刻 raw 全部非负
    seen = torch.zeros(S, R, dtype=torch.bool, device=raw.device)
    for k in range(K):
        d_k = dest[:, k:k + 1]                                           # [S,1]
        dup_k = seen.gather(1, d_k).squeeze(1)                           # 目的 rank 已出现？
        v_k = raw[:, k]
        raw[:, k] = torch.where(dup_k, -v_k - 1, v_k)
        seen.scatter_(1, d_k, True)
    return raw.reshape(-1)


class PlanningKernel:
    """规划 kernel 的 torch 参考实现（契约 §4；类名/签名与源码逐字一致）。

    ``__init__`` 仅比源码多一个关键字形参 ``arena``（唯一签名扩展点，契约 §0）。
    ``num_vblocks``/``num_sms`` 为 GPU 侧 C1 计数排序/网格规模的实现细节，
    torch 体不使用（属性照存以保契约）；``meta_stride``/``BARRIER_OFF``/
    ``TOPK0_OFF``/``ORDER_OFF``/``ORDER0_OFF`` 用于 meta 各区的偏移推算；
    ``NvS_capacity`` 即单源填充的 CAP。
    """

    def __init__(self, R, E, B, S, K, NvS_capacity, NvS, num_vblocks, meta_stride,
                 TPE_OFF, PLAN_OFF, BARRIER_OFF, TOPK0_OFF, ORDER_OFF, ORDER0_OFF,
                 token_padding, num_sms, *, arena):
        self.R, self.E, self.B, self.S, self.K = R, E, B, S, K
        self.N = self.S * self.K
        self.NvS_capacity, self.NvS, self.num_vblocks = NvS_capacity, NvS, num_vblocks
        self.meta_stride = meta_stride
        self.TPE_OFF, self.PLAN_OFF, self.BARRIER_OFF = TPE_OFF, PLAN_OFF, BARRIER_OFF
        self.TOPK0_OFF, self.ORDER_OFF, self.ORDER0_OFF = TOPK0_OFF, ORDER_OFF, ORDER0_OFF
        self.token_padding, self.num_sms = token_padding, num_sms
        # —— 参考实现扩展（唯一签名扩展点）——
        self.arena = arena

    def __call__(self, tpe, topk, meta, mc, dst, cu_seqlens,
                 experts_to_copy, zero_fill, remote_stats, alloc, group_tokens, z,
                 local_hist, bar, rank: int, stream):
        """torch 体：rank 差异化的 Phase A-D（结构对齐源码，见模块 docstring）。

        形参与源码 planning.py:366-368 逐字一致（cute.Pointer → 同名
        torch.Tensor）：

        - tpe [E] int32：本 rank tokens_per_expert；
        - topk [N] int32：本 rank topk 专家号（S*K 展平）；
        - meta [meta_stride] int32：本 rank 对称 meta chunk（已在 arena 注册，
          块 = 1 元素）；跨 rank 读写经 arena.push_all/pull_all；
        - mc：GPU 多播（multicast）指针，昇腾无等价物，形参保留不使用
          （B.5 的 3ER 表分发由 rank0 经 arena.push_all 逐 peer 推送替代）；
        - dst [N] int32 / cu_seqlens [E+B] int32 / experts_to_copy [R,B] int32 /
          zero_fill [E+B,2] int32 / remote_stats [2] int32：输出，原地写；
        - alloc/group_tokens/z/local_hist：GPU 版 rank0 的全局 scratch，torch
          体为纯局部中间量，形参保留不使用；
        - bar：GPU 软件 grid barrier 计数器，形参保留不使用（同步经
          arena.barrier()，对应源码三处 cross_rank_barrier）；
        - rank：本 rank 号（须与 arena.rank 一致）；
        - stream：CUDA stream，形参保留、接受 None、不使用。
        """
        R, E, B, S, K, NvS = self.R, self.E, self.B, self.S, self.K, self.NvS
        N = self.N
        arena = self.arena
        rank = int(rank)
        assert 0 <= rank < R, f"rank({rank}) 越界 [0,{R})"
        assert int(arena.rank) == rank, (
            f"rank 形参({rank}) 与 arena.rank({arena.rank}) 不一致"
        )
        assert int(arena.world_size) == R, (
            f"arena.world_size({arena.world_size}) 与 R({R}) 不一致"
        )

        # ---- 入参校验（dtype 契约：meta/tpe/topk 与全部输出均为 int32）----
        def _chk_i32(t: torch.Tensor, numel: int, name: str) -> None:
            assert isinstance(t, torch.Tensor) and t.dtype == torch.int32, (
                f"{name} 须为 int32 张量，got {getattr(t, 'dtype', None)}"
            )
            assert t.numel() == numel, (
                f"{name} 元素数应为 {numel}，got {t.numel()}"
            )

        _chk_i32(tpe, E, "tpe")
        _chk_i32(topk, N, "topk")
        _chk_i32(dst, N, "dst")
        _chk_i32(cu_seqlens, E + B, "cu_seqlens")
        _chk_i32(experts_to_copy, R * B, "experts_to_copy")
        _chk_i32(zero_fill, 2 * (E + B), "zero_fill")
        _chk_i32(remote_stats, 2, "remote_stats")
        assert meta.dim() == 1 and meta.dtype == torch.int32, (
            f"meta 须为一维 int32（本 rank chunk），got {tuple(meta.shape)} {meta.dtype}"
        )
        src_info_off = self.BARRIER_OFF + _BARRIER_SLOTS
        assert meta.numel() >= src_info_off + NvS, (
            f"meta chunk 过小：numel={meta.numel()} < SRC_INFO_OFF+NvS="
            f"{src_info_off + NvS}"
        )
        dev = meta.device

        # PLAN 子区偏移（源码 546-552）
        pb = self.PLAN_OFF
        alloc_sub = 0
        tpe_sub = E * R
        eoff_sub = 2 * E * R
        cu_sub = 3 * E * R
        zfr_sub = cu_sub + R * (E + B)
        etc_sub = zfr_sub + 2 * R * (E + B)
        stats_sub = etc_sub + R * B

        def _blocks(off: int, n: int) -> torch.Tensor:
            return off + torch.arange(n, dtype=torch.int64, device=dev)

        def _put(off: int, t: torch.Tensor, name: str) -> None:
            v = t.to(torch.int32).reshape(-1)
            assert off + v.numel() <= meta.numel(), (
                f"meta 直写越界：{name} 区 [{off}, {off + v.numel()}) 超出 "
                f"numel={meta.numel()}"
            )
            meta[off:off + v.numel()] = v

        # ================================================================
        # Phase A（601-609）：tpe 汇聚（全组 → rank0 chunk 的 TPE 区）
        # ================================================================
        # 展平块号寻址（对齐源码的地址公式 base + 目标rank*ms + 区内偏移）：
        # flat = pe*ms + off，ms = meta_stride（meta chunk 的块数，块=1 元素）。
        ms = int(self.meta_stride)
        # SRC_INFO 区 -1 初始化（空槽哨兵）：本轮 C.2 的远端发布只覆盖已分配
        # 槽，未覆盖的空槽必须保持 -1 供 dispatch builder 识别。本地位纯本地
        # 写，且远端写两个屏障后才到达（C.2），此处填充无竞争。
        meta[src_info_off:src_info_off + NvS].fill_(-1)
        # 源码 603：meta[0*ms + TPE_OFF + rank*E + i] = tpe[i]（写 rank0 chunk）
        arena.push_all(meta, _blocks(self.TPE_OFF + rank * E, E),
                       tpe.reshape(E, 1))
        if R > 1 and rank == 0:
            # rank0 额外把 topk/tpe 推入 rank1 的 chunk（TOPK0/TPE 区，C1 外包输入）
            # 源码 607：meta[1*ms + TOPK0_OFF + i] / meta[1*ms + TPE_OFF + i]
            idx = torch.cat([_blocks(self.TOPK0_OFF, N), _blocks(self.TPE_OFF, E)])
            pay = torch.cat([topk, tpe]).reshape(-1, 1)
            arena.push_all(meta, ms + idx, pay)
        arena.barrier()   # cross_rank_barrier #1

        # ================================================================
        # Phase B（610-960，仅 rank0）+ B.5 推送（multimem.st 的昇腾替代）
        # ================================================================
        if rank == 0:
            tpe_all = meta[self.TPE_OFF:self.TPE_OFF + R * E].reshape(R, E)
            tbl = _phase_b_tables(
                tpe_all, R=R, E=E, B=B, NvS=NvS,
                NvS_capacity=self.NvS_capacity, token_padding=self.token_padding,
            )
            # 写自己 chunk 的整个 PLAN 区（3ER 广播段 + CU/ZFR/ETC/STATS）
            _put(pb + alloc_sub, tbl["alloc_cumsum"], "PLAN.ALLOC")
            _put(pb + tpe_sub, tbl["tpe_cumsum"], "PLAN.TPE")
            _put(pb + eoff_sub, tbl["expert_offsets"], "PLAN.EOFF")
            _put(pb + cu_sub, tbl["cu_all"], "PLAN.CU")
            _put(pb + zfr_sub, tbl["zfr_all"], "PLAN.ZFR")
            _put(pb + etc_sub, tbl["etc_all"], "PLAN.ETC")
            _put(pb + stats_sub, tbl["stats_all"], "PLAN.STATS")
            if R > 1:
                # B.5：3ER 表逐 peer 推送（等价源码 954-960 multimem.st 广播）
                # 展平坐标：写每个 peer chunk 的 PLAN 区同一段
                # （flat = d*ms + PLAN_OFF + i，payload 为同一份 bc）
                bc = torch.cat([
                    tbl["alloc_cumsum"].reshape(-1),
                    tbl["tpe_cumsum"].reshape(-1),
                    tbl["expert_offsets"].reshape(-1),
                ]).to(torch.int32)
                n3 = 3 * E * R
                flat_bc = torch.cat([d * ms + _blocks(pb, n3)
                                     for d in range(1, R)])
                pay_bc = bc.reshape(1, n3).expand(R - 1, n3) \
                           .reshape(-1, 1).contiguous()
                arena.push_all(meta, flat_bc, pay_bc)

        # ================================================================
        # Phase C.1（398-517，rank≠0；rank0 的 C1 外包 rank1）
        # ================================================================
        if R == 1:
            order = _phase_c1_order(topk)
            _put(self.ORDER_OFF, order, "ORDER")
        elif rank != 0:
            order = _phase_c1_order(topk)
            _put(self.ORDER_OFF, order, "ORDER")
            if rank == 1:
                # 代算 rank0 的 C1（961-970）：读 TOPK0 区 → 结果推回 rank0 的 ORDER 区
                # 源码 970：meta[0*ms + ORDER_OFF + i] = order0[i]
                tk0 = meta[self.TOPK0_OFF:self.TOPK0_OFF + N]
                order0 = _phase_c1_order(tk0)
                _put(self.ORDER0_OFF, order0, "ORDER0")
                arena.push_all(meta, _blocks(self.ORDER_OFF, N),
                               order0.reshape(N, 1).to(torch.int32))
        # rank0 不跑 C1（忙于 Phase B）
        arena.barrier()   # cross_rank_barrier #2（3ER 推送与 ORDER 回写对全组可见）

        # ================================================================
        # Phase C.2（1048-1077，全组）：逐 rank dst 与 src_info 远端发布
        # ================================================================
        order = meta[self.ORDER_OFF:self.ORDER_OFF + N].to(torch.int64)
        tpe_cumsum = meta[pb + tpe_sub:pb + tpe_sub + R * E].reshape(R, E).to(torch.int64)
        alloc_cumsum = meta[pb + alloc_sub:pb + alloc_sub + E * R].reshape(E, R).to(torch.int64)
        expert_offsets = meta[pb + eoff_sub:pb + eoff_sub + R * E].reshape(R, E).to(torch.int64)
        dst_row, src_writes = _phase_c2_rank(
            order, topk, tpe, tpe_cumsum, alloc_cumsum, expert_offsets,
            rank, R=R, E=E, S=S, K=K, NvS=NvS,
        )
        # 展平坐标单次发布（对齐源码 1073：meta[dr*ms + SRC_INFO_OFF + loff]
        # = src_val）；pe==rank 的项由 transport 本地拷贝，不再保留本地直写
        # 分支（GPU 语义：所有 chunk RW 映射、本地 VA 直写）。
        if src_writes:
            flat_src = torch.cat([
                dr * ms + src_info_off + loffs
                for dr, (loffs, _vals) in sorted(src_writes.items())
            ])
            pay_src = torch.cat([
                vals for _dr, (_loffs, vals) in sorted(src_writes.items())
            ]).reshape(-1, 1).to(torch.int32)
        else:
            # 本 rank 无发出项：以 n=0 空 push_all 保持全组集合调用形态一致
            flat_src = torch.empty(0, dtype=torch.int64, device=dev)
            pay_src = torch.empty(0, 1, dtype=torch.int32, device=dev)
        arena.push_all(meta, flat_src, pay_src)
        arena.barrier()   # cross_rank_barrier #3（src_info 发布对全组可见）

        # ================================================================
        # Phase D（1030-1047 拉取 + 1079-1113 dedup，全组）
        # ================================================================
        # 从 rank0 chunk 拉本 rank 的 CU/ZFR/STATS 切片 + 全组 ETC 表
        n1, n2, n3 = E + B, 2 * (E + B), R * B
        req = torch.cat([
            _blocks(pb + cu_sub + rank * n1, n1),
            _blocks(pb + zfr_sub + rank * n2, n2),
            _blocks(pb + etc_sub, n3),
            _blocks(pb + stats_sub + rank * 2, 2),
        ])
        got = arena.pull_all(meta, req).reshape(-1)
        cu_seqlens.copy_(got[:n1].to(torch.int32))
        zero_fill.copy_(got[n1:n1 + n2].reshape(E + B, 2).to(torch.int32))
        experts_to_copy.copy_(got[n1 + n2:n1 + n2 + n3].reshape(R, B).to(torch.int32))
        remote_stats.copy_(got[n1 + n2 + n3:].to(torch.int32))

        # dedup 负编码（本地，k 序）
        dst.copy_(_phase_d_dedup(dst_row, S=S, K=K, R=R, NvS=NvS).to(torch.int32))


# ============================================================
# Host 侧：输出分配 + launch（对齐源码 planning.py:1136-1316）
# ============================================================
def allocate_planning_outputs(ctx: dict) -> tuple[MoonEPCommPlan, torch.Tensor]:
    """分配 (MoonEPCommPlan, cu_seqlens) 二元组（对齐源码 planning.py:1202-1250）。

    plan 自带的 dedup 张量在此一并分配（内容留待 DispatchKernel 的
    build_dedup_map 物化，对齐源码"fresh planning leaves their contents for
    the dispatch builder to materialize"）。

    与源码差异：不做 _round4 超配（那是 GPU 向量化写的越界防护，纯 torch 无
    此问题）；扩展字段 src_info [NvS] 一并分配（契约 §2）。
    """
    E = ctx["E"]
    B = ctx.get("B", 0)
    S = ctx["S"]
    K = ctx["K"]
    N = S * K
    R = ctx["R"]
    NvS = ctx["NvS"]
    dev = ctx["meta_buf"].device

    dst = torch.empty(N, dtype=torch.int32, device=dev)
    cu_seqlens = torch.empty(E + B, dtype=torch.int32, device=dev)
    experts_to_copy = torch.empty(R, B, dtype=torch.int32, device=dev)
    zero_fill_ranges = torch.empty(E + B, 2, dtype=torch.int32, device=dev)
    remote_stats = torch.empty(2, dtype=torch.int32, device=dev)
    dup_groups = torch.empty(NvS, 3, dtype=torch.int32, device=dev)
    dup_loffs = torch.empty(NvS, dtype=torch.int32, device=dev)
    dup_counts = torch.empty(2, dtype=torch.int32, device=dev)
    src_info = torch.empty(NvS, dtype=torch.int32, device=dev)

    plan = MoonEPCommPlan(
        dst=dst,
        experts_to_copy=experts_to_copy,
        zero_fill_ranges=zero_fill_ranges,
        remote_stats=remote_stats,
        dup_groups=dup_groups,
        dup_loffs=dup_loffs,
        dup_counts=dup_counts,
        N=N,
        R=R,
        E=E,
        B=B,
        NvS=NvS,
        K=K,
        src_info=src_info,
    )
    return plan, cu_seqlens


def _check_planning_outputs(ctx: dict, cu_seqlens, plan) -> None:
    """对齐源码 planning.py:1253-1264（逐字）。"""
    assert isinstance(plan, MoonEPCommPlan)
    E = ctx['E']
    B = ctx.get('B', 0)
    assert plan.N == ctx['S'] * ctx['K']
    assert plan.R == ctx['R']
    assert plan.E == E
    assert plan.B == B
    assert plan.NvS == ctx['NvS']
    assert plan.K == ctx['K']
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()
    assert tuple(cu_seqlens.shape) == (E + B,)


def _check_dedup_encoding_bounds(ctx: dict) -> None:
    """对齐源码 planning.py:1267-1291（逐字；KIDX_BITS 常量在模块顶层内联）。"""
    R = int(ctx['R'])
    S = int(ctx['S'])
    K = int(ctx['K'])
    NvS = int(ctx['NvS'])
    NvS_BITS = 32 - 1 - KIDX_BITS
    N = S * K
    int32_max = 2**31 - 1
    assert N <= NvS, (
        f"src_info NvS-stride encoding requires S*K <= NvS, got S*K={N}, NvS={NvS}"
    )
    assert R * NvS <= int32_max, (
        "src_info linear encoding requires R*NvS <= int32_max: "
        f"R={R}, NvS={NvS}, R*NvS={R * NvS}, int32_max={int32_max}"
    )
    assert R <= 128, (
        f"dst duplicate canonicalization rank bitset requires R <= 128, got {R}"
    )
    assert K <= (1 << KIDX_BITS) - 1, (
        f"primary_packed encoding requires K <= {(1 << KIDX_BITS) - 1}, got {K}"
    )
    assert NvS <= (1 << NvS_BITS) - 1, (
        f"primary_packed encoding requires NvS <= {(1 << NvS_BITS) - 1}, got {NvS}"
    )
    assert K <= 32, f"kmask bitmask requires K <= 32, got {K}"


def launch_planning(
    ctx: dict,
    topk_experts_flat,
    tokens_per_expert,
    cu_seqlens,
    plan,
) -> None:
    """Run the planning kernel; may reuse caller-allocated output objects.

    Results are written in-place into ``plan.dst``, ``plan.experts_to_copy``,
    ``plan.zero_fill_ranges``, ``plan.remote_stats`` and ``cu_seqlens``.
    The plan-owned dedup structures are materialized in the fresh dispatch
    builder; the reuse path with an existing plan reuses these tensors
    directly.

    参考实现差异（契约 §0/§4）：

    - 内部取 ``ctx['arena']`` 构造 PlanningKernel 并调用（常量配置键与源码
      ``_launch_planning_kernel`` 的 ``_get_compiled`` 实参表逐字对应）；
    - 源码经 ``ctx['meta_mc']/ctx['alloc']/ctx['group_tokens']/ctx['z']/
      ctx['local_hist']/ctx['grid_sync_bar']`` 传 GPU scratch；这些形参在
      torch 体中不使用，此处按 GPU 同形临时分配占位（mc 以 meta 占位）；
    - 返回前把扩展字段并入 plan：``plan.src_info`` 原地拷入 meta 的
      SRC_INFO 本 rank 切片（契约 §2）。
    """
    assert int(ctx['B']) > 0, f"planning requires B > 0, got B={int(ctx['B'])}"
    _check_planning_outputs(ctx, cu_seqlens, plan)
    _check_dedup_encoding_bounds(ctx)

    R = int(ctx['R'])
    E = int(ctx['E'])
    NvS = int(ctx['NvS'])
    kernel = PlanningKernel(
        R,
        E,
        int(ctx['B']),
        int(ctx['S']),
        int(ctx['K']),
        int(ctx['NvS_capacity']),
        NvS,
        int(ctx['num_vblocks']),
        int(ctx['meta_chunk_padded']),
        int(ctx['TPE_OFF']),
        int(ctx['PLAN_OFF']),
        int(ctx['BARRIER_OFF']),
        int(ctx['TOPK0_OFF']),
        int(ctx['ORDER_OFF']),
        int(ctx['ORDER0_OFF']),
        int(ctx['token_padding']),
        int(ctx['num_sms']),
        arena=ctx['arena'],
    )

    meta = ctx['meta_buf']
    dev = meta.device
    # GPU scratch 形参的同形占位（torch 体不使用，见 __call__ docstring）
    alloc = torch.empty(R * E, dtype=torch.int32, device=dev)
    group_tokens = torch.empty(R, dtype=torch.int32, device=dev)
    z = torch.empty(R * R, dtype=torch.int32, device=dev)
    local_hist = torch.empty(int(ctx['num_vblocks']) * E, dtype=torch.int32, device=dev)
    bar = torch.zeros(1, dtype=torch.int32, device=dev)

    kernel(
        tokens_per_expert,          # tpe [E] int32
        topk_experts_flat,          # topk [N] int32
        meta,                       # meta 本 rank chunk
        meta,                       # mc 占位（GPU 多播句柄，不使用）
        plan.dst,
        cu_seqlens,
        plan.experts_to_copy,
        plan.zero_fill_ranges,
        plan.remote_stats,
        alloc,
        group_tokens,
        z,
        local_hist,
        bar,
        int(ctx['rank']),
        None,                       # stream 留参（参考实现同步执行）
    )

    # 扩展字段并入 plan（契约 §2/§4）：SRC_INFO 本 rank 切片回读
    src_info_off = int(ctx.get("SRC_INFO_OFF", int(ctx["BARRIER_OFF"]) + _BARRIER_SLOTS))
    plan.src_info.copy_(meta[src_info_off:src_info_off + NvS])


def physical_tokens_per_expert(cu_seqlens: torch.Tensor, rank: int, epn: int, E: int) -> torch.Tensor:
    """diff(cu_seqlens) → [masters(epn) | slots(B)] 的本 rank 物理组计数 [epn+B]。

    cu_seqlens 的组 0..E-1 按全局专家号排列，本 rank 的本地专家段对应下标
    [rank*epn, (rank+1)*epn)；组 E..E+B-1 为副本槽段。diff 的首项即
    cu_seqlens[0]（前一项视为 0）。
    """
    counts = cu_seqlens - torch.cat((cu_seqlens.new_zeros(1), cu_seqlens[:-1]))
    masters = counts[rank * epn:(rank + 1) * epn]
    slots = counts[E:]
    return torch.cat((masters, slots))
