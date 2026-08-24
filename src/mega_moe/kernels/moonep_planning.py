# coding=utf-8
"""MoonEP planning 的 Triton 融合实现（v2，三 kernel bring-up 形态）。

语义基准：mega_moe/moonep_ref/moonep/planning.py（MoonEP torch 参考实现
@23d71348）。输出与 oracle 逐位一致（全 int64 内部计算、无浮点）。

与参考实现的编排差异（结果逐位等价）：oracle 是 rank0 集中算 Phase B +
B.5 推表、rank1 代算 rank0 的 C.1；本实现**全 rank 同构执行**（无
rank0/1 分工），按 bisheng 编译器形状 bug 的规避边界拆成三个 kernel：

- **kernel 1a ``moonep_plan_gather``（Phase A）**：本 rank topk 直方图得
  tpe → 写对称 tpe_all 本 rank 行 → putmem 逐 peer 发布 + fence + 尾部
  barrier（所有 rank 离开前 putmem 落地，后续 kernel 读全组可见——
  moonep_prefetch 尾 barrier 同款契约）；src_info 预填 -1（空槽哨兵）。
- **kernel 1b ``moonep_plan_tables``（Phase B）**：七张规划表每 rank 冗余
  自算（纯整数确定性 ⇒ 各 rank 逐位一致，oracle B.5 推表的替代）。
- **kernel 2 ``moonep_plan_dst``（Phase C+D）**：C.1 稳定计数排序（全
  rank 各算各的，rank0 不再外包；直方图重算——原版 c1 同款）→ C.2 逐
  排序位算 dst + src_info 发布（dl.symm_at indexed 直写目的 rank，对齐
  oracle src_writes 展开坐标 push）→ fence + barrier（发布对全组可见）
  → D dedup 负编码。

**为什么三个 kernel 而不是单 kernel**：``moonep_plan_fused``（同文件保
留，A~D 全融合）已验证逐位正确，但本地构建的 bishengir-compile 存在形状
相关的 convert-hfusion-to-hivm legalization bug（CASE-19）——多段共存时
（单 kernel 仅 E=4/R=2 可编译；{B.0+B.1+C.1} 三段共存即失败），逐段拆小
即可编译。规避均已 warmup 扫描验证：B.0/alloc_cs 逐 rank 行 1D 累加替代
2D tl.cumsum；B.4 在 E≤64 走无 indexed gather/scatter 的 onehot 选择版；
C.1 onehot/cumsum 中间量 i32；B.1/B.2 贪心循环 ``if R > 1`` 编译期剔除。
详见 docs/moonep_dev_cases.md CASE-19 与
memory:bisheng-convert-hfusion-shape-bug。编译器修复后切回单 kernel
（golden 测试保持对 fused 的逐位对拍锚点）。

**R == 1 不支持**（v1，CASE-19；实际使用无此形态）。

通信原语（仓内先例）：putmem + sub_vec_id()==0 守卫
（tests/kernel/moonep/test_step0_smoke.py ``_smoke_roundtrip``）；
dl.symm_at + tl.store（kernels/combine_fc1_bwd.py）；putmem/symm_at 之后
fence 再 barrier——barrier 不隐含 RMA 落地（CASE-14，moonep_combine.py）。

**集体性契约**：gather 尾 barrier 与 dst 的 src_info 后 barrier 均为全组
集体操作，全 rank 必须每步无条件一致 launch（禁止任何 skip 分支，同
moonep_prefetch 尾 barrier 纪律）。

v1 约束：E 为 2 的幂（⇒ R、epn 均为 2 的幂）、R>1、E%R==0、N%K==0、
R≤64（dedup 位掩码）、R·NvS < 2^31（dst/src_info 线性编码）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id

__all__ = [
    "MoonepPlanBuffers",
    "launch_moonep_planning",
    "moonep_plan_gather",
    "moonep_plan_tables",
    "moonep_plan_dst",
    "moonep_plan_fused",
]


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _device_id(device) -> int:
    s = str(device)
    return int(s.split(":")[-1]) if ":" in s else 0


class MoonepPlanBuffers:
    """planning 的设备缓冲（宿主持有）。

    对称（跨 rank 读写，全 rank 同形同序分配）：
        ``tpe_all`` int32 [R,E]——Phase A 全组 tpe 汇聚（各 rank 写自己的
        行，putmem 发布给 peer）；``src_info`` int32 [NvS]——C.2 远端发布
        的落点（outs["src_info"] 的数据源，kernel 1a 预填 -1）。
    本地：``order`` i32 [N]、``dst_row`` i64 [N]、七表（tpec [R,E] /
    alloc_cs [E,R] / eoff [R,E] / cu [R,E+B] / zfr [R,E+B,2] / etc [R,B] /
    stats [R,2]，全 i64）+ 中间表（gtok [R] / ecnt [E] / z [R,R] /
    alloc [R,E] i64、sel [R,E] i32）+ ``expoff`` i64 [E]。

    仅 ``eoff`` 依赖每步 ``zero_()``（本轮无 token 的专家不写、须保持 0）；
    其余缓冲每步被整块覆盖。
    """

    def __init__(self, R: int, E: int, B: int, N: int, NvS: int, device, *,
                 tpe_all: torch.Tensor | None = None,
                 src_info: torch.Tensor | None = None):
        import shmem as ash

        did = _device_id(device)
        self._owned: list[torch.Tensor] = []
        if tpe_all is None:
            tpe_all = ash.aclshmem_create_tensor(
                [R, E], torch.int32, device_id=did)
            self._owned.append(tpe_all)
        else:
            assert tpe_all.dtype == torch.int32 and \
                tuple(tpe_all.shape) == (R, E)
        if src_info is None:
            src_info = ash.aclshmem_create_tensor(
                [NvS], torch.int32, device_id=did)
            self._owned.append(src_info)
        else:
            assert src_info.dtype == torch.int32 and src_info.numel() == NvS
        self.tpe_all = tpe_all
        self.src_info = src_info

        mk = lambda shape, dt: torch.empty(shape, dtype=dt, device=device)
        self.order = mk(N, torch.int32)
        self.dst_row = mk(N, torch.int64)
        self.gtok = mk(R, torch.int64)
        self.ecnt = mk(E, torch.int64)
        self.z = mk((R, R), torch.int64)
        self.alloc = mk((R, E), torch.int64)
        self.tpec = mk((R, E), torch.int64)
        self.alloc_cs = mk((E, R), torch.int64)
        self.eoff = mk((R, E), torch.int64)
        self.etc = mk((R, B), torch.int64)
        self.stats = mk((R, 2), torch.int64)
        self.cu = mk((R, E + B), torch.int64)
        self.zfr = mk((R, E + B, 2), torch.int64)
        self.sel = mk((R, E), torch.int32)
        self.expoff = mk(E, torch.int64)

    def finalize(self):
        import shmem as ash

        for ten in self._owned:
            ash.aclshmem_free_tensor(ten)
        self._owned.clear()


# ---------------------------------------------------------------------------
# kernel 1a：Phase A（tpe allgather；尾 barrier 保证全组 putmem 落地）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_plan_gather(
    topk_ptr,               # int32 [N] 本 rank topk（展开序）
    tpe_all_ptr,            # 对称 int32 [R,E]（本 rank 行自写 + peer putmem）
    src_info_ptr,           # 对称 int32 [NvS]（预填 -1；远端写在 kernel 2）
    R: tl.constexpr, E: tl.constexpr, N: tl.constexpr, NvS: tl.constexpr,
    NP: tl.constexpr, NV: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    BLOCK_HIST: tl.constexpr, BLOCK_D: tl.constexpr,
):
    offs_e = tl.arange(0, E)

    # A0 直方图（= oracle 侧调用方的 bincount，语义同一数据）
    tpe = tl.zeros((E,), dtype=tl.int64)
    for s0 in range(0, NP, BLOCK_HIST):
        offs = s0 + tl.arange(0, BLOCK_HIST)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        tpe += tl.sum(tl.where(offs_e[None, :] == v[:, None], 1, 0), axis=0)

    # A1 src_info 预填 -1（空槽哨兵；远端写在 kernel 2 的 barrier 后才到）
    for s0 in range(0, NV, BLOCK_D):
        offs = s0 + tl.arange(0, BLOCK_D)
        tl.store(src_info_ptr + offs,
                 tl.full((BLOCK_D,), -1, tl.int32), mask=offs < NvS)

    # A2 写本 rank 行 + putmem 发布（同 offset：dst/src 同一本地指针表示）
    row = tpe_all_ptr + LOCAL_RANK * E
    tl.store(row + offs_e, tpe.to(tl.int32))
    if sub_vec_id() == 0:
        for peer in range(R):
            if peer != LOCAL_RANK:
                libshmem_device.putmem(row, row, E * 4, peer)
        libshmem_device.fence()          # CASE-14：barrier 不隐含 RMA 落地
    # 尾 barrier（全组契约）：所有 rank 离开本 kernel 前 putmem 均已落地，
    # 后续 kernel（tables）读 tpe_all 全组可见——moonep_prefetch 尾 barrier
    # 同款模式
    libshmem_device.barrier_all_vec()


# ---------------------------------------------------------------------------
# kernel 1b：Phase B（七张表，每 rank 冗余自算）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_plan_tables(
    tpe_all_ptr,            # 对称 int32 [R,E]（gather 后全组可见）
    # Phase B 七表 + 中间表
    gtok_ptr, ecnt_ptr,     # int64 [R], int64 [E]
    z_ptr, alloc_ptr,       # int64 [R,R], int64 [R,E]
    tpec_ptr,               # int64 [R,E]（tpe 沿 rank 维 inclusive 前缀）
    alloc_cs_ptr,           # int64 [E,R]（alloc 沿 rank 维前缀的转置）
    eoff_ptr,               # int64 [R,E]（须每步 zero_）
    etc_ptr, stats_ptr,     # int64 [R,B], int64 [R,2]
    cu_ptr, zfr_ptr,        # int64 [R,E+B], int64 [R,E+B,2]
    sel_ptr,                # int32 [R,E]
    expoff_ptr,             # int64 [E] 本 rank tpe 的 exclusive 前缀
    R: tl.constexpr, E: tl.constexpr, EPN: tl.constexpr, B: tl.constexpr,
    N: tl.constexpr, TP: tl.constexpr,
    G: tl.constexpr,        # next_pow2(E+B)
    LOCAL_RANK: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offs_e = tl.arange(0, E)
    offs_r = tl.arange(0, R)

    # ---- Phase B：七张表（每 rank 冗余自算，逐位复刻 oracle 平局规则；
    # tpe_all 的全组可见性由 kernel 1a 的尾部 barrier 保证）----
    # B.0 tpe_cumsum / expert_count / group_tokens（分块 over E）。
    # 不用 [R,BE] 的 2D tl.cumsum：与后段组合时部分形状触发 bisheng
    # legalization 失败（CASE-19），逐 rank 行 1D 累加数学等价
    gtok = tl.zeros((R,), dtype=tl.int64)
    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        run = tl.zeros((BLOCK_E,), dtype=tl.int64)
        for r in range(R):
            t = tl.load(tpe_all_ptr + r * E + oe, mask=me,
                        other=0).to(tl.int64)                        # [BE]
            run = run + t                                            # inclusive
            tl.store(tpec_ptr + r * E + oe, run, mask=me)
        tl.store(ecnt_ptr + oe, run, mask=me)
        hgrp = oe // EPN
        gtok += tl.sum(tl.where(hgrp[None, :] == offs_r[:, None],
                                run[None, :], 0), axis=1)
    tl.store(gtok_ptr + offs_r, gtok)

    # expoff：本 rank 行 exclusive 前缀（喂 kernel 2 的 C.2）
    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        r = tl.load(tpe_all_ptr + LOCAL_RANK * E + oe, mask=me,
                    other=0).to(tl.int64)
        tl.store(expoff_ptr + oe, tl.cumsum(r, axis=0) - r, mask=me)

    # B.1 单源填充迁移矩阵 z[R,R]（水位贪心；≤R-1 轮，平局取小下标）。
    # R==1 不支持（CASE-19）：循环体编译期剔除
    bal = gtok - N                     # CAP = N = S·K
    z = tl.zeros((R, R), dtype=tl.int64)
    if R > 1:
        active = True
        for _it in range(R - 1):
            mx = tl.max(bal, axis=0)
            mn = tl.min(bal, axis=0)
            ok = active & (mx > 0) & (mn < 0)
            s = tl.min(tl.where(bal == mx, offs_r, R), axis=0)   # 平局取小
            d = tl.min(tl.where(bal == mn, offs_r, R), axis=0)   # 平局取小
            move = -mn                  # 一次补满亏空
            z = tl.where(ok & (offs_r[:, None] == s) & (offs_r[None, :] == d),
                         move, z)
            bal = tl.where(ok & (offs_r == s), bal - move, bal)
            bal = tl.where(ok & (offs_r == d), 0, bal)
            active = ok
    tl.store(z_ptr + offs_r[:, None] * R + offs_r[None, :], z)

    # B.2 按属主组贪心切分 alloc[R,E]（组间串行；组内 ≤R+EPN 轮，平局取小）
    offs_le = tl.arange(0, EPN)
    for h in range(R):
        quotas = tl.sum(tl.where(offs_r[:, None] == h, z, 0), axis=0)  # z[h,:]
        rem = tl.load(ecnt_ptr + h * EPN + offs_le)
        acc = tl.zeros((R, EPN), dtype=tl.int64)
        acc = tl.where(offs_r[:, None] == h, rem[None, :], acc)   # 属主初值
        if R > 1:                        # R==1 无迁移：acc 即属主初值
            active = True
            for _it in range(R + EPN):
                mq = tl.max(quotas, axis=0)
                mr = tl.max(rem, axis=0)
                ok = active & (mq > 0) & (mr > 0)
                d = tl.min(tl.where(quotas == mq, offs_r, R), axis=0)
                le = tl.min(tl.where(rem == mr, offs_le, EPN), axis=0)
                take = tl.minimum(mr, mq)
                quotas = tl.where(ok & (offs_r == d), mq - take, quotas)
                rem = tl.where(ok & (offs_le == le), mr - take, rem)
                acc = tl.where(ok & (offs_r[:, None] == d)
                               & (offs_le[None, :] == le), acc + take, acc)
                acc = tl.where(ok & (offs_r[:, None] == h)
                               & (offs_le[None, :] == le), mr - take, acc)
                active = ok
        tl.store(alloc_ptr + offs_r[:, None] * E + (h * EPN + offs_le)[None, :],
                 acc)

    # alloc_cumsum [E,R] = alloc.cumsum(0).T（分块转置存；逐 rank 行 1D 累加）
    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        run = tl.zeros((BLOCK_E,), dtype=tl.int64)
        for r in range(R):
            a = tl.load(alloc_ptr + r * E + oe, mask=me, other=0)
            run = run + a
            tl.store(alloc_cs_ptr + oe * R + r, run, mask=me)

    # B.3 top-B 副本槽（B 轮清零链；平局取大下标）；stats[.][1] 写属主组
    stats0 = tl.zeros((R,), dtype=tl.int64)
    stats1 = tl.zeros((R,), dtype=tl.int64)
    for d in range(R):
        a_row = tl.load(alloc_ptr + d * E + offs_e)
        is_loc = (offs_e >= d * EPN) & (offs_e < (d + 1) * EPN)
        rc = tl.where(is_loc, 0, a_row)
        stats0 += tl.where(offs_r == d,
                           tl.sum(tl.where(rc > 0, 1, 0), axis=0), 0)
        sel = tl.zeros((E,), dtype=tl.int32)
        for b in range(B):
            best = tl.max(rc, axis=0)
            ok = best > 0
            idx = tl.max(tl.where(rc == best, offs_e, -1), axis=0)  # 取大
            tl.store(etc_ptr + d * B + b, tl.where(ok, idx, -1).to(tl.int64))
            sel = tl.where(ok & (offs_e == idx), 1, sel)
            rc = tl.where(ok & (offs_e == idx), 0, rc)
            stats1 += tl.where(ok & (offs_r == idx // EPN), 1, 0)
        tl.store(sel_ptr + d * E + offs_e, sel)
    tl.store(stats_ptr + offs_r * 2, stats0)
    tl.store(stats_ptr + offs_r * 2 + 1, stats1)

    # B.4 VM 段布局：常驻段+槽段统一到 [G] 向量，cumsum 一次向量化。
    # 双路径：E≤64 走无 indexed gather/scatter 的 onehot 选择版（indexed
    # 路径在部分形状触发 bisheng legalization 失败，CASE-19）；大 E 走
    # 原 indexed 版（[G,E] 选择 tile 超 UB，且 indexed 路径在 E=128/256
    # 生产形状编译通过）
    offs_g = tl.arange(0, G)
    valid = offs_g < E + B
    is_res = offs_g < E
    is_slot = (offs_g >= E) & valid
    for d in range(R):
        sel_g = tl.load(sel_ptr + d * E + tl.where(is_res, offs_g, 0),
                        mask=is_res, other=1)
        se = tl.load(etc_ptr + d * B + tl.where(is_slot, offs_g - E, 0),
                     mask=is_slot, other=-1)
        eid = tl.where(is_res, offs_g.to(tl.int64), se)    # pad 段 eid=-1
        if E <= 64:
            cnt = tl.zeros((G,), dtype=tl.int64)
            for e0 in range(0, E, BLOCK_E):
                oe = e0 + tl.arange(0, BLOCK_E)
                me = oe < E
                a_row = tl.load(alloc_ptr + d * E + oe, mask=me, other=0)
                cnt += tl.sum(tl.where((eid[:, None] == oe[None, :])
                                       & (eid[:, None] >= 0),
                                       a_row[None, :], 0), axis=1)
        else:
            cnt = tl.load(alloc_ptr + d * E + tl.where(eid >= 0, eid, 0),
                          mask=eid >= 0, other=0)
        cnt = tl.where(is_res & (sel_g != 0), 0, cnt)      # 被选槽的常驻段空
        padded = tl.where(cnt > 0, ((cnt + TP - 1) // TP) * TP, 0)
        cu_i = tl.cumsum(padded, axis=0)                   # inclusive 段尾
        start = cu_i - padded                              # exclusive 段首
        tl.store(cu_ptr + d * (E + B) + offs_g, cu_i, mask=valid)
        if E <= 64:
            # eoff[e] = start_g（eid_g==e 且 cnt_g>0）；每 e 至多命中一个 g，
            # 未命中写 0（缓冲每步 zero_，语义等价 oracle 的"只写非空段"）
            for e0 in range(0, E, BLOCK_E):
                oe = e0 + tl.arange(0, BLOCK_E)
                me = oe < E
                eoff_val = tl.sum(
                    tl.where((eid[None, :] == oe[:, None].to(tl.int64))
                             & (cnt > 0)[None, :], start[None, :], 0), axis=1)
                tl.store(eoff_ptr + d * E + oe, eoff_val, mask=me)
        else:
            tl.store(eoff_ptr + d * E + eid, start, mask=valid & (cnt > 0))
        extra = padded - cnt
        pz = valid & (extra > 0)
        tl.store(zfr_ptr + d * (E + B) * 2 + offs_g * 2,
                 tl.where(pz, start + cnt, 0), mask=valid)
        tl.store(zfr_ptr + d * (E + B) * 2 + offs_g * 2 + 1,
                 tl.where(pz, extra, 0), mask=valid)


# ---------------------------------------------------------------------------
# kernel 2：Phase C（order / dst / src_info 发布）+ Phase D（dedup）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_plan_dst(
    topk_ptr,               # int32 [N] 本 rank topk
    src_info_ptr,           # 对称 int32 [NvS]（peer symm_at 直写 + 本地条目）
    dst_out_ptr,            # int32 [N] 输出（Phase D 负编码后）
    order_ptr,              # int32 [N] C.1 稳定序
    dst_row_ptr,            # int64 [N] C.2 原始 dst
    tpec_ptr, alloc_cs_ptr, eoff_ptr,     # kernel 1b 的表（i64）
    expoff_ptr,             # int64 [E]
    R: tl.constexpr, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    NvS: tl.constexpr, NP: tl.constexpr,
    LOCAL_RANK: tl.constexpr,
    BLOCK_HIST: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    offs_e = tl.arange(0, E)
    offs_r = tl.arange(0, R)

    # C.1 稳定计数排序（阶段 1 直方图重算——原版 c1 同款；i32 中间量）
    tpe = tl.zeros((E,), dtype=tl.int64)
    for s0 in range(0, NP, BLOCK_HIST):
        offs = s0 + tl.arange(0, BLOCK_HIST)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        tpe += tl.sum(tl.where(offs_e[None, :] == v[:, None], 1, 0), axis=0)
    start_e = (tl.cumsum(tpe, axis=0) - tpe).to(tl.int32)   # 各专家起始位
    running = tl.zeros((E,), dtype=tl.int32)
    for s0 in range(0, NP, BLOCK_HIST):
        offs = s0 + tl.arange(0, BLOCK_HIST)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        onehot = ((offs_e[None, :] == v[:, None]) & m[:, None]).to(tl.int32)
        pre = tl.cumsum(onehot, axis=0) - onehot          # 块内 exclusive 前缀
        rank_in_blk = tl.sum(pre * onehot, axis=1)
        e_start = tl.sum(onehot * (start_e + running)[None, :], axis=1)
        tl.store(order_ptr + e_start + rank_in_blk, offs.to(tl.int32), mask=m)
        running += tl.sum(onehot, axis=0)

    # C.2 逐排序位算 dst + src_info 发布
    for s0 in range(0, NP, BLOCK_N):
        pos = s0 + tl.arange(0, BLOCK_N)
        m = pos < N
        o = tl.load(order_ptr + pos, mask=m, other=0).to(tl.int64)
        e = tl.load(topk_ptr + o, mask=m, other=0).to(tl.int64)

        prev = tl.zeros((BLOCK_N,), dtype=tl.int64)
        if LOCAL_RANK > 0:
            prev = tl.load(tpec_ptr + (LOCAL_RANK - 1) * E + e,
                           mask=m, other=0)
        expoff = tl.load(expoff_ptr + e, mask=m, other=0)
        gidx = prev + (pos - expoff)                      # 全组专家序

        rows = tl.load(alloc_cs_ptr + e[:, None] * R + offs_r[None, :],
                       mask=m[:, None], other=0)          # [BLOCK,R]
        lo = tl.sum(tl.where(rows <= gidx[:, None], 1, 0), axis=1)
        pc = tl.sum(tl.where(offs_r[None, :] == (lo - 1)[:, None], rows, 0),
                    axis=1)
        eoff = tl.load(eoff_ptr + lo * E + e, mask=m, other=0)
        drl = lo * NvS + eoff + (gidx - pc)               # dst_row（全非负）
        tl.store(dst_row_ptr + o, drl, mask=m)

        # src_info 发布：目的 rank 的槽出处（对齐 oracle src_writes）
        if sub_vec_id() == 0:
            val = (LOCAL_RANK * NvS + o).to(tl.int32)
            loff = drl - lo * NvS
            for peer in range(R):
                pm = m & (lo == peer)
                if peer == LOCAL_RANK:
                    tl.store(src_info_ptr + loff, val, mask=pm)
                else:
                    tl.store(dl.symm_at(src_info_ptr, peer) + loff, val,
                             mask=pm)
            libshmem_device.fence()

    libshmem_device.barrier_all_vec()     # src_info 全组发布可见（下游消费）

    # Phase D：dedup 负编码
    S: tl.constexpr = N // K
    one = tl.full((BLOCK_D,), 1, dtype=tl.int64)
    for t0 in range(0, S, BLOCK_D):
        tok = t0 + tl.arange(0, BLOCK_D)
        m = tok < S
        seen = tl.zeros((BLOCK_D,), dtype=tl.int64)
        for k in range(K):
            raw = tl.load(dst_row_ptr + tok * K + k, mask=m, other=0)
            dest = raw // NvS
            dup = (seen >> dest) & one
            enc = tl.where(dup == one, -raw - one, raw)
            tl.store(dst_out_ptr + tok * K + k, enc.to(tl.int32), mask=m)
            seen = seen | (one << dest)


# ---------------------------------------------------------------------------
# 单 kernel 融合版（编译器 bug 修复前的基准锚点，golden 测试对拍用；
# 仅 E=4/R=2 形状可编译，bug 详见模块 docstring / CASE-19）
# ---------------------------------------------------------------------------
@triton.jit
def moonep_plan_fused(
    topk_ptr, tpe_all_ptr, src_info_ptr, dst_out_ptr, order_ptr, dst_row_ptr,
    gtok_ptr, ecnt_ptr, z_ptr, alloc_ptr, tpec_ptr, alloc_cs_ptr, eoff_ptr,
    etc_ptr, stats_ptr, cu_ptr, zfr_ptr, sel_ptr, expoff_ptr,
    R: tl.constexpr, E: tl.constexpr, EPN: tl.constexpr, B: tl.constexpr,
    N: tl.constexpr, K: tl.constexpr, NvS: tl.constexpr, TP: tl.constexpr,
    NP: tl.constexpr, NV: tl.constexpr, G: tl.constexpr,
    LOCAL_RANK: tl.constexpr, BLOCK_HIST: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """A~D 全融合（三 kernel 版的逐行拼接，语义与三段完全一致）。"""
    offs_e = tl.arange(0, E)
    offs_r = tl.arange(0, R)

    tpe = tl.zeros((E,), dtype=tl.int64)
    for s0 in range(0, NP, BLOCK_HIST):
        offs = s0 + tl.arange(0, BLOCK_HIST)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        tpe += tl.sum(tl.where(offs_e[None, :] == v[:, None], 1, 0), axis=0)

    for s0 in range(0, NV, BLOCK_D):
        offs = s0 + tl.arange(0, BLOCK_D)
        tl.store(src_info_ptr + offs,
                 tl.full((BLOCK_D,), -1, tl.int32), mask=offs < NvS)

    row = tpe_all_ptr + LOCAL_RANK * E
    tl.store(row + offs_e, tpe.to(tl.int32))
    if sub_vec_id() == 0:
        for peer in range(R):
            if peer != LOCAL_RANK:
                libshmem_device.putmem(row, row, E * 4, peer)
        libshmem_device.fence()
    libshmem_device.barrier_all_vec()

    gtok = tl.zeros((R,), dtype=tl.int64)
    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        run = tl.zeros((BLOCK_E,), dtype=tl.int64)
        for r in range(R):
            t = tl.load(tpe_all_ptr + r * E + oe, mask=me,
                        other=0).to(tl.int64)
            run = run + t
            tl.store(tpec_ptr + r * E + oe, run, mask=me)
        tl.store(ecnt_ptr + oe, run, mask=me)
        hgrp = oe // EPN
        gtok += tl.sum(tl.where(hgrp[None, :] == offs_r[:, None],
                                run[None, :], 0), axis=1)
    tl.store(gtok_ptr + offs_r, gtok)

    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        r = tl.load(tpe_all_ptr + LOCAL_RANK * E + oe, mask=me,
                    other=0).to(tl.int64)
        tl.store(expoff_ptr + oe, tl.cumsum(r, axis=0) - r, mask=me)

    bal = gtok - N
    z = tl.zeros((R, R), dtype=tl.int64)
    if R > 1:
        active = True
        for _it in range(R - 1):
            mx = tl.max(bal, axis=0)
            mn = tl.min(bal, axis=0)
            ok = active & (mx > 0) & (mn < 0)
            s = tl.min(tl.where(bal == mx, offs_r, R), axis=0)
            d = tl.min(tl.where(bal == mn, offs_r, R), axis=0)
            move = -mn
            z = tl.where(ok & (offs_r[:, None] == s) & (offs_r[None, :] == d),
                         move, z)
            bal = tl.where(ok & (offs_r == s), bal - move, bal)
            bal = tl.where(ok & (offs_r == d), 0, bal)
            active = ok
    tl.store(z_ptr + offs_r[:, None] * R + offs_r[None, :], z)

    offs_le = tl.arange(0, EPN)
    for h in range(R):
        quotas = tl.sum(tl.where(offs_r[:, None] == h, z, 0), axis=0)
        rem = tl.load(ecnt_ptr + h * EPN + offs_le)
        acc = tl.zeros((R, EPN), dtype=tl.int64)
        acc = tl.where(offs_r[:, None] == h, rem[None, :], acc)
        if R > 1:
            active = True
            for _it in range(R + EPN):
                mq = tl.max(quotas, axis=0)
                mr = tl.max(rem, axis=0)
                ok = active & (mq > 0) & (mr > 0)
                d = tl.min(tl.where(quotas == mq, offs_r, R), axis=0)
                le = tl.min(tl.where(rem == mr, offs_le, EPN), axis=0)
                take = tl.minimum(mr, mq)
                quotas = tl.where(ok & (offs_r == d), mq - take, quotas)
                rem = tl.where(ok & (offs_le == le), mr - take, rem)
                acc = tl.where(ok & (offs_r[:, None] == d)
                               & (offs_le[None, :] == le), acc + take, acc)
                acc = tl.where(ok & (offs_r[:, None] == h)
                               & (offs_le[None, :] == le), mr - take, acc)
                active = ok
        tl.store(alloc_ptr + offs_r[:, None] * E + (h * EPN + offs_le)[None, :],
                 acc)

    for e0 in range(0, E, BLOCK_E):
        oe = e0 + tl.arange(0, BLOCK_E)
        me = oe < E
        run = tl.zeros((BLOCK_E,), dtype=tl.int64)
        for r in range(R):
            a = tl.load(alloc_ptr + r * E + oe, mask=me, other=0)
            run = run + a
            tl.store(alloc_cs_ptr + oe * R + r, run, mask=me)

    stats0 = tl.zeros((R,), dtype=tl.int64)
    stats1 = tl.zeros((R,), dtype=tl.int64)
    for d in range(R):
        a_row = tl.load(alloc_ptr + d * E + offs_e)
        is_loc = (offs_e >= d * EPN) & (offs_e < (d + 1) * EPN)
        rc = tl.where(is_loc, 0, a_row)
        stats0 += tl.where(offs_r == d,
                           tl.sum(tl.where(rc > 0, 1, 0), axis=0), 0)
        sel = tl.zeros((E,), dtype=tl.int32)
        for b in range(B):
            best = tl.max(rc, axis=0)
            ok = best > 0
            idx = tl.max(tl.where(rc == best, offs_e, -1), axis=0)
            tl.store(etc_ptr + d * B + b, tl.where(ok, idx, -1).to(tl.int64))
            sel = tl.where(ok & (offs_e == idx), 1, sel)
            rc = tl.where(ok & (offs_e == idx), 0, rc)
            stats1 += tl.where(ok & (offs_r == idx // EPN), 1, 0)
        tl.store(sel_ptr + d * E + offs_e, sel)
    tl.store(stats_ptr + offs_r * 2, stats0)
    tl.store(stats_ptr + offs_r * 2 + 1, stats1)

    offs_g = tl.arange(0, G)
    valid = offs_g < E + B
    is_res = offs_g < E
    is_slot = (offs_g >= E) & valid
    for d in range(R):
        sel_g = tl.load(sel_ptr + d * E + tl.where(is_res, offs_g, 0),
                        mask=is_res, other=1)
        se = tl.load(etc_ptr + d * B + tl.where(is_slot, offs_g - E, 0),
                     mask=is_slot, other=-1)
        eid = tl.where(is_res, offs_g.to(tl.int64), se)
        if E <= 64:
            cnt = tl.zeros((G,), dtype=tl.int64)
            for e0 in range(0, E, BLOCK_E):
                oe = e0 + tl.arange(0, BLOCK_E)
                me = oe < E
                a_row = tl.load(alloc_ptr + d * E + oe, mask=me, other=0)
                cnt += tl.sum(tl.where((eid[:, None] == oe[None, :])
                                       & (eid[:, None] >= 0),
                                       a_row[None, :], 0), axis=1)
        else:
            cnt = tl.load(alloc_ptr + d * E + tl.where(eid >= 0, eid, 0),
                          mask=eid >= 0, other=0)
        cnt = tl.where(is_res & (sel_g != 0), 0, cnt)
        padded = tl.where(cnt > 0, ((cnt + TP - 1) // TP) * TP, 0)
        cu_i = tl.cumsum(padded, axis=0)
        start = cu_i - padded
        tl.store(cu_ptr + d * (E + B) + offs_g, cu_i, mask=valid)
        if E <= 64:
            for e0 in range(0, E, BLOCK_E):
                oe = e0 + tl.arange(0, BLOCK_E)
                me = oe < E
                eoff_val = tl.sum(
                    tl.where((eid[None, :] == oe[:, None].to(tl.int64))
                             & (cnt > 0)[None, :], start[None, :], 0), axis=1)
                tl.store(eoff_ptr + d * E + oe, eoff_val, mask=me)
        else:
            tl.store(eoff_ptr + d * E + eid, start, mask=valid & (cnt > 0))
        extra = padded - cnt
        pz = valid & (extra > 0)
        tl.store(zfr_ptr + d * (E + B) * 2 + offs_g * 2,
                 tl.where(pz, start + cnt, 0), mask=valid)
        tl.store(zfr_ptr + d * (E + B) * 2 + offs_g * 2 + 1,
                 tl.where(pz, extra, 0), mask=valid)

    start_e = (tl.cumsum(tpe, axis=0) - tpe).to(tl.int32)
    running = tl.zeros((E,), dtype=tl.int32)
    for s0 in range(0, NP, BLOCK_HIST):
        offs = s0 + tl.arange(0, BLOCK_HIST)
        m = offs < N
        v = tl.load(topk_ptr + offs, mask=m, other=E)
        onehot = ((offs_e[None, :] == v[:, None]) & m[:, None]).to(tl.int32)
        pre = tl.cumsum(onehot, axis=0) - onehot
        rank_in_blk = tl.sum(pre * onehot, axis=1)
        e_start = tl.sum(onehot * (start_e + running)[None, :], axis=1)
        tl.store(order_ptr + e_start + rank_in_blk, offs.to(tl.int32), mask=m)
        running += tl.sum(onehot, axis=0)

    for s0 in range(0, NP, BLOCK_N):
        pos = s0 + tl.arange(0, BLOCK_N)
        m = pos < N
        o = tl.load(order_ptr + pos, mask=m, other=0).to(tl.int64)
        e = tl.load(topk_ptr + o, mask=m, other=0).to(tl.int64)
        prev = tl.zeros((BLOCK_N,), dtype=tl.int64)
        if LOCAL_RANK > 0:
            prev = tl.load(tpec_ptr + (LOCAL_RANK - 1) * E + e,
                           mask=m, other=0)
        expoff = tl.load(expoff_ptr + e, mask=m, other=0)
        gidx = prev + (pos - expoff)
        rows = tl.load(alloc_cs_ptr + e[:, None] * R + offs_r[None, :],
                       mask=m[:, None], other=0)
        lo = tl.sum(tl.where(rows <= gidx[:, None], 1, 0), axis=1)
        pc = tl.sum(tl.where(offs_r[None, :] == (lo - 1)[:, None], rows, 0),
                    axis=1)
        eoff = tl.load(eoff_ptr + lo * E + e, mask=m, other=0)
        drl = lo * NvS + eoff + (gidx - pc)
        tl.store(dst_row_ptr + o, drl, mask=m)
        if sub_vec_id() == 0:
            val = (LOCAL_RANK * NvS + o).to(tl.int32)
            loff = drl - lo * NvS
            for peer in range(R):
                pm = m & (lo == peer)
                if peer == LOCAL_RANK:
                    tl.store(src_info_ptr + loff, val, mask=pm)
                else:
                    tl.store(dl.symm_at(src_info_ptr, peer) + loff, val,
                             mask=pm)
            libshmem_device.fence()
    libshmem_device.barrier_all_vec()

    S: tl.constexpr = N // K
    one = tl.full((BLOCK_D,), 1, dtype=tl.int64)
    for t0 in range(0, S, BLOCK_D):
        tok = t0 + tl.arange(0, BLOCK_D)
        m = tok < S
        seen = tl.zeros((BLOCK_D,), dtype=tl.int64)
        for k in range(K):
            raw = tl.load(dst_row_ptr + tok * K + k, mask=m, other=0)
            dest = raw // NvS
            dup = (seen >> dest) & one
            enc = tl.where(dup == one, -raw - one, raw)
            tl.store(dst_out_ptr + tok * K + k, enc.to(tl.int32), mask=m)
            seen = seen | (one << dest)


# ---------------------------------------------------------------------------
# 宿主编排
# ---------------------------------------------------------------------------
def launch_moonep_planning(
    bufs: MoonepPlanBuffers,
    outs: dict,
    topk: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    ep_group,
    S: int,
    K: int,
    E: int,
    B: int,
    NvS: int,
    token_padding: int,
) -> None:
    """跑一次融合 planning（三 kernel），结果原地写入 ``outs`` 的各张量。

    outs 契约（与 v1 完全一致，下游 dispatch/combine/backward 零改动）：
        dst [S*K]、cu_seqlens [E+B]、experts_to_copy [R,B]、
        zero_fill_ranges [E+B,2]、remote_stats [2]、src_info [NvS]
        （全部 int32 设备张量，预分配）；附加宿主侧 _tbl（七表 cpu/int64
        dict，cu_all 被 dispatch/backward 消费）与 _dst_all（cpu/int64
        [R,N]，build_recv_counts 输入）。
    """
    import torch.distributed as dist

    N = S * K
    R = world_size
    dev = topk.device
    assert topk.dtype == torch.int32 and topk.numel() == N
    assert R > 1, "R==1 不支持（bisheng 形状 bug，CASE-19）"
    assert R <= 64, f"dedup 位掩码要求 R<=64，got {R}"
    assert N % K == 0 and E % R == 0 and (E & (E - 1)) == 0, \
        f"v1 约束：N%K==0 且 E 为 2 的幂，got N={N} K={K} E={E}"

    # eoff 只在 cnt>0 时写：每步清零防残值（其余表被整块覆盖）
    bufs.eoff.zero_()

    hist = max(8, min(256, _next_pow2(max(1, 8192 // E))))
    moonep_plan_gather[(1, 1, 1)](
        topk, bufs.tpe_all, bufs.src_info,
        R=R, E=E, N=N, NvS=NvS, NP=_next_pow2(N), NV=_next_pow2(NvS),
        LOCAL_RANK=rank, BLOCK_HIST=hist, BLOCK_D=256,
    )
    moonep_plan_tables[(1, 1, 1)](
        bufs.tpe_all,
        bufs.gtok, bufs.ecnt, bufs.z, bufs.alloc,
        bufs.tpec, bufs.alloc_cs, bufs.eoff, bufs.etc, bufs.stats,
        bufs.cu, bufs.zfr, bufs.sel, bufs.expoff,
        R=R, E=E, EPN=E // R, B=B, N=N, TP=token_padding,
        G=_next_pow2(E + B), LOCAL_RANK=rank, BLOCK_E=min(32, E),
    )
    moonep_plan_dst[(1, 1, 1)](
        topk, bufs.src_info, outs["dst"], bufs.order, bufs.dst_row,
        bufs.tpec, bufs.alloc_cs, bufs.eoff, bufs.expoff,
        R=R, E=E, N=N, K=K, NvS=NvS, NP=_next_pow2(N),
        LOCAL_RANK=rank,
        BLOCK_HIST=hist, BLOCK_N=256, BLOCK_D=256,
    )

    # ---- outs 切片（设备→设备；int32 出口 cast 与 v1 约定一致）----
    outs["cu_seqlens"].copy_(bufs.cu[rank].to(torch.int32))
    outs["experts_to_copy"].copy_(bufs.etc.to(torch.int32))
    outs["zero_fill_ranges"].copy_(bufs.zfr[rank].to(torch.int32))
    outs["remote_stats"].copy_(bufs.stats[rank].to(torch.int32))
    outs["src_info"].copy_(bufs.src_info)

    # ---- _tbl：宿主七表（dispatch send_meta / backward 需 cu_all cpu/i64）----
    outs["_tbl"] = {
        "tpe_cumsum": bufs.tpec.cpu(),
        "alloc_cumsum": bufs.alloc_cs.cpu(),
        "expert_offsets": bufs.eoff.cpu(),
        "cu_all": bufs.cu.cpu(),
        "zfr_all": bufs.zfr.cpu(),
        "etc_all": bufs.etc.cpu(),
        "stats_all": bufs.stats.cpu(),
    }

    # ---- _dst_all：全组 dst（build_recv_counts 输入；HCCL 设备张量）----
    dst_list = [torch.empty(N, dtype=torch.int32, device=dev)
                for _ in range(R)]
    dist.all_gather(dst_list, outs["dst"], group=ep_group)
    outs["_dst_all"] = torch.stack([d.to(torch.int64).cpu() for d in dst_list])
