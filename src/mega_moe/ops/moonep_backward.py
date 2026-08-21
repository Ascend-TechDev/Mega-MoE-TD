# coding=utf-8
"""MoonEP 后向（B-1/B-2：dx / dw 打通；B-3 wgrad 见 moonep_backward_plan）。

核心（docs/moonep_backward_plan.md）：**plan 在反向全量复用**——

- dy 散布：`moonep_dispatch_push`（PUSH_RW=0）原样复用——dy[t] 按 send 表
  推到各 rank 的 ``dy_recv[loff]``，布局与 forward payload 同构；
- FC2 dgrad：`gact = dy_recv @ dn_e`（段 GEMM，复用 moonep_fc2_gemm 的
  通用形态；权重视图 = ``down.transpose(-1,-2)``，N(out)=F、K(red)=H）。
  gact 是 **w 相乘前**的 ∂L/∂act（D1 同构分界）；
- swiglu_bwd：**classic 原样复用**（其 dc 输入语义恰好是 pre-w 的
  ∂L/∂(w·sw)、dscale 输出即 ∂L/∂w）——w 在此乘入；
- FC1 dgrad：`gy = dAB @ gu_e`（段 GEMM；视图 = ``gate_up.transpose``，
  N=H、K=2F）；
- dx：gy 行按 src_info 推回 + top-k 求和——**`moonep_combine_push`/
  `moonep_topk_reduce` 原样换指针**；
- dw：v1 宿主聚合（dscale 按 src_info 重排到 [S,K]；量小，设备化后续）。

dup（v1 照发语义）：K 份贡献独立，无需特殊处理。
"""

from __future__ import annotations

import torch

from mega_moe.kernels.moonep_combine import (
    moonep_combine_push,
    moonep_fc2_gemm,
    moonep_topk_reduce,
)
from mega_moe.kernels.moonep_dispatch import moonep_dispatch_push
from mega_moe.kernels.swiglu_bwd import swiglu_bwd_triton
from mega_moe.runtime.moonep_routing import (
    build_moonep_send_meta,
    build_moonep_segment_meta,
)

__all__ = ["launch_moonep_backward_dx"]


def launch_moonep_backward_dx(
    op,                        # MoonepForward（复用其 workspace/outs/saved）
    dy: torch.Tensor,          # bf16 [S, H]（loss 对 forward 输出的梯度）
) -> dict:
    """B-1/B-2：返回 dict(dx [S,H] bf16, dw [S,K] fp32, gact/dAB/gy 调试量)。

    前置：op.forward 已跑（saved/outs 就绪）；单 in-flight（下一个 forward
    前完成反向）。
    """
    t = op.topo
    dev = op._device
    outs = op._outs
    saved = op._saved
    rows_pad = saved["rows_pad"]
    ws = op.ws
    cu_all = outs["_tbl"]["cu_all"].to(torch.int64)

    seg_counts, seg_offsets, _ = build_moonep_segment_meta(
        cu_all[op.rank],
        outs["experts_to_copy"].cpu()[op.rank], op.rank, t.epn, t.E, t.B)

    # ---- B-1a：dy 散布（send 表复用；send 表宿主重建，plan 不变）----
    send = build_moonep_send_meta(outs["dst"].cpu(), cu_all, t.K, t.epn,
                                  t.E, t.B, t.NvS)
    d = lambda x: x.to(dev)
    dy_c = dy.contiguous()
    moonep_dispatch_push[(op.num_cores, 1, 1)](
        dy_c, ws.dy_recv, dy_c, dy_c,        # rw 指针占位（PUSH_RW=0 不用）
        d(send["send_src_idx"]), d(send["send_offv"]), d(send["send_loff"]),
        d(send["run_dst"]), d(send["run_seg"]), d(send["run_start"]),
        d(send["run_count"]), int(send["run_dst"].numel()),
        t.H, dy_c.stride(0),
        NUM_PROGRAM_CORES=op.num_cores, BLOCK_M=op.block, PUSH_RW=0,
    )

    # ---- B-1b：FC2 dgrad → gact（pre-w）----
    dn_t = ws.down.transpose(-1, -2)         # 逻辑 [Seg, F, H]
    gact = torch.empty((rows_pad, t.F), dtype=torch.bfloat16, device=dev)
    moonep_fc2_gemm[(op.num_cores, 1, 1)](
        ws.dy_recv, dn_t, gact,
        d(seg_counts), d(seg_offsets),
        t.F, t.H,
        ws.dy_recv.stride(0), ws.dy_recv.stride(1),
        dn_t.stride(0), dn_t.stride(1), dn_t.stride(2),
        gact.stride(0), gact.stride(1),
        NUM_PROGRAM_CORES=op.num_cores, SEG=t.seg,
        BLOCK_SIZE_M=op.block, BLOCK_SIZE_N=op.block, BLOCK_SIZE_K=op.block,
    )

    # ---- B-2a：swiglu_bwd（classic 复用；w 在此乘入，dscale=∂L/∂w）----
    dAB, dscale = swiglu_bwd_triton(
        gact, saved["fc1_out"], saved["rw_recv"][:rows_pad].contiguous())

    # ---- B-2b：FC1 dgrad → gy ----
    # gy[m,h] = Σ_c dAB[m,c]·gu_phys[h,c] → b[c_red, h] = phys + h·2F + c
    # ⇒ 传 gate_up 原样（stride_out=2F, stride_red=1）——转置方向与 FC1
    # 前向相反（CASE-13 纪律：以 b_ptrs 步长公式推导）
    gu_t = ws.gate_up
    gy = torch.empty((rows_pad, t.H), dtype=torch.bfloat16, device=dev)
    moonep_fc2_gemm[(op.num_cores, 1, 1)](
        dAB, gu_t, gy,
        d(seg_counts), d(seg_offsets),
        t.H, 2 * t.F,
        dAB.stride(0), dAB.stride(1),
        gu_t.stride(0), gu_t.stride(1), gu_t.stride(2),
        gy.stride(0), gy.stride(1),
        NUM_PROGRAM_CORES=op.num_cores, SEG=t.seg,
        BLOCK_SIZE_M=op.block, BLOCK_SIZE_N=op.block, BLOCK_SIZE_K=op.block,
    )

    # ---- B-2c：gy 推回 + K 求和 → dx（combine kernel 换指针）----
    moonep_combine_push[(op.num_cores, 1, 1)](
        gy, outs["src_info"], ws.grad_buf, rows_pad,
        NvS=t.NvS, H=t.H, NUM_PROGRAM_CORES=op.num_cores)
    dx = torch.empty((t.S, t.H), dtype=torch.bfloat16, device=dev)
    moonep_topk_reduce[(1, 1, 1)](
        ws.grad_buf, dx, N=t.N, K=t.K, H=t.H,
        BLOCK_T=8, BLOCK_H=max(16, 1 << (t.H - 1).bit_length()))

    # ---- dw：dscale 在【接收方】，须按源归集（v1 小 allgather；B-4 设备化）----
    import torch.distributed as dist

    rows_pad_all = [int(cu_all[r][-1]) for r in range(t.R)]
    # dscale 定长收集（rows_pad_max 全缓冲；HCCL 须设备张量）
    ds_send = torch.zeros(t.rows_pad_max, dtype=torch.bfloat16, device=dev)
    ds_send[:rows_pad] = dscale
    ds_l = [torch.empty_like(ds_send) for _ in range(t.R)]
    dist.all_gather(ds_l, ds_send, group=op.ep_group)
    si_l = [torch.empty(t.NvS, dtype=torch.int32, device=dev)
            for _ in range(t.R)]
    dist.all_gather(si_l, outs["src_info"], group=op.ep_group)
    ds_l = [x.cpu() for x in ds_l]
    si_l = [x.cpu() for x in si_l]
    # offv 空间是【每个源 rank 各一份 [0,N)】——只取 si 编码的源 == 本 rank
    # 的条目（CASE-16：按 VM 属主遍历会把 R 份 offv 叠进同一 N 槽）
    dw_flat = torch.zeros(t.N, dtype=torch.float32)
    for x in range(t.R):
        si_x = si_l[x][:rows_pad_all[x]].to(torch.int64)
        msk = (si_x >= 0) & (si_x // t.NvS == op.rank)
        dw_flat[si_x[msk] % t.NvS] = \
            ds_l[x][:rows_pad_all[x]].to(torch.float32)[msk]
    dw = dw_flat.view(t.S, t.K).clone()

    return {"dx": dx, "dw": dw, "gact": gact, "dAB": dAB, "gy": gy,
            "dscale": dscale}
