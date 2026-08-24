# coding=utf-8
"""M3：dispatch（payload/权重散布 + FC1 GEMM 融合 kernel）对拍（2 rank）。

验收：
1. VM 行内容 == 宿主期望（全组 dst 解码散布，**dup 条目照发**——v1 语义）
   逐位（bf16）；
2. routing_weight_recv == 期望权重（fp32 逐位）；
3. padding 行 == 0（zero_fill 先序 launch 生效）；
4. fc1_output 合法行 ≈ 宿主 GEMM（fp32 参照，容差）——GEMM 半边为
   dispatch_fc1 原函数（Seg 替换 EPR），主要验证元数据接线正确。
"""

import functools

import pytest
import torch
import torch.distributed as dist

import mega_moe.moonep_ref  # noqa: F401
from mega_moe.kernels.moonep_dispatch import (
    MoonepDispatchState,
    launch_moonep_dispatch_fc1,
)
from mega_moe.kernels.moonep_planning import (
    MoonepPlanBuffers,
    launch_moonep_planning,
)
from mega_moe.kernels.moonep_zero_fill import launch_moonep_zero_fill
from mega_moe.runtime.moonep_routing import build_moonep_segment_meta
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace

_S, _K, _EPN, _H, _F, _B, _TP = 64, 2, 2, 256, 128, 1, 2
_BLOCK_M = _BLOCK_N = _BLOCK_K = 128


def _gen_inputs(S, K, E, R, H, seed=11):
    g = torch.Generator().manual_seed(seed)
    topks, tpes, hiddens, rws = [], [], [], []
    for r in range(R):
        if r == 0:
            logits = torch.zeros(E)
            logits[0] = 4.0
            pick = torch.multinomial(torch.softmax(logits, -1)
                                     .repeat(S * K, 1), 1,
                                     generator=g).reshape(-1)
        else:
            pick = torch.randint(0, E, (S * K,), generator=g)
        topk = pick.to(torch.int32)
        topks.append(topk)
        tpes.append(torch.bincount(topk.to(torch.int64), minlength=E)
                    .to(torch.int32))
        hiddens.append((torch.randn(S, H, generator=g) * 0.5)
                       .to(torch.bfloat16))
        rws.append(torch.rand(S * K, generator=g).to(torch.float32))
    return topks, tpes, hiddens, rws


def _expected_vm_and_weights(dst_all, hiddens, rws, NvS, H, K):
    """宿主期望：dup 负 dst 解码后照发（v1 语义），padding 行保持 0。"""
    R, N = dst_all.shape
    vm_exp = torch.zeros(R, NvS, H, dtype=torch.bfloat16)
    w_exp = torch.zeros(R, NvS, dtype=torch.float32)
    d = dst_all
    raw = torch.where(d < 0, -d - 1, d)
    for sr in range(R):
        dr = raw[sr] // NvS
        loff = raw[sr] - dr * NvS
        for x in range(R):
            m = dr == x
            vm_exp[x][loff[m]] = hiddens[sr][ (torch.nonzero(m).reshape(-1) // K) ]
            w_exp[x][loff[m]] = rws[sr][torch.nonzero(m).reshape(-1)]
    return vm_exp, w_exp


def _worker(rank, world_size):
    import tests._moe_testkit as kit

    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    E = world_size * _EPN
    topo = MoonepTopology(S=_S, K=_K, E=E, R=world_size, B=_B, H=_H, F=_F,
                          token_padding=_TP, dispatch_block_m=_BLOCK_M)
    N, NvS, epn = topo.N, topo.NvS, topo.epn
    topks, tpes, hiddens, rws = _gen_inputs(_S, _K, E, world_size, _H)

    heap = max(MoonepWorkspace.required_bytes(topo) * 2, 256 << 20)
    with kit.aclshmem_session(rank, world_size, heap):
        ws = MoonepWorkspace(topo, rank, device)
        pbufs = MoonepPlanBuffers(world_size, E, _B, N, NvS, device,
                                  tpe_all=ws.tpe_all, src_info=ws.src_info)
        state = MoonepDispatchState()
        try:
            gen = torch.Generator().manual_seed(100 + rank)
            gu_local = (torch.randn(epn, _H, 2 * _F, generator=gen)
                        * 0.1).to(torch.bfloat16)
            dn_local = (torch.randn(epn, _H, _F, generator=gen)
                        * 0.1).to(torch.bfloat16)
            gu_view, _ = ws.register_weights(gu_local.to(device),
                                             dn_local.to(device))
            del gu_local, dn_local

            outs = {
                "dst": torch.empty(N, dtype=torch.int32, device=device),
                "cu_seqlens": torch.empty(E + _B, dtype=torch.int32,
                                          device=device),
                "experts_to_copy": torch.empty(world_size, _B,
                                               dtype=torch.int32,
                                               device=device),
                "zero_fill_ranges": torch.empty(E + _B, 2, dtype=torch.int32,
                                                device=device),
                "remote_stats": torch.empty(2, dtype=torch.int32,
                                            device=device),
                "src_info": torch.empty(NvS, dtype=torch.int32,
                                        device=device),
            }
            launch_moonep_planning(
                pbufs, outs, topks[rank].to(device),
                rank=rank, world_size=world_size, ep_group=ep_group,
                S=_S, K=_K, E=E, B=_B, NvS=NvS, token_padding=_TP)

            # 全组 dst（宿主）
            dst_list = [torch.empty(N, dtype=torch.int32, device=device)
                        for _ in range(world_size)]
            dist.all_gather(dst_list, outs["dst"], group=ep_group)
            dst_all = torch.stack([t.to(torch.int64).cpu()
                                   for t in dst_list])

            launch_moonep_zero_fill(ws.vm, ws.routing_weight_recv,
                                    outs["zero_fill_ranges"], _H)
            fc1_out, rows_pad, _ep = launch_moonep_dispatch_fc1(
                ws.vm, ws.routing_weight_recv, ws.signal_mem, ws.gate_up,
                hiddens[rank].to(device), rws[rank].to(device), outs,
                dst_all, rank=rank, epn=epn, E=E, B=_B, NvS=NvS, K=_K, H=_H,
                state=state, num_cores=8,
                block_m=_BLOCK_M, block_n=_BLOCK_N, block_k=_BLOCK_K)
            torch.npu.synchronize(device)

            # ---- 校验 ----
            vm_exp, w_exp = _expected_vm_and_weights(dst_all, hiddens, rws,
                                                     NvS, _H, _K)
            assert torch.equal(ws.vm[:NvS].cpu(), vm_exp[rank]), \
                "VM 行内容 ≠ 期望散布"
            assert torch.equal(ws.routing_weight_recv[:NvS].cpu(),
                               w_exp[rank]), "权重散布 ≠ 期望"

            # fc1 数值粗检（合法段行；padding 行应为精确 0）
            cu = outs["cu_seqlens"].cpu()
            seg_counts, seg_offsets, _ = build_moonep_segment_meta(
                cu, outs["experts_to_copy"].cpu()[rank], rank, epn, E, _B)
            vm_cpu = ws.vm[:NvS].cpu().to(torch.float32)
            gu_cpu = ws.gate_up.cpu().to(torch.float32)
            got = fc1_out.cpu().to(torch.float32)
            for s in range(seg_counts.numel()):
                c, o = int(seg_counts[s]), int(seg_offsets[s])
                if c == 0:
                    continue
                exp = vm_cpu[o:o + c] @ gu_cpu[s]  # b[k,n]=phys[k,n] → out = vm @ W_phys
                ok = torch.allclose(got[o:o + c], exp, rtol=5e-2, atol=5e-2)
                assert ok, f"fc1 段{s} 数值偏差"
            # padding 行：zfr 区间行应为 0（fc1 输出 0 @ W = 0）
            zfr = outs["zero_fill_ranges"].cpu()
            for gi in range(zfr.shape[0]):
                st, cnt2 = int(zfr[gi, 0]), int(zfr[gi, 1])
                if cnt2 > 0:
                    assert float(got[st:st + cnt2].abs().sum()) == 0.0, \
                        "padding 行 fc1 输出非 0"
        finally:
            pbufs.finalize()
            ws.finalize()


@pytest.mark.dist
@pytest.mark.npu
def test_dispatch_bitexact_r2(dist_test):
    dist_test(functools.partial(_worker), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
