# 分阶段核查 rank0：vm→fc1→act→fc2→combine_buf→out
import functools
import pytest, torch
import torch.distributed as dist
import mega_moe.moonep_ref  # noqa
from mega_moe.ops.moonep_forward import MoonepForward
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace
from mega_moe.runtime.moonep_routing import build_moonep_segment_meta
from mega_moe.kernels.weighted_swiglu import weighted_swiglu_forward
import sys
sys.path.insert(0, '.')
from tests.layer.test_moonep_forward import _reference, _silu, _S, _K, _EPN, _H, _F, _B, _TP

def _worker(rank, world_size):
    import tests._moe_testkit as kit
    device = f"npu:{rank}"; ep_group = dist.group.WORLD
    E = world_size * _EPN; epn = _EPN
    g = torch.Generator().manual_seed(23)
    topk_l, hs_l, rw_l = [], [], []
    for r in range(world_size):
        if r == 0:
            logits = torch.zeros(E); logits[0] = 4.0
            pick = torch.multinomial(torch.softmax(logits,-1).repeat(_S,1),1,generator=g)
        else:
            pick = torch.randint(0,E,(_S,1),generator=g)
        topk = pick.reshape(_S,1).repeat(1,_K)
        topk[:,1] = torch.randint(0,E,(_S,),generator=g)
        topk_l.append(topk.to(torch.int32))
        hs_l.append((torch.randn(_S,_H,generator=g)*0.5).to(torch.bfloat16))
        rw_l.append(torch.rand(_S,_K,generator=g).to(torch.float32))
    gu_all = torch.stack([(torch.randn(epn,_H,2*_F,generator=torch.Generator().manual_seed(500+r))*0.1).to(torch.bfloat16) for r in range(world_size)])
    dn_all = torch.stack([(torch.randn(epn,_H,_F,generator=torch.Generator().manual_seed(600+r))*0.1).to(torch.bfloat16) for r in range(world_size)])
    topo = MoonepTopology(S=_S,K=_K,E=E,R=world_size,B=_B,H=_H,F=_F,token_padding=_TP,dispatch_block_m=128)
    heap = max(MoonepWorkspace.required_bytes(topo)*2, 256<<20)
    with kit.aclshmem_session(rank, world_size, heap):
        op = MoonepForward(ep_group, max_tokens_per_rank=_S, hidden_size=_H, ffn_dim=_F,
                           top_k=_K, num_experts=E, num_slots=_B, token_padding=_TP, num_cores=8, block_size=128)
        try:
            op.register_weights(gu_all[rank].to(device), dn_all[rank].to(device))
            out = op.forward(hs_l[rank].to(device), topk_l[rank].to(device), rw_l[rank].to(device))
            op.sync()
            # 复算 planning 拿段表
            cu = op._outs["cu_seqlens"].cpu()
            seg_c, seg_o, _ = build_moonep_segment_meta(cu, op._outs["experts_to_copy"].cpu()[rank], rank, epn, E, _B)
            rows_pad = int(seg_o[-1])
            vm = op.ws.vm[:rows_pad].cpu().to(torch.float32)
            gu_c = op.ws.gate_up.cpu().to(torch.float32)
            dn_c = op.ws.down.cpu().to(torch.float32)
            # 期望 vm：按 src_info 反推每行的 (sr, offv) → payload
            si = op._outs["src_info"][:rows_pad].cpu().to(torch.int64)
            NvS = topo.NvS
            vm_exp = torch.zeros(rows_pad, _H)
            for r0 in range(rows_pad):
                info = int(si[r0])
                sr, offv = info // NvS, info % NvS
                vm_exp[r0] = hs_l[sr][offv // _K].to(torch.float32)
            print(f"[rank{rank}] stage0 vm ok={torch.allclose(vm, vm_exp, atol=2e-2)}", flush=True)
            # fc1/act/fc2 期望（按段）
            seg_expert = torch.cat([torch.arange(rank*epn,(rank+1)*epn), op._outs["experts_to_copy"].cpu()[rank].to(torch.int64)])
            for s0 in range(seg_c.numel()):
                c0,o0 = int(seg_c[s0]), int(seg_o[s0])
                if c0==0: continue
                e = int(seg_expert[s0])
                gu = gu_all[e//epn, e%epn].to(torch.float32)
                exp_fc1 = vm_exp[o0:o0+c0] @ gu
                # act 权重按行（rw_recv）
                w = op.ws.routing_weight_recv[o0:o0+c0].cpu()
                gate, up = exp_fc1[:, :_F], exp_fc1[:, _F:]
                exp_act = (_silu(gate)*up) * w[:,None]
                dn = dn_all[e//epn, e%epn].to(torch.float32)
                exp_fc2 = exp_act @ dn.T
                print(f"[rank{rank}] seg{s0}(e{e}) fc1-like ok={torch.allclose(vm[o0:o0+c0] @ gu_c[s0], exp_fc1, atol=3e-2)}", flush=True)
            # combine_buf 期望：buf[offv] = fc2(row of that entry)
            # 直接验 out 每错 token 的 k 分量来源
            ref = _reference(topk_l[rank], rw_l[rank], hs_l[rank], gu_all, dn_all, epn, E)
            got = out.to(torch.float32).cpu()
            bad = (got-ref).abs().max(dim=1).values > 0.5
            nbad = int(bad.sum())
            print(f"[rank{rank}] out bad_tokens={nbad}/{_S}", flush=True)
            if nbad:
                t0 = int(torch.nonzero(bad)[0])
                print(f"[rank{rank}] first bad t={t0} topk={topk_l[rank][t0].tolist()} rw={rw_l[rank][t0].tolist()} got[:4]={got[t0,:4].tolist()} ref[:4]={ref[t0,:4].tolist()}", flush=True)
        finally:
            op.finalize()

@pytest.mark.dist
@pytest.mark.npu
def test_stages(dist_test):
    dist_test(functools.partial(_worker), world_size=2)
