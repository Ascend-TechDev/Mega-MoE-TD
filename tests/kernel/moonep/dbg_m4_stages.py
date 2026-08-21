# 分阶段核查（真实中间张量）：vm→fc1→act→fc2→combine_buf→out
# 用 op._saved 的实际 tensor 逐级比对，定位 flaky 的第一级。
import functools
import sys
import pytest, torch
import torch.distributed as dist
sys.path.insert(0, '.')
import mega_moe.moonep_ref  # noqa
from mega_moe.ops.moonep_forward import MoonepForward
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace
from mega_moe.runtime.moonep_routing import build_moonep_segment_meta
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
    heap = max(MoonepWorkspace.required_bytes(topo)*2,256<<20)
    with kit.aclshmem_session(rank, world_size, heap):
        op = MoonepForward(ep_group, max_tokens_per_rank=_S, hidden_size=_H, ffn_dim=_F,
                           top_k=_K, num_experts=E, num_slots=_B, token_padding=_TP, num_cores=8, block_size=128)
        try:
            op.register_weights(gu_all[rank].to(device), dn_all[rank].to(device))
            out = op.forward(hs_l[rank].to(device), topk_l[rank].to(device), rw_l[rank].to(device))
            op.sync()
            cu = op._outs["cu_seqlens"].cpu()
            seg_c, seg_o, _ = build_moonep_segment_meta(cu, op._outs["experts_to_copy"].cpu()[rank], rank, epn, E, _B)
            rows_pad = int(seg_o[-1]); NvS = topo.NvS
            si = op._outs["src_info"][:rows_pad].cpu().to(torch.int64)
            # ---- stage0: vm（按 src_info 反推）----
            vm = op.ws.vm[:rows_pad].cpu().to(torch.float32)
            vm_exp = torch.zeros(rows_pad, _H)
            for r0 in range(rows_pad):
                info = int(si[r0])
                if info < 0: continue
                vm_exp[r0] = hs_l[info // NvS][info % NvS // _K].to(torch.float32)
            s0 = torch.allclose(vm, vm_exp, atol=2e-2)
            # ---- stage1: fc1（真实张量）----
            fc1 = op._saved["fc1_out"][:rows_pad].cpu().to(torch.float32)
            seg_expert = torch.cat([torch.arange(rank*epn,(rank+1)*epn), op._outs["experts_to_copy"].cpu()[rank].to(torch.int64)])
            fc1_ok = True; worst = 0.0
            for s0i in range(seg_c.numel()):
                c0,o0 = int(seg_c[s0i]), int(seg_o[s0i])
                if c0==0: continue
                e = int(seg_expert[s0i])
                exp = vm_exp[o0:o0+c0] @ gu_all[e//epn,e%epn].to(torch.float32)
                d = float((fc1[o0:o0+c0]-exp).abs().max()); worst=max(worst,d)
                if d > 5e-2: fc1_ok = False
            # ---- stage2: act ----
            act = op._saved["act"][:rows_pad].cpu().to(torch.float32)
            w = op.ws.routing_weight_recv[:rows_pad].cpu()
            act_ok = True; worst_a = 0.0
            for s0i in range(seg_c.numel()):
                c0,o0 = int(seg_c[s0i]), int(seg_o[s0i])
                if c0==0: continue
                e = int(seg_expert[s0i])
                f1 = fc1[o0:o0+c0]  # 用真实 fc1（隔离下游错误）
                exp = (_silu(f1[:, :_F])*f1[:, _F:]) * w[o0:o0+c0,None].cpu()
                d = float((act[o0:o0+c0]-exp).abs().max()); worst_a=max(worst_a,d)
                if d > 5e-2: act_ok = False
            # ---- stage3: fc2（真实张量，act 为基）----
            fc2 = op._saved["fc2_out"][:rows_pad].cpu().to(torch.float32)
            fc2_ok = True; worst_f = 0.0
            for s0i in range(seg_c.numel()):
                c0,o0 = int(seg_c[s0i]), int(seg_o[s0i])
                if c0==0: continue
                e = int(seg_expert[s0i])
                exp = act[o0:o0+c0] @ dn_all[e//epn,e%epn].to(torch.float32).T
                d = float((fc2[o0:o0+c0]-exp).abs().max()); worst_f=max(worst_f,d)
                if d > 5e-2: fc2_ok = False
            # ---- stage4: combine_buf（每条目行 == 源处 fc2 行）----
            cb = op.ws.combine_buf.cpu().to(torch.float32)
            cb_exp = torch.zeros_like(cb)
            for r0 in range(rows_pad):
                info = int(si[r0])
                if info < 0: continue
                offv = info % NvS
                cb_exp[offv] = fc2[r0]
            cb_ok = bool((cb-cb_exp).abs().max() < 5e-2)
            # ---- stage5: out ----
            ref = _reference(topk_l[rank], rw_l[rank], hs_l[rank], gu_all, dn_all, epn, E)
            got = out.to(torch.float32).cpu()
            nbad = int((((got-ref).abs().max(dim=1).values) > 0.5).sum())
            print(f"[rank{rank}] STAGES vm={s0} fc1={fc1_ok}({worst:.3f}) act={act_ok}({worst_a:.3f}) "
                  f"fc2={fc2_ok}({worst_f:.3f}) cbuf={cb_ok} bad_out={nbad}", flush=True)
        finally:
            op.finalize()

@pytest.mark.dist
@pytest.mark.npu
def test_stages(dist_test):
    dist_test(functools.partial(_worker), world_size=2)
