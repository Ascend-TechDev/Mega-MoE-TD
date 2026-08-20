# coding=utf-8
"""M4 / L2：MoonEP 前向端到端对拍（2 rank）——纵向打通验收。

MoonepForward.forward 全链（planning → prefetch → zero_fill → dispatch_fc1
→ weighted_swiglu → FC2+combine）vs 宿主 fp32 参照（逐 entry 精确 FFN +
加权和求和），OUTPUT 容差（4e-2/4e-2）。

路由用 skew（rank0 重压 e0）保证副本槽路径被真实走到。
"""

import functools

import pytest
import torch
import torch.distributed as dist

from mega_moe.ops.moonep_forward import MoonepForward
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace

_S, _K, _EPN, _H, _F, _B, _TP = 64, 2, 2, 256, 128, 1, 2


def _silu(x):
    return x * torch.sigmoid(x)


def _reference(topk, rw, x, gu_all, dn_all, epn, E):
    """宿主 fp32 参照：out[t] = Σ_k w·(dn_e @ (silu(gate)·up))。"""
    S, K = topk.shape
    H, F = x.shape[1], gu_all.shape[2] // 2
    out = torch.zeros(S, H, dtype=torch.float32)
    for t in range(S):
        for k in range(K):
            e = int(topk[t, k])
            gu = gu_all[e // epn, e % epn].to(torch.float32)   # [H, 2F]
            dn = dn_all[e // epn, e % epn].to(torch.float32)   # [H, F]
            y = x[t].to(torch.float32)
            gate = y @ gu[:, :F]
            up = y @ gu[:, F:]
            act = _silu(gate) * up
            out[t] += float(rw[t, k]) * (act @ dn.T)
    return out


def _worker(rank, world_size):
    import tests._moe_testkit as kit

    device = f"npu:{rank}"
    ep_group = dist.group.WORLD
    E = world_size * _EPN
    epn = _EPN

    g = torch.Generator().manual_seed(23)
    topk_l, hs_l, rw_l = [], [], []
    for r in range(world_size):
        if r == 0:
            logits = torch.zeros(E)
            logits[0] = 4.0
            pick = torch.multinomial(torch.softmax(logits, -1)
                                     .repeat(_S, 1), 1, generator=g)
        else:
            pick = torch.randint(0, E, (_S, 1), generator=g)
        topk = pick.reshape(_S, 1).repeat(1, _K)
        # 打破 K 份相同：第二份独立采样（保证 dup/跨卡多样性）
        pick2 = torch.randint(0, E, (_S,), generator=g)
        topk[:, 1] = pick2
        topk_l.append(topk.to(torch.int32))
        hs_l.append((torch.randn(_S, _H, generator=g) * 0.5).to(torch.bfloat16))
        rw_l.append(torch.rand(_S, _K, generator=g).to(torch.float32))

    # 全组权重（host 留参照；e 的真身在 rank e//epn）
    gu_all = torch.stack([
        (torch.randn(epn, _H, 2 * _F,
                     generator=torch.Generator().manual_seed(500 + r)) * 0.1
        ).to(torch.bfloat16) for r in range(world_size)])
    dn_all = torch.stack([
        (torch.randn(epn, _H, _F,
                     generator=torch.Generator().manual_seed(600 + r)) * 0.1
        ).to(torch.bfloat16) for r in range(world_size)])

    topo = MoonepTopology(S=_S, K=_K, E=E, R=world_size, B=_B, H=_H, F=_F,
                          token_padding=_TP, dispatch_block_m=128)
    heap = max(MoonepWorkspace.required_bytes(topo) * 2, 256 << 20)
    with kit.aclshmem_session(rank, world_size, heap):
        op = MoonepForward(ep_group, max_tokens_per_rank=_S, hidden_size=_H,
                           ffn_dim=_F, top_k=_K, num_experts=E, num_slots=_B,
                           token_padding=_TP, num_cores=8, block_size=128)
        try:
            op.register_weights(gu_all[rank].to(device),
                                dn_all[rank].to(device))
            out = op.forward(hs_l[rank].to(device), topk_l[rank].to(device),
                             rw_l[rank].to(device))
            op.sync()

            ref = _reference(topk_l[rank], rw_l[rank], hs_l[rank],
                             gu_all, dn_all, epn, E)
            got = out.to(torch.float32).cpu()
            diff = (got - ref).abs()
            rel = diff / ref.abs().clamp_min(1e-3)
            ok = torch.allclose(got, ref, rtol=4e-2, atol=4e-2)
            max_abs = float(diff.max())
            print(f"[rank{rank}] L2 fwd ok={ok} max_abs={max_abs:.4f} "
                  f"max_rel={float(rel.max()):.3f}", flush=True)
            assert ok, (f"前向输出偏差超限：max_abs={max_abs:.4f}")
            assert out.dtype == torch.bfloat16 and out.shape == (_S, _H)
        finally:
            op.finalize()


@pytest.mark.dist
@pytest.mark.npu
def test_moonep_forward_e2e_r2(dist_test):
    dist_test(functools.partial(_worker), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
