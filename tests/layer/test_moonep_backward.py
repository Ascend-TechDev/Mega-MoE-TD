# coding=utf-8
"""M4b B-1/B-2：MoonEP 反向 dx/dw 对拍（2 rank）——autograd oracle。

前向（op.forward）→ dy → launch_moonep_backward_dx → (dx, dw)；
oracle = 宿主 fp32 可微参照对 (x, w) 的 autograd 梯度（GRAD 容差
2e-2/1e-2）。
"""

import functools

import pytest
import torch
import torch.distributed as dist

from mega_moe.ops.moonep_backward import launch_moonep_backward_dx
from mega_moe.ops.moonep_forward import MoonepForward
from mega_moe.runtime.moonep_workspace import MoonepTopology, MoonepWorkspace

_S, _K, _EPN, _H, _F, _B, _TP = 64, 2, 2, 256, 128, 1, 2


def _reference_autograd(topk, rw, x, gu_all, dn_all, epn):
    """fp32 可微参照（x/rw 为叶子）。返回 out [S,H]。"""
    S, K = topk.shape
    H, F = x.shape[1], gu_all.shape[3] // 2
    out = torch.zeros(S, H, dtype=torch.float32)
    for t_i in range(S):
        for k in range(K):
            e = int(topk[t_i, k])
            gu = gu_all[e // epn, e % epn]
            dn = dn_all[e // epn, e % epn]
            y = x[t_i]
            gate = y @ gu[:, :F]
            up = y @ gu[:, F:]
            act = torch.nn.functional.silu(gate) * up * rw[t_i, k]
            out[t_i] = out[t_i] + act @ dn.T
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
        topk[:, 1] = torch.randint(0, E, (_S,), generator=g)
        topk_l.append(topk.to(torch.int32))
        hs_l.append((torch.randn(_S, _H, generator=g) * 0.5).to(torch.bfloat16))
        rw_l.append(torch.rand(_S, _K, generator=g).to(torch.float32))
    gu_all = torch.stack([
        (torch.randn(epn, _H, 2 * _F,
                     generator=torch.Generator().manual_seed(500 + r)) * 0.1
        ).to(torch.bfloat16) for r in range(world_size)])
    dn_all = torch.stack([
        (torch.randn(epn, _H, _F,
                     generator=torch.Generator().manual_seed(600 + r)) * 0.1
        ).to(torch.bfloat16) for r in range(world_size)])

    # ---- autograd oracle（宿主 fp32，本 rank 的 x/w 为叶子）----
    x_leaf = hs_l[rank].to(torch.float32).requires_grad_(True)
    w_leaf = rw_l[rank].clone().requires_grad_(True)
    gu_f = gu_all.to(torch.float32)
    dn_f = dn_all.to(torch.float32)
    ref = _reference_autograd(topk_l[rank], w_leaf, x_leaf, gu_f, dn_f, epn)
    dy_cpu = (torch.randn(_S, _H,
                          generator=torch.Generator().manual_seed(99)) * 0.3)
    dx_ref, dw_ref = torch.autograd.grad(ref, (x_leaf, w_leaf),
                                         grad_outputs=dy_cpu)

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
            _out = op.forward(hs_l[rank].to(device), topk_l[rank].to(device),
                              rw_l[rank].to(device))
            op.sync()
            bwd = launch_moonep_backward_dx(op, dy_cpu.to(device).to(
                torch.bfloat16))
            op.sync()

            from tests._numeric import GRAD_ATOL, GRAD_RTOL, cmp_grad

            dx = bwd["dx"].to(torch.float32).cpu()
            dw = bwd["dw"].cpu()
            # 仓库标准梯度比对（全局 max 分母；dscale 为 classic bf16 设计）
            ok_dx, *_ = cmp_grad("dx", dx, dx_ref, GRAD_RTOL, GRAD_ATOL)
            ok_dw, *_ = cmp_grad("dw", dw, dw_ref, GRAD_RTOL, GRAD_ATOL)
            mdx = float((dx - dx_ref).abs().max())
            mdw = float((dw - dw_ref).abs().max())
            print(f"[rank{rank}] BWD dx_ok={ok_dx} max={mdx:.4f} | "
                  f"dw_ok={ok_dw} max={mdw:.4f}", flush=True)
            assert ok_dx, f"dx 偏差 {mdx:.4f}"
            assert ok_dw, f"dw 偏差 {mdw:.4f}"
        finally:
            op.finalize()


@pytest.mark.dist
@pytest.mark.npu
def test_moonep_backward_dx_dw_r2(dist_test):
    dist_test(functools.partial(_worker), world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
