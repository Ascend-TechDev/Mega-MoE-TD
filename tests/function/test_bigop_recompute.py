"""Check the recompute baseline without cached expert inputs/activations."""

import importlib

import pytest
import torch
import torch.nn.functional as F


def test_bigop_recompute_without_forward_caches(monkeypatch):
    pytest.importorskip("bigop")
    module = importlib.import_module("mega_moe._goldens.bigop_ref")

    def swiglu(x, dim=-1):
        gate, up = x.chunk(2, dim=dim)
        return F.silu(gate) * up

    def swiglu_backward(grad, x, dim=-1):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            return torch.autograd.grad(swiglu(x, dim), x, grad)[0]

    def matmul(x, weights, counts):
        return torch.cat([rows @ weight for rows, weight in
                          zip(x.split(counts.tolist()), weights)])

    def wgrad(x, grad, counts):
        return torch.stack([rows.T @ dy for rows, dy in
                            zip(x.split(counts.tolist()), grad.split(counts.tolist()))])

    def all_to_all(output, source, **kwargs):
        output.copy_(source)

    monkeypatch.setattr(module.torch_npu, "npu_swiglu", swiglu)
    monkeypatch.setattr(module.torch_npu, "npu_swiglu_backward", swiglu_backward)
    monkeypatch.setattr(module, "_grouped_matmul", matmul)
    monkeypatch.setattr(module, "_grouped_wgrad", wgrad)
    monkeypatch.setattr(module.dist, "all_to_all_single", all_to_all)
    generator = torch.Generator().manual_seed(71)
    hidden = torch.randn(4, 5, generator=generator)
    dy = torch.randn(4, 5, generator=generator)
    sort = torch.tensor([6, 7, 2, 3, 0, 1, 4, 5])
    local_sort = torch.tensor([3, 0, 5, 6, 7, 1, 2, 4])
    saved = dict(
        batch_size=4, hidden_dim=5, topk=2, total_recv=8, total_send=8,
        output=hidden, sort_idxs=sort, local_sort_idxs=local_sort,
        inv_sort=sort.argsort(), inv_local=local_sort.argsort(),
        splits_send_list=[8], splits_recv_list=[8], ep_group=None,
        expert_counts=torch.tensor([4, 4]),
        fc1_output=torch.randn(8, 6, generator=generator),
        fc1_combined=torch.randn(2, 6, 5, generator=generator),
        fc2=torch.randn(2, 5, 3, generator=generator),
        recv_weights_sorted=torch.rand(8, generator=generator),
    )
    previous = None
    for offset in (0.0, 0.5):
        source = hidden + offset
        saved["fc1_output"] = saved["fc1_output"] + offset
        cached = dict(saved,
                      recv_hidden_sorted=source[sort // 2][local_sort],
                      swiglu_out_weighted=swiglu(saved["fc1_output"])
                      * saved["recv_weights_sorted"].unsqueeze(-1))
        expected = module.moe_backward_bigop(cached, dy)
        actual = module.moe_backward_bigop(saved, dy, recompute=True,
                                          hidden_states=source)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        if previous is not None:
            assert not torch.equal(actual["grad_fc1_1"], previous)
        previous = actual["grad_fc1_1"]
