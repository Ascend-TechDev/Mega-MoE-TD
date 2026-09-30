"""Host checks for route-to-token gradient gathering, without device launches."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("topk", [1, 4, 16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("routing", ["permutation", "duplicates", "empty"])
def test_dispatch_gather_matches_route_expansion(monkeypatch, topk, strided, routing):
    module = importlib.import_module("mega_moe.kernels.dispatch_fc2_bwd")
    generator = torch.Generator().manual_seed(31)
    tokens, hidden = 11, 7
    base = torch.randn(tokens, hidden * 2, generator=generator, dtype=torch.bfloat16)
    dy = base[:, ::2] if strided else base[:, :hidden].contiguous()
    routes = torch.randperm(tokens * topk, generator=generator).to(torch.int32)
    if routing == "duplicates":
        routes = routes[torch.tensor([0, 0, 3, 2, 3, 1])]
    elif routing == "empty":
        routes = routes[:0]
    original_routes = routes.clone()
    cache = {"bwd_expert_sort": routes, "total_send": routes.numel()}
    monkeypatch.setattr(module, "_dispatch_static_maps", lambda saved: cache)
    for grad in (dy, dy.neg()):
        expected = grad.repeat_interleave(topk, dim=0)[routes.long()]
        prepared = module._prepare_dispatch_fc2_bwd({"topk": topk}, grad)
        assert torch.equal(prepared["gco"], expected)
        assert prepared["gco"].is_contiguous()
        assert prepared["gco"].dtype == grad.dtype
        assert prepared["gco"].shape == (routes.numel(), hidden)
        assert prepared["bwd_expert_sort"] is routes
        assert "gco" not in cache
        assert torch.equal(routes, original_routes)
