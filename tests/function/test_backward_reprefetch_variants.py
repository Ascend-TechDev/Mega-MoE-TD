"""A/B calls restore their protocol and preserve the ideal-control boundary."""

import os
from types import SimpleNamespace

import pytest

from benchmark.layer._backward_moonep_case import ReprefetchVariant


def test_reprefetch_variants_keep_independent_sync_modes(monkeypatch):
    for name in ("FUSED", "PIPELINE", "FINE_SYNC"):
        monkeypatch.setenv("MOE_MEGA_REPREFETCH_" + name, "1")
    for name in ("MOE_MEGA_REPREFETCH_TRANSPORT", "MOE_MEGA_SAFE_WGRAD",
                 "MOE_SAVED_RECOMPUTE"):
        monkeypatch.setenv(name, "0")

    def owner():
        return tuple(os.environ[key] for key in (
            "MOE_MEGA_REPREFETCH_FUSED", "MOE_MEGA_REPREFETCH_PIPELINE",
            "MOE_MEGA_REPREFETCH_FINE_SYNC", "MOE_SAVED_RECOMPUTE"))

    fine = ReprefetchVariant(owner, "udma")
    control = ReprefetchVariant(owner, "udma_fused_pipeline", fine_sync=False)
    serial = ReprefetchVariant(owner, "udma_fused_barrier")
    standalone = ReprefetchVariant(owner, "udma_standalone")
    saved = ReprefetchVariant(owner, "udma", recompute=False)
    for _ in range(3):
        assert fine() == ("1", "1", "1", "1")
        assert serial() == ("1", "0", "0", "1")
        assert standalone() == ("0", "0", "0", "1")
        assert saved() == ("1", "1", "1", "0")
        assert control() == ("1", "1", "0", "1")


def test_pipeline_remains_opt_in(monkeypatch):
    monkeypatch.setenv("MOE_MEGA_REPREFETCH_FUSED", "1")
    monkeypatch.delenv("MOE_MEGA_REPREFETCH_PIPELINE", raising=False)
    monkeypatch.delenv("MOE_MEGA_REPREFETCH_FINE_SYNC", raising=False)
    variant = ReprefetchVariant(lambda: None, "udma")
    assert variant.fused and not variant.pipeline and not variant.fine_sync
    with pytest.raises(ValueError, match="requires the fused pipeline"):
        ReprefetchVariant(lambda: None, "udma", preloaded=True)


@pytest.mark.parametrize("phase", ["MOE_MEGA_P1", "MOE_MEGA_P23"])
@pytest.mark.parametrize("recompute", [False, True])
def test_fine_sync_rejects_missing_producer_before_submission(monkeypatch, phase, recompute):
    from mega_moe.kernels import mega_bwd

    for name in ("REPREFETCH", "REPREFETCH_FUSED", "REPREFETCH_PIPELINE",
                 "REPREFETCH_FINE_SYNC", "P1", "P23"):
        monkeypatch.setenv("MOE_MEGA_" + name, "1")
    monkeypatch.setenv("MOE_MEGA_REPREFETCH_TRANSPORT", "udma")
    monkeypatch.setenv("MOE_SAVED_RECOMPUTE", str(int(recompute)))
    monkeypatch.setenv(phase, "0")
    saved = dict(use_moonep=True, ep_rank=0, world_size=8,
                 active_physical_experts_per_rank=5, home_experts_per_rank=4)
    # No weights or device buffers: rejection must precede prefetch submission.
    with pytest.raises(ValueError, match="fine sync requires its P1 and P23 producers"):
        mega_bwd.mega_backward_triton(saved, SimpleNamespace(device="cpu"), None)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_preloaded_preserves_launches_and_restores_hooks(monkeypatch, recompute, fail):
    from mega_moe.kernels import mega_bwd

    for name in ("FUSED", "PIPELINE", "FINE_SYNC"):
        monkeypatch.setenv("MOE_MEGA_REPREFETCH_" + name, "1")
    monkeypatch.setenv("MOE_SAVED_RECOMPUTE", "1")
    monkeypatch.setenv("MOE_MEGA_REPREFETCH_TRANSPORT", "udma")
    monkeypatch.setenv("MOE_MEGA_SAFE_WGRAD", "0")
    calls = []

    class Original:
        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(hash="same-pipeline")
            return launch

    original = Original()
    name = "kernel_moe_backward_mega_recompute" if recompute else "kernel_moe_backward_mega"
    monkeypatch.setattr(mega_bwd, "_kernel_replica_repush_udma", original)
    monkeypatch.setattr(mega_bwd, name, original)

    def owner():
        assert os.environ["MOE_MEGA_REPREFETCH_FINE_SYNC"] == "1"
        mega_bwd._kernel_replica_repush_udma[(1, 1, 1)](submit=1)
        kernel = getattr(mega_bwd, name)[(1, 1, 1)](
            repref_desc_count=65, repref_epoch=999)
        if fail:
            raise RuntimeError("test launch failure")
        return kernel

    control = ReprefetchVariant(owner, "udma", preloaded=True, recompute=recompute)
    control.preloaded_epoch = 123
    if fail:
        with pytest.raises(RuntimeError, match="test launch failure"):
            control()
    else:
        control()
    assert calls == [dict(submit=0), dict(repref_desc_count=0, repref_epoch=123)]
    assert control.early_kernel_hash == control.kernel_hash == "same-pipeline"
    assert getattr(mega_bwd, name) is original
    assert mega_bwd._kernel_replica_repush_udma is original


def test_unused_slot_probe_preserves_owner_flows_and_copy_multiplicity():
    import torch
    from benchmark.layer._backward_moonep_case import add_unused_replica_slots

    etc = torch.full((8, 4), -1, dtype=torch.int32)
    for peer, expert in ((2, 4), (3, 5), (4, 0), (5, 1), (6, 2), (7, 6)):
        etc[peer, 0] = expert
    extended = add_unused_replica_slots(etc)
    assert torch.equal(extended[etc >= 0], etc[etc >= 0])
    assert int((extended != etc).sum()) == 2
    assert extended[2, 1] == 7
    assert extended[4, 1] == 3
    assert len(set(extended[extended >= 0].tolist())) == 8
    assert torch.all(extended[:2] == -1)


def test_unused_slot_probe_large_case_adds_six_unique_experts():
    import torch
    from benchmark.layer._backward_moonep_case import add_unused_replica_slots

    etc = torch.full((8, 112), -1, dtype=torch.int32)
    for peer, start, count in ((2, 112, 21), (3, 133, 21), (7, 154, 5),
                               (4, 0, 6), (5, 6, 6), (6, 12, 6)):
        etc[peer, :count] = torch.arange(start, start + count)
    extended = add_unused_replica_slots(etc)
    assert int((extended != etc).sum()) == 6
    assert len(set(extended[extended >= 0].tolist())) == 71
    for peer, slot in torch.nonzero(extended != etc).tolist():
        assert int(extended[peer, slot]) // 112 == int(etc[peer, 0]) // 112
