# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Host-only tests for the receive-capacity-factor default resolution.

The MoonEP default is the proven tight bound 2.0 (B.0-B.3 pins every
destination's receive at S*topk; the backward wgrad pad adds at most one
more S*topk), the plain-EP default stays world_size (unbalanced receive),
and an explicit factor is always honored as-is."""

import pytest

from mega_moe.config import MoEForwardConfig


@pytest.mark.parametrize("world", [2, 8, 16, 64])
def test_default_plain_ep_is_world(world):
    cfg = MoEForwardConfig(enable_moonep=False)
    assert cfg.resolved_receive_capacity_factor(world) == float(world)


@pytest.mark.parametrize("world", [2, 8, 16, 64])
def test_default_moonep_is_tight_bound(world):
    cfg = MoEForwardConfig(enable_moonep=True)
    # Routing-independent: the bound does not depend on world_size once
    # balancing is active (only the unbalanced path does).
    assert cfg.resolved_receive_capacity_factor(world) == 2.0


@pytest.mark.parametrize("moonep", [False, True])
@pytest.mark.parametrize("factor", [1.0, 1.25, 2.0, 16.0])
def test_explicit_factor_wins(moonep, factor):
    cfg = MoEForwardConfig(
        enable_moonep=moonep, receive_capacity_factor=factor)
    assert cfg.resolved_receive_capacity_factor(16) == float(factor)


@pytest.mark.parametrize(
    "s,topk", [(1024, 8), (64, 8), (2048, 1)])
def test_tight_bound_covers_invariant_plus_pad(s, topk):
    # The two quantities the window must cover under MoonEP: the pinned
    # per-destination receive (== S*topk) and the wgrad pad bound
    # (max_rows_w <= the rank's own receive == S*topk).
    cfg = MoEForwardConfig(enable_moonep=True)
    rows = cfg.resolved_receive_capacity_factor(16) * s * topk
    assert rows >= 2 * s * topk


def test_below_one_is_rejected():
    with pytest.raises(ValueError):
        MoEForwardConfig(receive_capacity_factor=0.5)
