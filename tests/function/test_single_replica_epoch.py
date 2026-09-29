"""All writers of pooled ready slabs must reserve a distinct SET epoch."""

from types import SimpleNamespace

import pytest
import torch

from mega_moe.ops.forward import FusedMoEForward
from mega_moe.runtime.replica_weight_prefetch import ReplicaWeightBuffers


def buffers():
    return ReplicaWeightBuffers(None, None, 4, (8, 16), (8, 8), 0)


def operator(shared, tile=1, replica=1):
    return SimpleNamespace(
        enable_moonep=True, _replica_weight_buffers=shared,
        _tile_signal_epoch=tile, _replica_weight_epoch=replica,
        _replica_weight_cache_valid=True)


def launch_epoch(op):
    value = FusedMoEForward._reserve_single_signal_epoch(op)
    op._tile_signal_epoch += 1  # successful forward launch
    return value


def test_first_forward_then_early_backward_cannot_accept_stale_ready():
    shared = buffers()
    op = operator(shared)
    forward = launch_epoch(op)
    backward = shared.next_push_epoch()
    assert forward == 1
    assert backward > forward
    assert launch_epoch(op) > backward


def test_pooled_layers_and_saved_forward_share_one_sequence():
    shared = buffers()
    layers = [operator(shared), operator(shared, tile=20, replica=40)]
    observed = []
    for _ in range(4):
        for op in layers:
            observed.extend([launch_epoch(op), launch_epoch(op)])
            observed.append(shared.next_push_epoch())  # early backward
            observed.append(shared.next_push_epoch(floor=1))  # grad return
            saved = shared.next_push_epoch(floor=op._replica_weight_epoch - 1)
            op._replica_weight_epoch = saved + 1
            observed.append(saved)
            assert not op._replica_weight_cache_valid
    assert all(a < b for a, b in zip(observed, observed[1:]))
    assert observed[5] >= 40


def test_epoch_exhaustion_never_wraps_to_stale_values():
    shared = buffers()
    shared.push_epoch = torch.iinfo(torch.int32).max - 1
    with pytest.raises(RuntimeError, match="epoch exhausted"):
        launch_epoch(operator(shared))
    assert shared.push_epoch == torch.iinfo(torch.int32).max - 1


def test_non_moonep_tile_sequence_is_unchanged():
    op = operator(None, tile=7)
    op.enable_moonep = False
    assert launch_epoch(op) == 7
    assert launch_epoch(op) == 8
