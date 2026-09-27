# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Route addressing and masking checks; NPU tests cover BF16 numerics."""

import numpy as np
import pytest

from tests.function.test_single_kernel_routing_metadata import (
    Language, Pointer, production_helpers, tensor,
)


class ReductionLanguage(Language):
    float32 = np.float32
    # Integer-valued inputs make the host sum exact. The hardware correctness
    # gate separately checks the BF16 output after changing accumulation order.
    bfloat16 = np.float32
    static_range = staticmethod(range)

    def __init__(self):
        super().__init__()
        self.loads = 0

    def load(self, pointer, mask=True, other=0):
        offsets, active = np.broadcast_arrays(pointer.offsets, np.asarray(mask, dtype=bool))
        assert np.all((offsets[active] >= 0) & (offsets[active] < pointer.values.size))
        values = np.full(offsets.shape, other, dtype=pointer.values.dtype)
        values[active] = pointer.values[offsets[active]]
        self.loads += 1
        return tensor(values)

    def store(self, pointer, values, mask=True):
        offsets, values, active = np.broadcast_arrays(
            pointer.offsets, values, np.asarray(mask, dtype=bool))
        selected = offsets[active]
        assert np.all((selected >= 0) & (selected < pointer.values.size))
        assert np.unique(selected).size == selected.size
        pointer.values[selected] = values[active]
        pointer.writes.extend((self.lane, int(offset)) for offset in selected)


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("topk", [1, 3, 4, 5, 8, 16, 17])
@pytest.mark.parametrize("tokens", [0, 7, 65])
@pytest.mark.parametrize("hidden", [17, 65])
@pytest.mark.parametrize("capacity_ok", [False, True])
def test_reduce_restores_routes_and_masks_tail(
        topk, tokens, hidden, capacity_ok, interleave):
    language = ReductionLanguage()
    helpers = production_helpers(("_reduce_topk_rows",), {
        "tl": language,
        "range": lambda *args: (tensor(np.int64(i)) for i in range(*args)),
    })
    rng = np.random.default_rng(426 + topk + tokens + hidden)
    rows = tokens * topk
    combined = rng.integers(-3, 4, size=(rows, hidden)).astype(np.float32)
    routes = rng.permutation(rows).astype(np.int32)
    routes[::3] = -1
    routes[1::7] = -5
    if tokens:
        routes[-topk:] = -1
    output = Pointer(np.full(tokens * hidden, np.nan, dtype=np.float32))
    for worker in range(8):
        language.lane = worker
        helpers._reduce_topk_rows(
            worker, Pointer(combined.reshape(-1)), Pointer(routes), output,
            tokens, capacity_ok, 8, hidden, topk, 32, interleave)

    expected = np.zeros((tokens, hidden), dtype=np.float32)
    if capacity_ok:
        for token in range(tokens):
            for route in routes[token * topk:(token + 1) * topk]:
                if route >= 0:
                    expected[token] += combined[route]
    else:
        assert language.loads == 0
    np.testing.assert_array_equal(output.values.reshape(tokens, hidden), expected)
    assert sorted(offset for _, offset in output.writes) == list(range(tokens * hidden))
