# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Matched full/trimmed Kimi imbalance without duplicate top-k experts."""

import torch


KIMI_OWNER_COUNTS = (19, 27, 11, 11, 15, 15, 15, 15)


def kimi_skewed_routes(tokens: int, num_experts: int, topk: int, rank: int):
    """Return CPU int32 routes with the moderate-wide profile's owner loads.

    Distribute each owner's 128-route quota over complete tokens first, then
    cycle through its experts. The original flat owner-period construction
    repeats experts within a token when there are only four experts per owner.
    """
    world = len(KIMI_OWNER_COUNTS)
    if topk not in (8, 16) or num_experts % world:
        raise ValueError("Kimi skewed routes require W8 and top-k 8 or 16")
    period = sum(KIMI_OWNER_COUNTS) // topk
    if tokens % period:
        raise ValueError("tokens must contain whole Kimi owner periods")
    epn = num_experts // world
    quotas = [[count // period for count in KIMI_OWNER_COUNTS] for _ in range(period)]
    extra = 0
    for owner, count in enumerate(KIMI_OWNER_COUNTS):
        for _ in range(count % period):
            quotas[extra % period][owner] += 1
            extra += 1
    if max(max(row) for row in quotas) > epn:
        raise ValueError("too few local experts for distinct per-token choices")
    cursor = [rank % epn] * world
    routes = []
    for token in range(tokens):
        row = []
        for owner, count in enumerate(quotas[token % period]):
            row.extend(owner * epn + (cursor[owner] + j) % epn for j in range(count))
            cursor[owner] += count
        routes.append(row)
    return torch.tensor(routes, dtype=torch.int32)
