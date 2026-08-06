# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared MoE test-shape definitions.

One dataclass describes every forward/backward test shape so the two sides
share field names and ordering. The field order follows the backward test's
historical tuple ``(name, ntokens, hidden, ffn, topk, num_experts)``:
``num_experts`` is explicit and now used by both forward and backward shapes.
``experts_per_rank`` remains supported for world-size scaling but is unused by
the current shape lists; DSV4 sets ``global_tokens`` so
:meth:`resolved_tokens_per_rank` divides it across ranks.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MoETestShape:
    label: str
    tokens: int                       # tokens_per_rank (used when global_tokens is None)
    hidden: int
    ffn: int
    topk: int
    num_experts: int = 0              # explicit (backward style); 0 => epr * world_size
    experts_per_rank: int = 0         # forward epr scaling (mutually exclusive with num_experts)
    drop_frac: float = 0.0
    global_tokens: Optional[int] = None  # DSV4 uses a global token count divided across ranks

    def resolved_num_experts(self, world_size: int) -> int:
        if self.num_experts:
            return self.num_experts
        return self.experts_per_rank * world_size

    def resolved_tokens_per_rank(self, world_size: int) -> int:
        if self.global_tokens is not None:
            if self.global_tokens % world_size != 0:
                raise ValueError("global token count must be divisible by world_size")
            return self.global_tokens // world_size
        return self.tokens


# ---------------------------------------------------------------------------
# Forward correctness shapes — all use explicit num_experts (unified with the
# backward convention). DSV4-smoke additionally sets global_tokens.
# ---------------------------------------------------------------------------

FORWARD_SHAPES = [
    MoETestShape("S", 128, 256, 512, 2, num_experts=8),
    MoETestShape("M", 256, 512, 1024, 2, num_experts=8),
    MoETestShape("L", 512, 1024, 2048, 2, num_experts=8),
    MoETestShape("S-drop", 128, 256, 512, 2, num_experts=8, drop_frac=0.3),
    MoETestShape("EPR4", 256, 512, 1024, 2, num_experts=16),
    # DeepSeek-V4-Pro routed-expert shape from the sibling NVIDIA dsv4 case:
    # H=7168, F=3072, K=6, E=384. This BF16 correctness smoke excludes the
    # shared-expert branch and the model's FP4 expert format. It deliberately
    # keeps a smaller global-token count; the performance benchmark owns the
    # exact 2K/8K/32K/128K per-rank workloads.
    MoETestShape("DSV4-smoke", 0, 7168, 3072, 6, num_experts=384, global_tokens=1024),
    # Kimi-K3 architecture (H=3584, F=3072, top-k=16), reduced to a 128-expert
    # "small" variant so it stays a light correctness smoke. The full 896-expert
    # model lives in FORWARD_SHAPES_KIMI (opt-in via MOE_KIMI=1, mirroring the
    # backward test), since it needs >=4 cards to fit comfortably.
    MoETestShape("Kimi-K3-small", 2048, 3584, 3072, 16, num_experts=128),
]


# Full Kimi-K3 (896 experts) correctness shapes, opt-in via MOE_KIMI=1 in
# tests/layer/test_moe_forward.py (mirrors BACKWARD_SHAPES_KIMI). Too heavy for
# the default 2-rank smoke; run at >=4 ranks.
FORWARD_SHAPES_KIMI = [
    MoETestShape("Kimi-K3", 2048, 3584, 3072, 16, num_experts=896),
]


# Backward correctness shapes, selected by env in tests/layer/test_moe_backward.py.
BACKWARD_SHAPES_SMALL = [
    MoETestShape("small", 512, 512, 256, 4, num_experts=128),
    MoETestShape("small", 1024, 512, 256, 4, num_experts=128),
    MoETestShape("small", 2048, 512, 256, 4, num_experts=128),
    MoETestShape("small", 512, 1024, 512, 8, num_experts=128),
]

BACKWARD_SHAPES_KIMI = [
    MoETestShape("Kimi-K3", 2048, 3584, 3072, 16, num_experts=896),
    MoETestShape("Kimi-K3", 4096, 3584, 3072, 16, num_experts=896),
    MoETestShape("Kimi-K3", 8192, 3584, 3072, 16, num_experts=896),
]

BACKWARD_SHAPES_PERF = [
    MoETestShape("Qwen3-30B-A3B",    4096, 2048,  768,  8, num_experts=128),
    MoETestShape("Qwen3-30B-A3B",    8192, 2048,  768,  8, num_experts=128),
    MoETestShape("Qwen3-30B-A3B",   16384, 2048,  768,  8, num_experts=128),
    MoETestShape("DeepSeek-MoE-16B", 4096, 2048, 1408,  6, num_experts=64),
    MoETestShape("Qwen3-235B-A22B",  4096, 4096, 1536,  8, num_experts=128),
    MoETestShape("Qwen3-Next-80B",   4096, 2048,  512, 10, num_experts=512),
    MoETestShape("Qwen3-Omni-30B",   4096, 1024,  384,  6, num_experts=128),
    MoETestShape("Kimi-K3",          4096, 3584, 3072, 16, num_experts=896),
    MoETestShape("Kimi-K3",          8192, 3584, 3072, 16, num_experts=896),
    MoETestShape("Kimi-K3",         16384, 3584, 3072, 16, num_experts=896),
]

# MegaMoEBackwardFunction autograd shapes (fixed num_experts=128).
BACKWARD_FUNCTION_SHAPES = [
    MoETestShape("fn", 512, 512, 256, 4, num_experts=128),
    MoETestShape("fn", 1024, 512, 256, 4, num_experts=128),
    MoETestShape("fn", 512, 1024, 512, 8, num_experts=128),
]
