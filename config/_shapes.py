# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Central model / shape configuration for Mega-MoE tests and benchmarks.

Single source of truth consumed by both the correctness tests (tests/layer/*)
and the performance benchmarks (benchmark/layer/*). Re-exported by
``config/__init__`` so callers do ``from config import MoETestShape,
FORWARD_SHAPES, MODEL_PROFILES, select_perf_shapes, ...``.

Two kinds of config live here:

* :class:`MoETestShape` plus the ``FORWARD_*`` / ``BACKWARD_*`` shape lists —
  one dataclass instance per (model, token count). The field order follows the
  backward test's historical tuple ``(name, ntokens, hidden, ffn, topk,
  num_experts)``. ``num_experts`` is explicit (now used by both sides);
  ``experts_per_rank`` remains supported for world-size scaling but is unused by
  the current lists; DSV4-smoke sets ``global_tokens`` so
  :meth:`resolved_tokens_per_rank` divides it across ranks.
* :data:`MODEL_PROFILES` — forward-benchmark profiles (arch + capacity +
  per-token ``bench_configs`` with env-selectable slugs), consumed by
  ``benchmark/layer/bench_full_forward.py`` via ``_activate_model_profile``.
"""

import os
from dataclasses import dataclass
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
    # DeepSeek-V4-Pro routed-experts shape from the sibling NVIDIA dsv4 case:
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


# Backward correctness shapes, selected by env in tests/layer/test_moe_backward.py
# and benchmark/layer/bench_backward.py (see select_perf_shapes).
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

# Kimi-K3 architecture scaled to 16 experts (the minimum: topk=16 requires
# E >= topk). Default backward benchmark shape when RANK_SIZE=2 (see rank_size /
# bench_backward): at 2 cards epr=8, within the backward kernel's launch-grid
# limit (<= physical aicore num); E=128 (epr=64) would be skipped on 2 cards.
BACKWARD_SHAPES_KIMI_SMALL = [
    MoETestShape("Kimi-K3-small", 2048, 3584, 3072, 16, num_experts=16),
    MoETestShape("Kimi-K3-small", 4096, 3584, 3072, 16, num_experts=16),
    MoETestShape("Kimi-K3-small", 8192, 3584, 3072, 16, num_experts=16),
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


# ---------------------------------------------------------------------------
# Forward benchmark profiles (benchmark/layer/bench_full_forward.py).
# arch + receive-capacity + per-token bench configs as (slug, tokens_per_rank).
# Consumed by _activate_model_profile; slugs are env-selectable via
# MOE_FULL_BENCH_CONFIG (DSV4 via MOE_DSV4_BENCH_CONFIG). Tiling is env-driven
# (_layer_tiling_overrides), so tiling_overrides stays {} here.
# ---------------------------------------------------------------------------

MODEL_PROFILES = {
    "QWEN": {
        "hidden": 2048,
        "ffn_dim": 768,
        "topk": 8,
        "num_experts": 128,
        "capacity": 1.25,
        "tiling_overrides": {},
        "bench_configs": [
            ("2K", 2048),
            ("8K", 8192),
            ("16K", 16384),
            ("32K", 32768),
        ],
    },
    # Uses the CASE_SET=dsv4 model dimensions from the sibling NVIDIA benchmark
    # launcher, with token counts interpreted per rank for the Ascend workload.
    # This is the routed-expert BF16 shape only: it excludes the shared-expert
    # branch and does not model the model's FP4 expert format.
    "DSV4": {
        "hidden": 7168,
        "ffn_dim": 3072,
        "topk": 6,
        "num_experts": 384,
        "capacity": 4.0,
        "tiling_overrides": {},
        "bench_configs": [
            ("dsv4_pro_2k", 2048),
            ("dsv4_pro_4k", 4096),
            ("dsv4_pro_8k", 8192),
            ("dsv4_pro_32k", 32768),
            ("dsv4_pro_128k", 131072),
        ],
    },
    "KIMI-K3": {
        "hidden": 3584,
        "ffn_dim": 3072,
        "topk": 16,
        "num_experts": 896,
        "capacity": 1.25,
        "tiling_overrides": {},
        "bench_configs": [
            ("kimi_k3_4k", 4096),
            ("kimi_k3_8k", 8192),
            ("kimi_k3_16k", 16384),
        ],
    },
    # Kimi-K3 architecture scaled to 128 experts so it fits on 2 cards.
    # Selected by test_bench_full_forward_kimi_k3 when RANK_SIZE=2.
    "KIMI-K3-SMALL": {
        "hidden": 3584,
        "ffn_dim": 3072,
        "topk": 16,
        "num_experts": 128,
        "capacity": 1.25,
        "tiling_overrides": {},
        "bench_configs": [
            ("kimi_k3_small_4k", 4096),
            ("kimi_k3_small_8k", 8192),
            ("kimi_k3_small_16k", 16384),
        ],
    },
}


def select_perf_shapes(spec):
    """Select backward perf shapes from the ``MOE_PERF_CONFIGS`` env value.

    ``spec == "1"`` -> all of :data:`BACKWARD_SHAPES_PERF`; otherwise a
    comma-separated list of model labels, matched case-insensitively against
    each shape's ``label`` (e.g. ``"Kimi-K3"`` or
    ``"Qwen3-30B-A3B,DeepSeek-MoE-16B"``). Raises :class:`ValueError` if no
    label matches, listing the available labels.
    """
    if spec == "1":
        return list(BACKWARD_SHAPES_PERF)
    wanted = {part.strip().lower() for part in spec.split(",") if part.strip()}
    selected = [s for s in BACKWARD_SHAPES_PERF if s.label.lower() in wanted]
    if not selected:
        available = sorted({s.label for s in BACKWARD_SHAPES_PERF})
        raise ValueError(
            f"MOE_PERF_CONFIGS={spec!r} matched no perf shape; "
            f"available labels: {available}"
        )
    return selected


def rank_size(default: int = 8) -> int:
    """Read the ``RANK_SIZE`` env (must be 2 or 8) for the perf benchmarks.

    The Kimi-K3 benchmarks use this to pick world_size and, by extension, the
    model variant: ``2`` -> Kimi-K3-small (forward 128 experts / backward 16
    experts, fits on 2 cards); ``8`` -> full Kimi-K3 (896 experts).
    """
    rs = int(os.environ.get("RANK_SIZE", str(default)))
    if rs not in (2, 8):
        raise ValueError(f"RANK_SIZE must be 2 or 8, got {rs}")
    return rs


def shape_slug(shape):
    """Stable slug like ``kimi_k3_small_4k`` for a MoETestShape.

    ``label`` lowercased with ``-`` -> ``_``, plus a ``<tokens/1024>k`` suffix.
    Matches the forward benchmark's bench_config slug style (e.g. the
    KIMI-K3-SMALL @4096 slug ``kimi_k3_small_4k``), so the two benchmarks share
    one config-name vocabulary.
    """
    name = shape.label.lower().replace("-", "_")
    return f"{name}_{shape.tokens // 1024}k"
