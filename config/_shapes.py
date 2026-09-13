# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Declarative MoE cases shared by functional tests and benchmarks.

``tokens`` is always the number of tokens owned by one rank.  The registry is
deliberately static: selecting a direction or a tag is done by Python code
(``select_cases``), never by mutating process-global shapes from environment
variables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable


_VALID_DIRECTIONS = frozenset({"forward", "backward"})
Capacity = float | Callable[[int], float]


@dataclass(frozen=True)
class CaseSpec:
    """One concrete test or benchmark execution."""

    case_id: str
    direction: str
    model: str
    tokens: int
    world_size: int
    hidden: int
    ffn: int
    topk: int
    num_experts: int
    tags: frozenset[str]
    capacity_factor: float = 1.0
    drop_frac: float = 0.0

    @property
    def tokens_per_rank(self) -> int:
        """Explicit alias used by result serialization and human-readable labels."""
        return self.tokens

    @property
    def experts_per_rank(self) -> int:
        if self.num_experts % self.world_size:
            raise ValueError(
                f"{self.case_id}: num_experts={self.num_experts} is not divisible "
                f"by world_size={self.world_size}"
            )
        return self.num_experts // self.world_size

    def validate(self) -> "CaseSpec":
        normalized_id = self.case_id.lower()
        model_id = normalized_id.replace("-", "_")
        if _slug(self.model) not in model_id:
            raise ValueError(f"{self.case_id}: case_id must contain the model slug")
        direction_slug = "fwd" if self.direction == "forward" else "bwd"
        if f"-{direction_slug}-" not in normalized_id:
            raise ValueError(f"{self.case_id}: case_id must contain {direction_slug!r}")
        if f"-w{self.world_size}-" not in normalized_id:
            raise ValueError(f"{self.case_id}: case_id must contain the world size")
        if f"-{_token_slug(self.tokens)}" not in normalized_id:
            raise ValueError(f"{self.case_id}: case_id must contain the token slug")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"{self.case_id}: invalid direction {self.direction!r}")
        if self.tokens <= 0:
            raise ValueError(f"{self.case_id}: tokens must be positive")
        if self.world_size <= 0:
            raise ValueError(f"{self.case_id}: world_size must be positive")
        if self.hidden <= 0 or self.ffn <= 0 or self.topk <= 0:
            raise ValueError(f"{self.case_id}: dimensions and topk must be positive")
        if self.num_experts < self.topk:
            raise ValueError(f"{self.case_id}: num_experts must be >= topk")
        _ = self.experts_per_rank
        if not math.isfinite(self.capacity_factor) or self.capacity_factor < 1.0:
            raise ValueError(f"{self.case_id}: capacity_factor must be >= 1")
        if not math.isfinite(self.drop_frac) or not 0.0 <= self.drop_frac < 1.0:
            raise ValueError(f"{self.case_id}: drop_frac must be in [0, 1)")
        return self

    def as_dict(self) -> dict:
        result = asdict(self)
        result["tags"] = sorted(self.tags)
        result["tokens_per_rank"] = self.tokens
        return result


@dataclass(frozen=True)
class CaseGroup:
    """A compact Cartesian-product description of related concrete cases."""

    prefix: str
    direction: str
    model: str
    tokens: tuple[int, ...]
    worlds: tuple[int, ...]
    hidden: int
    ffn: int
    topk: int
    num_experts: int
    tags: frozenset[str]
    capacity_factor: Capacity = 1.0
    drop_frac: float = 0.0


def _token_slug(tokens: int) -> str:
    if tokens % 1024 == 0:
        return f"t{tokens // 1024}k"
    return f"t{tokens}"


def _slug(label: str) -> str:
    return label.lower().replace("-", "_").replace(" ", "_")


def _expand(group: CaseGroup) -> tuple[CaseSpec, ...]:
    """Expand one group into its world/token Cartesian product."""
    if group.direction not in _VALID_DIRECTIONS:
        raise ValueError(f"{group.prefix}: invalid direction {group.direction!r}")
    cases = []
    for world_size in group.worlds:
        for token_count in group.tokens:
            capacity = (
                group.capacity_factor(world_size)
                if callable(group.capacity_factor)
                else group.capacity_factor
            )
            case_id = f"{group.prefix}-w{world_size}-{_token_slug(token_count)}"
            cases.append(
                CaseSpec(
                    case_id=case_id,
                    direction=group.direction,
                    model=group.model,
                    tokens=token_count,
                    world_size=world_size,
                    hidden=group.hidden,
                    ffn=group.ffn,
                    topk=group.topk,
                    num_experts=group.num_experts,
                    tags=group.tags,
                    capacity_factor=capacity,
                    drop_frac=group.drop_frac,
                ).validate()
            )
    return tuple(cases)


def _functional_capacity(world_size: int) -> float:
    """Keep smoke tests' historical world-size-specific receive capacity."""
    return float(world_size)


_CASE_GROUPS = (
    # Functional forward: five representative shapes only.
    CaseGroup(
        prefix="functional-fwd-s", direction="forward", model="S",
        tokens=(128,), worlds=(2, 4, 8), hidden=256, ffn=512, topk=2,
        num_experts=8, tags=frozenset({"functional", "forward", "smoke"}),
        capacity_factor=_functional_capacity,
    ),
    CaseGroup(
        prefix="functional-fwd-m", direction="forward", model="M",
        tokens=(256,), worlds=(2, 4, 8), hidden=512, ffn=1024, topk=2,
        num_experts=8, tags=frozenset({"functional", "forward", "smoke"}),
        capacity_factor=_functional_capacity,
    ),
    CaseGroup(
        prefix="functional-fwd-l", direction="forward", model="L",
        tokens=(512,), worlds=(2, 4, 8), hidden=1024, ffn=2048, topk=2,
        num_experts=8, tags=frozenset({"functional", "forward", "smoke"}),
        capacity_factor=_functional_capacity,
    ),
    CaseGroup(
        prefix="functional-fwd-s-drop", direction="forward", model="S-drop",
        tokens=(128,), worlds=(2, 4, 8), hidden=256, ffn=512, topk=2,
        num_experts=8, tags=frozenset({"functional", "forward", "smoke"}),
        capacity_factor=_functional_capacity, drop_frac=0.3,
    ),
    CaseGroup(
        prefix="functional-fwd-epr4", direction="forward", model="EPR4",
        tokens=(256,), worlds=(2, 4, 8), hidden=512, ffn=1024, topk=2,
        num_experts=16, tags=frozenset({"functional", "forward", "smoke"}),
        capacity_factor=_functional_capacity,
    ),

    # Functional backward: two small representative dimensions at all
    # supported process counts.  Kimi's large functional cases are omitted;
    # they duplicate coverage while making the default suite prohibitively
    # expensive.
    CaseGroup(
        prefix="functional-bwd-small-h512-f256-k4", direction="backward", model="small",
        tokens=(512,), worlds=(2, 4, 8), hidden=512, ffn=256, topk=4,
        num_experts=128, tags=frozenset({"functional", "backward", "smoke"}),
    ),
    CaseGroup(
        prefix="functional-bwd-small-h1024-f512-k8", direction="backward", model="small",
        tokens=(512,), worlds=(2, 4, 8), hidden=1024, ffn=512, topk=8,
        num_experts=128, tags=frozenset({"functional", "backward", "smoke"}),
    ),

    # Forward performance: all models use the same per-rank token sweep.
    CaseGroup(
        prefix="performance-fwd-qwen", direction="forward", model="QWEN",
        tokens=(4096, 8192, 16384), worlds=(2, 4, 8), hidden=2048, ffn=768,
        topk=8, num_experts=128, tags=frozenset({"performance", "forward", "slow"}),
        capacity_factor=1.25,
    ),
    CaseGroup(
        prefix="performance-fwd-dsv4", direction="forward", model="DSV4",
        tokens=(4096, 8192, 16384), worlds=(2, 4, 8), hidden=7168, ffn=3072,
        topk=6, num_experts=384,
        tags=frozenset({"performance", "forward", "dsv4", "slow"}),
        capacity_factor=4.0,
    ),
    CaseGroup(
        prefix="performance-fwd-kimi-k3", direction="forward", model="KIMI-K3",
        tokens=(4096, 8192, 16384), worlds=(4, 8), hidden=3584, ffn=3072,
        topk=16, num_experts=896,
        tags=frozenset({"performance", "forward", "kimi", "slow"}),
        capacity_factor=1.25,
    ),
    # Small expert-parallel projection used before scaling Kimi to multiple
    # machines.  Keep the model dimensions and T4K/T16K loads while reducing the
    # routed expert topology to four local experts on each of eight ranks.
    CaseGroup(
        prefix="performance-fwd-kimi-k3-trimmed",
        direction="forward",
        model="KIMI-K3-TRIMMED",
        tokens=(4096, 16384),
        worlds=(8,),
        hidden=3584,
        ffn=3072,
        topk=8,
        num_experts=32,
        tags=frozenset({"performance", "forward", "kimi", "trimmed", "slow"}),
        capacity_factor=1.25,
    ),

    # Matched unbalanced Kimi cases: only total expert count differs. Keep the
    # old top-k=8 trimmed case above for existing performance regressions.
    CaseGroup(
        prefix="performance-fwd-kimi-k3-trimmed-skewed",
        direction="forward", model="KIMI-K3-TRIMMED",
        tokens=(4096, 16384), worlds=(8,), hidden=3584, ffn=3072,
        topk=16, num_experts=32,
        tags=frozenset({"performance", "forward", "kimi", "trimmed", "moonep-skewed", "slow"}),
        capacity_factor=1.6875,
    ),
    CaseGroup(
        prefix="performance-fwd-kimi-k3-skewed",
        direction="forward", model="KIMI-K3",
        tokens=(4096, 16384), worlds=(8,), hidden=3584, ffn=3072,
        topk=16, num_experts=896,
        tags=frozenset({"performance", "forward", "kimi", "moonep-skewed", "slow"}),
        capacity_factor=1.6875,
    ),

    # Backward performance has only validated profiles.  There is no DSV4
    # backward profile yet, so it is intentionally not synthesized here.
    CaseGroup(
        prefix="performance-bwd-qwen3-30b-a3b", direction="backward",
        model="Qwen3-30B-A3B", tokens=(4096, 8192, 16384), worlds=(2, 4, 8),
        hidden=2048, ffn=768, topk=8, num_experts=128,
        tags=frozenset({"performance", "backward", "slow"}),
    ),
    CaseGroup(
        prefix="performance-bwd-kimi-k3", direction="backward", model="Kimi-K3",
        tokens=(2048, 4096, 8192, 16384), worlds=(4, 8), hidden=3584, ffn=3072,
        topk=16, num_experts=896,
        tags=frozenset({"performance", "backward", "kimi", "slow"}),
    ),
)


_CASES: list[CaseSpec] = []
for group in _CASE_GROUPS:
    _CASES.extend(_expand(group))

CASE_REGISTRY = tuple(_CASES)
_CASE_BY_ID = {case.case_id: case for case in CASE_REGISTRY}
if len(_CASE_BY_ID) != len(CASE_REGISTRY):
    raise ValueError("case_id values must be unique")


def select_cases(*, direction: str, tags: Iterable[str] = ()) -> tuple[CaseSpec, ...]:
    """Select cases by fixed code-level direction and tag predicates."""
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_VALID_DIRECTIONS)}")
    wanted = frozenset(tags)
    return tuple(
        case
        for case in CASE_REGISTRY
        if case.direction == direction and wanted.issubset(case.tags)
    )


def resolve_case(case_id: str) -> CaseSpec:
    try:
        return _CASE_BY_ID[case_id]
    except KeyError as exc:
        raise KeyError(f"unknown MoE case_id {case_id!r}") from exc


def case_dict(case: CaseSpec) -> dict:
    return case.as_dict()


__all__ = [
    "CaseGroup",
    "CaseSpec",
    "CASE_REGISTRY",
    "select_cases",
    "resolve_case",
    "case_dict",
]
