# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Public case-registry API for tests and benchmarks."""

from config._shapes import (
    CASE_REGISTRY,
    CaseGroup,
    CaseSpec,
    case_dict,
    resolve_case,
    select_cases,
)

__all__ = [
    "CaseGroup",
    "CaseSpec",
    "CASE_REGISTRY",
    "select_cases",
    "resolve_case",
    "case_dict",
]
