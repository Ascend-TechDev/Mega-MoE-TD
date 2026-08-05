# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Shared numerical-comparison helpers for Mega-MoE tests.

All bf16 correctness thresholds live here so the forward and backward tests
share one source of truth. The values are deliberately kept identical to the
pre-unification constants:

* layout tensors (expert ids, dispatch order) — exact, ``0/0``
* forward fc1 / SwiGLU activations — ``2e-2 / 2e-2``
* forward full ``[tokens, hidden]`` output — ``4e-2 / 4e-2``
* backward grad tensors — ``2e-2 / 1e-2``

``assert_close`` is the pass/fail verdict (wraps ``torch.testing.assert_close``
on fp32 values). ``diagnose`` and ``cmp_grad`` produce per-tensor failure
detail for the human-readable summary printed by the dist tests.
"""

import torch

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LAYOUT_RTOL = 0
LAYOUT_ATOL = 0

APPROX_RTOL = 2e-2
APPROX_ATOL = 2e-2

OUTPUT_RTOL = 4e-2
OUTPUT_ATOL = 4e-2

GRAD_RTOL = 2e-2
GRAD_ATOL = 1e-2


# ---------------------------------------------------------------------------
# Pass/fail + diagnostics
# ---------------------------------------------------------------------------

def assert_close(actual, expected, *, rtol, atol):
    """Pass/fail verdict. Compares in fp32 to match the historical golden path."""
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=rtol, atol=atol)


def diagnose(actual, expected):
    """Return a one-line diagnostic string (max abs + relative error)."""
    diff = (actual.float() - expected.float()).abs()
    gmax = float(expected.abs().float().max().item()) + 1e-9
    return f"max={diff.max().item():.1e} rel={diff.max().item() / gmax:.1e}"


def cmp_grad(name, tri, gold, rtol=GRAD_RTOL, atol=GRAD_ATOL):
    """Diagnostic grad comparison returning ``(ok, max_abs, rel, n_bad)``.

    Kept as the per-tensor detail helper used by the backward tests' summary
    print. The pass/fail verdict itself should use :func:`assert_close` so the
    threshold lives in one place; this helper mirrors the same
    ``|tri - gold| <= atol + rtol * |gold|`` rule for the printed detail.
    """
    tri = tri.float(); gold = gold.float()
    d = (tri - gold).abs()
    max_d = float(d.max().item())
    gmax = float(gold.abs().max().item())
    n_bad = int((d > atol + rtol * gmax).sum().item())
    ok = n_bad == 0
    return ok, max_d, max_d / (gmax + 1e-9), n_bad
