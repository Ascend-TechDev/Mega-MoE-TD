"""TopK routing generation shared by communication benchmarks and tests.

移植自原仓 MoonEP/tests/generate_topk_routing.py（语义逐字一致）；差异仅为
设备从 cuda 改为 cpu（参考实现纯 torch、CPU 可跑），balanced 基线在 cpu
生成器下给出等价的 round-robin 排列。
"""

import torch


def generate_topk_routing(S, K, E, R, bias_ratio, dev, seed, rank=0):
    """Generate topk_experts and tokens_per_expert with benchmark semantics.

    bias_ratio is the sigma of the underlying lognormal expert-logit
    distribution:
      0   -> round-robin balanced baseline
      0.5 -> mild skew
      1   -> typical dropless-MoE skew
      2   -> heavy skew
      5   -> near-degenerate routing

    `seed` seeds rank-shared state (the expert-logit distribution and the
    round-robin expert permutation); `rank` seeds the per-token draws. This
    is bit-for-bit identical across ranks of the same world on cpu: shared
    generator <- base seed, inner generator <- ep rank, so hot experts line
    up across ranks while per-token draws stay independent.
    """
    g_shared = torch.Generator(device=dev).manual_seed(seed)
    g_local = torch.Generator(device=dev).manual_seed(rank)
    if bias_ratio == 0.0:
        epn = E // R
        toks = torch.arange(S, device=dev)
        ks = torch.arange(K, device=dev)
        target_rank = (toks[:, None] + ks[None, :]) % R
        target_local = ((toks[:, None] // R) + ks[None, :]) % epn
        perm = torch.randperm(epn, device=dev, generator=g_local)
        topk = (target_rank * epn + perm[target_local]).to(torch.int32)
    else:
        logits = torch.exp(torch.normal(
            mean=0.0, std=bias_ratio, size=(E,), device=dev, generator=g_shared
        ))
        probs = logits[None, :].expand(S, E)
        topk = torch.multinomial(
            probs, K, replacement=False, generator=g_local
        ).to(torch.int32)

    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe
