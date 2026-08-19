"""共享测试设施（组织格式对齐原仓 MoonEP/tests/kernel_test_utils.py）。

与原仓的差异（契约 §0 的 torch 参考实现适配）：

- 运行时为单进程 SimTransport（多线程模拟 R 个 rank），无 torchrun/NCCL：
  ``dist_env``/``init_case``/``gather_tensor``/``assert_*_all_ranks`` 合并为
  ``build_world``（返回 sim/arenas/bufs 三元组，R 个 rank 数据在进程内
  直接可达）+ ``run_ranks``（线程驱动 + 异常汇总）；
- 断言 helpers 退化为纯 assert（线程内抛出即失败，run_ranks 汇总重抛）；
- 朴素 oracle（散布/dup 表/src_info）从 test_sim_cases.py 的 cfg 版收编，
  统一挂在 KernelCase 上，供各 per-module 测试文件复用。
"""

import os
import sys
import threading
from dataclasses import dataclass

import pytest
import torch

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_EXPERT_PARALLEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_THIS_DIR, _EXPERT_PARALLEL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_topk_routing import generate_topk_routing

DEFAULT_TOKEN_PADDING = 128

# 活跃世界登记（conftest 的 autouse cleanup fixture 消费）
_ACTIVE_WORLDS = []


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class KernelCase:
    """一个用例的维度/路由配置（字段与原仓 kernel_test_utils.KernelCase
    逐字一致；新增 ``R`` 为 SimTransport 世界大小，替代 torchrun 的
    world size）。"""

    name: str
    S: int
    K: int
    epn: int
    H: int
    num_sms: int
    B: "int | None" = None
    token_padding: int = DEFAULT_TOKEN_PADDING
    routing: str = "balanced"
    bias_ratio: float = 0.0
    seed: int = 42
    min_R: int = 1
    max_R: "int | None" = None
    R: int = 4
    zero_copy: bool = False

    def E(self, R):
        return R * self.epn

    @property
    def N(self):
        return self.S * self.K

    def NvS(self, R):
        epn = self.E(R) // R
        return self.S * self.K + (self.token_padding - 1) * 2 * epn


def case_params(cases):
    return [pytest.param(case, id=case.name) for case in cases]


def skip_if_unsupported_world_size(case, R=None):
    R = case.R if R is None else R
    if R < case.min_R:
        pytest.skip(f"case {case.name} requires R >= {case.min_R}, got R={R}")
    if case.max_R is not None and R > case.max_R:
        pytest.skip(f"case {case.name} requires R <= {case.max_R}, got R={R}")


def build_world(case):
    """按 case 构造 SimTransport 世界并登记（替代原仓 init_case）。

    返回 (sim, arenas, bufs)：R 个 arena 视角与 R 个 Buffer；ctx 经
    ``bufs[r]._require_ctx()`` 取（原仓 init_case 的等价物）。
    """
    from moonep import Buffer
    from sim_transport import SimTransport

    skip_if_unsupported_world_size(case)
    sim = SimTransport(case.R)
    arenas = [sim.arena_for(r) for r in range(case.R)]
    bufs = [
        Buffer(case.S, case.H, case.K, case.E(case.R), case.R,
               B=case.B, num_sms=case.num_sms,
               token_padding=case.token_padding, arena=arenas[r])
        for r in range(case.R)
    ]
    _ACTIVE_WORLDS.append((sim, bufs))
    return sim, arenas, bufs


def destroy_active_worlds():
    while _ACTIVE_WORLDS:
        _sim, bufs = _ACTIVE_WORLDS.pop()
        for buffer in bufs:
            if not buffer.destroyed:
                buffer.destroy()


def run_ranks(sim, per_rank_fn, tag, timeout=660):
    """每 rank 一个线程执行 per_rank_fn(r)；屏障 armed，异常汇总抛出。

    （SimTransport 集合时序要求的线程驱动，替代原仓 torchrun 多进程；
    异常带 traceback 汇总，便于定位线程内的裸 assert。）
    """
    import traceback

    errors = {}

    def _wrap(r):
        try:
            per_rank_fn(r)
        except BaseException as exc:  # noqa: BLE001 - 测试驱动汇总重抛
            errors[r] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__))
            sim._abort()

    sim._arm()
    try:
        ths = [threading.Thread(target=_wrap, args=(r,), name=f"rank{r}")
               for r in range(sim.num_ranks)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout)
    finally:
        sim._disarm()
    for r, th in enumerate(ths):
        assert not th.is_alive(), f"[{tag}] rank{r} 线程超时（疑似屏障死锁）"
    if errors:
        msgs = "; ".join(f"rank{r}: {tb.strip().splitlines()[-1]}"
                         for r, tb in errors.items())
        detail = "\n".join(f"---- rank{r} ----\n{tb}" for r, tb in errors.items())
        raise AssertionError(f"[{tag}] 线程执行失败：{msgs}\n{detail}")


def make_topk(case, rank, R=None):
    """生成该 rank 的 (topk [S,K] int32, tpe [E] int32)。

    balanced/biased 走 generate_topk_routing（cpu）；确定性路由模式与
    原仓 kernel_test_utils.make_topk 逐字一致。
    """
    R = case.R if R is None else R
    skip_if_unsupported_world_size(case, R)
    E = case.E(R)

    if case.routing in {"balanced", "biased"}:
        if case.K > E:
            pytest.skip(f"case {case.name} requires K <= E, got K={case.K}, E={E}")
        bias = case.bias_ratio if case.routing == "biased" else 0.0
        return generate_topk_routing(
            case.S, case.K, E, R, bias, "cpu", case.seed, rank=rank
        )

    s = torch.arange(case.S)[:, None]
    k = torch.arange(case.K)[None, :]
    epn = case.epn

    if case.routing == "all_local":
        topk = rank * epn + ((s + k) % epn)
    elif case.routing == "all_remote":
        remote_rank = (rank + 1) % R
        topk = remote_rank * epn + ((s + k) % epn)
    elif case.routing == "single_expert":
        topk = torch.zeros((case.S, case.K), dtype=torch.long)
    elif case.routing == "duplicate_topk":
        expert = ((rank + 1) % R) * epn
        topk = torch.full((case.S, case.K), expert, dtype=torch.long)
    else:
        raise ValueError(f"unknown routing pattern: {case.routing}")

    topk = topk.to(torch.int32).contiguous()
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    return topk, tpe


def make_inputs(case, rank, seed=0):
    """hidden [S,H] bf16 与 weights [S,K] fp32（per-rank 种子独立）。"""
    gen = torch.Generator().manual_seed(seed + rank)
    hidden = torch.randn(case.S, case.H, dtype=torch.bfloat16, generator=gen)
    weights = torch.rand(case.S, case.K, dtype=torch.float32, generator=gen)
    return hidden, weights


def traceable_hidden(rank, S, H):
    """int16 位壳打点的可溯源 hidden：[:,0]=s（低 16 位），[:,1]=rank。"""
    hidden = torch.zeros(S, H, dtype=torch.bfloat16)
    hidden_i16 = hidden.view(torch.int16)
    s_idx = torch.arange(S, dtype=torch.int32)
    hidden_i16[:, 0] = s_idx.to(torch.int16)
    if H > 1:
        hidden_i16[:, 1] = torch.full((S,), rank, dtype=torch.int16)
    return hidden


def traceable_weights(rank, S, K):
    """int32 位壳打点的可溯源 weights：值 = rank*S*K + s*K + k。"""
    weights_i32 = (
        torch.arange(S * K, dtype=torch.int32).reshape(S, K).add_(rank * S * K)
    )
    return weights_i32.view(torch.float32)


def decode_dst(v, nvs):
    """dst 编码解码：v>=0 → raw=v；v<0 → raw=-v-1。返回 (dest_rank, loff)。"""
    raw = v if v >= 0 else -v - 1
    return raw // nvs, raw % nvs


# ---------------------------------------------------------------------------
# planning 不变量（对齐原仓 kernel_test_utils.planning_invariant_errors；
# meta 布局公式与 api.py 的逐字实现一致）
# ---------------------------------------------------------------------------
def planning_invariant_errors(case, ctx, dst, cu_seqlens, experts_to_copy):
    errors = []
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    NvS = int(ctx["NvS"])
    N = case.S * case.K

    planning_out_elems = (
        3 * E * R
        + R * (E + B)
        + 2 * R * (E + B)
        + R * B
        + 2 * R
    )
    n4 = _align_up(N, 4)
    expected_topk0_off = _align_up(int(ctx["PLAN_OFF"]) + planning_out_elems, 4)
    expected_order_off = expected_topk0_off + n4
    expected_order0_off = expected_order_off + n4
    expected_barrier_off = expected_order0_off + n4
    expected_src_info_off = expected_barrier_off + 3
    layout_checks = (
        ("TOPK0_OFF", expected_topk0_off),
        ("ORDER_OFF", expected_order_off),
        ("ORDER0_OFF", expected_order0_off),
        ("BARRIER_OFF", expected_barrier_off),
        ("SRC_INFO_OFF", expected_src_info_off),
    )
    for key, expected in layout_checks:
        actual = int(ctx[key])
        if actual != expected:
            errors.append(f"{key}={actual}, expected {expected}")
    if int(ctx["meta_chunk_padded"]) < expected_src_info_off + NvS:
        errors.append(
            f"meta_chunk_padded={int(ctx['meta_chunk_padded'])} is smaller than "
            f"src_info end={expected_src_info_off + NvS}"
        )

    if dst.dtype != torch.int32 or tuple(dst.shape) != (N,):
        errors.append(f"dst must be int32 [{N}], got {dst.dtype} {tuple(dst.shape)}")
    else:
        dst_cpu = dst.cpu() if dst.device.type != "cpu" else dst
        raw_dst = torch.where(dst_cpu < 0, -dst_cpu - 1, dst_cpu)
        dest_rank = torch.div(raw_dst, NvS, rounding_mode="floor")
        local_off = raw_dst % NvS
        if not torch.all((dest_rank >= 0) & (dest_rank < R)):
            errors.append("dst contains an out-of-range destination rank")
        if not torch.all((local_off >= 0) & (local_off < NvS)):
            errors.append("dst contains an out-of-range local offset")

    cu_cpu = cu_seqlens.cpu() if cu_seqlens.device.type != "cpu" else cu_seqlens
    if cu_seqlens.dtype != torch.int32 or tuple(cu_seqlens.shape) != (E + B,):
        errors.append(
            f"cu_seqlens must be int32 [{E + B}], "
            f"got {cu_seqlens.dtype} {tuple(cu_seqlens.shape)}"
        )
    else:
        prev = 0
        for gid, cur_t in enumerate(cu_cpu.tolist()):
            cur = int(cur_t)
            seg_len = cur - prev
            if seg_len < 0:
                errors.append(f"cu_seqlens decreases at group {gid}")
                break
            if seg_len and seg_len % case.token_padding != 0:
                errors.append(
                    f"group {gid} segment length {seg_len} is not divisible "
                    f"by token_padding={case.token_padding}"
                )
                break
            prev = cur
        if int(cu_cpu[-1].item()) > NvS:
            errors.append(f"cu_seqlens total {int(cu_cpu[-1].item())} exceeds NvS={NvS}")

    copy_cpu = experts_to_copy.cpu() if experts_to_copy.device.type != "cpu" else experts_to_copy
    if experts_to_copy.dtype != torch.int32 or tuple(experts_to_copy.shape) != (R, B):
        errors.append(
            f"experts_to_copy must be int32 [{R}, {B}], "
            f"got {experts_to_copy.dtype} {tuple(experts_to_copy.shape)}"
        )
    elif not torch.all((copy_cpu == -1) | ((copy_cpu >= 0) & (copy_cpu < E))):
        errors.append("experts_to_copy contains invalid expert ids")

    return errors


# ---------------------------------------------------------------------------
# 朴素 oracle（dispatch/combine 的独立重算；收编自 test_sim_cases.py 的
# cfg 版，供 per-module 测试复用）
# ---------------------------------------------------------------------------
def naive_scatter_hidden(plans, src, case, R=None):
    """朴素 dispatch payload 散布 + epilogue 扇出 + padding 清零。"""
    R = case.R if R is None else R
    nvs = case.NvS(R)
    N = case.N
    out = [torch.zeros(nvs, case.H, dtype=torch.bfloat16) for _ in range(R)]
    for sr in range(R):
        dst = plans[sr].dst
        for offv in range(N):
            v = int(dst[offv])
            if v < 0:
                continue
            dr, loff = decode_dst(v, nvs)
            out[dr][loff] = src[sr][offv // case.K]
    for r in range(R):
        n_groups = int(plans[r].dup_counts[0])
        for gidx in range(n_groups):
            primary_loff = int(plans[r].dup_groups[gidx, 0])
            dup_start = int(plans[r].dup_groups[gidx, 1])
            dup_count = int(plans[r].dup_groups[gidx, 2])
            for j in range(dup_count):
                out[r][int(plans[r].dup_loffs[dup_start + j])] = out[r][primary_loff]
    for r in range(R):
        zfr = plans[r].zero_fill_ranges
        for gidx in range(case.E(R) + int(plans[r].experts_to_copy.shape[1])):
            start, count = int(zfr[gidx, 0]), int(zfr[gidx, 1])
            if count > 0:
                out[r][start:start + count].zero_()
    return out


def naive_scatter_weights(plans, w, case, R=None):
    """朴素 dispatch 权重散布（WEIGHTS 区全槽，含负 dst 槽；从不去重）。"""
    R = case.R if R is None else R
    nvs = case.NvS(R)
    N = case.N
    out = [torch.zeros(nvs, dtype=torch.float32) for _ in range(R)]
    for sr in range(R):
        dst = plans[sr].dst
        w_flat = w[sr].reshape(-1)
        for offv in range(N):
            dr, loff = decode_dst(int(dst[offv]), nvs)
            out[dr][loff] = w_flat[offv]
    for r in range(R):
        zfr = plans[r].zero_fill_ranges
        for gidx in range(case.E(R) + int(plans[r].experts_to_copy.shape[1])):
            start, count = int(zfr[gidx, 0]), int(zfr[gidx, 1])
            if count > 0:
                out[r][start:start + count].zero_()
    return out


def oracle_dup_tables(plans, case, tag, R=None):
    """dup 表 oracle：从全组 dst 独立重算（组内 dup 按 kidx 升序）。"""
    R = case.R if R is None else R
    nvs = case.NvS(R)
    N = case.N
    dst_all = torch.stack([p.dst for p in plans])
    for r in range(R):
        prim_of_key = {}
        dup_slots = []
        for sr in range(R):
            for offv in range(N):
                v = int(dst_all[sr, offv])
                dr, loff = decode_dst(v, nvs)
                if dr != r:
                    continue
                key = (sr, offv // case.K)
                if v >= 0:
                    assert key not in prim_of_key, \
                        f"[{tag}] rank{r} 组 {key} 出现多个 primary 槽"
                    prim_of_key[key] = loff
                else:
                    dup_slots.append((sr, offv // case.K, offv % case.K, loff))
        expect = {}
        for sr, tok, kidx, loff in dup_slots:
            key = (sr, tok)
            assert key in prim_of_key, \
                f"[{tag}] rank{r} dup 槽 {loff} 找不到同组 primary"
            expect.setdefault(prim_of_key[key], []).append((kidx, loff))
        for p in expect:
            expect[p] = [loff for _, loff in sorted(expect[p])]

        n_groups = int(plans[r].dup_counts[0])
        n_loffs = int(plans[r].dup_counts[1])
        assert n_groups == len(expect), \
            f"[{tag}] rank{r} dup 组数 {n_groups} != 朴素重算 {len(expect)}"
        assert n_loffs == len(dup_slots), \
            f"[{tag}] rank{r} dup 槽位数 {n_loffs} != 朴素重算 {len(dup_slots)}"
        actual = {}
        for gidx in range(n_groups):
            p = int(plans[r].dup_groups[gidx, 0])
            st = int(plans[r].dup_groups[gidx, 1])
            ct = int(plans[r].dup_groups[gidx, 2])
            actual[p] = [int(plans[r].dup_loffs[st + j]) for j in range(ct)]
        assert actual == expect, \
            f"[{tag}] rank{r} dup 表内容与朴素重算不一致：{actual} vs {expect}"


def oracle_src_info(plans, case, tag, R=None):
    """src_info oracle：本 rank 槽位出处 == src_rank*NvS+offv（-1 空槽）。"""
    R = case.R if R is None else R
    nvs = case.NvS(R)
    N = case.N
    dst_all = torch.stack([p.dst for p in plans])
    for r in range(R):
        expect = torch.full((nvs,), -1, dtype=torch.int32)
        for sr in range(R):
            for offv in range(N):
                dr, loff = decode_dst(int(dst_all[sr, offv]), nvs)
                if dr == r:
                    expect[loff] = sr * nvs + offv
        assert torch.equal(plans[r].src_info, expect), \
            f"[{tag}] rank{r} src_info 与朴素重算不一致"


def assert_close(name, actual, expected, atol=0.05):
    diff = (actual.float() - expected.float()).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    assert max_err < atol, f"{name} max_err={max_err} >= atol={atol}"
