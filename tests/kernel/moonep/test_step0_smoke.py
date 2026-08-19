# coding=utf-8
"""Step 0 smoke（M0b）：planning/prefetch Triton 移植依赖的四类原语在 910B
后端的可用性验证。任何一个失败都影响 M1 路线（见方案 L0 节的回退策略）。

① tl.sort / tl.cumsum / tl.argmax(tie_break_left) 编译运行 + 数值正确
② 动态标量 while 循环（B.1/B.2 贪心的控制流形态）编译运行
③ (1,1,1) 网格 kernel 内 putmem + barrier_all_vec（routing.py 同款）2 rank
④ 多张对称张量按序分配后 putmem 跨张量寻址正确（MoonepWorkspace 前提）
"""

import pytest
import torch
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cann.extension import sub_vec_id

_NPU_AVAILABLE = False
try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# ① 原语数值正确性（单 NPU）
#
# 注意 tl.sort 的 910B 后端约束（smoke 实测）：sort 与 kernel 内其它算术
# 共存时，非排序通路的奇数 lane 会被破坏（疑似 sort 降级的 layout/register
# 污染）——纯 load+sort+store 的 kernel 正确。因此 sort 必须独占一个
# kernel：造键（复合键 e*N+i）与排序分开两次 launch。
# ---------------------------------------------------------------------------
@triton.jit
def _smoke_make_key(x_ptr, key_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(x_ptr + offs)
    tl.store(key_ptr + offs, v.to(tl.int64) * N + offs)  # 复合键：构造性稳定


@triton.jit
def _smoke_sort_only(key_ptr, skey_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    key = tl.load(key_ptr + offs)
    tl.store(skey_ptr + offs, tl.sort(key))


@triton.jit
def _smoke_reduce(x_ptr, cum_ptr, am_left_ptr, am_right_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(x_ptr + offs)
    tl.store(cum_ptr + offs, tl.cumsum(v, axis=0))
    tl.store(am_left_ptr, tl.argmax(v, axis=0, tie_break_left=True))
    # tie_break_left=False 在 910B 后端不生效（实测行为同 True）——B.3 的
    # 平局取大须用手写 max-index-of-max
    m = tl.max(v, axis=0)
    tl.store(am_right_ptr, tl.max(tl.where(v == m, offs, -1), axis=0))


@triton.jit
def _smoke_while(x_ptr, out_ptr, N: tl.constexpr):
    """贪心循环：每轮取 argmax（平局取小）累加并清零——B.1/B.2 的控制流形态。

    910B 约束：while 内不支持 break——用布尔守卫做"空转轮"，每轮用 mask
    屏蔽掉 m<=0 后的更新（B.1/B.2 的贪心循环同款写法）。
    """
    offs = tl.arange(0, N)
    v = tl.load(x_ptr + offs).to(tl.int64)
    acc = tl.zeros((), dtype=tl.int64)
    remaining = N
    active = True
    while active:
        m = tl.max(v, axis=0)
        ok = m > 0
        idx = tl.argmax(v, axis=0, tie_break_left=True)
        v = tl.where(ok & (offs == idx), 0, v)
        acc += tl.where(ok, m * remaining, 0)
        remaining = remaining - 1
        active = ok & (remaining > 0)
    tl.store(out_ptr, acc)


@pytest.mark.skipif(not _NPU_AVAILABLE, reason="需要 NPU")
@pytest.mark.parametrize("N", [8, 128])
def test_tensor_primitives(N):
    dev = "npu:0"
    torch.manual_seed(0)
    # 含重复值（打平局）与打平的 argmax 用例
    x = torch.randint(0, 4, (N,), dtype=torch.int32, device=dev)
    x[0] = x[1] = x[2] = 7  # 强制三重平局在头部

    order = torch.empty(N, dtype=torch.int32, device=dev)
    sorted_v = torch.empty(N, dtype=torch.int32, device=dev)
    key = torch.empty(N, dtype=torch.int64, device=dev)
    skey = torch.empty(N, dtype=torch.int64, device=dev)
    _smoke_make_key[(1,)](x, key, N=N)
    _smoke_sort_only[(1,)](key, skey, N=N)
    ref = torch.sort(x.to(torch.int64), stable=True)
    assert (skey // N).cpu().tolist() == ref.values.tolist(), "sort 值错误"
    assert (skey % N).cpu().tolist() == ref.indices.tolist(), \
        "复合键稳定性被破坏（order 与 stable argsort 不一致）"

    cum = torch.empty(N, dtype=torch.int64, device=dev)
    am_l = torch.empty(1, dtype=torch.int32, device=dev)
    am_r = torch.empty(1, dtype=torch.int32, device=dev)
    _smoke_reduce[(1,)](x.to(torch.int64), cum, am_l, am_r, N=N)
    assert cum.cpu().tolist() == torch.cumsum(x.to(torch.int64), 0).tolist()
    assert am_l.item() == 0, f"平局取小失败: {am_l.item()} != 0"
    assert am_r.item() == 2, f"平局取大失败: {am_r.item()} != 2（三重平局应取 2）"

    # 贪心宿主重算
    v = x.to(torch.int64).tolist()
    acc, remaining = 0, N
    while remaining > 0:
        m = max(v)
        if m <= 0:
            break
        i = v.index(m)
        v[i] = 0
        acc += m * remaining
        remaining -= 1
    out = torch.empty(1, dtype=torch.int64, device=dev)
    _smoke_while[(1,)](x.to(torch.int64), out, N=N)
    assert out.item() == acc, f"动态 while 贪心结果错误: {out.item()} != {acc}"


# ---------------------------------------------------------------------------
# ③④ 对称堆 putmem / barrier（2 rank dist）
# ---------------------------------------------------------------------------
@triton.jit
def _smoke_roundtrip(
    sym_ptr, out_ptr,
    LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr, ELEMS: tl.constexpr,
):
    """各 rank 写自己行 → putmem 给所有 peer → barrier → 回读右邻行。"""
    offs = tl.arange(0, ELEMS)
    row = sym_ptr + LOCAL_RANK * ELEMS
    tl.store(row + offs, offs + LOCAL_RANK * 1000)
    if sub_vec_id() == 0:
        for peer in range(WORLD_SIZE):
            if peer != LOCAL_RANK:
                # routing.py 惯例：dst 与 src 为同一本地指针——语义是"把我的
                # 行发布到 peer 堆上相同偏移"；长度参数是字节数
                libshmem_device.putmem(row, row, ELEMS * 4, peer)
    libshmem_device.barrier_all_vec()
    peer = (LOCAL_RANK + 1) % WORLD_SIZE
    tl.store(out_ptr + offs, tl.load(sym_ptr + peer * ELEMS + offs))


@triton.jit
def _smoke_multi_tensor(
    t2_ptr, t3_ptr, out2_ptr, out3_ptr,
    LOCAL_RANK: tl.constexpr, WORLD_SIZE: tl.constexpr,
    N2: tl.constexpr, N3: tl.constexpr,
):
    """非首个对称张量的跨 rank putmem 寻址（MoonepWorkspace 分配序前提）。"""
    offs2 = tl.arange(0, N2)
    offs3 = tl.arange(0, N3)
    tl.store(t2_ptr + LOCAL_RANK * N2 + offs2, offs2 + LOCAL_RANK * 2000)
    tl.store(t3_ptr + LOCAL_RANK * N3 + offs3, offs3 + LOCAL_RANK * 3000)
    if sub_vec_id() == 0:
        for peer in range(WORLD_SIZE):
            if peer != LOCAL_RANK:
                # 同 offset 发布（dst == src 本地指针），routing.py 惯例
                libshmem_device.putmem(t2_ptr + LOCAL_RANK * N2,
                                       t2_ptr + LOCAL_RANK * N2, N2 * 4, peer)
                libshmem_device.putmem(t3_ptr + LOCAL_RANK * N3,
                                       t3_ptr + LOCAL_RANK * N3, N3 * 4, peer)
    libshmem_device.barrier_all_vec()
    peer = (LOCAL_RANK + 1) % WORLD_SIZE
    tl.store(out2_ptr + offs2, tl.load(t2_ptr + peer * N2 + offs2))
    tl.store(out3_ptr + offs3, tl.load(t3_ptr + peer * N3 + offs3))


def _run_roundtrip(rank, world_size):
    import shmem as ash

    import tests._moe_testkit as kit

    device = f"npu:{rank}"
    with kit.aclshmem_session(rank, world_size, 256 * 1024 * 1024):
        total, elems = 2 * 16, 16
        sym = ash.aclshmem_create_tensor([total], torch.int32,
                                         device_id=rank)
        out = torch.empty(elems, dtype=torch.int32, device=device)
        try:
            _smoke_roundtrip[(1, 1, 1)](
                sym, out, LOCAL_RANK=rank, WORLD_SIZE=world_size,
                ELEMS=elems,
            )
            expect = torch.arange(elems, dtype=torch.int32) \
                + ((rank + 1) % world_size) * 1000
            assert out.cpu().tolist() == expect.tolist(), \
                f"roundtrip 值错误: {out.cpu().tolist()[:4]}..."
        finally:
            ash.aclshmem_free_tensor(sym)


def _run_multi_tensor(rank, world_size):
    import shmem as ash

    import tests._moe_testkit as kit

    device = f"npu:{rank}"
    with kit.aclshmem_session(rank, world_size, 256 * 1024 * 1024):
        # 与 MoonepWorkspace 同款约束：所有 rank 按**相同顺序、相同形状**分配
        t1 = ash.aclshmem_create_tensor([8], torch.int32, device_id=rank)
        t2 = ash.aclshmem_create_tensor([2 * 16], torch.int32,
                                        device_id=rank)
        t3 = ash.aclshmem_create_tensor([2 * 8], torch.int32, device_id=rank)
        n2, n3 = 16, 8
        out2 = torch.empty(n2, dtype=torch.int32, device=device)
        out3 = torch.empty(n3, dtype=torch.int32, device=device)
        try:
            _smoke_multi_tensor[(1, 1, 1)](
                t2, t3, out2, out3, LOCAL_RANK=rank, WORLD_SIZE=world_size,
                N2=n2, N3=n3,
            )
            peer = (rank + 1) % world_size
            assert out2.cpu().tolist() == (torch.arange(n2) + peer * 2000).tolist()
            assert out3.cpu().tolist() == (torch.arange(n3) + peer * 3000).tolist()
        finally:
            ash.aclshmem_free_tensor(t1)
            ash.aclshmem_free_tensor(t2)
            ash.aclshmem_free_tensor(t3)


@pytest.mark.dist
@pytest.mark.npu
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="需要 NPU")
def test_step0_putmem_barrier_roundtrip(dist_test):
    dist_test(_run_roundtrip, world_size=2)


@pytest.mark.dist
@pytest.mark.npu
@pytest.mark.skipif(not _NPU_AVAILABLE, reason="需要 NPU")
def test_step0_multi_symmetric_tensor(dist_test):
    dist_test(_run_multi_tensor, world_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--import-mode=importlib"]))
