# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""All-to-all transport baselines for the mega-MoE exchange sizing.

The fullnet mega path moves one exchange per direction per layer at
S*topk*H bf16 bytes per rank (kimik3 single-node: 1024*8*7168*2 = 112MB).
This benchmark pins the platform floors for that shape so the mega
windows (fwd ``waves``, bwd w2/w3) can be read against what the fabric
itself can do:

  1. HCCL ``all_to_all_single`` (equal and skewed splits) — the platform
     collective baseline.
  2. ACLSHMEM symmetric-heap put_signal exchange (MTE|UDMA combo engine,
     the family the mega kernels ride) — the transport floor without any
     compute riding on it.

Timing is the blocking per-call latency (sync, t0, op, sync, t1); per rep
the SLOWEST rank's time is the exchange time (all_reduce MAX).
Correctness is asserted per size (each received row must carry its
sender's tag) so a silent drop cannot read as speed.

Run (8 ranks, single node):
  pytest tests/layer/test_all2all_baseline.py -x -q
"""

import statistics
import time

import pytest
import torch
import torch.distributed as dist

from tests import _moe_testkit as kit

# kimik3 single-node mega exchange: 1024 tokens * topk 8 * H 7168 * bf16.
# The sweep brackets it (8/32/64/112/224 MB per rank out).
_SIZES_MB = (8, 32, 64, 112, 224)
_REPS = 10
_WARMUP = 3


def _dev(rank):
    dev_id = kit.resolve_local_device(rank)
    torch.npu.set_device(kit.device_str(dev_id))


def _timed(fn, reps=_REPS, warmup=_WARMUP):
    """Blocking per-call latency of fn, reduced to the slowest rank."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    times = []
    for _ in range(reps):
        torch.npu.synchronize()
        dist.barrier()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append(time.perf_counter() - start)
    # HCCL on this build has no kDouble collective — float32 ms values are
    # plenty for a p50/p90 latency table.
    t = torch.tensor(times, dtype=torch.float32, device="npu") * 1e3
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return [float(x) for x in t.cpu().tolist()]


def _report(label, total_mb, times_ms):
    p50 = statistics.median(times_ms)
    gbps = total_mb / 1024.0 / (p50 / 1e3)
    if dist.get_rank() == 0:
        print(
            f"[all2all-bench] {label:34s} {total_mb:6.0f}MB/rank "
            f"p50={p50:8.3f}ms p90={sorted(times_ms)[int(len(times_ms)*0.9)]:8.3f}ms "
            f"per-rank={gbps:7.2f}GB/s",
            flush=True,
        )


def _run_hccl_case(rank, world_size, total_mb, skewed=False):
    _dev(rank)
    W = world_size
    per_peer = total_mb * 1024 * 1024 // 2 // W  # bf16 elems per peer
    if skewed:
        # one heavy peer + one light (a routing hotspot shape); sum fixed
        weights = [1.6, 0.4] + [1.0] * (W - 2)
        splits = [int(per_peer * W * w // sum(weights)) for w in weights]
        splits[-1] += per_peer * W - sum(splits)
    else:
        splits = [per_peer] * W
    total = sum(splits)
    inp = torch.full((total,), rank, dtype=torch.bfloat16, device="npu")
    out = torch.empty_like(inp)

    def fn():
        dist.all_to_all_single(out, inp, output_split_sizes=splits,
                               input_split_sizes=splits)

    times = _timed(fn)
    off = 0
    for i, n in enumerate(splits):
        got = out[off:off + min(n, 64)]
        assert torch.all(got == float(i)), f"hccl block {i} corrupted"
        off += n
    _report(f"hccl all_to_all {'skewed' if skewed else 'equal'}", total_mb,
            times)


def _run_shmem_case(rank, world_size, total_mb):
    import shmem as ash
    from shmem.core.direct import ComparisonType, SignalOp
    from shmem.core.rma import put_signal, quiet, signal_wait
    from shmem.core.utils import Buffer

    _dev(rank)
    W = world_size
    per_peer = total_mb * 1024 * 1024 // 2 // W
    dev_id = kit.resolve_local_device(rank)
    # The session is owned by the caller: re-initializing aclshmem seconds
    # after a finalize on the same bootstrap port (8666) trips its TIME_WAIT
    # residue and fails with aclshmem_init ret != 0 (r7), so the whole shmem
    # sweep shares ONE session and each size allocates its tensors inside it
    # (worst case 2*(8+32+64+112+224)MB = 880MB of the 2GB heap).
    # send[p]  = my outgoing chunk destined for peer p
    # recv[i]  = rank i's contribution to me (row written by rank i)
    # sig[i]   = write count on recv row i (exactly 1 writer per row)
    send = ash.aclshmem_create_tensor(
        [W, per_peer], dtype=torch.bfloat16, device_id=dev_id)
    recv = ash.aclshmem_create_tensor(
        [W, per_peer], dtype=torch.bfloat16, device_id=dev_id)
    sig = ash.aclshmem_create_tensor(
        [W], dtype=torch.int64, device_id=dev_id)
    send.copy_(torch.full((W, per_peer), rank, dtype=torch.bfloat16,
                          device="npu"))
    recv.zero_()
    sig.zero_()
    torch.npu.synchronize()
    dist.barrier()  # heaps allocated and zeroed; offsets symmetric

    row_bytes = per_peer * 2
    my_row = rank * row_bytes

    def fn():
        for p in range(W):
            if p == rank:
                recv[rank].copy_(send[rank])  # self leg stays local
                continue
            # dst = peer p's recv[rank] at the same symmetric offset;
            # src = my send[p]; the +1 lands on peer p's sig[rank]
            put_signal(
                Buffer(recv.data_ptr() + my_row, row_bytes),
                Buffer(send.data_ptr() + p * row_bytes, row_bytes),
                Buffer(sig.data_ptr() + rank * 8, 8),
                1, SignalOp.SIGNAL_ADD, remote_pe=p)
        quiet(0)
        for i in range(W):
            if i == rank:
                continue
            signal_wait(Buffer(sig.data_ptr() + i * 8, 8), 1,
                        ComparisonType.CMP_GE, stream=0)
        torch.npu.synchronize()
        sig.zero_()
        torch.npu.synchronize()
        dist.barrier()

    times = _timed(fn)
    for i in range(W):
        got = recv[i, :64]
        assert torch.all(got == float(i)), f"shmem row {i} corrupted"
    _report("shmem put_signal (MTE|UDMA)", total_mb, times)


def _all2all_worker(rank, world_size):
    for mb in _SIZES_MB:
        _run_hccl_case(rank, world_size, mb)
    _run_hccl_case(rank, world_size, 112, skewed=True)
    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(2),
                              enable_udma=True):
        for mb in _SIZES_MB:
            _run_shmem_case(rank, world_size, mb)


def test_all2all_baselines(dist_test):
    """8-rank equal/skewed HCCL + shmem put exchange floors."""
    dist_test(_all2all_worker, world_size=8, backend="hccl")
