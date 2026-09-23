# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""G2 cross-node transport probe: one put_signal, both directions, no MoE layers.

The r14-r26 rootinfo saga closed with the generator source (wheel
unofficial-ascend-tools 0.0.7rc2, rankinfo/product_pod.py): the official
``/etc/hccl_rootinfo.json`` is a PER-MACHINE self-description — rank_list is
enumerated from the local ``/dev/davinci*`` and no remote-server entry is ever
written.  The r14-r22 configuration was therefore already the official form,
and the original fault (cross-node writes invisible, r18-r22 counts rows)
lives at the transport or addressing layer, not in rootinfo.

This probe strips everything down to the one primitive the counts exchange
needs: ``aclshmemx_putmem_signal`` from the python API.  Each rank fills its
symmetric tensor with ``MAGIC + rank``, pushes it to the peer's tensor with an
additive signal, and after an HCCL barrier reads its LOCAL tensor.  The local
value can only be the peer's magic if the peer's put landed on this NPU.
Both directions are exercised simultaneously; the verdict is gathered via
HCCL (which demonstrably works cross-node) and asserted on both ranks.
"""

import pytest
import torch
import torch.distributed as dist

from tests import _moe_testkit as kit

MAGIC = 0x5A5A


def run_g2_transport_probe_case(rank, world_size):
    import shmem as ash
    from shmem.core.direct import SignalOp
    from shmem.core.rma import put_signal
    from shmem.core.utils import Buffer

    dev_id = kit.resolve_local_device(rank)
    torch.npu.set_device(kit.device_str(dev_id))
    peer = 1 - rank

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(1)):
        n = 64
        data = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)
        sig = ash.aclshmem_create_tensor([1], dtype=torch.int64, device_id=dev_id)
        data.fill_(0)
        sig.fill_(0)
        torch.npu.synchronize()
        dist.barrier()  # both heaps allocated and zeroed; offsets are symmetric

        # leg 1: bidirectional put_signal — my magic into the peer's tensor,
        # additive signal on the peer's signal word.
        data.fill_(MAGIC + rank)
        torch.npu.synchronize()
        data_buf = Buffer(data.data_ptr(), data.numel() * data.element_size())
        sig_buf = Buffer(sig.data_ptr(), sig.numel() * sig.element_size())
        put_signal(data_buf, data_buf, sig_buf, 1, SignalOp.SIGNAL_ADD,
                   remote_pe=peer)
        dist.barrier()
        torch.npu.synchronize()

        got = data.tolist()
        got_sig = int(sig.item())
        want = MAGIC + peer
        # The peer's put overwrote our local fill at the same symmetric offset,
        # so every element must now carry the PEER's magic.
        bad_idx = next((i for i, v in enumerate(got) if v != want), -1)
        print(f"[probe r{rank}] heap data@{hex(data.data_ptr())} sig@{hex(sig.data_ptr())} "
              f"data[0]={got[0]:#x} want={want:#x} bad_idx={bad_idx} sig={got_sig} (want 1)",
              flush=True)

        ok = (bad_idx == -1) and (got_sig == 1)
        verdict = torch.tensor([1 if ok else 0], dtype=torch.int64,
                               device=kit.device_str(dev_id))
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        assert int(verdict.item()) == 1, (
            f"cross-node put_signal probe failed on rank {rank}: "
            f"data[0]={got[0]:#x} want={want:#x} bad_idx={bad_idx} sig={got_sig}"
        )
        print(f"[probe r{rank}] PASS: cross-node put_signal landed both directions", flush=True)


@pytest.mark.dist
@pytest.mark.functional
def test_g2_transport_probe_w2(dist_test):
    dist_test(run_g2_transport_probe_case, world_size=2)


def run_g2_pull_probe_case(rank, world_size):
    """G2 pull-leg probe: getmem (PULL) both directions, no push involved.

    The r27d/r27f/r28 push matrix showed every engine carries bulk in exactly
    one cross-node direction (udma: node1->node0 only; mte: node0->node1 only;
    the dead direction drops silently at every log level).  A PULL inverts the
    initiator: if getmem works where same-engine put dies, the counts exchange
    can be redesigned pull-side and the whole G2 data plane unblocks without
    any vendor fix.  Each rank fills its symmetric tensor with MAGIC + rank,
    then getmems the peer's tensor into a second symmetric buffer — the
    symmetric heap maps the same VA on every PE, so the remote source address
    equals the local one.
    """
    import shmem as ash

    dev_id = kit.resolve_local_device(rank)
    torch.npu.set_device(kit.device_str(dev_id))
    peer = 1 - rank

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(1)):
        n = 64
        data = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)
        recv = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)
        data.fill_(0)
        recv.fill_(0)
        torch.npu.synchronize()
        dist.barrier()  # heaps allocated symmetrically: data/recv offsets match

        data.fill_(MAGIC + rank)
        torch.npu.synchronize()
        dist.barrier()  # peer's fill visible-to-pull

        ash.aclshmem_getmem(recv.data_ptr(), data.data_ptr(),
                            data.numel() * data.element_size(), peer)
        torch.npu.synchronize()
        dist.barrier()  # both pulls issued before anyone reads

        got = recv.tolist()
        want = MAGIC + peer
        bad_idx = next((i for i, v in enumerate(got) if v != want), -1)
        print(f"[pull r{rank}] heap data@{hex(data.data_ptr())} recv@{hex(recv.data_ptr())} "
              f"recv[0]={got[0]:#x} want={want:#x} bad_idx={bad_idx}", flush=True)

        ok = bad_idx == -1
        verdict = torch.tensor([1 if ok else 0], dtype=torch.int64,
                               device=kit.device_str(dev_id))
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        assert int(verdict.item()) == 1, (
            f"cross-node getmem probe failed on rank {rank}: "
            f"recv[0]={got[0]:#x} want={want:#x} bad_idx={bad_idx}"
        )
        print(f"[pull r{rank}] PASS: cross-node getmem landed both directions", flush=True)


@pytest.mark.dist
@pytest.mark.functional
def test_g2_pull_probe_w2(dist_test):
    dist_test(run_g2_pull_probe_case, world_size=2)


def run_g2_pull1_probe_case(rank, world_size):
    """G2 asymmetric pull probe: ONLY rank 1 (node1) getmems from rank 0.

    The r27-r30 matrix: every cross-node op node1 INITIATES under udma works
    (push, r27d/r28/r29A); every node0-initiated op dies (push silent-drops,
    getmem SIGABRTs with ERR02005, r29B); mixed engine masks fail init (r30).
    The untested cell — node1-initiated getmem from node0 — carries the
    "node1 initiates everything" workaround architecture: node1 would push
    its own payloads AND pull node0's.  Rank 0 stays passive here (no local
    RMA), so a node1 crash surfaces as a spawn ProcessExitedException on
    rank 0 rather than a wrong-data assert.
    """
    import shmem as ash

    dev_id = kit.resolve_local_device(rank)
    torch.npu.set_device(kit.device_str(dev_id))
    peer = 1 - rank

    with kit.aclshmem_session(rank, world_size, kit.get_ash_size_bytes(1)):
        n = 64
        data = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)
        recv = ash.aclshmem_create_tensor([n], dtype=torch.int64, device_id=dev_id)
        data.fill_(0)
        recv.fill_(0)
        torch.npu.synchronize()
        dist.barrier()

        data.fill_(MAGIC + rank)
        torch.npu.synchronize()
        dist.barrier()

        if rank == 1:
            ash.aclshmem_getmem(recv.data_ptr(), data.data_ptr(),
                                data.numel() * data.element_size(), peer)
            torch.npu.synchronize()
        dist.barrier()

        ok = True
        if rank == 1:
            got = recv.tolist()
            want = MAGIC + peer
            bad_idx = next((i for i, v in enumerate(got) if v != want), -1)
            print(f"[pull1 r1] data@{hex(data.data_ptr())} recv@{hex(recv.data_ptr())} "
                  f"recv[0]={got[0]:#x} want={want:#x} bad_idx={bad_idx}", flush=True)
            ok = bad_idx == -1
        verdict = torch.tensor([1 if ok else 0], dtype=torch.int64,
                               device=kit.device_str(dev_id))
        dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
        assert int(verdict.item()) == 1, (
            f"node1-initiated getmem probe failed on rank {rank}"
        )
        print(f"[pull1 r{rank}] PASS: node1-initiated cross-node getmem works", flush=True)


@pytest.mark.dist
@pytest.mark.functional
def test_g2_pull1_probe_w2(dist_test):
    dist_test(run_g2_pull1_probe_case, world_size=2)
