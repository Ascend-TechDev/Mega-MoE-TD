# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# ============================================================================
#  dispatch_fc2_bwd.py  —  step 1: dispatch-A2A(home->expert) + fc2 input-grad
#  C2 all-core per-tile signal/wait pipeline (mirrors forward dispatch_fc1):
#  the Vector side pushes source-local BLOCK_M tiles by (dst,expert) bucket with
#  signal_op SET, the Cube side merged-window dl.wait/consume_token + contiguous
#  GEMM. No barrier_all — push and GEMM overlap on every AI core. peer_mem is
#  written expert-major and read contiguously, so no local_sort gather is needed.
# ============================================================================

import os

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .common import ncore, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K


def _dispatch_static_maps(saved):
    """Build the dy-independent expert-major dispatch maps once, cached on `saved`.

    The producer pushes grad buckets in (dst, expert) order and the consumer reads
    peer_mem expert-major contiguously, so every map here is expert-major (mirror
    of forward dispatch_fc1), not the rank-major sort_idxs layout.

    A MoonEP ``saved_phys`` (``use_moonep``) already carries the physical
    ``(destination, slot)`` plan metadata, so the bincount / argsort / all_gather
    reconstruction below is skipped entirely (see
    :func:`_dispatch_static_maps_moonep`)."""
    cache = saved.get("_dispatch_cache")
    if cache is not None:
        return cache
    if saved.get("use_moonep"):
        return _dispatch_static_maps_moonep(saved)
    device = f"npu:{saved['ep_rank']}"
    pe = saved["ep_rank"]; W = saved["world_size"]; H = saved["hidden_dim"]
    ep_group = saved["ep_group"]; total_send = saved["total_send"]

    EPR = saved["experts_per_rank"]
    flat = saved["selected_experts"].to(torch.int64).to(device).reshape(-1)   # [total_send] global expert
    send_counts_re = torch.bincount(flat, minlength=W * EPR).to(torch.int32).reshape(W, EPR)  # [W,EPR]
    all_send_e = torch.stack(all_gather_list(send_counts_re.reshape(-1), ep_group)).reshape(W, W, EPR)
    recv_counts_re = all_send_e[:, pe, :]                                       # [W, EPR] (source, le)
    recv_per_expert = recv_counts_re.sum(0).to(torch.int32)                     # [EPR]
    recv_expert_offs = torch.zeros(EPR + 1, dtype=torch.int32, device=device)
    recv_expert_offs[1:] = recv_per_expert.cumsum(0).to(torch.int32)            # [EPR+1]
    bwd_expert_sort = torch.argsort(flat.to(torch.float32), stable=True).to(torch.int32)  # [total_send]
    send_counts_flat = send_counts_re.reshape(-1)
    send_bucket_starts = torch.zeros(W * EPR + 1, dtype=torch.int32, device=device)
    send_bucket_starts[1:] = send_counts_flat.cumsum(0).to(torch.int32)
    dst_total = all_send_e.sum(0)                                               # [W, EPR]
    expert_base = torch.zeros_like(dst_total)
    expert_base[:, 1:] = dst_total.cumsum(1)[:, :-1]
    source_prefix = all_send_e[:pe].sum(0).to(torch.int32) if pe > 0 else \
        torch.zeros(W, EPR, dtype=torch.int32, device=device)
    send_bucket_dst_starts = (expert_base.to(torch.int32) + source_prefix).reshape(-1).contiguous()

    # sanity: expert-major metadata must match the rank-major counts already in saved
    assert int(send_counts_re.sum().item()) == total_send, \
        f"send_counts_re sum {send_counts_re.sum().item()} != total_send {total_send}"
    assert int(recv_per_expert.sum().item()) == saved["total_recv"], \
        f"recv_per_expert sum {recv_per_expert.sum().item()} != total_recv {saved['total_recv']}"

    _local_max = torch.tensor([int(send_counts_re.max().item())], dtype=torch.int64, device=device)
    dist.all_reduce(_local_max, op=dist.ReduceOp.MAX, group=ep_group)
    _global_max_bwd_tiles = max(1, (int(_local_max.item()) + 64 - 1) // 64)
    cache = dict(
        M=saved["M"], N=saved["ffn_dim"], K=H, E=EPR,
        fc2=saved["fc2"].contiguous(),
        total_send=total_send, H=H, total_recv=saved["total_recv"],
        # expert-major signal/wait metadata
        send_counts_re=send_counts_flat.contiguous(), send_bucket_starts=send_bucket_starts,
        send_bucket_dst_starts=send_bucket_dst_starts,
        recv_per_expert=recv_per_expert, recv_expert_offs=recv_expert_offs,
        bwd_expert_sort=bwd_expert_sort,
        # receive counts [W,EPR] (source, expert) for merged-window overlap, and the
        # source-tile slot capacity (max tiles any (dst,expert) bucket produces).
        recv_counts_re=recv_counts_re.contiguous(),
        max_bwd_tiles=_global_max_bwd_tiles,
    )
    saved["_dispatch_cache"] = cache
    return cache


def _dispatch_static_maps_moonep(saved):
    """Physical-slot dispatch maps consumed straight from the saved MoonEP plan.

    ``saved_phys`` snapshots the planner's ``(destination, physical slot)``
    bucket metadata, which is exactly the expert-major layout this step needs:

    * ``plan_send_counts_by_rank_expert``  [W, epn+B]  per-(dst, slot) sends
    * ``plan_send_bucket_starts``          [W*(epn+B)] send-order bucket starts
    * ``plan_send_bucket_dst_starts``      [W*(epn+B)] receive-side offsets in
      the destination's (slot, source) row order
    * ``plan_recv_counts_by_source_expert`` [W, epn+B] + ``expert_counts`` /
      ``plan_received_expert_offsets``     the receive-side slot tables
    * ``sort_idxs`` (== plan send_route_indices)  send position -> flat route id

    ``E`` therefore becomes the *physical* expert stride ``epn + B`` so the
    bucket indexing and the ``_bwd_tile_signal_mem`` slot formula
    ``(source * E + slot) * MAX_BWD_TILES + tile`` stay aligned on both the
    producer and the consumer side. Only ``max_bwd_tiles`` still needs a
    collective: it bounds the per-(source, slot) 64-row push tiles, which every
    rank derives from its own plan send counts and MAX-reduces over the group.
    """
    device = f"npu:{saved['ep_rank']}"
    ep_group = saved["ep_group"]
    physical_experts = int(saved["physical_experts_per_rank"])
    send_counts = saved["plan_send_counts_by_rank_expert"].to(device)
    _local_max = torch.tensor(
        [int(send_counts.max().item())], dtype=torch.int64, device=device
    )
    dist.all_reduce(_local_max, op=dist.ReduceOp.MAX, group=ep_group)
    cache = dict(
        M=saved["M"], N=saved["ffn_dim"], K=saved["hidden_dim"], E=physical_experts,
        fc2=saved["fc2"].contiguous(),
        replica_fc2=saved["replica_down"].contiguous(),
        home_experts=int(saved["home_experts_per_rank"]),
        active_experts=int(saved["active_physical_experts_per_rank"]),
        use_moonep=True,
        total_send=saved["total_send"], H=saved["hidden_dim"],
        total_recv=saved["total_recv"],
        send_counts_re=send_counts.reshape(-1).contiguous(),
        send_bucket_starts=(
            saved["plan_send_bucket_starts"].to(device).contiguous()
        ),
        send_bucket_dst_starts=(
            saved["plan_send_bucket_dst_starts"].to(device).contiguous()
        ),
        recv_per_expert=saved["expert_counts"].to(device).contiguous(),
        recv_expert_offs=(
            saved["plan_received_expert_offsets"].to(device).contiguous()
        ),
        bwd_expert_sort=saved["sort_idxs"].to(device).contiguous(),
        recv_counts_re=(
            saved["plan_recv_counts_by_source_expert"].to(device).contiguous()
        ),
        max_bwd_tiles=max(1, (int(_local_max.item()) + 63) // 64),
    )
    saved["_dispatch_cache"] = cache
    return cache


def _prepare_dispatch_fc2_bwd(saved, dy):
    """Build the expert-major gco (grad_combined_out_flat) for the per-tile
    signal/wait dispatch. Static maps are cached on `saved`; only the dy-dependent
    gco is built per call."""
    p = _dispatch_static_maps(saved)
    topk = saved["topk"]
    # gco in expert-major (bwd_expert_sort) order: the producer pushes (dst,expert)
    # buckets and the consumer reads peer_mem contiguously — no sort_idxs gather.
    gco = dy.repeat_interleave(topk, dim=0)[p["bwd_expert_sort"].to(torch.int64)].contiguous()
    p = dict(p)
    p["gco"] = gco
    return p


# ============================================================================
# C2 per-tile signal/wait: producer publishes each source-local BLOCK_M tile
# with signal_op SET(epoch); consumer merged-window per-tile dl.wait/consume_token
# + contiguous GEMM. No barrier_all — readiness fences only the window a Cube GEMM
# is about to read. Mirrors the verified all-core tile-readiness pipeline of
# forward _kernel_dispatch_fc1.
# ============================================================================
@triton.jit
def _fc2_bwd_gemm_one_mn_tile(
    input_ptr, weight_ptr, output_ptr,
    expert_id, m_off, m_size, n_tile, N, K,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    dtype: tl.constexpr,
):
    """fc2 input-grad GEMM one tile. A = peer_mem[m_off, H] CONTIGUOUS (no gather),
    B = fc2[expert][H, ffn], out = grad_swiglu[m_off, ffn]. WEIGHT_EXPERT_BASE
    re-bases the weight table (home table: base 0; replica table: base epn),
    mirroring forward dispatch_fc1's dual-weight launches."""
    if m_size > 0:
        om = tl.arange(0, BLOCK_M); on_ = tl.arange(0, BLOCK_N); ok = tl.arange(0, BLOCK_K)
        m_offs = m_off + om; m_mask = om < m_size
        n_offs = n_tile * BLOCK_N + on_; n_mask = n_offs < N
        wb = (expert_id.to(tl.int64) - WEIGHT_EXPERT_BASE) * stride_we
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ks in range(0, K, BLOCK_K):
            k_offs = ks + ok; k_mask = k_offs < K
            ao = m_offs[:, None] * stride_im + k_offs[None, :] * stride_ik
            a = tl.load(input_ptr + ao, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            bo = k_offs[:, None] * stride_wk + n_offs[None, :] * stride_wn
            b = tl.load(weight_ptr + wb + bo, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(a, b)
        co = m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
        tl.store(output_ptr + co, acc.to(output_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _dispatch_grad_source_tiles(
    pid, num_cores,
    gco_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    signal_epoch,
    H: tl.constexpr, stride_gm,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    MAX_BWD_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
):
    """Per-(dst,expert,tile) putmem + fence + signal_op SET. Adapted from forward
    _dispatch_one_source_tile_task (dispatch_fc1.py:710-762): the staging buffer
    (gco) is already expert-major contiguous, so each BLOCK_M tile is one bulk
    putmem (no per-token gather). Expert-major work order so every dst receives
    early expert tiles concurrently. slot = (LOCAL_RANK*EPR+expert)*MAX_BWD_TILES
    +source_tile; the consumer waits the same slot keyed by source_id == LOCAL_RANK."""
    num_tasks: tl.constexpr = WORLD_SIZE * EXPERTS_PER_RANK
    for work_id in range(pid, num_tasks, num_cores):
        dst_rank = work_id % WORLD_SIZE
        expert_id = work_id // WORLD_SIZE
        task_id = dst_rank * EXPERTS_PER_RANK + expert_id
        task_start = tl.load(send_bucket_starts_ptr + task_id)
        task_count = tl.load(send_counts_re_ptr + task_id)
        task_dst_start = tl.load(send_bucket_dst_starts_ptr + task_id)
        num_source_tiles = tl.cdiv(task_count, BLOCK_M)
        for source_tile in range(num_source_tiles):
            tile_start = source_tile * BLOCK_M
            tile_count = tl.minimum(BLOCK_M, task_count - tile_start)
            libshmem_device.putmem(
                peer_mem_ptr + (task_dst_start + tile_start) * H,
                gco_ptr + (task_start + tile_start) * stride_gm,
                tile_count * H * 2, dst_rank)
            libshmem_device.fence()
            signal_slot = (
                (LOCAL_RANK * EXPERTS_PER_RANK + expert_id) * MAX_BWD_TILES
                + source_tile
            )
            libshmem_device.signal_op(
                signal_mem_ptr + signal_slot * 16,
                signal_epoch,
                libshmem_device.ACLSHMEM_SIGNAL_SET,
                dst_rank)


@triton.jit
def _fc2_bwd_gemm_merged_tiles_wait(
    pid, ncore,
    peer_mem_ptr, signal_mem_ptr, fc2_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    signal_epoch,
    N, K,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    TILE_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
    FIRST_EXPERT: tl.constexpr, LAST_EXPERT: tl.constexpr,
    WEIGHT_EXPERT_BASE: tl.constexpr,
    MAX_BWD_TILES: tl.constexpr, dtype: tl.constexpr,
    b1_signal_ptr, b1_epoch,
    SIGNAL_ON: tl.constexpr, LOCAL_RANK: tl.constexpr,
):
    """Consume merged expert M windows as their source tiles become ready. Adapted
    from forward _triton_grouped_gemm_expert_n_merged_tiles_wait
    (dispatch_fc1.py:539-629). GEMM ownership is keyed by (expert, n_tile); source
    rank participates only in the readiness dependency, so source fragments no
    longer force independent full-weight scans. waitValue = signal_epoch (SET mode,
    ONE value per epoch). Reuses the contiguous-read _fc2_bwd_gemm_one_mn_tile
    (stride_im=H, stride_ik=1).

    FIRST_EXPERT/LAST_EXPERT/WEIGHT_EXPERT_BASE restrict the consumed expert
    range to one weight table (forward dispatch_fc1's dual-launch pattern): the
    home table serves slots [0, epn) and the replica table serves
    [epn, active). EXPERTS_PER_RANK stays the physical slot stride of the
    bucket/signal tables.

    BLOCK_M is the GEMM tile height AND the merged readiness window; TILE_M is
    the producer's push/signal tile height (64, matches the signal-slot layout).
    A BLOCK_M=128 window simply waits every 64-row source tile overlapping it
    before running one double-height GEMM — the readiness protocol itself is
    unchanged, which decouples the GEMM L0A fill (a-tile [64,BK] bf16 = half of
    64KB L0A) from the transport tile granularity.

    SIGNAL_ON=1 (MOE_MEGA_TILE_B1=1, mega backward only) additionally fires a
    LOCAL readiness SET after each (expert, n_tile, m_window) tile of the
    output lands: fence() + signal_op to this rank's b1 slot, the probe-3
    producer side.  Slot layout (expert*num_n_tiles + n_tile)*max_win +
    m_window, where max_win — the max window count any expert occupies — is
    derived from recv_per_expert INSIDE the kernel, so producer and consumer
    share one slot formula with no host-side table.  The downstream swiglu
    phase then merged-waits a window's num_n_tiles slots instead of crossing
    the B1 barrier."""
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    max_win = 1
    if SIGNAL_ON:
        for e0 in range(EXPERTS_PER_RANK):
            sz0 = tl.load(recv_per_expert_ptr + e0)
            max_win = tl.maximum(max_win, tl.cdiv(sz0, BLOCK_M))
    first_task = FIRST_EXPERT * num_n_tiles
    last_task = LAST_EXPERT * num_n_tiles
    for task_id in range(pid + first_task, last_task, ncore):
        expert_id = task_id // num_n_tiles
        n_tile = task_id % num_n_tiles
        expert_size = tl.load(recv_per_expert_ptr + expert_id)
        expert_off = tl.load(recv_expert_offs_ptr + expert_id)
        if expert_size > 0:
            num_m_windows = tl.cdiv(expert_size, BLOCK_M)
            for m_window in range(num_m_windows):
                window_start = m_window * BLOCK_M
                window_size = tl.minimum(BLOCK_M, expert_size - window_start)
                window_end = window_start + window_size
                source_start = 0
                ready_token = 0
                # peer_mem is expert-major then source-major. Acquire every
                # dispatch tile whose rows overlap this merged expert window.
                for source_id in range(WORLD_SIZE):
                    source_size = tl.load(
                        recv_counts_re_ptr
                        + source_id * EXPERTS_PER_RANK
                        + expert_id)
                    source_end = source_start + source_size
                    overlap_start = tl.maximum(window_start, source_start)
                    overlap_end = tl.minimum(window_end, source_end)
                    if overlap_start < overlap_end:
                        first_source_tile = (
                            overlap_start - source_start
                        ) // TILE_M
                        last_source_tile = (
                            overlap_end - source_start - 1
                        ) // TILE_M
                        for source_tile in range(
                            first_source_tile, last_source_tile + 1
                        ):
                            signal_slot = (
                                (source_id * EXPERTS_PER_RANK + expert_id)
                                * MAX_BWD_TILES
                                + source_tile
                            )
                            token = dl.wait(
                                signal_mem_ptr + signal_slot * 16,
                                1, "gpu", "acquire",
                                waitValue=signal_epoch)
                            ready_token += token
                    source_start = source_end
                ready_input_ptr = dl.consume_token(peer_mem_ptr, ready_token)
                _fc2_bwd_gemm_one_mn_tile(
                    ready_input_ptr, fc2_ptr, output_ptr,
                    expert_id, expert_off + window_start, window_size, n_tile, N, K,
                    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
                    BLOCK_M, BLOCK_N, BLOCK_K, WEIGHT_EXPERT_BASE, dtype)
                if SIGNAL_ON:
                    # B1 readiness: this (expert, n_tile, m_window) output tile
                    # has landed — probe 3's local cube->vector fence/signal
                    # chain, producer side (fence() orders the FixPipe store
                    # before the SET).
                    libshmem_device.fence()
                    libshmem_device.signal_op(
                        b1_signal_ptr
                        + ((expert_id * num_n_tiles + n_tile) * max_win
                           + m_window) * 16,
                        b1_epoch, libshmem_device.ACLSHMEM_SIGNAL_SET,
                        LOCAL_RANK)


@triton.jit(do_not_specialize=["signal_epoch"])
def kernel_dispatch_fc2_bwd_tile_signal(
    gco_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    H: tl.constexpr, stride_gm,
    fc2_ptr, output_ptr,
    replica_fc2_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    N, K,
    stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    stride_rwe, stride_rwk, stride_rwn,
    signal_epoch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PUSH_BLOCK_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
    HOME_EXPERTS_PER_RANK: tl.constexpr, ACTIVE_EXPERTS_PER_RANK: tl.constexpr,
    USE_REPLICA_WEIGHTS: tl.constexpr,
    MAX_BWD_TILES: tl.constexpr, LOCAL_RANK: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
):
    """C2 fused per-tile dispatch + fc2 input-grad. The Vector side pushes
    source-local BLOCK_M tiles + signal_op SET; the Cube side merged-window
    dl.wait/consume_token + contiguous GEMM. They run concurrently on every AI
    core (all-core pipeline), fenced per source-tile — no barrier_all. Mirrors the
    verified ALL_CORE_PIPELINE + tile-readiness path of forward
    _kernel_dispatch_fc1. sub_vec_id()==0 gates the push (a mixed kernel has two
    vector sub-cores that would otherwise duplicate the putmem).

    The Cube side runs one GEMM sweep per weight table (forward dispatch_fc1's
    dual-launch pattern): the home table (fc2_ptr) serves slots
    [0, HOME_EXPERTS_PER_RANK) and, when USE_REPLICA_WEIGHTS, the replica table
    (replica_fc2_ptr, re-based at HOME_EXPERTS_PER_RANK) serves
    [HOME_EXPERTS_PER_RANK, ACTIVE_EXPERTS_PER_RANK). EXPERTS_PER_RANK is the
    physical slot stride of the bucket / signal-slot tables, so it equals
    epn + B on the MoonEP path and epn otherwise (where the replica sweep is
    compiled out and the behavior is unchanged)."""
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(axis=0)
    dtype = tl.bfloat16
    with al.scope(core_mode="vector", disable_auto_sync=True):
        if sub_vec_id() == 0:
            _dispatch_grad_source_tiles(
                pid, num_cores,
                gco_ptr, peer_mem_ptr, signal_mem_ptr,
                send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
                signal_epoch, H, stride_gm,
                LOCAL_RANK, WORLD_SIZE, EXPERTS_PER_RANK, MAX_BWD_TILES, PUSH_BLOCK_M, BLOCK_H_PUSH)
    with al.scope(core_mode="cube", disable_auto_sync=True):
        _fc2_bwd_gemm_merged_tiles_wait(
            pid, num_cores,
            peer_mem_ptr, signal_mem_ptr, fc2_ptr, output_ptr,
            recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
            signal_epoch,
            N, K, stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
            BLOCK_M, BLOCK_N, BLOCK_K, PUSH_BLOCK_M,
            WORLD_SIZE, EXPERTS_PER_RANK,
            0, HOME_EXPERTS_PER_RANK, 0,
            MAX_BWD_TILES, dtype,
            signal_mem_ptr, 0, SIGNAL_ON=0, LOCAL_RANK=LOCAL_RANK)
        if USE_REPLICA_WEIGHTS:
            _fc2_bwd_gemm_merged_tiles_wait(
                pid, num_cores,
                peer_mem_ptr, signal_mem_ptr, replica_fc2_ptr, output_ptr,
                recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
                signal_epoch,
                N, K, stride_im, stride_ik,
                stride_rwe, stride_rwk, stride_rwn, stride_om, stride_on,
                BLOCK_M, BLOCK_N, BLOCK_K, PUSH_BLOCK_M,
                WORLD_SIZE, EXPERTS_PER_RANK,
                HOME_EXPERTS_PER_RANK, ACTIVE_EXPERTS_PER_RANK,
                HOME_EXPERTS_PER_RANK,
                MAX_BWD_TILES, dtype,
                signal_mem_ptr, 0, SIGNAL_ON=0, LOCAL_RANK=LOCAL_RANK)


def _dispatch_gemm_tile():
    """Step1 fc2 input-grad GEMM tile, env-tunable (does NOT touch
    common.BLOCK_SIZE_*). Defaults: BM=128 / BN=256 / BK=128 — measured best on
    Kimi-K3 w8 t2k (42.5 -> 35.5ms e2e; BM=64 left the a-tile at half of L0A,
    per AscendKernelWiki pattern-low-mte-utilization-small-tile); the
    transport/signal granularity stays 64 (PUSH_BLOCK_M), so a taller GEMM tile
    only widens the merged readiness window."""
    return (
        int(os.environ.get("MOE_DISPATCH_GEMM_BM", "128")),
        int(os.environ.get("MOE_DISPATCH_GEMM_BN", "256")),
        int(os.environ.get("MOE_DISPATCH_GEMM_BK", "128")),
    )


def _ensure_bwd_signal_mem(saved, W, EPR, MAX_BWD_TILES):
    """Lazy-alloc the shared backward tile-signal workspace on `saved`.

    One SET slot per (source, expert, tile). Slot layout must match
    producer/consumer: (rank*EPR+expert)*MAX_BWD_TILES+tile, with rank as the
    source id. 16 int32 elements per slot (= 64 bytes) mirrors the forward
    workspace's signal slot granularity. Shared by the standalone step-1
    launcher and the one-kernel mega backward (mega_bwd.py); the SET epoch
    lives alongside it in ``saved["_bwd_tile_signal_epoch"]``."""
    import shmem as ash
    signal_mem = saved.get("_bwd_tile_signal_mem")
    if signal_mem is None:
        signal_mem = ash.aclshmem_create_tensor(
            [W * EPR * MAX_BWD_TILES * 16],
            dtype=torch.int32,
            device_id=saved["ep_rank"])
        signal_mem.zero_()
        saved["_bwd_tile_signal_mem"] = signal_mem
    return signal_mem


def _launch_dispatch_fc2_bwd_tile_signal(prep, peer_mem, out, saved):
    W = saved["world_size"]
    EPR = prep["E"]
    MAX_BWD_TILES = prep["max_bwd_tiles"]
    fc2 = prep["fc2"]; H = prep["H"]; N = prep["N"]; K = prep["K"]
    # Dual weight tables (MoonEP physical slots); the home-only path points the
    # replica entry at the home table and compiles the second GEMM sweep out.
    replica_fc2 = prep.get("replica_fc2", fc2)
    home_experts = prep.get("home_experts", EPR)
    active_experts = prep.get("active_experts", EPR)
    use_replica_weights = prep.get("use_moonep", False)
    signal_mem = _ensure_bwd_signal_mem(saved, W, EPR, MAX_BWD_TILES)
    # SET-mode epoch: producer writes signal_epoch, consumer waits waitValue=
    # signal_epoch. Bump after launch so the next call sees a fresh value (no
    # need to zero the slots — SET overwrites unconditionally).
    signal_epoch = saved.get("_bwd_tile_signal_epoch", 1)
    saved["_bwd_tile_signal_epoch"] = signal_epoch + 1
    gemm_bm, gemm_bn, gemm_bk = _dispatch_gemm_tile()
    kernel_dispatch_fc2_bwd_tile_signal[(ncore(), 1, 1)](
        prep["gco"], peer_mem, signal_mem,
        prep["send_bucket_starts"], prep["send_counts_re"], prep["send_bucket_dst_starts"],
        H, prep["gco"].stride(0),
        fc2, out, replica_fc2,
        prep["recv_per_expert"], prep["recv_expert_offs"], prep["recv_counts_re"],
        N, K, H, 1, fc2.stride(0), fc2.stride(1), fc2.stride(2), N, 1,
        replica_fc2.stride(0), replica_fc2.stride(1), replica_fc2.stride(2),
        signal_epoch,
        BLOCK_M=gemm_bm, BLOCK_N=gemm_bn, BLOCK_K=gemm_bk, PUSH_BLOCK_M=64,
        WORLD_SIZE=W, EXPERTS_PER_RANK=EPR,
        HOME_EXPERTS_PER_RANK=home_experts,
        ACTIVE_EXPERTS_PER_RANK=active_experts,
        USE_REPLICA_WEIGHTS=use_replica_weights,
        MAX_BWD_TILES=MAX_BWD_TILES,
        LOCAL_RANK=saved["ep_rank"], BLOCK_H_PUSH=256, num_warps=8)
    return out


def dispatch_fc2_bwd_triton(saved, dy, peer_mem):
    """Step 1: returns (grad_swiglu [M,ffn], grad_fc2_out_sorted [M,H]).

    Single path: the C2 all-core per-tile signal/wait pipeline — Vector putmem +
    signal_op SET, Cube merged-window dl.wait/consume_token + contiguous GEMM, no
    barrier_all, no local_sort gather. peer_mem is written expert-major and IS the
    sorted layout, so grad_fc2_out_sorted is an identity view over it."""
    prep = _prepare_dispatch_fc2_bwd(saved, dy)
    out = torch.empty(prep["M"], prep["N"], dtype=dy.dtype, device=dy.device)
    _launch_dispatch_fc2_bwd_tile_signal(prep, peer_mem, out, saved)
    # expert-major peer_mem IS the sorted layout -> identity view (no gather)
    grad_fc2_out_sorted = peer_mem.view(-1)[:prep["total_recv"] * prep["H"]].view(
        prep["total_recv"], prep["H"]).contiguous()
    return out, grad_fc2_out_sorted
