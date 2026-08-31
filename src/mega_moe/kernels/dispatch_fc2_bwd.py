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
from .shmem_transport import mte_put
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import sub_vec_id

from .common import ncore, all_gather_list, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K


def _dispatch_static_maps(saved):
    """Build the dy-independent expert-major dispatch maps once, cached on `saved`.

    The producer pushes grad buckets in (dst, expert) order and the consumer reads
    peer_mem expert-major contiguously, so every map here is expert-major (mirror
    of forward dispatch_fc1), not the rank-major sort_idxs layout."""
    cache = saved.get("_dispatch_cache")
    if cache is not None:
        return cache
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
    dtype: tl.constexpr,
):
    """fc2 input-grad GEMM one tile. A = peer_mem[m_off, H] CONTIGUOUS (no gather),
    B = fc2[expert][H, ffn], out = grad_swiglu[m_off, ffn]."""
    if m_size > 0:
        om = tl.arange(0, BLOCK_M); on_ = tl.arange(0, BLOCK_N); ok = tl.arange(0, BLOCK_K)
        m_offs = m_off + om; m_mask = om < m_size
        n_offs = n_tile * BLOCK_N + on_; n_mask = n_offs < N
        wb = expert_id.to(tl.int64) * stride_we
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
    """Per-(dst,expert,tile) MTE put + fence + signal_op SET. Adapted from forward
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
            mte_put(
                peer_mem_ptr + (task_dst_start + tile_start) * H,
                gco_ptr + (task_start + tile_start) * stride_gm,
                tile_count * H,
                dst_rank,
                BLOCK_ELEMENTS=8192,
            )
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
    MAX_BWD_TILES: tl.constexpr, dtype: tl.constexpr,
):
    """Consume merged expert M windows as their source tiles become ready. Adapted
    from forward _triton_grouped_gemm_expert_n_merged_tiles_wait
    (dispatch_fc1.py:539-629). GEMM ownership is keyed by (expert, n_tile); source
    rank participates only in the readiness dependency, so source fragments no
    longer force independent full-weight scans. waitValue = signal_epoch (SET mode,
    ONE value per epoch). Reuses the contiguous-read _fc2_bwd_gemm_one_mn_tile
    (stride_im=H, stride_ik=1).

    BLOCK_M is the GEMM tile height AND the merged readiness window; TILE_M is
    the producer's push/signal tile height (64, matches the signal-slot layout).
    A BLOCK_M=128 window simply waits every 64-row source tile overlapping it
    before running one double-height GEMM — the readiness protocol itself is
    unchanged, which decouples the GEMM L0A fill (a-tile [64,BK] bf16 = half of
    64KB L0A) from the transport tile granularity."""
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tasks = EXPERTS_PER_RANK * num_n_tiles
    for task_id in range(pid, num_tasks, ncore):
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
                    BLOCK_M, BLOCK_N, BLOCK_K, dtype)


@triton.jit(do_not_specialize=["signal_epoch"])
def kernel_dispatch_fc2_bwd_tile_signal(
    gco_ptr, peer_mem_ptr, signal_mem_ptr,
    send_bucket_starts_ptr, send_counts_re_ptr, send_bucket_dst_starts_ptr,
    H: tl.constexpr, stride_gm,
    fc2_ptr, output_ptr,
    recv_per_expert_ptr, recv_expert_offs_ptr, recv_counts_re_ptr,
    N, K, stride_im, stride_ik, stride_we, stride_wk, stride_wn, stride_om, stride_on,
    signal_epoch,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    PUSH_BLOCK_M: tl.constexpr,
    WORLD_SIZE: tl.constexpr, EXPERTS_PER_RANK: tl.constexpr,
    MAX_BWD_TILES: tl.constexpr, LOCAL_RANK: tl.constexpr,
    BLOCK_H_PUSH: tl.constexpr,
):
    """C2 fused per-tile dispatch + fc2 input-grad. The Vector side pushes
    source-local BLOCK_M tiles + signal_op SET; the Cube side merged-window
    dl.wait/consume_token + contiguous GEMM. They run concurrently on every AI
    core (all-core pipeline), fenced per source-tile — no barrier_all. Mirrors the
    verified ALL_CORE_PIPELINE + tile-readiness path of forward
    _kernel_dispatch_fc1. sub_vec_id()==0 gates the push (a mixed kernel has two
    vector sub-cores that would otherwise duplicate the putmem)."""
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
            WORLD_SIZE, EXPERTS_PER_RANK, MAX_BWD_TILES, dtype)


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


def _launch_dispatch_fc2_bwd_tile_signal(prep, peer_mem, out, saved):
    import shmem as ash
    W = saved["world_size"]
    EPR = prep["E"]
    MAX_BWD_TILES = prep["max_bwd_tiles"]
    fc2 = prep["fc2"]; H = prep["H"]; N = prep["N"]; K = prep["K"]
    # Lazy-alloc one SET slot per (source, expert, tile). Slot layout must match
    # producer/consumer: (rank*EPR+expert)*MAX_BWD_TILES+tile, with rank as the
    # source id. 16 int32 elements per slot (= 64 bytes) mirrors the forward
    # workspace's signal slot granularity.
    signal_mem = saved.get("_bwd_tile_signal_mem")
    if signal_mem is None:
        signal_mem = ash.aclshmem_create_tensor(
            [W * EPR * MAX_BWD_TILES * 16],
            dtype=torch.int32,
            device_id=saved["ep_rank"])
        signal_mem.zero_()
        saved["_bwd_tile_signal_mem"] = signal_mem
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
        fc2, out,
        prep["recv_per_expert"], prep["recv_expert_offs"], prep["recv_counts_re"],
        N, K, H, 1, fc2.stride(0), fc2.stride(1), fc2.stride(2), N, 1,
        signal_epoch,
        BLOCK_M=gemm_bm, BLOCK_N=gemm_bn, BLOCK_K=gemm_bk, PUSH_BLOCK_M=64,
        WORLD_SIZE=W, EXPERTS_PER_RANK=EPR, MAX_BWD_TILES=MAX_BWD_TILES,
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
