# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Differentiable torch + HCCL EP-MoE forward, reused by the fused backward.

This is *not* a legacy reference: ``MegaMoEBackwardFunction.forward`` runs this
module's :func:`moe_forward` to produce the differentiable output and the
``saved`` intermediates that the 5-op triton backward consumes. The hand-written
torch backward golden (``tests/_goldens/backward.py``) reuses the same grouped
matmul primitives and :func:`moe_forward` so that the golden and the production
autograd path share one source of truth.

Layout invariants (routing weights applied in SwiGLU; combine reduce is a plain
sum over topk — no token dropping, matching the 06 tutorial):

    dispatch(home->expert) -> sort by local expert
    -> fc1(gate=H@fc1_1^T, up=H@fc1_2^T) -> SwiGLU(silu(gate)*up * scale)
    -> fc2(swiglu@fc2^T) -> reverse-A2A(expert->home) -> view(B,topk,H).sum(1)
"""

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

# NPU bf16 torch.matmul is M-shape-dependent: rows not a multiple of the native
# tile (256) select a different Cube kernel that re-rounds by ~1 ULP. Padding the
# token (M) dimension of every grouped matmul to a multiple of 256 makes the
# golden bit-stable and match the triton kernel's BLOCK_M=64 tiling (see 06).
GOLDEN_MATMUL_M_TILE = 256

# Tile width used by the metadata builder (ported from the 06 tutorial).
BLOCK_SIZE_M = 64


# ----------------------------------------------------------------------------
# Grouped matmul primitives (differentiable, M-padded to 256)
# ----------------------------------------------------------------------------

def _pad_m(slc, cnt):
    """Pad the row (M) dim of a [cnt, *] slice up to a multiple of GOLDEN_MATMUL_M_TILE."""
    pad = (-cnt) % GOLDEN_MATMUL_M_TILE
    if pad:
        slc = torch.nn.functional.pad(slc, (0, 0) * (slc.dim() - 1) + (0, pad))
    return slc


def grouped_matmul(a, weight, expert_counts, transpose=True):
    """Per-expert matmul over the token (M) dim, which is sorted/grouped by expert.

    a       : [M, K]   (rows sorted by local expert, contiguous per expert)
    weight  : [E, N, K] (transpose=True  -> out[m,n] = sum_k a[m,k] * weight[e,n,k]  => a @ weight[e].T, out [M, N])
                          (transpose=False -> out[m,n] = sum_k a[m,k] * weight[e,k,n]  => a @ weight[e],   out [M, N])
    expert_counts : [E] int

    Each expert slice is M-padded to 256 before the matmul (see GOLDEN_MATMUL_M_TILE).
    """
    out_parts = []
    start = 0
    for e in range(weight.shape[0]):
        cnt = int(expert_counts[e].item())
        if cnt > 0:
            slc = a[start:start + cnt]
            slc = _pad_m(slc, cnt)
            w = weight[e]
            res = (slc @ w.T) if transpose else (slc @ w)
            out_parts.append(res[:cnt])
            start += cnt
    if not out_parts:
        return torch.zeros(a.shape[0], weight.shape[1] if transpose else weight.shape[2],
                           dtype=a.dtype, device=a.device)
    return torch.cat(out_parts, dim=0)


def grouped_transposed_matmul(grad_out, orig_in, expert_counts):
    """Weight-grad grouped matmul: grad_weight[e] = grad_out[e]^T @ orig_in[e].

    grad_out : [M, N]   (rows sorted by expert)
    orig_in  : [M, K]   (rows sorted by expert)
    -> grad_weight [E, N, K]

    M-padded to 256 per expert (zero rows contribute nothing, but keep the same
    Cube kernel path as the input-grad matmuls for numerical consistency).
    """
    E = weight_E_from_counts(expert_counts)
    N = grad_out.shape[1]
    K = orig_in.shape[1]
    grad_weight = torch.zeros(E, N, K, dtype=grad_out.dtype, device=grad_out.device)
    start = 0
    for e in range(E):
        cnt = int(expert_counts[e].item())
        if cnt > 0:
            g = _pad_m(grad_out[start:start + cnt], cnt)      # [cnt_pad, N]
            o = _pad_m(orig_in[start:start + cnt], cnt)       # [cnt_pad, K]
            grad_weight[e] = g.T @ o                          # [N, K]
            start += cnt
    return grad_weight


def weight_E_from_counts(expert_counts):
    return int(expert_counts.shape[0])


# ----------------------------------------------------------------------------
# Metadata builder (ported from 06, BLOCK_SIZE_M=64)
# ----------------------------------------------------------------------------

def prepare_moe_metadata(expert_counts):
    """Build tile-based GroupGEMM metadata from per-expert token counts."""
    E = expert_counts.shape[0]
    expert_counts = expert_counts.to(torch.int64)
    tiles_per_expert = (expert_counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_total = int(tiles_per_expert.sum().item())
    meta_expert_ids = torch.zeros(num_total, dtype=torch.int32)
    meta_split_cum = torch.zeros(num_total, dtype=torch.int32)
    meta_tile_num = torch.zeros(num_total, dtype=torch.int32)
    meta_tile_num_cum = torch.zeros(num_total, dtype=torch.int32)
    idx = 0; tile_acc = 0; token_acc = 0
    for e in range(E):
        nt = int(tiles_per_expert[e].item())
        for t in range(nt):
            meta_expert_ids[idx] = e
            meta_split_cum[idx] = token_acc
            meta_tile_num[idx] = t
            meta_tile_num_cum[idx] = tile_acc
            idx += 1
        tile_acc += nt
        token_acc += int(expert_counts[e].item())
    num_tiles_total = torch.tensor([num_total], dtype=torch.int32)
    split_size_cum_per_expert = torch.zeros(E + 1, dtype=torch.int32)
    split_size_cum_per_expert[1:] = expert_counts.cumsum(0)
    return (split_size_cum_per_expert, meta_expert_ids, meta_split_cum,
            meta_tile_num, meta_tile_num_cum, num_tiles_total)


# ----------------------------------------------------------------------------
# Autograd-aware all-to-all
# ----------------------------------------------------------------------------

class AllToAll(torch.autograd.Function):
    """Autograd-aware all_to_all_single.

    dist.all_to_all_single is not autograd-aware on this torch_npu build, so we
    wrap it. The all-to-all is its own transpose: the backward of
    all_to_all(out_split=A, in_split=B) is all_to_all(out_split=B, in_split=A).
    This is exactly the combine<->dispatch direction swap used in the hand bwd.
    """

    @staticmethod
    def forward(ctx, input, output_split_sizes, input_split_sizes, group):
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group
        feat = input.shape[1:]
        out_rows = int(sum(output_split_sizes))
        out = input.new_empty((out_rows, *feat))
        dist.all_to_all_single(out, input, output_split_sizes=output_split_sizes,
                               input_split_sizes=input_split_sizes, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # swap split roles: grad_input = all_to_all(grad_output, out_split=B, in_split=A)
        feat = grad_output.shape[1:]
        in_rows = int(sum(ctx.input_split_sizes))
        grad_input = grad_output.new_empty((in_rows, *feat))
        dist.all_to_all_single(grad_input, grad_output,
                               output_split_sizes=ctx.input_split_sizes,
                               input_split_sizes=ctx.output_split_sizes,
                               group=ctx.group)
        return grad_input, None, None, None


def _a2a(input, output_split_sizes, input_split_sizes, group):
    return AllToAll.apply(input, output_split_sizes, input_split_sizes, group)


# ----------------------------------------------------------------------------
# Forward  (differentiable; returns output + saved dict for the hand bwd)
# ----------------------------------------------------------------------------

def moe_forward(hidden_states, routing_weights, selected_experts,
                fc1_1, fc1_2, fc2, ep_group, topk, return_saved=False):
    """EP-MoE forward. routing_weights [B,topk], selected_experts [B,topk] (global ids).

    Mirrors GPU torch_moe_fwd / 06 build_moe_fwd_inputs. No token dropping.
    When return_saved=True, also returns a dict of all intermediates needed by
    the torch backward golden (detached under torch.no_grad() by the caller).
    """
    dtype = hidden_states.dtype
    device = hidden_states.device
    world_size = dist.get_world_size(ep_group)
    batch_size, hidden_dim = hidden_states.shape
    experts_per_rank = fc1_1.shape[0]
    num_experts = experts_per_rank * world_size
    ffn_dim = fc1_1.shape[1]
    ep_rank = dist.get_rank(ep_group)

    # ---- Dispatch: expand + sort by dest rank ----
    expanded_hidden = hidden_states.repeat_interleave(topk, dim=0)        # [B*topk, H]
    flat_weights = routing_weights.reshape(-1)                            # [B*topk]
    flat_indices = selected_experts.reshape(-1).to(torch.int64)           # [B*topk]
    expert_ranks = flat_indices // experts_per_rank
    local_experts = flat_indices % experts_per_rank

    send_counts = torch.bincount(expert_ranks, minlength=world_size).to(torch.int64)
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    splits_send_list = send_counts.tolist()
    splits_recv_list = recv_counts.tolist()
    total_send = int(send_counts.sum().item())
    total_recv = int(recv_counts.sum().item())

    sort_idxs = torch.argsort(expert_ranks, stable=True)
    sorted_hidden = expanded_hidden[sort_idxs]
    sorted_weights = flat_weights[sort_idxs].to(dtype)
    sorted_local_experts = local_experts[sort_idxs]

    tokens_recv = _a2a(sorted_hidden, splits_recv_list, splits_send_list, ep_group)
    weights_recv = _a2a(sorted_weights, splits_recv_list, splits_send_list, ep_group)
    expert_ids_recv = torch.empty((total_recv,), dtype=torch.int64, device=device)
    dist.all_to_all_single(expert_ids_recv, sorted_local_experts.to(torch.int64),
                           output_split_sizes=splits_recv_list,
                           input_split_sizes=splits_send_list, group=ep_group)

    # ---- Sort received tokens by local expert ----
    local_expert_ids = expert_ids_recv % experts_per_rank
    local_sort_idxs = torch.argsort(local_expert_ids, stable=True)
    recv_hidden_sorted = tokens_recv[local_sort_idxs]
    recv_weights_sorted = weights_recv[local_sort_idxs]
    expert_counts = torch.bincount(local_expert_ids, minlength=experts_per_rank).to(torch.int32)

    meta = prepare_moe_metadata(expert_counts)
    (split_size_cum_per_expert, meta_expert_ids, meta_split_cum,
     meta_tile_num, meta_tile_num_cum, num_tiles_total) = meta

    # ---- fc1 + SwiGLU + fc2 ----
    fc1_combined = torch.cat([fc1_1, fc1_2], dim=1)                      # [E, 2*ffn, H]
    fc1_out = grouped_matmul(recv_hidden_sorted, fc1_combined, expert_counts, transpose=True)
    gate, up = fc1_out.chunk(2, dim=-1)                                  # each [M, ffn]
    swiglu_out = (torch.nn.functional.silu(gate.float()) * up.float())
    swiglu_out_weighted = (swiglu_out * recv_weights_sorted.float().unsqueeze(-1)).to(dtype)
    fc2_out = grouped_matmul(swiglu_out_weighted, fc2, expert_counts, transpose=True)  # [M, H]

    # ---- Combine: reverse-A2A + topk plain sum ----
    inv_local = torch.argsort(local_sort_idxs)
    fc2_out_unsorted = fc2_out[inv_local]                                # back to arrival order
    combined_out_flat = _a2a(fc2_out_unsorted, splits_send_list, splits_recv_list, ep_group)
    inv_sort = torch.argsort(sort_idxs)
    combined_full = combined_out_flat[inv_sort]                          # [B*topk, H]
    output = combined_full.view(batch_size, topk, hidden_dim).sum(dim=1)

    if not return_saved:
        return output

    saved = dict(
        output=output, dy=None,
        # dims / group
        batch_size=batch_size, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
        num_experts=num_experts, experts_per_rank=experts_per_rank, topk=topk,
        world_size=world_size, ep_rank=ep_rank, ep_group=ep_group,
        # dispatch / sort invariants
        sort_idxs=sort_idxs, local_sort_idxs=local_sort_idxs,
        inv_local=inv_local, inv_sort=inv_sort,
        splits_send_list=splits_send_list, splits_recv_list=splits_recv_list,
        total_send=total_send, total_recv=total_recv, M=int(expert_counts.sum().item()),
        expert_counts=expert_counts,
        split_size_cum_per_expert=split_size_cum_per_expert,
        meta_expert_ids=meta_expert_ids, meta_split_cum=meta_split_cum,
        meta_tile_num=meta_tile_num, meta_tile_num_cum=meta_tile_num_cum,
        num_tiles_total=num_tiles_total,
        # fwd activations
        recv_hidden_sorted=recv_hidden_sorted, fc1_output=fc1_out,
        gate=gate, up=up, swiglu_out_weighted=swiglu_out_weighted,
        recv_weights_sorted=recv_weights_sorted, fc2_out=fc2_out,
        # weights
        fc1_1=fc1_1, fc1_2=fc1_2, fc2=fc2, fc1_combined=fc1_combined,
    )
    return output, saved


__all__ = [
    "AllToAll",
    "BLOCK_SIZE_M",
    "GOLDEN_MATMUL_M_TILE",
    "grouped_matmul",
    "grouped_transposed_matmul",
    "moe_forward",
    "prepare_moe_metadata",
    "weight_E_from_counts",
]
