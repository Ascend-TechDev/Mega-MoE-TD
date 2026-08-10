# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Torch-NPU grouped-GEMM + HCCL forward performance baseline.

This is deliberately separate from :mod:`tests._moe_baselines`: the latter is
an independent correctness reference, while this module models the optimized
Torch-NPU performance alternative measured by the layer benchmark.  Every
shape comes from the explicitly supplied :class:`config.CaseSpec`; there is no
module-global active case.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:  # pragma: no cover - distributed Ascend jobs require torch-npu
    torch_npu = None

from config import CaseSpec


ACTIVATION_DTYPE = torch.bfloat16
ROUTING_TRANSPORT_DTYPE = torch.float32


class GroupedForwardBaseline:
    """One immutable case's grouped-GEMM + HCCL performance baseline."""

    def __init__(self, spec: CaseSpec, ep_group):
        if spec.direction != "forward" or "performance" not in spec.tags:
            raise ValueError(f"grouped forward baseline received {spec.case_id}")
        self.spec = spec.validate()
        self.ep_group = ep_group
        self.experts_per_rank = self.spec.experts_per_rank

    @staticmethod
    def _grouped_matmul(inputs, weight_kn, group_list):
        if torch_npu is None:
            raise RuntimeError("grouped forward baseline requires torch_npu")
        if inputs.shape[0] == 0:
            return torch.empty(
                (0, weight_kn.shape[-1]),
                dtype=ACTIVATION_DTYPE,
                device=inputs.device,
            )
        return torch_npu.npu_grouped_matmul(
            [inputs],
            [weight_kn],
            group_list=group_list,
            split_item=3,
            group_type=0,
            group_list_type=0,
            output_dtype=ACTIVATION_DTYPE,
        )[0]

    @torch.no_grad()
    def preprocess(self, hidden_states, selected_experts, routing_weights):
        """Stable post-router filtering/sort plus HCCL count exchange."""
        tokens_per_rank = hidden_states.shape[0]
        world_size = dist.get_world_size(group=self.ep_group)
        flat_expert = selected_experts.reshape(-1).long()
        flat_routing = routing_weights.reshape(-1)
        valid_mask = (flat_expert >= 0) & (flat_expert < self.spec.num_experts)
        valid_expert = flat_expert[valid_mask]
        valid_routing = flat_routing[valid_mask]
        route_token = torch.arange(
            tokens_per_rank, device=hidden_states.device
        ).repeat_interleave(self.spec.topk)
        valid_token = route_token[valid_mask]

        # Global-expert order is destination-rank-major then local-expert-major.
        sort_idx = torch.argsort(valid_expert.to(torch.float32), stable=True)
        expert_send = valid_expert[sort_idx].to(torch.int32).contiguous()
        token_send = hidden_states[valid_token[sort_idx]].contiguous()
        routing_send_fp32 = valid_routing[sort_idx].contiguous()
        destination = (
            expert_send.long() // self.experts_per_rank
        ).to(torch.int32)
        send_counts = torch.bincount(
            destination, minlength=world_size
        ).to(torch.int32)
        recv_counts = torch.empty(
            (world_size,), dtype=torch.int32, device=hidden_states.device
        )
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        return {
            "tokens_per_rank": tokens_per_rank,
            "valid_mask": valid_mask,
            "sort_idx": sort_idx,
            "token_send": token_send,
            "routing_send_fp32": routing_send_fp32,
            "expert_send": expert_send,
            "send_splits": send_counts.cpu().tolist(),
            "recv_splits": recv_counts.cpu().tolist(),
            "num_send": int(send_counts.sum().item()),
            "num_recv": int(recv_counts.sum().item()),
        }

    @torch.no_grad()
    def dispatch_fc1(self, state, torch_w1_kn):
        """HCCL payload dispatch, local expert grouping, and packed FC1."""
        device = state["token_send"].device
        recv_rows = state["num_recv"]
        token_recv = torch.empty(
            (recv_rows, self.spec.hidden),
            dtype=ACTIVATION_DTYPE,
            device=device,
        )
        routing_recv = torch.empty(
            (recv_rows,), dtype=ROUTING_TRANSPORT_DTYPE, device=device
        )
        expert_recv = torch.empty(
            (recv_rows,), dtype=torch.int32, device=device
        )
        routing_send = state["routing_send_fp32"].to(ROUTING_TRANSPORT_DTYPE)
        dist.all_to_all_single(
            token_recv,
            state["token_send"],
            output_split_sizes=state["recv_splits"],
            input_split_sizes=state["send_splits"],
            group=self.ep_group,
        )
        dist.all_to_all_single(
            routing_recv,
            routing_send,
            output_split_sizes=state["recv_splits"],
            input_split_sizes=state["send_splits"],
            group=self.ep_group,
        )
        dist.all_to_all_single(
            expert_recv,
            state["expert_send"],
            output_split_sizes=state["recv_splits"],
            input_split_sizes=state["send_splits"],
            group=self.ep_group,
        )

        local_expert = expert_recv.remainder(self.experts_per_rank)
        local_sort = torch.argsort(local_expert.to(torch.float32), stable=True)
        token_grouped = token_recv[local_sort].contiguous()
        routing_grouped = routing_recv[local_sort].contiguous()
        expert_counts = torch.bincount(
            local_expert.long(), minlength=self.experts_per_rank
        )
        group_list = torch.cumsum(expert_counts, dim=0).to(torch.int64)
        state.update(
            {
                "local_sort": local_sort,
                "routing_grouped": routing_grouped,
                "group_list": group_list,
                "fc1_output": self._grouped_matmul(
                    token_grouped, torch_w1_kn, group_list
                ),
            }
        )
        return state

    @staticmethod
    @torch.no_grad()
    def weighted_swiglu(state):
        gate, up = state["fc1_output"].chunk(2, dim=-1)
        return (
            F.silu(gate.float())
            * up.float()
            * state["routing_grouped"].float().unsqueeze(-1)
        ).to(ACTIVATION_DTYPE)

    def _restore_routes_and_reduce(
        self, back_sorted, sort_idx, valid_mask, tokens_per_rank
    ):
        inverse_sort = torch.argsort(sort_idx)
        valid_rows = back_sorted[inverse_sort]
        if valid_rows.shape[0] == tokens_per_rank * self.spec.topk:
            combined = valid_rows.view(
                tokens_per_rank, self.spec.topk, self.spec.hidden
            )
        else:
            combined = torch.zeros(
                (tokens_per_rank * self.spec.topk, self.spec.hidden),
                dtype=ACTIVATION_DTYPE,
                device=back_sorted.device,
            )
            combined[valid_mask] = valid_rows
            combined = combined.view(
                tokens_per_rank, self.spec.topk, self.spec.hidden
            )

        # Match production: fixed route-slot order, FP32 accumulation, then BF16.
        output_fp32 = torch.zeros(
            (tokens_per_rank, self.spec.hidden),
            dtype=torch.float32,
            device=back_sorted.device,
        )
        for route_slot in range(self.spec.topk):
            output_fp32 += combined[:, route_slot].float()
        return output_fp32.to(ACTIVATION_DTYPE)

    @torch.no_grad()
    def fc2_combine(self, state, weighted_activation, torch_w2_kn):
        """FC2, reverse HCCL A2A, and fixed-order FP32 top-k reduction."""
        fc2_grouped = self._grouped_matmul(
            weighted_activation, torch_w2_kn, state["group_list"]
        )
        arrival_order = torch.argsort(state["local_sort"])
        fc2_arrival = fc2_grouped[arrival_order].contiguous()
        back_sorted = torch.empty(
            (state["num_send"], self.spec.hidden),
            dtype=ACTIVATION_DTYPE,
            device=weighted_activation.device,
        )
        dist.all_to_all_single(
            back_sorted,
            fc2_arrival,
            output_split_sizes=state["send_splits"],
            input_split_sizes=state["recv_splits"],
            group=self.ep_group,
        )
        return self._restore_routes_and_reduce(
            back_sorted,
            state["sort_idx"],
            state["valid_mask"],
            state["tokens_per_rank"],
        )

    @torch.no_grad()
    def full_post_routing(
        self,
        hidden_states,
        selected_experts,
        routing_weights,
        torch_w1_kn,
        torch_w2_kn,
    ):
        state = self.preprocess(
            hidden_states, selected_experts, routing_weights
        )
        state = self.dispatch_fc1(state, torch_w1_kn)
        weighted_activation = self.weighted_swiglu(state)
        return self.fc2_combine(state, weighted_activation, torch_w2_kn)

    @staticmethod
    def _arrival_to_grouped_from_plan(routing_plan):
        counts = routing_plan.receive_counts_by_source_expert.to(torch.int64)
        source_prefix = torch.cumsum(counts, dim=0) - counts
        starts = (
            routing_plan.received_expert_offsets[:-1].to(torch.int64).unsqueeze(0)
            + source_prefix
        )
        flat_counts = counts.reshape(-1)
        flat_starts = starts.reshape(-1)
        segment_offsets = torch.cumsum(flat_counts, dim=0) - flat_counts
        num_rows = routing_plan.num_received_routes
        repeated_starts = torch.repeat_interleave(
            flat_starts, flat_counts, output_size=num_rows
        )
        repeated_offsets = torch.repeat_interleave(
            segment_offsets, flat_counts, output_size=num_rows
        )
        within_segment = (
            torch.arange(num_rows, dtype=torch.int64, device=counts.device)
            - repeated_offsets
        )
        return repeated_starts + within_segment

    @torch.no_grad()
    def fc2_combine_from_dispatch(
        self, weighted_activation, torch_w2_kn, dispatch_result, tokens_per_rank
    ):
        """Grouped-GEMM + HCCL FC2+combine on the candidate stage input."""
        routing_plan = dispatch_result.routing_plan
        group_list = torch.cumsum(
            routing_plan.received_routes_per_expert.to(torch.int64), dim=0
        )
        fc2_grouped = self._grouped_matmul(
            weighted_activation, torch_w2_kn, group_list
        )
        arrival_to_grouped = self._arrival_to_grouped_from_plan(routing_plan)
        fc2_arrival = fc2_grouped[arrival_to_grouped].contiguous()
        recv_splits = (
            routing_plan.receive_counts_by_source_expert
            .sum(dim=1, dtype=torch.int32)
            .cpu()
            .tolist()
        )
        send_splits = (
            routing_plan.send_counts_by_rank_expert
            .sum(dim=1, dtype=torch.int32)
            .cpu()
            .tolist()
        )
        back_sorted = torch.empty(
            (routing_plan.num_sent_routes, self.spec.hidden),
            dtype=ACTIVATION_DTYPE,
            device=weighted_activation.device,
        )
        dist.all_to_all_single(
            back_sorted,
            fc2_arrival,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.ep_group,
        )
        return self._restore_routes_and_reduce(
            back_sorted,
            routing_plan.stable_sort_indices,
            routing_plan.valid_route_mask,
            tokens_per_rank,
        )


__all__ = ["GroupedForwardBaseline"]
