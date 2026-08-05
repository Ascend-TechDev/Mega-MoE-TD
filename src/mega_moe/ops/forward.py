# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standalone BF16 Ascend Mega-MoE post-routing forward.

The public boundary starts with selected experts and FP32 routing weights;
router matmul, softmax, and top-k selection are intentionally excluded.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed

from ..config import MoEForwardConfig
from ..kernels.dispatch_fc1 import _kernel_dispatch_fc1
from ..kernels.fc2_combine import (
    build_route_to_send,
    launch_fc2_combine,
    prepare_fc2_combine_metadata,
)
from ..kernels.weighted_swiglu import weighted_swiglu_forward
from ..runtime.routing import MoERoutingPlan, build_routing_plan
from ..runtime.workspace import create_moe_forward_context


@dataclass
class DispatchFC1Result:
    """Outputs and route identity produced by the dispatch plus FC1 stage."""

    routing_plan: MoERoutingPlan
    fc1_output: torch.Tensor
    dispatched_tokens: torch.Tensor
    received_routing_weights: torch.Tensor
    send_route_indices: torch.Tensor


class FusedMoEForward(torch.nn.Module):
    """Optimized fused EP All-to-All + grouped GEMM MoE op for Ascend NPU.

    Dispatch and FC1 overlap through readiness signals.  The default path uses
    expert-major merged-M windows; retained alternatives cover Qwen,
    DeepSeek/DSV4, and Kimi shapes until same-snapshot A/B data proves them
    redundant. ``dispatch_fc1_weighted_swiglu`` extends the supported path
    through weighted SwiGLU. FC2, route transport, route restoration, and
    top-k reduction then run in the dedicated combine kernels.
    """

    def __init__(
        self,
        ep_group: Optional[torch.distributed.ProcessGroup],
        *,
        max_tokens_per_rank: int,
        hidden_size: int,
        top_k: int,
        num_experts: int,
        config: Optional[MoEForwardConfig] = None,
    ):
        """Create one BF16-only post-routing MoE forward instance.

        ``ep_group`` must match the complete ACLSHMEM world.  The optional
        config contains only stage-specific capacity, schedule, and tiling
        controls; activation and expert-weight dtypes are fixed to BF16.
        """
        super().__init__()
        if max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if config is not None and not isinstance(config, MoEForwardConfig):
            raise TypeError("config must be a MoEForwardConfig instance or None")

        self.ep_group = ep_group
        if ep_group is not None:
            self.rank = ep_group.rank()
            self.world_size = ep_group.size()
        else:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        if num_experts % self.world_size:
            raise ValueError("num_experts must be divisible by the EP world size")

        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.experts_per_rank = num_experts // self.world_size
        self.activation_dtype = torch.bfloat16
        self.config = config if config is not None else MoEForwardConfig()
        self.receive_capacity_factor = (
            self.config.resolved_receive_capacity_factor(self.world_size)
        )
        self.num_aicore_programs = self.config.num_aicore_programs
        self.dispatch_producer_cores = (
            self.config.resolved_dispatch_producer_cores()
        )
        self.dispatch_readiness = self.config.dispatch_readiness
        self.dispatch_fc1_schedule = self.config.dispatch_fc1_schedule
        self.activation = self.config.activation
        self.situ_beta = self.config.situ_beta
        self.situ_linear_beta = self.config.situ_linear_beta

        # Tile SET slots and expert ADD counters use disjoint signal regions.
        self._tile_signal_epoch = 1
        self._expert_signal_epoch = 1
        self.context = create_moe_forward_context(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            top_k=top_k,
            num_experts=num_experts,
            rank=self.rank,
            world_size=self.world_size,
            receive_capacity_factor=self.receive_capacity_factor,
            dispatch_fc1_block_size_m=self.config.dispatch_fc1_block_size_m,
        )

        # FC2/combine workspaces are ordinary tensors and remain lazy so
        # dispatch-only stage tests do not pay their memory cost.
        self._combine_fc2_buf = None
        self._route_to_send = None
        self._fc2_tile_expert = None
        self._fc2_tile_row_start = None
        self._fc2_tile_row_count = None
        self._reverse_tile_rank = None
        self._reverse_tile_src_start = None
        self._reverse_tile_dst_start = None
        self._reverse_tile_row_count = None
        self._max_fc2_tile_slots = 0
        self._max_reverse_tile_slots = 0
        self._dispatch_send_staging = None
        self._dispatch_route_staging = None
        self._routing_weights_keepalive = None

        # All ranks must observe zeroed symmetric buffers before first use.
        torch.npu.synchronize()
        torch.distributed.barrier(group=self.ep_group)

    # ===================== buffer-management =====================
    def sync(self):
        torch.npu.synchronize()

    def finalize(self):
        self._dispatch_send_staging = None
        self._dispatch_route_staging = None
        self._routing_weights_keepalive = None
        self._combine_fc2_buf = None
        self._route_to_send = None
        self._fc2_tile_expert = None
        self._fc2_tile_row_start = None
        self._fc2_tile_row_count = None
        self._reverse_tile_rank = None
        self._reverse_tile_src_start = None
        self._reverse_tile_dst_start = None
        self._reverse_tile_row_count = None
        self.context.finalize()

    def _ensure_combine_buffers(self):
        """Allocate reusable local workspaces for FC2 and route restoration."""
        if self._combine_fc2_buf is not None:
            return

        max_recv = self.context.peer_mem.numel() // self.hidden_size
        max_send = self.max_tokens_per_rank * self.top_k
        block_m = self.config.fc2_combine_block_size_m
        max_m_tiles = (max_recv + block_m - 1) // block_m
        self._max_fc2_tile_slots = max_m_tiles + self.experts_per_rank
        if self.config.fc2_combine_transport == "direct_pull":
            max_transport_tiles = (max_send + block_m - 1) // block_m
        else:
            max_transport_tiles = max_m_tiles
        self._max_reverse_tile_slots = (
            max_transport_tiles + self.world_size * self.experts_per_rank
        )
        combine_rows = (
            max_send
            if self.config.fc2_combine_transport == "direct_pull"
            else max_recv
        )
        device = self.context.peer_mem.device
        self._combine_fc2_buf = torch.empty(
            (combine_rows, self.hidden_size),
            dtype=self.activation_dtype,
            device=device,
        )
        self._route_to_send = torch.empty(
            max_send, dtype=torch.int32, device=device)

        def make_int_workspace(size):
            return torch.empty(size, dtype=torch.int32, device=device)

        self._fc2_tile_expert = make_int_workspace(self._max_fc2_tile_slots)
        self._fc2_tile_row_start = make_int_workspace(self._max_fc2_tile_slots)
        self._fc2_tile_row_count = make_int_workspace(self._max_fc2_tile_slots)
        self._reverse_tile_rank = make_int_workspace(self._max_reverse_tile_slots)
        self._reverse_tile_src_start = make_int_workspace(self._max_reverse_tile_slots)
        self._reverse_tile_dst_start = make_int_workspace(self._max_reverse_tile_slots)
        self._reverse_tile_row_count = make_int_workspace(self._max_reverse_tile_slots)

    def _validate_topk_indices(self, selected_experts: torch.Tensor):
        if selected_experts.ndim != 2:
            raise ValueError(
                "selected_experts must have shape [tokens, top_k], got "
                f"{tuple(selected_experts.shape)}"
            )
        if selected_experts.shape[1] != self.top_k:
            raise ValueError(
                f"selected_experts must have top_k={self.top_k}, got shape "
                f"{tuple(selected_experts.shape)}"
            )
        if selected_experts.dtype != torch.int32:
            raise TypeError(
                "selected_experts must use torch.int32, got "
                f"{selected_experts.dtype}"
            )
        if selected_experts.device != self.context.peer_mem.device:
            raise ValueError("selected_experts must be on the operator's NPU device")
        if not selected_experts.is_contiguous():
            raise ValueError("selected_experts must be contiguous")
        if selected_experts.shape[0] > self.max_tokens_per_rank:
            raise ValueError(
                f"input has {selected_experts.shape[0]} tokens, exceeding "
                f"max_tokens_per_rank={self.max_tokens_per_rank}"
            )

    def _validate_dispatch_inputs(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
    ):
        self._validate_topk_indices(selected_experts)
        if hidden_states.ndim != 2:
            raise ValueError(
                f"hidden_states must have shape [tokens, hidden], got {tuple(hidden_states.shape)}"
            )
        expected_shape = (selected_experts.shape[0], self.hidden_size)
        if hidden_states.shape != expected_shape:
            raise ValueError(
                f"hidden_states must have shape {expected_shape}, got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != self.activation_dtype:
            raise TypeError(f"hidden_states must use {self.activation_dtype}, got {hidden_states.dtype}")
        if hidden_states.device != selected_experts.device:
            raise ValueError(
                "hidden_states and selected_experts must be on the same device"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous")

    # ===================== routing metadata =========================
    def build_routing_plan(
        self,
        selected_experts: torch.Tensor,
    ) -> MoERoutingPlan:
        """Build stable dispatch/count metadata for one set of selected experts."""
        self._validate_topk_indices(selected_experts)
        return build_routing_plan(self.context, selected_experts)

    def _prepare_combine_metadata(
        self,
        dispatch_result: DispatchFC1Result,
    ) -> dict:
        """Build schedule-required FC2 metadata and the route inverse."""
        self._ensure_combine_buffers()
        plan = dispatch_result.routing_plan
        send_route_indices = dispatch_result.send_route_indices

        num_routes = plan.num_input_tokens * self.top_k
        if num_routes > self._route_to_send.numel():
            raise ValueError(
                f"route count {num_routes} exceeds configured capacity "
                f"{self._route_to_send.numel()}"
            )
        if send_route_indices.numel() != plan.num_sent_routes:
            raise ValueError(
                "send_route_indices length does not match the dispatch send count"
            )
        route_to_send = self._route_to_send[:num_routes]
        build_route_to_send(send_route_indices, route_to_send)

        block_m = self.config.fc2_combine_block_size_m
        m_tiles = (plan.num_received_routes + block_m - 1) // block_m
        num_fc2_slots = m_tiles + self.experts_per_rank
        direct_pull = self.config.fc2_combine_transport == "direct_pull"
        if direct_pull:
            # Sum(ceil(bucket_count / block_m)) is bounded by
            # ceil(num_sent / block_m) + num_global_experts - 1.
            sent_tiles = (plan.num_sent_routes + block_m - 1) // block_m
            num_reverse_slots = (
                sent_tiles + self.world_size * self.experts_per_rank
            )
        else:
            num_reverse_slots = (
                m_tiles + self.world_size * self.experts_per_rank
            )
        if num_fc2_slots > self._max_fc2_tile_slots:
            raise ValueError("FC2 tile metadata exceeds its configured workspace")
        if num_reverse_slots > self._max_reverse_tile_slots:
            raise ValueError("reverse-A2A tile metadata exceeds its configured workspace")

        prepare_fc2_combine_metadata(
            plan.receive_counts_by_source_expert,
            plan.received_expert_offsets,
            self.context.metadata_counts_mem,
            plan.send_bucket_starts,
            plan.send_bucket_receive_offsets,
            self._fc2_tile_expert,
            self._fc2_tile_row_start,
            self._fc2_tile_row_count,
            self._reverse_tile_rank,
            self._reverse_tile_src_start,
            self._reverse_tile_dst_start,
            self._reverse_tile_row_count,
            num_fc2_slots,
            num_reverse_slots,
            local_rank=self.rank,
            world_size=self.world_size,
            experts_per_rank=self.experts_per_rank,
            num_bins_pad=self.context.metadata_num_bins,
            block_m=block_m,
            direct_pull=direct_pull,
            build_fc2_tiles=(
                self.config.fc2_gemm_schedule == "tile_n_major"
            ),
        )
        return {
            "route_to_send": route_to_send,
            "num_fc2_slots": num_fc2_slots,
            "num_reverse_slots": num_reverse_slots,
        }

    # ===================== dispatch + FC1 ===========================
    def dispatch_fc1(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_plan: MoERoutingPlan,
        fc1_weight: torch.Tensor,
        fc1_output: Optional[torch.Tensor] = None,
        *,
        routing_weights: torch.Tensor,
        final_barrier: bool = True,
    ) -> DispatchFC1Result:
        """Dispatch routes and execute local-expert FC1 in one overlap kernel.

        FC1 weights must use contiguous ``[expert, K, N]`` model-load layout
        so the Cube B tile has contiguous N.  The kernel consumes a transposed
        logical view without a hot-path materialization.
        """
        hidden_size = self.hidden_size
        device = hidden_states.device
        self._validate_dispatch_inputs(hidden_states, selected_experts)
        if routing_plan.selected_experts is not selected_experts:
            raise ValueError(
                "routing_plan must be built from this exact selected_experts tensor"
            )
        if routing_plan.num_input_tokens != hidden_states.shape[0]:
            raise ValueError("routing plan token count does not match hidden_states")
        if fc1_weight.ndim != 3:
            raise ValueError("fc1_weight must be a 3D local-expert weight tensor")
        if fc1_weight.shape[0] != self.experts_per_rank:
            raise ValueError(
                f"fc1_weight has {fc1_weight.shape[0]} local experts, "
                f"expected {self.experts_per_rank}"
            )
        if fc1_weight.dtype != self.activation_dtype:
            raise TypeError(
                f"fc1_weight must use {self.activation_dtype}, got {fc1_weight.dtype}"
            )
        if fc1_weight.device != hidden_states.device:
            raise ValueError("fc1_weight and hidden_states must be on the same device")
        if fc1_weight.shape[1] != hidden_size:
            raise ValueError(
                "fc1_weight must use [experts_per_rank, hidden_size, output_size] "
                f"layout, got {tuple(fc1_weight.shape)}"
            )
        if fc1_weight.shape[2] <= 0:
            raise ValueError("fc1_weight output_size must be positive")
        if not fc1_weight.is_contiguous():
            raise ValueError("fc1_weight must be contiguous in [expert, K, N] layout")

        if routing_weights.shape != selected_experts.shape:
            raise ValueError(
                "routing_weights must match selected_experts shape [tokens, top_k]"
            )
        if routing_weights.dtype != torch.float32:
            raise TypeError(
                f"routing_weights must use torch.float32, got {routing_weights.dtype}"
            )
        if routing_weights.device != hidden_states.device:
            raise ValueError(
                "routing_weights and hidden_states must be on the same device"
            )
        if not routing_weights.is_contiguous():
            raise ValueError("routing_weights must be contiguous")

        route_indices = self.context.row_route_indices[: selected_experts.numel()]
        if routing_plan.num_sent_routes == selected_experts.numel():
            kept_route_indices = route_indices
        else:
            kept_route_indices = route_indices[routing_plan.valid_route_mask]
        send_route_indices = kept_route_indices[
            routing_plan.stable_sort_indices
        ].to(torch.int32).contiguous()

        num_received_routes = routing_plan.num_received_routes
        direct_bucket_schedule = self.dispatch_fc1_schedule in (
            "allcore_expert",
            "allcore_expert_mn",
            "allcore_expert_n",
        )
        if direct_bucket_schedule:
            staging_rows = selected_experts.numel()
            if (
                self._dispatch_send_staging is None
                or self._dispatch_send_staging.shape[0] < staging_rows
            ):
                self._dispatch_send_staging = torch.empty(
                    (staging_rows, hidden_size),
                    dtype=self.activation_dtype,
                    device=device,
                )
                self._dispatch_route_staging = torch.empty(
                    staging_rows, dtype=torch.float32, device=device
                )
            num_sent_routes = routing_plan.num_sent_routes
            torch.index_select(
                hidden_states,
                0,
                routing_plan.send_token_indices,
                out=self._dispatch_send_staging[:num_sent_routes],
            )
            torch.index_select(
                routing_weights.view(-1),
                0,
                send_route_indices,
                out=self._dispatch_route_staging[:num_sent_routes],
            )

        # Present a logical [expert, N, K] view without materializing a second
        # multi-GiB weight table.  N remains contiguous in the physical KN
        # model-load layout.
        weight_for_gemm = fc1_weight.transpose(-1, -2)

        _, output_size, reduction_size = weight_for_gemm.shape
        block_n = self.config.fc1_gemm_block_size_n
        block_k = self.config.fc1_gemm_block_size_k
        block_m = self.config.dispatch_fc1_block_size_m
        if output_size % block_n or reduction_size % block_k:
            raise ValueError(
                "FC1 block_n and block_k must divide the output and reduction dimensions"
            )

        if fc1_output is None:
            output = torch.empty(
                (num_received_routes, output_size),
                dtype=self.activation_dtype,
                device=device,
            )
        else:
            output = fc1_output
            if output.shape != (num_received_routes, output_size):
                raise ValueError(
                    f"fc1_output must have shape {(num_received_routes, output_size)}, "
                    f"got {tuple(output.shape)}"
                )
            if output.dtype != self.activation_dtype:
                raise TypeError(
                    f"fc1_output must use {self.activation_dtype}, got {output.dtype}"
                )
            if output.device != device:
                raise ValueError(
                    "fc1_output and hidden_states must be on the same device"
                )
            if not output.is_contiguous():
                raise ValueError("fc1_output must be contiguous")

        dispatched_tokens = self.context.peer_mem[
            :num_received_routes * hidden_size
        ].view(num_received_routes, hidden_size)
        tile_readiness = self.dispatch_readiness == "tile"
        signal_epoch = (
            self._tile_signal_epoch if tile_readiness else self._expert_signal_epoch
        )
        _kernel_dispatch_fc1[self.num_aicore_programs, 1, 1](
            hidden_states,
            self._dispatch_send_staging
            if direct_bucket_schedule
            else hidden_states,
            self.context.peer_mem,
            routing_weights,
            self._dispatch_route_staging
            if direct_bucket_schedule
            else routing_weights,
            self.context.routing_weight_mem,
            self.context.signal_mem,
            weight_for_gemm,
            output,
            routing_plan.send_token_indices,
            send_route_indices,
            routing_plan.send_bucket_receive_offsets,
            routing_plan.send_bucket_starts,
            routing_plan.send_counts_by_rank_expert,
            routing_plan.received_routes_per_expert,
            routing_plan.received_expert_offsets,
            routing_plan.receive_counts_by_source_expert,
            signal_epoch,
            hidden_size,
            output_size,
            reduction_size,
            hidden_states.stride(0),
            hidden_states.stride(1),
            weight_for_gemm.stride(0),
            weight_for_gemm.stride(1),
            weight_for_gemm.stride(2),
            output.stride(0),
            output.stride(1),
            N_DISPATCH_CORES=self.dispatch_producer_cores,
            NUM_CONSUMER_CORES=(
                self.num_aicore_programs - self.dispatch_producer_cores
            ),
            NUM_PROGRAM_CORES=self.num_aicore_programs,
            LOCAL_RANK=self.rank,
            WORLD_SIZE=self.world_size,
            EXPERTS_PER_RANK=self.experts_per_rank,
            MAX_SOURCE_TILES=self.context.max_source_tiles,
            TILE_READINESS=tile_readiness,
            COUNT_DERIVED_SCHEDULE=self.dispatch_fc1_schedule == "count",
            ALL_CORE_PIPELINE=self.dispatch_fc1_schedule
            in (
                "allcore",
                "allcore_expert",
                "allcore_expert_mn",
                "allcore_expert_n",
                "allcore_expert_n_tile",
            ),
            DIRECT_EXPERT_DISPATCH=direct_bucket_schedule,
            EXPERT_N_TILE_CONSUMER=(
                self.dispatch_fc1_schedule == "allcore_expert_n_tile"
            ),
            MN_TILE_FC1=self.dispatch_fc1_schedule == "allcore_expert_mn",
            N_TILE_FC1=self.dispatch_fc1_schedule == "allcore_expert_n",
            FINAL_BARRIER=final_barrier,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )
        if tile_readiness:
            self._tile_signal_epoch += 1
        else:
            self._expert_signal_epoch += 1

        received_routing_weights = self.context.routing_weight_mem[
            :num_received_routes
        ]
        # The Triton launch is asynchronous.  Retain the public FP32 send
        # tensor for this single-in-flight operation until the next dispatch.
        self._routing_weights_keepalive = routing_weights
        return DispatchFC1Result(
            routing_plan=routing_plan,
            fc1_output=output,
            dispatched_tokens=dispatched_tokens,
            received_routing_weights=received_routing_weights,
            send_route_indices=send_route_indices,
        )

    def weighted_swiglu(
        self,
        dispatch_result: DispatchFC1Result,
    ) -> torch.Tensor:
        """Apply the configured gated activation and route scaling in FP32.

        ``self.activation`` selects between SwiGLU (``"swiglu"``) and SiTU-GLU
        (``"situglu"``); ``self.situ_beta`` / ``self.situ_linear_beta`` configure
        the SiTU-GLU branch and are ignored for SwiGLU.
        """
        return weighted_swiglu_forward(
            dispatch_result.fc1_output,
            dispatch_result.received_routing_weights,
            self.num_aicore_programs,
            activation=self.activation,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )

    def dispatch_fc1_weighted_swiglu(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        gate_up_weight: torch.Tensor,
        *,
        final_dispatch_barrier: bool = True,
    ):
        """Build routing, run dispatch+FC1, then weighted SwiGLU."""
        self._validate_dispatch_inputs(hidden_states, selected_experts)
        if gate_up_weight.ndim != 3:
            raise ValueError(
                "gate_up_weight must have shape [experts_per_rank, hidden, 2F]"
            )
        if gate_up_weight.shape[0] != self.experts_per_rank:
            raise ValueError(
                f"gate_up_weight has {gate_up_weight.shape[0]} local experts, "
                f"expected {self.experts_per_rank}"
            )
        if gate_up_weight.shape[1] != self.hidden_size:
            raise ValueError(
                "gate_up_weight must use [experts_per_rank, hidden, 2F] layout "
                f"with hidden_size={self.hidden_size}, got {tuple(gate_up_weight.shape)}"
            )
        packed_output_size = gate_up_weight.shape[2]
        if packed_output_size <= 0 or packed_output_size % 2:
            raise ValueError(
                "gate_up_weight output dimension must be a positive even value (2F)"
            )
        if gate_up_weight.dtype != self.activation_dtype:
            raise TypeError(
                f"gate_up_weight must use {self.activation_dtype}, "
                f"got {gate_up_weight.dtype}"
            )
        if gate_up_weight.device != hidden_states.device:
            raise ValueError(
                "gate_up_weight and hidden_states must be on the same device"
            )
        if not gate_up_weight.is_contiguous():
            raise ValueError(
                "gate_up_weight must be contiguous and pre-packed before forward"
            )

        routing_plan = self.build_routing_plan(selected_experts)
        dispatch_result = self.dispatch_fc1(
            hidden_states,
            selected_experts,
            routing_plan,
            gate_up_weight,
            routing_weights=routing_weights,
            final_barrier=final_dispatch_barrier,
        )
        weighted_activation = self.weighted_swiglu(dispatch_result)
        return weighted_activation, dispatch_result

    # ===================== FC2 + combine ============================
    def fc2_combine(
        self,
        weighted_activation: torch.Tensor,
        down_weight: torch.Tensor,
        dispatch_result: DispatchFC1Result,
        combine_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run BF16 FC2, reverse all-to-all, route restore, and top-k sum."""
        if weighted_activation.ndim != 2:
            raise ValueError(
                "weighted_activation must have shape [received_routes, ffn_dim]"
            )
        num_received_routes, ffn_size = weighted_activation.shape
        plan = dispatch_result.routing_plan
        if num_received_routes != plan.num_received_routes:
            raise ValueError(
                f"weighted_activation has {num_received_routes} rows but the "
                f"routing plan requires {plan.num_received_routes}"
            )
        if weighted_activation.dtype != self.activation_dtype:
            raise TypeError(
                f"weighted_activation must use {self.activation_dtype}, "
                f"got {weighted_activation.dtype}"
            )
        if weighted_activation.device != self.context.peer_mem.device:
            raise ValueError(
                "weighted_activation must be on the operator's NPU device"
            )
        if not weighted_activation.is_contiguous():
            raise ValueError("weighted_activation must be contiguous")

        expected_weight_shape = (
            self.experts_per_rank,
            self.hidden_size,
            ffn_size,
        )
        if down_weight.shape != expected_weight_shape:
            raise ValueError(
                f"down_weight must have shape {expected_weight_shape}, "
                f"got {tuple(down_weight.shape)}"
            )
        if down_weight.dtype != self.activation_dtype:
            raise TypeError(
                f"down_weight must use {self.activation_dtype}, "
                f"got {down_weight.dtype}"
            )
        if down_weight.device != weighted_activation.device:
            raise ValueError(
                "down_weight and weighted_activation must be on the same device"
            )
        if not down_weight.is_contiguous():
            raise ValueError(
                "down_weight must be contiguous with layout "
                "[experts_per_rank, hidden_size, ffn_size]"
            )

        expected_output_shape = (plan.num_input_tokens, self.hidden_size)
        if combine_output is None:
            output = torch.empty(
                expected_output_shape,
                dtype=self.activation_dtype,
                device=weighted_activation.device,
            )
        else:
            output = combine_output
            if output.shape != expected_output_shape:
                raise ValueError(
                    f"combine_output must have shape {expected_output_shape}, "
                    f"got {tuple(output.shape)}"
                )
            if output.dtype != self.activation_dtype:
                raise TypeError(
                    f"combine_output must use {self.activation_dtype}, "
                    f"got {output.dtype}"
                )
            if output.device != weighted_activation.device:
                raise ValueError(
                    "combine_output and weighted_activation must be on the same device"
                )
            if not output.is_contiguous():
                raise ValueError("combine_output must be contiguous")

        block_n = self.config.fc2_gemm_block_size_n
        block_k = self.config.fc2_gemm_block_size_k
        if self.hidden_size % block_n or ffn_size % block_k:
            raise ValueError(
                "FC2 block_n and block_k must divide the output and reduction dimensions"
            )
        combine_metadata = self._prepare_combine_metadata(dispatch_result)
        direct_pull = self.config.fc2_combine_transport == "direct_pull"
        # Reverse push uses this buffer for local FC2 rows.  Direct pull writes
        # FC2 to peer_mem, then pulls remote bucket segments into this ordinary
        # local workspace in stable-send order before the top-k reduction.
        workspace_rows = (
            plan.num_sent_routes if direct_pull else num_received_routes
        )
        fc2_workspace = self._combine_fc2_buf[:workspace_rows]
        launch_fc2_combine(
            weighted_activation,
            down_weight,
            fc2_workspace,
            self.context.peer_mem,
            combine_metadata["route_to_send"],
            output,
            plan.received_routes_per_expert,
            plan.received_expert_offsets,
            self._fc2_tile_expert,
            self._fc2_tile_row_start,
            self._fc2_tile_row_count,
            self._reverse_tile_rank,
            self._reverse_tile_src_start,
            self._reverse_tile_dst_start,
            self._reverse_tile_row_count,
            combine_metadata["num_fc2_slots"],
            combine_metadata["num_reverse_slots"],
            plan.num_sent_routes,
            topk=self.top_k,
            num_cores=self.num_aicore_programs,
            block_m=self.config.fc2_combine_block_size_m,
            block_n=block_n,
            block_k=block_k,
            world_size=self.world_size,
            expert_n_persistent=(
                self.config.fc2_gemm_schedule == "expert_n_persistent"
            ),
            direct_pull=direct_pull,
            reverse_vector_workers=(
                self.config.fc2_reverse_vector_workers
            ),
            reduce_vector_workers=self.config.fc2_reduce_vector_workers,
        )
        return output

    # ===================== full forward ================================
    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the complete BF16 post-routing MoE forward.

        The caller provides selected experts and FP32 routing weights. Router
        matmul, softmax, and top-k selection are outside this boundary.
        """
        weighted_activation, dispatch_result = self.dispatch_fc1_weighted_swiglu(
            hidden_states,
            selected_experts,
            routing_weights,
            gate_up_weight,
            final_dispatch_barrier=False,
        )
        # Weighted SwiGLU applies each route weight exactly once before FC2.
        return self.fc2_combine(
            weighted_activation,
            down_weight,
            dispatch_result,
        )


def pack_gate_up_weights(
    gate_weight_local: torch.Tensor,
    up_weight_local: torch.Tensor,
) -> torch.Tensor:
    """Pack local BF16 gate/up projections as contiguous ``[E, H, 2F]``."""
    if gate_weight_local.ndim != 3 or up_weight_local.ndim != 3:
        raise ValueError(
            "gate and up weights must have shape "
            "[experts_per_rank, ffn_size, hidden_size]"
        )
    if gate_weight_local.shape != up_weight_local.shape:
        raise ValueError(
            "gate and up weight shapes must match, got "
            f"{gate_weight_local.shape} and {up_weight_local.shape}"
        )
    if (
        gate_weight_local.dtype != torch.bfloat16
        or up_weight_local.dtype != torch.bfloat16
    ):
        raise TypeError("gate and up weights must both use torch.bfloat16")
    if gate_weight_local.device != up_weight_local.device:
        raise ValueError("gate and up weights must be on the same device")
    return torch.cat(
        (gate_weight_local.transpose(1, 2), up_weight_local.transpose(1, 2)),
        dim=2,
    ).contiguous()


__all__ = ["DispatchFC1Result", "FusedMoEForward", "pack_gate_up_weights"]
