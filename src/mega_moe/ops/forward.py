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
    _FC2_REMOTE_STORE_BLOCK,
    _FC2_TRANSPORT_BLOCK_M,
    _fc2_reduce_block_n,
    build_route_to_send,
    launch_fc2_combine,
    prepare_fc2_remote_store_metadata,
)
from ..runtime.routing import (
    MoERoutingPlan,
    build_routing_plan,
)
from ..runtime.workspace import create_moe_forward_context
from ..kernels.weighted_swiglu import weighted_swiglu_forward


_FC2_PIPELINE_GROUP_EXPERTS = 16


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

    Dispatch and FC1 overlap through per-source-tile readiness signals in the
    fixed all-core expert/N-tile pipeline.
    ``dispatch_fc1_weighted_swiglu`` extends the supported path
    through weighted SwiGLU. FC2, route transport, route restoration, and
    top-k reduction then run in the dedicated combine kernels. Instances are
    single-in-flight because their workspaces, streams, and events are reused.
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
        config contains only stage-specific capacity and tiling controls;
        activation and expert-weight dtypes are fixed to BF16.
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
        self.num_aivector_programs = self.config.num_aivector_programs
        self._fc2_pipeline_group_experts = min(
            _FC2_PIPELINE_GROUP_EXPERTS, self.experts_per_rank
        )
        self._fc2_remote_store_block = _FC2_REMOTE_STORE_BLOCK
        self._fc2_transport_block_m = _FC2_TRANSPORT_BLOCK_M
        # Keep the public metadata value for provenance; the implementation is
        # fixed to this validated default and no longer branches on it.
        self.dispatch_fc1_schedule = self.config.dispatch_fc1_schedule
        # Preserve the target repository optional SiTU-GLU activation.
        self.activation = self.config.activation
        self.situ_beta = self.config.situ_beta
        self.situ_linear_beta = self.config.situ_linear_beta

        # Tile SET slots publish per-source-tile readiness for the Cube consumer.
        self._tile_signal_epoch = 1
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

        # FC2/combine workspaces remain lazy so
        # dispatch-only stage tests do not pay their memory cost.
        self._combine_fc2_buf = None
        self._combine_fc2_storage = None
        self._combine_pipeline_cube_stream = None
        self._combine_pipeline_vector_stream = None
        self._combine_pipeline_group_events = []
        self._combine_pipeline_start_event = None
        self._combine_pipeline_done_event = None
        self._route_to_send = None
        self._pull_tile_rank = None
        self._pull_tile_src_start = None
        self._pull_tile_dst_start = None
        self._pull_tile_row_count = None
        # Grouped remote-store metadata: one contiguous descriptor segment per
        # (local expert group, source rank).  These ordinary-GM workspaces are
        # rebuilt by the single metadata kernel for each forward.
        self._pull_group_segment_starts = None
        self._pull_group_segment_counts = None
        self._max_pull_tile_slots = 0
        self._routing_weights_keepalive = None

        # All ranks must observe zeroed symmetric buffers before first use.
        torch.npu.synchronize()
        torch.distributed.barrier(group=self.ep_group)

    # ===================== buffer-management =====================
    def sync(self):
        torch.npu.synchronize()

    def finalize(self):
        # Grouped FC2 may still have work queued on its dedicated Cube/Vector
        # streams even when the caller never explicitly synchronized.  Drain
        # every stream before releasing any symmetric allocation they can
        # still reference.
        if self.context.peer_mem is not None:
            torch.npu.synchronize(self.context.peer_mem.device)
        if self._combine_fc2_storage is not None:
            import shmem as ash

            ash.aclshmem_free_tensor(self._combine_fc2_storage)
            self._combine_fc2_storage = None
        self._routing_weights_keepalive = None
        self._combine_fc2_buf = None
        self._combine_pipeline_cube_stream = None
        self._combine_pipeline_vector_stream = None
        self._combine_pipeline_group_events = []
        self._combine_pipeline_start_event = None
        self._combine_pipeline_done_event = None
        self._route_to_send = None
        self._pull_tile_rank = None
        self._pull_tile_src_start = None
        self._pull_tile_dst_start = None
        self._pull_tile_row_count = None
        self._pull_group_segment_starts = None
        self._pull_group_segment_counts = None
        self.context.finalize()

    def _ensure_combine_buffers(self):
        """Allocate reusable local workspaces for FC2 and route restoration."""
        if self._combine_fc2_buf is not None:
            return

        import shmem as ash

        max_send = self.max_tokens_per_rank * self.top_k
        max_receive = self.context.peer_mem.numel() // self.hidden_size
        max_receive_tiles = (
            max_receive + self._fc2_transport_block_m - 1
        ) // self._fc2_transport_block_m
        self._max_pull_tile_slots = (
            max_receive_tiles + self.world_size * self.experts_per_rank
        )
        device = self.context.peer_mem.device
        self._combine_fc2_storage = ash.aclshmem_create_tensor(
            [max_send * self.hidden_size],
            dtype=self.activation_dtype,
            device_id=self.rank,
        )
        self._combine_fc2_buf = self._combine_fc2_storage.view(
            max_send, self.hidden_size
        )
        self._route_to_send = torch.empty(
            max_send, dtype=torch.int32, device=device)

        def make_int_workspace(size):
            return torch.empty(size, dtype=torch.int32, device=device)

        self._pull_tile_rank = make_int_workspace(self._max_pull_tile_slots)
        self._pull_tile_src_start = make_int_workspace(self._max_pull_tile_slots)
        self._pull_tile_dst_start = make_int_workspace(self._max_pull_tile_slots)
        self._pull_tile_row_count = make_int_workspace(self._max_pull_tile_slots)
        # At most one group per local expert is useful.  Segment metadata is
        # indexed [group, source-rank], so its size is independent of the
        # token/route count and remains tiny compared with the FC2 workspace.
        self._pull_group_segment_starts = make_int_workspace(
            self.experts_per_rank * self.world_size
        )
        self._pull_group_segment_counts = make_int_workspace(
            self.experts_per_rank * self.world_size
        )

    def _ensure_group_pipeline_runtime(self, group_experts: int):
        """Create reusable streams/events for the coarse FC2 pipeline."""
        device = self.context.peer_mem.device
        if self._combine_pipeline_cube_stream is None:
            self._combine_pipeline_cube_stream = torch.npu.Stream(device=device)
            self._combine_pipeline_vector_stream = torch.npu.Stream(device=device)
            self._combine_pipeline_start_event = torch.npu.Event()
            self._combine_pipeline_done_event = torch.npu.Event()
        num_groups = (
            self.experts_per_rank + group_experts - 1
        ) // group_experts
        while len(self._combine_pipeline_group_events) < num_groups:
            self._combine_pipeline_group_events.append(torch.npu.Event())

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

        transport_block_m = self._fc2_transport_block_m
        pipeline_group_experts = self._fc2_pipeline_group_experts
        # The production remote-store descriptors retain the expert-major
        # receive layout and coalesce each bucket at the separately tuned
        # transport row size.  This is intentionally independent from the
        # FC2 GEMM M tile: a large RMA request must not force a huge Cube
        # accumulator tile.
        receive_tiles = (
            plan.num_received_routes + transport_block_m - 1
        ) // transport_block_m
        num_pull_slots = receive_tiles + self.world_size * self.experts_per_rank
        if num_pull_slots > self._max_pull_tile_slots:
            raise ValueError(
                "remote-store tile metadata exceeds its configured workspace"
            )

        prepare_fc2_remote_store_metadata(
            self.context.metadata_counts_mem,
            plan.received_expert_offsets,
            self._pull_tile_rank,
            self._pull_tile_src_start,
            self._pull_tile_dst_start,
            self._pull_tile_row_count,
            num_pull_slots,
            local_rank=self.rank,
            world_size=self.world_size,
            experts_per_rank=self.experts_per_rank,
            num_bins_pad=self.context.metadata_num_bins,
            block_m=transport_block_m,
            group_segment_starts=self._pull_group_segment_starts,
            group_segment_counts=self._pull_group_segment_counts,
            group_experts=pipeline_group_experts,
        )
        return {
            "route_to_send": route_to_send,
            "num_pull_slots": num_pull_slots,
            "pipeline_group_experts": pipeline_group_experts,
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
        signal_epoch = self._tile_signal_epoch
        _kernel_dispatch_fc1[self.num_aicore_programs, 1, 1](
            hidden_states,
            self.context.peer_mem,
            routing_weights,
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
            NUM_PROGRAM_CORES=self.num_aicore_programs,
            LOCAL_RANK=self.rank,
            WORLD_SIZE=self.world_size,
            EXPERTS_PER_RANK=self.experts_per_rank,
            MAX_SOURCE_TILES=self.context.max_source_tiles,
            FINAL_BARRIER=final_barrier,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )
        self._tile_signal_epoch += 1

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

        self.activation selects SwiGLU or SiTU-GLU.  The activation
        parameters are ignored for the default SwiGLU path.
        """
        return weighted_swiglu_forward(
            dispatch_result.fc1_output,
            dispatch_result.received_routing_weights,
            self.num_aivector_programs,
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
        # FC2 writes destination-local rows to one symmetric workspace; the
        # Vector pipeline stores them directly into each source rank's
        # symmetric send-order workspace before top-k reduction.
        fc2_workspace = self._combine_fc2_buf[:plan.num_sent_routes]
        pipeline_group_experts = combine_metadata["pipeline_group_experts"]
        reduce_block_n = _fc2_reduce_block_n(num_received_routes)
        self._ensure_group_pipeline_runtime(pipeline_group_experts)
        launch_fc2_combine(
            weighted_activation,
            down_weight,
            fc2_workspace,
            self.context.peer_mem,
            combine_metadata["route_to_send"],
            output,
            plan.received_routes_per_expert,
            plan.received_expert_offsets,
            self._pull_tile_rank,
            self._pull_tile_src_start,
            self._pull_tile_dst_start,
            self._pull_tile_row_count,
            combine_metadata["num_pull_slots"],
            plan.num_sent_routes,
            topk=self.top_k,
            num_program_cores=self.num_aicore_programs,
            num_vector_programs=self.num_aivector_programs,
            reduce_block_n=reduce_block_n,
            block_m=self.config.fc2_combine_block_size_m,
            block_n=block_n,
            block_k=block_k,
            world_size=self.world_size,
            pipeline_group_experts=pipeline_group_experts,
            group_segment_starts=self._pull_group_segment_starts,
            group_segment_counts=self._pull_group_segment_counts,
            pipeline_cube_stream=self._combine_pipeline_cube_stream,
            pipeline_vector_stream=self._combine_pipeline_vector_stream,
            pipeline_group_events=self._combine_pipeline_group_events,
            pipeline_start_event=self._combine_pipeline_start_event,
            pipeline_done_event=self._combine_pipeline_done_event,
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
