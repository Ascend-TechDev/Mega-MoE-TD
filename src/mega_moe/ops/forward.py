# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standalone BF16 Ascend Mega-MoE post-routing forward.

The public boundary starts with selected experts and FP32 routing weights;
router matmul, softmax, and top-k selection are intentionally excluded.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import os

import torch
import torch.distributed

from ..config import MoEForwardConfig
from ..kernels.dispatch_fc1 import _kernel_dispatch_fc1
from ..kernels.fused_forward import _kernel_fused_forward
from ..kernels.replica_weight_prefetch import (
    _kernel_compact_local_replica_descriptors,
)
from ..kernels.fc2_combine import (
    _fc2_reduce_block_n,
    _launch_fc2_combine,
    _validate_putmem_descriptor_capacity,
    build_route_to_send,
    prepare_fc2_device_put_metadata,
)
from ..runtime.routing import (
    MoERoutingPlan,
    build_routing_plan,
)
from ..runtime.replica_weight_prefetch import (
    allocate_replica_weight_buffers,
    fence_replica_weight_prefetch_async,
    replica_weight_push_geometry,
)
from ..runtime.workspace import create_moe_forward_context
from ._native_saved import (
    assemble_native_saved,
    capture_activations,
    snapshot_plan_metadata,
)


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
    The production full path shadows weighted activation under FC2, route
    transport, route restoration, and top-k reduction. Instances are
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
        self.enable_moonep = self.config.enable_moonep
        self.enable_single_kernel_forward = (
            self.config.enable_single_kernel_forward
        )
        if self.enable_moonep and self.world_size & (self.world_size - 1):
            raise ValueError(
                "MoonEP Triton planning requires a power-of-two EP world size"
            )
        self.replica_budget = self.experts_per_rank if self.enable_moonep else 0
        self.physical_experts_per_rank = (
            self.experts_per_rank + self.replica_budget
        )
        self.receive_capacity_factor = (
            self.config.resolved_receive_capacity_factor(self.world_size)
        )
        self.num_aicore_programs = self.config.num_aicore_programs
        self.num_aivector_programs = self.config.num_aivector_programs
        if (
            self.enable_single_kernel_forward
            and self.world_size > self.num_aicore_programs
        ):
            raise ValueError(
                "single-kernel forward requires world_size no larger than "
                "the physical AICore count"
            )
        self._fc2_pipeline_group_experts = min(
            _FC2_PIPELINE_GROUP_EXPERTS, self.physical_experts_per_rank
        )
        # Preserve the target repository optional SiTU-GLU activation.
        self.activation = self.config.activation
        self.situ_beta = self.config.situ_beta
        self.situ_linear_beta = self.config.situ_linear_beta

        # Tile SET slots publish per-source-tile readiness for the Cube consumer.
        self._tile_signal_epoch = 1
        # Routing metadata points into a single-in-flight workspace.  An
        # identity token plus monotonically increasing generation makes staged
        # dispatch reject plans copied across operators or superseded by a
        # later build, without adding device work to the forward hot path.
        self._routing_owner_token = object()
        self._routing_generation = 0
        self.context = create_moe_forward_context(
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden_size,
            top_k=top_k,
            num_experts=num_experts,
            rank=self.rank,
            world_size=self.world_size,
            receive_capacity_factor=self.receive_capacity_factor,
            dispatch_fc1_block_size_m=self.config.dispatch_fc1_block_size_m,
            enable_moonep=self.enable_moonep,
            ep_group=self.ep_group,
        )

        # FC2/combine workspaces remain lazy so
        # dispatch-only stage tests do not pay their memory cost.
        self._combine_fc2_buf = None
        self._combine_fc2_storage = None
        self._combine_pipeline_cube_stream = None
        self._combine_pipeline_vector_stream = None
        self._combine_pipeline_transfer_stream = None
        self._combine_pipeline_group_events = []
        self._combine_pipeline_activation_events = []
        self._combine_pipeline_start_event = None
        self._combine_pipeline_done_event = None
        self._route_to_send = None
        self._pull_tile_rank = None
        self._pull_tile_src_start = None
        self._pull_tile_dst_start = None
        self._pull_tile_row_count = None
        self._single_fc1_output = None
        self._single_weighted_activation = None
        self._single_core_bucket_cursors = None
        self._single_send_token_indices = None
        self._single_send_route_indices = None
        self._routing_weights_keepalive = None
        self._replica_weight_buffers = None
        self._replica_weight_cache_key = None
        self._replica_weight_cache_valid = False
        self._replica_experts_cache = None
        self._replica_prepared_experts_cpu = None
        self._replica_weight_source_refs = None
        self._replica_prefetch_stream = None
        self._replica_dispatch_done_event = None
        self._replica_prefetch_done_event = None
        self._replica_prefetch_pending = False
        self._replica_step_has_replicas = False
        self._replica_step_prepared = False
        self._replica_down_descriptor_count = 0
        self._replica_weight_epoch = 1
        self._active_replica_weight_epoch = 1
        self._forward_stream = None
        # Persistent buffers for the large native-saved activations (see
        # _get_saved_workspace).  Allocated once on first saved-forward and
        # reused every step at a fixed address; MOE_SAVED_WORKSPACE=0 restores
        # the per-step allocation (leaks ~3.8 GiB/iter on the integrated
        # training path — the autograd ctx pins the saved dict).
        self._saved_ws = None
        # Fixed-address staging for a strided down (fc2) weight view (see
        # _materialize_down_weight) — same persistent-buffer rationale.
        self._fc2_ws = None

        # All ranks must observe zeroed symmetric buffers before first use.
        torch.npu.synchronize()
        torch.distributed.barrier(group=self.ep_group)

    # ===================== buffer-management =====================
    def _get_saved_workspace(self, hidden_states, gate_up_weight):
        """Fixed-address buffers for the large native-saved activations.

        Sized once to the receive capacity (peer_mem bound) and handed out
        every step as ``[:M]`` slices for ``fc1_output`` /
        ``recv_hidden_sorted`` / ``recv_weights_sorted`` /
        ``swiglu_out_weighted``.  The blocks never return to the caching
        allocator, which matters twice on the integrated training path:

        * the autograd ctx of ``MegaMoEFunction`` outlives the graph and pins
          the saved dict — with per-step allocations that pinned dict grows
          the live set by ~3.8 GiB/iteration (OOM at iter 5 on kimi-k3 w8);
          views of these persistent buffers pin nothing new;
        * recycled blocks get picked up by the next allocation while the
          operator's own-stream / free-flight (ffts) work may still be using
          them — the free-flight tasks outlive torch stream semantics, so any
          cross-step reuse races the dispatch signal chain (ffts spin →
          vector core timeout).  A fixed address cannot be raced.
        """
        if self._saved_ws is None:
            max_recv = self.context.peer_mem.numel() // self.hidden_size
            ffn = gate_up_weight.shape[2] // 2
            dev = hidden_states.device
            self._saved_ws = {
                "fc1_output": torch.empty(
                    max_recv, 2 * ffn,
                    dtype=self.activation_dtype, device=dev),
                "recv_hidden_sorted": torch.empty(
                    max_recv, self.hidden_size,
                    dtype=self.activation_dtype, device=dev),
                "recv_weights_sorted": torch.empty(
                    max_recv, dtype=self.activation_dtype, device=dev),
                "swiglu_out_weighted": torch.empty(
                    max_recv, ffn,
                    dtype=self.activation_dtype, device=dev),
            }
        return self._saved_ws

    def _materialize_down_weight(self, down_weight):
        """Fixed-address contiguous staging for a strided down (fc2) view.

        The integrated host stores down_proj as ``[E, F, H]`` and used to hand
        over a fresh ``transpose(1, 2).contiguous()`` copy every step (~84 MB
        per layer at the kimi-k3 shape) because the GEMM contract asked for a
        contiguous table.  That per-step copy is retained by the autograd
        ctx's pinned ``saved`` dict on the training path — +336 MB/iter of
        live-set growth.  A stride view is acceptable to both fc2 GEMMs (they
        address the table through explicit strides), but staging it here is
        strictly better: the copy lands in a persistent buffer (address never
        changes, so no block is ever recycled under a free-flight task) and
        ``copy_`` refreshes it every step, so optimizer updates flow through.
        ``MOE_SAVED_WORKSPACE=0`` restores the historical per-step
        ``.contiguous()``.
        """
        if down_weight.is_contiguous():
            return down_weight
        if os.environ.get("MOE_SAVED_WORKSPACE", "1") == "0":
            return down_weight.contiguous()
        if (
            self._fc2_ws is None
            or tuple(self._fc2_ws.shape) != tuple(down_weight.shape)
        ):
            self._fc2_ws = torch.empty(
                down_weight.shape,
                dtype=down_weight.dtype,
                device=down_weight.device,
            )
        self._fc2_ws.copy_(down_weight)
        return self._fc2_ws

    def sync(self):
        torch.npu.synchronize()

    def finalize(self):
        # Grouped FC2 may still have work queued on its dedicated Cube, Vector,
        # and transfer streams. Drain every stream before releasing
        # any symmetric allocation they can still reference.
        if self.context.peer_mem is not None:
            torch.npu.synchronize(self.context.peer_mem.device)
        if self.enable_moonep:
            # Quiesce owner-push RMA before freeing any symmetric destination
            # table.  This is a teardown-only collective; normal forwards keep
            # the overlap path free of a host synchronization.
            torch.distributed.barrier(group=self.ep_group)
        if self._combine_fc2_storage is not None:
            import shmem as ash

            ash.aclshmem_free_tensor(self._combine_fc2_storage)
            self._combine_fc2_storage = None
        if self._replica_weight_buffers is not None:
            self._replica_weight_buffers.finalize()
            self._replica_weight_buffers = None
        self._replica_weight_cache_key = None
        self._replica_weight_cache_valid = False
        self._replica_experts_cache = None
        self._replica_prepared_experts_cpu = None
        self._replica_weight_source_refs = None
        self._replica_prefetch_stream = None
        self._replica_dispatch_done_event = None
        self._replica_prefetch_done_event = None
        self._replica_prefetch_pending = False
        self._replica_step_has_replicas = False
        self._replica_step_prepared = False
        self._replica_down_descriptor_count = 0
        self._replica_weight_epoch = 1
        self._active_replica_weight_epoch = 1
        self._forward_stream = None
        self._routing_weights_keepalive = None
        self._combine_fc2_buf = None
        self._combine_pipeline_cube_stream = None
        self._combine_pipeline_vector_stream = None
        self._combine_pipeline_transfer_stream = None
        self._combine_pipeline_group_events = []
        self._combine_pipeline_activation_events = []
        self._combine_pipeline_start_event = None
        self._combine_pipeline_done_event = None
        self._route_to_send = None
        self._pull_tile_rank = None
        self._pull_tile_src_start = None
        self._pull_tile_dst_start = None
        self._pull_tile_row_count = None
        self._single_fc1_output = None
        self._single_weighted_activation = None
        self._single_core_bucket_cursors = None
        self._single_send_token_indices = None
        self._single_send_route_indices = None
        self.context.finalize()

    def _ensure_replica_weight_buffers(
        self,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        """Allocate fixed-B symmetric tables before routing collectives."""
        if not self.enable_moonep:
            return
        if self._replica_weight_buffers is None:
            self._replica_weight_buffers = allocate_replica_weight_buffers(
                gate_up_weight,
                down_weight,
                rank=self.rank,
                world_size=self.world_size,
            )
            torch.npu.synchronize(gate_up_weight.device)
            torch.distributed.barrier(group=self.ep_group)

    def _begin_replica_prefetch(
        self,
        experts_to_copy_cpu: torch.Tensor,
        experts_to_copy_device: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        """Prepare one cache-miss owner-push after ETC/weight readiness."""
        if not self.enable_moonep:
            return
        if self._replica_weight_buffers is None:
            raise RuntimeError("replica buffers must be allocated before planning")
        if self._replica_prefetch_pending and not self._replica_weight_cache_valid:
            raise RuntimeError(
                "replica prefetch is still in flight after an incomplete forward; "
                "finalize and recreate the operator"
            )

        has_replica = int(bool((experts_to_copy_cpu >= 0).any().item()))
        self._replica_step_has_replicas = bool(has_replica)
        self._replica_step_prepared = True
        self._replica_prepared_experts_cpu = experts_to_copy_cpu.clone()
        if not has_replica:
            self._replica_prefetch_pending = False
            self._replica_down_descriptor_count = 0
            return

        weight_key = (
            gate_up_weight.data_ptr(),
            gate_up_weight._version,
            down_weight.data_ptr(),
            down_weight._version,
        )
        if self.config.moonep_enable_replica_cache:
            local_cache_hit = (
                self._replica_weight_cache_valid
                and weight_key == self._replica_weight_cache_key
                and self._replica_experts_cache is not None
                and torch.equal(
                    experts_to_copy_cpu, self._replica_experts_cache
                )
                and self._replica_weight_source_refs is not None
                and self._replica_weight_source_refs[0] is gate_up_weight
                and self._replica_weight_source_refs[1] is down_weight
            )
            cache_flag = torch.tensor(
                [int(local_cache_hit)],
                dtype=torch.int32,
                device=gate_up_weight.device,
            )
            torch.distributed.all_reduce(
                cache_flag,
                op=torch.distributed.ReduceOp.MIN,
                group=self.ep_group,
            )
            if bool(cache_flag.item()):
                self._replica_prefetch_pending = False
                self._replica_down_descriptor_count = 0
                return

        device = gate_up_weight.device
        if self._replica_prefetch_stream is None:
            self._replica_prefetch_stream = torch.npu.Stream(device=device)
            self._replica_dispatch_done_event = torch.npu.Event()
            self._replica_prefetch_done_event = torch.npu.Event()
        epoch = self._replica_weight_epoch
        if epoch >= torch.iinfo(torch.int32).max:
            raise RuntimeError(
                "replica weight readiness epoch exhausted; recreate the operator"
            )
        # Count and compact only owner-local remote copies.  The mixed MTE
        # dispatch/FC1 kernel consumes this list directly.
        local_owner_start = self.rank * self.experts_per_rank
        local_owner_end = local_owner_start + self.experts_per_rank
        locally_owned = (
            (experts_to_copy_cpu >= local_owner_start)
            & (experts_to_copy_cpu < local_owner_end)
        )
        local_copy_count = int(
            locally_owned.sum().item()
            - locally_owned[self.rank].sum().item()
        )
        descriptor_workspace = self.context.replica_down_descriptors
        if descriptor_workspace is None:
            raise RuntimeError("MoonEP down-prefetch descriptors are not allocated")
        if local_copy_count > descriptor_workspace.numel():
            raise RuntimeError("MoonEP down-prefetch descriptor workspace exhausted")
        self._replica_down_descriptor_count = local_copy_count
        # Compact the tiny ETC list on device in slot-major/peer-interleaved
        # order.  The current stream makes it visible to the following mixed
        # kernel without any host nonzero or H2D descriptor copy.
        _kernel_compact_local_replica_descriptors[1, 1, 1](
            experts_to_copy_device,
            descriptor_workspace,
            LOCAL_RANK=self.rank,
            WORLD_SIZE=self.world_size,
            EXPERTS_PER_RANK=self.experts_per_rank,
        )
        self._replica_prefetch_pending = True
        self._replica_weight_cache_valid = False
        self._active_replica_weight_epoch = epoch
        self._replica_weight_epoch += 1
        self._replica_weight_cache_key = weight_key
        self._replica_experts_cache = experts_to_copy_cpu.clone()
        self._replica_weight_source_refs = (gate_up_weight, down_weight)
    def _finish_replica_prefetch(self):
        """Return replica views; kernel-side signals preserve slot overlap."""
        if not self.enable_moonep:
            return None, None
        return self._replica_weight_buffers.gate_up, self._replica_weight_buffers.down

    def _fence_replica_prefetch_after_dispatch(self, device):
        """Fence fused gate/down RMA after the mixed kernel completes."""
        if not self._replica_prefetch_pending:
            return None
        current_stream = torch.npu.current_stream(device)
        dispatch_done_event = self._replica_dispatch_done_event
        prefetch_stream = self._replica_prefetch_stream
        done_event = self._replica_prefetch_done_event
        dispatch_done_event.record(current_stream)
        with torch.npu.stream(prefetch_stream):
            prefetch_stream.wait_event(dispatch_done_event)
            fence_replica_weight_prefetch_async(
                num_barrier_programs=self.num_aicore_programs
            )
            done_event.record(prefetch_stream)
        return done_event

    def _ensure_combine_buffers(self):
        """Allocate reusable local workspaces for FC2 and route restoration."""
        if self._combine_fc2_buf is not None:
            return

        max_send = self.max_tokens_per_rank * self.top_k
        _validate_putmem_descriptor_capacity(max_send, self.hidden_size)

        import shmem as ash

        physical_experts_per_rank = getattr(
            self, "physical_experts_per_rank", self.experts_per_rank
        )
        descriptor_slots = self.world_size * physical_experts_per_rank
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

        self._pull_tile_rank = make_int_workspace(descriptor_slots)
        self._pull_tile_src_start = make_int_workspace(descriptor_slots)
        self._pull_tile_dst_start = make_int_workspace(descriptor_slots)
        self._pull_tile_row_count = make_int_workspace(descriptor_slots)

    def _ensure_single_kernel_buffers(self, gate_up_weight: torch.Tensor):
        """Allocate fixed-capacity ordinary buffers for the fused launch."""
        self._ensure_combine_buffers()
        max_recv = self.context.peer_mem.numel() // self.hidden_size
        max_send = self.max_tokens_per_rank * self.top_k
        ffn_size = gate_up_weight.shape[2] // 2
        device = self.context.peer_mem.device

        expected_fc1 = (max_recv, 2 * ffn_size)
        expected_activation = (max_recv, ffn_size)
        if self._single_fc1_output is None:
            self._single_fc1_output = torch.empty(
                expected_fc1,
                dtype=self.activation_dtype,
                device=device,
            )
            self._single_weighted_activation = torch.empty(
                expected_activation,
                dtype=self.activation_dtype,
                device=device,
            )
            self._single_core_bucket_cursors = torch.empty(
                (self.num_aicore_programs, self.context.metadata_num_bins),
                dtype=torch.int32,
                device=device,
            )
            self._single_send_token_indices = torch.empty(
                max_send,
                dtype=torch.int32,
                device=device,
            )
            self._single_send_route_indices = torch.empty(
                max_send,
                dtype=torch.int32,
                device=device,
            )
        elif (
            tuple(self._single_fc1_output.shape) != expected_fc1
            or tuple(self._single_weighted_activation.shape)
            != expected_activation
        ):
            raise ValueError(
                "single-kernel workspaces were initialized for a different "
                "FFN size; recreate the operator"
            )

    def _ensure_group_pipeline_runtime(
        self,
        group_experts: int,
        active_experts_per_rank: int,
        active_group_ids: tuple[int, ...] | None = None,
    ):
        """Create reusable streams/events for the coarse FC2 pipeline."""
        device = self.context.peer_mem.device
        if self._combine_pipeline_cube_stream is None:
            self._combine_pipeline_cube_stream = torch.npu.Stream(device=device)
            self._combine_pipeline_vector_stream = torch.npu.Stream(device=device)
            self._combine_pipeline_transfer_stream = torch.npu.Stream(device=device)
            self._combine_pipeline_start_event = torch.npu.Event()
            self._combine_pipeline_done_event = torch.npu.Event()
        num_groups = (
            len(active_group_ids)
            if active_group_ids is not None
            else (active_experts_per_rank + group_experts - 1) // group_experts
        )
        while len(self._combine_pipeline_group_events) < num_groups:
            self._combine_pipeline_group_events.append(torch.npu.Event())
        while len(self._combine_pipeline_activation_events) < num_groups:
            self._combine_pipeline_activation_events.append(torch.npu.Event())

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

    def _validate_gate_up_weight(
        self,
        gate_up_weight: torch.Tensor,
        device: torch.device,
    ):
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
        if gate_up_weight.device != device:
            raise ValueError("gate_up_weight and hidden_states must be on the same device")
        if not gate_up_weight.is_contiguous():
            raise ValueError(
                "gate_up_weight must be contiguous and pre-packed before forward"
            )

    # ===================== routing metadata =========================
    def build_routing_plan(
        self,
        selected_experts: torch.Tensor,
        *,
        moonep_plan_hook: Optional[Callable] = None,
    ) -> MoERoutingPlan:
        """Build stable dispatch/count metadata for one set of selected experts."""
        self._validate_topk_indices(selected_experts)
        if self.enable_moonep and moonep_plan_hook is None:
            # Metadata-only planning does not prepare replica weights.  Invalidate
            # any prior staged state so a new plan cannot consume an older ETC.
            self._replica_step_prepared = False
            self._replica_prepared_experts_cpu = None
        # Every build rewrites the shared metadata workspace, so invalidate any
        # older staged plan even when its selected_experts tensor is unchanged.
        self._routing_generation += 1
        generation = self._routing_generation
        plan = build_routing_plan(
            self.context,
            selected_experts,
            moonep_plan_hook=moonep_plan_hook,
            moonep_fused_balanced_count=(
                self.config.moonep_fused_balanced_count
            ),
            moonep_fused_route_mapping=(
                self.config.moonep_fused_route_mapping
            ),
        )
        plan.owner_token = self._routing_owner_token
        plan.generation = generation
        plan.selected_experts_version = selected_experts._version
        return plan

    def _prepare_combine_metadata(
        self,
        dispatch_result: DispatchFC1Result,
    ) -> dict:
        """Build device-put FC2 descriptors and the route inverse."""
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

        pipeline_group_experts = self._fc2_pipeline_group_experts

        prepare_fc2_device_put_metadata(
            self.context.metadata_counts_mem,
            plan.received_expert_offsets,
            self._pull_tile_rank,
            self._pull_tile_src_start,
            self._pull_tile_dst_start,
            self._pull_tile_row_count,
            local_rank=self.rank,
            world_size=self.world_size,
            experts_per_rank=self.physical_experts_per_rank,
            num_bins_pad=self.context.metadata_num_bins,
        )
        return {
            "route_to_send": route_to_send,
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
        replica_fc1_weight: Optional[torch.Tensor] = None,
        replica_weight_ready: Optional[torch.Tensor] = None,
        replica_weight_epoch: int = 0,
        down_weight_to_prefetch: Optional[torch.Tensor] = None,
        replica_down_weight_to_prefetch: Optional[torch.Tensor] = None,
        replica_down_ready: Optional[torch.Tensor] = None,
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
        if self.enable_moonep and (
            routing_plan.owner_token is not self._routing_owner_token
            or routing_plan.generation != self._routing_generation
        ):
            raise RuntimeError(
                "MoonEP routing plan belongs to another or superseded operator"
            )
        if (
            self.enable_moonep
            and routing_plan.selected_experts_version != selected_experts._version
        ):
            raise RuntimeError(
                "MoonEP selected_experts was modified after routing plan build"
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

        send_route_indices = routing_plan.send_route_indices
        if send_route_indices is None:
            route_indices = self.context.row_route_indices[: selected_experts.numel()]
            if routing_plan.num_sent_routes == selected_experts.numel():
                kept_route_indices = route_indices
            else:
                kept_route_indices = route_indices[routing_plan.valid_route_mask]
            send_route_indices = kept_route_indices[
                routing_plan.stable_sort_indices
            ].to(torch.int32).contiguous()

        num_received_routes = routing_plan.num_received_routes
        if routing_plan.physical_experts_per_rank != self.physical_experts_per_rank:
            raise ValueError(
                "routing plan physical expert stride does not match the operator"
            )
        active_experts_per_rank = routing_plan.active_physical_experts_per_rank
        if not self.experts_per_rank <= active_experts_per_rank <= self.physical_experts_per_rank:
            raise ValueError(
                "routing plan active physical expert bound must be between the "
                "home and allocated physical expert counts"
            )
        # Present a logical [expert, N, K] view without materializing a second
        # multi-GiB weight table.  N remains contiguous in the physical KN
        # model-load layout.
        weight_for_gemm = fc1_weight.transpose(-1, -2)
        if replica_fc1_weight is None:
            replica_weight_for_gemm = weight_for_gemm
        else:
            if replica_fc1_weight.shape != fc1_weight.shape:
                raise ValueError("replica_fc1_weight must match fc1_weight shape")
            if (
                replica_fc1_weight.dtype != fc1_weight.dtype
                or replica_fc1_weight.device != fc1_weight.device
                or not replica_fc1_weight.is_contiguous()
            ):
                raise ValueError(
                    "replica_fc1_weight must be a contiguous tensor matching "
                    "fc1_weight dtype and device"
                )
            replica_weight_for_gemm = replica_fc1_weight.transpose(-1, -2)

        _, output_size, reduction_size = weight_for_gemm.shape
        block_n = self.config.fc1_gemm_block_size_n
        block_k = self.config.fc1_gemm_block_size_k
        dispatch_block_m = self.config.dispatch_fc1_block_size_m
        gemm_block_m = self.config.fc1_gemm_block_size_m
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
        down_push_args = (
            down_weight_to_prefetch,
            replica_down_weight_to_prefetch,
            replica_down_ready,
        )
        prefetch_replica_down = any(arg is not None for arg in down_push_args)
        if prefetch_replica_down and not all(arg is not None for arg in down_push_args):
            raise ValueError(
                "down_weight_to_prefetch, replica_down_weight_to_prefetch, "
                "and replica_down_ready must be provided together"
            )
        down_weight_elements = 1
        down_chunk_elements = 1
        down_num_chunks = 1
        gate_up_weight_elements = 1
        gate_up_chunk_elements = 1
        gate_up_num_chunks = 1
        down_weight_for_push = fc1_weight
        replica_down_weight_for_push = fc1_weight
        experts_to_copy_for_push = send_route_indices
        down_ready_for_push = self.context.signal_mem
        down_descriptor_ids_for_push = self.context.signal_mem
        down_descriptor_count = 0
        if self.enable_moonep:
            if self._replica_weight_buffers is None or not self._replica_step_prepared:
                raise RuntimeError(
                    "MoonEP dispatch requires replica preparation by full forward"
                )
            if (
                routing_plan.experts_to_copy_cpu is None
                or self._replica_prepared_experts_cpu is None
                or not torch.equal(
                    routing_plan.experts_to_copy_cpu,
                    self._replica_prepared_experts_cpu,
                )
            ):
                raise RuntimeError(
                    "MoonEP routing plan does not match the prepared replica layout"
                )
            if replica_weight_ready is None:
                raise ValueError(
                    "MoonEP FC1 requires a replica gate/up readiness table"
                )
            if (
                type(replica_weight_epoch) is not int
                or not 0 < replica_weight_epoch <= torch.iinfo(torch.int32).max
            ):
                raise ValueError(
                    "replica_weight_epoch must be a positive int32 value"
                )
            if (
                replica_weight_ready.dtype != torch.int32
                or replica_weight_ready.device != device
                or not replica_weight_ready.is_contiguous()
                or replica_weight_ready.numel() < self.replica_budget * 16
            ):
                raise ValueError(
                    "replica_weight_ready must provide 16 contiguous int32 "
                    "values per replica slot on the input device"
                )
            if prefetch_replica_down:
                expected_prefix = (self.experts_per_rank, hidden_size)
                for name, weight in (
                    ("down_weight_to_prefetch", down_weight_to_prefetch),
                    (
                        "replica_down_weight_to_prefetch",
                        replica_down_weight_to_prefetch,
                    ),
                ):
                    if (
                        weight.ndim != 3
                        or tuple(weight.shape[:2]) != expected_prefix
                        or weight.dtype != self.activation_dtype
                        or weight.device != device
                        or not weight.is_contiguous()
                    ):
                        raise ValueError(
                            f"{name} must be contiguous BF16 [experts, hidden, ffn] "
                            "on the input device"
                        )
                if replica_down_weight_to_prefetch.shape != down_weight_to_prefetch.shape:
                    raise ValueError(
                        "replica_down_weight_to_prefetch must match "
                        "down_weight_to_prefetch shape"
                    )
                if (
                    replica_down_ready.dtype != torch.int32
                    or replica_down_ready.device != device
                    or not replica_down_ready.is_contiguous()
                    or replica_down_ready.numel() < self.replica_budget * 16
                ):
                    raise ValueError(
                        "replica_down_ready must provide 16 contiguous int32 "
                        "values per replica slot on the input device"
                    )
                experts_to_copy_for_push = routing_plan.experts_to_copy
                if (
                    experts_to_copy_for_push is None
                    or tuple(experts_to_copy_for_push.shape)
                    != (self.world_size, self.experts_per_rank)
                    or experts_to_copy_for_push.dtype != torch.int32
                    or experts_to_copy_for_push.device != device
                    or not experts_to_copy_for_push.is_contiguous()
                ):
                    raise ValueError(
                        "MoonEP down prefetch requires its contiguous device ETC table"
                    )
                down_weight_for_push = down_weight_to_prefetch
                replica_down_weight_for_push = replica_down_weight_to_prefetch
                down_ready_for_push = replica_down_ready
                down_descriptor_ids_for_push = (
                    self.context.replica_down_descriptors
                )
                down_descriptor_count = self._replica_down_descriptor_count
                if down_descriptor_ids_for_push is None:
                    raise RuntimeError(
                        "MoonEP down-prefetch descriptors are not allocated"
                    )
                down_weight_elements = (
                    down_weight_to_prefetch.numel() // self.experts_per_rank
                )
                gate_up_weight_elements = (
                    fc1_weight.numel() // self.experts_per_rank
                )
                gate_up_chunk_elements, gate_up_num_chunks = (
                    replica_weight_push_geometry(
                        gate_up_weight_elements,
                        chunk_bytes=(
                            self.config.moonep_replica_gate_up_chunk_bytes
                        ),
                    )
                )
                down_chunk_elements, down_num_chunks = (
                    replica_weight_push_geometry(
                        down_weight_elements,
                        chunk_bytes=(
                            self.config.moonep_replica_down_chunk_bytes
                        ),
                    )
                )
        else:
            if prefetch_replica_down:
                raise ValueError("replica down prefetch requires MoonEP")
            # The home-only specialization never dereferences this pointer.
            replica_weight_ready = self.context.signal_mem
            replica_weight_epoch = 0
        _kernel_dispatch_fc1[self.num_aicore_programs, 1, 1](
            hidden_states,
            self.context.peer_mem,
            routing_weights,
            self.context.routing_weight_mem,
            self.context.signal_mem,
            weight_for_gemm,
            replica_weight_for_gemm,
            replica_weight_ready,
            down_weight_for_push,
            replica_down_weight_for_push,
            experts_to_copy_for_push,
            down_ready_for_push,
            down_descriptor_ids_for_push,
            down_descriptor_count,
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
            replica_weight_epoch,
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
            HOME_EXPERTS_PER_RANK=self.experts_per_rank,
            USE_REPLICA_WEIGHTS=self.enable_moonep,
            PREFETCH_REPLICA_DOWN=prefetch_replica_down,
            EXPERTS_PER_RANK=self.physical_experts_per_rank,
            ACTIVE_EXPERTS_PER_RANK=active_experts_per_rank,
            GATE_UP_WEIGHT_ELEMENTS_PER_EXPERT=gate_up_weight_elements,
            GATE_UP_CHUNK_ELEMENTS=gate_up_chunk_elements,
            GATE_UP_NUM_WEIGHT_CHUNKS=gate_up_num_chunks,
            DOWN_WEIGHT_ELEMENTS_PER_EXPERT=down_weight_elements,
            DOWN_CHUNK_ELEMENTS=down_chunk_elements,
            DOWN_NUM_WEIGHT_CHUNKS=down_num_chunks,
            DOWN_EARLY_PROGRAMS=(
                self.config.moonep_replica_down_early_programs
            ),
            DOWN_EARLY_DESCRIPTORS_PER_PROGRAM=(
                self.config.moonep_replica_down_early_descriptors_per_program
            ),
            MAX_SOURCE_TILES=self.context.max_source_tiles,
            FINAL_BARRIER=final_barrier,
            DISPATCH_BLOCK_SIZE_M=dispatch_block_m,
            GEMM_BLOCK_SIZE_M=gemm_block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            **(
                {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"}
                if gemm_block_m * block_n > 128 * 256
                else {}
            ),
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

    def lend_replica_weight_tables_for_grad(self):
        """Hand the symmetric replica weight tables to a physical backward.

        The M3 backward sinks its replica weight gradients into these same
        slots, so the caller must not run another forward on this operator until
        the borrowed transport has been reduced.  Lending therefore invalidates
        the replica weight cache immediately and permanently: a gradient left in
        a slot by an aborted backward can never be consumed as a weight, and the
        next forward re-pushes every replica it needs.

        Returns a :class:`mega_moe.runtime.replica_grad_transport.ReplicaGradTransport`
        holding a *reference* to the live buffers plus cloned planning
        snapshots (the routing workspace is reused in place, and a later
        metadata-only ``build_routing_plan`` clears the operator's staged ETC).
        """
        if not self.enable_moonep:
            raise RuntimeError("replica grad transport requires MoonEP")
        if self._replica_weight_buffers is None:
            raise RuntimeError("replica weight buffers have not been allocated")
        if self._replica_weight_cache_key is None or not self._replica_weight_cache_valid:
            raise RuntimeError(
                "replica weights are not published; run a full forward before "
                "lending the tables"
            )
        if self._replica_experts_cache is None:
            raise RuntimeError("no replica layout snapshot is available to lend")

        from ..runtime.replica_grad_transport import ReplicaGradTransport

        experts_to_copy_cpu = self._replica_experts_cache.clone()
        self._replica_weight_cache_valid = False
        return ReplicaGradTransport(
            buffers=self._replica_weight_buffers,
            experts_to_copy_cpu=experts_to_copy_cpu,
            experts_to_copy_device=experts_to_copy_cpu.to(
                self.context.peer_mem.device
            ),
            rank=self.rank,
            world_size=self.world_size,
            experts_per_rank=self.experts_per_rank,
            num_barrier_programs=self.num_aicore_programs,
            gate_up_chunk_bytes=self.config.moonep_replica_gate_up_chunk_bytes,
            down_chunk_bytes=self.config.moonep_replica_down_chunk_bytes,
        )

    # ===================== FC2 + combine ============================
    def _fc2_combine_shadow_activation(
        self,
        down_weight: torch.Tensor,
        dispatch_result: DispatchFC1Result,
        replica_down_weight: Optional[torch.Tensor] = None,
        prefetch_done_event=None,
        *,
        return_weighted_activation: bool = False,
        workspace: Optional[dict] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Produce grouped activation while FC2 consumes earlier groups.

        With ``return_weighted_activation=True`` also returns the shadowed
        ``weighted_activation`` buffer (native-saved ``swiglu_out_weighted``);
        the default keeps the historical single return value for the existing
        stage-level callers (the layer benchmark drives this helper directly).
        ``workspace`` (from _get_saved_workspace) supplies a fixed-address
        slice for that buffer; None keeps the per-step allocation.
        """
        packed_dim = dispatch_result.fc1_output.shape[1]
        if packed_dim <= 0 or packed_dim % 2:
            raise ValueError("FC1 output dimension must be a positive even value")
        num_recv = dispatch_result.routing_plan.num_received_routes
        if workspace is not None:
            weighted_activation = workspace["swiglu_out_weighted"][:num_recv]
            if weighted_activation.shape[1] != packed_dim // 2:
                raise ValueError(
                    "workspace swiglu buffer width "
                    f"{weighted_activation.shape[1]} != fc1 output half "
                    f"{packed_dim // 2}"
                )
        else:
            weighted_activation = torch.empty(
                (num_recv, packed_dim // 2),
                dtype=self.activation_dtype,
                device=dispatch_result.fc1_output.device,
            )
        num_received_routes, ffn_size = weighted_activation.shape
        plan = dispatch_result.routing_plan
        if replica_down_weight is None:
            replica_down_weight = down_weight
        if weighted_activation.device != self.context.peer_mem.device:
            raise ValueError(
                "weighted_activation must be on the operator's NPU device"
            )
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
        if (
            replica_down_weight.shape != expected_weight_shape
            or replica_down_weight.dtype != down_weight.dtype
            or replica_down_weight.device != down_weight.device
            or not replica_down_weight.is_contiguous()
        ):
            raise ValueError(
                "replica_down_weight must be a contiguous tensor matching "
                "down_weight shape, dtype, and device"
            )

        output = torch.empty(
            (plan.num_input_tokens, self.hidden_size),
            dtype=self.activation_dtype,
            device=weighted_activation.device,
        )

        block_n = self.config.fc2_gemm_block_size_n
        block_k = self.config.fc2_gemm_block_size_k
        if self.hidden_size % block_n or ffn_size % block_k:
            raise ValueError(
                "FC2 block_n and block_k must divide the output and reduction dimensions"
            )
        combine_metadata = self._prepare_combine_metadata(dispatch_result)
        # FC2 stages destination-local rows in symmetric GM. Device-put workers
        # move each source segment into send order before top-k reduction.
        fc2_workspace = self._combine_fc2_buf[:plan.num_sent_routes]
        pipeline_group_experts = combine_metadata["pipeline_group_experts"]
        reduce_block_n = _fc2_reduce_block_n(num_received_routes)
        active_experts_per_rank = plan.active_physical_experts_per_rank
        self._ensure_group_pipeline_runtime(
            pipeline_group_experts,
            active_experts_per_rank,
            None,
        )
        _launch_fc2_combine(
            weighted_activation,
            down_weight,
            replica_down_weight,
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
            plan.num_sent_routes,
            topk=self.top_k,
            num_program_cores=self.num_aicore_programs,
            num_vector_programs=self.num_aivector_programs,
            reduce_block_n=reduce_block_n,
            block_m=self.config.fc2_combine_block_size_m,
            block_n=block_n,
            block_k=block_k,
            world_size=self.world_size,
            home_experts_per_rank=self.experts_per_rank,
            physical_experts_per_rank=self.physical_experts_per_rank,
            active_experts_per_rank=active_experts_per_rank,
            pipeline_group_experts=pipeline_group_experts,
            pipeline_group_ids=None,
            pipeline_cube_stream=self._combine_pipeline_cube_stream,
            pipeline_vector_stream=self._combine_pipeline_vector_stream,
            pipeline_transfer_stream=self._combine_pipeline_transfer_stream,
            pipeline_group_events=self._combine_pipeline_group_events,
            pipeline_start_event=self._combine_pipeline_start_event,
            pipeline_done_event=self._combine_pipeline_done_event,
            prefetch_done_event=prefetch_done_event,
            pipeline_activation_events=self._combine_pipeline_activation_events,
            activation_fc1_output=dispatch_result.fc1_output,
            activation_routing_weights=dispatch_result.received_routing_weights,
            activation_id=0 if self.activation == "swiglu" else 1,
            activation_situ_beta=float(self.situ_beta),
            activation_situ_linear_beta=(
                float(self.situ_linear_beta)
                if self.situ_linear_beta is not None
                else 0.0
            ),
            activation_has_linear_beta=self.situ_linear_beta is not None,
        )
        if self._replica_prefetch_pending:
            # This flag also records that both symmetric tables hold a valid
            # published epoch for correctness/debug consumers.  Disabling the
            # cache skips only the next-step collective hit lookup above.
            self._replica_weight_cache_valid = True
            self._replica_prefetch_pending = False
        if return_weighted_activation:
            return output, weighted_activation
        return output

    def _forward_single_kernel(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the home-expert routing-to-combine implementation once."""
        self._ensure_single_kernel_buffers(gate_up_weight)
        num_tokens = hidden_states.shape[0]
        num_routes = selected_experts.numel()
        ffn_size = gate_up_weight.shape[2] // 2
        max_received_routes = (
            self.context.peer_mem.numel() // self.hidden_size
        )
        if max_received_routes >= torch.iinfo(torch.int32).max:
            raise ValueError("dispatch receive offsets exceed int32 range")
        gate_up_for_gemm = gate_up_weight.transpose(-1, -2)
        output = torch.empty(
            (num_tokens, self.hidden_size),
            dtype=self.activation_dtype,
            device=hidden_states.device,
        )
        signal_epoch = self._tile_signal_epoch
        launch_options = (
            {"limit_auto_multi_buffer_of_local_buffer": "no-l0c"}
            if max(
                self.config.fc1_gemm_block_size_m
                * self.config.fc1_gemm_block_size_n,
                self.config.fc2_combine_block_size_m
                * self.config.fc2_gemm_block_size_n,
            )
            > 128 * 256
            else {}
        )
        _kernel_fused_forward[self.num_aicore_programs, 1, 1](
            hidden_states,
            selected_experts,
            routing_weights,
            gate_up_for_gemm,
            down_weight,
            self.context.peer_mem,
            self.context.routing_weight_mem,
            self.context.signal_mem,
            self._combine_fc2_buf,
            self._single_fc1_output,
            self._single_weighted_activation,
            output,
            self.context.metadata_counts_mem,
            self.context.metadata_send_bucket_starts,
            self.context.metadata_send_bucket_dst_starts,
            self.context.metadata_recv_counts_re,
            self.context.metadata_recv_per_expert,
            self.context.metadata_recv_expert_offs,
            self.context.metadata_stats,
            self._single_core_bucket_cursors,
            self._single_send_token_indices,
            self._single_send_route_indices,
            self._route_to_send,
            self._pull_tile_rank,
            self._pull_tile_src_start,
            self._pull_tile_dst_start,
            self._pull_tile_row_count,
            num_routes,
            signal_epoch,
            float(self.situ_beta),
            (
                float(self.situ_linear_beta)
                if self.situ_linear_beta is not None
                else 0.0
            ),
            hidden_states.stride(0),
            hidden_states.stride(1),
            gate_up_for_gemm.stride(0),
            gate_up_for_gemm.stride(1),
            gate_up_for_gemm.stride(2),
            down_weight.stride(0),
            down_weight.stride(1),
            down_weight.stride(2),
            NUM_PROGRAM_CORES=self.num_aicore_programs,
            LOCAL_RANK=self.rank,
            WORLD_SIZE=self.world_size,
            NUM_EXPERTS=self.world_size * self.experts_per_rank,
            EXPERTS_PER_RANK=self.experts_per_rank,
            TOPK=self.top_k,
            HIDDEN=self.hidden_size,
            FFN=ffn_size,
            MAX_RECEIVED_ROUTES=max_received_routes,
            NUM_BINS_PAD=self.context.metadata_num_bins,
            MAX_SOURCE_TILES=self.context.max_source_tiles,
            DISPATCH_BLOCK_M=self.config.dispatch_fc1_block_size_m,
            FC1_BLOCK_M=self.config.fc1_gemm_block_size_m,
            FC1_BLOCK_N=self.config.fc1_gemm_block_size_n,
            FC1_BLOCK_K=self.config.fc1_gemm_block_size_k,
            FC2_BLOCK_M=self.config.fc2_combine_block_size_m,
            FC2_BLOCK_N=self.config.fc2_gemm_block_size_n,
            FC2_BLOCK_K=self.config.fc2_gemm_block_size_k,
            ACTIVATION=0 if self.activation == "swiglu" else 1,
            HAS_LINEAR_BETA=self.situ_linear_beta is not None,
            **launch_options,
        )
        self._tile_signal_epoch += 1
        self._routing_weights_keepalive = routing_weights
        if self.receive_capacity_factor < self.world_size:
            required_received_routes = int(
                self.context.metadata_stats[1].item()
            )
            if required_received_routes > max_received_routes:
                raise ValueError(
                    f"peer buffer capacity {max_received_routes} routes is "
                    "smaller than the required receive size "
                    f"{required_received_routes}"
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
        *,
        return_saved: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Run the complete BF16 post-routing MoE forward.

        The caller provides selected experts and FP32 routing weights. Router
        matmul, softmax, and top-k selection are outside this boundary.

        With ``return_saved=True`` the operator also returns the native
        ``saved`` dict for the fused backward as ``(output, saved)`` — the
        full home-layout contract (metadata / permutation / scalar /
        activation / weight-reference sections, see
        :mod:`mega_moe.ops._native_saved`), extended with the MoonEP
        physical ``[home | replica]`` sections when ``enable_moonep``.
        ``return_saved=False`` (default) keeps the previous behavior exactly.
        """
        # Preserve the public validation order before routing launches any work.
        self._validate_dispatch_inputs(hidden_states, selected_experts)
        self._validate_gate_up_weight(gate_up_weight, hidden_states.device)
        current_stream = torch.npu.current_stream(hidden_states.device)
        if self._forward_stream is None:
            self._forward_stream = current_stream
        elif self._forward_stream != current_stream:
            raise RuntimeError(
                "FusedMoEForward is single-stream; recreate the operator for "
                "a different caller stream"
            )
        if (
            routing_weights.shape != selected_experts.shape
            or routing_weights.dtype != torch.float32
            or routing_weights.device != hidden_states.device
            or not routing_weights.is_contiguous()
        ):
            raise ValueError(
                "routing_weights must be contiguous FP32 with the same shape "
                "and device as selected_experts"
            )
        # Framework callers hand down_proj as a transposed stride view of
        # their [E, F, H] table (MindSpeed-MM megamoe dispatcher); the FC2
        # kernels address raw memory and need a row-major [E, H, F].  Copy
        # once here so both callers work; the fresh allocation also keeps
        # the replica weight cache honestly miss-per-step, which matches
        # optimizer-updated weights.
        if (
            self.enable_single_kernel_forward
            and not return_saved
            and not down_weight.is_contiguous()
        ):
            raise ValueError(
                "single-kernel forward requires contiguous down_weight"
            )
        if not down_weight.is_contiguous():
            down_weight = down_weight.contiguous()
        expected_down_shape = (
            self.experts_per_rank,
            self.hidden_size,
            gate_up_weight.shape[2] // 2,
        )
        if (
            down_weight.ndim != 3
            or tuple(down_weight.shape) != expected_down_shape
            or down_weight.dtype != self.activation_dtype
            or down_weight.device != hidden_states.device
        ):
            raise ValueError(
                "down_weight must be BF16 with shape "
                f"{expected_down_shape} on the input device (a strided view "
                "is staged into the operator's persistent fc2 buffer)"
            )
        down_weight = self._materialize_down_weight(down_weight)
        ffn_size = gate_up_weight.shape[2] // 2
        if (
            gate_up_weight.shape[2] % self.config.fc1_gemm_block_size_n
            or self.hidden_size % self.config.fc1_gemm_block_size_k
            or self.hidden_size % self.config.fc2_gemm_block_size_n
            or ffn_size % self.config.fc2_gemm_block_size_k
        ):
            raise ValueError(
                "configured FC1/FC2 N and K tiles must divide the weight dimensions"
            )
        if self.enable_single_kernel_forward and not return_saved:
            return self._forward_single_kernel(
                hidden_states,
                selected_experts,
                gate_up_weight,
                down_weight,
                routing_weights,
            )
        # Symmetric replica allocations must be complete before any rank enters
        # the routing count exchange.  The planner then starts owner-push RMA
        # as soon as its ETC device copy is queued.
        self._ensure_replica_weight_buffers(gate_up_weight, down_weight)
        self._ensure_combine_buffers()
        if self.enable_moonep:
            def moonep_plan_hook(experts_to_copy_cpu, experts_to_copy_device):
                self._begin_replica_prefetch(
                    experts_to_copy_cpu,
                    experts_to_copy_device,
                    gate_up_weight,
                    down_weight,
                )
        else:
            moonep_plan_hook = None
        routing_plan = self.build_routing_plan(
            selected_experts,
            moonep_plan_hook=moonep_plan_hook,
        )
        # T1: snapshot the plan metadata right at the planning sync point,
        # before any later stage can rewrite the shared planning workspace.
        plan_snapshot = (
            snapshot_plan_metadata(self, routing_plan)
            if return_saved
            else None
        )
        if self.enable_moonep:
            replica_gate_up_weight, replica_down_weight = (
                self._finish_replica_prefetch()
            )
        else:
            replica_gate_up_weight, replica_down_weight = gate_up_weight, down_weight
        # Native-saved path: hand out fixed-address workspace slices for the
        # large activations (see _get_saved_workspace) instead of per-step
        # allocations.  MOE_SAVED_WORKSPACE=0 restores the historical
        # per-step torch.empty behavior.
        ws = (
            self._get_saved_workspace(hidden_states, gate_up_weight)
            if return_saved and os.environ.get("MOE_SAVED_WORKSPACE", "1") != "0"
            else None
        )
        dispatch_result = self.dispatch_fc1(
            hidden_states,
            selected_experts,
            routing_plan,
            gate_up_weight,
            fc1_output=(
                ws["fc1_output"][: routing_plan.num_received_routes]
                if ws is not None else None
            ),
            routing_weights=routing_weights,
            replica_fc1_weight=replica_gate_up_weight,
            replica_weight_ready=(
                self.context.replica_gate_ready
                if self.enable_moonep
                else None
            ),
            replica_weight_epoch=self._active_replica_weight_epoch,
            down_weight_to_prefetch=(
                down_weight
                if self._replica_prefetch_pending
                else None
            ),
            replica_down_weight_to_prefetch=(
                replica_down_weight
                if self._replica_prefetch_pending
                else None
            ),
            replica_down_ready=(
                self.context.replica_down_ready
                if self._replica_prefetch_pending
                else None
            ),
            final_barrier=False,
        )
        prefetch_done_event = self._fence_replica_prefetch_after_dispatch(
            hidden_states.device
        )
        # T2: snapshot the activations between dispatch completion and the FC2
        # launch — FC2's device-put staging overwrites the peer-memory receive
        # view, so this is the only window for the recv_hidden_sorted clone.
        activations = (
            capture_activations(self, dispatch_result, workspace=ws)
            if return_saved else None
        )
        # Produce expert-major activation groups on FC2's otherwise-idle
        # Vector stream while the Cube stream consumes earlier groups.
        combine_result = self._fc2_combine_shadow_activation(
            down_weight,
            dispatch_result,
            replica_down_weight=replica_down_weight,
            prefetch_done_event=prefetch_done_event,
            return_weighted_activation=return_saved,
            workspace=ws,
        )
        if not return_saved:
            return combine_result
        result, weighted_activation = combine_result
        # swiglu_out_weighted is shadowed inside the FC2 pipeline and joins the
        # saved dict through the helper's two-tuple return (same buffer the
        # pipeline fully writes before the combine completes).
        activations["swiglu_out_weighted"] = weighted_activation
        # T3: assemble the native saved dict.
        saved = assemble_native_saved(
            self,
            routing_plan,
            plan_snapshot,
            activations,
            hidden_states=hidden_states,
            gate_up_weight=gate_up_weight,
            down_weight=down_weight,
            selected_experts=selected_experts,
        )
        return result, saved


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
