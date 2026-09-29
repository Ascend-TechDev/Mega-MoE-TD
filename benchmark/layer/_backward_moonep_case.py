"""Same-input single-kernel MoonEP fixture, with real backward reprefetch.

Forward and saved-contract construction stay outside backward timing. The
replica tables are overwritten after forward: finite gradients then require
backward to restore the weights, as under pooled layer reuse.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from benchmark.layer._kimi_routes import kimi_skewed_routes
from benchmark.layer.bench_moe_suite import _make_local_weights
from mega_moe import FusedMoEForward, MoEForwardConfig, moe_backward_triton
from mega_moe._goldens._torch_forward_for_backward import moe_forward
from mega_moe.ops._single_saved_adapter import enrich_single_kernel_saved
from mega_moe.ops.function import _MEGA_PERSISTENT_KEYS
from tests import _moe_testkit as kit


class ReprefetchVariant:
    def __init__(self, owner, transport, safe_wgrad=None, fine_sync=None,
                 recompute=True, preloaded=False, trace_kernel=False):
        self.owner, self.transport = owner, transport
        self.fused = (transport.startswith("udma_fused") or (
            transport == "udma"
            and os.environ.get("MOE_MEGA_REPREFETCH_FUSED", "1") == "1"))
        self.pipeline = (self.fused and transport != "udma_fused_barrier"
                         and (transport == "udma_fused_pipeline" or
                              os.environ.get("MOE_MEGA_REPREFETCH_PIPELINE", "0") == "1"))
        self.safe_wgrad = (os.environ.get("MOE_MEGA_SAFE_WGRAD", "0") == "1"
                           if safe_wgrad is None else safe_wgrad)
        self.fine_sync = self.pipeline and (
            os.environ.get("MOE_MEGA_REPREFETCH_FINE_SYNC", "0") == "1"
            if fine_sync is None else fine_sync)
        self.early_kernel_hash = None
        self.recompute = recompute
        self.preloaded = preloaded
        self.trace_kernel = trace_kernel or preloaded
        self.kernel_hash = None
        if preloaded and not self.pipeline:
            raise ValueError("preloaded overlap control requires the fused pipeline")

    def prepare(self):
        os.environ["MOE_SAVED_RECOMPUTE"] = "1" if self.recompute else "0"
        self.owner.prepare()
        if self.preloaded:
            # An ideal-residency control, not a production bypass: poison as
            # usual, perform REAL UDMA refresh+quiet+publication outside the
            # measured backward, then retain every in-kernel consumer wait.
            from mega_moe.kernels.common import ncore
            from mega_moe.kernels.mega_bwd import launch_replica_grad_barrier
            from mega_moe.kernels.replica_weight_prefetch import _kernel_replica_repush_udma
            saved = self.owner.native_saved
            self.preloaded_epoch = saved["replica_buffers"].next_push_epoch()
            chunk = int(os.environ.get("MEGAMOE_UDMA_CHUNK_BYTES", 64 * 1024 * 1024)) // 2
            h, f = self.owner.case.hidden, self.owner.case.ffn
            _kernel_replica_repush_udma[(ncore(), 1, 1)](
                saved["fc1_combined"], saved["fc2"],
                saved["replica_gate_up"], saved["replica_down"],
                saved["replica_gate_ready"].view(torch.uint64),
                saved["replica_down_ready"].view(torch.uint64),
                saved["experts_to_copy"], self.preloaded_epoch,
                LOCAL_RANK=self.owner.rank, WORLD_SIZE=self.owner.case.world_size,
                EPR=self.owner.case.num_experts // self.owner.case.world_size,
                GU_ELEMS=2 * h * f, GU_CHUNK=chunk,
                DN_ELEMS=h * f, DN_CHUNK=chunk)
            launch_replica_grad_barrier(ncore())
            torch.npu.synchronize()
            dist.barrier()
        if not self.recompute:
            # Only used by --check-saved-kernel's untimed correctness gate.
            # Materialize exactly the two caches the ordinary kernel needs.
            from mega_moe._goldens.bigop_ref import _redispatch_hidden
            saved = self.owner.native_saved
            saved["recv_hidden_sorted"] = _redispatch_hidden(self.owner.hidden, saved)
            ab = saved["fc1_output"]
            ffn = self.owner.case.ffn
            weighted = torch.empty((ab.shape[0], ffn), dtype=torch.bfloat16,
                                   device=ab.device)
            for start in range(0, ab.shape[0], 2048):
                values = ab[start:start + 2048].float()
                if ab.dtype == torch.float8_e4m3fn:
                    group = saved["fc1_scale_group_size"]
                    values = (values.view(values.shape[0], -1, group)
                              * saved["fc1_output_scale"][start:start + 2048, :, None]).flatten(1)
                gate, up = values.chunk(2, dim=1)
                weighted[start:start + 2048] = (
                    torch.nn.functional.silu(gate) * up
                    * saved["recv_weights_sorted"][start:start + 2048, None])
            saved["swiglu_out_weighted"] = weighted

    def __call__(self):
        os.environ["MOE_MEGA_REPREFETCH_TRANSPORT"] = (
            "udma" if self.transport.startswith("udma") else self.transport)
        os.environ["MOE_MEGA_REPREFETCH_FUSED"] = "1" if self.fused else "0"
        os.environ["MOE_MEGA_REPREFETCH_PIPELINE"] = "1" if self.pipeline else "0"
        os.environ["MOE_MEGA_REPREFETCH_FINE_SYNC"] = "1" if self.fine_sync else "0"
        os.environ["MOE_SAVED_RECOMPUTE"] = "1" if self.recompute else "0"
        os.environ["MOE_MEGA_SAFE_WGRAD"] = "1" if self.safe_wgrad else "0"
        if not self.trace_kernel:
            return self.owner()
        from mega_moe.kernels import mega_bwd
        name = ("kernel_moe_backward_mega_recompute" if self.recompute
                else "kernel_moe_backward_mega")
        original = getattr(mega_bwd, name)
        variant = self

        class Launch:
            def __getitem__(self, grid):
                launch = original[grid]

                def measured(*args, **kwargs):
                    if variant.preloaded:
                        # Both scalar args are do_not_specialize: the same
                        # compiled pipeline consumes the real preload epoch,
                        # while zero outgoing descriptors suppress issuance.
                        kwargs["repref_desc_count"] = 0
                        kwargs["repref_epoch"] = variant.preloaded_epoch
                    kernel = launch(*args, **kwargs)
                    variant.kernel_hash = getattr(kernel, "hash", None)
                    return kernel
                return measured

        original_early = mega_bwd._kernel_replica_repush_udma

        class EarlyLaunch:
            def __getitem__(self, grid):
                launch = original_early[grid]

                def measured(*args, **kwargs):
                    if variant.preloaded:
                        kwargs["submit"] = 0
                    kernel = launch(*args, **kwargs)
                    variant.early_kernel_hash = getattr(kernel, "hash", None)
                    return kernel
                return measured

        setattr(mega_bwd, name, Launch())
        if self.fine_sync:
            mega_bwd._kernel_replica_repush_udma = EarlyLaunch()
        try:
            return self.owner()
        finally:
            setattr(mega_bwd, name, original)
            mega_bwd._kernel_replica_repush_udma = original_early


def add_unused_replica_slots(original):
    """Extend existing flows without adding duplicate-expert P6 ordinals."""
    etc = original.clone()
    epn = etc.shape[1]
    used = set(etc[etc >= 0].tolist())
    for peer in range(etc.shape[0]):
        assigned = etc[peer][etc[peer] >= 0].tolist()
        if not assigned:
            continue
        owners = {expert // epn for expert in assigned}
        if len(owners) != 1:
            raise ValueError("unused-slot probe requires one existing owner per receiver")
        free = torch.nonzero(etc[peer] < 0).flatten().tolist()
        owner = next(iter(owners))
        available = [e for e in range(owner * epn, (owner + 1) * epn) if e not in used]
        if not free or not available:
            continue
        expert = available[0]
        etc[peer, free[0]] = expert
        used.add(expert)
    if torch.equal(etc, original):
        raise ValueError("no unused expert/slot remains on existing owner-peer flows")
    return etc


class SingleKernelMoonEPCase:
    def __init__(self, case, rank, output_dir, *, add_unused_replica=False):
        if case.world_size != 8:
            raise ValueError("single-kernel MoonEP benchmark requires W8")
        self.rank, self.case = rank, case
        self.add_unused_replica = add_unused_replica
        self.output_dir = Path(output_dir)
        device = f"npu:{rank}"
        self.ep_group = dist.group.WORLD
        self.packed, self.down, _ = _make_local_weights(
            case, case.num_experts // case.world_size, rank, device,
        )
        torch.manual_seed(43 + rank * 1000)
        self.hidden = torch.randn(case.tokens, case.hidden, dtype=torch.bfloat16, device=device)
        self.gates = torch.softmax(torch.randn(case.tokens, case.topk, device=device), -1)
        self.routes = kimi_skewed_routes(case.tokens, case.num_experts, case.topk, rank).to(device)
        torch.manual_seed(44 + rank * 1000)
        self.dy = torch.randn_like(self.hidden)
        self.logical_output, self.saved = moe_forward(
            self.hidden, self.gates, self.routes,
            self.packed[:, :, :case.ffn].transpose(1, 2),
            self.packed[:, :, case.ffn:].transpose(1, 2),
            self.down, self.ep_group, case.topk, return_saved=True,
        )
        # First symmetric allocation; preserve receive slack for masked wgrad
        # loads and the all-source layout required by the production adapter.
        self.peer_mem = kit.make_moonep_backward_peer_mem(
            case.tokens * case.topk * case.world_size, case.tokens * case.topk,
            case.hidden, self.hidden.dtype, rank, self.ep_group,
        )
        self.op = FusedMoEForward(
            self.ep_group, max_tokens_per_rank=case.tokens,
            hidden_size=case.hidden, top_k=case.topk, num_experts=case.num_experts,
            config=MoEForwardConfig(
                receive_capacity_factor=case.capacity_factor,
                enable_moonep=True, enable_single_kernel_forward=True,
            ),
        )
        self.signal_mem, self.epoch, self.persistent = None, 0, {}
        self.prepared = self.forward_gate_done = False

    def variant(self, transport, safe_wgrad=None, fine_sync=None,
                recompute=True, preloaded=False, trace_kernel=False):
        return ReprefetchVariant(
            self, transport, safe_wgrad=safe_wgrad, fine_sync=fine_sync,
            recompute=recompute, preloaded=preloaded, trace_kernel=trace_kernel)

    def prepare(self):
        from benchmark.layer.bench_backward_bigop import compare_gradient

        output, saved = self.op.forward(
            self.hidden, self.routes, self.packed, self.down, self.gates,
            return_saved=True,
        )
        saved = enrich_single_kernel_saved(
            self.op, saved, hidden_states=self.hidden,
            gate_up_weight=self.packed, down_weight=self.down,
        )
        if self.add_unused_replica:
            # Diagnostic plan extension: no routes change. Each existing
            # receiver gets a zero-token slot from the SAME owner. Do not
            # invent new owner-peer flows: those change P6 credit topology.
            # The unused slot must still be acquired and drained at exit.
            original_etc = saved["experts_to_copy_cpu"]
            etc = add_unused_replica_slots(original_etc)
            epn = self.case.num_experts // self.case.world_size
            for slot in torch.nonzero(etc[self.rank] != original_etc[self.rank]).flatten().tolist():
                if saved["expert_counts"][epn + slot].item() != 0:
                    raise AssertionError("unused-replica probe would change an occupied slot")
            saved["experts_to_copy_cpu"] = etc
            saved["experts_to_copy"] = etc.to(self.hidden.device)
            saved["active_physical_experts_per_rank"] = (
                epn + int(torch.nonzero(etc >= 0)[:, 1].max()) + 1)
        if (int(saved["topk"]) != self.case.topk
                or tuple(saved["selected_experts"].shape)
                != (self.case.tokens, self.case.topk)):
            raise AssertionError(
                "forward/backward Top-K contract diverged: "
                f"case={self.case.topk}, saved={saved['topk']}, "
                f"selected_experts={tuple(saved['selected_experts'].shape)}"
            )
        if saved["total_send"] != self.case.tokens * self.case.topk:
            raise AssertionError("MoonEP forward dropped routes")
        if not self.forward_gate_done:
            forward_gate = compare_gradient(output, self.logical_output)
            replicas = int((saved["experts_to_copy_cpu"] >= 0).sum())
            report = {
                "forward_gate": forward_gate, "global_replica_slots": replicas,
                "experts_to_copy": saved["experts_to_copy_cpu"].tolist(),
                "physical_counts": saved["expert_counts"].cpu().tolist(),
                "total_recv": saved["total_recv"],
                "topk": int(saved["topk"]),
                "routes_shape": list(self.routes.shape),
                "selected_experts_shape": list(saved["selected_experts"].shape),
                "topk_consistent": (
                    int(saved["topk"]) == self.case.topk
                    and tuple(saved["selected_experts"].shape)
                    == (self.case.tokens, self.case.topk)
                ),
                "routing_input_dtype": str(self.gates.dtype),
                "same_logical_inputs": True, "single_kernel_forward": True,
                "replica_tables_overwritten_before_backward": True,
                "injected_unused_replica": self.add_unused_replica,
                "unused_assigned_slots": [
                    slot for slot, expert in enumerate(saved["experts_to_copy_cpu"][self.rank].tolist())
                    if expert >= 0 and saved["expert_counts"][self.case.num_experts // self.case.world_size + slot].item() == 0
                ],
                "global_reprefetch_payload_bytes": replicas * 3 * self.case.hidden * self.case.ffn * 2,
            }
            (self.output_dir / f"moonep_rank{self.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
            ok = torch.tensor([int(forward_gate["passed"] and replicas > 0)], device=self.hidden.device, dtype=torch.int32)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not ok.item():
                raise AssertionError("MoonEP same-input forward/replica gate failed")
            self.forward_gate_done = True
        # Every rank must finish forward before shared tables are overwritten.
        # This setup stays outside both implementations' timed boundaries.
        torch.npu.synchronize()
        dist.barrier()
        buffers = saved["replica_buffers"]
        buffers.gate_up.fill_(float("nan"))
        buffers.down.fill_(float("nan"))
        torch.npu.synchronize()
        dist.barrier()
        self.native_saved, self.prepared = saved, True

    def __call__(self):
        if not self.prepared:
            raise RuntimeError("prepare a fresh forward before each MoonEP backward")
        self.prepared = False
        saved = self.native_saved
        if self.signal_mem is not None:
            saved["_bwd_tile_signal_mem"] = self.signal_mem
        saved["_bwd_tile_signal_epoch"] = max(int(self.epoch), 1)
        saved.update(self.persistent)
        transport = self.op.lend_replica_weight_tables_for_grad(
            experts_to_copy_cpu=saved["experts_to_copy_cpu"])
        grads = moe_backward_triton(saved, self.dy, self.peer_mem,
                                    grad_transport=transport, hidden_states=self.hidden)
        self.signal_mem = saved.get("_bwd_tile_signal_mem")
        self.epoch = saved.get("_bwd_tile_signal_epoch", 1)
        self.persistent = {key: saved[key] for key in _MEGA_PERSISTENT_KEYS if key in saved}
        if not transport.sunk or not transport.reduced:
            raise AssertionError("replica gradient transport did not complete")
        return grads

    def close(self):
        self.op.finalize()
