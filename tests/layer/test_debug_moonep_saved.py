# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Differential probe: adapter single-kernel saved vs native multi-kernel saved.

Temporary debug tool (2026-09-22): stage 1 builds BOTH saved dicts from the
same inputs — the production single-kernel path (op.forward + adapter) and the
battle-tested NATIVE multi-kernel ``op.forward(..., return_saved=True)`` (the
exact saved type the mega backward was validated with) — and compares every
shared key.  Stage 2 runs the mega backward on each and localizes the
grad_fc1 error per home-expert slice.
"""

import os

import pytest
import torch
import torch.distributed as dist

from mega_moe import FusedMoEForward, MoEForwardConfig, pack_gate_up_weights
from mega_moe.ops import moe_backward_triton
from mega_moe.ops._single_saved_adapter import enrich_single_kernel_saved
from mega_moe.ops._torch_forward import moe_forward
from tests import _moe_testkit as kit
from tests._moe_baselines import (
    backward_torch_baseline,
    compare_backward_gradients,
    make_down_weights,
    make_gate_up_weights,
    make_routing_weights,
    prepare_inputs,
)
from mega_moe.runtime.device import device_str, resolve_local_device


def _per_expert_mismatch(tr_val, gold_val, tol):
    d = (tr_val.float() - gold_val.float()).abs()
    bad = (d > tol).view(tr_val.shape[0], -1).sum(1)
    return [int(v) for v in bad.cpu().tolist()]


def run_debug_probe(rank: int, world_size: int) -> None:
    os.environ["MOE_BWD_MEGA"] = "1"
    os.environ["MOE_SAVED_RECOMPUTE"] = "1"
    os.environ["MEGAMOE_REPLICA_POOL"] = "1"

    # Exact failing-test w2/step2 shapes and route skew (hot expert 16 =
    # rank1's home at W2, seeded random spread for the rest).
    tokens, hidden, ffn, topk, num_experts = 512, 512, 256, 4, 32
    situ_beta, situ_linear_beta = 4.0, 25.0
    epn = num_experts // world_size
    device = device_str(resolve_local_device(rank))
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    use_udma = os.environ.get("PROBE_UDMA", "1") == "1"
    with kit.aclshmem_session(
        rank, world_size, kit.get_ash_size_bytes(default_gb=2),
        enable_udma=use_udma,
    ):
        # Backward scratch FIRST (dl.symm_at offset-0 contract).
        peer_mem = kit.make_moonep_backward_peer_mem(
            tokens * topk * world_size, tokens * topk, hidden, dtype, rank,
            ep_group,
        )

        def make_op(single_kernel):
            return FusedMoEForward(
                ep_group,
                max_tokens_per_rank=tokens,
                hidden_size=hidden,
                top_k=topk,
                num_experts=num_experts,
                config=MoEForwardConfig(
                    receive_capacity_factor=float(world_size),
                    activation="situglu",
                    situ_beta=situ_beta,
                    situ_linear_beta=situ_linear_beta,
                    enable_single_kernel_forward=single_kernel,
                    enable_moonep=True,
                    fc1_gemm_block_size_m=256,
                    fc2_combine_block_size_m=256,
                ),
            )

        mode = os.environ.get("PROBE_MODE", "single")
        dump_path = os.environ.get(
            "PROBE_DUMP", "/root/.claude/jobs/b01bb115/tmp/probe_native.pt")
        op = make_op(mode == "single")   # single-kernel or native operator
        try:
            w_gate, w_up = make_gate_up_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device)
            packed_w1 = pack_gate_up_weights(w_gate, w_up)
            w2 = make_down_weights(
                num_experts, hidden, ffn, world_size, rank, dtype, device)
            hs, _ = prepare_inputs(
                tokens, hidden, num_experts, topk, dtype, device,
                seed=2303 + rank)
            rw = make_routing_weights(tokens, topk, device, seed=2304 + rank)
            torch.manual_seed(2305 + rank)
            dy = torch.randn(tokens, hidden, dtype=dtype, device=device)
            torch.manual_seed(9000 + rank * 10 + 4)
            routes = torch.randint(
                0, num_experts, (tokens, topk), dtype=torch.int64)
            routes[:, 0] = num_experts // world_size
            ei = routes.to(device=device, dtype=torch.int32).contiguous()

            # ---- eager golden (same recipe as the autograd case) ----
            with torch.no_grad():
                _, gs = moe_forward(
                    hs, rw, ei, w_gate, w_up, w2, ep_group, topk,
                    return_saved=True)
                gate = gs["gate"].float()
                up = gs["up"].float()
                situ_a = (
                    situ_beta * torch.tanh(gate / situ_beta)
                    * torch.sigmoid(gate)
                )
                up_v = situ_linear_beta * torch.tanh(
                    up / situ_linear_beta)
                gs["swiglu_out_weighted"] = (
                    situ_a * up_v * gs["recv_weights_sorted"].float()
                    .unsqueeze(-1)
                ).to(dtype)
                gs["activation"] = "situglu"
                gs["situ_beta"] = situ_beta
                gs["situ_linear_beta"] = situ_linear_beta
                golden = backward_torch_baseline(gs, dy)
            print(f"[probe r{rank}] golden-ready", flush=True)

            if mode == "single":
                # ---- adapter saved (production single-kernel path) ----
                with torch.no_grad():
                    output, saved_min = op.forward(
                        hs, ei, packed_w1, w2, rw, return_saved=True)
                    saved = enrich_single_kernel_saved(
                        op, saved_min, hidden_states=hs,
                        gate_up_weight=packed_w1, down_weight=w2)
            else:
                # ---- NATIVE multi-kernel saved (validated contract) ----
                with torch.no_grad():
                    native_out, native_saved = op.forward(
                        hs, ei, packed_w1, w2, rw, return_saved=True)
                print(f"[probe r{rank}] native-forward-done", flush=True)

            int_keys = [
                "expert_counts", "split_size_cum_per_expert",
                "num_tiles_total", "meta_expert_ids", "meta_split_cum",
                "meta_tile_num", "meta_tile_num_cum",
                "sort_idxs", "local_sort_idxs", "inv_local", "inv_sort",
                "plan_send_counts_by_rank_expert",
                "plan_send_bucket_starts",
                "plan_send_bucket_dst_starts",
                "plan_recv_counts_by_source_expert",
                "plan_received_expert_offsets",
                "experts_to_copy",
            ]
            float_keys = ["fc1_output", "recv_weights_sorted"]

            if mode == "native":
                pass  # dump moved after the backward (save-between crashed)

            if mode == "single" and os.path.exists(dump_path + f".r{rank}"):
                nat_dump = torch.load(dump_path + f".r{rank}")
                msgs = []
                for key in int_keys + float_keys:
                    if key not in saved or key not in nat_dump:
                        msgs.append(f"{key}: MISSING")
                        continue
                    a, b = saved[key].cpu(), nat_dump[key]
                    if a.shape != b.shape:
                        msgs.append(f"{key}: SHAPE {tuple(a.shape)} vs "
                                    f"{tuple(b.shape)}")
                        continue
                    if key in float_keys:
                        fdiff = (a.float() - b.float()).abs()
                        n = int((fdiff > 0).sum().item())
                        if n:
                            msgs.append(f"{key}: FDIFF n={n} "
                                        f"max={fdiff.max().item():.6f}")
                    else:
                        a64 = a.to(torch.int64)
                        b64 = b.to(torch.int64)
                        if not torch.equal(a64, b64):
                            diff = (a64 != b64).nonzero()
                            msgs.append(f"{key}: DIFF n={diff.shape[0]} "
                                        f"first={diff[:6].tolist()}")
                print(f"[probe r{rank}] saved-vs-native "
                      + ("ALL-MATCH" if not msgs else " | ".join(msgs)),
                      flush=True)

            if mode == "native":
                # ---- mega backward on the NATIVE saved ----
                print(f"[probe r{rank}] native-bwd-enter "
                      f"saved_keys={len(native_saved)}", flush=True)
                with torch.no_grad():
                    nat_res = moe_backward_triton(
                        native_saved, dy, peer_mem,
                        grad_transport=op
                        .lend_replica_weight_tables_for_grad(),
                        hidden_states=hs)
                print(f"[probe r{rank}] native-bwd-done", flush=True)
                try:
                    torch.save(
                        {k: native_saved[k].detach().cpu()
                         for k in int_keys + float_keys
                         if k in native_saved},
                        dump_path + f".r{rank}",
                    )
                    print(f"[probe r{rank}] native-dump-ok", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe r{rank}] native-dump-ERR {exc!r}",
                          flush=True)
                ok_n, _ = compare_backward_gradients(nat_res, golden)
                pe = ""
                if not ok_n:
                    for nm in ("grad_fc1_1", "grad_fc1_2"):
                        pe += f" {nm}/exp=" + str(_per_expert_mismatch(
                            nat_res[nm], golden[nm], 0.05))
                print(f"[probe r{rank}] bwd-native "
                      f"{'PASS' if ok_n else 'FAIL'} {pe}", flush=True)
            else:
                # ---- mega backward on the adapter saved ----
                with torch.no_grad():
                    triton_result = moe_backward_triton(
                        saved, dy, peer_mem,
                        grad_transport=op
                        .lend_replica_weight_tables_for_grad(
                            experts_to_copy_cpu=saved.get(
                                "experts_to_copy_cpu")),
                        hidden_states=hs)
                all_ok, _ = compare_backward_gradients(
                    triton_result, golden)
                pe = ""
                if not all_ok:
                    for nm in ("grad_fc1_1", "grad_fc1_2"):
                        pe += f" {nm}/exp=" + str(_per_expert_mismatch(
                            triton_result[nm], golden[nm], 0.05))
                print(f"[probe r{rank}] bwd-single "
                      f"{'PASS' if all_ok else 'FAIL'} {pe}", flush=True)
                if not all_ok and os.environ.get("PROBE_GRAD_DEBUG") == "1":
                    # mismatch geometry on the hot expert: grad_fc1_1[0] is
                    # [F, H]; element (f, h) == slot flat h*2F + f, so the
                    # h-histogram separates the GU chunk0/chunk1 regions and
                    # the magnitude histogram separates "missing
                    # contribution" (~|contrib|) from ulp-level reorder.
                    d = (triton_result["grad_fc1_1"].float()
                         - golden["grad_fc1_1"].float()).abs()
                    bad = d[0] > 0.05
                    fh = bad.nonzero()
                    h_hist = torch.histc(
                        fh[:, 1].float(), bins=8, min=0, max=512)
                    f_hist = torch.histc(
                        fh[:, 0].float(), bins=4, min=0, max=256)
                    dmag = d[0][bad]
                    print(f"[probe r{rank}] grad-debug n={int(bad.sum())} "
                          f"max={dmag.max().item():.4f} "
                          f"mean={dmag.mean().item():.4f} "
                          f"h_hist={[int(v) for v in h_hist.tolist()]} "
                          f"f_hist={[int(v) for v in f_hist.tolist()]} "
                          f"first_idx={fh[:6].tolist()}", flush=True)
        finally:
            op.finalize()
            kit.ash.aclshmem_free_tensor(peer_mem)


@pytest.mark.dist
@pytest.mark.functional
def test_debug_moonep_saved_probe_w2(dist_test):
    dist_test(run_debug_probe, world_size=2)
