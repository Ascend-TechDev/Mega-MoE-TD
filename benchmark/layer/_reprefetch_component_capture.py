"""Serial down/gate-up transfer diagnostics on real poisoned MoonEP tables."""

import os

import torch
import torch.distributed as dist


def capture_components(variant, output_dir, rank, samples):
    from benchmark.layer.bench_backward_bigop import prepare_call, write_json
    from mega_moe.kernels.common import ncore
    from mega_moe.kernels.mega_bwd import launch_replica_grad_barrier
    from mega_moe.kernels.replica_weight_prefetch import _kernel_replica_repush_udma

    case = variant.owner.case
    chunk = int(os.environ.get("MEGAMOE_UDMA_CHUNK_BYTES", 64 * 1024 * 1024)) // 2
    rows = []
    for iteration in range(samples + 1):
        prepare_call(variant)
        saved = variant.owner.native_saved
        epoch = saved["replica_buffers"].next_push_epoch()
        torch.npu.synchronize()
        dist.barrier()
        torch.npu.synchronize()
        events = [torch.npu.Event(enable_timing=True) for _ in range(4)]
        events[0].record()
        for component, mask in enumerate((1, 2)):
            _kernel_replica_repush_udma[(ncore(), 1, 1)](
                saved["fc1_combined"], saved["fc2"],
                saved["replica_gate_up"], saved["replica_down"],
                saved["replica_gate_ready"].view(torch.uint64),
                saved["replica_down_ready"].view(torch.uint64),
                saved["experts_to_copy"], epoch,
                LOCAL_RANK=rank, WORLD_SIZE=case.world_size,
                EPR=case.num_experts // case.world_size,
                GU_ELEMS=2 * case.hidden * case.ffn, GU_CHUNK=chunk,
                DN_ELEMS=case.hidden * case.ffn, DN_CHUNK=chunk,
                WAIT_COMPLETION=True, PANEL_MASK=mask)
            events[component + 1].record()
        launch_replica_grad_barrier(ncore())
        events[3].record()
        torch.npu.synchronize()
        if iteration:
            rows.append({
                "down_ms": events[0].elapsed_time(events[1]),
                "gate_up_ms": events[1].elapsed_time(events[2]),
                "publish_ms": events[2].elapsed_time(events[3]),
                "total_ms": events[0].elapsed_time(events[3]),
            })
    write_json(output_dir / f"reprefetch_components_rank{rank}.json", {
        "boundary": "real standalone down+quiet, then gate/up+quiet, then publication barrier",
        "notes": "diagnostic serial schedule; one warmup excluded; take per-sample rank MAX for each standalone component; do not sum different rank maxima",
        "samples": rows,
    })
