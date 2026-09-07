# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Compile the selected single-kernel forward without allocating on NPU."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from config import resolve_case
from mega_moe.kernels.fused_forward import _kernel_fused_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="performance-fwd-kimi-k3-trimmed-w8-t4k")
    parser.add_argument("--arch", default="Ascend950DT_9582")
    parser.add_argument("--cores", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dump-sync-ir", action="store_true")
    parser.add_argument("--fc1-block", type=int, nargs=3, default=(256, 256, 256),
                        metavar=("M", "N", "K"))
    parser.add_argument("--fc2-block", type=int, nargs=3, default=(256, 256, 256),
                        metavar=("M", "N", "K"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    case = resolve_case(args.case).validate()
    if not 0 <= args.rank < case.world_size:
        parser.error("rank must be in the selected case's world")
    if args.cores < case.world_size:
        parser.error("the pipeline requires at least one core per rank")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be new or empty")
    h, f = case.hidden, case.ffn
    fc1_m, fc1_n, fc1_k = args.fc1_block
    fc2_m, fc2_n, fc2_k = args.fc2_block
    if fc1_m != fc2_m:
        parser.error("dynamic waves require matching FC1 and FC2 M tiles")
    for value in (*args.fc1_block, *args.fc2_block):
        if value < 16 or value & (value - 1):
            parser.error("GEMM tile dimensions must be powers of two of at least 16")
    if max(fc1_m, fc2_m) > 256 or max(fc1_m * fc1_n, fc2_m * fc2_n) > 65536:
        parser.error("GEMM tiles exceed the operator accumulator limits")
    if h % fc1_k or f % fc2_k or h % fc2_n or (2 * f) % fc1_n:
        parser.error("GEMM N/K tiles must divide the weight dimensions")
    max_recv = math.ceil(case.tokens * case.topk * case.capacity_factor)
    constants = dict(
        stride_hidden_m=h, stride_hidden_k=1,
        stride_gate_up_e=h * 2 * f, stride_gate_up_n=1, stride_gate_up_k=2 * f,
        stride_down_e=h * f, stride_down_n=f, stride_down_k=1,
        NUM_PROGRAM_CORES=args.cores, LOCAL_RANK=args.rank,
        WORLD_SIZE=case.world_size, NUM_EXPERTS=case.num_experts,
        EXPERTS_PER_RANK=case.experts_per_rank, TOPK=case.topk,
        HIDDEN=h, FFN=f, MAX_RECEIVED_ROUTES=max_recv,
        NUM_BINS_PAD=triton.next_power_of_2(case.num_experts),
        MAX_SOURCE_TILES=triton.cdiv(case.tokens * case.topk, 128),
        MAX_PIPELINE_GROUPS=triton.cdiv(
            max_recv + case.experts_per_rank * (fc1_m - 1), 16 * fc1_m),
        DISPATCH_BLOCK_M=128, FC1_BLOCK_M=fc1_m, FC1_BLOCK_N=fc1_n,
        FC1_BLOCK_K=fc1_k, FC2_BLOCK_N=fc2_n, FC2_BLOCK_K=fc2_k,
        ACTIVATION=0, HAS_LINEAR_BETA=False, PIPELINE_GROUP_WINDOWS=16,
    )
    bf16_inputs = {
        "hidden_states_ptr", "gate_up_weight_ptr", "down_weight_ptr",
        "peer_mem_ptr", "combine_buf_ptr", "fc2_output_ptr",
        "weighted_activation_ptr", "output_ptr",
    }
    fp32_inputs = {"routing_weights_ptr", "routing_weight_recv_ptr"}
    signature = {}
    for name in _kernel_fused_forward.arg_names:
        if name in constants:
            signature[name] = "constexpr"
        elif name in bf16_inputs:
            signature[name] = "*bf16"
        elif name in fp32_inputs:
            signature[name] = "*fp32"
        elif name.endswith("_ptr"):
            signature[name] = "*i32"
        elif name in ("situ_beta", "situ_linear_beta"):
            signature[name] = "fp32"
        else:
            signature[name] = "i32"
    source = ASTSource(_kernel_fused_forward, signature, constants)
    options = dict(
        has_auto_blockify_blacklist_op=False,
        enable_dynamic_cv_pipeline=False, enable_mixed_cv=True,
        disable_auto_inject_block_sync=True, set_workspace_multibuffer=0,
        limit_auto_multi_buffer_buffer="only-cube",
    )
    if args.dump_sync_ir:
        options["debug"] = True
    if max(fc1_m * fc1_n, fc2_m * fc2_n) > 128 * 256:
        options["limit_auto_multi_buffer_of_local_buffer"] = "no-l0c"
    print(f"[compile-only] {args.case} selected dynamic pipeline", flush=True)
    kernel = triton.compile(source, target=GPUTarget("npu", args.arch, 0), options=options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension, assembly in kernel.asm.items():
        if isinstance(assembly, str):
            (args.output_dir / f"kernel.{extension}").write_text(assembly)
    if args.dump_sync_ir:
        from triton.runtime.cache import get_dump_manager

        sync_ir = get_dump_manager(kernel.hash).get_file("kernel.npuir.mlir")
        if not sync_ir:
            raise RuntimeError("backend did not save its synchronization IR")
        shutil.copyfile(sync_ir, args.output_dir / "kernel.npuir.mlir")
    result = dict(
        case_id=case.case_id, kernel_hash=kernel.hash, constants=constants,
        binary_bytes=len(kernel.kernel), metadata=kernel.metadata._asdict(),
        device_execution=False,
    )
    (args.output_dir / "compile_result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n")
    print(f"[compiled] {kernel.hash} ({len(kernel.kernel)} bytes)", flush=True)


if __name__ == "__main__":
    main()
