#!/usr/bin/env python3
"""Compile the production dispatch+FC1 Triton kernel without taking an NPU.

This captures the compiler inputs and the first three IR stages used to audit
AscendNPU-IR/Triton-Ascend transformations.  It deliberately imports the
kernel from this checkout rather than copying the implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import mega_moe.kernels.dispatch_fc1 as dispatch_fc1_module
import triton
import triton._C.libtriton as libtriton_module
import triton.backends.ascend.compiler as ascend_compiler_module
import triton_dist
from triton._C.libtriton import ir
from triton._C.libtriton import distributed as distributed_ir_module
from triton._C.libtriton import ascend as ascend_module
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend import _apply_ascend_patch
from triton.backends.ascend.compiler import (
    NPUOptions,
    make_ttir,
    min_dot_size,
    ttir_to_linalg,
)
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource

from mega_moe.kernels.dispatch_fc1 import _kernel_dispatch_fc1


POINTER_SIGNATURE = {
    "input_ptr": "*bf16",
    "peer_mem_ptr": "*bf16",
    "routing_weight_ptr": "*fp32",
    "routing_weight_recv_ptr": "*fp32",
    "signal_mem_ptr": "*i32",
    "weight_ptr": "*bf16",
    "output_ptr": "*bf16",
    "send_src_idx_ptr": "*i32",
    "send_route_idx_ptr": "*i32",
    "send_bucket_dst_starts_ptr": "*i32",
    "send_bucket_starts_ptr": "*i32",
    "send_counts_re_ptr": "*i32",
    "recv_per_expert_ptr": "*i32",
    "recv_expert_offs_ptr": "*i32",
    "recv_counts_re_ptr": "*i32",
}

SCALAR_SIGNATURE = {
    "signal_epoch": "i32",
    "stride_input_m": "i64",
    "stride_input_k": "i64",
    "stride_weight_0": "i64",
    "stride_weight_1": "i64",
    "stride_weight_2": "i64",
    "stride_output_m": "i64",
    "stride_output_n": "i64",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _write_text(output_dir: Path, name: str, text: str) -> dict[str, object]:
    path = output_dir / name
    path.write_text(text, encoding="utf-8")
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arch", default="Ascend950PR_9579")
    parser.add_argument("--tokens-per-rank", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--num-program-cores", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--fc1-output", type=int, default=6144)
    parser.add_argument("--dispatch-block-m", type=int, default=128)
    parser.add_argument("--gemm-block-m", type=int, default=256)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--final-barrier", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_experts % args.world_size:
        raise ValueError("num-experts must be divisible by world-size")
    if not 0 <= args.local_rank < args.world_size:
        raise ValueError("local-rank must be in [0, world-size)")

    experts_per_rank = args.num_experts // args.world_size
    max_source_tiles = (
        args.tokens_per_rank * args.top_k + args.dispatch_block_m - 1
    ) // args.dispatch_block_m
    signature = {**POINTER_SIGNATURE, **SCALAR_SIGNATURE}
    constants = {
        "hidden": args.hidden,
        "N": args.fc1_output,
        "K": args.hidden,
        "NUM_PROGRAM_CORES": args.num_program_cores,
        "LOCAL_RANK": args.local_rank,
        "WORLD_SIZE": args.world_size,
        "EXPERTS_PER_RANK": experts_per_rank,
        "MAX_SOURCE_TILES": max_source_tiles,
        "FINAL_BARRIER": args.final_barrier,
        "DISPATCH_BLOCK_SIZE_M": args.dispatch_block_m,
        "GEMM_BLOCK_SIZE_M": args.gemm_block_m,
        "BLOCK_SIZE_N": args.block_n,
        "BLOCK_SIZE_K": args.block_k,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _apply_ascend_patch()
    context = ir.context()
    ir.load_dialects(context)
    distributed_ir_module.ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend_module.load_dialects(context)
    options = NPUOptions(
        arch=args.arch,
        compile_on_910_95=True,
        enable_dynamic_cv_pipeline=True,
    )
    source = ASTSource(_kernel_dispatch_fc1, signature, constants)
    raw_ttir = ast_to_ttir(
        _kernel_dispatch_fc1,
        source,
        context,
        options,
        {"min_dot_size": min_dot_size(None)},
        {},
    )
    raw_record = _write_text(args.output_dir, "01-raw.ttir.mlir", str(raw_ttir))

    metadata = {**options.__dict__}
    optimized_ttir = make_ttir(raw_ttir, metadata, options)
    optimized_record = _write_text(
        args.output_dir, "02-optimized.ttir.mlir", str(optimized_ttir)
    )
    linalg = ttir_to_linalg(optimized_ttir, metadata, options, named_ops=True)
    linalg_record = _write_text(args.output_dir, "03-linalg.mlir", str(linalg))

    manifest = {
        "schema": "mega_moe_dispatch_fc1_ir_dump/v1",
        "kernel": "mega_moe.kernels.dispatch_fc1._kernel_dispatch_fc1",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "triton_version": getattr(triton, "__version__", None),
            "triton_dist_version": getattr(triton_dist, "__version__", None),
        },
        "loaded_sources": {
            "dump_script": _file_record(Path(__file__)),
            "mega_dispatch_fc1": _file_record(dispatch_fc1_module.__file__),
            "ascend_compiler": _file_record(ascend_compiler_module.__file__),
            "libtriton": _file_record(libtriton_module.__file__),
            "triton_dist": _file_record(triton_dist.__file__),
        },
        "signature": signature,
        "constants": constants,
        "options": {
            "arch": args.arch,
            "compile_on_910_95": True,
            "enable_dynamic_cv_pipeline": True,
        },
        "artifacts": [raw_record, optimized_record, linalg_record],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
