"""Lazy public exports for the Ascend MoE backward Triton kernels."""

import importlib


_LAZY_EXPORTS = {
    "BLOCK_SIZE_M": (".common", "BLOCK_SIZE_M"),
    "BLOCK_SIZE_N": (".common", "BLOCK_SIZE_N"),
    "BLOCK_SIZE_K": (".common", "BLOCK_SIZE_K"),
    "WGRAD_BLOCK_M": (".common", "WGRAD_BLOCK_M"),
    "WGRAD_BLOCK_N": (".common", "WGRAD_BLOCK_N"),
    "WGRAD_BLOCK_K": (".common", "WGRAD_BLOCK_K"),
    "ncore": (".common", "ncore"),
    "all_gather_list": (".common", "all_gather_list"),
    "swiglu_bwd_triton": (".swiglu_bwd", "swiglu_bwd_triton"),
    "kernel_swiglu_bwd": (".swiglu_bwd", "kernel_swiglu_bwd"),
    "transposed_grouped_gemm_triton": (
        ".transposed_grouped_gemm",
        "transposed_grouped_gemm_triton",
    ),
    "kernel_transposed_grouped_gemm": (
        ".transposed_grouped_gemm",
        "kernel_transposed_grouped_gemm",
    ),
    "dispatch_fc2_bwd_triton": (".dispatch_fc2_bwd", "dispatch_fc2_bwd_triton"),
    "kernel_dispatch_fc2_bwd": (".dispatch_fc2_bwd", "kernel_dispatch_fc2_bwd"),
    "combine_fc1_bwd_triton": (".combine_fc1_bwd", "combine_fc1_bwd_triton"),
    "kernel_combine_fc1_bwd": (".combine_fc1_bwd", "kernel_combine_fc1_bwd"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
