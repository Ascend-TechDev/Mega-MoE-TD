"""Ascend NPU triton mega-kernels for MoE backward, one module per kernel.

Mirrors the GPU ``TritonDistFusedEpMoeFunction.backward`` 5-op split and reuses
the 06 tutorial's ``barrier_all`` + ``dl.symm_at`` symmetric-memory idiom:

  * :mod:`benchmark.kernel.swiglu_bwd`            step2: SwiGLU backward
  * :mod:`benchmark.kernel.transposed_grouped_gemm` step3/5: weight-grad (fc2/fc1)
  * :mod:`benchmark.kernel.dispatch_fc2_bwd`       step1: dispatch-A2A + fc2 input-grad
  * :mod:`benchmark.kernel.combine_fc1_bwd`        step4: fc1 input-grad + reverse-A2A + gate-grad
"""

from .common import (  # noqa: F401
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    WGRAD_BLOCK_M,
    WGRAD_BLOCK_N,
    WGRAD_BLOCK_K,
    ncore,
    all_gather_list,
)
from .swiglu_bwd import swiglu_bwd_triton, kernel_swiglu_bwd  # noqa: F401
from .transposed_grouped_gemm import (  # noqa: F401
    transposed_grouped_gemm_triton,
    kernel_transposed_grouped_gemm,
)
from .dispatch_fc2_bwd import (  # noqa: F401
    dispatch_fc2_bwd_triton,
    kernel_dispatch_fc2_bwd,
)
from .combine_fc1_bwd import (  # noqa: F401
    combine_fc1_bwd_triton,
    kernel_fc1_input_grad_gemm,
    kernel_combine_push_reduce,
)
