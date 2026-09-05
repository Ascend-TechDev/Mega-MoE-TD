"""Ascend NPU triton mega-kernels for MoE backward, one module per kernel.

Mirrors the GPU ``TritonDistFusedEpMoeFunction.backward`` 5-op split and reuses
the 06 tutorial's ``barrier_all`` + ``dl.symm_at`` symmetric-memory idiom:

  * :mod:`mega_moe.kernels.swiglu_bwd` step2: SwiGLU backward
  * :mod:`mega_moe.kernels.transposed_grouped_gemm` step3/5: weight-grad
  * :mod:`mega_moe.kernels.dispatch_fc2_bwd` step1: dispatch-A2A + fc2 input-grad
  * :mod:`mega_moe.kernels.combine_fc1_bwd` step4: input-grad + reverse-A2A + gate-grad
  * :mod:`mega_moe.kernels.mega_bwd` MOE_BWD_MEGA=1: steps 1-5 in ONE launch
    (in-kernel barrier_all phase chain)
"""

from .common import (  # noqa: F401
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    WGRAD_BLOCK_M,
    WGRAD_BLOCK_N,
    WGRAD_BLOCK_K,
    ncore,
    nvec,
    all_gather_list,
)
from .swiglu_bwd import swiglu_bwd_triton, kernel_swiglu_bwd  # noqa: F401
from .transposed_grouped_gemm import (  # noqa: F401
    transposed_grouped_gemm_triton,
    kernel_transposed_grouped_gemm,
)
from .dispatch_fc2_bwd import (  # noqa: F401
    dispatch_fc2_bwd_triton,
)
from .combine_fc1_bwd import (  # noqa: F401
    combine_fc1_bwd_triton,
)
from .mega_bwd import (  # noqa: F401
    mega_backward_triton,
    kernel_moe_backward_mega,
)
