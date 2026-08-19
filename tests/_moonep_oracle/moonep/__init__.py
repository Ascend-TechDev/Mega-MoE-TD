# coding=utf-8
"""moonep：MoonEP 的 torch 语义级参考实现包（昇腾路径）。

以顶层包形式独立导入：把 expert_parallel/ 目录加入 sys.path 后直接
``import moonep`` 即可，不依赖 mindspeed_mm 父包。

绑定级设计契约见 docs/design.md：kernel 类名、``__init__`` 常量配置、
``__call__`` 入参/出参、以及 ``launch_*`` 宿主启动函数签名与 GPU 源码
（source_code/MoonEP/moonep/）逐字一致；``launch_*`` 内部调用本包的
torch kernel 实现。

说明：

- ShmemStreamTransport 的类定义在 buffer.py 顶层，其构造器内才惰性 import
  外部依赖（shmem / torch_npu），因此本 __init__ 顶层导入该类不会触发
  shmem/torch_npu 导入；环境不适配（未装 shmem/torch_npu、进程组未初始化）
  会在构造时直接报错并附安装指引；
- 数值验证的单进程模拟器为测试专用件 tests/sim_transport.py（不在生产包面内）；
- 8 个 launch_* 中，launch_planning / launch_dispatch /
  launch_dispatch_epilogue / launch_combine_prologue / launch_combine 经
  ``ctx['arena']`` 取 arena，launch_prefetch / launch_grad_reduce（无 ctx）
  经 ``resolve_arena_for(对称张量)`` 按 data_ptr 在全局注册表反查，
  launch_inter_rank_sync(ctx) 内部调 ``cross_rank_barrier(ctx['arena'])``。
"""

from .api import Buffer
from .buffer import (
    ShmemStreamTransport,
    SymmetricArena,
    register_arena,
    resolve_arena_for,
    unregister_arena,
)
from .combine import CombineKernel, launch_combine
from .combine_prologue import CombinePrologueKernel, launch_combine_prologue
from .dispatch import DispatchKernel, launch_dispatch
from .dispatch_epilogue import DispatchEpilogueKernel, launch_dispatch_epilogue
from .grad_reduce import GradReduceKernel, launch_grad_reduce
# 契约目录结构里的 inter_rank_sync.py（跨 rank 同步原语 + launch 形态）
from .inter_rank_sync import (
    cross_rank_barrier,
    inter_rank_sync,
    launch_inter_rank_sync,
)
from .planning import (
    MoonEPCommPlan,
    PlanningKernel,
    allocate_planning_outputs,
    launch_planning,
    physical_tokens_per_expert,
)
from .prefetch import PrefetchKernel, launch_prefetch

__all__ = [
    # 顶层 API
    "Buffer",
    # 规划与通信计划
    "MoonEPCommPlan",
    "allocate_planning_outputs",
    "physical_tokens_per_expert",
    # 对称内存封装与传输层
    "SymmetricArena",
    "ShmemStreamTransport",
    "register_arena",
    "unregister_arena",
    "resolve_arena_for",
    # kernel 类（类名/__init__ 常量配置/__call__ 签名与 GPU 源码逐字一致）
    "PlanningKernel",
    "DispatchKernel",
    "DispatchEpilogueKernel",
    "CombinePrologueKernel",
    "CombineKernel",
    "PrefetchKernel",
    "GradReduceKernel",
    # launch_* 宿主启动函数（签名与 GPU 源码逐字一致）
    "launch_planning",
    "launch_dispatch",
    "launch_dispatch_epilogue",
    "launch_combine_prologue",
    "launch_combine",
    "launch_prefetch",
    "launch_grad_reduce",
    "launch_inter_rank_sync",
    # 跨 rank 同步原语
    "cross_rank_barrier",
    "inter_rank_sync",
]
