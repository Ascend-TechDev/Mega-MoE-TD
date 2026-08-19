# coding=utf-8
"""MoonEP 参考实现的 vendor 副本 —— 仅作测试 oracle，勿在生产代码 import。

出处：
    /home/z00905891/MoonEP/MindSpeed-MM_MoonEP（moonep 分支）
    mindspeed_mm/fsdp/distributed/expert_parallel/moonep/
    commit 23d71348（2026-08 快照）

内容：`moonep/` 整包**字节级复制**（.py 文件与源仓 diff 为空；tests/ 子目录
只取 sim_transport / kernel_test_utils / generate_topk_routing 三个非测试
文件，conftest/pytest.ini/test_* 均不复制以免干扰本仓 pytest 收集）。
同步方式：源仓更新后重新 cp 并 `diff -rq` 核对，更新本文件头的 commit。

用途（moonep Triton 移植的分层验收，见 docs/plan 或 commit message）：
    - 纯函数 oracle：_phase_b_tables / _phase_c1_order / _phase_c2_rank /
      _phase_d_dedup —— Triton kernel 的 bit-exact 对拍标准答案
    - 全链 oracle：SimTransport 单进程 CPU 模拟 R rank 跑 PlanningKernel
      编排（build_world/run_ranks），与 NPU 上的 Triton planning 输出对拍
    - 不变量检查：planning_invariant_errors
    - prefetch oracle：launch_prefetch（M2 用）

CPU 纯净性：全部模块顶层只 import torch（及标准库）；ShmemStreamTransport
（NPU 生产后端）的 shmem/torch_npu 依赖在构造器内惰性 import，本副本永远
不会触发。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    # 让 `import moonep` 解析到本目录下的 vendor 副本（kernel_test_utils 的
    # 两级向上 sys.path 注入同样落到这里，无需改动其代码）。
    sys.path.insert(0, _HERE)

from moonep.planning import (  # noqa: E402
    MoonEPCommPlan,
    PlanningKernel,
    _phase_b_tables,
    _phase_c1_order,
    _phase_c2_rank,
    _phase_d_dedup,
    allocate_planning_outputs,
    launch_planning,
)
from moonep.prefetch import launch_prefetch as launch_prefetch_oracle  # noqa: E402

__all__ = [
    "MoonEPCommPlan",
    "PlanningKernel",
    "_phase_b_tables",
    "_phase_c1_order",
    "_phase_c2_rank",
    "_phase_d_dedup",
    "allocate_planning_outputs",
    "launch_planning",
    "launch_prefetch_oracle",
]

# kernel_test_utils / sim_transport 里的公开设施（build_world / run_ranks /
# KernelCase / make_topk / planning_invariant_errors）按需从
# moonep.tests.kernel_test_utils 导入——不在顶层 re-export，避免把 pytest
# 依赖强加给只想用纯函数 oracle 的调用方。
