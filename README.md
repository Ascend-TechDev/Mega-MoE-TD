# MEGA-MOE-TD

MEGA-MOE-TD 提供基于
[Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)
的 Ascend NPU MoE 前向与反向实现。前向覆盖 post-routing 边界，反向保留现有五阶段
mega-kernel 编排。

```text
selected_experts + FP32 routing_weights
  -> routing metadata
  -> dispatch + FC1
  -> weighted SwiGLU
  -> FC2 + reverse all-to-all + route restore + top-k reduction
```

router matmul、softmax 和 top-k 选专家不属于当前前向计时边界。激活和专家权重使用
BF16，routing weight 在公开接口、通信和 weighted SwiGLU 中保持 FP32。

## 安装

运行环境有两种配置途径：

- 按照
  [Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)
  源仓库配置 CANN、Triton、`torch-npu` 和 ACLSHMEM。
- 使用已经集成 triton-dist 的 Triton-Ascend 软件包。

环境配置完成后，从本仓库根目录安装。硬件相关依赖不会由本仓库自动安装。

```bash
python -m pip install -e . --no-deps
```

安装后，无论当前工作目录在哪里都应使用 `mega_moe` 包名，不再导入 `src`、
`functions`、`kernels` 或 `benchmark`。

## 目录结构

```text
src/mega_moe/
├── config.py                 # 前向配置与调度参数校验
├── ops/
│   ├── forward.py            # 完整前向编排与输入校验
│   ├── backward.py           # 五阶段反向编排及 autograd.Function
│   └── _legacy_backward_golden.py
├── runtime/
│   ├── workspace.py          # ACLSHMEM 对称内存及 workspace 生命周期
│   └── routing.py            # 路由过滤、排序、计数交换和 offset
└── kernels/                  # Triton JIT kernels 及 launcher

tests/
├── function/                 # autograd Function 测试
├── layer/                    # 完整前向/反向流程测试
└── kernel/{forward,backward}/

benchmark/layer/              # 完整前向性能入口与结果汇总
results/forward/              # 正式 forward benchmark JSON
```

## 公开接口

```python
from mega_moe import (
    FusedMoEForward,
    MoEForwardConfig,
    MoERoutingPlan,
    DispatchFC1Result,
    MegaMoEBackwardFunction,
    moe_backward_triton,
    pack_gate_up_weights,
)
```

典型前向调用：

```python
config = MoEForwardConfig(receive_capacity_factor=4.0)
moe = FusedMoEForward(
    ep_group,
    max_tokens_per_rank=8192,
    hidden_size=7168,
    top_k=6,
    num_experts=384,
    config=config,
)

packed_fc1 = pack_gate_up_weights(gate_weight_local, up_weight_local)
output = moe(
    hidden_states,
    selected_experts,
    packed_fc1,
    down_weight_local,
    routing_weights,
)
```

同一个 `FusedMoEForward` 实例是 single-in-flight 的。最后一次调用后按相同 rank 顺序
同步并释放对称内存：

```python
moe.sync()
torch.distributed.barrier(group=ep_group)
moe.finalize()
torch.distributed.barrier(group=ep_group)
```

## 正确性测试

先运行不需要多卡的公开接口和配置测试：

```bash
python -m pytest tests/layer/test_moe_forward.py -m "not dist" -v
```

forward pytest fixture 会自行建立多进程环境，不要再套 `torchrun`：

```bash
python -m pytest \
  tests/layer/test_moe_forward.py::test_forward_2ranks \
  -m dist -v -s
```

backward 五阶段和 autograd 测试使用 `torchrun`：

```bash
# legacy torch golden：手写 backward 与 autograd 交叉验证
torchrun --nproc-per-node=2 -m mega_moe.ops._legacy_backward_golden

# 五阶段 Triton backward 与 torch golden
torchrun --nproc-per-node=2 tests/layer/test_moe_backward.py

# MegaMoEBackwardFunction autograd 接口
torchrun --nproc-per-node=2 tests/function/test_moe_backward_function.py
```

## 现有性能结果

### Backward

最后一列按已有数据计算为 `torch(ms) / triton(ms)`，大于 1 表示 Triton 更快。

| 模型 | tokens | torch/ms | triton/ms | Triton speedup |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 4096 | 45.1 | 28.7 | **1.57x** |
| Qwen3-30B-A3B | 8192 | 56.6 | 111.6 | 0.51x |
| Qwen3-30B-A3B | 16384 | 83.8 | 224.1 | 0.37x |
| DeepSeek-MoE-16B | 4096 | 27.9 | 29.8 | 0.94x |
| Qwen3-235B-A22B | 4096 | 63.4 | 79.6 | 0.80x |
| Qwen3-Next-80B | 4096 | 136.4 | 31.4 | **4.34x** |
| Qwen3-Omni-30B | 4096 | 37.7 | 12.4 | **3.04x** |
| **平均** | | | | **1.65x** |

### Forward

以下展示当前正式 DSV4 W8、8K tokens/rank、KN 配置的四阶段 median，数据来自
[`bench_full_forward_dsv4_grouped_routefp32_w8.json`](results/forward/bench_full_forward_dsv4_grouped_routefp32_w8.json)。
加速比为 `Grouped baseline / Ascend candidate`，大于 1 表示 Ascend candidate 更快。

| Stage | Ascend candidate/ms | Grouped baseline/ms | speedup |
|---|---:|---:|---:|
| preprocess | 3.906 | 11.266 | **2.884x** |
| dispatch+FC1 | 25.191 | 27.056 | **1.074x** |
| weighted SwiGLU | 1.705 | 6.592 | **3.866x** |
| FC2+combine | 65.610 | 39.905 | 0.608x |

四阶段来自各自独立的完整执行 event 样本，只用于定位瓶颈，不能相加还原完整
post-routing forward latency。其余 world size 和模型的完整 JSON 仍保存在
`results/forward/`，可通过 `summarize_results.py` 汇总。