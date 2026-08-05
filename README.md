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

当前 forward 以 Kimi-K3 为首要性能目标，主 shape 为
`H=3584, F=3072, top_k=16, E=896`；同时保留 Qwen 和 DeepSeek-V4（DSV4）
profile，用于跨 shape 回归和调度 A/B。

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
config = MoEForwardConfig(receive_capacity_factor=1.25)
moe = FusedMoEForward(
    ep_group,
    max_tokens_per_rank=8192,
    hidden_size=3584,
    top_k=16,
    num_experts=896,
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

## Forward profile 与默认调度

`benchmark/layer/bench_full_forward.py` 当前包含以下 tokens/rank workload：

| Profile | H | F | top-k | Experts | tokens/rank |
|---|---:|---:|---:|---:|---|
| Kimi-K3 | 3584 | 3072 | 16 | 896 | 4K、8K、16K |
| Qwen | 2048 | 768 | 8 | 128 | 2K、8K、16K、32K |
| DSV4 routed BF16 | 7168 | 3072 | 6 | 384 | 2K、4K、8K、32K、128K |

当前 `MoEForwardConfig` 默认使用 24 个 AI Core program、tile readiness、
`allcore_expert_n_tile` dispatch+FC1、`expert_n_persistent` FC2 和 `direct_pull`
combine；FC1 tile 为 M/N/K=`128/256/128`，FC2 tile 为 `128/256/128`，本地
top-k reduction 使用两个 Vector worker。`tile_n_major`、`reverse_push` 及其他
dispatch schedule 仍保留用于同版本 A/B。

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

## Forward 性能测试

性能入口由 pytest fixture 创建多进程，不要再套 `torchrun`。例如 Kimi-K3 W8：

```bash
source /path/to/Triton-distributed-ascend/run.sh
export MOE_FULL_BENCH_CONFIG=kimi_k3_4k,kimi_k3_8k,kimi_k3_16k
export MOE_FUSED_ASH_SIZE_GB=5
python -m pytest -p tests.conftest \
  benchmark/layer/bench_full_forward.py::test_bench_full_forward_kimi_k3_8ranks \
  -m dist -v -s
```

另有 `test_bench_full_forward_kimi_k3_4ranks` 入口。可通过
`MOE_FUSED_FC2_GEMM_SCHEDULE`、`MOE_FUSED_FC2_COMBINE_TRANSPORT`、
`MOE_FUSED_FC2_REVERSE_VECTOR_WORKERS` 和 `MOE_FUSED_FC2_REDUCE_VECTOR_WORKERS`
切换 FC2/combine A/B 配置。正式输出写入 `results/forward/`。

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

#### Kimi-K3 W8

Kimi-K3 W8 测试已完成，全部正确性 gate 通过，未发生 OOM。完整 post-routing
forward 的加速比为 `Grouped golden / Ascend candidate`：

| tokens/rank | Ascend full | Grouped golden | 加速比 | 观测 HBM/卡 |
|---:|---:|---:|---:|---:|
| 4K | 42.928 ms | 81.095 ms | **1.889x** | ≈22.4 GiB |
| 8K | 74.329 ms | 150.165 ms | **2.020x** | ≈35.1 GiB |
| 16K | 137.703 ms | 302.747 ms | **2.199x** | ≈53.5 GiB |

- 8K、16K full forward 已达到 2x。
- 4K 尚差约 2.38 ms 才达到 2x。
- 16K 进程显存约 50 GiB，总 HBM 约 53.5 GiB，仍有约 12 GiB 余量。
- 16K 的 ASH 理论需求约 4.048 GiB，默认 4 GiB 不够，因此测试使用 5 GiB。

主要 Ascend 阶段耗时如下，单位均为 ms：

| tokens/rank | preprocess | dispatch+FC1 | weighted SwiGLU | FC2+combine |
|---:|---:|---:|---:|---:|
| 4K | 5.031 | 19.221 | 2.257 | 16.491 |
| 8K | 5.764 | 33.973 | 4.465 | 30.961 |
| 16K | 6.970 | 64.385 | 8.888 | 59.042 |

阶段中位数来自独立采样，不能严格相加。当前最明显的性能短板仍是 dispatch+FC1：
8K、16K 相对 golden 分别仅为 1.058x、1.052x。

与 Megatron mean latency 的对比如下：

| tokens/rank | Megatron mean | Triton mean | Megatron / Triton |
|---:|---:|---:|---:|
| 4096 | 45.136 ms | 43.130 ms | **1.047x** |
| 8192 | 68.541 ms | 74.582 ms | 0.919x |
| 16384 | 127.801 ms | 137.767 ms | 0.928x |

- 4K：Triton 快约 4.4%。
- 8K：Megatron 快约 8.8%。
- 16K：Megatron 快约 7.8%。

为支持 Kimi-K3 的 896 experts，本次修复了 1024-bin routing metadata 路径：

- 使用排序结果的向量化 lower-bound，绕过 910B1 无法正常完成的 1024-bin histogram。
- 在 masked load 前将 inactive lane 地址钳制到合法下标。
- all-drop 时提供有效哨兵地址，避免后端“无条件 load 后 select”造成越界。
- DSV4 的 512-bin histogram 路径保持不变。
- 只修改 Triton Python，没有修改 NPU-IR。

#### DSV4 W8 历史快照

以下展示仓库已归档的 DSV4 W8、8K tokens/rank、KN 历史正式快照的四阶段 median，数据来自
[`bench_full_forward_dsv4_grouped_routefp32_w8.json`](results/forward/bench_full_forward_dsv4_grouped_routefp32_w8.json)。
加速比为 `Grouped baseline / Ascend candidate`，大于 1 表示 Ascend candidate 更快。
本轮 forward 同步后尚未重新运行性能测试，因此这张表不表示新代码快照的复测结果。

| Stage | Ascend candidate/ms | Grouped baseline/ms | speedup |
|---|---:|---:|---:|
| preprocess | 3.906 | 11.266 | **2.884x** |
| dispatch+FC1 | 25.191 | 27.056 | **1.074x** |
| weighted SwiGLU | 1.705 | 6.592 | **3.866x** |
| FC2+combine | 65.610 | 39.905 | 0.608x |

四阶段来自各自独立的完整执行 event 样本，只用于定位瓶颈，不能相加还原完整
post-routing forward latency。其余 world size 和模型的完整 JSON 仍保存在
`results/forward/`，可通过 `summarize_results.py` 汇总。
