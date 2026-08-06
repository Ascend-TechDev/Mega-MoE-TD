# MEGA-MOE-TD

MEGA-MOE-TD 提供基于
[Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)
的 Ascend NPU MoE 前向与反向实现。

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
│   └── _torch_forward.py     # 可微 torch 前向（被 backward 复用）
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

## How to start

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
torchrun --nproc-per-node=2 -m tests._goldens.backward

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
