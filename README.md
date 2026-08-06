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
config/
└── _shapes.py                # 所有模型/shape 配置（MoETestShape + 各 shape 列表 + MODEL_PROFILES）

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
├── conftest.py               # @pytest.mark.dist 多进程 HCCL 启动夹具
├── _moe_dist_utils.py        # 多卡测试/基准共享工具（ACLSHMEM、peer_mem 等）
├── _numeric.py               # 数值比较阈值与判定
├── _goldens/                 # torch golden 参考实现（backward）
├── layer/                    # 完整前向/反向流程测试
└── kernel/{forward,backward}/  # kernel 级单测（占位）

benchmark/layer/              # 前向/反向 benchmark 入口与结果汇总
```

## How to start

### 正确性测试

基础环境依赖配置测试：

```bash
python -m pytest tests/layer/test_moe_forward.py -m "not dist" -v
```

前向两卡用例：
```bash
python -m pytest \
  tests/layer/test_moe_forward.py::test_forward_2ranks \
  -m dist -v -s
```

后向两卡用例：
```bash
python -m pytest \
  tests/layer/test_moe_backward.py::test_backward_2ranks \
  -m dist -v -s
```

### 性能测试（benchmark）


前向性能（`bench_full_forward.py`）：

```bash
MOE_FULL_BENCH_CONFIG=kimi_k3_4k \
python -m pytest -p tests.conftest \
  benchmark/layer/bench_full_forward.py::test_bench_full_forward_kimi_k3_8ranks \
  -m dist -v -s
```

后向性能（`bench_backward.py`）：

```bash
# MOE_PERF_CONFIGS: =1 跑全部 perf shape；给模型 label（大小写不敏感、逗号分隔）只跑该模型
MOE_PERF_CONFIGS=Kimi-K3 \
python -m pytest -p tests.conftest \
  benchmark/layer/bench_backward.py::test_bench_backward_2ranks \
  -m dist -v -s
```

## 现有性能结果

### Forward

#### Kimi-K3

完整*八卡* post-routing forward 的加速比为 `Grouped golden / Ascend candidate`：

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

与 Megatron mean latency 的对比如下：

| tokens/rank | Megatron mean | Triton mean | Megatron / Triton |
|---:|---:|---:|---:|
| 4096 | 45.136 ms | 43.130 ms | **1.047x** |
| 8192 | 68.541 ms | 74.582 ms | 0.919x |
| 16384 | 127.801 ms | 137.767 ms | 0.928x |

- 4K：Triton 快约 4.4%。
- 8K：Megatron 快约 8.8%。
- 16K：Megatron 快约 7.8%。

### Backward

两卡后向的计算加速比 `torch(ms) / triton(ms)`

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

