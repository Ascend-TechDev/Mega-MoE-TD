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
└── _shapes.py                # CaseGroup table + expanded CaseSpec registry

src/mega_moe/
├── config.py                 # 前向配置与调度参数校验
├── ops/
│   ├── forward.py            # 完整前向编排与输入校验
│   ├── backward.py           # 五阶段反向编排及 autograd.Function
│   └── _torch_forward.py     # backward 使用的私有可微 Torch 前向
├── runtime/
│   ├── workspace.py          # ACLSHMEM 对称内存及 workspace 生命周期
│   └── routing.py            # 路由过滤、排序、计数交换和 offset
├── kernels/                  # Triton JIT kernels 及 launcher

conftest.py                   # @pytest.mark.dist 多进程 HCCL 启动夹具

tests/
├── _moe_testkit.py           # ACLSHMEM 生命周期、peer memory、计时协议
├── _moe_baselines.py         # 独立 functional oracle 与 hand-written backward baseline
├── _numeric.py               # 数值比较阈值与判定
├── layer/test_moe_suite.py   # 参数化 functional forward/backward suite
└── kernel/{forward,backward}/  # kernel 级单测（占位）

benchmark/layer/              # 前向/反向 benchmark、profiler 与结果汇总
├── bench_moe_suite.py        # 参数化 performance forward/backward suite
├── _grouped_forward_baseline.py # Torch-NPU grouped-GEMM + HCCL performance baseline
└── summarize_results.py
```

## How to start

### 正确性测试

功能 suite 的入口是同一个参数化文件；可用 `-k` 或 node id 选择 case：

```bash
python -m pytest \
  tests/layer/test_moe_suite.py -k 'forward and smoke' \
  -m dist -v -s
python -m pytest \
  tests/layer/test_moe_suite.py -k 'backward and smoke' \
  -m dist -v -s
```

### 性能测试（benchmark）


性能 suite（`bench_moe_suite.py`）：

```bash
# 先查看显式的 model/world/tokens-per-rank node id，再选择一个 case
python -m pytest --collect-only -q benchmark/layer/bench_moe_suite.py
python -m pytest \
  'benchmark/layer/bench_moe_suite.py::test_bench_forward_case[performance-fwd-qwen-w8-t8k]' \
  -m dist -v -s
```

后向性能：

```bash
python -m pytest \
  'benchmark/layer/bench_moe_suite.py::test_bench_backward_case[performance-bwd-kimi-k3-w4-t4k]' \
  -m dist -v -s
```

## 现有性能结果

### Forward

#### Kimi-K3

完整*八卡* post-routing forward 的加速比为 `Grouped baseline / Ascend candidate`：

| tokens/rank | Ascend full | Grouped baseline | 加速比 | 观测 HBM/卡 |
|---:|---:|---:|---:|---:|
| 2K  | 27.525 ms | 43.207 ms | **1.570x** | — |
| 4K  | 44.709 ms | 80.422 ms | **1.799x** | ≈22.4 GiB |
| 8K  | 79.312 ms | 149.837 ms | **1.889x** | ≈35.1 GiB |
| 16K | 137.703 ms | 302.747 ms | **2.199x** | ≈53.5 GiB |


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

#### Kimi-K3（八卡）

完整*八卡A3*后向`torch(ms) / triton(ms)`：

| tokens/rank | torch/ms | triton/ms | bigop/ms | tri/torch | tri/bigop |
|---:|---:|---:|---:|---:|---:|
| 2K  | 160.630 | 44.684 |  58.173 | **3.64x** | **1.29x**|
| 4K  | 200.162 | 78.22 |  88.392 | **2.553** | **1.12x** |