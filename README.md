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

### A2/A3整网测试（kimi k3 mock 训练）

通过宿主仓
[MindSpeed-MM_MoonEP](https://gitcode.com/jzhoujg/MindSpeed-MM_MoonEP.git)
的 `ep_plan.dispatcher: megamoe` 路径，在 8 卡上端到端跑 kimi k3 mock 训练：

```bash
# 1. 安装本仓库（宿主仓 import 的 mega_moe 即来源于此）
cd Mega-MoE-TD
python -m pip install -e . --no-deps

# 2. 拉取宿主仓并切到适配分支
git clone https://gitcode.com/jzhoujg/MindSpeed-MM_MoonEP.git
cd MindSpeed-MM_MoonEP
git checkout moonep

# 3. 生成 mock 数据（新 clone 没有 data/，yaml 引用
#    data/mocked_vl_data/mock_data_pic_num_1_textlen_700.json）
python mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py \
  --tokenizer_path mindspeed_mm/fsdp/models/kimi_k3 \
  --num_pics 1 --text_length 700 --save_dir ./data/mocked_vl_data/

# 4. 启动 8 卡训练（kimik3_config.yaml 在该分支已默认 dispatcher: megamoe）
bash examples/kimi_k3/finetune_kimik3.sh
```

融合反向是 native-saved 零重放，`recompute` / `enable_activation_offload` /
`megamoe_shared_op` 三项配置约束及原因见宿主仓
`examples/kimi_k3/README.md`（违反会在首个 backward 处失败）。

## 现有性能结果

### Forward

#### Kimi-K3

模型配置：总专家数 `896`，每个 token 选择 `top-k=16` 个专家。

完整 *950DT 八卡* Mega-MoE-TD forward 的 median耗时对比如下。加速比定义为
`Megatron moe-permute-fusion / Mega-MoE-TD`

| tokens/rank | Mega-MoE-TD | Megatron moe-permute-fusion | Grouped Torch | Megatron / Mega-MoE-TD | |
|---:|---:|---:|---:|---:|---:|
| 4K | 16.801 ms | 19.634 ms | 22.198 ms | **1.169x** | 
| 8K | 28.022 ms | 32.653 ms | 42.501 ms | **1.165x** | 
| 16K | 51.171 ms | 66.506 ms | 85.922 ms | **1.300x** |


Mega-MoE-TD 的主要阶段 median 耗时 如下：

| tokens/rank | Preprocess | Dispatch+FC1 | Weighted SwiGLU + FC2+Combine | E2E |
|---:|---:|---:|---:|---:|
| 4K | 0.912 ms | 9.712 ms | 6.694 ms | 16.801 ms |
| 8K | 0.946 ms | 16.635 ms | 11.085 ms | 28.022 ms |
| 16K | 0.946 ms | 30.770 ms | 19.878 ms | 51.171 ms |


### Backward

#### Kimi-K3（八卡）

完整*八卡A3*后向`torch(ms) / triton(ms)`：

| tokens/rank | torch/ms | triton/ms | bigop/ms | tri/torch | tri/bigop |
|---:|---:|---:|---:|---:|---:|
| 2K  | 160.630 | 44.684 |  58.173 | **3.64x** | **1.29x**|
| 4K  | 200.162 | 78.22 |  88.392 | **2.55x** | **1.12x** |

| 功能阶段 | bigop（单流串行） | mega-kernel（5 流重叠） |
|---|---:|---:|
| **A. dispatch + fc2-dx** | a2a 2.8 + gmm 3.89 ≈ **7.5** | step1 **14.76**（纯 GEMM ~3.9 + 通信 ~10.9） |
| **B. swiglu bwd** | ≈ **2.2** | 2.27（与 wgrad 并行，关键路径 ≈ 0） |
| **C. fc2 wgrad** | 3.98 + 回转置 9.6 = **13.6** | ~**4.5**（无回转置） |
| **D. combine + fc1-dx + gate** | a2a 2.8 + gmm 7.05 + Index 2.0 ≈ **12** | tiled GEMM 10.39 + barrier 4.97 + push 2.73 + reduce 0.74（名义 18.8，关键路径 ≈ 11.1） |
| **E. fc1 wgrad** | 7.75 + 回转置 9.6 = **17.4** | ~**7.6**（与其他阶段重叠） |
| 其他 | ~3.5 | ~1.9 |
| **E2E** | **52.90**（=串行相加 52.8） | **40.68**（名义和 49.3，重叠收益 ~8.6） |

#### Kimi-K3 mega-kernel 单 launch（`MOE_BWD_MEGA=1`）

`MOE_BWD_MEGA=1` 把非 MoonEP 五步反向合并为**一个 Triton kernel launch**，
相位间用 `libshmem_device.barrier_all()` 串行（GEMM `BM=256` 默认 tile）。下表为
*950DT 八卡* 与 bigop 基线（`mega_moe._goldens.bigop_ref.moe_backward_bigop`：
bigop grouped matmul/wgrad + `npu_swiglu_backward`，A2A 走 torch/HCCL）的
性能与峰值内存对比。协议：WARM=3 / ITERS=10 / rank-MAX，`torch.manual_seed`
随机路由；内存为 torch 峰值 `reserved`，mega 侧已计入池外 2 GiB aclshmem
对称堆（t8k 峰值需求 ~0.94 GiB，固定 2 GiB 配额）。

| tokens/rank | mega ms/iter | bigop ms/iter | 加速比 | mega 峰值 reserved | bigop 峰值 reserved | 内存差 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | **24.94** | 38.89 | **1.56x** | **25.3 GiB**¹ | 29.9 GiB | −4.6 GiB |
| 4K | **43.43** | 56.34 | **1.30x** | **29.9 GiB**¹ | 34.3 GiB | −4.4 GiB |
| 8K | **79.31** | 94.55 | **1.19x** | **39.6 GiB**¹ | 43.4 GiB | −3.7 GiB |

¹ = torch reserved + 2 GiB 池外 aclshmem 对称堆。

全序列档位性能领先 1.19–1.56x 且峰值内存省 3.7–4.6 GiB：单 kernel 直出
梯度，无 grouped-op 中间转置/重排缓冲；短序列端优势最大（单 launch 省去的
调度/启动开销占比高）。正确性门槛：w2/w8 functional suite + f0b probe2
（kimi 真形 5-key 全比对）/ probe3（epoch 复用 bit 级）全绿。

`MOE_BWD_MEGA=1` 同样覆盖 MoonEP 物理布局（`use_moonep` saved）：P1 增加
replica down 表的第二趟 fc2-dgrad 扫，P4a 的 GEMM 过 `tile_home_bound` 切
home/replica 双权重表（均为 standalone kernel 双表模式的内联）；当前向借出
对称 replica 表（`grad_transport`）时，M3 grad_reduce 链（seed+sink →
owner-pull → zero）作为 P6a/P6b/P6c 尾相位并入同一次 launch，transport 的
三条 barrier 变为内核 B5/B6/B7，尾 barrier #3 由 kernel 退出本身承担。fp32
累加按 home 专家分区、(peer, slot) 描述符序保持 —— 与 fused transport
kernel 逐位一致；host 侧 sink/`post_sink_hook` 被内核链取代（`sunk`/
`reduced` 仍置位，槽位级 bit 精确断言在该模式下跳过，HCCL oracle 归约比对
与 post-zero 检查仍然全量生效）。