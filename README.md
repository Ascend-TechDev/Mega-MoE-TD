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
└── kernels/                  # Triton JIT kernels 及 launcher
    └── fused_forward.py      # 路由到 combine 的单次 all-core launch

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

无 MoonEP 的单 kernel 前向通过
`MoEForwardConfig(enable_single_kernel_forward=True)` 显式启用。当前范围是
`return_saved=False` 且 `down_weight` 连续；路由、dispatch/FC1、加权激活、
FC2、反向传输和 top-k combine 均在一次物理 all-core kernel launch 内完成。
MoonEP 与 saved-forward 暂时继续使用原多 kernel 路径。

8 卡 Kimi-K3 T4K 正确性用例：

```bash
MOE_FUSED_ASH_SIZE_GB=6 PYTHONPATH=src:. python -m pytest \
  tests/layer/test_moe_suite.py::test_single_kernel_kimi_k3_t4k_w8 -v -s
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

#### 单 kernel 前向相位打点（`MOE_FWD_TIMING`）

`MOE_FWD_TIMING=1` 让单 kernel 前向的同一次 launch 携带 SYS_CNT 相位打点
（`MOV $0, SYS_CNT` 内联汇编，实测 ~1 GHz，host 按事件钟逐次校准），产物由
`read_last_forward_phase_timing()` 读出，`benchmark/layer/_fwd_phase_timing.py`
做纯 host 归约。采集是独立 pytest 用例
`tests/layer/test_fwd_phase_timing.py`，一个节点只跑一种变体
（case × saved/unsaved × save_fc1_dtype），JSON 落
`MOE_FWD_TIMING_OUT_DIR`（默认 `results/fwd_phase_timing`）：

```bash
MOE_FUSED_ASH_SIZE_GB=6 python -m pytest tests/layer/test_fwd_phase_timing.py \
    -k "phase_timing and fp16 and saved and trimmed-w8-t4k" -m dist -v -s
```

测量约束：Cube 引擎没有向量/标量 ALU，cube scope 内的数据路径运算会被
整块丢弃，因此 FC1/FC2 的 cube 侧墙钟全部由 vector lane 时钟括出
（FC1 按 UB 释放 ack 两步移位、FC2 按激活 signal→return 完成等待）；
跨核时间戳只做每列 max/min，不做核间差值。跨运行对比只认 ticks
（SYS_CNT 原始值）：校准在 958–993 ticks/µs 之间随运行波动（±3.6%，
三次采集），跨分支混比 μs/ms 会引入该量级误差。

8 卡 Kimi-K3 t4k、fp16 保存（warmup 5 / samples 20，校准 982.7 ticks/µs）
的 p50 结果（2026-09-18，优化前基线；段名按段内实际工作命名——打点落在
barrier 之后，2026-09-18 之前的段名整体错位一格，旧 JSON 的
`routing_zero_count`/`stable_cursors` 键分别对应此表的 `counts_publish`/
`route_scatter`）：

| 阶段（段，按内容命名） | p50 µs | 波管线内部（每核累计） | p50 |
|---|---:|---|---:|
| zero_histogram（清零+直方图） | 958 | dispatch_issue | ~23 µs |
| counts_publish（pid0 串行发布） | 2 680 | fc1_cube_wall | 1.88 ms |
| destination_metadata（收发表+波偏移） | 283 | 激活 vact lane0 / lane1 | 0.46 / 5.76 ms |
| stable_cursors（游标转换+pull starts） | 451 | save（fp16，两 lane） | ~0.21–0.23 ms |
| route_scatter（稳定散射） | 5 662 | return_issue（两 lane） | 2.85 ms |
| routing_metadata_total | 10 036 | reduce_issue（两 lane） | 0.29 ms |
| wave_pipeline | 18 834 | e2e（事件钟） | 28.85 ms |
| kernel_total | 28 849 | | |

FC2 按波墙钟（21 波）：wall p50 1.84 ms，从首波 1.58 ms 爬升到第 17 波
峰值 1.99 ms 后回落（尾波 1.36 ms）；residual p50 1.59 ms（wall−residual
≈ 250 µs/波 为跨核偏差与排队份额）。

结论（优化方向依据）：

- **routing 元数据 10.0 ms（kernel 的 35%）是首要优化目标**：其中
  route_scatter 5.66 ms + counts_publish 2.68 ms 占 8.3 ms——前者是单
  vector lane 上的稳定散射（另一个子核整段空转，且 pad bin 扫描有 12.5%
  纯浪费），后者是 pid0 单核对 896 bucket × 32 core 的逐标量串行归并
  （~28 672 次依赖链 load，~93 ns/次，与 2.68 ms 严丝合缝）。两者均无
  GEMM 工作；`feat/single-kernel-routing-metadata` 针对性优化（发布向量化、
  散射双 lane 分摊 + pad bin 裁剪，输出保持逐位相同的稳定序）已测到
  **routing_metadata_total 10.0 → 3.4 ms、e2e min 28.0 → 23.0 ms**
  （2026-09-19 两轮）；逐位稳定序的回归仍待 NPU 跑 test_moe_suite 单
  kernel 节点。
- **激活 lane 不对称：dispatch 竞争已否，"是否占关键路径"待一个消除实验**：
  vact lane0 0.46 ms vs lane1 5.76 ms（12×；全量 case 每 rank 65536 routes /
  112 专家 ≈ 585 行/专家，lane0 的 0.46 ms 就是这段激活的真实工作量级）。
  车道间唯一结构差异是 lane1 独担 dispatch；把 dispatch(step+1) 挪到激活
  signal 之后（`feat/single-kernel-vact-first`，4e8aec1 → 097d080）后
  vact_v1 不降反升（5.7M → 6.0M ticks，两轮复现）、e2e 中性 →
  **dispatch 访存竞争假设被否**，重排这条路线也拿不到收益。wave_pipeline
  同步下移 ~0.3M ticks：该括号装的是可被重排移进移出的等待，不是纯发射
  时间也非纯计算。两 lane 的 busy 合计仍远低于波周期（lane1 ≈ 8.6M、
  lane0 ≈ 3.1M vs 18.1M ticks），它更像与 cube 工作重叠的 slack——但
  "不在关键路径"要能用让等待消失的实验（dispatch 按波奇偶分摊双 lane 降
  突发、或 t16k 缩放判别）证实后才算数。在那之前不按瓶颈追，group 合并类
  优化不立项。
- fp16 保存的 e2e 代价 ≈ 其自身活动（双变体对照 +0.6 ms）；save 括号是
  发射口径（store 异步），0.22 ms/核 与 12.6 MB/lane-subcore 的写入量在
  合理 store 速率下相符——旧记录的“30× 带宽理论值”按错形状算，作废。
- return 2.85 ms/lane 两 lane 对称，但该括号含函数首部的 FC2 完成等待
  （dl.wait），其中真正的搬运份额待细分。
- FC2 每波 wall 缓升 ~20%，属轻度排队而非堆积，fill/drain 缩短的收益
  待逐波数据进一步量化。

### Backward

| 模块 | `kernel_moe_backward_mega` | `kernel_moe_backward_mega_recompute` |
  |---|:---:|:---:|
  | (env) | `MOE_SAVED_RECOMPUTE=0` | `MOE_SAVED_RECOMPUTE=1` |
  | dispatch A2A(P1) | √ | √ |
  | fc2 dgrad(P1) | √ | √ |
  | swiglu bwd(P2) | √ | √ |
  | fc2 wgrad(P3) | √ | √ |
  | fc1 dgrad(P4a) | √ | √ |
  | reverse A2A push(P4b) | √ | √ |
  | fc1 wgrad(P5a/P5b) | √ | √ |
  | topk reduce(P4c) | √ | √ |
  | MoonEP replica | √ | √ |
  | act 行重算 | — | √ |
  | re-dispatch 重发 | — | √ |
  | re-prefetch | — | √ |
  | 读 saved fc1_output | √ | √ |

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

#### Kimi-K3 mega-kernel A5 单 launch（`MOE_BWD_MEGA=1`）


| tokens/rank | mega ms/iter | bigop ms/iter | 加速比 | mega 峰值 reserved | bigop 峰值 reserved | 内存差 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | **24.94** | 38.89 | **1.56x** | **25.3 GiB**¹ | 29.9 GiB | −4.6 GiB |
| 4K | **43.43** | 56.34 | **1.30x** | **29.9 GiB**¹ | 34.3 GiB | −4.4 GiB |
| 8K | **79.31** | 94.55 | **1.19x** | **39.6 GiB**¹ | 43.4 GiB | −3.7 GiB |
