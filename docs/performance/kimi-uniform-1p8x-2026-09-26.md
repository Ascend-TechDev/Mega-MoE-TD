# Kimi K3 fused forward — MTE2 瓶颈定位与 wave32 优化（2026-09-26）

分支 `codex/kimi-uniform-1p8x`，基线 `64d8c03`，生产改动 `3c1b507`。
硬件 8 x Ascend950DT_9582（32 AICore / 64 AIVector），CANN 9.2.0-beta.2，
bishengir-compile 1.2.0。环境与复现入口同
[交接文档](../kimi-forward-optimization-handoff-2026-09-26.md) 第 3、10 节。

## 1. 目标与结果

目标：非 MoonEP 均衡用例达到 1.8x，AIC MAC ratio 提升到 90%。

**未达成。** 实测最优 1.563x，MAC ratio 从 80.1% 提升到 84.7%。
根因是 MTE2（GM->L1 搬运）饱和，且降低 MTE2 的三条路径均被硬件硬约束封死
（见第 4 节）。本文记录定位过程、已取得的收益、以及每条被否决路径的实测证据。

## 2. 瓶颈定位：不是 Cube 双缓冲，是 MTE2

用 torch_npu Level1 PipeUtilization PMU 采集（E32 均衡 top16 W8 T4K，
非 MoonEP，M256/N256/K128）。rank0 三次 launch：

| 流水 | wave16（旧默认） | wave32（新默认） |
|---|---:|---:|
| MAC | 79.20 / 80.13 / 80.87 % | 84.88 / 84.65 / 84.59 % |
| **MTE2** | 71.16 / 72.57 / 73.69 % | 77.83 / 77.46 / 77.43 % |
| MTE1 | 31.40 / 31.77 / 32.04 % | ~35.1 % |
| scalar | ~15.9 % | ~14.9 % |
| fixpipe | 2.30 / 2.32 / 2.33 % | ~2.5 % |
| MTE3 / vec | ~0 | ~0 |

**fixpipe 仅 2.3%、MTE3 与 vector 近似为零**，说明 CV ping-pong 与 UB 双缓冲
不是瓶颈；与 MAC 抢时间的是 MTE2。编译产物 IR 佐证：48 处
`hivm.address_space` 全部是 `ub`，没有一个 `l1`。

流量测算（FC1，每卡 E=4、每专家 16384 行，M256/N256/K128）：
A 面板重读 `F/pair_bn` = 24 次 = 11.3 GB，W 面板重读 `rows/BM` = 64 次 = 11.3 GB，
FC1+FC2 合计 **33.8 GB/卡/次 forward**，理论下限 0.65 GB，放大 35x。
按 MTE2 忙时 9.86 ms 折算约 3429 GB/s，超出单卡 HBM 现实带宽，
说明多数命中 L2，但 MTE2 端口本身已饱和。

### 工具修复

profiler 解析全部失败（`kernel_details.csv` 缺失）的根因是系统缺
`libsqlite3.so` 软链（只有 `.so.0`）。修复方式（不改系统）：

~~~bash
mkdir -p /tmp/libshim && ln -sf /usr/lib64/libsqlite3.so.0 /tmp/libshim/libsqlite3.so
export LD_LIBRARY_PATH=/tmp/libshim:$LD_LIBRARY_PATH
python -c "import torch_npu; torch_npu.profiler.profiler.analyse(profiler_path='<rankN_..._ascend_pt>')"
~~~

可对已采集数据重解析，无需重跑。PMU 在
`ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db` 的 `TASK_PMU_INFO` 表，
指标名经 `STRING_IDS` 解码。

## 3. 生产改动：wave windows 16 -> 32

wave window 决定一个 core 复用同一 A 面板跨越多少 row tile，是唯一能改变
MAC/MTE2 比例的参数。E32 均衡 top16 W8 T4K，M256/N256/K128 实测：

| windows | fused median | speedup |
|---:|---:|---:|
| 8 | 14.725 ms | 1.425x |
| 16（旧默认） | 14.214 ms | 1.475x |
| **32（新默认）** | **13.476 ms** | **1.562x** |
| 64 | 13.348 ms | 1.569x |

64 与 32 在噪声内（wave32 重复测得 13.535，噪声约 ±0.06 ms），且需要整个专家
放进一个 wave，故取 32。MoonEP 路径原本就是 32，**默认值未变**。

## 4. 被硬件硬约束封死的三条路径

### 4.1 放大 tile —— L0C 物理上限

`_MAX_FC1_GEMM_ACCUMULATOR_ELEMENTS = 256*256` 不是保守估计，而是精确贴合
Ascend950DT 的 L0C。一个 FC1 tile 同时持有 gate+up 累加器：
`m * (n/2) * 2 * 4B = m * n * 4B`，256x256 即 262144 B = 2097152 bits。

实测 M256/N512（FC1 单独、FC1+FC2 同时，两次独立尝试）：

~~~
cc overflow, requires 4194304 bits while 2097152 bits available!
error: Failed to run buildFinalHIVMPipelines pipeline
~~~

**2097152 bits = 256 KB 即 L0C 实际容量，当前配置已 100% 占满。** 这是硬件边界，
不可绕过。已在 `config.py` 注释中固化，避免重复探测。

### 4.2 显式 L1 驻留 —— API 与容量双重不可行

`triton.language.extra.cann.extension` 暴露 `L0A/L0B/L0C/L1/UB` 地址空间，
但 `al.copy` 与 `al.copy_from_ub_to_l1` **只支持 UB->L1，没有 GM->L1**。
走 GM->UB->L1 会让数据两次过 UB，反而增加 MTE2 并占用 vector 带宽。

即便 API 可用，A 面板 `256x3584x2B` = 1.75 MB、W 面板同为 1.75 MB，
合计 3.5 MB，远超单核 L1，**K 循环内的重复搬运是结构性的**。

### 4.3 编译器 multi-buffer 开关 —— 现有配置已是最优

基线 wave32 13.48 ms / 1.562x：

| 开关 | fused median | speedup | 结论 |
|---|---:|---:|---|
| `limit_auto_multi_buffer_buffer=no-limit` | 16.978 ms | 1.238x | 劣化 25% |
| `limit_auto_multi_buffer_buffer=only-vector` | 16.911 ms | 1.243x | 劣化 25% |
| `set_workspace_multibuffer=2` | 13.589 ms | 1.548x | 噪声内 |
| `enable_preload=True` | 13.620 ms | 1.542x | 噪声内 |
| `set_cv_pipeline_mode` | — | — | Triton 不识别该 kwarg |

`no-limit`/`only-vector` 的 25% 劣化印证了源码注释：auto-buffering vector
临时变量会挤掉后端的 L1 buffering，而 FC1 是 MTE2 受限，失去 L1 复用代价极大。
`only-cube` 保留。

### 4.4 dispatch 去重（同 rank 多专家只发一次 + 本地拷贝）

量化（torch.topk 均衡路由，T=4096，实测而非估算）：每 token 命中的不同目标
rank 数，E32/top16 为 **7.592**（16 路由 -> 47.4%，2.11x），E896/top16 为
**7.068**（-> 44.2%，2.26x）。跨卡传输可从 411 MB/卡 降到 195 MB/卡。

**但它不能提升 MAC ratio**：接收行数仍是 T x topk（本地拷贝把行展开回来），
FC1/FC2 的 33.8 GB MTE2 流量一字节不变。该优化改善的是通信暴露时间，
而通信在当前 profile 中不是瓶颈。**未实施。**

## 5. Block 选择：与 T 相关，各点已调至最优

FC1 与 FC2 的 M tile 必须相同（`config.py` 约束）。

| 用例 | M256/N256 | M128/N512 | 采用 |
|---|---:|---:|---|
| E32 均衡 t4k | **1.539x** | — | M256/N256 |
| E32 均衡 t8k | **1.437x** | 1.387x | M256/N256 |
| E32 均衡 t16k | **1.347x** | 1.321x | M256/N256 |
| E896 均衡 t4k | 1.303x | **1.416x** | M128/N512 |
| E896 均衡 t8k | 1.417x | **1.432x** | M128/N512 |
| E896 均衡 t16k | **1.563x** | 1.533x | M256/N256 |

其他否决项：FC1 K256 与 FC2 K256 均无收益（14.30 / 14.24 ms vs 基线 14.21）；
M512 被 `M <= 256` 挡住；wave64 在 t16k 为 1.350x，与 wave32 的 1.347x 同档。

## 6. 全覆盖回归矩阵

全部通过 normal / zero-receive / all-drop 正确性门（rtol=atol=0.05）与
occupancy gate，5 warmup + 50 measured，8 rank MAX，speedup = Torch median / fused median。

### 6.1 非 MoonEP 均衡（本次目标用例，各点最优 block、wave32）

| 用例 | fused median | Torch median | speedup |
|---|---:|---:|---:|
| E32 t4k | 13.602 ms | 20.939 ms | 1.539x |
| E32 t8k | 28.416 ms | 40.845 ms | 1.437x |
| E32 t16k | 59.244 ms | 79.803 ms | 1.347x |
| E896 t4k | 15.308 ms | 21.682 ms | 1.416x |
| E896 t8k | 28.699 ms | 41.103 ms | 1.432x |
| E896 t16k | 53.506 ms | 83.652 ms | 1.563x |

**均未达到 1.8x**，缺口 15%–25%。

### 6.2 非 MoonEP 偏斜：wave16（旧默认）vs wave32（新默认）回归对照

| 用例 | wave16 | wave32 | 变化 |
|---|---:|---:|---:|
| trimmed t4k | 1.442x | 1.523x | **+5.6%** |
| trimmed t8k | 1.414x | 1.407x | −0.5%（噪声） |
| trimmed t16k | 1.309x | 1.368x | **+4.5%** |
| full t4k | 1.444x | 1.453x | +0.6% |
| full t8k | 1.516x | 1.506x | −0.7%（噪声） |
| full t16k | 1.492x | 1.489x | −0.2%（噪声） |

wave16 实测值与交接文档第 5 节一致（如 trimmed t4k 1.442 vs 文档 1.446），
测量口径可比。**无实质回退**，最大负向变动 0.7% 在 ±0.06 ms 噪声量级内。

### 6.3 MoonEP 偏斜裁剪（默认 wave32，未改动）

| 用例 | 本次 | 交接文档 | 变化 |
|---|---:|---:|---:|
| trimmed t4k | 2.226x | 2.256x | −1.3% |
| trimmed t8k | 2.123x | 2.183x | −2.7% |
| trimmed t16k | 2.112x | 2.111x | +0.0% |

MoonEP 默认值未变，差异属运行间波动。**这三点仍是全矩阵唯一超过 1.8x 的用例。**

### 6.4 MoonEP 非裁剪（E896）

沿用交接文档第 8.7 节状态：后端编译阻塞，本次以 1200 s 预算重试仍未通过，
未产生性能数据。**不填零、不计为性能失败。**

## 7. 结论

1. 瓶颈是 MTE2 饱和而非 Cube 双缓冲，有 PMU 与 IR 双重证据。
2. wave16->32 是本轮唯一有效改动，MAC ratio 80.1%->84.7%，
   均衡用例 1.475x->1.539x（t4k），偏斜用例最高 +5.6%，无回退。
3. MAC ratio 要到 90% 必须降 MTE2，而放大 tile（L0C 硬上限）、显式 L1 驻留
   （无 GM->L1 API 且容量不足）、编译器 multi-buffer（现配置已最优）三条路
   全部封死。**在维持 fused 单 kernel、BF16、当前 L0C 的前提下，1.8x 不可达。**
4. dispatch 去重可降跨卡字节 2.1–2.26x，但不影响 MTE2，对 MAC ratio 无帮助。

若要继续逼近 1.8x，需要放宽约束，按预期收益排序：
- **FP8 计算**（非仅 saved）：MTE2 字节数直接减半，是唯一能大幅降低 33.8 GB 的手段；
- 拆分 FC1 为独立 kernel，用超出当前 fused kernel 寄存器/UB 预算的 tile 策略；
- 权重常驻改造：把 W 预排布为 L2 友好布局，降低 64 次重读的实际代价。

以上均超出"维持 fused triton forward 形态"的约束，需要明确授权。
