# Kimi K3 fused forward 第二轮优化交接文档（2026-09-26）

接续 [第一轮交接](kimi-forward-optimization-handoff-2026-09-26.md)。
本文给下一位 AI，记录本轮全部工作：瓶颈定位方法、生产改动、**所有无效尝试及其
被否决的原因**、硬件边界的实测证据、以及未解决的问题。

分支 `codex/kimi-uniform-1p8x`，基线 `64d8c03`（`codex/kimi-forward-perf`）。

---

## 0. 任务与达成情况

用户目标（按时间顺序演进）：

1. 裁剪/非裁剪、4/8/16k、8 卡相比 torch 明显提升，最好 ≥1.8x，维持 fused triton forward 形态；
   用户建议方向：同 rank 多专家的 token 只发一次 + 本地拷贝。
2. 补充：**非 MoonEP 的均衡用例**也要到 1.8x。
3. 补充：AIC MAC ratio 提到 90%，怀疑 cube 编译器双缓冲问题，用 msprof pipeline 确认。
4. 补充：只跑 forward 测试，不跑 backward。
5. 补充：全覆盖跑性能测试，确保无回退。
6. 补充：定位并解决 E896+MoonEP 编译阻塞。

**达成**：MAC ratio 80.1% → 84.7%；均衡用例最高 1.563x；全覆盖无回退。
**未达成**：1.8x（缺口 15%–25%）；MAC ratio 90%；E896+MoonEP 编译仍阻塞。

**结论：在维持 fused 单 kernel + BF16 + 当前 L0C 的前提下，1.8x 不可达。**
理由见第 4 节三条硬约束，每条都有硬件层面的实测证据。

---

## 1. 环境（与第一轮相同，补充一处关键修复）

硬件 8 x Ascend950DT_9582，`get_aicore_num()=32`、`get_aivector_core_num()=64`。
CANN 9.2.0-beta.2，bishengir-compile 1.2.0，Python 3.11.10，torch 2.10.0+cpu，
torch_npu 2.10.0.post1.dev20260528。环境脚本见第一轮文档第 3.2 节，本轮沿用：

~~~bash
source /home/vllm_kimiw/.venv-udma/bin/activate
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.2.0-beta.2
export PATH=/home/vllm_kimiw/.venv-udma/lib/python3.11/site-packages/ascendnpuir/bin:/usr/local/Ascend/cann-9.2.0-beta.2/bin:$PATH
export TRITON_BACKENDS_IN_TREE=1
export TRITON_DISABLE_FFTS=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MOE_FUSED_ASH_SIZE_GB=16
export MOE_FWD_TIMING=0
export PYTHONPATH=src:.
export TRITON_CACHE_DIR=/tmp/kimi_u18/cache
~~~

### 1.1 【重要】profiler 解析失败的根因与修复

第一轮拿不到完整 PipeTimeline/PMU，根因**不是** msopprof 能力问题，而是：

系统只有 `/usr/lib64/libsqlite3.so.0`，**缺 `libsqlite3.so` 软链**，
导致 torch_npu profiler 的 `CANNTimelineParser`/`KernelViewParser`/`DbParser`
等全部 task 静默 `run failed`，`kernel_details.csv` 不生成。

修复（不改系统）：

~~~bash
mkdir -p /tmp/libshim && ln -sf /usr/lib64/libsqlite3.so.0 /tmp/libshim/libsqlite3.so
export LD_LIBRARY_PATH=/tmp/libshim:$LD_LIBRARY_PATH
# 可对已采集数据重解析，无需重跑 8 卡
python -c "import torch_npu; torch_npu.profiler.profiler.analyse(profiler_path='<rankN_..._ascend_pt>')"
~~~

PMU 数据在 `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db` 的
`TASK_PMU_INFO` 表（列 `globalTaskId,name,value`），指标名要经 `STRING_IDS` 表解码。
提取脚本：

~~~python
import sqlite3, collections
c = sqlite3.connect(db)
names = {r[0]: r[1] for r in c.execute('select id,value from STRING_IDS')}
per = collections.defaultdict(dict)
for gid, nm, v in c.execute('select globalTaskId,name,value from TASK_PMU_INFO'):
    per[gid][names.get(nm, str(nm))] = float(v)
# aic_total_time 单位 ns；aic_total_cycles 是 32 核求和（/time 得 ~48.8 GHz = 32 x 1.525 GHz）
# ratio = aic_<pipe>_time / aic_total_time
~~~

注意 `aic_*_ratio` 列存的是被截断的整数（0 或 1），**不可直接用**，
必须用 `aic_<pipe>_time / aic_total_time` 自己算。

---

## 2. 瓶颈定位：MTE2 饱和，不是 Cube 双缓冲

### 2.1 PMU 实测（E32 均衡 top16 W8 T4K，非 MoonEP，M256/N256/K128，rank0 三次 launch）

| 流水 | wave16 | wave32 |
|---|---:|---:|
| MAC | 79.20 / 80.13 / 80.87 % | 84.88 / 84.65 / 84.59 % |
| **MTE2** | 71.16 / 72.57 / 73.69 % | 77.83 / 77.46 / 77.43 % |
| MTE1 | 31.40 / 31.77 / 32.04 % | ~35.1 % |
| scalar | ~15.9 % | ~14.9 % |
| fixpipe | 2.30 / 2.32 / 2.33 % | ~2.5 % |
| MTE3 / vec | ~0 | ~0 |

**fixpipe 只有 2.3%、MTE3 与 vector 近似 0** —— 这直接证伪了"cube 双缓冲/CV
ping-pong 不足"的假设。若 UB 双缓冲不够，特征应是 fixpipe 高或 MAC 有大段空洞。
与 MAC 抢时间的是 MTE2。

IR 佐证：编译产物 `kernel.source` 中 48 处 `hivm.address_space` **全部是 `ub`，
没有一个 `l1`**，`tt.dot` 的操作数直接从 GM 经 MTE2 读入。

### 2.2 MTE2 流量测算

FC1 结构（`_partition_pipeline_fc1_activation_group_ub`）：
`pair_block_m = BLOCK_M`、`pair_block_n = BLOCK_N // 2`，每个 cube step 做
gate 和 up 两个 dot，K 方向 `BLOCK_K` 切片循环。

流量闭式（每卡每次 forward）：

- A 流量 `= rows * H * 2B * (F / pair_bn)`，即 A 面板按 n_tile 数重读
- W 流量 `= rows * F * H * 4B / BM`，即 W 面板按 row_tile 数重读

E32 均衡 t4k（rows=65536、H=3584、F=3072、BM=256、pair_bn=128）：
A 重读 24 次 = 11.3 GB，W 重读 64 次 = 11.3 GB，FC1 合计 22.5 GB；
加 FC2 的 5.64+5.64 GB，**总计 33.8 GB**，理论下限 0.65 GB，**放大 35 倍**。
按 MTE2 忙时 9.86 ms 折算约 3429 GB/s，超单卡 HBM 现实带宽，
说明多数命中 L2，但 **MTE2 端口本身已饱和**。

### 2.3 tile 遍历顺序（重要，避免重复分析）

`row_step = FC1_CORES % cube_group_row_parts`、`n_step = FC1_CORES // cube_group_row_parts`。
E32 + wave16：`row_parts=16`、`FC1_CORES=32` → `row_step=0`、`n_step=2`，
即**每个 core 的 row_part 固定、只推进 n_tile**（A 驻留、W 全流式）。

**不要尝试改成 W-stationary**：BM=256 时 A 块（256x3584x2B）与 W 块
（3584x128x2Bx2）**都是 1.84 MB，完全对称**，换方向流量数学上一字节不变。
我算过，已否决。

---

## 3. 生产改动（唯一有效）

提交 `3c1b507`：`single_kernel_group_windows` 默认 16 → 32（两条路径统一）。

wave window 决定一个 core 复用同一 A 面板跨越多少 row tile。
E32 均衡 top16 W8 T4K，M256/N256/K128 实测：

| windows | fused median | speedup | 备注 |
|---:|---:|---:|---|
| 8 | 14.725 ms | 1.425x | |
| 16（旧默认） | 14.214 ms | 1.475x | |
| **32（新默认）** | **13.476 ms** | **1.562x** | 重复测 13.535，噪声 ±0.06 ms |
| 64 | 13.348 ms | 1.569x | 与 32 同档，需整专家入 wave |

MAC ratio 80.1% → 84.7%。**MoonEP 路径原本就是 32，默认值未变**，
所以回归风险只在非 MoonEP 侧（第 5 节已验证无回退）。

另有提交 `99f2320`（性能报告）和本轮文档。
`src/mega_moe/config.py` 与 `src/mega_moe/ops/forward.py` 的其余改动**只有注释**，
把实测边界固化下来避免重复探测。

---

## 4. 三条硬约束：为什么 1.8x 不可达

要提 MAC ratio 必须降 MTE2 流量，三条路都被封死，**每条都有实测证据，不要重试**。

### 4.1 放大 tile —— L0C 物理上限（硬件硬挡）

`_MAX_FC1_GEMM_ACCUMULATOR_ELEMENTS = 256*256` **不是保守估计**。
一个 FC1 tile 同时持有 gate+up 两个 FP32 累加器：
`m * (n/2) * 2 * 4B = m * n * 4B`，256x256 即 262144 B = **2097152 bits**。

实测放开该 cap 后编译 M256/N512（FC1 单独、FC1+FC2 同时，两次独立尝试）：

~~~
loc(...): error: cc overflow, requires 4194304 bits while 2097152 bits available!
loc(...): error: Failed to run buildFinalHIVMPipelines pipeline
[ERROR] Failed to run BiShengIR regbase pipeline
~~~

**2097152 bits = 256 KB 就是 Ascend950DT 每核 L0C 实际容量，当前配置已 100% 占满。**
`M ≤ 256` 与 `M*N ≤ 65536` 都是硬件边界。已在 config.py 注释中固化。

### 4.2 显式 L1 驻留 —— API 与容量双重不可行

`triton.language.extra.cann.extension` 的 `ascend_address_space` 暴露
`L0A/L0B/L0C/L1/UB`，但可用的拷贝原语只有：

- `al.copy(src, dst)` —— 文档明确"Copies data from the **Unified Buffer (UB)** to
  the Unified Buffer (UB) or L1 Buffer"
- `al.copy_from_ub_to_l1(src, dst)`

**没有 GM→L1**。走 GM→UB→L1 会让数据两次过 UB，反而增加 MTE2 并占用 vector 带宽。

即便 API 可用：A 面板 `256x3584x2B = 1.75 MB`、W 面板同为 1.75 MB，合计 3.5 MB，
远超单核 L1，**K 循环内的重复搬运是结构性的**。

### 4.3 编译器 multi-buffer 开关 —— 现配置已最优

基线 wave32 13.48 ms / 1.562x：

| 开关 | fused median | speedup | 结论 |
|---|---:|---:|---|
| `limit_auto_multi_buffer_buffer=no-limit` | 16.978 ms | 1.238x | **劣化 25%** |
| `limit_auto_multi_buffer_buffer=only-vector` | 16.911 ms | 1.243x | **劣化 25%** |
| `set_workspace_multibuffer=2` | 13.589 ms | 1.548x | 噪声内 |
| `enable_preload=True` | 13.620 ms | 1.542x | 噪声内 |
| `set_cv_pipeline_mode=skew` | — | — | Triton 报 `unrecognised` kwarg |

`no-limit`/`only-vector` 的 25% 劣化印证了 `forward.py` 原注释：
auto-buffering vector 临时变量会挤掉后端的 L1 buffering，而 FC1 是 MTE2 受限，
失去 L1 复用代价极大。**`only-cube` 必须保留。**

注意：`bishengir-compile --help` 里说
"Ascend950/RegBase applies no-limit when this flag is not explicitly set"，
看起来像是"代码里硬编码 only-cube 压制了硬件默认"，**但实测 no-limit 更慢 25%**，
所以这个默认值描述不能作为优化依据。

---

## 5. 全覆盖回归矩阵

全部通过 normal / zero-receive / all-drop 正确性门（rtol=atol=0.05）与 occupancy gate，
5 warmup + 50 measured，8 rank MAX，speedup = Torch median / fused median。

### 5.1 非 MoonEP 均衡（目标用例，各点最优 block、wave32）

| 用例 | block | fused median | Torch | speedup |
|---|---|---:|---:|---:|
| E32 t4k | M256/N256/K128 | 13.602 ms | 20.939 | 1.539x |
| E32 t8k | M256/N256/K128 | 28.416 ms | 40.845 | 1.437x |
| E32 t16k | M256/N256/K128 | 59.244 ms | 79.803 | 1.347x |
| E896 t4k | M128/N512/K128 | 15.308 ms | 21.682 | 1.416x |
| E896 t8k | M128/N512/K128 | 28.699 ms | 41.103 | 1.432x |
| E896 t16k | M256/N256/K128 | 53.506 ms | 83.652 | 1.563x |

### 5.2 非 MoonEP 偏斜：wave16（旧）vs wave32（新）—— 回归对照

| 用例 | wave16 | wave32 | 变化 |
|---|---:|---:|---:|
| trimmed t4k | 1.442x | 1.523x | **+5.6%** |
| trimmed t8k | 1.414x | 1.407x | −0.5%（噪声） |
| trimmed t16k | 1.309x | 1.368x | **+4.5%** |
| full t4k | 1.444x | 1.453x | +0.6% |
| full t8k | 1.516x | 1.506x | −0.7%（噪声） |
| full t16k | 1.492x | 1.489x | −0.2%（噪声） |

wave16 实测与第一轮文档第 5 节一致（trimmed t4k 1.442 vs 文档 1.446），口径可比。
**无实质回退。**

### 5.3 MoonEP 偏斜裁剪（默认 wave32，未改动）

| 用例 | 本轮 | 第一轮文档 | 变化 |
|---|---:|---:|---:|
| trimmed t4k | 2.226x | 2.256x | −1.3% |
| trimmed t8k | 2.123x | 2.183x | −2.7% |
| trimmed t16k | 2.112x | 2.111x | +0.0% |

**这三点仍是全矩阵唯一超过 1.8x 的用例。**

### 5.4 MoonEP 非裁剪（E896）

仍编译阻塞，见第 7 节。

---

## 6. 全部无效尝试清单（不要重复）

### 6.1 block 配置

| 配置 | 结果 | 否决原因 |
|---|---:|---|
| FC1 K256 | 14.304 ms / 1.472x | 无收益。**K 不影响 MTE2 流量**：A 流量只与 BN 有关、W 流量只与 BM 有关 |
| FC2 K256 | 14.239 ms / 1.475x | 同上 |
| M128/N256（均衡 E32 t4k） | 15.577 ms / 1.352x | 明显更差 |
| M128/N512（均衡 E32 t4k） | 14.600 ms / 1.448x | 差于 M256/N256 |
| M128/N512（E32 t8k） | 1.387x | 差于 1.437x |
| M128/N512（E32 t16k） | 1.321x | 差于 1.347x |
| M256/N256（E896 t4k） | 1.303x | 差于 M128/N512 的 1.416x |
| M256/N512 | **编译失败** | L0C cc overflow，见 4.1 |
| M512/N256 | **配置拒绝** | `M ≤ 256` 硬约束 |
| dispatch block 128 / 512 | 1.561x / 1.559x | 与 256 同档，无影响 |
| wave 8 | 1.425x | 更差 |
| wave 64 | 1.569x(t4k) / 1.350x(t16k) | 与 32 同档，需整专家入 wave，不取 |

**注意约束**：`config.py` 要求单 kernel forward 的
`fc1_gemm_block_size_m == fc2_combine_block_size_m`，改 M 必须两边一起改。

### 6.2 被证伪的假设

- **"大 T 下应改用大 N"**：我推理 T=16k 时激活工作集 1.88 GB 超 L2、
  A 重读 24 次退化为真 HBM 流量，故应减少 n_tiles。
  **实测证伪**：E32 t16k 用 M128/N512 是 1.321x，反而差于 M256/N256 的 1.347x。
  原因是 W 重读翻倍的代价超过 A 重读减半的收益。
  唯一的例外是 **E896 t16k**，M256/N256 (1.563x) 确实优于 M128/N512 (1.533x)。
- **W-stationary 遍历**：见 2.3，流量数学上不变，未实施。

### 6.3 dispatch 去重（用户建议方向，已量化但未实施）

用 `torch.topk` 均衡路由实测（T=4096，非解析估算）每 token 命中的不同目标 rank 数：

| 用例 | 不同 rank 数 | 占 topk 比例 | 传输降幅 |
|---|---:|---:|---:|
| E32 / top16 | 7.592 | 47.4% | 2.11x |
| E896 / top16 | 7.068 | 44.2% | 2.26x |

跨卡传输可从 411 MB/卡 降到 195 MB/卡，**方向本身成立**。

**但不实施，因为它不能提升 MAC ratio**：接收行数仍是 `T x topk`
（本地拷贝把行展开回来喂给各专家的 GEMM），FC1/FC2 的 33.8 GB MTE2 流量
**一个字节都不会少**。它改善的是通信暴露时间，而 PMU 显示通信不是当前瓶颈
（MTE3≈0、vector 空闲）。若将来 MTE2 瓶颈被解开、通信重新暴露出来，
这个优化才有价值。

---

## 7. 未解决：E896 + MoonEP 编译阻塞

### 7.1 现象与变量隔离（compile-only，不占 NPU）

| 配置 | 编译耗时 |
|---|---:|
| E32 非 MoonEP | 24 s |
| E32 + MoonEP | 33 s |
| E896 非 MoonEP | 28 s |
| **E896 + MoonEP** | **> 1200 s（未完成）** |

**只有 E896 与 MoonEP 的组合爆炸**，单独任一维度都正常。

### 7.2 确认是编译器 CPU-bound

`bishengir-compile` 子进程持续 100% CPU（父 python 仅 6%，
一开始我误看成父进程而以为是阻塞，实为编译器在算）。

### 7.3 已排除的两个候选根因（都做了实测，不要重试）

1. **`_single_moonep_b0` 的 `tl.static_range(0, E, BLOCK_E)` 全展开**
   （`fused_forward.py` 传 `min(32, NUM_EXPERTS & -NUM_EXPERTS)`）：
   E896 时 `896 & -896 = 128` 但被 min 压成 32 → **28 次全展开**（E32 只有 1 次）。
   改成 `min(128, ...)` 使展开降到 7 次后，**编译仍 >4.5 分钟未完成，排除**。
2. **`_moonep_b3_destination` 的 O(B x BE) 选择排序**
   （`for slot in range(B)` 内含 `tl.max`/`tl.argmax`，
   E896 时 B=112、BE=`next_pow2(896)`=1024，乘积 114688，是 E32 的 896 倍）：
   诊断性地把 B 压到 4 后，**编译仍 >3.5 分钟未完成，排除**。

### 7.3b 第三个被排除的根因：MoonEP 规划整体

把内联的 b0 + b2 + b3 + alloc_cumsum **全部移除**后重编（诊断性，破坏正确性），
编译**仍 >3.5 分钟未完成**（正常配置 28-33 s）。

**所以 MoonEP 规划三阶段整体都不是爆炸源**，爆炸在 MoonEP 的**数据路径**侧。

这同时否定了 7.6 中"把规划拆成独立小 kernel"的建议 —— 拆了也不会解决编译问题，
下一位不要按那条走。

已知 MoonEP 数据路径的两个膨胀点（`fused_forward.py:1512` 附近）：

1. `for replica_kind in tl.static_range(2 if MOONEP else 1)` ——
   MoonEP 下把**整个 FC1 流水实例化两份**（home 权重表 + replica 权重表），
   因为"Ascend block-pointer pass 无法合并 home/replica 指针"。
2. `full_rows = (end - begin == WAVE_WINDOWS * BLOCK_M) & (not MOONEP)` ——
   MoonEP 下 `full_rows` 恒为 False，FC1 只能走**带 mask 的通用路径**，
   其代码量显著大于 no-mask 快路径。

两者叠加解释了 MoonEP 使 `kernel.source` 膨胀 63%（718 KB → 1168 KB）。
但**仍未解释 E896 特异性**（E32+MoonEP 同样有这两个膨胀点却 33 s 编完），
所以推测是"MoonEP 的大 IR × E896 的大 constexpr 宽度"在某个超线性 pass 上相乘。
下一位应优先拿 per-pass timing 定位，而不是继续猜构造。

### 7.3c 第四个被排除的根因：FC1 `full_rows` 恒假分支

`fused_forward.py` 中 `full_rows = (end - begin == WAVE_WINDOWS * BLOCK_M) & (not MOONEP)`。
我推测 MoonEP 下它虽恒为 False 但是**运行时值**，Triton 会把 no-mask 快路径和
mask 路径两份都 trace，叠加 `replica_kind` 的 2 份 = FC1 实例化 4 份。

改成编译期 `if MOONEP: full_rows = False` 后实测（E32+MoonEP t4k）：

| | source | ttir | 编译 |
|---|---:|---:|---:|
| 改前 | 1167607 | 573906 | 33 s |
| 改后 | 1166212 | **573906（逐字节相同）** | 37 s |

**ttir 完全一致 —— Triton 本来就已经折叠了这个分支，该假设错误，改动零收益，已回滚。**
E896+MoonEP 在该改动下编译仍 >3.7 分钟未完成。

### 7.3d 重要观察：是超线性慢，不是死锁

standalone 用 90 分钟预算跑 `/tmp/kimi_u18/e896_moonep_kernel.mlir`，
持续 100% CPU 跑到 **20 分钟以上仍在推进**（无死锁特征、无内存爆炸）。
所以这是**编译时间超线性增长**问题，理论上给足预算可能能编出来，
下一位可以先用一个很长的预算（如 2-4 小时）确认它到底能不能收敛 ——
如果能，短期可用"预编译 + 缓存 npubin"绕过，不必先解决编译器问题。

### 7.4 IR 体量对比（线索）

| 配置 | kernel.ttir | kernel.source |
|---|---:|---:|
| E32 非 MoonEP | 433 KB | 718 KB |
| E32 + MoonEP | 574 KB | **1168 KB** |
| E896 非 MoonEP | 444 KB | 725 KB |
| E896 + MoonEP（卡住的 kernel.mlir） | — | 706 KB |

MoonEP 把 source 撑大 63%，但 E32+MoonEP 仍 33s 编完。
推测是 **MoonEP 的大 IR 体量 × E896 的大 constexpr 宽度**共同触发某个超线性 pass，
而非单一构造。

### 7.5 已备好的复现材料（下一位直接用）

卡住的 MLIR 与完整编译器 argv 已保存，**可脱离 8 卡环境独立复现**：

- MLIR：`/tmp/kimi_u18/e896_moonep_kernel.mlir`（706 KB）
- argv：`/tmp/kimi_u18/bishengir_args.txt`

独立复现命令（已验证能复现卡顿）：

~~~bash
/home/vllm_kimiw/.venv-udma/lib/python3.11/site-packages/ascendnpuir/bin/bishengir-compile \
  /tmp/kimi_u18/e896_moonep_kernel.mlir \
  --target=Ascend950DT_9582 --enable-auto-multi-buffer=True \
  --enable-auto-bind-sub-block=False --disable-ffts --set-workspace-multibuffer=0 \
  --limit-auto-multi-buffer-of-local-buffer=no-l0c --limit-auto-multi-buffer-buffer=only-cube \
  --enable-mixed-cv=True --disable-auto-inject-block-sync=True --enable-auto-blockify-loop \
  --enable-hfusion-compile=true --enable-triton-kernel-compile=true \
  --link-aicore-bitcode=.../libdevice.10.bc \
  --link-aicore-bitcode=.../meta_op.mix.aic.c310.bc \
  --link-aicore-bitcode=.../meta_op.mix.aiv.c310.bc \
  --mlir-timing --mlir-timing-display=list \
  -o /tmp/out --enable-vf-merge-level=1
~~~

### 7.6 建议的下一步

1. 用上面的 standalone 命令加 `--mlir-timing` 跑到**真正结束**（本轮给过 90 分钟预算），
   拿到 per-pass 耗时表直接锁定热点 pass。这是最高效的路径，不要再靠猜。
   环境无 gdb/py-spy，无法采样栈，只能靠 pass timing。
2. 若确认是某个 pass 超线性，尝试用 `--mlir-disable-pass=<name>` 或对应开关绕过。
3. **不要**再试"把 MoonEP 规划拆成独立 kernel" —— 7.3b 已实测证明移除整个规划
   也不能解决编译阻塞。
4. 若 pass timing 指向 FC1 的 `replica_kind` 双实例化，可考虑让 home/replica
   共用一份 FC1 代码、用运行时选择权重基址（需先确认该 Ascend block-pointer
   限制在当前工具链版本是否仍然存在——原注释可能已过时，就像 al.sort 的 A5 限制一样）。

---

## 8. 测试

**只跑 forward 测试**（用户明确要求不跑 backward）：

~~~bash
python -m pytest tests/kernel/forward tests/function tests/fstage/test_f0b_probes.py \
  tests/layer/test_fwd_phase_timing.py -q -k "not slow"
~~~

结果：**606 passed**。

已知与本轮无关的基线失败（在**未改动的** `64d8c03` 上同样失败，已用 git stash 验证）：
`tests/fstage/test_mega_bwd_probes.py::test_mega_probe5_ub_pingpong`，以及全量跑时
约 92% 处的另一个 backward 用例。两者都在 backward 侧。

---

## 9. 复现入口

单点：

~~~bash
python benchmark/layer/profile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-trimmed-top16-w8-t4k --benchmark-only \
  --fc1-block 256 256 128 --fc2-block 256 256 128 \
  --dispatch-block 256 --wave-windows 32 --output-dir /tmp/point
~~~

均衡用例 id：`performance-fwd-kimi-k3-trimmed-top16-w8-{t4k,t8k,t16k}`（E32）、
`performance-fwd-kimi-k3-w8-{t4k,t8k,t16k}`（E896）。
偏斜用例把 `-trimmed-top16`/无后缀换成 `-trimmed-skewed`/`-skewed`，MoonEP 加 `--moonep`。

compile-only（不占 NPU，用于编译问题迭代）：

~~~bash
python benchmark/layer/compile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-skewed-w8-t4k --moonep \
  --fc1-block 128 512 128 --fc2-block 128 512 128 \
  --dispatch-block 256 --wave-windows 32 --output-dir /tmp/probe
~~~

**跑测试前务必等 NPU 空闲**，benchmark 有 occupancy gate 会直接拒绝：

~~~bash
npu-smi info | grep -c "No running processes"   # 需为 8
ps -ef | grep profile_single_kernel | grep -v grep | wc -l   # 需为 0
~~~

本轮有两次点被上一个点的收尾进程挡住而误报 FAIL，串行跑务必加 idle 等待。
另注意 `pgrep -f <pattern>` 会匹配到自己的 shell，用
`ps -ef | grep "[p]attern"` 避免自匹配。

---

## 10. 本轮产物路径

- 均衡六点：`/tmp/kimi_u18/mx_*`
- block 扫描：`/tmp/kimi_u18/sw_*`、`/tmp/kimi_u18/bt_*`
- 全覆盖回归：`/tmp/kimi_u18/fc_*`
- pipe profile（含 PMU）：`/tmp/kimi_u18/prof_trim16_t4k`、`/tmp/kimi_u18/prof_wave32`
- compile-only 对照：`/tmp/kimi_u18/cc_*`
- 卡住的 MLIR 与 argv：`/tmp/kimi_u18/e896_moonep_kernel.mlir`、`bishengir_args.txt`
- sqlite shim：`/tmp/kimi_u18/libshim/`
