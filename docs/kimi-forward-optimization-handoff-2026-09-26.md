# Kimi K3 fused forward 性能优化完整交接记录

生成时间：2026-09-26，Asia/Shanghai。本文给下一位 AI 接手使用，保留问题背景、实验口径、源码状态、工具链、每类尝试的结果、否决理由和原始产物路径。性能矩阵的逐卡遥测和 50 个原始样本不在本文重复展开，见矩阵 Markdown/JSON。

> 2026-09-26 清理更新：本文第 2 节的 HEAD/源码指纹与第 5 节矩阵对应清理前的测量快照；当前代码、删除项、最终验证和遗留事项见 [清理与验收记录](performance/kimi-forward-cleanup-validation-2026-09-26.md)。原矩阵 9/12 的历史数据不覆盖。用户新增顺序：先清理、验证、提交、push，之后补测三个 full + MoonEP 点并确认根因。

## 1. 原始问题与用户后来明确的要求

### 1.1 原始用户输入（问题原文整理）

用户最初给出的目标是：

- 当前 main 分支的未裁剪专家、8 卡 Kimi K3 fused Triton forward 明显慢于裁剪专家版本。
- 裁剪专家相对 Torch 的加速比约 1.6x，但未裁剪版本达不到。
- 用户给出的初始数据：
  - main：fused forward median 21.552 ms，mean 21.585 ms，P95 21.832 ms；
  - PR #68：median 17.693 ms，mean 17.745 ms，P95 18.074 ms。
- 要求基于 origin/main 新建 worktree 测试和修改，分析根因并优化性能。
- 可以使用 /home/vllm_kimiw/repo/msopprof 做流水和性能瓶颈分析。
- 可以参考记忆使用 sys_count 打点。
- 跑测试前等待 NPU 空闲。
- 使用 UDMA 环境时必须参考并 source udma 虚拟环境。

### 1.2 后续任务边界

用户随后明确要求先暂停新的优化探索，跑系统性能矩阵：

- 使用当前最优的 fused forward Triton kernel；
- Kimi K3，8 卡；
- 每卡 token 长度 4k、8k、16k；
- 裁剪专家 E32 / 非裁剪全量 E896；
- MoonEP 开 / 关；
- 共 12 个组合；
- 每个数据点记录 case、token 数、top-k、形状和 block 配置、MoonEP 开关、加速比、逐卡功耗和频率；
- 结果写入 MD；
- 路由采用现有偏斜路由，以实际触发 MoonEP 迁移；
- 裁剪版 top-k 固定为 16，与全量版一致。

因此当前优先级是完成并审计矩阵报告；不要把新的 kernel 优化猜想混入矩阵，也不要用其他 block 或其他 compiler plan 的结果替换原始 12 点。

### 1.3 关于 forward、autograd 和 backward

这次基准是 forward-only：

- benchmark 直接调用 op.forward 的 return_saved=False 路径；
- hidden、权重和 routing 输入不需要梯度；
- 不调用 backward，也不测 autograd graph、saved tensor 或梯度通信；
- benchmark 源码中的 torch.autograd.profiler.record_function 只是在 profiling 分支中给 CPU/NPU trace 加标签，不代表发生了 backward；
- --benchmark-only 跳过 profiling 分支；
- save_fc1_dtype=bf16 是配置字段，return_saved=False 时不保存训练中间值；
- 不应把本次 forward 延迟解释成训练 forward+backward 延迟。

## 2. 当前仓库和交付状态

### 2.1 原矩阵测量时的 worktree、分支和源码

工作目录：

/home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf

当前分支：

codex/kimi-forward-perf

当前 HEAD：

7a33afb Record compiler and direct-pull performance experiments

基线和提交顺序：

- origin/main：eb737e7
- 5ad5f02：vectorize routing metadata、减少 wave entry scan、分片 return acquire
- 9142dba：phase timing 节点先做 accuracy gate
- 8046e45：把 route inverse reset 从 histogram 中分离
- 07a5de8：把 route inverse reset 放到 fused kernel 前的 host stream
- 7074180：为 EXP-L 记录和 reset 机制增加测试
- 7c93cb1：Reduce sparse-expert overhead in Kimi fused forward
- 7a33afb：记录 compiler 和 direct-pull 实验

当前 worktree 的未提交变化是 benchmark/报告相关，不是新的生产 kernel 优化：

- benchmark/layer/profile_single_kernel_forward.py：增加 --record-host-intervals，记录每个 rank 的 candidate/baseline host 调用区间；
- config/_shapes.py：为 skewed 和 top16 matrix case 增加 8192 token 档；
- benchmark/layer/run_kimi_forward_matrix.py：矩阵运行器；
- benchmark/layer/summarize_kimi_forward_matrix.py：矩阵汇总器；
- docs/performance/kimi-k3-forward-matrix-2026-09-26.md；
- docs/performance/kimi-k3-forward-matrix-2026-09-26.json；
- 本交接文档和人类简报。

原矩阵冻结的 fused kernel SHA256：

cbeae461ed961ee9c5785faf1ed80fb1ed5c4d74d348909977ddb4f2f4faf5fb

矩阵中共同使用的主要源码指纹：

- src/mega_moe/kernels/fused_forward.py：cbeae461ed961ee9c5785faf1ed80fb1ed5c4d74d348909977ddb4f2f4faf5fb
- src/mega_moe/ops/forward.py：dbc053daf793610dff3fe5c0b5ce9846a48435d654d061ed23f895b3239d7273
- src/mega_moe/kernels/fused_moonep.py：2ef421ad78a4fceff3d56326a55f42ad38e83213177da96347fe810f387e2561
- benchmark/layer/_kimi_routes.py：b09f9b2b0d4147a5599b1c9a2c6ca9aff070295f7164508652e393082b57c9d7

### 2.2 文档和矩阵是否完整

已经写入的矩阵文档：

- 主报告：docs/performance/kimi-k3-forward-matrix-2026-09-26.md
- 结构化报告：docs/performance/kimi-k3-forward-matrix-2026-09-26.json

两份文件都写入了已经得到的 9 个有效点、P95、输入参数、block、MoonEP、逐 rank 功耗/频率、正确性门、占用检查、版本和原始产物路径。

但是主矩阵还不完整：

- 计划数据点：12；
- 有效性能数据：9；
- 缺失：3；
- 缺失的全部是 E896 + MoonEP：
  - full_t4k_moonep1；
  - full_t8k_moonep1；
  - full_t16k_moonep1。
- 这三个原始点都停在首次正常 correctness 调用触发的后端编译，没有完成 correctness gate，也没有 forward 延迟、加速比或有效计时期功耗/频率；
- 不能填零，不能称作性能失败，不能用裁剪 MoonEP 或其他 block 的结果替代；
- 因而“12 点全部完成”的要求尚未满足。

旧的优化分析文档：

docs/kimi-forward-performance-analysis.md

它详细记录了 2026-09-25 之前的优化和诊断，但它不是新的 4k/8k/16k、E32/E896、MoonEP 开关 12 点矩阵。不要把旧文档中的 4k uniform/legacy 结果和新 skewed 矩阵拼成一个完整表。

旧的结构化实验记录：

docs/performance/kimi-forward-2026-09-25.json

它保存了原优化期间的 measurements、local probes、L2/PMU、frequency、UDMA、compile 和 direct-pull 记录，下一位 AI 应优先读取这个 JSON，而不是凭记忆重做实验。

### 2.3 最新未完成补测

为补齐 full + MoonEP，做过以下额外尝试，均没有产生可写入主矩阵的结果：

1. 与原点相同的 full E896/t4k/M128-N512/K128/wave32，使用原始 profile，900 秒预算。后端停在 linalg_to_bin_enable_npu_compile_910_95，returncode=-15。
2. compile-only，E896/t4k/MoonEP，M256/N256/K128，wave32，240 秒预算超时。
3. compile-only，E896/t4k/MoonEP，M128/N256/K128，wave32，240 秒预算超时。
4. 另建的 isolated largest-first wrapper 原意是对 full+t4k+MoonEP 只改变 compiler plan_memory_strategy=largest-first，但实际启动的 bishengir-compile 命令行没有出现 --plan-memory-strategy=largest-first，说明该 wrapper 没有把参数传入真正的 Triton launch；这次不能作为 largest-first 结论。它仍在普通 compiler plan 下持续约 5 分钟，没有 status.json 或 benchmark_result.json。文档任务开始时该 probe 仍占用 NPU/编译资源，随后结束并完整保留目录：
   - /tmp/kimi_forward_matrix_20260926_completion/full_t4k_moonep1_largest
   - wrapper：/tmp/kimi_forward_matrix_20260926_completion/profile_largest_wrapper.py

这些补测不能证明 CANN 版本错误，也不能证明性能差；它们只能证明当前 full+MoonEP 特化编译压力仍没有在规定预算内解决。

## 3. 固定硬件、软件和环境

### 3.1 硬件

- 8 x Ascend950DT；
- world size=8；
- 每 rank 32 AICore programs、64 AIVector programs；
- 对称堆大小：每 rank 16 GiB；
- 设备运行前和运行中检查 npu-smi，无外部进程时才开始每个点；
- 逐卡采样 rank/device 0…7。

### 3.2 UDMA 环境和 CANN

Python：

/home/vllm_kimiw/.venv-udma/bin/python

虚拟环境：

/home/vllm_kimiw/.venv-udma

CANN runtime：

/usr/local/Ascend/cann-9.2.0-beta.2

矩阵使用的 NPU-IR compiler：

/home/vllm_kimiw/.venv-udma/lib/python3.11/site-packages/ascendnpuir/bin/bishengir-compile

compiler 版本：

bishengir-compile 1.2.0，Ascend NPU-IR commit b229bc6ccbcc，日期 2026-09-03，LLVM 19.1.7

运行时：

- Python 3.11.10；
- torch 2.10.0+cpu；
- torch_npu 2.10.0.post1.dev20260528；
- triton runtime 3.6.0；
- distribution metadata 中同时存在 triton_dist 3.4.0、triton_ascend 3.2.2、triton 3.5.0，实际 import 的 runtime 以 run_metadata.json 为准。

矩阵 environment.sh：

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
export TRITON_CACHE_DIR=/tmp/kimi_forward_matrix_20260926_completion/cache
unset MOE_ASH_ENGINE
unset ASCEND_LAUNCH_BLOCKING
~~~

早期普通 forward 有时把 CANN bin 放在 PATH 前面；矩阵最终记录了真实 compiler 路径和 device library 指纹，不能只看 ASCEND_HOME_PATH 判断工具链。

### 3.3 CANN/UDMA 判断

已确认：

- 正常普通回传 forward 的 runtime 来自 CANN 9.2.0-beta.2；
- 正常路径含 rtsGetHardwareSyncAddr 符号，并成功通过 NPU correctness；
- 早期 UDMA 链接失败的根因是 PATH 选中了 CANN 下的 compiler，却链接了 UDMA/上游特有的 ACLSHMEM device symbol；
- 显式把 ascendnpuir/bin 放到 PATH 首位并链接配套 meta-op bitcode 后，UDMA 原型可以编译和过 correctness；
- 因此“rtsGetHardwareSyncAddr 限制”不是当前普通 forward 失败的证据，早期环境异常与 UDMA device-library 混用也不能合并解释；
- current full+MoonEP 缺失点发生在后端编译阶段，不能写成硬件端运行错误。

## 4. 矩阵输入、路由和计时协议

### 4.1 12 个主矩阵点

每卡 token：

4096、8192、16384

专家：

- trimmed：E=32，E/rank=4；
- full：E=896，E/rank=112。

top-k：

16，裁剪和全量完全相同。

hidden/FFN：

- hidden=3584；
- FFN=3072。

形状：

- hidden/output：[T, 3584]；
- selected experts 和 routing weights：[T, 16]；
- trimmed W1：[4, 3584, 6144]；
- trimmed W2：[4, 3584, 3072]；
- full W1：[112, 3584, 6144]；
- full W2：[112, 3584, 3072]；
- 在分布式逻辑中全局专家数分别为 32、896。

dtype：

- hidden、W1、W2、output：BF16；
- routing weights：FP32；
- selected experts：INT32；
- activation：SwiGLU；
- capacity factor=1.6875；
- drop_frac=0。

随机种子：

- local weights：42+rank；
- inputs：43+rank*1000；
- hidden 在 route logits 之前生成；
- MoonEP 开/关使用相同的确定性输入和路由生成逻辑。

### 4.2 偏斜路由

owner quotas：

[19, 27, 11, 11, 15, 15, 15, 15] / 128

所有 global experts 都 active。每个 token 的 top-k=16 个 expert 不重复；在每个 owner 内按 rank%epn 的 cursor 轮询 local expert。

每 rank 接收的原始 owner route 总数：

- T=4096：[77824, 110592, 45056, 45056, 61440, 61440, 61440, 61440]；
- T=8192：上面的 2 倍；
- T=16384：上面的 4 倍。

E32 和 E896 使用相同 owner 负载分布，但 expert identity 和权重数量不同。E32 的 MoonEP 规划实际 copies/rank 为 [0,0,1,1,1,1,1,1]，每次 forward 传输权重 396361728 bytes。MoonEP replica cache 关闭，每次 forward 刷新。

### 4.3 fused 和 Torch 的边界

fused measured boundary：

router 之后的完整 forward，包括：

1. route metadata；
2. dispatch；
3. FC1；
4. weighted SwiGLU；
5. FC2；
6. combine；
7. forward 内的 workspace/route_to_send reset。

输入生成、权重生成、operator 初始化、JIT 首次编译、correctness gates 不计入 50 次 event 延迟。

Torch baseline：

Torch-NPU grouped-GEMM + HCCL，始终不执行 MoonEP。MoonEP 开/关的 fused 对比不能解释成 Torch 也开/关了 MoonEP。

计时：

- warmup=5；
- measured iterations=50；
- clock=npu_event；
- 每次样本取 8 rank MAX；
- speedup=Torch grouped-GEMM + HCCL median / fused median；
- P95 从 50 个原始样本按排序后位置 0.95*(n-1) 线性插值。

host interval：

- 每 rank 记录 candidate 和 baseline；
- 记录从调用进入开始，到 event 完成并进入 rank-MAX 前；
- 摘要工具排除 5 个 warmup，并只取各 rank 自己的 50 个 measured intervals 的并集；
- 不把 inter-call 空隙、启动等待或整个 telemetry 文件错误归给另一个 rank；
- DCMI 只是离散采样，不是逐 kernel 周期积分。

correctness：

每个有效点在 timing 前通过：

- normal routes；
- zero-receive/empty-expert；
- negative/out-of-range all-drop；

rtol=0.05，atol=0.05。还要通过运行期间 device occupancy gate。原 benchmark 没有额外的 changed-hidden gate；旧实验 JSON 里出现的 changed-input 数字来自其他 prototype，不应写入主矩阵 correctness。

## 5. 现有矩阵结果

以下是主矩阵 9 个有效点的重算值。时间顺序为 fused median / mean / P95，再给 Torch median / mean / P95。

| point | block FC1/FC2 | wave | fused ms | Torch ms | speedup |
|---|---|---:|---:|---:|---:|
| trimmed_t4k_moonep0 | 256,256,128 / 256,256,128 | 16 | 22.476 / 22.489 / 22.958 | 32.492 / 32.523 / 32.635 | 1.446x |
| trimmed_t4k_moonep1 | 256,256,128 / 256,256,128 | 32 | 14.379 / 14.358 / 14.818 | 32.439 / 32.469 / 32.572 | 2.256x |
| full_t4k_moonep0 | 128,512,128 / 128,512,128 | 16 | 22.470 / 22.535 / 23.387 | 33.138 / 33.173 / 33.218 | 1.475x |
| trimmed_t8k_moonep0 | 256,256,128 / 256,256,128 | 16 | 45.306 / 45.251 / 45.661 | 64.246 / 64.346 / 65.168 | 1.418x |
| trimmed_t8k_moonep1 | 256,256,128 / 256,256,128 | 32 | 29.340 / 29.410 / 30.240 | 64.054 / 64.101 / 64.179 | 2.183x |
| full_t8k_moonep0 | 128,512,128 / 128,512,128 | 16 | 43.670 / 43.684 / 44.232 | 65.473 / 65.515 / 65.557 | 1.499x |
| trimmed_t16k_moonep0 | 256,256,128 / 256,256,128 | 16 | 95.512 / 95.564 / 96.300 | 126.903 / 126.980 / 127.113 | 1.329x |
| trimmed_t16k_moonep1 | 256,256,128 / 256,256,128 | 32 | 60.215 / 60.200 / 60.728 | 127.107 / 127.189 / 127.543 | 2.111x |
| full_t16k_moonep0 | 128,512,128 / 128,512,128 | 16 | 86.088 / 85.941 / 86.581 | 129.420 / 129.489 / 129.757 | 1.503x |

缺失：

| point | block | reason |
|---|---|---|
| full_t4k_moonep1 | 128,512,128 / 128,512,128, wave32 | 900 s backend compile budget exceeded before correctness |
| full_t8k_moonep1 | 128,512,128 / 128,512,128, wave32 | 900 s backend compile budget exceeded before correctness |
| full_t16k_moonep1 | 128,512,128 / 128,512,128, wave32 | 900 s backend compile budget exceeded before correctness |

### 5.1 逐卡遥测在哪里

每个有效点的：

- telemetry.jsonl；
- benchmark/host_call_intervals_rank0.json 至 rank7.json；
- benchmark/benchmark_result.json；
- benchmark/run_metadata.json；
- benchmark/device_occupancy.jsonl；

都在 /tmp/kimi_forward_matrix_20260925_1426/<point>。

主 MD 已将每个 valid point 的 rank0…7 frequency 和 power 以 median [min–max], n 写出。Power 是 DCMI raw unit 0.1 W 转成 W；frequency type=7。缺失点没有合法计时期遥测，不应使用 compile-phase telemetry 伪造 forward 功耗。

## 6. 从问题到瓶颈的分析过程和工具

### 6.1 公平性先行

最早把 E32/top-k=8 的 1.6x 和 E896/top-k=16 直接对比是不公平的。固定 E32、token、hidden、FFN，只把 top-k 从 8 改成 16：

- top-k=8：fused 6.841 ms，Torch 11.642 ms，1.702x；
- top-k=16：fused 14.307 ms，Torch 21.306 ms，1.489x；
- fused 延迟增长 2.066x，Torch 增长 1.821x。

因此 1.6x 目标的一部分差异来自 top-k 和 route count，而不是只有 E32/E896。后续矩阵将 top-k 固定 16。

### 6.2 host 回归和 AST shim

用于 routing、scatter、wave task、return counter 和 JIT argument binding 的 host tests：

- 曾有 routing optimization 文档记录 391 passed；
- 叠加 timing/profile/saved/compile 相关测试后，主机回归记录为 494 passed；
- 保存分支和归约相关针对性回归记录为 174 passed；
- Python syntax check 和 git diff --check 通过。

这些 host tests 只能验证整数公式、任务表覆盖、signal protocol、调用绑定和边界，不能证明 Ascend lowering、UB、跨 rank 可见性或 NPU 性能。必须看硬件 correctness 和 event benchmark。

### 6.3 msopprof / Pipe / L2

使用 /home/vllm_kimiw/repo/msopprof 和现有 profile 工具做过：

- Pipe/Cube utilization；
- matched top-k、matched tile 的 L2 profile；
- routing/wave phase 的辅助 trace。

一个独立 FC1 helper 用 msopprof --aic-metrics=PipeTimeline，命令返回成功但产物没有 timeline.bin 或 TRACE 数据块，只有基础耗时；追加 Default,PipeTimeline 后得到 PMU 表，但仍没有完整流水事件。因此不能声称拿到了完整 PipeTimeline trace。

可复核的结果：

- 完整 fused 的 AIC MAC ratio 中位数约 81.95%；
- Cube utilization 约 97.22%，分母不同，不能说 97% 的时间都在有效 MAC；
- matched L2 profile 的 Victim Rate：full 11.69%，trimmed 7.45%；
- Cube local L2 read-miss raw count：full 20.38M，trimmed 7.05M；
- l2_cache.csv Hit Rate 为 N/A，raw miss 不能直接变成整体 hit rate；
- profile kernel duration 与端到端 event 的大小顺序不完全一致，所以缓存数据只能支持“缓存行为不同”，不能单独归因全部端到端差额；
- 112 experts x 585 rows 的独立 FC1 helper：core0 Cube ratio 91.42%，MTE2 ratio 96.70%，但没有远端通信，不能代替 8 卡 full profile。

### 6.4 SYS_CNT / phase timing

通过 MOE_FWD_TIMING=1 和 compile-only timing variant：

- phase stamps 放入同一 fused launch；
- routing metadata 约 0.23 ms；
- 主要时间仍在 wave pipeline；
- fc1_cube_wall、fc2_wave_wall、return_wait wall 都包含依赖等待，不是纯 GEMM 或纯 Scalar；
- 当前正式矩阵使用 MOE_FWD_TIMING=0，不将 timing binary 的 phase 数字当成生产 forward 延迟；
- timing/saved 二进制有独立 UB 和编译约束，必须单独 correctness。

### 6.5 频率和功耗

使用 DCMI v2 只读接口：

- 约每 10–20 ms 轮询；
- frequency type=7；
- power raw unit 0.1 W；
- host monotonic interval 对齐；
- 不修改频率、功耗或限频配置。

诊断 full E896/top-k16：

- fused 频率为 1350–1650 MHz；
- 320 个 full fused 采样中 201 个低于 1650 MHz；
- 同轮 Torch 448 个采样均为 1650 MHz；
- matched E32/top-k16 fused 312 个采样均为 1650 MHz；
- full fused power median 约 894.7 W，trimmed top16 约 855.8–876.1 W；
- 28/30 core 的频率可恢复 1650 MHz，但延迟分别为 16.098/16.499 ms，慢于 32 core 的约 15.1 ms。

结论：full 的降频与高工作集/功耗相关，是差额的一项证据；不能仅凭离散 DCMI 样本算出每个优化项的精确贡献。

### 6.6 编译器和进程问题

所有硬件点都先等待 NPU idle，运行期间由 benchmark occupancy monitor 写 device_occupancy.jsonl。普通 forward 成功时没有外部 NPU 进程。

遇到的问题：

- npu-smi 查询曾出现一次 timeout；后续 DCMI 采样完成，正式表只用完成的读数；
- 长编译触发 faulthandler 每 120 s 的重复线程栈，但不表示 NPU kernel 崩溃；
- full+MoonEP 的后端编译在 linalg_to_bin_enable_npu_compile_910_95 长时间占用 compiler；
- 进程被 timeout/SIGTERM 时只记录 returncode=-15 和 stop_reason，未把它当成 correctness fail；
- earlier fallback wrapper 有相对路径、预创建 output、source relative_to 错误；后来使用原生 profile/compile-only 入口；
- 一个名为 largest-first 的 isolated probe 曾运行约 5 分钟没有结果；后检查确认实际 compiler argv 没有传入该策略参数，因此它只能作为无效诊断产物保留，不能作为策略结论。

## 7. 有效果并保留的生产方案

### 7.1 Routing metadata 和 compact wave

提交 5ad5f02 及后续修改做了：

- stable cursor 改为 core/expert block exclusive scan；
- destination、wave offsets 和 pull starts 复用归约；
- metadata 扫描向量化；
- E>=128 的 non-MoonEP scatter 使用 32-route block、32x32 pairwise matching；
- E<128 保留 dense path；
- 原 routing order、stable send order、route inverse 语义保持；
- wave_task_offsets 和 wave_tasks 由 routing 阶段构建；
- FC1、FC2、dispatch、return 消费同一个 compact task table；
- 任务表按 wave-major、expert-major 保持旧调度顺序；
- 容量溢出时不写 table，沿用全局 capacity guard。

E896、R=65536 时，expert-ID 扫描由约 28 遍减少到两个 vector lane 各一遍；pairwise 约 4.19M 元素，旧 dense matching 约 58.72M 元素。不能把元素比直接当成端到端 speedup，因为仍有 histogram、gather、loop 和跨卡等待。

### 7.2 小 bucket worker rotation

E896 的 source/expert bucket 平均很小，历史第一个 source tile 总落到同一 peer lane，导致其余 lane 空转。只对单 tile bucket 按 expert 号轮转 worker，大 bucket 保留旧分配。

记录：

- full baseline variant skip_idle_fc1：16.046 ms；
- dispatch_expert_rotation：15.388 ms；
- repeat：15.185 ms；
- final Pipe 配套：15.130 ms。

改善约 0.861 ms，约 5.4%，但不同进程存在测量波动。原因是把小 bucket 的 dispatch 发射从固定 lane 串行化变成了更均匀的 lane 工作。

### 7.3 M128/N512 和尾块路径

full E896、top-k16、M256 需要计算 688128 个 tile rows，而有效 route rows 约 524288，填充约 31.25%；M128 时约 9.62%。trimmed M256 填充约 1.46%，所以同一 tile 不适合两个专家规模。

代表结果：

- full M256/N256：17.677 ms，1.224x；
- full M128/N256：18.033 ms，1.215x；
- full M128/N512/K128：稳定约 15.1 ms，约 1.43–1.45x；
- full dispatch M512：16.293 ms，1.332x；
- full dispatch M128：15.159 ms，接近但不优于 dispatch M256；
- trimmed top16 M256/N256：14.307 ms，1.489x；
- trimmed top16 M128/N512：14.637 ms，1.455x。

因此主 full block 保留 M128/N512/K128，主 trimmed block 保留 M256/N256/K128。

代码还让单个满 row tile 走 no-mask FC1 path，跳过没有 FC1 tile 的 core 的 helper 初始化；尾行仍保留 mask 和数值语义。这个路径与 compact task、worker rotation 一起形成最终收益，单独结果容易受噪声影响。

### 7.4 Return checker 和四累加归约

Return path：

- 原实现每个 vector lane 反复扫描全部 destination/wave；
- W8、约 21 waves、32 cores 时 remote counter acquire 约 10752 次；
- 新实现由 checker 分片 acquire 原始 counters，经 fence 发布本地 epoch，再由 reducer 等 checker slab；
- 仍保持 remote counter 的严格可见性和旧 epoch 防护；
- 单独收益不大，但减少了尾部空扫描。

Reduce：

- 把 top-k combine 的串行 FP32 accumulator 拆成四条连续 DMA/accumulator chain；
- local probe 0.278 ms -> 0.230 ms；
- paired full AB/BA：15.336 -> 15.242 ms；
- 50 对中 28 对新版本更快；
- FP32 加法顺序变化，完整 correctness gate 在 rtol/atol=0.05 下通过；
- saved FC1 path 保留单链，因为四链 + BF16 save compile-only 超过 300 s。

### 7.5 route inverse host reset

kernel 内 reset 在完整 E896 上出现 dropped route 错误。当前方案：

- launch 前同 stream 执行 self._route_to_send[:num_routes].fill_(-1)；
- scatter 覆盖有效 route；
- dropped route 保持 -1；
- 正确性通过；
- 计时包含这个 fill，不能把矩阵 fused 数字称为纯单个 kernel duration；
- 这是 correctness root fix，不应当声称它本身提升了性能。

## 8. 没有效果、被否决或仅诊断使用的尝试

以下数值来自 /tmp/kimi_forward_perf，除特别说明均为 E896、8 卡、T4K、top-k16、5 warmup/50 event，candidate median / Torch median / speedup。

### 8.1 基线和 block/dispatch

| 实验目录 | 配置 | 结果 | 否决原因 |
|---|---|---|---|
| pr68_exact_env | PR #68，M256/N256，legacy dispatch | 17.655 / 21.782 / 1.234x | 作为 baseline；与当前优化相比慢 |
| full_dispatch256_m256n256 | M256/N256，dispatch legacy | 17.642 / 21.685 / 1.229x | full 尾块填充和稀疏 bucket 不适合 |
| full_m128n256 | M128/N256 | 18.033 / 21.906 / 1.215x | N256 计算/流水不足 |
| full_m128n512_dispatch512 | M128/N512，dispatch512 | 16.293 / 21.709 / 1.332x | dispatch source tile 过大 |
| full_dispatch128_rotation | dispatch128 | 15.159 / 21.739 / 1.434x | 接近但没有稳定胜过 dispatch256 |
| full_m128n512 的不同 no-op/重复运行 | M128/N512/dispatch256 | 15.095–15.478 ms | 测量波动，不把单次最好值解释成新方案 |

### 8.2 FC1 和 task path

| 实验目录 | 配置 | 结果 | 否决原因 |
|---|---|---|---|
| skip_idle_fc1 | 跳过空 FC1 core helper | 16.046 / 21.775 / 1.357x | 只是保留在后续组合中，单独不够 |
| fc1_row_assignment_full | 改 row assignment | 15.436 / 21.684 / 1.405x | 没有超过 retained path |
| fc1_bounded_read_full | bounded read | 15.312 / 21.785 / 1.423x | 小于噪声范围，未保留为独立策略 |
| fc1_task_groups8_full | task groups=8 | 18.140 / 21.746 / 1.199x | 任务分组加重调度 |
| fc1_task_groups_full | 另一组 task grouping | 25.625 / 22.043 / 0.860x | 明显变慢 |
| full_m128n512_fc1_fullrow | full-row FC1 | 16.310 / 21.708 / 1.331x | standalone 方案变慢 |
| full_m128n512_fc1_fullrow_repeat | full-row repeat | 16.119 / 21.705 / 1.347x | 没有稳定收益 |
| full_m128n512_fc1_fullrow_allow_l0c | allow L0C multibuffer | 22.274 / 21.780 / 0.978x | UB/L0C 约束导致更慢 |
| full_m128n512_fc1_fullrow_w32 | full-row + wave32 | 16.419 / 22.148 / 1.349x | wave32 代价更高 |
| fc1_m64n1024k64_full | M64/N1024/K64 | 22.685 / 21.715 / 0.957x | tile 数和重复工作增加 |
| m64_n1024_act16_full_retry | M64/N1024/K128 | 31.662 ms | 更慢，未保留 |
| fc1_options_cube_merge | enable_cube_block_merge | local median 8.542 ms vs baseline 8.540 ms | 无收益且尾部样本变差 |
| fc1_options_no_preload | enable_preload=False | local 8.542 ms | 无收益 |
| fc1_options_k64 | FC1 K64 | local 9.977 ms vs K128 8.540 ms | 更慢 |
| fc1_loop_unswitch_fixpipe_probe | 把整行/尾行判断移出 K loop | local 10.375 ms | 数值正确但更慢 |
| fc1_weight_evict_last | weight load evict_last | local 8.541 ms；full 15.244 ms | 无稳定端到端收益 |

local probe 的参数是 world=1、112 experts、每 expert 585 rows、5 warmup/20 samples，只用于 helper 选择，不能与 8 卡端到端相加。

### 8.3 FC2、completion 和 return

| 实验目录 | 配置 | 结果 | 否决原因 |
|---|---|---|---|
| fc2_options_k | K128/K64/K256 | 4.484 / 5.817 / 6.579 ms | K128 最好 |
| fc2_n256_full | FC2 N256 | 15.264 / 21.679 / 1.420x | 不如 retained path |
| full_m128n512_k256 | K256 | 22.168 / 21.713 / 0.979x | 明显更慢 |
| fc2_home_nk_load | FC2 home N/K load | 16.220 / 21.691 / 1.337x | 变慢 |
| fc2_incremental_full | incremental FC2 | 15.162 / 21.712 / 1.432x | 与噪声相近，无稳定证据 |
| full_m128n512_fc2_fullrow | FC2 full-row | 17.080 / 21.886 / 1.281x | 变慢 |
| fc2_completion_relay | single Vector completion relay | 15.372 / 21.709 / 1.412x | 变慢，功耗仍约 895 W |
| fc2_completion_counter | merge all Cube completion into one ADD counter | 首轮 correctness 超过 120 s，无性能结果 | 未证明可见性/依赖协议 |
| return_chunk32k_full | enlarge return chunk | 15.177 / 21.744 / 1.433x | 无稳定改善 |
| weight_evict_last_full | return/weight eviction variant | 15.244 / 21.762 / 1.428x | 无稳定改善 |

### 8.4 wave、核心数和频率

| 实验目录 | 配置 | 结果 | 否决原因 |
|---|---|---|---|
| final_wave_windows_32 | wave=32 | 15.388 / 21.737 / 1.413x | 比 wave16 差 |
| final_wave_windows_64 | wave=64 | 15.816 / 21.704 / 1.372x | 更差 |
| dcmi_full_cores30 | AICore=30 | 16.499 / 21.988 / 1.333x | 频率改善但吞吐下降 |
| dcmi_full_cores28 | AICore=28 | 16.098 / 21.889 / 1.360x | 频率改善但吞吐下降 |
| dcmi_full_cores32 | AICore=32 | 15.211 / 21.949 / 1.443x | retained |

### 8.5 direct pull、UDMA 和编译器 plan

| 实验 | 结果 | 否决原因 |
|---|---|---|
| direct_pull_reduce_full | 直接从远端 FC2 symmetric memory 拉取/归约，默认 plan | compiler timeout，未执行 NPU |
| direct_pull_reduce_serial_full | 单链 direct pull，默认 plan | compile timeout；pass 停在 hivm-plan-memory-regbase |
| direct_pull_reduce_serial_largest_full | 单链 + plan_memory_strategy=largest-first | 23.872 / 21.915 / 0.918x | 通过 gate 但明显慢 |
| standard_serial_largest_full | 普通回传 + largest-first，capacity=1.25/旧 uniform case | 15.212 / 21.720 / 1.428x | plan 本身没有超过普通 path |
| standard_largest_full | 四链/普通回传 + largest-first | UB 1884160 bits > 1769472 bits | compiler UB overflow |
| udma_return_matched_toolchain | matched UDMA compiler 的 UDMA return | 15.829 / 21.990 / 1.389x | 比普通回传多约 0.532 ms |
| npuir_standard_full | matched UDMA/NPU-IR compiler 的普通回传 | 15.297 / 21.996 / 1.438x | 工具链一致性通过，但没有 1.6x |

早期 CANN compiler + UDMA device library 混用产生 aclshmem symbol 缺失；切换到配套 ascendnpuir compiler 后解决了链接问题。这个修复不能等价成性能优化。

### 8.6 saved、FP8/FP16 和 compile-only

compile-only 目录：

- /tmp/kimi_forward_perf/final_matched_workspace_compile
- /tmp/kimi_forward_perf/compile_full_npuir_toolchain
- /tmp/kimi_forward_perf/reduce_interleaved_timing_compile
- /tmp/kimi_forward_perf/reduce_control_bf16_compile
- /tmp/kimi_forward_perf/reduce_saved_fallback_bf16_compile
- /tmp/kimi_forward_perf/reduce_saved_fallback_fp8_n256_compile
- /tmp/kimi_forward_perf/reduce_saved_fallback_fp16_n256_compile
- /tmp/kimi_forward_perf/m64_n1024_act16_compile
- /tmp/kimi_forward_perf/skip_idle_fc1_compile
- /tmp/kimi_forward_perf/dispatch_rowlimit_full_compile

已成功的 compile-only 只证明对应 IR 能编译；保存分支没有被当成本次 return_saved=False 性能。

失败/超时的 saved 情况：

- M128/N512 FP8 saved 在旧归约和兼容处理后均未在 300 s 内完成；
- FP16 大分块也在 300 s 后超时；
- 旧版和新版送入 CANN 的 IR 去掉 debug location 后完全一致；
- 因此不能把该失败归因于某个单独的数学改动，也不能归因于 CANN 版本选错。

### 8.7 full + MoonEP 主矩阵缺失

原始配置：

- E896，top-k16，skewed route；
- FC1/FC2 M128/N512/K128；
- dispatch M256；
- wave32；
- MoonEP every-forward replica refresh；
- CANN/UDMA path 为矩阵最终记录的 ascendnpuir toolchain。

三个点都在后端 compile 中超过 900 s，returncode=-15。compile-only 改成 M256/N256 和 M128/N256 仍各超过 240 s。名为 largest-first 的 isolated probe 也没有在约 5 min 内退出，且实际 argv 未携带该策略参数，不能据此判断 largest-first。

下一位 AI 若补测，必须：

1. 使用同一 frozen source；
2. 保留原始配置的失败 artifact；
3. 任何 alternate block、alternate plan 或 alternate compiler 单独命名为 supplemental；
4. 只有通过 normal/empty/all-drop correctness 和 occupancy gate 后才写 speedup；
5. 不把 compiler elapsed、DCMI compile-phase 样本写成 forward latency/power。

## 9. 现阶段对根因的最可靠解释

已被多类证据共同支持的原因：

1. E896 的每个 source/expert bucket 很薄，固定 lane dispatch 造成空转；worker rotation 有约 0.86 ms 级改善。
2. E896 的有效路由行相对 M256 不整齐，M256 尾块填充约 31.25%，M128 将其降至约 9.62%。
3. E896 需要维护更多 expert metadata、更多 bucket、更多权重切换和更多跨卡同步；完整工作集约 7.399 GB/rank，E32 约 0.264 GB/rank。
4. full fused 测量中出现持续降频，E32/Torch 同轮没有同样程度的降频。
5. top-k16 比 top-k8 本身会扩大 route/GEMM 工作量，历史 1.6x 不能直接迁移。
6. L2 raw miss 和 Victim Rate 支持缓存行为差异，但不能单独给剩余差额定量归因。
7. full+MoonEP 还有显著 compile pressure；当前缺失是 compiler gate，不是已经测出一个慢的 runtime kernel。

仍不能证明的事情：

- 不能仅靠 DCMI 证明功耗限频造成了全部 0.5–2.5 ms；
- 不能仅靠 AIC MAC ratio 把时间分给 FC1/FC2/dispatch；
- 不能把 rtsGetHardwareSyncAddr、CANN mismatch 和当前 full+MoonEP compile timeout 视为同一根因；
- 不能把 direct-pull 的慢结果归因到单条 remote load，因为地址表、symmetric allocation、回传协议和 compiler plan 同时变化。

## 10. 复现入口

### 10.1 单点普通 forward

~~~bash
cd /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf
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
export TRITON_CACHE_DIR=/tmp/kimi_forward_matrix_next_cache

python benchmark/layer/profile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-skewed-w8-t4k \
  --benchmark-only --record-host-intervals \
  --fc1-block 128 512 128 --fc2-block 128 512 128 \
  --dispatch-block 256 --wave-windows 16 \
  --output-dir /tmp/kimi_single_point
~~~

MoonEP 只在 candidate 命令末尾加 --moonep，并把 wave 改为 32。先执行 npu-smi info，确认 8 张卡无外部 process；output-dir 必须新建或为空。

### 10.2 编译-only

~~~bash
python benchmark/layer/compile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-skewed-w8-t4k \
  --moonep \
  --fc1-block 128 512 128 \
  --fc2-block 128 512 128 \
  --dispatch-block 256 --wave-windows 32 \
  --output-dir /tmp/kimi_compile_probe
~~~

compile-only 不执行 device kernel，成功也不代表 runtime correctness 或 performance。

### 10.3 主矩阵运行器

~~~bash
python benchmark/layer/run_kimi_forward_matrix.py \
  --output-dir /tmp/kimi_forward_matrix_next \
  --routing-profile skewed --trimmed-topk 16 \
  --sample-interval 0.01 --case-timeout 900
~~~

运行器：

- 每个 point 前执行 npu-smi idle check；
- 记录 idle_checks.jsonl；
- 启动 profile；
- 通过 DCMI v2 采 frequency/power；
- 检查 child NPU occupancy；
- 用 frozen kernel SHA 防止运行中源码改变；
- 每个点写 job.json、running.json、status.json、benchmark.log、telemetry.jsonl；
- 失败时保留 artifact，不写假 speedup。

### 10.4 汇总器

~~~bash
python benchmark/layer/summarize_kimi_forward_matrix.py \
  /tmp/kimi_forward_matrix_20260925_1426 \
  docs/performance/kimi-k3-forward-matrix-2026-09-26.md \
  --diagnostic-status /tmp/kimi_forward_matrix_20260925_1426/fallback_largest_t4k_retry5/full_t4k_moonep1/status.json \
  --diagnostic-status /tmp/kimi_forward_matrix_20260926_completion/m256n256k128.status.json \
  --diagnostic-status /tmp/kimi_forward_matrix_20260926_completion/m128n256k128.status.json
~~~

汇总器会：

- 验证 12 个唯一组合；
- 验证每个成功点有 50 个 candidate/baseline 样本；
- 验证 8 个 rank 的区间文件和 telemetry；
- 按 rank 自己的 measured interval union 筛选功耗频率；
- 重算 median/mean/P95；
- 验证 correctness/occupancy gate；
- 验证 source SHA、shape、route owner counts；
- 把完整输入和失败原因写入 MD；
- 同步写出同名 JSON。

## 11. 原始产物索引

本轮主矩阵：

/tmp/kimi_forward_matrix_20260925_1426

该目录下每个 point 有：

- job.json；
- status.json；
- running.json；
- benchmark.log；
- telemetry.jsonl；
- benchmark/benchmark_result.json；
- benchmark/run_metadata.json；
- benchmark/host_call_intervals_rank0.json … rank7.json；
- benchmark/device_occupancy.jsonl；
- benchmark/occupancy_result.json。

本轮补测：

/tmp/kimi_forward_matrix_20260926_completion

包含：

- environment.sh；
- compile_configs.py；
- compile_driver.log；
- m256n256k128.status.json；
- m128n256k128.status.json；
- compile_m256n256k128/；
- compile_m128n256k128/；
- full_t4k_moonep1_largest/；
- profile_largest_wrapper.py；
- 对应 cache 子目录。

历史优化和性能数据：

/tmp/kimi_forward_perf

重点目录：

- pr68_exact_env；
- pr68_dispatch256_m128n512；
- full_final_profile；
- full_stable_l2_profile；
- trimmed_final_profile；
- trimmed_top16_control_retry；
- trimmed_top16_matched_tiles；
- dispatch_expert_rotation；
- dispatch_expert_rotation_repeat；
- dispatch_single_tile_nomask；
- skip_idle_fc1；
- full_m128n512_*；
- final_wave_windows_32；
- final_wave_windows_64；
- dcmi_full_cores28；
- dcmi_full_cores30；
- dcmi_full_cores32；
- reduce_interleaved_full；
- reduce_interleaved_paired_full；
- reduce_saved_fallback_full；
- fc1_options_*；
- fc2_options_k；
- fc2_completion_relay；
- fc2_completion_counter；
- direct_pull_reduce_*；
- udma_return_matched_toolchain；
- npuir_standard_full；
- compile_*。

## 12. 下一位 AI 的接手顺序

1. 先阅读本文、docs/kimi-forward-performance-analysis.md 和矩阵 JSON。
2. 先确认 NPU idle 和当前 compiler path，不要复用 stale Triton cache 来宣称成功。
3. 优先解决 full+MoonEP 原始 M128/N512/K128、wave32 的 compile/correctness gate；只做可解释的 compiler-pass/IR 对照。
4. 若尝试 alternate plan 或 block，建立 supplemental 目录和 supplemental 表，不覆盖主矩阵。
5. 成功后运行 full E896 + MoonEP 的 4k/8k/16k 三点；每点仍需 normal、empty、all-drop、occupancy、50 event、逐卡 DCMI。
6. 只有 12 点齐全后，才能回答“完整矩阵性能报告是否完成”。
7. 在 full+MoonEP 解决前，不要继续散射新的 kernel 微优化；否则会让现有 9 点和未来 3 点使用不同 frozen source。
8. 任何 1.6x 结论都要写清楚 top-k、route profile、capacity、tile、计时 boundary 和 Torch baseline；不要把历史 E32/top-k8 的 1.702x 作为 E896/top-k16 目标的直接证明。
