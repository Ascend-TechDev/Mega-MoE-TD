# E896 MoonEP fused-forward 编译退化：定位、修复、验证与交接

本报告对应实验批次 `/tmp/kimi_e896_rootcause_20260927`。
历史修复实验基线为 `b210dab`，工作分支为 `codex/e896-compile-fix`。
此前 forward MAC / paired-FC1 / dispatch 去重阶段已结束，见
`kimi-forward-mac-optimization-round3-summary-2026-09-27.md`；本次没有继续做其他性能优化。

**下一个 AI 请先读第 8 节交接说明，再查阅第 1–7 节历史实验细节。**
从 `7c93cb1`（含）开始的历史已按原顺序整理为三个代码提交和最后一个文档提交；
当前共同基线为 `7074180`，编译修复为 `00ecb6a`，提交映射见 8.1。
本次交接只补充文档，没有新增性能试验或重跑 NPU 验证。交接时重新核对发现：
旧 `/tmp` 路径下的关键原始日志、IR、perf、JUnit 和等待脚本已不存在；
仓库内的实验矩阵、硬件验收 JSON 与代码仍在，具体证据边界见 8.5。

## 1. 结论与当前验证边界

已定位并修复目标配置的**编译退化**。首个触发退化的 forward 提交是
`7c93cb1`，问题发生在 Ascend 原生编译器 `hivm-plan-memory-regbase` 阶段，
涉及 `MemPlanRegBase::SpecAlloc` / `GetOverlapBufferLife`，不是 Triton Python
前端卡死，也不是 MoonEP 的 `range(EPN)` 被隐式静态展开。

两个独立的触发改动需要同时处理：

1. FC1 activation Vector epilogue 根据运行时 `full_row_tile` 在无 mask / 有 mask
   两个 GM store 分支间选择。与 MoonEP home / replica 的 UB 生命周期组合后，
   触发内存地址规划的病态搜索。
2. top-k combine 的四路独立累加，经过 Vector function merge 后形成大缓冲区参数集合，
   同样能让该内存规划阶段长时间无法完成。

修复仅在 MoonEP fused 路径禁用上述两项代码形态：使用兼容完整 tile / 尾 tile 的
masked activation store，以及原有串行 top-k 累加。Cube 整行 load、routing-weight
load、idle-lane guard、dispatch worker rotation 和最后 wave acquire 均保留。
仍为一个 fused Triton forward kernel，未拆 FC1、未改 FP8、未做权重 L2 预排布，
未删 MoonEP planning、UDMA push、replica 或同步阶段。

截至最终实机验证版：

- E896+MoonEP 普通、保存原始 BF16 FC1、TIMING 三种变体均生成完整二进制。
- 新增的四项真实冷编译回归通过；现有 358 项 host 功能测试通过。
- **完整 E896 skewed MoonEP 八卡验证通过**：normal、zero-receive / empty-expert、
  negative / out-of-range all-drop，及 CPU planner 的 experts_to_copy / alloc_cumsum
  一致性均通过。算子有实际 replica，每卡复制数为 `[0,0,21,21,6,6,6,5]`。
- 正确性通过后完成 5 warmup / 50 measured / NPU event / 跨 rank MAX 计时：
  fused 中位数 **30.681696 ms**，Torch grouped-GEMM + HCCL 中位数
  **33.294735 ms**；该次运行比值为 **1.085166x**，延迟降低 **7.8482%**。
  这是修复后算子与 Torch 基线的测量，不是修复前后的性能 A/B，也未测量新 MAC ratio。
- 之前两轮 600 s 空闲门禁返回 75 的资源阻塞已经解除。用户恢复工作后，八卡
  均无外部进程，本轮正常启动并以 exit 0 结束；21 条 occupancy 记录全部通过，
  最终设备无残留进程。未修改或终止此前的外部 VLLM 进程。
- 这是对当前编译器的等价算子代码规避，未修改厂商编译器实现；不声称已解决
  `SpecAlloc` 对所有图的最坏情况复杂度。保存 FC1 变体仍有明显的编译耗时，见第 5 节。

## 2. 基线、配置与归因更正

### 2.1 固定复现配置

| 项 | 配置 |
|---|---|
| case | `performance-fwd-kimi-k3-skewed-w8-t4k` |
| experts / world / top-k | 896 / 8 / 16 |
| 每卡 tokens / H / FFN | 4096 / 3584 / 3072 |
| FC1 / FC2 tile | 128x512x128 / 128x512x128 |
| dispatch M / wave windows | 256 / 32 |
| capacity factor | 1.6875 |
| MoonEP / saved FC1 / TIMING | true / false / false，另有变体回归 |
| 原生 target | `Ascend950DT_9582` |
| 环境 | `/home/vllm_kimiw/udma-patch/activate-udma.sh` |
| bishengir | 1.2.0，revision `b229bc6ccbcc`，LLVM 19.1.7 |

每次 probe 使用新 cache、新输出目录；保存实际 constexpr、选项、原生编译器输入
MLIR、完整 argv 和状态。早期 probe 会覆盖 `DISPATCH_BLOCK_M`，因此 probe 的
`constants.json` 是实际参数依据，不能用未经覆盖的 CLI result 常量代替它。
冷编译总时间含 Python / 前端 / 原生编译；`compiler_status.json` 单独记录原生编译。
编译探针不执行 MoE 数据运算，但 Triton / NPU runtime 可能初始化设备上下文。

### 2.2 撤回此前不成立的 range 归因

本次核验早期曾把 `a7293ab` worktree 的成功编译归因于把 `range(EPN)` 改为
`tl.range(0, EPN)`。现明确撤回：

- `a7293ab` 使用较旧的 forward；历史失败来自 `b210dab` 性能分支，二者不是
  同一份 forward 代码。早期 dispatch 参数也不一致。
- 未改循环的 `a7293ab`，dispatch128，同样冷编译成功，实测 49.62 s。
- 在同一 b210 配置上，普通 range 与 tl.range 的编译器输入去除 debug locations 后
  完全相同，均为 372457 个字符；a729 两种循环形式也相同。
- 本地 Triton 的普通 range 和 tl.range 都生成循环 op，静态展开由 tl.static_range
  单独处理。没有证据支持“EPN 循环隐式展开是根因”。
- 已删除 backward-debug worktree 中这一未经证实的补丁；保留原补丁证据
  `previous_unverified_range.patch`。此前“约 33 s 因该改写而修复”的说法不能使用。

`a7293ab` 的原有提交属于前一 AI 工作成果；本次没有把它们计为新增修复。

## 3. 提交回归与最小消融

`07a5de8` 的 fused_forward 和 compile-only 入口与 `7c93cb1` 直接父提交完全相同
（git diff 验证）。相同 E896/MoonEP/tile/dispatch/wave 配置下：

| 代码 / 消融 | 冷编译结果 | 解释 |
|---|---:|---|
| `5ad5f02` | 50.12 s 成功，native 17.65 s | 较早快照 |
| `07a5de8` / 7c 的直接父代码 | 39.86 s 成功，native 19.08 s | 最后通过的代码 |
| `7c93cb1` | 120.22 s timeout | 首次退化 |
| `b210dab` dispatch256 | 240.22 s timeout | 原始目标 |
| `b210dab` dispatch128 | 240.22 s timeout | 不是 dispatch256 单独导致 |
| 仅回退四路 reduction | timeout | FC1 store 触发点仍在 |
| 仅关闭 FC1 Cube/Vector 动态整行路径 | timeout | 四路 reduction 触发点仍在 |
| 仅取消 idle-lane guard | timeout | guard 不是必要修复 |
| 仅取消 dispatch rotation | timeout | rotation 不是必要修复 |
| 等待所有 return waves | timeout | last-wave acquire 不是必要修复 |
| 静态整行路径 + 去 idle guard，保留四路 reduction | timeout | 仍未解决 reduction |
| 静态整行路径 + 串行 reduction，保留 idle guard | 39.41 s 成功 | 两项组合足够 |
| Cube 静态整行 + 串行 reduction | 150.22 s timeout | Vector 触发点仍在 |
| Vector 静态整行 + 串行 reduction | 38.71 s 成功 | Cube 快路径可保留 |
| 仅 routing-weight load 回退 + 串行 reduction | 150.22 s timeout | routing load 不是必要修复 |
| **仅 activation store 回退 + 串行 reduction** | **38.71 s 成功** | **最终最小组合** |

完整的 31 条状态记录（包括 capture-only、失败试验和 IR replay）在
`kimi-e896-compile-experiment-matrix-2026-09-27.json`。不能把其中每一条 success
都当成生成二进制成功，需同时看 arguments.capture_only 和 native 返回值。
所有有界 timeout 只证明在记录的观察窗内没有完成，不能外推为数学意义上的死循环。

其他失败诊断：

- 把 reduction block 从 4096 减到 2048，但保留 FC1 动态 store：150.22 s timeout。
- 直接把四路 reduction 的外层 tl.static_range 改为运行时 range，未重写分支变量
  作用域：前端报 `NameError('value0 is not defined')`，17.19 s 失败；未作为修复。
- 第一版计时命令因环境没有 `/usr/bin/time`，在编译开始前失败；后续使用 Python
  monotonic + 子进程状态计时，该次启动失败不计为编译故障。

## 4. 编译阶段与缓冲区证据

### 4.1 直接阶段证据

对原始 b210 的保存 MLIR 单独重放，打开标准 MLIR dump 与 pass ID：

```text
[PassID] hivm-plan-memory-regbase/module/2
```

该 pass 是 90 s 有界 replay 结束前最后进入的 pass。前端已经结束，原生编译器
尚未产出二进制，排除了“Python 模型构造还没进入编译”的解释。

对“已去掉 FC1 动态整行、仍保留四路 reduction”的卡住编译器做 12 s perf 采样：
1951 samples，lost samples=0，主要 self CPU 占比：

- `GetOverlapBufferLife`：51.31%。
- `SpecAlloc`：12.23%。
- malloc：6.05%，其中可见 BufferLife DenseMap 扩容调用。
- 其余包括 DenseMap lookup、MergeBufferVec、UpdateOutline。

调用链为 GetOverlapBufferLife → SpecAlloc → PlanMemAddressOfWholeLocalBuffer →
plan → PlanMemoryPass::runOnOperation。这里的采样对象是上述消融变体，不能伪称
对原始 b210 做了同一份采样。它与原始 b210 的 pass trace 共同定位问题阶段。

### 4.2 为什么不能只看 IR 大小或 alloc 数量

原始 b210 输入有 91 个 memref.alloc。只回退 activation store 并使用串行 reduction
的成功输入仍有 91 个 alloc，却能完成。因而“alloc 个数超限”不足以解释问题；
关键是控制流、缓冲区别名/生命周期与规划器搜索行为。

原始失败路径的合并 reduction Vector function，签名含 16 个
`memref<4096xbf16>` 和 4 个 `memref<4096xf32>`，对应参数缓冲区体积为 192 KiB。
成功的串行 reduction IR 分成较小的 VF，较大的签名为 4 个 BF16 + 2 个 FP32
4096 元素缓冲区，即 64 KiB。**这不是全 kernel UB 峰值测量**，而是可核对的
VF 参数集合差异，支持“四路累加经合并后加重同时活跃缓冲区规划”的机制判断。

FC1 store 的精确证据是源码消融：保留 Cube/load/其他分支，仅移除运行时 store
分叉，在串行 reduction 条件下即可恢复编译。未拿到厂商 C++ 实现的完整搜索轨迹，
因此不声称已证明某个内部循环的渐进复杂度，也不把超时伪装成已报告 UB overflow。

这与此前排除“1.8x 双模型对象”“CANN 算子规格”“坏 case / 坏 cache”的记录一致，
但本轮新增了精确的最后通过 / 首次失败提交、必要代码组合、pass 和 CPU 调用栈证据。

## 5. 最终代码与验证

生产变更仅在 `src/mega_moe/kernels/fused_forward.py`：

- FC1 helper 追加 constexpr `DYNAMIC_ACTIVATION_STORE=True`，保留已有位置参数语义。
- 两个 helper 调用显式传 `DYNAMIC_ACTIVATION_STORE=not MOONEP`。
- 完整静态 FULL_GROUP 仍可走完整 store；MoonEP 的运行时完整/尾 tile 统一走原有
  masked store。相同数据、地址、输出精度，未修改等待协议。
- reduction 改为 `INTERLEAVE_ACCUMULATORS=not (SAVE_FC1 or MOONEP)`。
  MoonEP 使用原有串行 FP32 累加，非 MoonEP 的原有选择不变。累加顺序改变后的
  BF16 数值容差必须由八卡 correctness 验证，不能仅靠 host 整数测试。

最终源码 SHA256：`bc24fac1c12c428cf79e8dcfa10af31d5a3d2a72d1befc4f4f9640b0d15dfb04`。
原始提交链中，`118f707` 归档 forward 性能总结，`537c652` 提交本节修复、编译回归
与证据报告。此前曾整理为 `0b8c1b2`（代码和测试）与 `b6d0c89`（文档和证据）；
最初原链保存在 `backup/e896-compile-fix-before-squash-8a7a976`。
随后按用户要求扩大整理范围到 `7c93cb1`（含），按功能保留原代码顺序，文档放最后；
当前编译修复为 `00ecb6a`，完整映射及此次备份见 8.1。各代码阶段与原对应版本一致，
文档仅更新历史映射和接手命令。交接时再次核对，当前 kernel 源码 hash 与硬件验收 JSON
保存的 hash 一致。历史记录中的原始提交号仍代表当时测试来源，不替换为新提交号。

| 最终补丁变体 | 冷编译总时间 | 原生编译时间 | PlanMemoryRegBase 各次 wall time |
|---|---:|---:|---|
| 普通 E896+MoonEP | 52.83 s | 19.61 s | 1.198 / 0.652 / 0.438 s |
| 保存 BF16 FC1 | 135.27 s | 101.27 s | 1.236 / 0.678 / **81.994 s** |
| TIMING 插桩 | 56.79 s | 23.17 s | 1.495 / 0.886 / 0.603 s |

保存 FC1 的 native 编译仍约 101 s，不能声称所有变体均已降至 20 s。
早期较宽的 Vector 回退补丁 `fixed_e896_full` 已被最终更小补丁替代，不作为最终
源码性能数据。最终数据来自 `fixed_v2_e896_*`，均保存源码快照和 hash。

新增 `tests/function/test_e896_moonep_compile.py`：真实编译 E896/top16/H3584/F3072
单 fused kernel，分别覆盖 skewed T4K、skewed T4K saved BF16、uniform T4K、
skewed T8K。使用独立 cache、实际二进制和常量检查、180 s / saved 300 s 防挂预算；
超时会清理该测试拥有的整个进程组。4 passed，239.17 s，JUnit 在
`/tmp/kimi_e896_rootcause_20260927/compile_regression.xml`。

现有 reduction、routing metadata、wave tasks 三组 host 测试先前为 358 passed，46.54 s；
最终收尾再次通过 358 项，45.09 s，并保存 `final_host_regression.xml` 和日志，
使验收记录可独立复核。

### 5.1 完整八卡实机验证

运行源码为干净提交 `6483aaa`，其中生产修复来自 `537c652`；fused_forward SHA256
与本节最终编译探针完全一致。完整命令见第 6 节，输出目录为
`/tmp/kimi_e896_rootcause_20260927/hardware_e896_skewed/`，终态 exit 0。
没有使用缩小专家数、关闭 MoonEP、跳过 replica、绕过同步或调宽数值容差等替代测试。

- 全部八卡 normal、空专家 / zero-receive、负数 / 越界 all-drop 输出与 Torch
  grouped baseline 比较通过，容差为入口原有 `rtol=0.05, atol=0.05`。
- MoonEP `experts_to_copy` 与 `alloc_cumsum` 均逐元素匹配 CPU oracle。复制数
  `[0,0,21,21,6,6,6,5]` 合计 65，每次 forward 刷新复制权重
  4,293,918,720 bytes，保留 upstream PIPE_S UDMA 路径。
- 占用监控共 21 条记录，覆盖 before_spawn、running、after_join，全部 passed，
  无 foreign process；首末时间为 `2026-09-27T01:32:50.197059+00:00` 和
  `2026-09-27T01:34:00.513544+00:00`。这段时间包括初始化、编译和正确性，
  不能作为单次 forward 延迟。

在上述 correctness gates 之后，使用既有计时入口采样，post-router full-forward
边界包含 routing metadata、dispatch、FC1、weighted SwiGLU、FC2、combine：

| 实现 | 样本数 | min ms | median ms | mean ms | P95 ms | max ms |
|---|---:|---:|---:|---:|---:|---:|
| 修复后 single fused kernel | 50 | 30.645697 | 30.681696 | 30.692342 | 30.804825 | 30.867830 |
| Torch grouped-GEMM + HCCL | 50 | 33.183441 | 33.294735 | 33.308049 | 33.443272 | 33.680782 |

P95 使用排序后 `(n-1)*0.95` 位置的线性插值；所有统计从保存的 50 个样本重新计算，
不是从三位小数的汇总倒推。中位数差为 2.613039 ms，比值为 1.085166x。
这是一次完整有效运行，不据此声称稳定跨运行收益；修复前相同目标编译无法在预算内
完成，因而没有有效的修复前后运行时 A/B。该结果也不能与此前 trimmed E32
paired-FC1 的 13 ms 数据混用，更不能声称达到 MAC 90% 或 1.8x。

随仓库保存的 `kimi-e896-moonep-hardware-validation-2026-09-27.json` 包含原始
100 个计时样本、统计、门禁、配置、源码 hash、21 条 occupancy 结论及原始文件 hash。
本次只补齐 E896 编译修复的实机验收，没有重新启动已停止的性能优化试验。
保存 FC1、TIMING、uniform T4K、skewed T8K 的已述结果仍限于编译验证；
本节八卡数值结论仅覆盖上述 E896 skewed T4K unsaved 配置。

## 6. 可复现命令

```bash
cd /home/vllm_kimiw/repo/MOE_Kimi-e896-compile-fix
source /home/vllm_kimiw/udma-patch/activate-udma.sh
MOE_RUN_COMPILE_TESTS=1 PYTHONPATH=src:. python -m pytest -q \
  tests/function/test_e896_moonep_compile.py
```

单次冷编译（输出目录必须新建，环境脚本会给出新 cache）：

```bash
python benchmark/layer/compile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-skewed-w8-t4k --moonep \
  --fc1-block 128 512 128 --fc2-block 128 512 128 \
  --dispatch-block 256 --wave-windows 32 --output-dir results/e896-new-compile-result
```

完整正确性和既有计时入口（执行前先确认八卡利用率均为 0，且无外部进程）：

```bash
npu-smi info
# 仅在确认全部八卡空闲后执行；不满足条件时停止，不启动测试。
MOE_FUSED_ASH_SIZE_GB=16 PYTHONPATH=src:. \
  python benchmark/layer/profile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-skewed-w8-t4k --moonep \
  --fc1-block 128 512 128 --fc2-block 128 512 128 \
  --dispatch-block 256 --wave-windows 32 --benchmark-only \
  --output-dir results/e896-new-hardware-result
```

`results/e896-new-hardware-result` 必须为空或尚不存在，复跑应使用新的目录名。
入口自带 before_spawn / running / after_join 占用检查及完整 correctness gates。
旧 `/tmp/moonep_wait_idle_exec.py` 在交接时已不存在，因此这里直接使用仓库入口；
没有因等待脚本缺失而省略全卡空闲要求。环境初始化仍须先执行本节开头的 UDMA 激活。

运行前必须同时检查利用率和进程表。前两轮门禁因外部 VLLM 占用而超时，
没有启动算子；最终本轮在八卡均无外部进程后启动，完整通过后退出。
当时分别记录了等待日志与成功实机日志，不能把等待超时归为算子失败；
交接时原始文件的可用性见 8.5，不能假定这些历史路径仍可读取。

## 7. 原始证据索引与最终状态

历史根目录：`/tmp/kimi_e896_rootcause_20260927/`。以下是当时的证据索引，
不是当前文件存在性清单；交接时已确认的缺失项见 8.5。

- `probe.py`：提交/参数/函数替换/超时隔离的编译探针；每次状态记录其完整参数。
- `experiment_matrix.json`：原始矩阵；同内容随本报告保存到 docs/performance。
- `a729_original_d128_v2/`、`b210_tlrange_capture/`：旧 range 归因更正。
- `commit_07a5de8/`、`commit_7c93cb1/`、`b210_original_d256/`：回归提交证据。
- `b7c_activation_store_static/`、`b7c_routing_load_static/`：最终 store 定位。
- `static_rows.perf.data`：CPU 栈采样；需先恢复原始文件，才能用 `perf report --stdio --no-children -i ...` 重放。
- `b210_standard_ir_trace/`、`good_standard_ir_trace/`：pass trace 和规划前 IR。
- `fixed_v2_e896_full/`、`fixed_v2_e896_saved/`、`fixed_v2_e896_timing/`：最终编译。
- `pytest_compile_regression/`、`compile_regression.xml`：新增四项回归。
- `final_host_regression.xml`、`final_host_regression.log`：最终 358 项 host 回归。
- `hardware_e896_skewed.log`：空闲门禁 / 八卡测试日志，需根据终态判断是否实际运行。
- `hardware_idle_gate_status.json`、`hardware_idle_gate_after_timeout.txt`：首轮门禁
  超时与残留外部进程证据。`operator_launched=false`，不是算子执行失败。
- `hardware_e896_skewed_retry2.log`、`hardware_blocked_audit.json`、
  `hardware_blocked_npu_snapshot.txt`：第二轮返回 75、等待进程已退出、无算子
  输出及连续三轮相同资源阻塞的历史核验，已被最终成功实机运行解除。
- `hardware_e896_skewed_resume.log`：恢复工作后的成功完整运行日志。
- `hardware_e896_skewed/benchmark_result.json`、`run_metadata.json`、
  `device_occupancy.jsonl`、`occupancy_result.json`：正确性、CPU oracle、计时样本、
  源码/环境、全过程占用监控；随仓库的 hardware-validation JSON 提供摘要和 hash。

forward 性能总结已归档；E896 退化提交、源码触发组合、原生编译阶段已定位，
修复与冷编译回归已提交，目标配置的完整八卡数值与占用门禁已通过。
这表示当时约定的 E896 编译修复验收完成，不表示 MAC 90%、1.8x、全部配置验证
或主分支集成已经完成。遗留工作、实验代码及证据可用性在第 8 节单独交接。
后续如向厂商提报，应先恢复或重新生成第 4 节的 IR / pass / perf 原始证据；
保存 FC1 的较长编译耗时和其他未做实机测试的形状仍应按本报告边界看待，
不属于本轮已经验证的跨配置性能结论。

## 8. 给下一个 AI 的完整交接

### 8.1 任务来源、完成范围与当前提交

本会话先接手 forward MAC / MTE2 判断核验，执行 paired-FC1 与 dispatch 去重试验；
按用户停止指令归档性能总结后，再执行用户另行指定的 E896 MoonEP 编译问题排查。
前一 Claude 会话 `641079e4` 的四个 commit、两份文档、MR 及 `a7293ab` 的原有成果，
均不是本会话新增工作。

当前应从以下位置接手，不能把主 worktree 或 paired 实验分支当作编译修复分支：

```text
工作区：/home/vllm_kimiw/repo/MOE_Kimi-e896-compile-fix
分支：codex/e896-compile-fix
当前整理基线：7074180e1c06eb3c4f72ac540db3c30d9aac9bcf（7c93cb1 的父提交）
历史编译实验基线：b210dabb558722c36407dacb5d8d0e56721ef9e8
代码 1：17b2e6776b77bc4206d9cd11b5a36028f64435dc
代码 2：592820acaec25aa27100918341febfd4909caf45
代码 3：00ecb6a5f8a2cc4ff9a7045d690ab9c1804f2e53
最后一个提交：HEAD，docs: collect forward results and handoff
此次整理前备份：backup/forward-before-squash-20260927-b6d0c89
原始五提交备份：backup/e896-compile-fix-before-squash-8a7a976
```

本次范围内的 10 个提交整理为 4 个，代码先后顺序不变：

| 顺序 | 当前提交 | 原始来源 | 内容 |
|---|---|---|---|
| 1 | `17b2e67` | `7c93cb1`、`801f496` 的非文档改动 | 稀疏路由、工作区和归约清理，以及配套测试、基准工具 |
| 2 | `592820a` | `3c1b507` | 默认 wave windows 调整为 32 |
| 3 | `00ecb6a` | `0b8c1b2`，其原始修复为 `537c652` | E896 编译规避和冷编译回归 |
| 4 | `HEAD` | 整理范围内所有 `docs/` 改动 | 报告、交接、JSON、图片，集中放在最后 |

只合并相邻的同功能代码阶段，没有改变代码内容。三个代码提交分别与原
`801f496`、`3c1b507`、`0b8c1b2` 的非文档文件完全一致。
文档提交不含源码、测试或工具改动，后续可按用户要求单独舍弃该提交；
基线原有文档不属于本次新增文档范围。文档 HEAD 的准确 SHA 用 `git rev-parse HEAD` 查看。
验收 JSON 中的 `run_commit=6483aaa`、`production_fix_commit=537c652` 是历史真实来源，
必须保留；不能为让 SHA 看起来最新而改写实验记录。

### 8.2 已修改的代码，以及它解决的具体问题

编译修复提交 `00ecb6a` 仅涉及两份文件；前两项性能代码提交另见 8.1：

| 文件 | 改动和定位 | 验证边界 |
|---|---|---|
| `src/mega_moe/kernels/fused_forward.py` | `_partition_pipeline_fc1_activation_group_ub` 追加 `DYNAMIC_ACTIVATION_STORE`；`_run_dynamic_wave_pipeline` 两处调用传入 `not MOONEP`，MoonEP 的动态完整/尾 tile 统一 masked activation store | 避免与 home/replica UB 生命周期组合的运行时 store 分叉，保留静态 FULL_GROUP 快路径、Cube load 和 routing-weight load |
| 同一 kernel 文件 | reduction 调用使用 `INTERLEAVE_ACCUMULATORS=not (SAVE_FC1 or MOONEP)` | MoonEP 使用原有串行 FP32 累加，避免四累加器合并后的复杂缓冲区规划；累加顺序改变已通过目标八卡容差验证 |
| `tests/function/test_e896_moonep_compile.py` | `test_e896_moonep_cold_compile_finishes`，4 个真实冷编译参数组，独立 cache、二进制/constexpr 校验、180 s / saved 300 s 超时清理 | 编译回归，不代替八卡数值或 backward 验证 |

已解决的是指定编译器和 E896 配置下的原生内存规划退化：原 b210 目标在 240 s
预算内无法完成，最终普通变体冷编译 52.83 s、native 19.61 s，并能实际运行。
这是算子源码规避，未修改厂商 `SpecAlloc` 算法，不是对所有图的复杂度保证。
没有拆分 forward kernel、使用 FP8、做额外权重 L2 预排布，也没有关闭 MoonEP
planning、UDMA、replica 或同步来换取编译成功。

### 8.3 性能阶段做了什么，实验代码在哪里

paired-FC1 在初始化时将权重按 panel 交错，kernel 连续宽 load 加单 dot，
不使用 tl.join，保留 ROW_SPLIT Fixpipe。新增原始 gate/up probe、correctness-only
入口和同进程 AB/BA 计时。六组 raw 输出均 bad=0、max_abs=0、nonfinite=0，
E32 trimmed 八卡 correctness 通过。两轮 AB/BA 中位延迟降低 0.6728% / 0.7772%，
profiler MAC ratio 中位数从 84.05% 到 86.00%，最高样本 87.1%。
MTE2 时间中位数仅下降约 0.92%，因此不能把低 MAC 的唯一根因归给 MTE2。
三条硬件/编排约束没有被证明构成绝对上限，90% 与 1.8x 均未达成。

dispatch 去重实现了 canonical route、activation 单份发送、alias metadata 和
接收端展开；首次 NPU 编译报 UB overflow，需求 14,858,496 bits，可用 1,769,472 bits。
因此没有有效的去重 correctness / MAC / 端到端 A/B，不能写成“测过且没有收益”。
用户要求该尝试之后停止其他性能优化，本会话已按此停止。

工作区路径均以 `/home/vllm_kimiw/repo/` 为前缀；以下是交接时重新核对的状态：

| 工作区目录 | HEAD / 状态 | 下个 AI 应如何处理 |
|---|---|---|
| `MOE_Kimi-e896-compile-fix` | 三个代码提交，最后一个文档提交 | 正式编译修复和交接文档所在位置 |
| `MOE_Kimi-kimi-forward-perf` | `b210dab`；3 份 tracked 文件修改，214 insertions / 72 deletions；另有未跟踪 probe | paired 仅为未提交实验，不能当作已合入修复或直接覆盖 |
| `MOE_Kimi-dispatch-dedup-20260926` | `b210dab`；4 份 tracked 文件修改，140 insertions / 9 deletions | 编译失败的原型，不能视为可上线代码 |
| `MOE_Kimi-mac-baseline-20260926` | `b210dab`，干净 | 原性能对照基线 |
| `MOE_Kimi-fc1-backward-debug` | `a7293ab`，干净 | 旧 AI 的工作；本会话错误 range 补丁已撤销 |
| `MOE_Kimi` | `eb737e7`；tracked 文件未改；未跟踪 `.codegraph/` 与根目录旧性能总结 | 主分支尚未集成本修复；根目录总结是旧副本，优先读本目录版本 |

paired 具体文件：`src/mega_moe/ops/forward.py`、
`src/mega_moe/kernels/fused_forward.py`、`benchmark/layer/profile_single_kernel_forward.py`、
新增 `benchmark/layer/probe_paired_fc1.py`。
dispatch 具体文件：`src/mega_moe/ops/forward.py`、
`src/mega_moe/kernels/fused_forward.py`、`src/mega_moe/kernels/dispatch_fc1.py`、
`benchmark/layer/profile_single_kernel_forward.py`。
这些 dirty worktree 是实验现场；整理提交没有将它们混进当前代码提交。

### 8.4 已完成验证与尚未覆盖的配置

| 项目 | 已有结果 | 不得扩大解释为 |
|---|---|---|
| E896 skewed T4K，unsaved，8 卡 MoonEP | 完整二进制、normal / empty-expert / all-drop 数值、CPU oracle、21 条 occupancy 均通过；65 个真实 replica | 所有形状、所有路由、所有环境都通过 |
| saved BF16 FC1 | 冷编译成功；135.27 s 总计，101.27 s native | saved raw FC1 数值及 backward 已做完整八卡验证 |
| TIMING 变体 | 冷编译成功；56.79 s 总计，23.17 s native | 本次已取得新的 MAC profiler / 插桩运行结果 |
| uniform T4K、skewed T8K | 真实冷编译回归通过 | 已跑相应形状的完整八卡数值与性能 |
| 4 项编译回归、358 项 host 测试 | 历史执行通过，结果摘要保存在验收 JSON | 本次文档交接又重跑了一遍测试 |
| 目标 E896 性能 | 5 warmup、50 samples/实现；30.681696 ms vs Torch 33.294735 ms，1.085166x | 修复前后 A/B、多轮稳定收益、达到 1.8x，或可与 E32 的 13 ms 混比 |

历史测试曾受外部 VLLM 占用阻塞，两轮空闲门禁各等 600 s 后退出 75，未启动算子；
之后资源释放并通过验收。这个阻塞已在历史运行中解除，不应照搬成下次接手的
当前阻塞，也不能未经查询就认定现在八卡空闲。

### 8.5 证据当前可用性：必须保留的缺口

本次交接重新核对了文件是否实际存在，结果如下：

- 仓库内 `kimi-e896-compile-experiment-matrix-2026-09-27.json` 可读，包含 31 条
  探针、失败、capture-only、replay 等记录；不能当作 31 次完整编译成功。
- 仓库内 `kimi-e896-moonep-hardware-validation-2026-09-27.json` 可读，包含
  两组各 50 个实际计时样本、21 条 occupancy 摘要、4/358 项回归结果摘要、
  环境与源码 hash、原始文件 hash。当前 kernel 的 SHA256 与其中记录一致。
- 旧 `/tmp/kimi_mac_resume_20260926/` 下已检查的 raw_result、两轮 AB/BA、
  paired_evidence_summary、dedup_correctness.log 均不存在。paired 的细节当前只能
  从已提交性能报告和保留的实验代码查阅，无法从这些旧路径再次重算置信区间。
- 旧 `/tmp/kimi_e896_rootcause_20260927/` 下已检查的 probe.py、JUnit XML、
  原始 benchmark_result、static_rows.perf.data、good/bad IR trace 目录均不存在。
- `/tmp/moonep_wait_idle_exec.py` 也不存在；UDMA 激活脚本仍存在。

以上不否定当时已完成并记录的测试，但当前无法直接重放缺失的 perf / IR 或重新
对原始文件逐个验 hash。hash 不能还原丢失文件，验收 JSON 也不是原始 IR 的备份。
若需要独立审计或向编译器团队提交最小复现，应先查找外部归档；若无归档，
在明确需要复现的范围后，用原始备份提交和当前代码重新生成必要证据，保存到
持久化目录。不要把报告中的历史路径存在性当作已重新确认，也不要编造缺失日志。

### 8.6 遗留工作及建议顺序

| 状态 / 优先级 | 未完成项 | 建议下一步与验收条件 |
|---|---|---|
| 交接优先 | 部分原始证据仅在旧临时目录保存，当前缺失 | 查外部归档；必要复现前先固定提交、配置和环境，并持久化原始日志、IR、perf 与完整 JSON |
| 待集成 | 编译修复尚未合入当前 main | 在用户确定的目标分支按顺序审核前三个代码提交；单独选取 `00ecb6a` 前确认目标已有前置改动。本次未创建新 MR 或执行合入 |
| 验证缺口 | saved FC1、TIMING、uniform T4K、skewed T8K 缺完整实机覆盖 | 若目标使用这些配置，按各自真实配置补数值/同步/占用验证；saved/backward 需单独验证，不复用 unsaved 结论 |
| 已知残余 | saved FC1 native 编译仍约 101 s，其中一次 PlanMemory 约 82 s | 若继续处理编译问题，可围绕 saved 变体复现和向厂商提报；当前仅有源码规避，厂商算法未修改 |
| 验证边界 | 非 MoonEP 的 constexpr 选择保持原逻辑，但未新增全面实机回归 | 集成到更广使用范围时按影响面补回归，不能把 host 测试当全设备验证 |
| 未达成且已停止 | MAC 90%、端到端 1.8x | 当前不自动继续优化；需用户重新明确授权后才启动新的优化方向 |
| 待决策 | paired 只有约 0.67%–0.78% 收益，代码未提交 | 保留实验现场，待用户决定是否值得产品化；不能自动合入当前编译修复 |
| 未完成且已停止 | dispatch 去重没有有效 correctness / A/B | 若以后获授权再做，先解决 UB footprint 并通过 correctness，再计时；现在不能声称已证实无收益 |

上表是明确的遗留清单，不是授权下个 AI 自动执行所有项目。本次用户要求是把
交接写入文档；没有借此启动新的优化、NPU 试验或主分支合并。

### 8.7 接手时的检查顺序与不能重复的错误

先执行只读检查，确认分支和实验改动仍然存在：

```bash
cd /home/vllm_kimiw/repo/MOE_Kimi-e896-compile-fix
git status --short
git log --reverse --oneline 7074180..HEAD
git show --stat 00ecb6a
git diff 592820a 00ecb6a -- src/mega_moe/kernels/fused_forward.py
git status --short --untracked-files=normal
```

下一步按实际任务选读：性能结论看同目录性能总结；编译根因看本报告第 2–5 节
和实验矩阵；数值与计时看硬件验收 JSON。若是集成任务，优先处理代码提交；
若是继续编译器排查，先处理原始 IR/perf 缺失；若是性能任务，先确认用户已经
解除停止指令。使用含 `.codegraph/` 的主仓库定位代码时，按 AGENTS.md 先用
CodeGraph；本修复 worktree 交接时没有 `.codegraph/`，无需自行建立索引。

需要重跑时沿用 UDMA 环境与第 6 节入口，先查八卡空闲，再运行 correctness，
之后才接受计时；每次保存唯一 cache、输出目录、源码 hash 和真实参数。
只查询或等待设备不等于启动了算子；观察超时不等于进程结束，不要盲目重复启动。

必须保留的纠错和约束：

- `range` 改 `tl.range` 的根因归因已撤回；不要重新应用旧错误补丁。
- 不把不同 checkout、dispatch 参数、E32/E896、profiler 与 AB/BA 的结果混比。
- 192 KiB / 64 KiB 是合并 VF 参数缓冲区体积，不是全 kernel UB 峰值。
- `GetOverlapBufferLife` / `SpecAlloc` 的 perf 采样来自特定消融变体，不冒充原始 b210。
- 有界编译 timeout 不能证明死循环，源码 workaround 不等于厂商算法彻底修复。
- 保持单 fused Triton forward；FP8、独立 FC1 kernel、权重 L2 预排布需用户先确认。
- 用户允许常规命令直接执行；仅不可逆危险操作需要确认。不要删除 dirty 实验现场。
- 不把前一 AI 的 commit / MR 归为本次成果，不把历史资源占用当作当前设备状态。
