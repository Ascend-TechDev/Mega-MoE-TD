# Kimi K3 8 卡 fused forward 性能矩阵

> 生成时间：2026-09-25T16:45:07.991661+00:00。原始矩阵：`/tmp/kimi_forward_matrix_20260925_1426`。
> **已取得有效性能数据 9/12 点；尚缺 3 点。不能将编译超时视为性能结果或正确性失败。**

暂停新优化探索，固定当前已验证的 fused forward 源码。比较每卡 4096/8192/16384 tokens、裁剪 E=32 / 全量 E=896、MoonEP 开/关。两种专家规模都使用 top-k=16 和现有偏斜路由。

## 测量与输入口径

- **只测 forward**：直接执行 `op.forward(..., return_saved=False)`；输入无需梯度，不调用 backward。`torch.autograd.profiler.record_function` 只是 profiler 标记，此次 `--benchmark-only` 跳过 profiling 分支，不属于计时操作。
- **计时边界**：router 之后的完整 forward，包含路由元数据、dispatch、FC1、weighted SwiGLU、FC2、combine，以及调用中的工作区重置。权重/输入生成、算子初始化、JIT 首次编译和正确性检查均在计时外；不是纯 GEMM 或单个设备 kernel 的裸时间。
- 每点先验证 normal、zero-receive/empty-expert、negative/out-of-range all-drop，容差 rtol=atol=0.05；然后 5 次预热、50 次 NPU event 采样，每样本取 8 卡 MAX。没有额外的 changed-hidden 正确性门。
- median、mean、P95 全部从 50 个原始样本重算。P95 使用排序后位置 `0.95*(n-1)` 的线性插值。加速比 = `Torch grouped-GEMM + HCCL median / fused median`。
- **MoonEP 开关只作用于 fused 候选**；Torch baseline 始终是无 MoonEP 的 grouped-GEMM + HCCL。同一专家规模与 token 档位的开/关使用相同输入种子和路由；wave windows 随开关为 32/16，因此开关差异也包含运行配置变化。
- 偏斜 owner 配额为 `[19,27,11,11,15,15,15,15]/128`，所有全局专家均活跃。每个 token 的 16 个专家互不重复；各 owner 内轮询分配。裁剪/全量的 owner 负载相同，专家身份及权重形状随专家数改变。
- 权重种子 `42+rank`；输入种子 `43+rank*1000`。hidden states 在 routing logits 之前生成。BF16：hidden、W1、W2、output；FP32：routing weights；INT32：selected experts。activation=SwiGLU；capacity factor=1.6875；drop_frac=0；对称堆每卡 16 GiB。
- 每卡有 32 个 AICore programs / 64 个 AIVector programs；dispatch block M=256。MoonEP replica cache 关闭，每次 forward 刷新迁移权重。`save_fc1_dtype=bf16` 是配置字段，本次 `return_saved=False` 不保存训练中间值。
- **逐卡遥测**：DCMI v2 只读采样，目标间隔 10 ms，frequency type=7，功耗原始单位 0.1 W。分别按 rank0…7 各自的 50 个测量调用区间的并集筛选；排除 5 次预热与调用间空隙。区间结束点在 NPU event 完成后、rank-MAX 集合通信前。
- 遥测表显示 `中位数 [最小–最大], 有效样本数`。这是 host/event 区间内离散设备读数，不是每个 kernel 的精确能耗积分；驱动读数刷新率可能低于轮询频率。每次硬件运行前等待 8 卡空闲，并检查运行期间的外部进程占用。

## 固定版本与工具链

- worktree HEAD：`7a33afb`；kernel SHA256：`cbeae461ed961ee9c5785faf1ed80fb1ed5c4d74d348909977ddb4f2f4faf5fb`。这是包含 PR #68 及后续已验证优化的当前 worktree，不应标为裸 main 或裸 PR #68。
- Python：`/home/vllm_kimiw/.venv-udma/bin/python`（3.11.10）；Torch=2.10.0+cpu；torch_npu=2.10.0.post1.dev20260528；运行时 Triton=3.6.0。
- CANN：`/usr/local/Ascend/cann-9.2.0-beta.2`；编译器：`/home/vllm_kimiw/.venv-udma/lib/python3.11/site-packages/ascendnpuir/bin/bishengir-compile`。
- 编译器版本：`bishengir-compile 1.2.0 (https://gitcode.com/Ascend/AscendNPU-IR.git b229bc6ccbcc 2026-09-03) llvm 19.1.7 3254a1b1c59d Release build`；`TRITON_DISABLE_FFTS=1`。
- 原始 Triton cache：`/tmp/kimi_forward_matrix_20260925_1426_cache`。各点 `run_metadata.json` 保存完整环境、包版本、device library 与相关源码 SHA256；不以其他环境的旧结果替代。

## 全部 12 个计划数据点

时间列均为 median / mean / P95，单位 ms；`—` 表示没有有效计时。

| 数据点 | case | tokens/rank | E / top-k | MoonEP | FC1 M,N,K | FC2 M,N,K | wave | fused ms | Torch ms | 加速比 | 状态 |
|---|---|---:|---:|:---:|---|---|---:|---:|---:|---:|---|
| `trimmed_t4k_moonep0` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t4k` | 4096 | 32 / 16 | 关 | 256,256,128 | 256,256,128 | 16 | 22.476 / 22.489 / 22.958 | 32.492 / 32.523 / 32.635 | 1.446× | 正确性/占用通过 |
| `trimmed_t4k_moonep1` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t4k` | 4096 | 32 / 16 | 开 | 256,256,128 | 256,256,128 | 32 | 14.379 / 14.358 / 14.818 | 32.439 / 32.469 / 32.572 | 2.256× | 正确性/占用通过 |
| `full_t4k_moonep0` | `performance-fwd-kimi-k3-skewed-w8-t4k` | 4096 | 896 / 16 | 关 | 128,512,128 | 128,512,128 | 16 | 22.470 / 22.535 / 23.387 | 33.138 / 33.173 / 33.218 | 1.475× | 正确性/占用通过 |
| `full_t4k_moonep1` | `performance-fwd-kimi-k3-skewed-w8-t4k` | 4096 | 896 / 16 | 开 | 128,512,128 | 128,512,128 | 32 | — | — | — | 未取得计时 |
| `trimmed_t8k_moonep0` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t8k` | 8192 | 32 / 16 | 关 | 256,256,128 | 256,256,128 | 16 | 45.306 / 45.251 / 45.661 | 64.246 / 64.346 / 65.168 | 1.418× | 正确性/占用通过 |
| `trimmed_t8k_moonep1` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t8k` | 8192 | 32 / 16 | 开 | 256,256,128 | 256,256,128 | 32 | 29.340 / 29.410 / 30.240 | 64.054 / 64.101 / 64.179 | 2.183× | 正确性/占用通过 |
| `full_t8k_moonep0` | `performance-fwd-kimi-k3-skewed-w8-t8k` | 8192 | 896 / 16 | 关 | 128,512,128 | 128,512,128 | 16 | 43.670 / 43.684 / 44.232 | 65.473 / 65.515 / 65.557 | 1.499× | 正确性/占用通过 |
| `full_t8k_moonep1` | `performance-fwd-kimi-k3-skewed-w8-t8k` | 8192 | 896 / 16 | 开 | 128,512,128 | 128,512,128 | 32 | — | — | — | 未取得计时 |
| `trimmed_t16k_moonep0` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t16k` | 16384 | 32 / 16 | 关 | 256,256,128 | 256,256,128 | 16 | 95.512 / 95.564 / 96.300 | 126.903 / 126.980 / 127.113 | 1.329× | 正确性/占用通过 |
| `trimmed_t16k_moonep1` | `performance-fwd-kimi-k3-trimmed-skewed-w8-t16k` | 16384 | 32 / 16 | 开 | 256,256,128 | 256,256,128 | 32 | 60.215 / 60.200 / 60.728 | 127.107 / 127.189 / 127.543 | 2.111× | 正确性/占用通过 |
| `full_t16k_moonep0` | `performance-fwd-kimi-k3-skewed-w8-t16k` | 16384 | 896 / 16 | 关 | 128,512,128 | 128,512,128 | 16 | 86.088 / 85.941 / 86.581 | 129.420 / 129.489 / 129.757 | 1.503× | 正确性/占用通过 |
| `full_t16k_moonep1` | `performance-fwd-kimi-k3-skewed-w8-t16k` | 16384 | 896 / 16 | 开 | 128,512,128 | 128,512,128 | 32 | — | — | — | 未取得计时 |

## 当前可支持的比较

- 4k：裁剪无 MoonEP / 裁剪有 MoonEP / 全量无 MoonEP 对 Torch 的加速比分别为 1.446× / 2.256× / 1.475×；裁剪开启 MoonEP 后的 forward 中位数改善 1.563×。
- 8k：裁剪无 MoonEP / 裁剪有 MoonEP / 全量无 MoonEP 对 Torch 的加速比分别为 1.418× / 2.183× / 1.499×；裁剪开启 MoonEP 后的 forward 中位数改善 1.544×。
- 16k：裁剪无 MoonEP / 裁剪有 MoonEP / 全量无 MoonEP 对 Torch 的加速比分别为 1.329× / 2.111× / 1.503×；裁剪开启 MoonEP 后的 forward 中位数改善 1.586×。

在本次 top-k=16、偏斜路由的受控输入中，无 MoonEP 的全量 forward 中位数与裁剪相当或更低。历史 1.6× 比较需要同时核对 top-k、路由和计时边界；不能用本矩阵推断不同口径的历史结论。全量 + MoonEP 缺失，尚不能比较两种专家规模下 MoonEP 的完整收益。

## 每点参数、命令与逐卡遥测

### 1. `trimmed_t4k_moonep0`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t4k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=4096；global tokens=32768；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[4096,3584]`；indices/routing weights=`[4096,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=关；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[77824, 110592, 45056, 45056, 61440, 61440, 61440, 61440]；active global experts=32。
- 执行时间：2026-09-25T14:24:44.927027+00:00 → 2026-09-25T14:25:32.707931+00:00；进程 returncode=0；总用时=47.78096646000631 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t4k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=22.476 / 22.489 / 22.958 ms；Torch=32.492 / 32.523 / 32.635 ms；加速比=1.446×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t4k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t4k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 110 | 793.9 [715.9–813.1], 110 | 1650 [1650–1650], 164 | 722.2 [699.3–747.2], 164 |
| 1 | 1500 [1350–1650], 112 | 897.8 [862.5–901.0], 112 | 1650 [1650–1650], 163 | 818.8 [797.4–849.7], 163 |
| 2 | 1650 [1650–1650], 108 | 658.1 [638.8–677.7], 108 | 1650 [1650–1650], 163 | 623.3 [607.0–647.1], 163 |
| 3 | 1650 [1650–1650], 108 | 654.2 [636.3–672.6], 108 | 1650 [1650–1650], 163 | 617.4 [596.2–643.5], 163 |
| 4 | 1650 [1650–1650], 108 | 722.0 [692.6–740.6], 108 | 1650 [1650–1650], 163 | 660.5 [638.9–687.4], 163 |
| 5 | 1650 [1650–1650], 108 | 740.6 [715.9–761.6], 108 | 1650 [1650–1650], 163 | 674.2 [654.3–702.8], 163 |
| 6 | 1650 [1650–1650], 110 | 728.7 [698.0–745.9], 110 | 1650 [1650–1650], 163 | 669.6 [645.8–691.9], 163 |
| 7 | 1650 [1650–1650], 111 | 710.5 [679.0–728.0], 111 | 1650 [1650–1650], 162 | 654.4 [634.7–681.5], 162 |

### 2. `trimmed_t4k_moonep1`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t4k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=4096；global tokens=32768；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[4096,3584]`；indices/routing weights=`[4096,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=开；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[77824, 110592, 45056, 45056, 61440, 61440, 61440, 61440]；active global experts=32。
- 执行时间：2026-09-25T14:25:34.031049+00:00 → 2026-09-25T14:26:27.071460+00:00；进程 returncode=0；总用时=53.040433040005155 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t4k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=14.379 / 14.358 / 14.818 ms；Torch=32.439 / 32.469 / 32.572 ms；加速比=2.256×；50 样本；正确性与外部占用检查通过。
- MoonEP 迁移：copies/rank=[0, 0, 1, 1, 1, 1, 1, 1]；每次 forward 合计权重复制 396361728 bytes；transport=upstream PIPE_S UDMA；刷新=every_forward。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t4k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t4k_moonep1/benchmark --moonep
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1600 [1350–1650], 75 | 892.1 [771.9–900.2], 75 | 1650 [1650–1650], 160 | 713.8 [693.2–755.0], 160 |
| 1 | 1650 [1450–1650], 74 | 888.2 [797.9–898.1], 74 | 1650 [1650–1650], 160 | 804.2 [778.9–829.0], 160 |
| 2 | 1650 [1550–1650], 73 | 883.1 [769.9–896.0], 73 | 1650 [1650–1650], 160 | 615.7 [595.4–673.0], 160 |
| 3 | 1650 [1550–1650], 72 | 880.6 [762.3–896.6], 72 | 1650 [1650–1650], 161 | 608.5 [585.0–668.0], 161 |
| 4 | 1650 [1650–1650], 73 | 877.4 [754.9–892.8], 73 | 1650 [1650–1650], 161 | 657.0 [629.2–708.8], 161 |
| 5 | 1650 [1350–1650], 74 | 891.9 [790.8–900.2], 74 | 1650 [1650–1650], 162 | 663.3 [639.2–729.1], 162 |
| 6 | 1650 [1350–1650], 73 | 890.4 [784.4–900.6], 73 | 1650 [1650–1650], 162 | 659.2 [638.0–710.5], 162 |
| 7 | 1650 [1500–1650], 73 | 876.6 [765.7–895.0], 73 | 1650 [1650–1650], 162 | 648.6 [626.1–693.7], 162 |

### 3. `full_t4k_moonep0`

- case=`performance-fwd-kimi-k3-skewed-w8-t4k`；model=KIMI-K3；world=8；tokens/rank=4096；global tokens=32768；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[4096,3584]`；indices/routing weights=`[4096,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=关；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[77824, 110592, 45056, 45056, 61440, 61440, 61440, 61440]；active global experts=896。
- 执行时间：2026-09-25T14:26:28.386823+00:00 → 2026-09-25T14:27:16.809672+00:00；进程 returncode=0；总用时=48.42287446997943 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t4k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=22.470 / 22.535 / 23.387 ms；Torch=33.138 / 33.173 / 33.218 ms；加速比=1.475×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t4k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t4k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 112 | 829.0 [754.5–847.1], 112 | 1650 [1650–1650], 167 | 726.0 [692.6–770.7], 167 |
| 1 | 1500 [1400–1650], 111 | 899.3 [869.7–900.7], 111 | 1650 [1650–1650], 167 | 851.5 [822.1–870.8], 167 |
| 2 | 1650 [1650–1650], 111 | 702.5 [678.1–727.2], 111 | 1650 [1650–1650], 166 | 655.5 [632.1–681.3], 166 |
| 3 | 1650 [1650–1650], 111 | 694.0 [673.9–714.6], 111 | 1650 [1650–1650], 166 | 646.7 [623.9–674.4], 166 |
| 4 | 1650 [1650–1650], 111 | 760.9 [735.6–780.9], 111 | 1650 [1650–1650], 166 | 696.6 [669.8–728.4], 166 |
| 5 | 1650 [1650–1650], 112 | 781.3 [748.9–804.9], 112 | 1650 [1650–1650], 166 | 704.6 [664.3–735.3], 166 |
| 6 | 1650 [1650–1650], 113 | 775.2 [743.3–797.4], 113 | 1650 [1650–1650], 166 | 679.2 [647.6–724.7], 166 |
| 7 | 1650 [1650–1650], 114 | 759.7 [733.5–777.9], 114 | 1650 [1650–1650], 165 | 667.3 [635.7–703.0], 165 |

### 4. `full_t4k_moonep1`

- case=`performance-fwd-kimi-k3-skewed-w8-t4k`；model=KIMI-K3；world=8；tokens/rank=4096；global tokens=32768；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[4096,3584]`；indices/routing weights=`[4096,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=开；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[77824, 110592, 45056, 45056, 61440, 61440, 61440, 61440]；active global experts=896。
- 执行时间：2026-09-25T14:27:18.218943+00:00 → 2026-09-25T14:42:19.500332+00:00；进程 returncode=-15；总用时=901.2814241400047 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t4k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- **未取得有效 forward 延迟/加速比**：Explicit per-case execution budget of 900 seconds exceeded。原始三个全量 + MoonEP 点均停在首次正确性调用触发的后端编译，尚未完成正确性门和计时；此处不把编译期功耗冒充 forward 功耗。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t4k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t4k_moonep1/benchmark --moonep
```


### 5. `trimmed_t8k_moonep0`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t8k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=8192；global tokens=65536；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[8192,3584]`；indices/routing weights=`[8192,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=关；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[155648, 221184, 90112, 90112, 122880, 122880, 122880, 122880]；active global experts=32。
- 执行时间：2026-09-25T14:42:36.370950+00:00 → 2026-09-25T14:43:20.166694+00:00；进程 returncode=0；总用时=43.79575920000207 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t8k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=45.306 / 45.251 / 45.661 ms；Torch=64.246 / 64.346 / 65.168 ms；加速比=1.418×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t8k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t8k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 219 | 797.9 [759.1–826.0], 219 | 1650 [1650–1650], 321 | 717.9 [683.5–754.9], 321 |
| 1 | 1550 [1400–1650], 224 | 899.0 [887.9–901.2], 224 | 1650 [1650–1650], 317 | 814.6 [786.4–838.1], 317 |
| 2 | 1650 [1650–1650], 220 | 654.6 [622.7–687.7], 220 | 1650 [1650–1650], 317 | 617.8 [594.4–655.6], 317 |
| 3 | 1650 [1650–1650], 221 | 639.1 [607.3–673.9], 221 | 1650 [1650–1650], 316 | 611.3 [587.9–648.8], 316 |
| 4 | 1650 [1650–1650], 220 | 720.1 [683.7–754.9], 220 | 1650 [1650–1650], 316 | 659.1 [631.6–696.4], 316 |
| 5 | 1650 [1650–1650], 220 | 739.9 [704.3–771.9], 220 | 1650 [1650–1650], 316 | 668.9 [638.8–706.0], 316 |
| 6 | 1650 [1650–1650], 220 | 723.3 [690.8–756.4], 220 | 1650 [1650–1650], 316 | 663.0 [635.6–699.8], 316 |
| 7 | 1650 [1650–1650], 222 | 704.3 [675.3–735.9], 222 | 1650 [1650–1650], 316 | 648.5 [621.3–686.1], 316 |

### 6. `trimmed_t8k_moonep1`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t8k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=8192；global tokens=65536；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[8192,3584]`；indices/routing weights=`[8192,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=开；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[155648, 221184, 90112, 90112, 122880, 122880, 122880, 122880]；active global experts=32。
- 执行时间：2026-09-25T14:43:21.478888+00:00 → 2026-09-25T14:44:16.399959+00:00；进程 returncode=0；总用时=54.921103750006296 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t8k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=29.340 / 29.410 / 30.240 ms；Torch=64.054 / 64.101 / 64.179 ms；加速比=2.183×；50 样本；正确性与外部占用检查通过。
- MoonEP 迁移：copies/rank=[0, 0, 1, 1, 1, 1, 1, 1]；每次 forward 合计权重复制 396361728 bytes；transport=upstream PIPE_S UDMA；刷新=every_forward。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t8k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t8k_moonep1/benchmark --moonep
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1550 [1250–1650], 145 | 895.6 [875.6–902.5], 145 | 1650 [1650–1650], 318 | 725.5 [695.1–760.8], 318 |
| 1 | 1550 [1300–1650], 145 | 895.5 [882.6–902.0], 145 | 1650 [1650–1650], 317 | 824.0 [797.5–844.1], 317 |
| 2 | 1650 [1400–1650], 145 | 891.8 [874.4–900.4], 145 | 1650 [1650–1650], 317 | 624.1 [601.4–666.5], 317 |
| 3 | 1600 [1350–1650], 145 | 892.4 [875.7–900.2], 145 | 1650 [1650–1650], 317 | 619.7 [591.4–659.9], 317 |
| 4 | 1600 [1350–1650], 147 | 894.1 [877.8–900.9], 147 | 1650 [1650–1650], 318 | 666.0 [638.3–705.1], 318 |
| 5 | 1550 [1300–1650], 147 | 896.0 [880.7–900.8], 147 | 1650 [1650–1650], 318 | 675.4 [642.8–711.5], 318 |
| 6 | 1600 [1350–1650], 147 | 895.0 [877.1–901.3], 147 | 1650 [1650–1650], 318 | 671.3 [641.6–708.1], 318 |
| 7 | 1650 [1400–1650], 147 | 891.2 [866.9–900.4], 147 | 1650 [1650–1650], 319 | 655.0 [624.0–692.8], 319 |

### 7. `full_t8k_moonep0`

- case=`performance-fwd-kimi-k3-skewed-w8-t8k`；model=KIMI-K3；world=8；tokens/rank=8192；global tokens=65536；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[8192,3584]`；indices/routing weights=`[8192,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=关；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[155648, 221184, 90112, 90112, 122880, 122880, 122880, 122880]；active global experts=896。
- 执行时间：2026-09-25T14:44:17.726944+00:00 → 2026-09-25T14:45:10.722771+00:00；进程 returncode=0；总用时=52.99585368001135 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t8k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=43.670 / 43.684 / 44.232 ms；Torch=65.473 / 65.515 / 65.557 ms；加速比=1.499×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t8k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t8k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 218 | 828.0 [792.5–859.2], 218 | 1650 [1650–1650], 323 | 751.4 [714.4–790.8], 323 |
| 1 | 1500 [1400–1650], 217 | 899.4 [890.5–901.3], 217 | 1650 [1650–1650], 323 | 860.0 [823.1–881.9], 323 |
| 2 | 1650 [1650–1650], 217 | 693.6 [661.4–730.1], 217 | 1650 [1650–1650], 322 | 649.0 [616.1–685.2], 322 |
| 3 | 1650 [1650–1650], 217 | 684.6 [653.3–719.4], 217 | 1650 [1650–1650], 323 | 643.8 [616.8–678.3], 323 |
| 4 | 1650 [1650–1650], 216 | 748.4 [709.1–787.5], 216 | 1650 [1650–1650], 324 | 679.0 [646.1–725.7], 324 |
| 5 | 1650 [1650–1650], 216 | 763.0 [722.8–793.4], 216 | 1650 [1650–1650], 324 | 690.1 [657.9–730.1], 324 |
| 6 | 1650 [1650–1650], 217 | 756.6 [717.6–791.7], 217 | 1650 [1650–1650], 324 | 686.7 [655.8–726.7], 324 |
| 7 | 1650 [1650–1650], 217 | 738.0 [703.3–769.0], 217 | 1650 [1650–1650], 324 | 671.1 [642.2–711.7], 324 |

### 8. `full_t8k_moonep1`

- case=`performance-fwd-kimi-k3-skewed-w8-t8k`；model=KIMI-K3；world=8；tokens/rank=8192；global tokens=65536；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[8192,3584]`；indices/routing weights=`[8192,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=开；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[155648, 221184, 90112, 90112, 122880, 122880, 122880, 122880]；active global experts=896。
- 执行时间：2026-09-25T14:45:12.038908+00:00 → 2026-09-25T15:00:13.264414+00:00；进程 returncode=-15；总用时=901.225554870005 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t8k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- **未取得有效 forward 延迟/加速比**：Explicit per-case execution budget of 900 seconds exceeded。原始三个全量 + MoonEP 点均停在首次正确性调用触发的后端编译，尚未完成正确性门和计时；此处不把编译期功耗冒充 forward 功耗。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t8k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t8k_moonep1/benchmark --moonep
```


### 9. `trimmed_t16k_moonep0`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t16k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=16384；global tokens=131072；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[16384,3584]`；indices/routing weights=`[16384,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=关；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[311296, 442368, 180224, 180224, 245760, 245760, 245760, 245760]；active global experts=32。
- 执行时间：2026-09-25T15:00:30.170966+00:00 → 2026-09-25T15:01:20.075562+00:00；进程 returncode=0；总用时=49.90463253000053 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t16k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=95.512 / 95.564 / 96.300 ms；Torch=126.903 / 126.980 / 127.113 ms；加速比=1.329×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t16k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t16k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 471 | 781.3 [724.9–832.3], 471 | 1650 [1650–1650], 630 | 721.7 [670.7–783.0], 630 |
| 1 | 1650 [1400–1650], 472 | 897.2 [879.9–901.5], 472 | 1650 [1650–1650], 630 | 825.6 [782.1–854.7], 630 |
| 2 | 1650 [1650–1650], 472 | 651.1 [599.2–710.2], 472 | 1650 [1650–1650], 632 | 619.9 [575.3–689.0], 632 |
| 3 | 1650 [1650–1650], 473 | 647.5 [595.6–705.1], 473 | 1650 [1650–1650], 632 | 612.7 [568.1–678.4], 632 |
| 4 | 1650 [1650–1650], 473 | 716.8 [659.8–774.9], 473 | 1650 [1650–1650], 631 | 660.7 [612.5–729.8], 631 |
| 5 | 1650 [1650–1650], 473 | 736.1 [674.0–800.0], 473 | 1650 [1650–1650], 631 | 673.3 [627.9–745.1], 631 |
| 6 | 1650 [1650–1650], 473 | 726.3 [666.6–781.7], 473 | 1650 [1650–1650], 629 | 667.2 [622.5–732.3], 629 |
| 7 | 1650 [1650–1650], 477 | 706.4 [649.7–761.2], 477 | 1650 [1650–1650], 629 | 653.7 [610.9–718.8], 629 |

### 10. `trimmed_t16k_moonep1`

- case=`performance-fwd-kimi-k3-trimmed-skewed-w8-t16k`；model=KIMI-K3-TRIMMED；world=8；tokens/rank=16384；global tokens=131072；E=32；E/rank=4；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[16384,3584]`；indices/routing weights=`[16384,16]`；W1=`[4,3584,6144]`；W2=`[4,3584,3072]`。
- MoonEP=开；FC1 block=[256, 256, 128]；FC2 block=[256, 256, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[311296, 442368, 180224, 180224, 245760, 245760, 245760, 245760]；active global experts=32。
- 执行时间：2026-09-25T15:01:21.427040+00:00 → 2026-09-25T15:02:20.040743+00:00；进程 returncode=0；总用时=58.61372751998715 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/trimmed_t16k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=60.215 / 60.200 / 60.728 ms；Torch=127.107 / 127.189 / 127.543 ms；加速比=2.111×；50 样本；正确性与外部占用检查通过。
- MoonEP 迁移：copies/rank=[0, 0, 1, 1, 1, 1, 1, 1]；每次 forward 合计权重复制 396361728 bytes；transport=upstream PIPE_S UDMA；刷新=every_forward。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-trimmed-skewed-w8-t16k --benchmark-only --record-host-intervals --fc1-block 256 256 128 --fc2-block 256 256 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/trimmed_t16k_moonep1/benchmark --moonep
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1350–1650], 296 | 894.3 [877.3–901.7], 296 | 1650 [1650–1650], 631 | 729.8 [677.7–790.7], 631 |
| 1 | 1500 [1250–1650], 300 | 898.4 [879.8–901.5], 300 | 1650 [1650–1650], 631 | 831.7 [789.5–862.6], 631 |
| 2 | 1650 [1300–1650], 295 | 895.4 [874.3–902.1], 295 | 1650 [1650–1650], 631 | 629.4 [582.4–697.6], 631 |
| 3 | 1650 [1300–1650], 294 | 894.5 [870.9–902.4], 294 | 1650 [1650–1650], 631 | 621.9 [579.4–691.2], 631 |
| 4 | 1650 [1300–1650], 295 | 895.6 [874.0–901.8], 295 | 1650 [1650–1650], 632 | 671.5 [626.0–739.4], 632 |
| 5 | 1550 [1350–1650], 299 | 897.5 [881.5–901.6], 299 | 1650 [1650–1650], 632 | 681.3 [635.1–751.3], 632 |
| 6 | 1650 [1350–1650], 296 | 894.5 [873.6–901.3], 296 | 1650 [1650–1650], 631 | 673.5 [632.4–740.8], 631 |
| 7 | 1650 [1450–1650], 299 | 890.5 [871.3–900.5], 299 | 1650 [1650–1650], 631 | 661.6 [619.1–725.4], 631 |

### 11. `full_t16k_moonep0`

- case=`performance-fwd-kimi-k3-skewed-w8-t16k`；model=KIMI-K3；world=8；tokens/rank=16384；global tokens=131072；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[16384,3584]`；indices/routing weights=`[16384,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=关；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=16；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[311296, 442368, 180224, 180224, 245760, 245760, 245760, 245760]；active global experts=896。
- 执行时间：2026-09-25T15:02:21.387214+00:00 → 2026-09-25T15:03:20.705323+00:00；进程 returncode=0；总用时=59.31814101000782 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t16k_moonep0`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- 结果：fused median/mean/P95=86.088 / 85.941 / 86.581 ms；Torch=129.420 / 129.489 / 129.757 ms；加速比=1.503×；50 样本；正确性与外部占用检查通过。
- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t16k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 16 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t16k_moonep0/benchmark
```

| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |
|---:|---|---|---|---|
| 0 | 1650 [1650–1650], 426 | 832.4 [775.7–879.8], 426 | 1650 [1650–1650], 645 | 733.7 [676.6–799.4], 645 |
| 1 | 1500 [1400–1650], 427 | 899.7 [881.4–900.8], 427 | 1650 [1600–1650], 644 | 863.8 [809.7–896.1], 644 |
| 2 | 1650 [1650–1650], 427 | 683.8 [629.3–743.9], 427 | 1650 [1650–1650], 643 | 629.5 [586.3–705.7], 643 |
| 3 | 1650 [1650–1650], 427 | 676.3 [621.2–734.6], 427 | 1650 [1650–1650], 643 | 623.1 [578.5–694.4], 643 |
| 4 | 1650 [1650–1650], 427 | 749.9 [690.3–807.7], 427 | 1650 [1650–1650], 643 | 673.2 [623.7–744.2], 643 |
| 5 | 1650 [1650–1650], 427 | 762.5 [700.8–826.5], 427 | 1650 [1650–1650], 643 | 684.8 [634.5–758.3], 643 |
| 6 | 1650 [1650–1650], 428 | 753.2 [696.3–812.4], 428 | 1650 [1650–1650], 644 | 681.7 [630.7–753.3], 644 |
| 7 | 1650 [1650–1650], 428 | 736.5 [681.1–792.7], 428 | 1650 [1650–1650], 644 | 666.6 [619.2–736.7], 644 |

### 12. `full_t16k_moonep1`

- case=`performance-fwd-kimi-k3-skewed-w8-t16k`；model=KIMI-K3；world=8；tokens/rank=16384；global tokens=131072；E=896；E/rank=112；top-k=16；capacity=1.6875。
- 形状：hidden/output=`[16384,3584]`；indices/routing weights=`[16384,16]`；W1=`[112,3584,6144]`；W2=`[112,3584,3072]`。
- MoonEP=开；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave=32；其余公共配置见测量口径。
- 路由计划的原始 owner routes/rank=[311296, 442368, 180224, 180224, 245760, 245760, 245760, 245760]；active global experts=896。
- 执行时间：2026-09-25T15:03:22.010830+00:00 → 2026-09-25T15:18:23.289285+00:00；进程 returncode=-15；总用时=901.2784931700153 s（包含初始化/编译/验证，不是 forward 延迟）。
- 原始目录：`/tmp/kimi_forward_matrix_20260925_1426/full_t16k_moonep1`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。
- **未取得有效 forward 延迟/加速比**：Explicit per-case execution budget of 900 seconds exceeded。原始三个全量 + MoonEP 点均停在首次正确性调用触发的后端编译，尚未完成正确性门和计时；此处不把编译期功耗冒充 forward 功耗。

```bash
/home/vllm_kimiw/.venv-udma/bin/python /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf/benchmark/layer/profile_single_kernel_forward.py --case performance-fwd-kimi-k3-skewed-w8-t16k --benchmark-only --record-host-intervals --fc1-block 128 512 128 --fc2-block 128 512 128 --dispatch-block 256 --wave-windows 32 --output-dir /tmp/kimi_forward_matrix_20260925_1426/full_t16k_moonep1/benchmark --moonep
```


## 缺失点的补测记录

原配置全量 + MoonEP 三点均超过 900 s 运行预算，编译阶段未退出。以下尝试只用于补齐基准，没有改动 frozen kernel 源码，也没有可报告的额外性能数据。

- `/tmp/kimi_forward_matrix_20260925_1426/fallback_largest_t4k_retry5/full_t4k_moonep1/status.json`：returncode=1；duration=181.26928351001698 s；详见对应日志。
- `/tmp/kimi_forward_matrix_20260926_completion/m256n256k128.status.json`：returncode=-15；duration=240.21737284000847 s；Compile-only 240 second budget exceeded。
- `/tmp/kimi_forward_matrix_20260926_completion/m128n256k128.status.json`：returncode=-15；duration=240.21670372999506 s；Compile-only 240 second budget exceeded。

这些失败不能证明 CANN 版本不匹配，也不是 `rtsGetHardwareSyncAddr` 的运行时报错。缺失点仍需成功编译、通过三个正确性场景和设备占用检查后，才能补入有效性能表。所有中间产物保留，报告旁的同名 JSON 保存重算统计、逐卡遥测摘要、状态与审计信息。
