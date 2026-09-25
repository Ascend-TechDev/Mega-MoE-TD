# E896 MoonEP 三个失败用例补测与根因（2026-09-26）

## 结论

`full_t4k_moonep1`、`full_t8k_moonep1` 和 `full_t16k_moonep1` 都在第一次 normal correctness forward 触发 Triton 到 BiShengIR 的编译阶段卡住，未进入 kernel correctness、性能计时或功耗采样。三个用例的共同根因是当前 E896 + MoonEP fused forward 生成的混合 Cube/Vector IR 让 Ascend950 RegBase 的 `hivm-plan-memory-regbase` 规划过程非常复杂：默认策略长时间搜索，受控的 `largest-first` 策略较快进入规划失败后的向量化失败。现有证据不支持把它描述成运行期显存不足、NPU 执行错误或 `rtsGetHardwareSyncAddr` 问题。

这说明当前 kernel 的资源形状和编译器规划路径需要进一步处理；不能据此断言硬件损坏，也不能把它当作已经解决的性能优化。生产源码没有为这些诊断试验改动。

## 原始三点

三点来自原始 12 点矩阵，使用同一 frozen fused source、CANN 环境和输入协议：

| point | tokens/rank | E | top-k | FC1/FC2 block | dispatch M | MoonEP | wave windows | 结果 |
|---|---:|---:|---:|---|---:|---|---:|---|
| `full_t4k_moonep1` | 4096 | 896 | 16 | `[128,512,128]` | 256 | 开 | 32 | 901.28 s 后被用例预算 SIGTERM，编译未完成 |
| `full_t8k_moonep1` | 8192 | 896 | 16 | `[128,512,128]` | 256 | 开 | 32 | 901.23 s 后被用例预算 SIGTERM，编译未完成 |
| `full_t16k_moonep1` | 16384 | 896 | 16 | `[128,512,128]` | 256 | 开 | 32 | 901.28 s 后被用例预算 SIGTERM，编译未完成 |

共同输入为 KIMI-K3、8 卡、hidden=3584、ffn=3072、BF16 activation/weights、FP32 routing weights、SwiGLU、capacity factor=1.6875、drop fraction=0。MoonEP 关闭的 E896 full 点可以正常编译和运行；E32 trimmed 的 MoonEP 点也可以正常编译和运行。这两个对照说明失败由 E896 MoonEP 组合触发，而不是 MoonEP API 在所有场景都不可用。

原始 artifacts 保留在 `/tmp/kimi_forward_matrix_20260925_1426/full_t{4,8,16}k_moonep1/`。每个 `status.json` 的 `returncode=-15` 和 `stop_reason=Explicit per-case execution budget of 900 seconds exceeded`；每个 benchmark log 的 Python stack 都停在 `triton.backends.ascend.compiler.py` 的 `subprocess.run` 编译调用。

## 编译环境和复现方式

原始矩阵和诊断均使用：

- Python 3.11.10，`torch 2.10.0+cpu`，`torch_npu 2.10.0.post1.dev20260528`，Triton runtime 3.6.0。
- `ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2`。
- NPU compiler 为环境中的 `ascendnpuir/bin/bishengir-compile`，目标 `Ascend950DT_9582`。
- `TRITON_BACKENDS_IN_TREE=1`、`TRITON_DISABLE_FFTS=1`、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`MOE_FUSED_ASH_SIZE_GB=16`、`MOE_FWD_TIMING=0`；没有设置 `ASCEND_LAUNCH_BLOCKING` 或旧的 `MOE_ASH_ENGINE`。
- 原始 benchmark 是 8 个 multiprocessing rank；每个 rank 首次进入 forward 时都可能等待同一份 kernel 的编译/缓存。

原始 benchmark 的 900 秒是整个 case 的显式预算，不是单独的 compiler timeout。三点都在 correctness 入口结束，因而没有有效 latency、speedup、frequency 或 power 数据，不能把失败点填写成性能为 0。

## 受控编译诊断

为避免再次启动 8 卡 benchmark，使用第一次 E896 MoonEP t4k 生成的 Triton IR dump 做单进程 compiler-only 试验。诊断只改变 compiler option，不改仓库源码。dump 目录为 `/root/.triton/dump/UCALNWYHA3XTCOKLQRSXHSOI7AJXWBE775XVTZSLPWP6SXHCXNTQ/`，输入 `kernel.mlir` 约 9549 行、714 KB，包含 47 个函数、99 个 local allocation 和 MoonEP UDMA 操作。

| 诊断 | 结果 | 证据 |
|---|---|---|
| 默认 memory strategy + tuning mode | 240 s 超时 | 进程持续占用约 100% CPU；没有输出 binary |
| `largest-first` + tuning mode | 37.01 s，compiler exit 1 | `PlanMemoryRegBase Failed`，随后 `Attempted to vectorize, but failed` at `kernel.mlirbc:2274:21` |
| `largest-first` + `enable-preload=false` | 36.71 s，exit 1 | 相同 PlanMemoryRegBase/向量化失败 |
| `largest-first` + `enable-vf-merge-level=0` | 39.36 s，exit 1 | 相同失败阶段 |
| `enable-auto-multi-buffer=false` | 180 s 超时 | 没有绕过规划耗时 |
| `largest-first`，wave 32→16 | 140 s 超时 | 仅减小 wave window 仍未完成编译 |
| `largest-first`，N tile 512→256 | 140 s 超时 | 仅减小 GEMM N tile 仍未完成编译 |

对默认策略进行 `perf record` 采样的结果是：约 55.62% CPU cycles 在 `mlir::hivm::MemPlanRegBase::GetOverlapBufferLife`，约 12.44% 在 `MemPlanRegBase::SpecAlloc`。在一次 `largest-first` 试验的 MLIR timing 中，`PlanMemoryRegBase` 占 27.759 s，总编译时间 56.204 s。这个调用栈解释了“看起来像卡死”的现象：进程在做编译器的缓冲区生命周期重叠分析和试分配，而不是在卡上执行模型。

编译器 stdout 还给出两类资源复用警告：

```text
cc: not reusing DMA buffers needs 8388608 bits while 2097152 bits available
cbuf: not reusing DMA buffers needs 10485760 bits while 4194304 bits available
```

它们只表示“不复用时”需要的空间超过可用空间，提示必须进行 buffer reuse，并不等价于最终的 UB overflow。此次失败日志没有出现明确的 `UB overflow` 或设备侧 allocation error；失败点是 RegBase planner 后的 vectorization error。

## 根因判断

### 已经确认的部分

1. 失败发生在 host-side `bishengir-compile`，不是 kernel 已加载后的 NPU 运行期。
2. 失败集中在 E896 + MoonEP；E896 MoonEP 关闭和 E32 MoonEP 开启均有成功对照。
3. 当前 IR 同时包含 MoonEP 的 UDMA/同步路径、动态 dispatch/combine 路径、混合 Cube/Vector GEMM 以及多缓冲临时对象。E896 的物理专家数和 pipeline 工作集显著大于 E32，因此 buffer lifetime 和候选复用关系更多。
4. 编译时间主要消耗在 `GetOverlapBufferLife` 和 `SpecAlloc`，与“编译器内存规划搜索空间爆炸”一致。

### 尚不能声称的部分

- 不能声称这是机器 DRAM 或 NPU 显存耗尽；compiler RSS 约 180–230 MiB，设备没有执行到申请工作区的阶段。
- 不能声称某一个特定 buffer 已被证明超过 UB 容量；当前 warning 是复用前估算，失败信息没有给出明确 overflow buffer。
- 不能声称关闭多缓冲、换 wave 或缩小 N tile 已经解决；三个受控变体在 140–180 秒内仍未完成。
- 不能把这次结果归因于 `rtsGetHardwareSyncAddr`、UDMA 运行时或 CANN 版本不匹配；相同环境下 MoonEP trimmed 和 MoonEP-off 路径可编译运行。

因此当前最准确的结论是：**代码生成的 E896 MoonEP kernel 资源/生命周期形状让当前 AscendNPU-IR RegBase memory planner 进入极慢或失败的路径；这属于 kernel 配置与编译器规划能力的组合限制，尚未证明是不可绕过的硬件容量上限。**

## 遗留事项

- 三个原始 point 仍没有性能数据，主矩阵保持 9/12 有效；不要用失败点计算平均加速比。
- 若继续处理，应先做 compiler-only 的缩小 IR 或分 kernel 设计，并记录新 source hash 与独立 artifact；不要把修改 compiler option 的结果伪装成原始矩阵数据。
- 可能的工程方向包括减少同一 kernel 中同时存活的 MoonEP planner/dispatch 临时对象、把超大 E896 MoonEP 路径拆分成阶段、或使用经过验证的预编译 cache。每个方向都需要重新过 correctness 和性能门。

