# Kimi K3 fused forward 性能优化简明总结

生成时间：2026-09-26（Asia/Shanghai）。本文件给人类阅读；下一位 AI 的完整交接记录见：

- docs/kimi-forward-optimization-handoff-2026-09-26.md
- docs/kimi-forward-performance-analysis.md
- docs/performance/kimi-k3-forward-matrix-2026-09-26.md
- docs/performance/kimi-k3-forward-matrix-2026-09-26.json
- docs/performance/kimi-forward-cleanup-validation-2026-09-26.md
- docs/performance/kimi-forward-cleanup-validation-2026-09-26.json

## 先说结论

这次优化把完整专家 E896、top-k=16、8 卡、每卡 4k tokens 的 fused forward 从本机复现 PR #68 的 17.655 ms 降到历史最佳稳定结果 15.095 ms；同次 Torch 基线为 21.697 ms，加速比 1.437x。清理后使用当前源码复验得到 15.129 ms、Torch 21.740 ms、1.437x，未见材料回退。相对 PR #68，fused 延迟下降 2.560 ms，约 14.5%。仍没有达到用户最初要求的 1.6x。

历史裁剪 E32/top-k=8 的 1.702x 不能直接和 E896/top-k=16 比较。固定 top-k=16 后，裁剪版无 MoonEP 的单次 4k 对照为 14.307 ms、21.306 ms、1.489x；因此原先的差距有一部分来自 top-k，而不是专家数本身。

## 这次尝试了什么

有明显效果并保留到当前代码的方案：

- 针对 E896 大专家数场景，将单 tile 的 dispatch worker 按 expert 号轮转，减少一个小 bucket 长期只落在同一 lane 的串行化。
- 用紧凑的 routing metadata 和 wave task 表，避免每个 wave、每个本地专家都重新扫描稠密表；FC1、FC2、dispatch 和 return 共用任务表。
- 用 M128/N512/K128 替代完整场景原来的 M256/N256/K128。E896 的有效行分布使 M256 产生约 31.25% 的尾块填充，M128 约 9.62%。
- 满行 tile 直接走无 mask 路径，并跳过没有 FC1 tile 的 core 的 helper 初始化。
- 末尾 return 检查改为 checker 分片和本地 epoch 接力；收益较小，但减少了尾部重复扫描。
- top-k combine 使用四条 FP32 累加链。局部归约从约 0.278 ms 降到 0.230 ms；端到端同进程 AB/BA 对照约从 15.336 ms 到 15.242 ms，50 对中 28 对更快。
- route_to_send 的独立 host reset 是正确性修复，不是主要性能优化；它会在计时边界内增加一次设备 fill，但解决了 kernel 内 reset 导致的 dropped route 错误。

没有效果、太小或被否决的方案：

- 直接从远端 FC2 对称缓冲区拉取并归约：单链 largest-first 版本 23.872 ms、Torch 21.915 ms、0.918x，明显慢于普通回传。
- wave windows=32/64：15.388/15.816 ms，分别 1.413x/1.372x，均差于 wave=16。
- 降到 30/28 个 AICore：16.499/16.098 ms；频率升到 1650 MHz，但端到端更慢。
- FC1 M64/N1024/K64 或 K128：22.685/31.662 ms；更小尾块没有抵消额外 tile 和重复工作。
- FC2 K64/K256：局部 5.817/6.579 ms，慢于 K128 的 4.484 ms；端到端 FC2 N256、不同完成方式均未形成稳定收益。
- FC1 task grouping、row assignment、bounded read、完整行特判的多个变体在 E896 上为 16.119–25.625 ms，或只在噪声范围内改善。
- 关闭/改变 preload、cube block merge、weight eviction、return chunk、FC2 completion counter 等局部选项没有稳定端到端收益；completion counter 方案还在首轮正确性调用超过预算。
- UDMA 回传在匹配的 UDMA 编译器下通过正确性，但为 15.829 ms、1.389x；普通回传同工具链为 15.297 ms、1.438x，因此没有接入。
- 保存 FC1、FP8/FP16 兼容分支的若干大 tile 组合 compile-only 超过 300 秒；这不是普通 forward 性能结果，也不能归因于 CANN 版本错误。

## 当前矩阵性能报告

矩阵固定：8 卡 Ascend950DT、当前 fused forward 源码、top-k=16、现有偏斜路由、每卡 4k/8k/16k tokens、E32/E896、MoonEP 开关。时间为 fused median / mean / P95，单位 ms；加速比是 Torch median / fused median。

| 数据点 | fused | Torch | 加速比 | 状态 |
|---|---:|---:|---:|---|
| E32, 4k, MoonEP 关 | 22.476 / 22.489 / 22.958 | 32.492 / 32.523 / 32.635 | 1.446x | 通过 |
| E32, 4k, MoonEP 开 | 14.379 / 14.358 / 14.818 | 32.439 / 32.469 / 32.572 | 2.256x | 通过 |
| E896, 4k, MoonEP 关 | 22.470 / 22.535 / 23.387 | 33.138 / 33.173 / 33.218 | 1.475x | 通过 |
| E32, 8k, MoonEP 关 | 45.306 / 45.251 / 45.661 | 64.246 / 64.346 / 65.168 | 1.418x | 通过 |
| E32, 8k, MoonEP 开 | 29.340 / 29.410 / 30.240 | 64.054 / 64.101 / 64.179 | 2.183x | 通过 |
| E896, 8k, MoonEP 关 | 43.670 / 43.684 / 44.232 | 65.473 / 65.515 / 65.557 | 1.499x | 通过 |
| E32, 16k, MoonEP 关 | 95.512 / 95.564 / 96.300 | 126.903 / 126.980 / 127.113 | 1.329x | 通过 |
| E32, 16k, MoonEP 开 | 60.215 / 60.200 / 60.728 | 127.107 / 127.189 / 127.543 | 2.111x | 通过 |
| E896, 16k, MoonEP 关 | 86.088 / 85.941 / 86.581 | 129.420 / 129.489 / 129.757 | 1.503x | 通过 |
| E896, 4k/8k/16k, MoonEP 开 | — | — | — | 首次正确性调用的后端编译超过 900 s |

矩阵报告明确是 9/12 个有效性能点。缺失的三个点没有正确性结果、forward 延迟、加速比或有效计时期功耗频率，不能填 0，也不能称为性能失败。每个有效点都保留了 50 个原始 event 样本、8 卡逐 rank host 区间、DCMI v2 功耗/频率遥测、正确性门和占用检查。

清理后验收单点的完整参数、逐卡频率/功耗、源码指纹和 428 项 host/JIT 回归记录见 docs/performance/kimi-forward-cleanup-validation-2026-09-26.md；三处 E896 + MoonEP 仍待后续补测。

## 为什么比裁剪专家慢

E896 每 rank 有 112 个本地专家，历史均匀偏斜诊断中约 7168 个 source/expert bucket 全部非空，每个 bucket 平均只有约 73 行；E32 的 bucket 更少、更厚。E896 因此更容易暴露 dispatch lane 空转、尾块填充、专家权重工作集、缓存和跨卡同步成本。

E896 的完整权重工作集每 rank 约 7.399 GB，E32 约 0.264 GB。连续测量还看到 E896 fused 频率在 1350–1650 MHz 间波动，而同轮 Torch 和 E32 对照大多保持 1650 MHz；E896 测量期 320 个读数中有 201 个低于 1650 MHz。这个事实支持降频是性能差额的一项因素，但不能把剩余差额全部归因于功耗。

msopprof 的 Pipe/Cube 指标显示 Cube utilization 约 97.22%，AIC MAC ratio 约 81.95%；SYS_CNT 中 routing metadata 约 0.23 ms，主要时间仍在 wave pipeline。这里的 wall 打点包含等待，不能把它当成纯 GEMM 时间。

## 文档和产物状态

- 主矩阵 Markdown：docs/performance/kimi-k3-forward-matrix-2026-09-26.md
- 主矩阵结构化 JSON：docs/performance/kimi-k3-forward-matrix-2026-09-26.json
- 优化分析原文：docs/kimi-forward-performance-analysis.md
- 详细 AI 交接：docs/kimi-forward-optimization-handoff-2026-09-26.md
- 主矩阵原始产物：/tmp/kimi_forward_matrix_20260925_1426
- 编译补测和中间文件：/tmp/kimi_forward_matrix_20260926_completion
- 更早的优化、profile、PMU、编译和回归产物：/tmp/kimi_forward_perf

当前矩阵文档已写入，但不完整：主矩阵计划 12 点、有效 9 点，缺 3 个 E896 + MoonEP 点。下一步应先决定如何让这三个原始配置通过编译和正确性门，再补入同一矩阵；任何改 block 或改 compiler plan 的补测都必须作为单独 supplemental 配置记录。
