# KIMI fused-forward MAC 利用率优化核验与 paired-FC1 / dispatch 去重实验总结

日期：2026-09-27（Asia/Shanghai）

本文只记录本次会话实际完成的工作和证据。此前 Claude 会话（session 641079e4）提交的 commit、文档和 MR 不计入本次会话的增量。本轮 dispatch 去重原型编译失败；本文件记录该停止点和全部可复现实验结果。

**交接入口：同目录 `kimi-e896-moonep-compile-root-cause-2026-09-27.md` 第 8 节。**
该节汇总本会话后续的 E896 编译修复、按功能整理的提交、实验 worktree、遗留工作及
用户停止指令。本文件描述的是已结束的性能阶段，不代表允许自动继续其他优化。
交接时重新核对，本文列出的旧 `/tmp/kimi_mac_resume_20260926/` 关键原始数据
已不在当前环境；下列结果是当时记录的实验结论，本次交接没有重跑或重新计算。
源码实验 worktree 仍保留，具体当前状态和证据缺口见上述交接入口。

## 1. 目标、基线和验证边界

目标是在保持 fused Triton forward 为单个 fused kernel 的条件下，提高 KIMI-K3 trimmed 八卡场景的 MAC ratio（观测约 84.7%，目标 90%）并降低 post-router full-forward 延迟。

固定 case：

- case：performance-fwd-kimi-k3-trimmed-top16-w8-t4k
- world size：8；每卡 token：4096；全局 token：32768
- hidden：3584；FFN：3072；top-k：16；experts：32
- 单 kernel blocks：FC1 256x256x128，FC2 256x256x128，dispatch 256，wave windows 32
- 计时边界：routing metadata、dispatch、FC1、SwiGLU、FC2、combine 的 post-router full forward
- 计时协议：5 warmup、50 measured、NPU event、跨 rank MAX
- 环境：/home/vllm_kimiw/udma-patch/activate-udma.sh；八卡空闲门禁和 occupancy 监控通过

基线 worktree 为 /home/vllm_kimiw/repo/MOE_Kimi-mac-baseline-20260926，HEAD b210dab，干净。paired 候选 worktree 为 /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf，HEAD 同为 b210dab，只有本次会话未提交改动。dispatch 去重在独立 worktree /home/vllm_kimiw/repo/MOE_Kimi-dispatch-dedup-20260926，HEAD 同为 b210dab。

## 2. 本次会话新增的 paired-FC1 工作

### 2.1 权重初始化阶段重排

文件 src/mega_moe/ops/forward.py：

- 增加 panel packing，把原始 [E,H,gate_F | up_F] 重排为 [gate_p0 | up_p0 | gate_p1 | up_p1 | ...]。
- 实现等价于 reshape(E,H,2,F//panel,panel)，permute(0,1,3,2,4)，contiguous，再 view(E,H,2F)。
- 增加显式 prepare_paired_fc1_weights；forward 热路径不重复 pack。
- 权重版本变化时拒绝复用旧 packed view。
- 默认 PAIRED_FC1=False 保留原来的双 gate/up load、双 accumulator、双 dot 路径。

### 2.2 单 dot FC1 kernel

文件 src/mega_moe/kernels/fused_forward.py：

- 为 FC1 UB partition helper 增加 PAIRED_FC1 编译期开关。
- paired 分支每个 K block 连续加载 [K, 2*pair_n]，用一个 tl.dot 得到交错 gate/up 结果。
- 使用 parity BF16 UB pair buffer 和 FP32 累加区，继续使用原 ROW_SPLIT Fixpipe 输出接口。
- 没有使用 tl.join。
- 仍为单 fused kernel；没有引入 FP8、独立 FC1 kernel 或额外 L2 权重预排布。

### 2.3 验证和计时入口

文件 benchmark/layer/profile_single_kernel_forward.py：

- 增加 MOE_FWD_PAIRED_FC1=1 初始化开关。
- 增加 correctness-only 模式。
- 增加同进程 AB/BA 交错计时模式，避免把不同进程漂移误算为收益。

新增 benchmark/layer/probe_paired_fc1.py 用于 raw FC1 gate/up 输出验证。

## 3. paired-FC1 正确性

### 3.1 Raw FC1

UDMA 环境下比较 baseline 与 packed paired 的 raw gate/up 输出，覆盖：

- rows=256, K=128, F=256
- rows=300, K=192, F=384（覆盖非整块尾部）
- rows=1024, K=3584, F=3072（覆盖多 K step 和 parity buffer 重用）

六个结果均为 bad=0、max_abs=0.0、nonfinite=0。

证据：/tmp/kimi_mac_resume_20260926/raw_result.json，status=passed。

### 3.2 八卡 fused forward

paired 版本通过 normal routes、zero-receive/empty-expert、negative/out-of-range all-drop 和 occupancy gate。

证据：

- /tmp/kimi_mac_resume_20260926/paired_correctness/correctness_result.json
- /tmp/kimi_mac_resume_20260926/paired_correctness/occupancy_result.json
- /tmp/kimi_mac_resume_20260926/paired_correctness/device_occupancy.jsonl

因此 paired 路径通过了 raw FC1 和目标 case 的八卡完整 correctness gate。

## 4. paired-FC1 性能

### 4.1 同进程 AB/BA（主要端到端证据）

每个变体 5 warmup、50 measured，baseline/paired 顺序逐样本交替，NPU event，跨 rank MAX；每个计时前有 rank barrier。两轮都在计时前通过完整 correctness 和 occupancy。

| 运行 | baseline 中位数 | paired 中位数 | 中位降低 | paired 平均节省 | paired 更快 |
|---|---:|---:|---:|---:|---:|
| paired_alternating1_fixed | 13.655808 ms | 13.563936 ms | 0.6728% | 0.079432 ms | 36/50 |
| paired_alternating2 | 13.282555 ms | 13.179317 ms | 0.7772% | 0.068439 ms | 34/50 |

配对差值 bootstrap 的平均节省 95% 区间：

- alternating1：[0.032356, 0.126400] ms
- alternating2：[0.021449, 0.119380] ms

证据：

- /tmp/kimi_mac_resume_20260926/paired_alternating1_fixed/paired_ab_result.json
- /tmp/kimi_mac_resume_20260926/paired_alternating2/paired_ab_result.json
- /tmp/kimi_mac_resume_20260926/paired_evidence_summary.json

结论：paired 在该固定 case 上有小幅、可重复的正收益，约 0.67%–0.78%，但没有达到 1.8x 目标，也没有证明 MAC ratio 可到 90%。

### 4.2 pipe profiler

baseline 和 paired 各做一轮相同配置的 pipe profiler，均通过 correctness 并成功解析。该组数据用于机制分析，主要端到端结论仍以同进程 AB/BA 为准。

| 指标 | baseline | paired | 变化 |
|---|---:|---:|---:|
| MAC ratio 中位数 | 84.05% | 86.00% | +1.95 个百分点 |
| MAC ratio 范围 | 77.9%–86.3% | 84.8%–87.1% | paired 整体抬高 |
| profiler kernel duration 中位数 | 12918.40 us | 12751.92 us | -1.29% |
| MAC time 中位数 | 10864.37 us | 10902.47 us | +0.35% |
| MTE2 time 中位数 | 9742.47 us | 9652.58 us | -0.92% |
| scalar time 中位数 | 1913.74 us | 1320.61 us | -31.0% |

证据：

- /tmp/kimi_mac_resume_20260926/baseline_pipe/mac_profile_summary.json
- /tmp/kimi_mac_resume_20260926/paired_pipe/mac_profile_summary.json
- /tmp/kimi_mac_resume_20260926/baseline_pipe/benchmark_result.json
- /tmp/kimi_mac_resume_20260926/paired_pipe/benchmark_result.json

独立 profiler run 的 full-e2e 中位数是 baseline 13.66 ms、paired 13.15 ms；torch_over_single_kernel_median 是 1.547x 和 1.616x。两轮不是同一进程交错样本，不能将 3.7% 直接当作稳定收益；同进程 AB/BA 的 0.67%–0.78% 更可靠。

## 5. 对原先瓶颈判断的核验

### 5.1 MAC 约 84.7% 是否已经证明由 MTE2 单独造成

没有证明为单一根因。

MTE2 占据长时间窗口，说明搬运和等待确实影响利用率；但 paired 只使 MTE2 中位数下降约 0.92%，MAC ratio 上升 1.95 个百分点，说明低 MAC 还包含 scalar/control、readiness wait、pipeline 排布和 rank/core 不均衡等因素。aic_mac_ratio 是 AIC task-cycle 指标，不等于端到端 TFLOPS，也不等于 Cube utilization。

### 5.2 L0C 满、multibuffer 已最优、无 GM-L1 copy path 是否已逐条证实

没有完成逐条硬件下界证明：

- 本轮没有改变 L0C 容量，也没有证明所有 multibuffer 排布已达到全局最优。
- 本轮没有引入额外 GM-L1 权重预排布，不能由本轮实验证明不存在可行 GM-L1 路径。
- paired 在保持这些条件不变时仍得到小幅收益，说明“现状完全不可改善”的强断言过早。
- 但收益很小且 90% 未达成，不能把该结果解读成已经找到足以消除剩余空转的方案。

### 5.3 单 fused kernel 下是否绝对无法继续提升

没有证实为不可能。paired 在单 fused kernel 约束下通过 raw FC1、八卡 correctness，并在同进程 AB/BA 中改善约 0.67%–0.78%，直接证明仍存在小幅可优化空间。与此同时，MAC ratio 仍未达到 90%，因此目标没有完成。

## 6. dispatch 去重尝试

尝试的设计：

1. 每个源 token 在每个目的 rank 选择 canonical route（最低目的地专家，tie-break 为 top-k slot）。
2. 只发送 canonical route 的 BF16 activation；其他重复 route 发送 route weight 和 alias metadata。
3. 接收端在 Cube 消费前，根据 alias metadata 把 canonical row 复制到重复 expert row，保持原 expert-major GEMM 布局。
4. 用本地 all-core barrier 排序复制和 Cube 读取，仍保持单 fused kernel。

首次八卡 correctness 编译即失败，日志：

ub overflow, requires 14858496 bits while 1769472 bits available

证据：/tmp/kimi_mac_resume_20260926/dedup_correctness.log。

因此没有合法的 dispatch 去重 MAC 或端到端 A/B 数据，不能下“dispatch 去重运行时没有收益”的结论；只能确认当前原型被 UB footprint 阻断。按当时停止指令，本轮在此停止，不再尝试其他去重实现或其他优化方向。

dispatch worktree 当前未提交 diff：

- HEAD b210dab
- benchmark/layer/profile_single_kernel_forward.py
- src/mega_moe/kernels/dispatch_fc1.py
- src/mega_moe/kernels/fused_forward.py
- src/mega_moe/ops/forward.py
- 140 insertions、9 deletions

这些代码不应视为可用生产实现；python 语法检查和 git diff --check 在进入 NPU lowering 前通过，NPU lowering 在 UB overflow 处失败。

## 7. 本次会话工作区和提交状态

截至本轮 forward 性能优化阶段停止时，尚未创建 commit、MR 或远程分支提交。
本文随后归档到 `codex/e896-compile-fix`；该分支后续的 E896 编译修复属于新任务，
详见 `kimi-e896-moonep-compile-root-cause-2026-09-27.md`，不计入上述性能结果。

主 worktree /home/vllm_kimiw/repo/MOE_Kimi：

- HEAD eb737e7
- 本次没有修改 tracked 文件
- 未跟踪 .codegraph/ 不由本轮创建

paired worktree /home/vllm_kimiw/repo/MOE_Kimi-kimi-forward-perf：

- HEAD b210dab
- 修改 benchmark/layer/profile_single_kernel_forward.py、src/mega_moe/kernels/fused_forward.py、src/mega_moe/ops/forward.py
- 新增 benchmark/layer/probe_paired_fc1.py
- tracked diff 214 insertions、72 deletions（不含新增 probe 文件内容）

baseline worktree /home/vllm_kimiw/repo/MOE_Kimi-mac-baseline-20260926：

- HEAD b210dab
- 干净

dispatch worktree /home/vllm_kimiw/repo/MOE_Kimi-dispatch-dedup-20260926：

- HEAD b210dab
- 失败原型未提交，详见第 6 节

## 8. 最终停止点和后续建议

1. paired-FC1 是本轮唯一通过 raw FC1、八卡 correctness 和正式同进程 AB/BA 的新优化。
2. 其机制成立，但实际收益小：端到端约 0.67%–0.78%，MAC ratio 中位数约 86.0%，最高样本 87.1%，仍明显低于 90%。
3. “完全被三条 MTE2 约束锁死”和“单 fused kernel 下绝对无法提升”均未被证实；本轮只证明存在小幅空间，没有证明 90% 或 1.8x 可达。
4. dispatch 去重没有运行时性能结论，当前实现停在 UB overflow 编译失败。
5. 若未来重新启动工作，应先由负责人审查 paired 未提交 diff 是否值得保留；dispatch 去重需先重新设计 UB footprint 和 metadata 组织，再进行 correctness 和 A/B。该工作不属于本轮已完成范围。

## 9. 证据文件索引

- Raw FC1：/tmp/kimi_mac_resume_20260926/raw_result.json
- paired 八卡 correctness：/tmp/kimi_mac_resume_20260926/paired_correctness/
- AB/BA：/tmp/kimi_mac_resume_20260926/paired_alternating1_fixed/paired_ab_result.json
- AB/BA 第二轮：/tmp/kimi_mac_resume_20260926/paired_alternating2/paired_ab_result.json
- 汇总：/tmp/kimi_mac_resume_20260926/paired_evidence_summary.json
- paired profiler：/tmp/kimi_mac_resume_20260926/paired_pipe/
- baseline profiler：/tmp/kimi_mac_resume_20260926/baseline_pipe/
- dispatch 编译失败：/tmp/kimi_mac_resume_20260926/dedup_correctness.log

本文件记录的 forward 性能优化阶段至此结束。
