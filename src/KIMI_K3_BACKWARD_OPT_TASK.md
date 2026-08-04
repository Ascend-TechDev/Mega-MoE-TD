# Kimi-K3 8卡后向优化任务 — 状态 handoff

> 任务：8卡 EP 下优化 Kimi-K3 后向。配置 `(ntokens, hidden=3584, ffn=3072, topk=16, E=896)`，
> 8卡 → 112 expert/rank。

## 最新进展（step3/5 torch 替代 — 已落地）

triton 的 `transposed_grouped_gemm` wgrad kernel 对 Kimi shape 有 codegen 病态
（step5 fc1_wgrad 0.16% peak，5805ms；step3 fc2_wgrad 102ms）。已加 torch fallback
（`_grouped_wgrad_torch`，逐专家 matmul），env 开关默认关（不影响 Qwen 等其它 shape）。

**Kimi 4096 8卡 per-stage 对比（prof，带 sync fence）：**

| 路径 | 时间 |
|---|---:|
| 全 triton (baseline) | 6057ms |
| hybrid (step5 torch) | 272ms |
| **hybrid (step3+5 torch)** | **183ms** ✅ |
| 全 torch e2e | 206ms |

step3+5 用 torch 后 hybrid (183ms) **反超全 torch (206ms) 1.12x**，比原全 triton 快 33x。
triton 在 step1/2/4 (comm+GEMM+swiglu) 仍比 torch 快 ~24ms，torch 在两个 wgrad 上碾压 triton。

剩余瓶颈（hybrid 模式）：**step4 combine_fc1 84ms (46%)** 和 **step1 dispatch_fc2 56ms (31%)**，
都是 comm+A2A+GEMM 融合阶段，triton 已比 torch 快，是当前真正的墙。

## 运行命令

```bash
# 8卡 per-stage profiling + torch e2e 对比（只跑 Kimi，1GB 对称堆够 4096）
PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
TRITON_CACHE_DIR=/tmp/triton_mb \
MOE_KIMI=1 MOE_ASH_GB=1 \
MOE_FC1_WGRAD_TORCH=1 MOE_FC2_WGRAD_TORCH=1 \
torchrun --nproc-per-node=8 --master_port=29514 debug/prof_backward.py

# 8卡端到端正确性+性能（Kimi only，2048 tokens 能放下双路径内存）
PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
TRITON_CACHE_DIR=/tmp/triton_mb \
MOE_KIMI=1 MOE_ASH_GB=1 \
MOE_FC1_WGRAD_TORCH=1 MOE_FC2_WGRAD_TORCH=1 \
torchrun --nproc-per-node=8 --master_port=29515 tests/layer/test_moe_backward.py
```

环境开关：
- `MOE_FC1_WGRAD_TORCH=1` / `MOE_FC2_WGRAD_TORCH=1`：step5/step3 wgrad 用 torch 替代 triton（Kimi 必开）。
- `MOE_ASH_GB`：对称内存堆 GB（8卡默认 2GB×8=16GB 超驱动上限，Kimi 用 1GB；4096 peer_mem 0.61GB 够）。
- `MOE_KIMI=1`：只跑 Kimi 配置。

> 注意：8卡多次跑前确认无残留进程（按 pid kill，勿用 `pkill -f torchrun` 会杀自身 shell），
> 并等 ~30s 让 HCCL 端口 16666 释放，否则报 EADDRINUSE / EI0020。

## 核心问题（triton wgrad 病态，已用 torch 绕过）

**4096 Kimi 8卡：step5 fc1_wgrad = 7.2s，占后向 96.6%。** 0.16% cube peak（Qwen 是 4%），
NPU 功耗仅 90W（cube 闲置 stall）。其它阶段合计 <300ms 可忽略。

Kimi 8卡 4096 实测 M(recv)=84917，skew 5.2x（最热 expert 3970 tokens vs 均值 758）。
wgrad tile 数 = E·ntn·ntk = 112·48·14 = 75264（Qwen 的 12x），24核 persistent 每核 3136 tile 串行。

## 诊断历程（逐个排除）

| 测试 | step5 结果 | 结论 |
|---|---|---|
| baseline (BM=256, N/K runtime arg) | 7204ms | — |
| 候选1: BM=64（回退 R1） | 6707ms | tile 大小不是根因 |
| micro-bench BM256BK128（修 L0B 超） | 55.7s（不可靠） | L0B 超 L0B 非根因 |
| grid=(total,) 75264 程序 | 报错 | 程序数过多不可行 |
| **constexpr N,K,ntn,ntk** | **5805ms (-20%)** | mask/整除是次要因，已保留 |

## 根因定性（未解决）

- **codegen/runtime 病态**：roofline 显示 compute~12ms、HBM~20ms，实测 5.8s = **290x 超出**。
  tile / mask / 整除 / L0B 都已排除或仅次要。
- **per-tile overhead ~1.85ms 主导**：BM64(12 iter) 与 BM256(3 iter) 时间几乎相同 →
  不是 m-loop 计算，是 tile 设新时代价（task 分解、split_size 标量 load、mask、store）。
- **编译本身病态慢 ~5min**（正常 wgrad 几秒）—— 编译器在 Kimi constexpr (N=6144, K=3584)
  上挣扎，生成差代码。这是最强信号。
- 75264 tile / 24核 persistent 串行 task loop 可能是结构根因。

## 第二个阻塞问题（8192/16384 跑不了）

8卡对称内存上限：`8×2GB=16GB` 超驱动对称内存总量上限 → 2GB 堆 `aclshmem_malloc` 失败。
改 1GB 堆后 4096（peer_mem 0.61GB）能跑，但 **8192 peer_mem=1.23GB、16384 ~2.4GB 仍失败**。
`prof_backward.py` 已加 skip-on-fail 跳过。要测长序列需先解决（减小 peer_mem 或调驱动对称内存配额）。

## 当前工作树改动（未提交）

- `src/mega_moe/ops/backward.py`：新增 `_grouped_wgrad_torch`（torch 逐专家 wgrad），
  step3/step5 用 `MOE_FC2_WGRAD_TORCH` / `MOE_FC1_WGRAD_TORCH` 开关切换 torch fallback（默认关）。
- `tests/layer/test_moe_backward.py`：`MOE_ASH_GB` 调对称堆；Kimi 配置加 2048（双路径内存能放下）；
  config 循环 skip-on-fail（8192/16384 peer_mem 超堆时跳过不崩）。

> 之前已提交（commit `657461d` + `6aeefad`）：
> - `657461d`：wgrad BM=256 + input-grad K=256 + empty + use_bytecode（Qwen 长序列 1.6x，不要回退）。
> - `6aeefad`：wgrad `N,K,num_tiles_n,num_tiles_k` 改 constexpr（Kimi -20%）+ MOE_KIMI 开关 + 本 handoff 文档。
> `debug/prof_backward.py`、`debug/bench_kimi_wgrad.py` 等诊断脚本未跟踪（与惯例一致）。

## 下一步建议

1. **啃 step4 combine_fc1 (84ms, 46%) / step1 dispatch_fc2 (56ms, 31%)**：hybrid 模式下的剩余墙，
   都是 comm+A2A+GEMM 融合阶段，triton 已比 torch 快。step4 的 push_reduce（reverse-A2A ~512MB + barrier）
   可能是 comm 主导，难压；step1 类似。可考虑 al.scope 融合（前向 fc2_combine 的做法，高风险）。
2. **解决 8 卡对称内存上限**：8192 peer_mem 1.23GB / 16384 ~2.4GB > 1GB 堆，跑不了。
   需减小 peer_mem 或调驱动对称内存配额，才能测长序列。
3. **triton wgrad 病态根因仍开放**（已用 torch 绕过）：5min 编译 + 290x 超 roofline + per-tile 1.85ms overhead。
   若要彻底修 triton 路径，查 AscendKernelWiki "persistent kernel + 大 tile 数 + 大非2幂 shape codegen" 或
   结构性重写 75264-tile task loop（2D grid 按 E 分维 / 非 persistent 分块）。
4. **验证 torch fallback 对 Qwen 不回归**：默认关，仅 Kimi 开；Qwen 跑基线确认 8192=65.9ms / 16384=133.5ms 不变。

## 关键数据备忘

- Kimi 4096 8卡 per-stage（constexpr 版）：total 6057ms，step5=5805(95.8%)，step3=102(1.7%)，
  step4=84(1.4%)，step1=57(0.9%)，step2=9(0.2%)。
- Qwen 16384 2卡（已优化，commit 657461d）：total 132ms，step5=42(31.5%)，step4=32(24%)，
  step1=29(22%)，step3=22(16.5%)。
- 910B1 L0 约束（bf16）：L0A 64KB (BN·BM≤32768)、L0B 64KB (BM·BK≤32768)、L0C 128KB (BN·BK≤32768)。
  当前 wgrad tile [BN=128,BM=256,BK=256] → b=[256,256]=128KB 超 L0B，但实测非根因。
