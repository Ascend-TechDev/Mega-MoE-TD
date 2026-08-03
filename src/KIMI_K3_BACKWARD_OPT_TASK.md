# Kimi-K3 8卡后向优化任务 — 状态 handoff

> 任务：8卡 EP 下优化 Kimi-K3 后向。配置 `(ntokens, hidden=3584, ffn=3072, topk=16, E=896)`，
> 8卡 → 112 expert/rank。当前聚焦 **step5 fc1_wgrad 病态慢**（占后向 96%）。
> 本文件记录进展与未决问题，方便后续恢复。

## 运行命令

```bash
# 8卡 per-stage profiling（只跑 Kimi 3 配置，1GB 对称堆够 4096）
PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
TRITON_CACHE_DIR=/tmp/triton_mb \
MOE_KIMI=1 MOE_ASH_GB=1 \
torchrun --nproc-per-node=8 --master_port=29514 debug/prof_backward.py

# 8卡端到端正确性+性能（Kimi only）
PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
TRITON_CACHE_DIR=/tmp/triton_mb \
MOE_KIMI=1 \
torchrun --nproc-per-node=8 --master_port=29515 tests/layer/test_moe_backward.py

# 单卡 wgrad 微基准（隔离诊断，编译慢~5min）
PATH=/home/z00905891/triton_dist/AscendNPU-IR/build/bin:$PATH \
TRITON_CACHE_DIR=/tmp/triton_mb python debug/bench_kimi_wgrad.py
```

> 注意：8卡多次跑前确认无残留进程（`pkill -9 -f prof_backward; pkill -9 -f torchrun`），
> 否则 HCCL 端口 16666 / torchrun 端口 29500 被占报 EADDRINUSE / EI0020。

## 核心问题

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

- `src/mega_moe/kernels/common.py`
  - `WGRAD_BLOCK_M=256`（R1，恢复）、`BLOCK_SIZE_K=256`（R2）、`BLOCK_SIZE_M=64`
- `src/mega_moe/kernels/transposed_grouped_gemm.py`
  - **`N, K, num_tiles_n, num_tiles_k` 改为 `tl.constexpr`**（Kimi -20%，新）—— 保留
  - `use_bytecode=True`、`torch.empty`（R4/R5）
- `tests/layer/test_moe_backward.py`：加 `MOE_KIMI=1` 配置分支（3 个 Kimi 配置）
- `debug/prof_backward.py`：加 `MOE_KIMI=1` 分支、diag 打印(total_recv/peer_mem)、
  skip-on-fail、`MOE_ASH_GB` 环境变量调对称堆大小
- `debug/bench_kimi_wgrad.py`：单卡 wgrad 隔离微基准（编译慢、数字不一致，仅参考）

> 之前已提交的 5 轮优化（commit `657461d`）：wgrad BM=256 + input-grad K=256 + empty + use_bytecode，
> 长序列 8192/16384 各 ~1.6x。对 Qwen 仍是有效优化，不要回退。

## 下一步建议

1. **查 AscendKernelWiki**：有无 "persistent kernel + 大 tile 数 + 大 E + 大非2幂 shape" 的已知 codegen 病态解法
   （`python3 scripts/query.py "persistent kernel low cube utilization many tiles stalling"`）。
2. **结构性重写 wgrad**：绕开 75264-tile 串行 task loop。候选：
   - 2D/3D grid 按 (E, ntn·ntk) 分维，让 program_id 直接映射 (e, tn, tk) 省整除；但总程序数仍 75264，需分块。
   - 非 persistent 分块调度：grid=(ncore,) 每核处理一段连续 (e,tn) 块，提高 L2 局部性（注意 uneven 会失衡，见 debug/bench_wgrad2.py 的 blocked 变体在 uneven 下崩）。
   - 按 E 维切分 + 多流。
3. **解决 8 卡对称内存上限**：让 8192/16384 能跑（减小 peer_mem 或调 ash 配额）。
4. **验证 constexpr 改动对 Qwen 不回归**（Qwen 长序列基线 8192=65.9ms / 16384=133.5ms）。

## 关键数据备忘

- Kimi 4096 8卡 per-stage（constexpr 版）：total 6057ms，step5=5805(95.8%)，step3=102(1.7%)，
  step4=84(1.4%)，step1=57(0.9%)，step2=9(0.2%)。
- Qwen 16384 2卡（已优化，commit 657461d）：total 132ms，step5=42(31.5%)，step4=32(24%)，
  step1=29(22%)，step3=22(16.5%)。
- 910B1 L0 约束（bf16）：L0A 64KB (BN·BM≤32768)、L0B 64KB (BM·BK≤32768)、L0C 128KB (BN·BK≤32768)。
  当前 wgrad tile [BN=128,BM=256,BK=256] → b=[256,256]=128KB 超 L0B，但实测非根因。
