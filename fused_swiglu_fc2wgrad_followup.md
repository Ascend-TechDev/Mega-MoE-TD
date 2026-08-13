# fused swiglu_bwd + fc2_wgrad 集成:负增益复盘与后续提炼方向

> 状态(2026-08-13):代码已集成进 `triton_dist/Mega-MoE-TD`(`src/mega_moe/kernels/fused_swiglu_bwd_fc2_wgrad.py` + `ops/backward.py` 的 `MOE_FUSED_SWIGLU_WGRAD` 分支),**默认关、未 commit**。正确性通过,但 Kimi-K3 w8 t4k 端到端**慢 ~28ms(112→140ms)**,故未启用。本文档供后续提炼解决使用。

## 1. 这个融合 kernel 是什么

外部工作流(`/home/d00841237/code/triton_gen/zhoujinggan/`,`report.md`)写的 `fused_swiglu_bwd_fc2_wgrad_v1`:把后向 **step2(SwiGLU bwd,Vector)+ step3(fc2 wgrad,Cube)** 融进单 launch,两 scope(`al.scope(core_mode=vector/cube, disable_auto_sync=True)`)数据独立、无 barrier → Ascend 同 AICore 上向量/立方双引擎并发,wall ≈ max(cube,vec) ≈ cube(swiglu 被藏住)。

三个结构性优化(相对仓库自带 `transposed_grouped_gemm`):
1. **消 host `.T.contiguous()`**:cube 直读 `[M,N]`(交换 stride),不做转置拷贝。
2. **cube 任务连续划分**:每个 program 走连续 (e,tn,tk) 段,局部性好。
3. **Cube/Vector 同 launch 并发**:swiglu 被 cube 藏住。

隔离 bench(910B1, M∈{4096,8192,16384}):相对 `torch-swiglu + npu_grouped_matmul` geomean **2.49×**,精度通过 0.05。

## 2. 集成所做(已落在 triton_dist)

- 新增 `fused_swiglu_bwd_fc2_wgrad.py`,带两个集成修正:
  - **cube tile 用本地 `FUSED_WBM=64`**(env `MOE_FUSED_WGRAD_BLOCK_M` 可调),**不 import 仓库 `WGRAD_BLOCK_M=256`**——BM=64 既是报告调优最优,也保证"直读 [M,N] 不转置"在 UB-bus-error 阈值之下(BM=256 会触发,这正是 standalone wgrad 要 `.T.contiguous()` 的原因)。
  - **连续划分加 remainder guard**(`blk=ceil(total/ncore)` + `if task<total`)——原码 `total//ncore` 对非整除 shape 会静默丢任务。
  - 保留:无 `use_bytecode`、无 `sub_vec_id` gate(纯本地 GM,无 SHMEM)。
- `backward.py`:`MOE_FUSED_SWIGLU_WGRAD=1` 时,step1 后一次性融合 step2+step3,返回 `(grad_fc1_output, grad_gate, grad_fc2)`;step5/step4 不变;不走 side stream。

## 3. 验证结果:负增益

| 路径 | Kimi-K3 w8 t4k 后向 | vs torch |
|---|---:|---:|
| fused-off(默认) | **112.0 ms** | 1.81× |
| **fused-on** | **140.6 ms** | 1.43× |

正确性门 **PASSED**(融合 kernel 数学正确),但端到端 **+25%**。

## 4. 根因(为什么 2.49× 没转化)

1. **真实 M 远大于报告测试 shape**。Kimi w8 t4k 的 `total_recv ≈ 6.5 万`行;报告最大只测到 M=16384。M=65000 时每专家 ~580 token,cube 每 task ~10 个 `tl.dot`(报告最大 3 个)。报告自诊断 cube 是 **memory/overhead-bound**,dot 数次线性增长 → 大 M 下 per-task 开销放大,triton cube wgrad 退化。
2. **真实 step2 已是快的 triton-swiglu**(`nvec`,B2 优化过),没什么"可藏";报告的 2.49× 里相当一部分是藏掉了较慢的 **torch**-swiglu,这部分在真实后向不存在。
3. **npu_grouped_matmul 是高度优化的厂商 op**,大 M 下比 triton cube wgrad 快。融合(消转置拷贝 + 连续划分 + 藏 swiglu)的优势只在**小 M** 时能赢它。

一句话:**该融合 kernel 的优势在小 M 成立,在 Kimi 这种大 M 下,triton cube wgrad 本身就比 npu-grouped 慢,融合救不回来**。

## 5. 后续提炼方向(待解决)

1. **按 M 分流(gating)**:测出 triton-fused-cube 与 npu-grouped 的**交叉点 M***。`M < M*` 走融合(短序列/小 batch 有收益),`M ≥ M*` 走 npu-grouped。先在隔离环境(对齐报告的 profiler 口径)用**真实 M=65000** 复测,确认交叉点。
2. **大 M 下让 triton cube wgrad 变快**:
   - 扫 `MOE_FUSED_WGRAD_BLOCK_M ∈ {128,256}`(大 M 下更大 tile → 更少 dot → 更低 overhead);但 BM=256 + 直读 [M,N] 有 UB-bus-error 风险,需验证。
   - 若 cube 是写带宽封顶(`grad_w` 输出 ~2.47GB@8K,大 M 更大,所有路径共担),则 tile 调参收益有限——需算法级改动(split-K、更优 reduce 布局)。
3. **只融合、不替换 wgrad 后端**:保留 npu-grouped 做 wgrad,仅把 swiglu 与之并发(但 B3 已证软件流并发是 12.5× 灾难;真并发只能靠 in-kernel `al.scope`,而 npu-grouped 是 CANN op、无法进 triton kernel)。→ 这条路目前不通,除非 cube wgrad 用 triton 且够快。
4. **精度复核**:fused swiglu 与 standalone 逐字节一致;cube wgrad 换 triton,仓库容差已过。如后续调 tile,重跑功能 suite 兜底。

## 6. 现状(快照)

- 默认后向 = **112ms**(B1+B2 优化 + B4 gate-pack;commit 至 `ad65b23`)。
- fused 代码**默认关、未 commit**;启用:`MOE_FUSED_SWIGLU_WGRAD=1`(可选 `MOE_FUSED_WGRAD_BLOCK_M`)。
- 关键结论:**不要在 Kimi 大 M 下默认启用**;小 M 场景待复测确认收益后再考虑 gating。
