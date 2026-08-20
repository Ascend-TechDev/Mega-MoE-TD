# MoonEP 负载均衡接入 Mega-MoE-TD：设计报告（v0.2 初稿）

---

## 1. 背景分析

### 1.1 MoonEP 仓库是什么

`MindSpeed-MM_MoonEP`（分支 moonep）中的
`mindspeed_mm/fsdp/distributed/expert_parallel/moonep/` 是 GPU 版 MoonEP 的
**torch 语义级参考实现（昇腾路径）**。

它解决的核心问题：**EP 的跨 rank 负载不均**——经典 all-to-all 里每 rank
接收量随路由波动，接收最多的 rank 成为木桶短板。MoonEP 用一次集中
planning 把每 rank 的接收量钉死在 CAP=S·K，代价是热点远程专家的 token
要迁到别的 rank 去算，为此引入**副本槽**（把热点远程专家的权重预取到
本地，见 §2.1）。

### 1.2 Mega-MoE-TD 是什么

基于 Triton-distributed-ascend 的 **Ascend NPU MoE 超大 kernel 库**：
A2A 通信与 GEMM 融合进单个 Triton kernel，kernel 内直接用
`libshmem_device.putmem / signal_op / barrier_all` 与
`dl.symm_at` 做跨 rank 数据移动。经典路径已完整落地：

- 前向 3 大 kernel：`dispatch_fc1`（Vector 半边 putmem 散布 ∥ Cube 半边
  按 per-tile SET 信号消费做 grouped GEMM）→ `weighted_swiglu`（纯本地，
  路由权重在此乘入）→ `fc2_combine`（FC2 GEMM + remote store 回源 +
  top-k 归约）。8 卡前向 1.57~2.20×；
- 反向 5-op：`dispatch_fc2_bwd` / `swiglu_bwd`（可与 fc2 wgrad 同核并发）/
  fc2 wgrad / `combine_fc1_bwd` / fc1 wgrad，平均 1.65×。

### 1.3 异同与关系

| 维度 | MoonEP 参考实现 | Mega-MoE-TD 经典路径 | 接入后的分工 |
|---|---|---|---|
| 通信规划 | ✅ planning 五表 + 副本槽 + dedup | ❌ 无规划，接收量随路由波动（capacity_factor=1.25 截断兜底） | **MoonEP 提供** |
| 权重预取 | ✅ prefetch（pull_into） | ❌ 无 | MoonEP 语义 |
| 传输形态 | arena push/pull 抽象（宿主逐块循环） | kernel 内 producer/consumer 信号流水 | **Mega-MoE 执行引擎** |
| GEMM | ❌ mock（框架逻辑之外） | ✅ grouped GEMM 双半边 | Mega-MoE 提供 |
| dedup 省带宽 | ✅ 负编码 + epilogue 扇出 | ❌ | MoonEP 语义（v2） |
| 冗余梯度归并 | ✅ grad_reduce | ❌（无副本概念） | MoonEP 语义 |
| 反向 | ✅ plan 复用三调用 | ✅ 5-op | 两者拼接 |

一句话：**MoonEP 出"规划语义 + oracle"，Mega-MoE 出"kernel 执行引擎 +
工程基建"**。接入 = 把 MoonEP 的 planning/prefetch 语义用 Mega-MoE 的
kernel 内通信原语重写，GEMM 半边尽量原样复用。

---

## 2. MoonEP 负载均衡算法梗概

### 2.1 关键概念与记号

| 记号 | 含义 |
|---|---|
| R / E / epn | EP 世界大小 / 全组专家数 / 每 rank 属主专家数 E÷R |
| S / K / N=S·K | 每 rank token 数 / top-k / 每 rank 条目数 |
| B | **副本槽数**（默认 = epn） |
| tp | token_padding（段长对齐粒度，v1 取 2） |
| **NvS** | 每 rank 接收区槽数上界 = S·K + (tp−1)·2·epn |
| **CAP = S·K** | 每 rank 接收量钉死值（容量均摊） |

**核心概念**：

- **容量均摊与迁移矩阵 z**：group_tokens[h] 是 h 组（属主 rank=h 的
  专家）的全组 token 总数。CAP 是硬上界，盈余组的 token 必须迁给亏空组。
  B.1 的"单源填充"性质：**每个目的 rank 的远程 token 只来自单一 home 组**
  （亏空一次补满，不分割）——这保证远程非零专家数 ≤ epn ≤ B。
- **top-B 副本槽（冗余专家专区）**：每个 rank 的权重表
  `gate_up/down [Seg=epn+B, ...]` 中，行 `[0, epn)` 是本 rank 属主专家
  （home 行），行 `[epn, Seg)` 是**专门开辟的副本槽区**。B.3 为每个目的
  rank 在其远程来源专家里做 B 轮 argmax 选举，选出 top-B 个热点远程专家
  写入 `experts_to_copy [R,B]`（全组槽表，-1=空槽）。prefetch 把这些专家
  的权重**整块搬进槽行**，此后这些"远程专家"的 GEMM 全部本地进行。
- **VM 布局**：接收区按"段"组织，段序 = 本地专家段（本地序升序）+
  副本槽段（槽序）；非空段按 tp 向上对齐，空段不占行（被选为槽的专家其
  常驻段为空，token 落槽段）。`cu_seqlens [E+B]` 记段边界，
  `zero_fill_ranges` 记每段 padding 区间。
- **dst 编码**：`dst = lo·NvS + loff`（目的 rank × 段内行号）；
  **dup 负编码** `dst = -raw-1`：同一 token 的多个条目落同一目的 rank 时，
  只有第 1 份（primary）搬 payload，其余只散布路由权重、payload 由
  epilogue 从 primary 行本地扇出——**dedup 是省 payload 带宽的根源**。
- **src_info [NvS]**：本 rank 每个接收槽的出处（src_rank·NvS+offv），
  dispatch 建 dup 表与 combine 权重 gather 都靠它。

### 2.2 planning 四阶段

| 阶段 | 谁跑 | 干什么 | 产物 |
|---|---|---|---|
| **A 上报** | 全组 | 各 rank 把 tpe[E] 推进 rank0 chunk 的 TPE 区自己那行；rank0 额外把 topk/tpe 预投放进 rank1 chunk（代算原料）→ bar#1 | rank0 手握全组 tpe |
| **B 宏观决策** | 仅 rank0 | B.0 tpe_cumsum/group_tokens → B.1 单源填充 z → B.2 alloc 贪心配额（alloc[d,e]，每 rank 行和 ≤ CAP）→ B.3 top-B 槽选举 → B.4 VM 段布局；3ER 表（alloc_cumsum/tpe_cumsum/expert_offsets）广播给全组 | 3ER 广播表 + CU/ZFR/ETC/STATS |
| **C 微观落位** | 全组 | C.1 计数排序（rank≠0 各算自己的；**rank1 代算 rank0 的**）→ bar#2 → C.2 逐排序位算 dst：gidx（全组专家序）→ 在 alloc_cumsum 上二分得目的 rank lo → loff；同时把 src_info 远端写进目的 rank | dst [N]、src_info 发布 |
| **D 发还+去重** | 全组 | 从 rank0 拉 CU/ZFR/ETC/STATS 切片写输出张量；本地 dedup 负编码（逐 token 扫 K 个 dst，同目的 rank 第 2 次起取负） | MoonEPCommPlan 完整输出 |

要点：B 回答"rank1 收 2 个 e0 的条目"，C 回答"你手上第 4 个 e0 条目 =
全组第 6 个 e0，落在 rank1 槽段第 6 行 dst=18"。B 只看计数（宏观），
C 才碰具体条目（微观）——这个分层让集中规划的通信量只有 O(R·E)。

### 2.3 prefetch 阶段在干什么

planning 之后、专家计算之前：对 `experts_to_copy` 的每个非空槽 b（选中
专家 e，属主 rank = e÷epn），把属主权重表的第 e 行**整块**
（gate_up [H,2F] + down [H,F] bf16）搬进本 rank 的槽行 `epn+b`。语义
等价于"把远程专家变成临时的本地专家"。参考实现是 pull 方向
（`pull_into` 对称堆直达），Mega-MoE v1 改 **push 方向**（owner 侧发起
putmem，因 repo 只有 putmem 惯例、无 getmem 用例）。路由稳定时可整表
跳过（`skip_if_same_as`，全组一致性由 etc 全组同表保证）。

### 2.4 Triton 语言对 planning 的支持面

planning 全部计算可归约为下表算子——910B 后端目前**全部有解**，坑已
全部踩平（CASE-01..13，样例在 `tests/kernel/moonep/test_step0_smoke.py`）：

| 子算法 | 所需算子 | 910B 状态 / 绕法 |
|---|---|---|
| C.1 计数排序 | 直方图累加、exclusive cumsum、one-hot 比较 + `tl.sum`、scatter store | ✅ 直接支持。**不能用 tl.sort**（CASE-01：与其它算术共存破坏奇数 lane，且须独占 kernel + 大 N 寄存器风险）——退回 GPU 原版计数排序，天然稳定。注意 one-hot 要求 **E 为 2 的幂**（tl.arange 约束） |
| B.1/B.2 贪心循环 | while 循环、argmax/argmin | ✅ 但 **while 内无 break**（CASE-03：布尔守卫空转 `active = ok & (remaining>0)`）；**`tie_break_left=False` 不生效**（CASE-02：平局取大用手写归约 `m=max(v); idx=max(where(v==m,offs,-1))`，B.3 必须用此写法） |
| C.2 dst 编码 | searchsorted、乘加、比较 | ✅ searchsorted 退化为 R 宽比较求和（`lo = Σ(alloc_rows ≤ gidx)`）；注意 `[BLOCK,R]` int64 临时量的 **UB 预算**（CASE-05：BLOCK 1024→256，int64→int32 是第一杠杆） |
| D dedup | 移位/与/或位运算 | ✅ per-token rank 位掩码（R≤64） |
| 跨 rank 数据流 | putmem / barrier_all(_vec) / signal_op | ✅ smoke 已验证（含 putmem 同偏移发布语义，CASE-04） |
| GEMM（dispatch/FC1） | cube dot | ✅ 复用 classic；**block 下限 128**（CASE-12：910B Cube 16×16×16 bf16 dot 触发 fixp 硬崩/挂，classic tail 最低也只收缩到 32/64——moonep 调用面统一 128 起） |
| 其余 | int32/int64 加减乘除、where、minimum | ✅ 无坑 |

工程结论：**算法面 Triton 全覆盖，风险全部在实现细节的坑里**（平局规则
逐位对齐、UB 预算、排序选型、GEMM 块下限），已沉淀案例库。

---

## 3. 预期的前向 / 后向数据流

### 3.1 前向（串行链）

```
router
  │ topk [S,K] int32, tpe [E] int32, route_weights [S,K] fp32
  ▼
┌─ planning（A/B/C/D，见 §2.2）
│    出：dst [N]、cu_seqlens [E+B]、experts_to_copy [R,B]、
│        zero_fill_ranges [E+B,2]、remote_stats [2]、src_info [NvS]
▼
┌─ zero_fill（独立先序 kernel：VM/权重接收区 padding 行清零，段长随 plan 变化）
▼
┌─ prefetch（push 式：owner 把槽选中权重整块推进各 consumer 槽行）
│        ∥ 可与 zero_fill/dispatch 的 payload 通路并行（无数据依赖）
▼
┌─ dispatch + FC1（融合 kernel，M3 已落地）
│    Vector 半边：按 (dst,seg) 连续 run 逐行 putmem payload(H·2B) +
│      路由权重(4B)，每 BLOCK_M tile 一次 fence+signal_op(SET, epoch)
│    Cube 半边：复用 dispatch_fc1 的 grouped GEMM（EXPERTS_PER_RANK→Seg），
│      按信号逐 tile 消费 → fc1_out [rows_pad, 2F]
│    （v1：dup 条目照发 payload；v2：dup 抑制 + 接收端 epilogue 扇出）
▼
┌─ weighted_swiglu（纯本地：FC1 输出 × 路由权重 + SwiGLU/SiTU 激活）
│    ※ 权重在此乘入（与经典路径同位）。参考实现 combine 为 fp32 无权求和，
│      权重已被此阶段烘进行里——dup 行在 prologue 无权求和才成立
▼
┌─ FC2 + combine（prolugue）（M4：combine_push 按 src_info 推回 + topk_reduce）
│    dup 组 fp32 无权求和写回 primary（combine prologue）
│    → K 份输出推回源 rank，fp32 累加回 token-major → 路由权重 gather
  │
  ▼
out [S,H]
```

### 3.2 后向（plan 全程复用，无二次 planning）

```
grad_out [S,H]
  │
  ├─① combine 的反向 = dispatch(grad_out, plan 复用, build_dedup_map=False)
  │     把输出梯度再散布回 VM 组序 → grad_nvsh [rows_pad,H]
  ├─② 专家计算反向（全本地，含槽段）：
  │     fc2 dgrad（dispatch_fc2_bwd 形态）→ swiglu bwd（可与 fc2 wgrad
  │     同核并发）→ fc1 dgrad（combine_fc1_bwd 形态）
  │     ⚠ 槽段专家的 wgrad 落在【副本】上，不是属主
  ├─③ dispatch 的反向 = combine(grad_nvsh, plan)：每 token K 份梯度求和
  │     回 token-major → grad_hidden [S,H]
  ├─④ grad_reduce（冗余专家梯度返还）：属主把全组 (r,b) 匹配槽的
  │     wgrad fp32 拉回、按 (r,b) 字典序累加进 home 行；barrier 后清零
  │     被消费槽位（自复位，可连发）
  └─⑤ 路由权重梯度：经 combine 的权重 gather 通路回程
```

与经典路径后向的**唯一结构性差异**是 ④（grad_reduce）——经典路径没有
副本概念不需要归并；①③ 则是经典 5-op 中 dispatch/combine 的换向复用。

---

## 4. 关键设计

### 4.1 设计一：planning

**输入输出契约**（`launch_moonep_planning`，`outs` 全预分配 int32）：

| 方向 | 量 | 形状 | 说明 |
|---|---|---|---|
| 入 | topk | [S·K] int32 | 展平专家号 |
| 入 | tpe | [E] int32 | = bincount(topk)，宿主可先算 |
| 出 | dst | [S·K] int32 | 正=lo·NvS+loff，负=-raw-1（dup） |
| 出 | cu_seqlens | [E+B] int32 | **全局**段边界（压缩 [epn+B] 视图消费侧派生，CASE-10） |
| 出 | experts_to_copy | [R,B] int32 | 全组槽表（-1 空槽） |
| 出 | zero_fill_ranges | [E+B,2] int32 | 段 padding [start,count) |
| 出 | remote_stats | [2] int32 | [远程非零专家数, 被选中槽数] |
| 出 | src_info | [NvS] int32 | 槽位出处（v1 由宿主 allgather dst 倒排重建） |

**rank0/rank1 协同拆解**（GPU 版为计算重叠，参考实现逐字保留）：

```
rank0:  Phase A ─┐              Phase B(算表+写PLAN+广播3ER)
rank1:  Phase A ─┼─ bar#1 ─┐    C1(自己) + 代算rank0的C1 ─┐
rankR:  Phase A ─┘         │    C1(自己)                 │
                           └──────── bar#2 ──────────────┤
全组：                            C2(算dst+发布src_info) ─ bar#3 → D
```

- **三角色**：rank0=会计（唯一跑 Phase B，O(R·E) 串行贪心）；rank1=兼职
  助理（代算 rank0 的 C.1，因为 rank0 忙着做账）；其余 rank 自扫门前。
- **三段数据流**：① Phase A rank0 把 topk（+tpe，契约保留的死数据流，
  参考实现无消费者）**预投放**进 rank1 chunk 的 TOPK0 区——push 而非
  pull，过了 bar#1 rank1 读自己本地 chunk 直接开算；② rank1 代算出
  order0 后**回推** rank0 的 ORDER 区（先写自己 ORDER0 区留档）；③
  bar#2 一个屏障保护两路生产者（rank0 的 3ER 广播 + rank1 的 order 回推
  + 全组各自 ORDER）。C.2 全组参加，rank0 读到的 ORDER 区就是 rank1 写的。
- **动机**：若 rank0 自算 C.1，时序为 A→B→C1→bar#2，C1 垫在 B 后串行、
  全组干等；外包后 rank0 的排序藏进 Phase B 窗口，bar#2 释放时间从
  B+C1 缩短为 max(B, 2·C1)。
- **为什么是 rank1**：GPU 源码就近约定，语义上任何 rank 都能代劳（C.1
  纯函数确定），但必须恰好一个，否则双写 rank0 的 ORDER 区。
- **脆弱点**：src_info 的 fill_(-1) 必须在 Phase A（远端写在 C.2 才到，
  先填后覆盖天然无竞争；挪到 bar#3 后会把别人刚发布的值抹掉）。R=1 退化
  为 rank0 自算 C.1，dedup 仍生效（同 token K 份全落 rank0）。
- **v1（Triton）变异与洞察**：宿主 HCCL allgather 取代 Phase A 两次
  push；代算保留（c1(topk_all[0]) → putmem 对称 order0 缓冲 → c2 头部
  barrier_all_vec）。**但 v1 的 Phase B 已挪宿主，rank0 设备空闲，重叠
  动机消失**——改成 rank0 自算可砍掉 putmem + c2 头 barrier + order0
  对称缓冲（planning 零设备侧跨 rank 通信），C.1 确定性保证 bit-exact
  不变。当前保留纯为结构对齐参考实现，列为 M5 调优候选。

**对 routing 的要求**（v1 约束，launcher 断言）：

1. topk 值域 [0,E)、tpe ≡ bincount(topk)；E % R == 0 且 **E 为 2 的幂**
   （C.1 one-hot 的 tl.arange 约束；⚠ Kimi-K3 的 E=896 不满足——需
   padding 到 1024 或 kernel 泛化，见 §7 待讨论）；N % K == 0；
   R ≤ 64（dedup 位掩码）；
2. S 每 rank 恒定（MoonepTopology v1 约束，变长 token 待扩展）；
3. NvS 容量公式成立的前提是 B.1 单源填充——由算法保证，不需 routing 额外
   约束；但 **remote_stats[0] ≤ B** 是 dispatch 恒等映射的硬前提（违反
   即 RuntimeError，报出实际需要的槽数，需调大 num_slots）；
4. 平局规则逐位对齐（B.1/B.2 取小、B.3 取大）——routing 任意，但测试
   oracle 依赖这些规则可复现。

### 4.2 设计二：prefetch

- **方向选择**：参考实现 pull（consumer 发起 pull_into 直写堆内槽行，
  对齐 GPU TMA 单跳）；v1 改 **push（owner 发起）**——repo 只有 putmem
  惯例（`getmem` 无用例，且 CASE-04 语义坑在 push 侧已验证），etc 表
  planning 后全组同表，每个 owner 自行筛"哪些 consumer 的槽选了我的
  专家"。bit-exact 语义不变：槽行 b 内容 == 属主 home 行逐位相等。
- **量级**：每槽 gate_up+down = (H·2F + H·F)·2 字节；B 个槽即上限
  B·3HF·2（Kimi-K3 量级 ≈112×3×3584×3072×2 ≈ **7.4 GB**/满槽）。与
  payload（token 侧）相比这是大头，但**每步只在 etc 变化时发生**。
- **跳过契约**：`skip_if_same_as`（上一轮 etc 相同则全组一致跳过）——
  kernel 尾部 barrier 是全组集合，跳过决策必须在所有 rank 一致（etc 全组
  同表天然满足）。
- **时序**：planning 之后、dispatch/FC1 之前；与 payload 通路无数据依赖，
  可并行（§4.6）。

### 4.3 设计三：dispatch 的协同设计（M3 已落地，97e23e0）

v1 结构镜像经典 `dispatch_fc1.py` 的 all-core 流水，双半边协同：

- **Vector 半边（sub_vec 0）**：宿主把本 rank dst 按 (dst,seg) 稳定排序
  成连续 run（`build_moonep_send_meta`），kernel 按 run 逐行 putmem
  payload（H bf16）与路由权重（4B），**每 BLOCK_M tile 一次
  fence + signal_op(SET, epoch)**——信号槽键 `(src·Seg+seg)·MAX_SRC_TILES+tile`。
  段内 loff 升序 = 段内 source-major 就绪前提（MoonEP gidx 构造天然满足）。
- **Cube 半边**：**原样复用** `dispatch_fc1` 的
  `_triton_grouped_gemm_expert_n_merged_tiles_wait`，仅把
  EXPERTS_PER_RANK 换成 **Seg = epn+B**——**压缩段 id 与 gate_up 权重表
  行号恒等**（本地专家段 [0,epn) + 槽段 [epn,Seg)），kernel 体零改动。
  这是整个接入最重要的复用决策：负载均衡不新写 GEMM，只换"每 rank
  多少段"这个常数。⚠ GEMM block 下限 128（CASE-12）。

#### 4.3.1 v2 合入设计：dup 抑制 + epilogue 扇出

**语义目标**（对齐参考实现）：dispatch 对负 dst 条目只发 4B 权重、不发
payload；接收端把 primary 行 payload **本地拷贝**到各 dup 行；combine
方向由 prologue 镜像合并（dup 行 fp32 求和回 primary，权重已在 swiglu
烘进行里故无权求和成立）。payload 通信由此达到下界：每 (token, 目的
rank) 恰好传一次，两个方向 dup 行都不过网。

**核心矛盾**：参考实现里 dispatch 与 epilogue 是**两次独立 launch**
（中间有 barrier 兜底）；Mega-MoE v1 把 payload 散布与 FC1 GEMM 融合在
一个 all-core kernel 里，Cube 按 per-tile 信号消费——dup 行若不随网
到达，Cube 可能在扇出完成前读到脏行。合入设计 = 在融合流水内找到
扇出的插入点并让就绪协议自洽。

**可行性根基（关键观察）**：同一 token 的 primary 与 dup 条目**必同源**
（都由持有该 token 的 rank 发出）→ 扇出的数据依赖只有"primary 行的源
tile 信号"，无跨源等待链，接收端扇出无死锁环。且 dup 表宿主即可构建
——v1 planning 已 allgather 了 dst_all，无需参考实现的 device 侧
`build_dedup_map`。


- **方案 A（v2a，bring-up 阶梯）——拆分 launch**：dispatch 抑制版
  （Vector-only push）→ 独立 `moonep_dispatch_epilogue` kernel（poll
  primary tile 源信号 → 扇出 → 完成屏障）→ GEMM-only launch 复用
  `_triton_grouped_gemm_expert_n_merged_tiles_wait`（M3 调试期的
  `m3_gemm_only_test.py` 就是现成脚手架）。优点：三步各自独立
  bit-exact 验收、协议最简；代价：牺牲 dispatch/FC1 重叠——只作阶梯，
  不作终态。

### 4.4 设计四：reprefetch（反向重预取）

- **为什么需要**：prefetch 的槽行内容绑定**本 step 的 etc**。标准训练
  fwd(t)→bwd(t)→fwd(t+1) 里，bwd 时槽行仍是 etc(t)，**无需重取**（当前
  v1 即此假设）。但 **1F1B 流水 / 梯度累积**下 fwd(t+1) 会先于 bwd(t)
  执行，槽行被 etc(t+1) 覆盖，bwd(t) 的 fc1/fc2 dgrad（需要槽段权重）
  就读到了错误的权重。
- **设计**：plan（或至少 etc 快照）随 activation 一起 stash；bwd 入口
  检查当前槽表 == plan 的 etc，不一致则按 stash 的 etc 重推一次 prefetch
  （全组一致决策，同 §4.2 跳过契约的镜像）。
- **代价与上界**：最坏每 micro-batch 一次全槽重推（B·3HF·2 字节）；
  路由稳定时 etc 逐 step 变化少，实际重推量远小于上界。
- **当前状态**：参考实现无 reprefetch（每层独占 Buffer，槽行生命周期
  覆盖 bwd）；Mega-MoE 单飞 workspace + 层间复用需要显式设计，列 M5。

### 4.5 设计五：grad_reduce（冗余专家梯度返还）

- **问题**：槽段专家的 wgrad 由**副本 rank**算出，参数真身在属主——
  优化器只认属主行，副本梯度必须归并回去。经典路径无此概念。
- **算法**（参考 `grad_reduce.py` 语义）：对本地属主专家 e，扫全组
  etc 找全部匹配槽 (r,b)（**不排除 r==rank 的自发槽**，与 GPU prescan
  一致），把各副本 reduce buffer 的槽行 fp32 拉回，**按 (r,b) 字典序**
  累加进 home 行（逐位可复现）；`barrier` 后**清零本 rank 被消费槽**
  （自复位，支持连发不需宿主干预）。
- **与训练框架的衔接**：MindSpeed 侧已有对应处理（814babbc "accumulate
  gradients from multiple EPLB expert replicas"）——接入时对齐该语义；
  FSDP2 的 flat grad 布局下，grad_reduce 的 home 行即参数真身行，无需
  二次搬运。
- **时序**：wgrad 全部落盘后、优化器 step 之前；与 dgrad 通路无依赖，
  可与其余反向 op 重叠。

### 4.6 设计六：通信-计算掩盖的思路和方法

按层次从算法到 kernel：

1. **planning 内部（rank 间计算重叠）**：rank0 的 Phase B ∥ 其余 rank 的
   C.1（§4.1 协同）。v1 的宿主 B 表使 rank0 设备空闲，反而可以砍掉代算
   ——掩盖思路在此处退化为"宿主与设备重叠"（allgather+B 表在宿主流，
   C.1/C.2/dedup 在设备流）。
2. **dispatch/FC1（传输-计算重叠，核心手段）**：Vector putmem 生产 ∥
   Cube GEMM 消费的 **per-tile signal/wait 生产者-消费者流水**——信号
   epoch 单调递增支持单飞 workspace 复用，Cube 不等整块到齐、按 tile
   细粒度消费。这是 Mega-MoE 已验证的机制，moonep dispatch 直接继承。
3. **prefetch ∥ payload 通路**：权重槽搬运与 zero_fill/dispatch 的
   payload 散布无数据依赖，可并行发起（不同 kernel 或双流）；权重量
   大但只在 etc 变化时发生。也可以放到GEMM处。
4. **combine 侧（计算-回传重叠）**：fc2_combine 的既有形态——按专家组
   Cube GEMM ∥ Vector remote store 逐 tile 回写源 rank，末尾独立 barrier
   kernel 收尾。moonep 的 VM 布局 + dup prologue 适配后直接继承
   （M4 combine_push 按 src_info 推回）。
5. **反向**：swiglu bwd（Vector）与 fc2 wgrad（Cube）同核并发
   （`al.scope(core_mode=...)`，fused_swiglu_bwd_fc2_wgrad 已验证）；
   step4 两流专家组流水（perf_dev_bac 经验：后向 145→117ms）；
   grad_reduce 与 dgrad 通路无依赖可重叠。
6. **权重侧摊销**：属主权重一次性 register 进对称表（训练全程驻留），
   每步只增量搬 B 个槽行；路由稳定时整表跳过。

---

## 5. 变更分析（相对经典路径）

### 5.1 内存影响分析

**对称堆（ACLSHMEM heap）是主要增量**。经典路径的堆只有 payload 级
缓冲；moonep 因 prefetch push 的目标必须在对称堆，**权重表整体迁入
对称堆并扩容 epn→Seg 行**。

以 Kimi-K3（H=3584, F=3072, K=16, E=896, R=8, S=4096, tp=2, B=epn=112,
Seg=224）逐项对比（每 rank）：

| 项 | 经典路径 | moonep | 增量 |
|---|---:|---:|---:|
| 接收区（peer_mem / vm） | max_recv=S·K·1.25=81920 行 ≈0.59 GB | NvS=65984 行 ≈0.47 GB | **-0.12 GB**（去掉 1.25 因子） |
| 权重表 gate_up+down | **普通显存** epn 行 ≈7.4 GB | **对称堆** Seg=epn+B 行 ≈14.8 GB | +7.4 GB（副本槽区）且改驻堆 |
| combine_buf | （fc2_combine 内含） | [N,H] bf16 ≈0.47 GB | +0.47 GB |
| signal_mem | R·epn·tiles·… | R·**Seg**·max_src_tiles·16 ≈59 MB | 小量级 |
| 规划暂存 | 无 | order0 等零头（M5 可砍） | 忽略 |
| **对称堆合计** | **≈0.7 GB** | **≈15.4 GB** | **+~15 GB** |

要点与结论：

1. **堆从 <1 GB 涨到 ~15 GB/rank**，几乎全部来自权重副本表。这正好对上
   参考实现 README 的"对称堆默认 16 GB/rank，专家权重表大时调大"
   （`MOONEP_SHMEM_HEAP_GB`）。会话 sizing 公式：`max(估算×2, 256MB)`
   （CASE-09 堆下限）。
2. **总 HBM 不翻倍**：权重从普通显存**搬**进对称堆（aclshmem 堆也是
   device memory），真正净增的是 +B 行副本区（≈7.4 GB）+ register 期间
   ~1× 单表的临时峰值（拷完 `empty_cache` 回收，workspace.py 注明）。
3. **层间复用是关键前提**：Kimi-K3 有数十个 MoE 层，全部常驻 Seg 行表
   不可行。依赖 FSDP2 逐层 allgather + **单飞（single-in-flight）纪律**：
   对称堆峰值 ≈ 1–2 层的表（15–30 GB），层退出即 `finalize` 逆序释放。
   这决定了 moonep 前向必须严格串行逐层、不能多层 in-flight——与经典
   路径的 workspace 复用纪律一致但约束更强（权重也进堆）。


### 5.2 性能分析


**代价来源**：

1. **planning 每步开销**：allgather topk/tpe + 宿主 B 表 + C.1/C.2/dedup
   kernels。对照：经典 preprocess 在 Kimi 4K/8 卡约 5 ms——planning 的
   预算即此量级。⚠ v1 有两处**宿主同步点**（`topk_all/tpe_all` 与
   dst_all 各一次 `.cpu()`），是当前最可疑的开销点，M5 目标（设备侧
   化或与上一步 overlap）。
2. **prefetch 带宽**：etc 变化时全槽重推上限 ≈7.4 GB/step（Kimi 满槽），
   ≈ 数 ms 级 @数百 GB/s；路由稳定时为 0（skip 契约）。但**训练初期
   路由随机，几乎每步全推**——冷启动阶段是净开销。
3. **grad_reduce 额外一趟**（后向）：槽 wgrad 拉回属主 ≈B·3HF·4B fp32，
   与 dgrad 无依赖可重叠（§4.6-5）。
4. dispatch 发送端 putmem 逐行、dup 照发（v1）——与经典同形态，v2 dedup
   才有净减。

**定量基线**（经典 Kimi-K3 8 卡 fwd，tokens/rank=4K，总 44.7 ms =
preprocess 5.0 + dispatch+FC1 19.2 + swiglu 2.3 + FC2+combine 16.5）：
moonep 的增量挂在 preprocess（+planning/prefetch），减量出现在
dispatch+FC1 与 FC2+combine 的 tail。**净效应依赖路由分布**——均衡路由
下可能净负（planning+prefetch 是纯开销），重倾斜下 dispatch 段收益放大。
M7 用 bias_ratio∈{0, 0.5, 1, 2} × B∈{epn/2, epn} 扫描定夺；当前 M3 只有
正确性数字，无 perf 数字。

**风险清单**：宿主 .cpu() 同步；prefetch 与 payload 争带宽（同行同时
发起）；planning 内多次小 kernel launch 的固定开销（N 小时可感知）。

### 5.3 接口变动分析

| 接口面 | 经典路径 | moonep 后 | 兼容策略 |
|---|---|---|---|
| 用户 op | `FusedMoEForward` | `MoonepForward`（M4 组装中） | 目标同签名（hidden/router 输出），编排内部换 planning+prefetch+Seg |
| 配置 | `MoEForwardConfig`（capacity_factor 等） | + `MoonepConfig`：num_slots(B)/token_padding/prefetch skip 策略；capacity_factor 语义被 NvS 公式取代 | M6：config 加 `planner='classic'\|'moonep'` 开关，默认 classic |
| 权重入口 | `pack_gate_up_weights`→[E_local,H,2F] 普通张量 | `register_weights`：一次性搬进对称表 [Seg,...]，**调用方须弃用原张量**（~1× 峰值后 empty_cache）；down 也须 register | 形状断言+文档；层数据加载处一次性迁移 |
| router 输入 | topk/weights 任意 int | topk 须 **int32 展平 [S·K]**、值域 [0,E)；tpe 可内部 bincount | launcher 内部 cast/断言 |
| 维度约束 | 宽松（变长 token、任意 E） | v1：**E 2 的幂**（⚠ Kimi 896 需 pad 1024 或 kernel 泛化）、S 恒定、N%K==0、R≤64、GEMM block≥128 | 约束集中写在 MoonepTopology + launcher 断言，违反早炸 |
| 反向接口 | 5-op 显式调用 + autograd | plan 全程复用：autograd ctx **多 stash 一份 plan（含 etc 快照）**；wgrad 后插 grad_reduce | `MegaMoEBackwardFunction` 平行新增 moonep 变体 |
| 训练循环 | 无感知 | 1F1B/梯度累积需 reprefetch 钩子（§4.4）；对称堆 sizing 环境变量 | M6 提供 ctx 检查钩子；默认关闭（假定 fwd→bwd 相邻） |
| 训练框架 | — | MindSpeed 侧已有 `ep_plan.dispatcher: shmem` 与 EPLB 多副本梯度累加（814babbc）——语义对齐点 | FSDP2 flat grad 布局 = grad_reduce home 行，零二次搬运 |
| 经典路径文件 | — | **零改动**（moonep_* 平行扩展；唯一交集是只读 import dispatch_fc1 的 Cube 半边函数） | 合入纪律 |

结论：对外接口（op 签名/输出语义 out [S,H]、权重乘入位置）保持不变；
变动集中在**初始化期**（权重 register、config）与 **autograd ctx**
（plan stash），运行期调用面与经典同构。

---

## 6. 测试方法

### 6.1 如何保持数据和原来的一致

三层闸门，从便宜到贵：

1. **宿主 oracle 逐位对拍（bit-exact，主力）**：`moonep_ref`（vendored
   参考实现）+selftest 保证 vendor 无损。planning 全部输出是整数表
   （无浮点），对拍标准就是**逐位相等**——`test_planning_bitexact.py`
   三用例（rand/dup/R=1）全绿即 M1 验收；prefetch 槽行、dispatch 的
   VM 行/权重/padding 清零同理逐位（`test_prefetch_bitexact.py` /
   `test_dispatch_bitexact.py`，M3 已全绿）。
2. **不变量断言（结构正确性）**：planning 不变量（meta 布局逐字复核 /
   dst 值域 / cu 单调且 tp 整除 / etc 值域 / NvS 上界）独立于 oracle 存在，
   换实现也成立；同输入二次规划逐位一致（确定性回归）。
3. **GEMM/浮点半边容差对拍**：融合 kernel 里的 GEMM 有浮点，退为 fp32
   宿主参照 + 容差（M3 段级 rtol/atol 5e-2）；M4 的 L2 整链对拍挂
   `torch_moe_fwd_golden`（4e-2）；最终对齐 `tests/layer/test_moe_suite.py`
   的可微 torch oracle（`ops/_torch_forward.py`）。

工程纪律：每里程碑一个 bitexact 测试文件 + 一个 commit；bring-up 案例
（CASE-nn）随里程碑追加进 dev_cases.md；vendor 升级走 cp + `diff -rq`
+ commit 注记流程。

---


单机8卡：32个专家 topk8

## 7. 仓库的合入规划

### 7.1 文件层次（对齐经典路径既有分层：kernels=kernel+launcher，
runtime=宿主元数据+对称堆，ops=用户态编排+autograd，tests/benchmark
按层分目录）

```
src/mega_moe/
├── kernels/                      # Triton kernel + launcher（现有分层）
│   ├── moonep_planning.py        # ✅ M1（c1/c2/dedup/push_r0_order）
│   ├── moonep_prefetch.py        # ✅ M2（push 式）
│   ├── moonep_zero_fill.py       # ✅ M3（97e23e0）
│   ├── moonep_dispatch.py        # ✅ M3（Vector run 推送 + Cube 复用 dispatch_fc1）
│   ├── moonep_combine_push.py    # 🔄 M4：按 src_info 推回 + topk_reduce
│   │                             #    （优先复用 fc2_combine，同 Seg 替换思路）
│   ├── moonep_combine_prologue.py# ⬜ M4 或并入 combine_push（dup 组求和，小 kernel）
│   ├── moonep_dispatch_epilogue.py # ⬜ v2a：dup 扇出独立 kernel（§4.3.1 方案 A）；
│   │                             #    v2b 并回 dispatch 融合 kernel（方案 B 组合信号）
│   ├── moonep_dispatch_bwd.py    # ⬜ M5：= dispatch 半边换向（grad 散布，复用 plan）
│   ├── moonep_combine_bwd.py     # ⬜ M5：= fc1 dgrad + 反向 push（combine_fc1_bwd 形态）
│   └── moonep_grad_reduce.py     # ⬜ M5：槽 wgrad 拉回属主 + 槽清零
├── runtime/
│   ├── moonep_routing.py         # ✅ M3（segment/send/recv 三构建器）
│   └── moonep_workspace.py       # ✅ M2（8 张对称张量定序 + register_weights）
├── ops/
│   └── moonep_forward.py         # 🔄 M4：MoonepForward.forward 组装
│                                 #    （对齐 ops/forward.py 形态）
├── config.py                     # ⬜ M6：MoonepConfig（num_slots/B、tp、跳过策略）
│                                 #    + planner 开关
└── moonep_ref/                   # oracle，不动（升级走 cp+diff 流程）

tests/kernel/moonep/              # per-milestone bitexact（现有模式延续）
tests/layer/                      # M6 起挂进 test_moe_suite（可微 oracle 对拍）
config/_shapes.py                 # M7：加 moonep CaseGroup（复用模型档）
benchmark/layer/                  # M7：bench_moe_suite 加 moonep case
docs/moonep_dev_cases.md          # ✅ 案例库 CASE-01..13，随里程碑追加
docs/moonep_integration_report.md # 本报告
```

### 7.2 分工


planning 部分 ： 赖俊杰、周靖淦

zero-fill, prefetch，dispatch, epilogue 扇出, combine: 王淳西

后向部分（reprefetch/dispatch/grad_reduce）：周靖淦，王淳西

---

## 附：待讨论问题（下一轮完善点）

1. **Kimi E=896 vs C.1 的 2 幂约束**：padding 到 1024（tpe/bincount
   minlength、槽表、段表全按 1024 走，浪费 128 个空段）或泛化 kernel
   （one-hot 改 masked 比较）——两者都便宜，选哪个？
2. **权重表的层间驻留策略**：FSDP2 逐层 allgather + 单飞纪律下，对称堆
   峰值 ≈1–2 层表（15–30 GB/rank）；是否需要"表常驻 + 槽行轮换"的变体
   （home 行常驻堆外、只槽区进堆）换取多层 in-flight？
3. **v2 dedup 的收益判定**：dup 照发（v1）多耗的 payload 带宽 vs 抑制+
   扇出的 kernel 复杂度——用 Kimi-K3 真实路由的 dup 率分布定夺；
4. **reprefetch 触发粒度**：etc 快照整表比对 vs 逐槽 diff 增量重推；
5. **combine 侧 Seg 替换是否像 dispatch 一样"零改动"**：fc2_combine 的
   专家组流水按 Seg 重排后 tile 数量变化，需 M4 实测；
6. **B 的取值策略**：默认 epn（全覆盖上界）是否最优——内存（§5.1：
   B 每行 ≈63 MB）vs 迁移量的权衡，M7 用 bias 谱系扫；
7. **planning 宿主化的最终形态**：v1 宿主 B 表 + 设备 C1/C2 的混合分工
   是否值得全设备化；两处 .cpu() 同步点（§5.2）的消除方案。
