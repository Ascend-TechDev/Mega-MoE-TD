# MoonEP 移植开发案例记录（dev_moonep 分支）

> 目的：bring-up 过程中处理过的问题按「案例」沉淀——现象、定位过程、
> 根因、解决、预防锚点。每个里程碑追加。格式：`CASE-nn 标题`。
> 关联：`tests/kernel/moonep/test_step0_smoke.py`（多数案例的可复现样例）。

---

## CASE-01 tl.sort 与其它算术共存会破坏奇数 lane

- **现象**：含复合键构造 + `tl.sort` 的 kernel，排序输入在**偶数下标**
  正确、奇数下标是 ±1e9 垃圾值；纯 load+sort+store 的 kernel 却完全正确。
- **定位**：①换 `zeros` 输出缓冲排除未初始化干扰——垃圾仍在，且 key 在
  sort **之前**存也一样烂；②把 `v*N+offs` 拆成单独 kernel 验证——算术
  本身全对。⇒ 排序的存在污染了同 kernel 内其它通路（layout/register
  污染），不是算术错。
- **解决**：sort 独占 kernel（造键、排序两次 launch）；int64 复合键
  `key = v*N + arange`，`skey % N` 即稳定 order（与 torch stable argsort
  逐位一致）。后续因大 N（65536×8B=512KB）寄存器风险，C.1 最终改用
  **计数排序**（GPU 原版算法），tl.sort 方案保留在 smoke 里作证据。
- **预防**：`test_step0_smoke.py::test_tensor_primitives`（N=8/128）。

## CASE-02 tl.argmax(tie_break_left=False) 不生效

- **现象**：三重平局期望取最大下标 2，返回 0（与 True 行为相同）。
- **根因**：910B 后端未实现该参数，静默按默认（取小）处理。
- **解决**：平局取大用手写归约 `m=tl.max(v); idx=tl.max(tl.where(v==m,
  offs, -1))`。B.3（experts_to_copy 的 top-B argmax）必须用这个写法。
- **预防**：同 CASE-01 测试的 am_r 断言。

## CASE-03 while 循环内不支持 break

- **现象**：`UnsupportedLanguageConstruct` 编译失败，指向 while 内的
  `break`。
- **解决**：布尔守卫空转——`active = ok & (remaining > 0)`，每轮用
  mask 屏蔽失效轮的更新（B.1/B.2 贪心循环的标准写法）。
- **预防**：同 CASE-01 测试的贪心 while 数值断言。

## CASE-04 putmem 目标寻址语义：同 offset 发布，不是"写到 peer 的行区"

- **现象**：2-rank roundtrip 读回全 0（新对称堆默认清零）。
- **定位**：画时序发现我把 dst 写成了 `sym_ptr + peer*ELEMS`（"peer 的
  行区域"），导致**我读的区域没有任何人写过**；对照 `runtime/routing.py`
  的用法：`putmem(row, row, size, peer)`——**dst 与 src 是同一个本地
  指针**，语义是"把我的这段发布到 peer 堆上**相同偏移**处"。
- **解决**：发布语义一律 dst==src；要写 peer 的**其它**偏移时，dst 传
  "该偏移的本地指针表示"（本地堆与 peer 堆同构）。
- **预防**：`test_step0_smoke.py::test_step0_putmem_barrier_roundtrip`。

## CASE-05 910B 单程序 kernel 的 UB 预算：BLOCK=1024 的 2D int64 中间量溢出

- **现象**：C.2 kernel 编译报 `ub overflow, requires 2031872 bits while
  1572864 bits available`（254KB > 196KB）。两次报错**字节数完全相同**
  是定位关键——说明挂的不是我刚改的 c1（rank0 根本不 launch c1），而是
  c2 的 `[BLOCK=1024, R]` int64 多个临时量（rows/where/乘加）。
- **解决**：`_C2_BLOCK` 1024→256 后通过。经验：2D `[BLOCK, R]` int64
  通路按 8-10 份临时量估 UB；int64 换 int32 是第一杠杆（c1 已换）。
- **预防**：目前靠编译期报错兜底；M5 调优时再系统化（tile 扫描）。

## CASE-06 oracle（MoonEP 参考实现）的 meta 对齐约束：3·E·R 须被 4 整除

- **现象**：R=1、epn=2（E=2）时 oracle `build_world` 断言
  `broadcast_elems (6) must be divisible by 4` 失败。
- **根因**：参考实现 api.py 的 meta 区对齐要求；R=1 且 E 小时容易撞上。
- **解决**：测试构造 R=1 用例时 epn 取 4（3·4·1=12 ✓）。
- **预防**：`test_planning_bitexact.py` 头部注释 + `test_oracle_selftest.py`
  同款处理。

## CASE-07 KernelCase 的 E/NvS 是方法不是属性 + mp.spawn 不能 pickle 闭包

- **现象**：①`case.NvS_capacity` AttributeError（不存在该属性，CAP 即
  `case.N`，`E/NvS` 是方法 `E(R)/NvS(R)`）；②dist worker 报
  `Can't pickle local object`。
- **解决**：①按方法调用；②worker 用模块级函数 + `functools.partial`
  传参（partial 的模块级函数可 pickle）。
- **预防**：两处测试文件即样例。

## CASE-08 src 生产代码不能 import tests/ —— oracle vendor 移位

- **现象**：v1 planning 宿主路径要直接调参考函数 `_phase_b_tables`，
  但它按原决定放在 `tests/_moonep_oracle/`，src 反向依赖 tests 不成立。
- **解决**：`git mv` 到 `src/mega_moe/moonep_ref/`（先例：经典路径的
  torch 参考实现就在 `ops/_torch_forward.py`）。sys.path bootstrap 保持
  原样（`moonep_ref/__init__.py` 注入自身目录使 `import moonep` 可达），
  tests 侧 import 同步替换。
- **预防**：vendor 同步流程不变（cp + `diff -rq` + 更新 commit 注记）。

## CASE-09 aclshmem_init 有堆大小下限（64MB init 失败）

- **现象**：两 rank 同时报 `aclshmem_init failed`——workspace 估算堆仅
  ~64MB 时。
- **解决**：堆下限取 256MB（smoke 验证过）；`MoonepWorkspace.required_bytes`
  仅作下界估算，session sizing 用 `max(估算×2, 256MB)`。
- **预防**：`test_prefetch_bitexact.py` 的 heap 计算行。

## CASE-10 cu/zfr 的形状是 [E+B] 不是 [epn+B]

- **现象**：M2 测试初版把 cu_seqlens/zero_fill_ranges 按 `topo.seg=epn+B`
  分配——planning 输出实际是**全局专家段** [E+B]（压缩段视图才是 epn+B，
  由 build_moonep_segment_meta 在消费侧派生）。
- **预防**：M2 测试的 outs 契约注释；M3 的 segment meta 构建器要显式做
  E+B → epn+B 的压缩。

---

## 里程碑状态（更新）

| # | 内容 | 状态 |
|---|---|---|
| M0 | oracle vendor + Step 0 smoke | ✅ aa93b4b / ee99e2f |
| M1 | Triton planning bit-exact（rand/dup/R=1） | ✅ dbed413 |
| M2 | MoonepWorkspace（8 张对称张量定序）+ prefetch push kernel 对拍 | ✅ 见本次提交 |
| M3 | dispatch + segment/send meta + FC1 GEMM 接线 | 待启动 |

## CASE-11 send_meta 的段压缩必须按【目的 rank】，不是本 rank

- **现象**：dispatch 融合 kernel aicore 超时（消费侧 GEMM `dl.wait` 等不到
  信号）。
- **定位**：二分法——push-only 变体（去 Cube 半边）VM/权重全对 ⇒ 推半边
  无辜；离线复算表数据全自洽 ⇒ 表值无辜；剩下信号槽键。`build_moonep_
  send_meta` 里 `_compress_seg(g, rank, ...)` 用了**本 rank**——段是目的
  rank 的本地段，必须 `g − dr·epn`；且段查表的 **cu 也必须是目的 rank
  的**（cu 每 rank 不同）。
- **解决**：planning 宿主把全组表经 `outs["_tbl"]` 捎带（cu_all），
  send_meta 按目的 rank 分组查表+压缩。
- **预防**：`test_dispatch_bitexact.py`（任何信号不匹配都会以超时显形）。

## CASE-12 910B Cube 不支持 16×16×16 的小块 bf16 dot（fixp 崩/挂）

- **现象**：BLOCK_M/N/K=16 时 kernel aicore 超时，设备 dump 报
  **fixp**（定点单元）错误；push-only（纯 Vector）同尺寸正常。
- **根因**：生产路径 classic GEMM 从不跑 16 方块（tail 最低收缩到 32/64），
  小尺寸 cube dot 在 910B 上触发硬件异常。
- **解决**：测试规格提到生产量级（H=256, 2F=256, BLOCK=128）后一次通过。
  **结论：moonep 的 GEMM 调用面 block 下限 128**（与 classic 一致）。
- **预防**：launcher 文档注释 + 本条案例；后续小规格功能用例只测
  planning/prefetch/push，不测 Cube GEMM。

## CASE-13 期望算式的转置方向（测试侧）

fc1 期望 = `vm @ gu[s]`（W 物理 [H,2F]，b[k,n]=phys[k,n]），不是
`@ gu[s].T`——以 classic b_ptrs 步长公式为准推导，不要凭直觉。

---

## 里程碑状态（更新）

| # | 内容 | 状态 |
|---|---|---|
| M0-M2 | oracle/smoke/planning/prefetch | ✅ |
| M3 | zero_fill + segment/send/recv 元数据 + dispatch 融合 kernel（复用 classic FC1 GEMM，Seg 替换 EPR）；VM/权重逐位 + fc1 数值全绿 | ✅ 见本次提交 |
| M4 | combine_push + topk_reduce + MoonepForward 组装 | 待启动 |

## CASE-14 设备流资源耗尽（EE1023）——挂死强杀的驱动级残留

- **现象**：所有卡 `SetDevice` 报 `Too many streams are created`；无进程
  持有设备 fd；容器内 `npu-smi set -t reset` 不可用。
- **根因**：aicore 超时被强杀的 run 泄漏驱动级流上下文，不随进程回收。
- **解决**：宿主机侧复位 NPU 后恢复。**预防：控制并发 pytest 进程数；
  挂死后先查设备状态再继续跑。**

## CASE-15 融合 dispatch（push∥GEMM+信号）对 self-putmem 行非确定性脏读

- **现象**：fc1 段级数值偶发错（同 seed 不同结果，M3 测试 6 跑 5 挂；
  提交时通过属运气）。行级 dump 锁定：**坏行全部落在 dst==self 的行块**
  （fence+signal 对自 putmem→cube 读的排序偶发失效；classic 同构生产
  无恙，深层根因未明——待与 triton-ascend 侧对齐）。
- **定位链**：真实中间量分阶段核查（VM 对/GEMM 错→排除散布）→ M3 独立
  回测复现 flaky → 行级 dump 的 per_src 分桶锁定自源块。
- **解决**：dispatch 改两段式 `push+fence+barrier → 无信号 plain GEMM`
  （与 combine 的 FC2 模式同款）。M3 6/6 绿、L2 3/3 绿且逐位确定。
  **融合流水（信号协议）留 M5 复原**（复原时优先查 self-putmem 语义）。

---

## 里程碑状态（更新）

| # | 内容 | 状态 |
|---|---|---|
| M0-M3 | oracle/smoke/planning/prefetch/dispatch | ✅（M3 两段式修复后 6/6 稳定） |
| M4a | **前向端到端**：MoonepForward 全链 + L2 对拍 | ✅ 3/3 绿且逐位确定（rank0 0.0165 / rank1 0.0136） |
| M4b | 后向（B-1 dy散布+dgrad → B-4 autograd 封装） | 设计定稿 docs/moonep_backward_plan.md，实施启动 |
| M5 | 性能基准 + 调优（含融合 dispatch 复原） | 待启动 |

## CASE-16 dw 归集：offv 空间是【每个源一份 [0,N)】，按 VM 属主遍历会叠槽

- **现象**：dw 两 rank 结果完全相同且大错（8.0 量级）；dump 发现
  rank0 的 dw 值 == rank1 的。
- **根因**：dscale 在【接收方】，条目的 offv 是**其源 rank** 的 [0,N)
  空间——遍历 (si, dscale) 对时不按 `si//NvS` 过滤源，R 份 offv 叠进
  同一 N 槽、后写覆盖前写（结果恰为最后一个源的视角，两 rank 同构
  同错）。
- **解决**：过滤 `si//NvS == 本 rank` 再填充。
- **预防**：dw 对拍（值对 = 映射错；值近似 = dtype 量化）。

## CASE-17 反向 dgrad 的权重视图方向与前向相反（CASE-13 反向版）

FC1 前向 b[k,n]=phys[k,n] 用 `gate_up.transpose`；FC1 **dgrad**
gy=dAB@gu_phys.T 则须传 `gate_up` **原样**（stride_out=2F、red=1）。
FC2 dgrad 同理用 `down.transpose`。纪律不变：以 b_ptrs 步长公式推导。

## CASE-18（预留）B-3 复跑挂：连续两次 pytest 间 aclshmem/hccl 资源残留

planning_b 全量回归（5 个 dist 用例单进程连跑）未复现；纪律保持：单文件
串行、挂死后先查 npu-smi 再继续。

## CASE-19 bisheng convert-hfusion-to-hivm 形状相关 legalization 失败（planning 单 kernel 化的边界）

- **现象**：`moonep_plan_fused`（A~D 单 kernel）在 E=4/R=2 编译并逐位
  通过；E8/R2、E8/R4、E16/R4、E4/R1、E128/R8、E256/R8 全部编译失败
  （`ConvertLinalgIRToBinary ... Failed to run BiShengHIR pipeline`；
  `--mlir-print-ir-after-failure` 定位到 convert-hfusion-to-hivm 的
  "failed to legalize operation 'linalg.generic'"）。
- **排除**：环境问题——tutorials 01-ascend-allgather-gemm 2 rank 全绿；
  各 phase 单独/两两组合编译全过；onehot/cumsum 最小模式单卡全过。
  CANN 8.5.0 自带 bishengir-compile 太旧（不认识 hivm.hir.custom）无法
  交叉验证；实际生效的是 PATH 里本地构建的 AscendNPU-IR（无 TRITON_
  NPU_COMPILER_PATH 时 wheel 不带 bishengir → which 解析）。
- **触发规律**：多段共存效应，无单一维度。最小复现 = {B.0+B.1+C.1}
  三段（任意两段 OK）；减法二分去掉 B.0 或 B.1 任一即过；A+B 同 kernel
  在 E8R2B3/E16R4 挂而各自单独过；B.4 的 indexed gather/scatter 在
  R=1 单段即挂。
- **已试无效**：C.1 i32 化、BLOCK_HIST/BLOCK_E 全扫描、phase 重排、
  tl.debug_barrier 切 fusion、prevec/autovec/simt_only/direct-hivm-
  lowering 编译旋钮、gather 尾 barrier vs tables 头 barrier 位置。
- **生效规避（当前三-kernel 形态的全部）**：拆 gather/tables/dst 三
  kernel；B.0/alloc_cs 逐 rank 行 1D 累加替代 2D tl.cumsum；B.4 双路径
  （E≤64 无 indexed 的分块 onehot 选择，大 E 用 indexed——E128/E256
  生产形状 indexed 路径编译通过）；C.1 onehot/cumsum i32；B.1/B.2 贪心
  循环 `if R > 1` 编译期剔除（R==1 不支持，实际使用无此形态）。
- **定位/复现工具**：失败 cache 目录的 `.bcmlir` 直接喂 bishengir-
  compile（flags 抄 `debug=True` warmup 打印的 cmd_list）+ `--mlir-
  print-ir-after-failure`；最小复现三段 kernel 在 /tmp/dbg_core.py
  （bring-up 会话产物，丢失可按本文重写）。详见
  memory:bisheng-convert-hfusion-shape-bug。
- **预防**：编译器修复前不改回单 kernel；golden 用例保留对 fused 的
  逐位对拍锚点，修复后合回时零漂移。

## 里程碑状态（更新）

| # | 内容 | 状态 |
|---|---|---|
| M0-M4a/B-1/B-2 | 同前（oracle/planning/prefetch/dispatch/前向 e2e/反向 dx/dw） | ✅ |
| **planning v2** | 单 kernel 化：kernel 内 tpe allgather（putmem+尾 barrier）+ B 七表冗余自算 + C.2 src_info symm_at 发布 + dedup；宿主 B 表/oracle 生产依赖/topk·tpe HCCL allgather/宿主 src_info 重建全部删除；rank0/1 分工与 order0 体系移除 | ✅ planning_b 分支：三-kernel 形态（CASE-19 规避）全绿（golden+单kernel锚点/ties4/b3tie/odd/determinism + planning/prefetch/dispatch/forward/backward 回归）；单 kernel 版保留为锚点待编译器修复 |
| B-3 | wgrad+槽归并 | 代码完成未提交（dev_moonep 工作区，planning_b 开发期间 stash） |
