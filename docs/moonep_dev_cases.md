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

---

## 里程碑状态

| # | 内容 | 状态 |
|---|---|---|
| M0 | oracle vendor（6/6 自测绿）+ Step 0 smoke（4/4 绿，产出 CASE-01..04） | ✅ aa93b4b / ee99e2f |
| M1 | Triton planning：C.1 计数排序 + putmem order0 + C.2 + dedup kernel；bit-exact rand/dup/R=1 全绿 | ✅ dbed413 |
| M2 | workspace + prefetch | 进行中 |
| M3-M5 | dispatch/GEMM 接线、forward 组装、基准 | 待启动 |
