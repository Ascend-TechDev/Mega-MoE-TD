# NPU 训练内存泄漏定位与修复 playbook

—— megamoe 整网（kimi k3 mock, 8 卡, FSDP2）泄漏案例复盘

> 案例：megamoe dispatcher 路径每迭代 alloc 精确 +3.8 GiB → 修复 → 仍有 +0.72 GiB/iter → 再修复到 +0.39 GiB/iter（剩余为框架侧独立泄漏）。
> 相关提交：算子仓 `c02f4f1`（workspace + fc2 staging）、框架仓 `6d272a96`（stride view）。
> 本文重点是**可复用的方法**，案例细节只作为每一步的例证。

---

## 0. 总体思路（先记住这三句话）

1. **先量化，再定位**：把"感觉在涨"变成"每迭代 +X GB 的直线"，斜率就是泄漏的大小，形状（线性/阶跃/锯齿）就是泄漏的类型。
2. **按 storage 找，不按 tensor 对象找**：同一块显存上可以挂几十个视图对象，对象计数会把"视图堆积"误报成内存增长。
3. **找到持有链 ≠ 找到修法**：持有链告诉你"谁不放"，但修法要结合对象的生命周期语义（本案例中"直接放掉"是错的，见 §4 安全销）。

---

## 1. 第一阶段：确认并量化泄漏

### 1.1 每迭代打印三个数

在训练循环里（每 iteration 结束、下 iteration 开始前）打：

```python
torch.npu.memory_allocated()    # 活跃显存（torch 认为还被引用的）
torch.npu.memory_reserved()     # 缓存池总保留（≥ allocated）
torch.npu.mem_get_info()        # (free, total) 设备级真实余量
```

得到形如 `iter N | alloc 17.8G reserved 46.5G free 8.5G` 的曲线。

### 1.2 读曲线

| 现象 | 结论 |
| --- | --- |
| alloc 每 iter 恒定 +X（线性爬升） | 有对象跨迭代存活，被某处引用钉住 → 走引用链定位（§3） |
| alloc 平、reserved 涨 | 缓存池碎片/不还 OS，通常不是泄漏 |
| alloc 锯齿且峰值爬升 | 峰值泄漏（同时存活的重叠变宽），看 `max_memory_allocated` |
| free 单调下降而 alloc 平 | torch 之外的显存占用（CANN workspace / 对称堆） |

**关键校验**：`allocated == active` 且增长与迭代严格同步 → 是 Python/pytorch 侧活引用，gc 一定能看到（§3 的方法成立）。
本案例：17.8 → 18.6 → 19.3 → 20.0 → 20.7 → 21.5，精确 +0.72 GiB/iter，线性无锯齿。

> 细节：泄漏量对齐到张量尺寸可以"猜"出泄漏对象。0.72 GiB/iter = 168 MiB × 4 层 = 恰好一个 `(4, 6144, 3584)` bf16 张量/层/迭代。这种"尺寸指纹"贯穿全程都在用。

---

## 2. 第二阶段：造一把好用的"显存解剖刀"

这是本案例最重要的可复用产出：`_dump_large_npu_tensors`（框架仓
`mindspeed_mm/fsdp/train/train_engine.py`，`MSMM_MEM_TRACE=1` 触发）。
分四层能力，**每层排除一类假设**：

### 2.1 storage 级分组（而不是对象级）

```python
objs = tuple(gc.get_objects())          # tuple 快照，见 §5.1 的坑
stores = {}                             # data_ptr -> rec
for obj in objs:
    if not isinstance(obj, torch.Tensor) or obj.device.type != "npu":
        continue
    st = obj.untyped_storage()
    if st.nbytes() < MIN_MB:            # 只看大块，压噪声
        continue
    rec = stores.get(st.data_ptr())
    if rec is None:                     # 每块显存只记一次
        rec = stores[st.data_ptr()] = dict(
            mb=st.nbytes() / 2**20, shape=tuple(obj.shape), dtype=str(obj.dtype),
            kinds=<obj 的引用链描述>, views=0)
    rec["views"] += 1                   # 顺带统计视图堆积（对象级信息不丢）
```

对比相邻两次 dump 的 `data_ptr` 集合：**新增指针 = 新分配的显存**；指针不变但 `views` 涨 = 只是视图堆积（无害）。
本案例第二轮靠这一步排除了假增长：`(4,6144,3584)` 组对象数 12→24 翻倍，但 storage 指针零新增——涨的全是常驻权重的视图。

### 2.2 引用链（谁持有这块显存）

```python
refs = gc.get_referrers(obj)
for r in refs:
    d = describe(r)                 # dict 打印前 6 个 key，list 打印长度和元素类型
    if 看起来是 saved 字典:          # 用特征 key 识别（如 "expert_counts"/"split_size"）
        for u in gc.get_referrers(r):   # 再向上追一跳——持有者才是真凶
            d += " <- " + describe(u)
```

识别特征：dump 里出现的 `dict('op','saved','peer_mem','state')` 一眼就是自定义 autograd Function 的 `ctx.__dict__`。
**必须向上多追一跳**：直接持有者往往只是中间结构（saved 字典、cache 字典），根因在它的持有者。

### 2.3 python 可见性 gap（泄漏是否 gc 可见）

```python
py_bytes = sum({st.data_ptr(): st.nbytes() for 所有 npu tensor 的 storage}.values())
print(alloc - py_bytes)     # gap
```

- gap ≈ 恒定小负数（本案例 -1.92 GiB = ACLSHMEM 对称堆，不走 caching allocator）且 python 可见总量与 alloc **同斜率**增长 → 泄漏全部 gc 可见，继续 §2.1；
- gap 随迭代增大 → 有 gc 看不见的持有者（C++ autograd 图、事件未决块），改用 §2.4。

### 2.4 memory_snapshot 交叉验证（给不可见显存定尺寸）

```python
snap = torch.npu.memory_snapshot()
# 遍历 seg.blocks，state == "active" 且 address 不落在任何 python storage 的
# [ptr, ptr+nbytes) 内的块 → 按 size 直方图，即"gc 不可见显存"的尺寸指纹
```

### 2.5 小块直方图

大块（>100MB）稳定时，把阈值降到 0.5MB 并按"精确 nbytes × 数量 × 一条持有链"聚合 top10。
本案例最后 0.72 GiB/iter 就是靠这个找到的：`84MB x N (4,3584,3072) held by dict('M','N','K','E','fc2',...)`。

---

## 3. 第三阶段：三轮定位实录（每轮一个教训）

### 第一轮：对象级引用链 → ctx 钉住 saved

- 持有链：`激活 ← dict(saved) ← list ← dict('op','saved',...)`（autograd ctx）。
- 教训 A（工具坑）：第一版 dump 里几乎所有张量的持有者都是 `list(len≈78万)`——那是 `gc.get_objects()` **自己的返回值**。必须先 `tuple(...)` 快照并在追链时排除自身与 dump frame（详见 §5.1）。
- 结论：`MegaMoEFunction` 把 `ctx.saved` 等挂在 ctx 属性上（不走 `save_for_backward`），集成路径上 ctx 被 FSDP2/引擎侧结构钉住跨迭代存活 → 每迭代钉住 4 个（每 MoE 层 1 个 backward 的）saved 字典，内含 ~955MB 激活/层。

### 第二轮：修"释放"→ 三连败 → 认识到"泄漏是安全销"

按持有链直觉修法是"backward 结束把 ctx.saved 清掉"，试了三种（置 None / `record_stream` 后置 None / 延迟一迭代释放）——**全部在 iter1-3 触发 ffts 自旋、vector core execution timed out**；而"泄漏版"从不卡。

对照实验（3/3 vs 3/3）定性：算子有自建流和 **ffts free-flight 任务**（寿命跨迭代、不受 torch stream 语义管辖）。saved 里的块一旦回池，会被下一迭代的分配拿去复用，与仍在飞的任务竞争 → 信号链数据污染 → 自旋超时。**泄漏一直在充当隐式安全销**：块被钉住所以从不复用。

> 这是全文最重要的教训：**持有链给你"谁不放"，不给你"能不能放"**。在异步执行/自建流/远端内存的环境里，"正确释放"和"过早释放"的表现分别是"泄漏"和"卡死"，要按对象的真实生命周期重新设计，而不是把引用掐断。

### 第三轮：storage 级 + 尺寸指纹 → 两个真凶

- **泄漏 A（+336MB/iter，我方）**：框架 dispatcher 每步做 `down.transpose(1,2).contiguous()`（~84MB/层，因为算子契约要求 contiguous），该副本进 `saved["fc2"]` 被钉住的 ctx 持有。尺寸指纹 `(4,3584,3072)` × 4 层与斜率严格吻合，持有链 `dict('M','N','K','E','fc2','total_send')`（dispatch_fc2_bwd 的 cache 字典）指认。
- **泄漏 B（+390MB/iter，框架侧）**：`SwapTensor`（GDN/KDA/flash-attn 的 skip_recompute offload，`OffloadManager`）逐迭代增长，形状是注意力中间态。与 dispatcher 无关（shmem 基线同样存在），非集成回归，移交框架侧。

---

## 4. 修复模式：常驻 workspace + 写透 staging

两条设计原则，都来自第二轮的教训（块不可回池竞争 → 干脆永不回池）：

### 4.1 大激活 → 算子常驻 buffer 的 `[:M]` 切片

```python
# forward.py: _get_saved_workspace —— 首次按接收容量分配，之后只发切片
self._saved_ws = {
    "fc1_output":         torch.empty(max_recv, 2*ffn, ...),
    "recv_hidden_sorted": torch.empty(max_recv, hidden, ...),
    ...
}
# 每步：saved["fc1_output"] = ws["fc1_output"][:routing_plan.num_received_routes]
```

- ctx 钉住的只是**视图** → 零增量泄漏；
- 地址固定 → 不进缓存池 → 与 free-flight 任务无竞争可能；
- `MOE_SAVED_WORKSPACE=0` 保留旧行为作回退开关。

### 4.2 每步都要变的权重 → 常驻 buffer + `copy_` 写透

权重被优化器每步原地更新，不能缓存一份旧的；也不能让框架每步 new 一份新的（那就是泄漏 A）：

```python
# 宿主：传 stride view，不再 .contiguous()
down = down.transpose(1, 2)

# 算子：_materialize_down_weight —— 常驻 buffer，每步 copy_ 刷新
if not down_weight.is_contiguous():
    if self._fc2_ws is None or 形状变了:
        self._fc2_ws = torch.empty(down_weight.shape, ...)
    self._fc2_ws.copy_(down_weight)     # 写透：优化器更新每步流入
    return self._fc2_ws
```

配套：算子两处 fc2 GEMM 本来就显式传 stride（能直接吃 view），forward 校验放宽接受 strided `[E,H,F]`。

### 4.3 为什么不选别的方案

| 方案 | 否决原因 |
| --- | --- |
| backward 尾部 `ctx.saved = None` / `record_stream` / 延迟释放 | 三种全部实测触发 ffts 自旋卡死（§3 第二轮） |
| 跨迭代缓存权重的 contiguous 副本 | 权重每步被优化器更新，缓存会算错数 |
| 算子直接吃 strided view（不 staging） | 数值正确，但 K 维非单位 stride 的 GEMM 读带宽受损；staging 只花一次 84MB copy（框架原本就在付这个代价） |

### 4.4 验证清单（修完必做）

1. **斜率**：alloc 曲线从 +0.72 → +0.39 GiB/iter（剩余 = 已知的框架侧泄漏 B）；
2. **数值**：同 seed 对照，loss 与修复前 **bit 级一致**（1.300527）——确认 staging 只是换地址不改数；
3. **稳定性**：10/10 iters 全绿无 OOM 无卡死（此前泄漏版 iter5 OOM、释放版 iter1-3 卡死）;
4. **回退开关**：`MOE_SAVED_WORKSPACE=0` 回到旧行为，用于二分定位回归。

---

## 5. 踩坑清单（都是实测踩过的）

### 5.1 `gc.get_referrers` 的自引用假象

`gc.get_objects()` 返回的 list 本身引用着所有对象，会作为一切对象的 referrer 出现（表现为 `list(len=78万, els=[cell,function,list,tuple])`）。
**修法**：`objs = tuple(gc.get_objects())` 快照；追链时 `excl = {id(objs), id(self)}` 排除自身；跳过 dump 自己的 frame（按 `co_name` 含 `dump_large`/`listcomp` 判别）。

### 5.2 对象计数 ≠ 显存增长

同一 storage 的视图（transpose/slice/chunk）对象会重复计数。**一切结论以 storage `data_ptr` 为准**，`views` 计数只作辅助信息。

### 5.3 负 gap 不是 bug

python 可见 storage 总和可以**大于** `memory_allocated`——对称堆（ACLSHMEM）、事件暂挂块等不走 caching allocator 的内存都在 python 侧可见但不在 alloc 里。gap 恒定 = 无信息；gap 增长 = 去查 §2.4。

### 5.4 "泄漏"与"卡死"可能是同一个问题的两个面

见 §3 第二轮。在动任何"释放"逻辑前，先问：这块内存的消费方是否有 torch stream 语义之外的生命周期（ffts/自建流/远端 RMA）？

### 5.5 尺寸指纹是最快的定位手段

泄漏斜率 ÷ 层数 ÷ backward 次数 → 单个张量的字节数 → 反查形状 → 直接 grep 代码里产生该形状的分配。配合 dump 里 storage 的 nbytes 分组，两步就能从"涨了"走到"这行代码"。

### 5.6 运维小项

- NPU 长跑必挂 watchdog：`run_cmd_watchdog.sh <name> <log> <cap> <command>`，idle 600s 自动 `py-spy dump` 后 kill——卡死时的栈是定位的唯一入口（本案例用它证实 serial 路径卡在框架 dataloader 而非算子）；
- 多轮实验日志按名字落盘（`/tmp/moonep_regress/*.log`），对照实验（泄漏版 vs 释放版 3/3 vs 3/3）是定性的关键证据。

---

## 6. 下次排查 checklist

1. [ ] 每迭代打印 alloc/reserved/free，确认斜率与形状（§1）
2. [ ] 斜率 ÷ 层数 ÷ microbatch 数 → 单张量尺寸指纹（§5.5）
3. [ ] storage 级 dump + 相邻两次 diff 指针集（§2.1）
4. [ ] 持有链向上追到根（ctx / module / dict / frame），排除 gc 假象（§2.2, §5.1）
5. [ ] gap 分析：泄漏是否 gc 可见；不可见走 memory_snapshot（§2.3, §2.4）
6. [ ] 大块稳定就降阈值看小块直方图（§2.5）
7. [ ] 设计修法前先回答：对象的真实生命周期？有无 torch stream 之外的使用方？（§3 第二轮）
8. [ ] 修复模式选择：常驻切片 / 写透 staging / 传 view（§4）
9. [ ] 验证：斜率 + bit 级数值 + N iters 全绿 + 回退开关（§4.4）
