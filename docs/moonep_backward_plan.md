# MoonEP 后向接入设计（moonep_backward，v1）

> 前置：M0-M4a（forward 链，dev_moonep）。本文件是 M4b 的实施蓝图：
> 反向五步算子、前向复用清单、saved 契约、槽梯度归并、对拍方案、
> 实施顺序。行号指 dev_moonep 分支当前代码。

## 0. 核心洞察：反向几乎全是前向部件的镜像

MoonEP 的 plan（dst/cu/etc/src_info + send/segment 元数据）描述的是
**布局映射**，与数据流向无关——反向只是把搬运方向倒过来：

| 前向部件 | 反向用途 | 复用方式 |
|---|---|---|
| `send 表`（offv→(dr,loff)） | dy 的散布（token→接收行） | **同一张表**，kernel 同构 |
| `src_info`（接收行→(sr,offv)） | gy 的推回（接收行→token） | **kernel 原样换指针** |
| `segment meta`（Seg 段表） | 全部四个 GEMM 的分段 | 同一张表 |
| `down/gate_up 对称表`（含槽） | dgrad 用；wgrad 的输出行序 | 同一张表 |
| `combine_push + topk_reduce` | dx 的汇聚 | **零改动复用**（换指针） |

因此反向的"通信"零新设计；新写的只有段 GEMM 的 dgrad/wgrad 变体
与槽梯度归并。

## 1. 五步算子（对照 classic backward.py 的 5-op 结构）

### B1 dispatch_bwd + FC2 dgrad（融合）
- **dy 散布**：token 的 K 个条目各需 dy[t]（每条目独立反传）。条目行
  映射 = forward send 表：`dy_buf[loff] = dy[offv//K]`。kernel 与
  `moonep_dispatch` 的 push 半边同构（逐行 putmem，无需信号——下游
  GEMM 前有 barrier）。
- **FC2 dgrad**：`gact[rows,F] = grad_z[rows,H] @ dn_e`——段 GEMM，
  结构 = `moonep_fc2_gemm` 换权重方向（dn 原样 [Seg,H,F]，out=Σ_n
  grad_z[n]·dn[n,f]，即 b[k,n]=phys[f? …] 按步长约定推导，同
  CASE-13 纪律：以 b_ptrs 步长公式为准，勿凭直觉）。
- 输出 `gact [rows_pad, F]`（∂L/∂act，含 w 已乘前的量——注意
  forward 的 w 在 swiglu 乘入，见 B2）。

### B2 swiglu_bwd（本地）
- ∂L/∂(g,u)[r] = gact[r] ⊙ [w·silu'(g)·u, w·silu(g)]，需要保存的
  fc1_out（gate/up）与 rw_recv。
- **直接复用** `kernels/swiglu_bwd.py::swiglu_bwd_triton`——行并行，
  入参换 `(gact, fc1_out, rw_recv)`，零改动。

### B3 FC2 wgrad（段 GEMM）+ 槽归并
- `d_dn_seg = grad_z_segᵀ @ act_seg`（[H,rows]×[rows,F]）——转置
  a-tile 分组 GEMM：复用 `kernels/transposed_grouped_gemm.py` /
  `_grouped_wgrad_npu`（ops/backward.py:75），喂 Seg 段表。act 用
  forward 保存（或由 fc1_out 重算，v1 保存）。
- **槽归并见 B5**。

### B4 FC1 dgrad + 反向汇聚（dx / dw）
- **FC1 dgrad**：`gy[rows,H] = gg_u[rows,2F] @ gu_e`——段 GEMM（同
  B1 结构）。
- **推回 + 求和 → dx**：gy 行按 src_info putmem 回源 rank 的
  `grad_buf[offv]`，再 per-token K 求和——**`moonep_combine_push` +
  `moonep_topk_reduce` 原样复用**（换 fc2_ptr→gy、combine_buf→
  grad_buf、output→dx）。
- **dw[t,k]**：∂L/∂w[r] = Σ_f gact[r,f]·(silu(g)·u)[r,f]（本地行约
  简，标量）；4B putmem 推回 offv 行 + K 求和（topk_reduce 的标量
  版或直接宿主）。注意 gact 是 **w 相乘前**的 ∂L/∂act（B1 里不乘
  w；w 在 B2 里乘）——这与 classic 的 gate-grad 打包同构。

### B5 槽梯度归并（reduce_grad 等价，v1 宿主实现）
- B3/B4 的 wgrad 写进对称梯度表 `[Seg,...]`：home 行直接正确；**槽行
  是副本处梯度**，必须归并回属主：`d_home[e] += Σ_slot d_slot`。
- v1 宿主归并（量极小：B 行/rank）：all_gather 槽梯度 → 按 etc 归属
  scatter_add 到 owner home 行 → 写回；fp32 逐位（参考实现
  grad_reduce 语义：home 归约 + 槽清零——**每步清零防跨步累积**）。
- v2 设备化方向：owner 侧 getmem 拉取 or 信号两阶段（参考 step4
  DIRECT_PULL 经验：getmem 只能 Vector，无 Cube 混用）。

### dup 条目（v1 语义对称）
forward v1 对负 dst 照发 payload ⇒ 反向 K 份贡献独立、无特殊处理。
若 v2 前向启用 payload 抑制 + 接收端扇出，反向需按 dup_groups 把
primary 行梯度扇出给 dup 条目——**复杂度前置到前向 v2 时一并设计**。

## 2. saved 契约（MoonepForward 需 stash 的反向状态）

```python
op.forward(..., return_saved=False)   # v1 增参；saved 由 op 持有
saved = {
    "plan": outs 全套（dst/cu/etc/zfr/stats/src_info/_tbl/_dst_all）,
    "fc1_out":  [rows_pad, 2F] bf16,      # B2 需要 gate/up
    "act":      [rows_pad, F] bf16,       # B3 wgrad 需要（或重算）
    "rw_recv":  视图（B2 需要；注意 single-in-flight 期间不得复用 ws）,
    "rows_pad": int, "epoch_state": …,
}
```
单 in-flight 纪律与经典一致：反向完成前不得发起下一个 forward
（workspace 的 vm/combine_buf 等被反向沿用）。

## 3. 梯度对拍方案（三档）

1. **autograd oracle（主力）**：宿主 fp32 参照（`test_moonep_forward
   ::_reference`）包 `torch.autograd.Function`，对 x/w/权重 求 autograd
   梯度作 oracle——省手写反向，数学上严格；
2. **容差**：`GRAD_*`（2e-2 / 1e-2，tests/_numeric.py），布局类 int
   表逐位；
3. **槽归并专项**：构造同一专家进多个 rank 槽（E=4,B=1,R=2 skew 即
   天然出现），断言 home 梯度 == 各副本处梯度之和（fp32 逐位）。

## 4. 实施顺序（4 个可独立验收的粒度）

| # | 内容 | 验收 |
|---|---|---|
| B-1 | dy 散布 kernel + FC2 dgrad + swiglu_bwd 接线 | gact/gg_u vs autograd 中间量 |
| B-2 | FC1 dgrad + combine_push/topk_reduce 复用 + dw | dx/dw vs autograd（全绿即"反向通"） |
| B-3 | 两个 wgrad（段 GEMM）+ 槽归并 | 权重梯度 vs autograd + 槽归并专项 |
| B-4 | autograd.Function 封装（MoonepBackwardFunction）+ 套件 + 与经典路径 perf 对照 | test_moonep_backward 全绿 |

## 5. 已知风险清单

- **b_ptrs 步长方向**：dgrad GEMM 的 b 矩阵方向与 forward 相反
  （CASE-13 的教训普适——每次都以步长公式推导）；
- **wgrad 转置 a-tile**：`transposed_grouped_gemm.py:16-23` 的坑
  （预转置连续 [N,M] 或 BLOCK_M≤64）；
- **两流重叠**：v1 全串行；B1 散布/B4 推回 与 GEMM 的重叠直接搬
  classic 的 per-tile epoch 信号协议（dispatch_fc2_bwd.py 同款）；
- **槽归并的跨步清零**：忘记清零 = 梯度跨步累积（静默错误）；
- **设备流资源**：本轮开发遭遇驱动级流表耗尽（aicore 挂死强杀残
  留，容器内不可复位）——**长跑测试须控制并发 pytest 进程数**，
  挂死后及时确认设备状态再继续（CASE-14 候选）。
