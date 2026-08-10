# bigop 融入 Mega-MoE-TD —— 第二套 MoE 反向 golden 接入方案

> **状态**:Step 1 ✅ 已完成;Step 2/3 ⬜ 待实施(**分阶段替代** torch 反向 golden,见下)
> **日期**:2026-08-07(初稿)/ 2026-08-08(按进展修正 + 策略调整为"分阶段替代")
> **用途**:把 bigop 作为 `3rdparty/` git submodule 接入 Mega-MoE-TD,复用其 compute 原语,
> 构建**独立的第二套 MoE 反向 golden**,**最终替代**现有 torch-loop golden 作为 Triton
> mega-kernel 的对比基准,形成更可靠的交叉验证。
> **实施顺序**:Step 1(打包安装,✅)→ Step 2(`bigop_ref.py`)→ Step 3(分阶段接入/替代)。

---

## 进展(截至 2026-08-10)

| 步骤 | 状态 | 证据 |
|---|---|---|
| **Step 1** 打包安装 | ✅ 完成 | bigop `68ca4cb`(加 `pyproject.toml` + `.gitignore`);mega_moe `.gitmodules` pin `68ca4cb`、`scripts/install_3rdparty.sh`、README 小节;commits `18419b9`/`65ea306`;`python -c "import bigop"` OK(editable → `3rdparty/bigop/bigop.py`)。 |
| **Step 2** `bigop_ref.py` | ✅ 完成 (2026-08-10) | `src/mega_moe/_goldens/bigop_ref.py` 已落地(compute 用 bigop,A2A 复用);2 卡 gold↔big 全绿(小 + Kimi-like shape,rel~5e-3);附独立 `__main__` 交叉验证入口(不依赖 Triton)。 |
| **Step 3** 接入/替代 | ✅ Phase1/2 完成 (2026-08-10) | Phase1:test_moe_backward.py 三列 tri/gold/big 全绿;**Phase2:tri 基准切 bigop(`[baseline=bigop]`),torch golden 降为非 gate 旁证,4 config 验证全绿**;bench_backward.py 加 bigop 计时列。 |

`<BIGOP_PKG>` = **`68ca4cb`**(bigop HEAD,`0484e96`+1 的打包提交)。

## 策略调整:从"additive 三列"到"分阶段替代"(本次修正核心)

初稿策略是"additive 三列、永不替换"。经核实后调整为**分阶段替代**——bigop 的独立 lineage
(UT/ST 与 Megatron bit-level 对齐)正是比手写 torch golden 更可信的基准,值得**最终替代**而非
永久并列。但**不裸替**:先用 torch golden 在仓内**验证 bigop 自身正确**,再提升。

**为什么替代是干净且值得的(已核实)**:
- `moe_backward_torch` **仅用于测试与基准**:`tests/layer/test_moe_backward.py`、
  `benchmark/layer/bench_backward.py:161`。生产 `src/mega_moe/ops/backward.py` 只 import
  共享前向 `moe_forward`,**不依赖反向 golden** → 替代不影响生产。
- torch golden 与生产前向 `_torch_forward_for_backward.py` **同源同维护**(其 docstring 自述
  "the golden and the production path share one source of truth"),正是 bigop 要消除的污染面。

**两阶段(降低"用未验证 golden 替换自验证 golden"的风险)**:
- **Phase 1 — 资格验证(additive)**:bigop 作第三列,跑 `gold↔big`。`gold↔big` 全绿 = 在仓内
  证明 bigop 是正确的反向参考(隔离到 compute 一维,适配错会发散到 ≫ 阈值,而 bf16 kernel 差 ~1e-3)。
- **Phase 2 — 提升/替代**:把 Triton 对比基准从 torch golden 切到 bigop(`tri↔big`),从正确性
  测试中退役 torch-loop 列。`moe_backward_torch` **保留**为 autograd 交叉验证的 meta-oracle
  (`backward.py:__main__::run_cross_check`,手写反向 vs autograd)与基准用,不删。

> 失败聚合:Phase 2 后 `run_test` 末尾 `all_reduce(MIN)` 逻辑不变,只是对比对象由 gold→big。

## 计划修正点(相对初稿,实施前必读)

1. **golden 路径**:`tests/_goldens/` → **`src/mega_moe/_goldens/`**。`test_moe_backward.py:35-36`
   已是 `from mega_moe._goldens.backward import ...`;`bigop_ref.py` 落 `src/mega_moe/_goldens/`。
2. **逐 op 映射已逐行核对真实代码,确认无误**(权重 `[E,out,in]` vs bigop `[E,K,N]` 对齐、
   wgrad 实参互换 + 转置、swiglu 因式 = `bigop.py:531-537`)。Step 2 给出可直接落地代码。
3. **遗漏项补上**:`benchmark/layer/bench_backward.py` 也用 golden,Phase 2 须一并切到 bigop。
4. **bigop compute 全程 bf16**:与 torch golden 的 FP32 swiglu + M-pad-256 GEMM 是不同 kernel
   路径,`gold↔big` 在 swiglu 链路 ~1e-3,落 `GRAD_RTOL/ATOL=2e-2/1e-2` 内。
5. **仓库两份副本**:计划路径以 `/home/z00905891/Mega-MoE-TD` 为准;shell cwd 是其兄弟
   `/home/z00905891/triton_dist/Mega-MoE-TD`(同分支同提交)。改动须落在 `python -c "import mega_moe"`
   解析的那份的 `src/mega_moe/`,两份保持一致。

---

## 目标(Context)

Mega-MoE-TD 现在只有**一套**手写 golden(`src/mega_moe/_goldens/backward.py::moe_backward_torch`,
纯 torch 循环 + HCCL)验证 Triton mega-kernel 反向。golden 与被测对象同源同维护、compute 路径
单一,一旦 golden 写错会被两边共享而难发现。

引入 bigop(`gitcode.com/jzhoujg/bigop.git`,已 UT/ST 证明与 Megatron bit-level 对齐)作为
**独立 golden**:其 compute 原语用 `npu_grouped_matmul`(split_item=2/3)+`npu_swiglu_backward`,
与现有 torch 循环 golden 是**不同 kernel 路径**。打包成可安装第三方库,`import bigop` 复用其
compute 原语构建 `moe_backward_bigop(saved, dy)`,**分阶段替代** torch golden。

## 核心策略(已被 Explore map 证实)

**两端 A2A 原样复用 mega_moe 现有 golden,只把中间 compute 核心换成 bigop 的 npu op。**

依据:mega_moe 的 `saved["recv_hidden_sorted"]` 已经是"派发后 + 本地排序 + 专家连续"的 token
序列,正好等价于 bigop 派发完成的 `global_input_tokens`;`saved["expert_counts"]` 等价于
bigop 的 `tokens_per_expert`。所以 bigop 的 router / permute / dispatch-A2A 这套(Megatron 布尔
`routing_map` 语义)**整段跳过**,只取它的 GEMM+SwiGLU 核心。

> **独立性边界(诚实声明)**:此设计把 bigop 与 torch golden 的差异**隔离到 compute kernel 一维**;
> 二者**共享** `saved`(及 `moe_forward`)、`combine_bwd_a2a`、`dispatch_bwd`。因此交叉验证覆盖
> compute 维度,**不覆盖** dispatch/A2A/前向 saved 构造——那部分仍由 torch golden 的 autograd
> 交叉验证(`run_cross_check`)兜底。要彻底独立需用 bigop 自有前向(本次非目标)。

## 逐 op 映射(实施核心,已核对)

`moe_backward_bigop(saved, dy)` 镜像现有 golden 的 7 步驱动(对应 5 mega-op),逐 op 标注来源:

| 步 | 现有 golden (`moe_backward_torch`) | 新 bigop golden | 来源 |
|---|---|---|---|
| 1a combine_bwd-a2a | `combine_bwd_a2a(dy,saved)`→`grad_fc2_out_sorted[M,H]` | **原样复用**(torch+HCCL A2A) | mega_moe |
| 1b fc2 input-grad | `grouped_matmul(grad_fc2_out, fc2, counts, transpose=False)`→`[M,F]` | `bigop._grouped_matmul(grad_fc2_out, fc2, counts)` | **bigop** |
| 2 swiglu bwd | 纯 torch FP32(`sigmoid`/`silu`,含算 `grad_gate`) | `npu_swiglu_backward` + probs 因式分解(见下) | **bigop** |
| 3 fc2 wgrad | `grouped_transposed_matmul(grad_fc2_out, swiglu_out_weighted, counts)`→`[E,H,F]` | `bigop._grouped_wgrad(...)` + 转置回(见下) | **bigop** |
| 4a fc1 input-grad | `grouped_matmul(grad_fc1_output, fc1_combined, counts, transpose=False)`→`[M,H]` | `bigop._grouped_matmul(grad_fc1_output, fc1_combined, counts)` | **bigop** |
| 4b dispatch_bwd | `dispatch_bwd(grad_recv_hidden_sorted, grad_gate, saved)`→`grad_hidden[B,H]`,`grad_routing_weights[B,topk]` | **原样复用**(torch+HCCL reverse-A2A) | mega_moe |
| 5 fc1 wgrad | `grouped_transposed_matmul(grad_fc1_output, recv_hidden_sorted, counts)`→`[E,2F,H]`→`chunk` | `bigop._grouped_wgrad(...)` + 转置回 + `chunk` | **bigop** |

返回字典键与现有 golden 完全一致(`grad_hidden, grad_routing_weights, grad_fc1_1, grad_fc1_2,
grad_fc2` + 中间量 `grad_fc2_out_sorted, grad_swiglu, grad_fc1_output, grad_gate,
grad_recv_hidden_sorted, grad_fc1`),便于直接 `cmp_grad` 与逐 op 定位。

## Weight / 原语适配(精确,已核实)

**布局约定差异**:mega_moe 权重存 `[E, out, in]`(Megatron 线性权重);bigop `_grouped_matmul`
要 `[E, K=in, N=out]`、`_grouped_wgrad` 返回 `[E, in, out]`。

| compute 步 | bigop 调用 | 适配操作 |
|---|---|---|
| 1b / 4a(input-grad) | `_grouped_matmul(grad, weight, counts)` | **无需适配** —— `fc2[E,H,F]`、`fc1_combined[E,2F,H]` 的尾两维正好是 bigop 要的 `[K,N]`,直传 |
| 3(fc2 wgrad) | `g=_grouped_wgrad(swiglu_out_weighted, grad_fc2_out, counts)` | 返回 `[E,F,H]`;`g.transpose(-1,-2).contiguous()`→`[E,H,F]`。**注意实参顺序与 mega_moe `grouped_transposed_matmul(grad_out, orig_in)` 相反**(bigop 是 `(orig_in, grad_out)`) |
| 5(fc1 wgrad) | `g=_grouped_wgrad(recv_hidden_sorted, grad_fc1_output, counts)` | 返回 `[E,H,2F]`;转置→`[E,2F,H]`;再 `chunk(2,dim=1)`→`grad_fc1_1, grad_fc1_2` |

**SwiGLU 反向(bigop 因式)**:bigop 把"乘 probs"与"swiglu 反向"拆开,
mega_moe `saved["fc1_output"]`(=gate_up `[M,2F]`)直接喂 `npu_swiglu_backward`:
```
grad_swiglu_for_npu = grad_swiglu * recv_weights_sorted.unsqueeze(-1)   # probs 因式
grad_fc1_output     = npu_swiglu_backward(grad_swiglu_for_npu, fc1_output)  # → [M,2F]
swiglu_out          = npu_swiglu(fc1_output)                              # 重算(probs-grad 需要)
grad_gate           = (grad_swiglu * swiglu_out).sum(-1)                  # = dScale, 喂 dispatch_bwd
```
数学上与现有 golden 的 `swiglu_bwd` 等价,但走 `npu_swiglu_backward` kernel(bf16),与现有
golden 的 FP32 torch 在这一步会有 ~1e-3 kernel 差(符合预期,非 bug)。

**`_grouped_wgrad` 铁律**:左矩阵必须 `input_tokens.T` 且**不可 `.contiguous()`**(否则丢失
group_type=2 转置标记 → EZ1001),见 `bigop.py:293-303`。**我们直传 mega_moe 的连续 saved 张量,
让 bigop 内部做 `.T`**——切勿预先转置再传入。`counts` 用 `saved["expert_counts"]`(int32),
bigop 内部自行转 int64,`group_list_type=1`(非 cumsum)两边一致。

## 落地步骤

### Step 1 — bigop 作为 `3rdparty/` git submodule 接入 + 打包安装 ✅ 已完成

设计要点保留备查:1.1 `3rdparty/bigop/` 单依赖子目录;1.2 给 bigop 加 `pyproject.toml`
(`py-modules=["bigop"]`)+ `.gitignore`(`*.egg-info/`、`build/`);1.3 `git submodule add`
并 pin commit;1.4 editable 安装(`pip install -e ... --no-deps`,匹配 submodule 工作流);
1.5 `scripts/install_3rdparty.sh`;1.6 升级标准动作;1.7 `.gitignore` 不忽略 gitlink。

**✅ 落地结果**:
- bigop:`68ca4cb build(packaging): add pyproject.toml (py-modules=bigop) + ignore egg-info/build`。
- mega_moe:`.gitmodules` 注册 `3rdparty/bigop` 并 pin `68ca4cb`;`scripts/install_3rdparty.sh`;
  README "测试期第三方依赖" 小节。
- 验证:`python -c "import bigop"` → `3rdparty/bigop/bigop.py`;`pip show bigop` editable。
  (clone 后 `3rdparty/bigop/` 默认未 checkout,需先跑 `bash scripts/install_3rdparty.sh`。)

### Step 2 — 新建 `src/mega_moe/_goldens/bigop_ref.py`
- `def moe_backward_bigop(saved, dy) -> dict`:按上方"逐 op 映射表"实现。
- 路径:`src/mega_moe/_goldens/bigop_ref.py`(golden 在包内)。
- 1a/4b `from mega_moe._goldens.backward import combine_bwd_a2a, dispatch_bwd`(复用)。
- compute `from bigop import _grouped_matmul, _grouped_wgrad`;swiglu 用 `torch_npu.npu_swiglu(_backward)`。
- 入口签名、返回键与 `moe_backward_torch` 完全一致。

可直接落地的实现(逐 op 已对真实代码核对):
```python
# src/mega_moe/_goldens/bigop_ref.py
# 第二套 MoE 反向 golden:A2A 两端复用 mega_moe 现有 torch+HCCL,compute 核心换 bigop 的
# npu 融合原语(npu_grouped_matmul / npu_swiglu(_backward))。入口/返回键与 moe_backward_torch 一致。
import torch
import torch_npu  # noqa: F401  (npu_swiglu, npu_swiglu_backward)

from bigop import _grouped_matmul, _grouped_wgrad
from mega_moe._goldens.backward import combine_bwd_a2a, dispatch_bwd


def moe_backward_bigop(saved, dy):
    """bigop-compute 反向 golden。返回键与 moe_backward_torch 完全一致。"""
    dy = dy.to(saved["output"].dtype)
    counts = saved["expert_counts"]                      # int32;bigop 内部转 int64

    # 1a combine_bwd-a2a —— 复用(torch+HCCL)
    grad_fc2_out_sorted = combine_bwd_a2a(dy, saved)     # [M,H]

    # 1b fc2 input-grad —— bigop:fc2[E,H,ffn]=[E,K=H,N=ffn],尾两维正合,无需适配
    grad_swiglu = _grouped_matmul(grad_fc2_out_sorted, saved["fc2"], counts)  # [M,ffn]

    # 2 swiglu bwd —— bigop 因式(*scale 的 bwd + npu_swiglu_backward)
    probs = saved["recv_weights_sorted"].unsqueeze(-1)            # [M,1]
    grad_swiglu_for_npu = grad_swiglu * probs                     # *scale 反向
    grad_fc1_output = torch_npu.npu_swiglu_backward(
        grad_swiglu_for_npu, saved["fc1_output"], dim=-1)         # [M,2F]
    swiglu_out = torch_npu.npu_swiglu(saved["fc1_output"], dim=-1)  # [M,F] 重算(probs-grad 需要)
    grad_gate = (grad_swiglu * swiglu_out).sum(dim=-1)            # [M] = dScale,喂 dispatch_bwd

    # 3 fc2 wgrad —— bigop:返回[E,ffn,H],转置回[E,H,ffn](实参顺序与 mega_moe 相反)
    g = _grouped_wgrad(saved["swiglu_out_weighted"], grad_fc2_out_sorted, counts)  # [E,ffn,H]
    grad_fc2 = g.transpose(-1, -2).contiguous()                  # [E,H,ffn]

    # 4a fc1 input-grad —— bigop:fc1_combined[E,2F,H]=[E,K=2F,N=H],尾两维正合,无需适配
    grad_recv_hidden_sorted = _grouped_matmul(grad_fc1_output, saved["fc1_combined"], counts)  # [M,H]

    # 4b dispatch_bwd —— 复用(torch+HCCL reverse-A2A)
    grad_hidden, grad_routing_weights = dispatch_bwd(grad_recv_hidden_sorted, grad_gate, saved)

    # 5 fc1 wgrad —— bigop:返回[E,H,2F],转置回[E,2F,H],再 chunk(2,dim=1)
    g = _grouped_wgrad(saved["recv_hidden_sorted"], grad_fc1_output, counts)  # [E,H,2F]
    grad_fc1 = g.transpose(-1, -2).contiguous()                  # [E,2F,H]
    grad_fc1_1, grad_fc1_2 = torch.chunk(grad_fc1, 2, dim=1)     # 各 [E,F,H]

    return dict(
        grad_hidden=grad_hidden, grad_routing_weights=grad_routing_weights,
        grad_fc1_1=grad_fc1_1, grad_fc1_2=grad_fc1_2, grad_fc2=grad_fc2,
        # 中间量(键与 moe_backward_torch 一致,便于逐 op 排查)
        grad_fc2_out_sorted=grad_fc2_out_sorted, grad_swiglu=grad_swiglu,
        grad_fc1_output=grad_fc1_output, grad_gate=grad_gate,
        grad_recv_hidden_sorted=grad_recv_hidden_sorted, grad_fc1=grad_fc1,
    )
```
> `saved["fc1_output"]` 即前向 `fc1_out`(gate_up `[M,2F]`),`npu_swiglu` 末维对 `[gate,up]`
> 做 `silu(gate)*up` → `[M,F]`,与 torch golden 的 `swiglu_out` 同义。bf16 kernel 与 FP32 golden
> 在此步 ~1e-3 差,符合预期。

### Step 3 — 测试接入(分两阶段)

文件:`tests/layer/test_moe_backward.py`。import:`from mega_moe._goldens.bigop_ref import moe_backward_bigop`,
建议 `pytest.importorskip("bigop")` 守卫(bigop 未装时退回 tri↔gold,不阻断现有测试)。

**Phase 1 — 资格验证(additive 三列)**。`run_one` 计算段加 `big`(`big` 无需 `peer_mem`,与 `gold` 并排):
```python
with torch.no_grad():
    gold = moe_backward_torch(saved, dy)
    big  = moe_backward_bigop(saved, dy)          # 新增
with torch.no_grad():
    tri = moe_backward_triton(saved, dy, peer_mem)
```
对比三组、逐 grad:
```python
checks = ["grad_hidden", "grad_routing_weights", "grad_fc1_1", "grad_fc1_2", "grad_fc2"]
pairs = [("tri/gold", tri, gold), ("tri/big", tri, big), ("gold/big", gold, big)]
for pname, a, b in pairs:
    for n in checks:
        ok, mx, rel, nbad = cmp_grad(f"{pname}/{n}", a[n], b[n]); all_ok &= ok
```
- **`gold/big` 全绿 = bigop 资格通过**,方可进入 Phase 2。
- 定位适配错:看中间量 `grad_swiglu`/`grad_fc1_output`(已返回)逐 op 缩小范围;真适配错发散 ≫1e-3,
  bf16 kernel 差 ~1e-3,阈值 `GRAD_RTOL/ATOL=2e-2/1e-2` 可区分。

**Phase 2 — 提升/替代**。Phase 1 稳定后,把 Triton 基准从 gold 切到 big(删 `tri/gold`、`gold/big`,
仅留 `tri/big`);从正确性测试退役 `moe_backward_torch` 调用。`moe_backward_torch` **保留**(autograd
meta-oracle + 基准用)。同步改 `benchmark/layer/bench_backward.py:161`(基准正确性参考切 bigop)。

> 不必删 `moe_backward_torch`:`backward.py:__main__::run_cross_check`(手写反向 vs autograd)仍是
> golden 自身的 meta-oracle;bench 仍可留 torch 作旁证。退役仅指"不再作 Triton 对比基准"。

## 风险与未决(本次新增,实施前评估)

1. **0-token 专家(首要运行时风险)**:torch golden `grouped_matmul` 对 `cnt==0` 专家 `if cnt>0` 跳过;
   bigop `npu_grouped_matmul` 拿原始 `group_list`(含 0)。`BACKWARD_SHAPES_SMALL`(128 专家/ws=2→64
   本地专家,~32 token/专家)随机路由下尾部可能产生 0 计数。**须在 Phase 1 实测**:`npu_grouped_matmul`
   对 `group_list` 含 0 的行为(报错/空转/错算)。若报错,在 bigop 侧或 bigop_ref 包一层 0-count 跳过
   (镜像 torch golden),或限定测试 shape 保证每专家 ≥1。
2. **M-pad/alignment**:torch golden 把每专家 M 补到 256(bf16 bit-stable);bigop 不补,走另一 Cube
   kernel。非 bug,但 `gold/big` 的 wgrad(input→ grad_fc1_*/grad_fc2/grad_hidden 链路)会比 swiglu 单步
   噪声略大,仍在阈值内。
3. **bf16 vs FP32 swiglu**:grad_gate/grad_routing/grad_hidden 经 swiglu 链路带 ~1e-3;若 `gold/big`
   超阈,先查因式分解与装箱,再疑原语;必要时 swiglu 步暂留 torch、只换 GEMM(部分 bigop 化可增量交付)。
4. **contiguity / wgrad 转置标记**:必须直传连续 saved 张量、由 bigop 内部 `.T`;预先 `.T.contiguous()`
   传入会触发 EZ1001。已在适配表标注,代码已遵守。
5. **独立性边界**:bigop 与 torch golden 共享 `saved`/A2A,交叉验证只覆盖 compute。dispatch/A2A/前向
   saved 的正确性仍靠 `run_cross_check`(autograd)兜底——Phase 2 退役 torch 列时**勿删**该交叉验证。
6. **基准同步**:`bench_backward.py` 用 golden 做正确性旁证,Phase 2 须切 bigop(或显式保留 torch 并注明)。

## 关键文件

| 仓库 | 文件 | 动作 |
|---|---|---|
| bigop | `pyproject.toml`、`.gitignore` | ✅ **已完成**(commit `68ca4cb`) |
| mega_moe | `.gitmodules`、`3rdparty/bigop`(gitlink) | ✅ **已完成**(pin `68ca4cb`) |
| mega_moe | `scripts/install_3rdparty.sh` | ✅ **已完成** |
| mega_moe | `README.md` | ✅ **已完成** |
| mega_moe | `src/mega_moe/_goldens/bigop_ref.py` | ⬜ **新增**(Step 2,compute 用 bigop,A2A 复用) |
| mega_moe | `tests/layer/test_moe_backward.py` | ⬜ **改**(Step 3,Phase1 三列 / Phase2 替代) |
| mega_moe | `benchmark/layer/bench_backward.py` | ⬜ **改**(Phase 2,golden 切 bigop;初稿遗漏) |

复用:`bigop._grouped_matmul/_grouped_wgrad`(`bigop.py:280/293`)、`torch_npu.npu_swiglu(_backward)`、
`mega_moe._goldens.backward.{combine_bwd_a2a, dispatch_bwd}`、`tests._numeric.cmp_grad`。

## 验证

1. `python -c "import bigop"` 成功(✅ 已通过)。
2. Phase 1,2 卡:`torchrun --nproc-per-node=2 -m pytest tests/layer/test_moe_backward.py::test_backward_2ranks -m dist -v -s`
   → 三组对比 dx/dw1/dw2/probs max_diff 落阈值;**`gold↔big` 全绿**才进 Phase 2。
3. Phase 2,同上但仅 `tri↔big`(bigop 为 primary golden)。
4. (可选)8 卡 `RANK_SIZE=8` / Kimi shapes 同套。

## 非目标(本次不做)

- 不接 bigop 的 router / aux_loss / 整块 `BigOp`(mega_moe 是 post-routing)。
- 不用 bigop 自有前向(保持"隔离 compute"设计;彻底独立前向是后续工作)。
- 不动 `src/mega_moe/` 生产路径与 Triton kernel。
- 不做 ST 端到端训练对比(本次只用 bigop 作 op 级 compute golden)。
- 不删 `moe_backward_torch`(保留为 autograd meta-oracle 与基准旁证)。
