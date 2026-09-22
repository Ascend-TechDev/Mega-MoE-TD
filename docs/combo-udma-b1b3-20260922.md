# Combo 引擎 MTE|UDMA + 后向/reprefetch UDMA 推模式 — 状态与迁移文档

日期：2026-09-22
仓库：`/home/t0303/project/mmt`（分支 `fix/single-kernel-forward-w8`，远端 `origin = gitcode.com/jzhoujg/Mega-MoE-TD.git`）
框架：`/home/t0303/project/framework`（分支 `moonep`，远端 `origin = gitcode.com/jzhoujg/MindSpeed-MM_MoonEP.git`）
设备：950DT（aarch64），8 卡；python venv：`/root/moe-venv/activate-moe.sh`

---

## 1. 总体目标

framework（moonep 分支）8 卡整网跑通：**前向 single-kernel + 后向 mega kernel + MoonEP（pooled replica tables）**，算子侧重算默认开（`MOE_SAVED_RECOMPUTE=1`）、reprefetch 默认开。MoonEP 相关路径（FC1_OFFLOAD / DOWN_DIRECT / 量化）全部补齐后再跑整网（用户指令"全部补齐再跑"）。

## 2. 方案背景（为什么改成 combo + UDMA 推）

三种引擎配置的死锁格局：

| session 引擎 | 前向 single-kernel | 后向（getmem owner-pull） |
|---|---|---|
| 纯 MTE | OK | OK（w2/w8 曾绿） |
| MTE\|UDMA combo | OK | **grad_fc1 损坏**（hot-expert owner rank；grad_fc2 正常 = pusher 死代码时的同款特征） |
| 纯 UDMA | 崩（无 MTE） | — |

结论：combo session 下 `libshmem_device.getmem` 家族不可用（环境问题，非本仓代码）；纯 MTE 下前向 UDMA QP push 无法工作。
**采用方案（用户拍板"方案2"）**：后向 grad 传输与 reprefetch 全部改成 **UDMA 推模式**，框架 session 改为 `MTE|UDMA`；"B1 不探针直接全套推模式"，"reprefetch 也改成 udma/推模式"。

## 3. 本轮发现的问题与修复

按发现顺序（全部在 `src/mega_moe/kernels/mega_bwd.py`，除非另注）：

### 3.1 KeyError: GRAD_UDMA 未识别
两个 mega kernel 定义（`kernel_moe_backward_mega` / `kernel_moe_backward_mega_recompute`）缺 `GRAD_UDMA: tl.constexpr,` 参数，而 wrapper 的 `mega_kwargs` 传了它 → 启动即 KeyError。
**修复**：两处 def 在 `GRAD_REDUCE` 后加 `GRAD_UDMA: tl.constexpr,`。

### 3.2 pusher 死代码（嵌套错误）
4 个调用点（window2 PUSH_DN ×2 kernel、window3 PUSH_GU ×2 kernel）写成了：
```python
if sub_vec_id() == 0:
    _mega_grad_owner_pull(...)
    if GRAD_UDMA:
        if sub_vec_id() == 1:      # 永假 — 死代码
            _mega_grad_udma_push_sweep(...)
```
首个 udma 运行因此"静默无 push"，产生与 combo+getmem 同款的损坏特征（127k mismatch，无挂死）。
**修复**：dedent 成 sibling——`==0` 跑 owner-pull，`==1` 跑 push sweep，同在 `al.scope(core_mode="vector", disable_auto_sync=True)` 内。

### 3.3 可见性竞态（w2 曾 43–46k 元素损坏，run-varying）
consumer 在 `dl.wait` 之后用**裸指针**读 staging 行 → 与仍在途的 payload 竞态。
**修复**：`token = dl.wait(...)` 后必须 `row = dl.consume_token(staging_base_ptr, token) + task*CHUNK`（仓内既定惯用法：`_mega_push_mtile`、`_mega_wgrad_sweep` WAIT_DISP 同款）。GU/DN 两分支都改。修后 w2 全绿。

### 3.4 w8 挂死 → **credit 协议在 W>2 成环**（最新，已写修复、**未验证**）
现象：`test_single_kernel_moonep_autograd_w8`（两 transport 均 udma，8GB 堆）564.77s 失败——第一次 backward（`out2.backward`）中 mega kernel 自旋，多核 aicore timeout/trap，表面错误 507014 出现在 `zero_consumed_replica_slots` 的 stream sync。日志：`/root/.claude/jobs/b01bb115/tmp/b1b3_autograd_w8b.log`。

根因（协议级，w2 单 peer 证明不了）：
- consumer（owner 侧 sub_vec0）：task 序、ordinal 序，先 credit 后等 arrival；
- pusher（peer 侧 sub_vec1）：自己 (dest,home) 行序、chunk 序，等 credit 再推；
- W=8 时多 owner/多 peer 交织可成环：consumer X 等 P2 的 arrival → P2 等 consumer Y 的 credit → Y 等 P3 的 arrival → P3 等 X 的 credit → **环**。

**修复：预 credit 扫**。每个 consumer 程序在进入任何 `dl.wait` 之前，先把自己所有 task 的**首 ordinal** credit 发出去（GU/DN 两分支对称）：
```python
if GRAD_UDMA:                                   # pre-credit sweep
    for task in range(pid, EPN * chunks, ncores):
        start, end = home_offsets[home], home_offsets[home+1]
        if end > start and desc_peer[start] != LOCAL_RANK:
            signal_op(cred + 2*task, epoch*256 + start, SET, first_peer)
for task in ...:                                # 原消费循环
    for ordinal in range(start, end):
        ...
        if ordinal > start:                     # start 已预发，勿重复
            signal_op(cred + 2*task, val, SET, peer)
        token = dl.wait(arr + 2*task, ..., waitValue=val)
        row = dl.consume_token(staging, token) + task*CHUNK
        _mega_grad_accum(...)
```
无死锁归纳：所有 0 号 credit 无条件先落地 → 每个 pusher 首个 credit-wait 必解除 → 所有 0 号 arrival 必到 → k 号 credit 在消费 k−1 后即发 → 归纳成立。同 task 内 exact-match `dl.wait` 安全：k+1 号 credit 只会在 k 号 arrival（pusher 已过其 credit-wait 并发出 put）之后才写。

### 3.5 真正的 w8 根因：ordinal 编码 ABI 不一致（2026-09-22 下午定位，已修）

预 credit 修复后 w8 仍挂 → 加 `MOE_MEGA_WAIT_DEBUG=1`（有界自旋 bail，见 3.6）+ `MOE_MEGA_GRAD_PROBE=1`（heap5 协议审计）两轮 102s 快速诊断定位：

- **site=12 记录**（pusher 等 credit）：`want=512 observed=513/514` —— credit 写**能落地**，但值对不上；
- **heap5 表 dump**：w8 实际 ETC 极稀疏（每 owner 仅 3 个单持有者 home：r1←{r0,r5,r6}、r2←{r3,r4,r7}，home_off=[0,1,2,3,3]，每个 home 仅 1 个持有者）；
- **根因**：consumer 的 credit/arrival 值用**全局描述符序号**（`epoch*256 + ordinal`，ordinal=start..end 是 owner 描述符表的全局下标），pusher 用 **home 内序号**（`epoch*256 + my_ord`，host 端 sched 按持有者数计的位置）。home0（start=0）两编码碰巧相同能通，home1/2 永远错位（513/514 vs 512）→ 双向 exact-match wait 全部卡死。w8 观察到的"credit 环"其实从未形成（单持有者 home 无链）；原始 w8 挂死由此 ABI 不一致完全解释。
- **修复**：consumer 统一改 home 内序号 —— pre-credit 值 `epoch*256`（首序号=0），循环内 `val = epoch*256 + (ordinal - start)`；pusher 不变（`my_ord` 本就是 home 内序号）。GU/DN 对称各两处。
- 观察到的 `observed=0` 条目是时序伪影：owner 的 stride-32 程序尚未预 credit 到该 task 时 pusher 已 1M 自旋 bail；正常路径（无 bail）不受影响。epoch 每 rank 均匀（同一 launch 全 rank 相同：2→4），无跨 rank 漂移。

### 3.6 快速死锁定位（用户要求缩短定位时间，已落地）

- CANN 无 aicore watchdog 的环境变量可调（libruntime.so 只暴露 rtSetOpExecuteTimeOut API）→ 改为**kernel 内有界自旋 bail**：`_wait_bounded_report`（dispatch_fc2_bwd.py）把 grad 传输 4 类 wait（site 10=GU arrival、11=DN arrival、12=GU credit、13=DN credit）换成 ~1M 次自旋后记录 `[site,slot,want,observed,expert,spins]` 到 `dbg_ptr+pid*8` 并**带错继续**。死锁从 ~10min watchdog 变成 **~102s** 在梯度比对处快速失败，且记录齐全（w8 实测：128 条 site-10 + 12 条 site-12，两步 backward 的记录分别可辨）。
- `MOE_MEGA_WAIT_DEBUG=1` 牺牲数值正确性（满足路径也无 acquire fence），只用于诊断；`MOE_MEGA_GRAD_PROBE=1`（host-only，不触发 kernel 重编译）dump 双方表 + epoch + scratch 指针 + launch 后四族 word 非零内容。

## 4. B1/B3/A 三个交付点

### B1（#9）：后向 grad transport UDMA 推反转 — 代码完成
- consumer/pusher/预 credit 如上；wrapper 侧：`grad_udma = MOE_MEGA_GRAD_TRANSPORT=udma`，scratch 由 `grad_transport.buffers.ensure_grad_push_scratch(...)`（task-indexed staging + arrival/credit words，一次分配零化），epoch 用 `next_push_epoch(floor=1)`（与 forward push、reprefetch 共享的 SET-only 单调计数器，跨调用免清零）。
- push schedule（host 端，每 rank）：扫全局 ETC `[W, EPN]`，本 rank 持有、它 rank 为 owner 的 slot 生成 `(dest, home, my_ord, my_slot)`，排序填 `grad_push_sched [epn6,4]`（-1 填充）。ordinal = 在 owner 的 (peer,slot) 字典序 descriptor list 中的位置，所有 rank 从同一 ETC 独立推导。
- **w2 已验证**：probe single+combo+udma PASS；autograd w2 GRAD=udma PASS（23.3s）；GRAD+REPREFETCH 均 udma PASS（14.19s，两趟 backward 交错、epoch/scratch 复用验证过）。
- **w8 未通过**（见 3.4，修复待验证）。

### B3（#8，completed）：reprefetch 改 UDMA 推 — 完成
`_kernel_replica_repush_udma`（`kernels/replica_weight_prefetch.py`）：peer QP 推、ready words 走 u64 视图（同 64B/slot ABI）、每 peer 尾 `_udma_quiet` 使 R1 collective barrier 成为合法发布边。wrapper 里 `MOE_MEGA_REPREFETCH_TRANSPORT=udma` 分支已在 mega 启动前独立 launch。w2 绿；w8 与 B1 一起待复验。

### A（#10）：框架侧 — 代码完成、待整网验证
`megamoe_ep_dispatcher.py`：`MEGAMOE_OP_ENGINE`（默认 `combo` = `MTE|UDMA`，`mte` 回退纯 MTE）。
`examples/kimi_k3/finetune_kimik3.sh`：已 export `MOE_MEGA_GRAD_TRANSPORT=udma`、`MOE_MEGA_REPREFETCH_TRANSPORT=udma`（另有 MOE_FUSED_ASH_SIZE_GB=8、MEGAMOE_REPLICA_POOL=1、MOE_BWD_MEGA=1、MOE_SAVED_RECOMPUTE=1、MOE_MEGA_REPREFETCH=1、MEGAMOE_SINGLE_KERNEL=1、MEGAMOE_FC1_OFFLOAD=1、MOE_DOWN_DIRECT=1）。

## 5. 关键协议细节（迁移后改代码必读）

- **QP ownership**：program `pid` 是目的 rank `pid` 的 QP 唯一 issuer（pusher 过滤 `dest == pid`），要求 **W ≤ ncore()**（32）。
- **UDMA API**：只有 `udma_put_nbi / udma_put_signal_nbi / udma_quiet`（无 get）；**同一 QP 上 WQE 也无顺序保证**；`put_signal_nbi(dst, src, cnt, signal_addr_u64, val_u64, peer)` —— payload+SET 一个 WQE，天然不乱序；对称偏移寻址（dst 指针的 offset 落在 peer 同偏移处）。
- **word 家族**：int32 成对视为 u64（`[tasks] u64 == [2*tasks] i32`）；SET-only、值 `epoch*256 + ordinal` 单调 → 免跨调用清零；ordinal 上限 255（实际 ≤ W−1=7）。
- **dl.wait/consume_token 惯用法**：`dl.wait` 返回 token，读远端写入的数据必须 `consume_token(data_ptr, token)` 拿可见性保证的指针；裸指针读 = 竞态。
- **sub_vec 分工**：consumer 在 `sub_vec_id()==0`、pusher 在 `==1`，必须同层 sibling（嵌套即死代码）。
- **950DT codegen 脆弱性**：本仓 mega kernel 有记录在案的整体误编译家族（见 mega_bwd.py 模块 docstring）——**任何源码改动都会重掷骰子**，所以每次改动后必须重跑 w2 功能门（`tests/layer/test_moe_suite.py` 的 moonep autograd w2/w8 + `test_debug_moonep_saved.py` 探针）。
- **签名冻结**：mega kernel 的死参数（reprefetch 旧组等）**故意保留**，删除会改变二进制并重掷误编译骰子（mega_bwd.py 内注释）。

## 6. 复现/验证命令（在 `/home/t0303/project/mmt`）

```bash
source /root/moe-venv/activate-moe.sh
export ASH_MASTER_PORT=$((RANDOM % 20000 + 40000))
export DIST_TEST_TIMEOUT_S=900
export MOE_FUSED_ASH_SIZE_GB=2          # w2 用 2；w8 用 8
export TRITON_CACHE_DIR=/tmp/triton-moe-<fresh-tag>
export MOE_MEGA_GRAD_TRANSPORT=udma
export MOE_MEGA_REPREFETCH_TRANSPORT=udma
python -m pytest "tests/layer/test_moe_suite.py::test_single_kernel_moonep_autograd_w2" -x -q -s
```
- w8 测试：`::test_single_kernel_moonep_autograd_w8`（**约 9.5 分钟，超过 Bash 工具 10 分钟上限时必须 run_in_background**）。
- 诊断开关：`PROBE_MODE=single|native`（探针/原生前向）、`PROBE_UDMA=0|1`（session 引擎对照）、`MOE_MEGA_WAIT_DEBUG=1`（dl.wait 超时点位/slot/want/observed）、`MOE_MEGA_HEAP_PROBE=1`（对称堆 offset 审计）、`PROBE_GRAD_DEBUG=1`（grad_fc1_1 直方图逐 bin）。
- Kimi w8 形状：E=32, epn=4, EPR=8（=2*epn，wrapper 有断言），W=8 → 每 home 最多 7 个 peer descriptor（首次吃多 ordinal 协议）。

## 7. 当前进度一览

| 项 | 状态 |
|---|---|
| probe w2（single + combo + udma 双 transport） | ✅ PASS |
| autograd w2（GRAD=udma） | ✅ PASS 23.3s |
| autograd w2（GRAD+REPREFETCH 均 udma） | ✅ PASS 14.19s |
| autograd w8（双 udma） | ❌ 564.77s 挂死 → 定位 credit 环 → **预 credit 修复已写、未验证** |
| A 框架 dispatcher/脚本 | ✅ 代码完成，随本次一并提交 |
| #6 FC1_OFFLOAD/DOWN_DIRECT/量化 MoonEP 补齐 | 分析完成、验证未跑（见 §8） |
| #5 整网 | 未开始（等上面全绿） |

## 8. 后续计划（按序）

1. **验证预 credit 修复**：
   a. w2 autograd 复跑（双 udma）——确认无 codegen 回归（协议上 w2 单 ordinal，行为应与旧版一致）；
   b. w8 复跑（双 udma，后台任务）；若仍挂，用 `MOE_MEGA_WAIT_DEBUG=1` 定位自旋的 wait（site/slot/want/observed），区分 credit-wait 还是 arrival-wait。
2. **#6 MoonEP 补齐验证**（分析已做完，按结论执行）：
   - FC1_OFFLOAD：`maybe_offload_fc1/maybe_reload_fc1`（`ops/function.py`）moonep 无关、结构性完备 → 跑 w2 autograd 加 `MEGAMOE_FC1_OFFLOAD=1` 验证；
   - DOWN_DIRECT：**不要**删 `forward.py` 里 `_down_direct` 的 `not self.enable_moonep` 条件——MoonEP 下 flat RMA push（`fused_moonep.py::_single_moonep_push` 的面板寻址）要求 home down 表连续，strided home 表会打乱推送；后向 `dispatch_fc2_bwd.py` 的 `fc2=saved["fc2"].contiguous()` 双胞胎在 saved fc2 已是 staged 连续 `_fc2_ws` 时本来就是 no-op → 验证该别名关系即可；
   - 量化（save_fc1_dtype=fp8）：mega 侧 FC1_FP8 反量化加载已支持（非 moonep 单卡用例有误差界断言），**缺 moonep 覆盖** → 在 `test_moe_suite.py` 给 `run_single_kernel_moonep_autograd_case` 加 fp8 save 子用例。
3. **#10/#5 整网**：framework `examples/kimi_k3/finetune_kimik3.sh` 8 卡跑；观察 `[mega-heap]` 审计行、首次迭代数值与 hang。
4. **cutover 决策（推迟到整网绿后，需用户确认）**：算子侧默认值仍是 `getmem/store`（`MOE_MEGA_GRAD_TRANSPORT`/`MOE_MEGA_REPREFETCH_TRANSPORT`），框架脚本显式 export udma；整网绿后再议是否把默认翻成 udma。
5. 遗留：REPREFETCH=1 的 W4 回归（旧项，未在本轮复现）；memory 文件更新（本次已做）。

## 9. 本次提交内容

**mmt（分支 `fix/single-kernel-forward-w8`）**：`mega_bwd.py`（B1 全套 + §3 四个修复）、`replica_weight_prefetch.py`（B3 udma re-push + helpers）、`runtime/replica_weight_prefetch.py`（push geometry/scratch）、`forward.py`/`function.py`/`_single_saved_adapter.py`（前向 moonep 侧配套）、`test_moe_suite.py`（w2/w8 autograd 用例）、`test_debug_moonep_saved.py`（探针+PROBE_GRAD_DEBUG 诊断）、`docs/`（本文 + dual-node-adaptation）、`extra-info/`（诊断产物，1.1MB）。

**framework（分支 `moonep`）**：`megamoe_ep_dispatcher.py`（combo 默认）、`finetune_kimik3.sh`（transport envs）、`kimik3_config.yaml`。

## 10. 迁移到新主机的最小步骤

1. 两仓 clone 对应分支（远端见文首）；mmt 安装到 venv（`pip install -e`，venv 路径可能不同，注意 `activate-moe.sh` 里的路径）。
2. 环境要求：950DT aarch64 + CANN/bishengir 工具链（与现机同版本——**版本漂移会重掷 mega kernel 误编译骰子**）；8 卡 HCCL/ASH 环境，`ASH_MASTER_PORT` 随机端口。
3. 从 §6 的 w2 门开始回归，再 w8，再整网。
