# 双机 G2 联调 · 对端(node1)执行手册

- 日期: 2026-09-23 · 版本 v1
- 代码: mmt 仓 `dual-node` 分支 @ `7f58f21`（origin 已推）
- 角色: **node1 = 本机**（133-C01-B07-OS5 / 141.61.95.30）；**node0 = 对侧**（133-C01-B07-OS1 / 141.61.95.70，由 node0 侧负责启动）
- 联络: 所有人工转达。每步结果按 §6 模板回传，由 node0 侧决定下一步。

## 0. 背景（3 行）

Mega-MoE-TD 的双机适配（8+8 卡 EP16）已完成代码改造，按门禁递进验证。当前到 **G2 = 首次真双机**：
G2a（2 节点×1 卡，W2 最小冒烟）→ G2b（2 节点×2 卡，W4 设备号拆分语义）。跨机数据面 UDMA 已确认可用，
引导用 ip_port（两节点显式一致的 `ASH_MASTER_ADDR=<node0-IP>` + 固定端口）。方案全文见仓内
`docs/dual-node-adaptation.md`（如需背景再读，执行本手册不需要读它）。

## 1. 环境基线（已核实，勿改动）

| 项 | 状态 |
|---|---|
| NPU | 8×Ascend950DT 全 OK；空载基线 HBM ~4.7GB/卡（驱动占用，正常） |
| 驱动 / npu-smi | 26.2.0.b007，与 node0 一致 |
| venv | `/root/moe-venv/activate-moe.sh`（torch 2.10.0+cpu / torch_npu 2.10.0.post1.dev20260528 / cann-shmem 1.6.0 / triton 3.6.0+triton_dist，**已验证 import 全绿**） |
| CANN | activate 内 source `~/Ascend92/cann-9.2.0-beta.2`（勿用 /usr/local/Ascend 的 9.1.0） |
| 连通 | node1→node0:29511 TCP 已通；同网段 141.61.95.0/24 |
| mmt 仓 | `/home/t0303/project/mmt`，origin 同源，起点 release_v1.0@5d68e95 |
| framework 仓 | 已切 dual-node（G2 不用它，勿动） |

## 2. 纪律（先读，全部来自共享机器教训）

1. **跑任何 NPU 任务前**：`npu-smi info` 查进程表——任何卡 HBM >20GB 视为他人占用，等待勿抢。
2. **跑前清孤儿**：`pgrep -f pytest` 有残留就 `pkill -f pytest`（强杀的 worker 会留 NPU 上下文毒后续测试）。
3. `TRITON_CACHE_DIR` 按手册指定目录用，**绝不裸奔默认目录**。
4. **triton_dist 的 wheel 永不 pip install**（捆绑错误 triton），环境里已有的够用。
5. **禁止任何 git push**；禁止切到/改动 release 分支；禁止改 venv 和 CANN 安装。
6. 计时/结论只认 pytest 输出与日志，不凭感觉。

## 3. Step 1（立即可执行）：切分支 + 单机冒烟（顺带预热 G2 cache）

```bash
cd /home/t0303/project/mmt
git status --short          # 应干净；仅 "3rdparty/bigop" 一行 dirty 可忽略
# ↑ 若出现其他未提交改动：停止，原样回报，勿 reset
git fetch origin dual-node && git checkout dual-node && git reset --hard origin/dual-node
git log --oneline -1        # 必须是 7f58f21

# bigop 子模块可选（G2 用不到；只为 git status 干净）:
# git submodule update 3rdparty/bigop

pkill -f pytest 2>/dev/null; sleep 2
npu-smi info | grep -A1 "NPU ID" | head
source /root/moe-venv/activate-moe.sh
MOE_FUSED_ASH_SIZE_GB=2 DIST_TEST_TIMEOUT_S=3600 \
ASH_MASTER_PORT=$((40000+RANDOM%20000)) \
TRITON_CACHE_DIR=/tmp/triton-moe-g2-1 \
python -m pytest "tests/layer/test_moe_suite.py::test_single_kernel_moonep_autograd_w2" \
    -m dist -v 2>&1 | tee /tmp/g2_smoke_node1.log
```

判读：
- **首轮含冷编译，5–20 分钟正常**（`DIST_TEST_TIMEOUT_S=3600` 已放宽；日志若是
  `dist workers did not finish within ...s` 才是超时，真实 traceback 才是失败）。
- 预期 `1 passed`。此步同时在 `/tmp/triton-moe-g2-1` 预热了 G2a 要用的 kernel cache（形状/world 与 G2a 完全一致）。
- 冒烟红：按 §6 采集回报，**不要自行改代码**。

Step 1 绿 → 回传结果，然后**等人工信号**："node0 已起"。

## 4. Step 2 = G2a（双机 2×1，W2 冒烟）：收到"node0 已起"后立即执行

node0 会先起并等我们（rendezvous 窗口约 30 分钟，不用抢，但别拖太久）：

```bash
cd /home/t0303/project/mmt && source /root/moe-venv/activate-moe.sh
pkill -f pytest 2>/dev/null; sleep 2

export MMT_NNODES=2 MMT_NODE_RANK=1 MEGAMOE_MULTI_NODE=1 \
       MASTER_ADDR=141.61.95.70 MASTER_PORT=29511 \
       ASH_MASTER_ADDR=141.61.95.70 ASH_MASTER_PORT=41888 \
       DIST_TEST_TIMEOUT_S=3600 MOE_FUSED_ASH_SIZE_GB=2 \
       TRITON_CACHE_DIR=/tmp/triton-moe-g2-1 \
       MOE_MEGA_GRAD_TRANSPORT=udma MOE_MEGA_REPREFETCH_TRANSPORT=udma MOE_MEGA_HEAP_PROBE=1
python -m pytest "tests/layer/test_moe_suite.py::test_single_kernel_moonep_autograd_w2" \
    -x -q -s 2>&1 | tee /tmp/g2a_node1.log
```

判读：
- 本机 spawn 1 个 worker（全局 rank1，npu:0）。预期 `1 passed`，日志里 `[mega-heap r1 ...]` 与 node0 侧
  `r0` 的 **`peer_off/sig_off/comb_off/rfc2_off` 逐字段一致**（偏移发散=跨 rank 写错对端 slab，视为失败）。
- 挂在开头无编译输出 >5 分钟：大概率引导/建网问题，按 §6 采集（重点 `dmesg | tail -30`）。
- 出现 `EE9999/EZ9999`（aicore trap）：记录即可，等待 node0 侧判断重试。

## 5. Step 3 = G2b（双机 2×2，W4 设备号拆分）：收到"G2a 双边绿"后执行

同 §4 的 env **完全不变**，只换用例（一条命令两个 case）：

```bash
python -m pytest \
  "tests/layer/test_moe_suite.py::test_forward_suite[functional-fwd-s-w4-t128]" \
  "tests/layer/test_moe_suite.py::test_backward_suite[functional-bwd-small-h512-f256-k4-w4-t512]" \
  -x -q -s 2>&1 | tee /tmp/g2b_node1.log
```

判读：本机 spawn 2 个 worker（全局 rank2/3，npu:0/1），预期 `2 passed`。
G2b 绿 = **G2 门禁整体通过**。之后的 G3(2×4)/G4(2×8) 命令由 node0 侧另发，勿自行推进。

## 6. 每步回传模板（喂给转达人即可）

```
[步骤名] exit=?
pytest 结尾 30 行（或失败 traceback 全文）
grep -E "mega-heap" 日志 | head -20        # G2a/G2b 必附
npu-smi info | tail -20                    # 失败时必附
dmesg | tail -30                           # 仅挂死/trap 时附
```

## 7. 禁止事项汇总

- 禁止：git push、改 release 分支、pip install / 卸载任何包、改 venv 与 CANN、自行跑 G3+、
  自行"修复"红测（采集回报即可）。
- bigop 子模块不用装（G2 不依赖）。
