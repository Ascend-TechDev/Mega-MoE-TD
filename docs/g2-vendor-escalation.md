# G2 双机 ACLSHMEM 跨节点传输缺陷 — vendor 升级材料

**日期**: 2026-09-23/24 · **发现轮次**: G2 r14-r32 · **状态**: 矩阵全闭合，双侧互证

## 1. 一句话摘要

两台 Atlas 950 SuperPoD 服务器（superpod_16383 的 srv_24/srv_28）之间，cann_shmem 的
**bulk 数据腿每个引擎只有一个方向能通（对向静默丢弃、无任何日志/错误）**，**跨节点 getmem
直接 SIGABRT（device aicore error 271）**，**混引擎堆在 init 阶段死锁超时**；同一会话内
signal 腿（控制面）双向全通——证明链路与引导层健康，故障锁定在引擎数据路径。

## 2. 环境

| 项 | node0 | node1 |
|---|---|---|
| 角色 | srv_24, 141.61.95.70, rank0 | srv_28, 141.61.95.30, rank1 |
| SuperPoD | superpod_16383（UB fabric, df30↔df34 已核实） | 同 |
| 芯片/固件 | Atlas 950 V120 / fw 9.1.13.0.b130 | 同型号（对端采集见附录） |
| CANN toolkit | 26.2.0.b007（version.cfg 串；安装目录名 cann-9.2.0-beta.2，同一 toolkit 双版本串，双侧一致） | 同 |
| cann_shmem wheel | 1.6.0 cp311，双侧统一安装，libshmem.so md5 `e3d35af94b238c49aa0a77a554d9cb40` | 同 md5 |
| python/torch | python 3.11.10 / torch 2.10.0 | 同 |
| OS | openEuler 24.03 LTS-SP3, kernel 6.6.0-145.3.29.160 | 同 |
| /etc/hccl_rootinfo.json | 官方生成器（unofficial-ascend-tools 0.0.7rc2）产物的每机自述形态，**本轮全程未改动**（md5 前后自证） | 同 |

## 3. 行为矩阵（r27d-r32，7 轮、双侧判据行互证）

对称堆 64×int64，MAGIC=0x5A5A+rank；每格至少一次双机同窗真联合执行。

| 操作 | n1→n0 | n0→n1 | 判据轮次 |
|---|---|---|---|
| put_signal bulk 腿 / **udma** | ✅ 落地 | ❌ **静默丢**（数据停留本地 fill 值，无错误无日志，DEBUG 级 plog 335 行零 op 记录） | r27d, r29A(combo 同形), r32b |
| put_signal bulk 腿 / **mte** | ❌ 静默丢 | ✅ 落地 | r27f |
| put_signal bulk 腿 / **combo**(MTE\|UDMA) | ✅ | ❌（=udma 形；**无 op 级按方向选引擎行为**） | r29A |
| put_signal **signal 腿**（任意引擎） | ✅ | ✅ | r27d/r27f/r28/r29A 四轮确认 |
| **getmem**（node0 发起） | — | ❌ **SIGABRT** ERR02005 DIST internal error | r29B |
| **getmem**（node1 发起） | ❌ SIGABRT + **device aicore error 271** | — | r31 |
| 混引擎堆（n0=mte / n1=udma） | init 120s 超时双侧死（见 §5） | 同 | r30 |
| **signal 腿连续 256×SIGNAL_SET 载荷** | ✅ 逐字落地 | ✅ 逐字落地（bad_slots=0） | r32/r32b（**全绿**，6µs/op） |

角色互换判别（r28）：rank/role/master/channel-client 四变量全翻，断向**跟机器走不跟角色走**
——每台机器恰好一个可用的 push 发起方向（node1 发起走 udma、node0 发起走 mte）。

## 4. 可复现用例（`scripts/g2_vendor_case/`，无仓依赖，每个 ~15s）

两侧同时执行（node0 先起、node1 合流）：

```
node0$ ./run_node0.sh push          # 用例1: 非对称静默丢
node1$ ./run_node1.sh push
node0$ ./run_node0.sh pull          # 用例2: getmem SIGABRT + aicore 271
node1$ ./run_node1.sh pull
node0$ ./run_node0.sh tunnel        # 用例3(对照): signal 腿双向载载荷 — 全绿
node1$ ./run_node1.sh tunnel
```

**用例1 期望输出**（两侧各一行判据）：
```
[push node0] data[0]=0x5a5b (peer magic 0x5a5b) -> peer bulk LANDED;  sig=1 (control plane OK)
[push node1] data[0]=0x5a5b (peer magic 0x5a5a) -> peer bulk SILENTLY DROPPED; sig=1 (control plane OK)
[push] VERDICT: n1->n0=LANDED n0->n1=DROPPED — BUG REPRODUCED ...
```
`MOE_ASH_ENGINE=mte` 时方向镜像（n0→n1 落地、n1→n0 丢）。

**用例2 期望输出**：node1 worker SIGABRT（`ERR02005 DIST internal error`），device plog 出现
`aicore error, error code = 271: "The address for scalar to access the internal buffer is out of bounds"`
（aic 拷贝任务 task_recycle，见附录 plog）；node0 挂在 barrier 直到 torchrun 收割。

**用例3 期望输出**：双侧 `bad_slots=0/256`，吞吐 ~6µs/op——同机同会话同堆，控制面路径
双向健康，故障被隔离在 bulk/pull 引擎路径。

## 5. 证据链（plog 索引）

| 轮次 | 证据 | 文件 |
|---|---|---|
| r31 | **device aicore error 271 完整栈** + ERR02005（node1 发起 getmem 崩） | `plogs-node1/plog-1616143_20260923220917605.log` |
| r31 | shmem 侧 init 全绿→finalize（证崩在 op 不在 init） | `plogs-node1/aclshmem_1616143_20260923220917.log` |
| r30 | 混堆 init 死：双侧同键 `SHM_(0)_S_0_1_GW` AllGather 120s 互空（node0 链：reserve_heap 成功→setup_heap -3 @shmem_init.cpp:652；node1 链：eidSlotCount 级联→reserve_heap -3） | node0 `aclshmem_4163467_20260923214919.log`; node1 `aclshmem_1605547_20260923215507.log` |
| r29B | node0 发起 getmem SIGABRT | node0 `aclshmem_4150513_20260923214109.log` |
| r28 | DEBUG 级 335 行 plog：channel/MR/WQCtx 全绿、**零 per-op 记录**（发送侧静默丢不可观测） | node0 `aclshmem_4134247_*.log` |
| r32/r32b | signal 腿隧道全绿判据行 + 6µs/op 吞吐 | 双侧任务输出（归档于 G2 channel 记录） |

时间线与逐步推理：G2 channel 记录（r14-r32，含 rootinfo 轨道关闭证明：官方生成器源码
rank_list 只从本地 /dev/davinci* 枚举，每机自述即官方形态，非配置错误）。

## 6. 诉求

1. **bulk 腿断向**：udma 引擎 node0→node1（及镜像 mte node1→node0）的 put_signal bulk 腿
   静默丢——要么修复，要么给出可诊断的错误返回/日志（当前 DEBUG 级都无痕迹）。
2. **getmem 跨节点崩溃**：两个发起方向都 SIGABRT（node1 发起附 aicore 271 寻径越界）——
   期望可用或明确报错，而非 device fault 打崩进程。
3. **混引擎堆 init**：MTE|UDMA 掩码两侧不一致时 GW 键交换（`SHM_(0)_S_0_1_GW`）120s 互空
   死锁——纯 MTE 堆不 post 该键是设计还是缺陷，请给出混引擎共存路径或明确不支持的定义。
4. （问题定性参考）signal 腿与 bootstrap/控制面双向全通，HCCL 集合通信跨机全绿
   （r16 旁证）——请聚焦 UDMA/MTE 数据面引擎的跨机路由/寻径。

## 7. 我方临时缓解（不影响上述诉求）

counts 级小载荷（1-4KB）可经 signal 腿逐字隧道（用例3，~6µs/op，2KB≈1.5ms）——
仅 host 级、低频交换可用；bulk/pull 修复前双机 MoE 数据面无法建立。
