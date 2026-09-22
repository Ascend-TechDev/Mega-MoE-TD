# MMT 大kernel 双机(跨节点)适配方案

> 适用范围:mega_moe 算子仓(本仓)+ MindSpeed-MM 框架仓(Kimi-K3 FSDP2 路径)
> 目标硬件:2 节点 × 8 卡 Ascend950DT(98GB HDM)
> 软件环境:cann-shmem 1.6.0(`import shmem as ash`)、triton 3.5.0 + triton_dist overlay、torch 2.10.0 + torch_npu
> 参考上游:[Ascend/Triton-distributed-ascend#184 — cross-node support for put/get mem, allgather-gemm and reverse-all2all](https://gitcode.com/Ascend/Triton-distributed-ascend/pull/184)
> 状态:**阶段 1-3 已实施**(两仓 `dual-node` 分支,自 `release_v1.0`/`5d68e95` 拉出;2026-09-22)。实施中的两个方案变更见 §3.2a。G0 单机回归绿(w2 51.76s);G2+ 双机验证因对端暂不可用而挂起。

---

## 1. 背景与目标

单机 w2/w8 前后向已通。本方案给出双机(2×8,EP world_size=16)所需的全部适配:

- **算子侧**(本仓 `src/mega_moe/`):拆分"ACLSHMEM 全局 PE"与"本机 NPU 设备号"两个被混用的语义
- **框架侧**(framework 仓 `mindspeed_mm/fsdp/distributed/expert_parallel/megamoe_ep_dispatcher.py`):修复设备号假设、ACLSHMEM 建链引导、启动脚本与配置
- **测试与验证**:G0→G5 递进门禁,单机默认行为零变化

**硬约束**
1. 不设新环境变量时,所有行为与现状逐位一致(单机回归零变化)
2. 对称 slab 尺寸必须跨 rank MAX(尺寸发散 → 对称堆 offset 错位 → signal 写错位 → 死锁)
3. SET epoch 单调递增、不重置(残留更大值会提前满足 `dl.wait`,表现为静默错值)
4. 本方案不改任何 slab 尺寸计算逻辑、不改任何 epoch 计数逻辑——改动仅限"设备号拆分"与"引导/引擎可配"两个正交维度

---

## 2. 调研结论

### 2.1 上游 PR#184 做了什么(对照基准)

| # | 问题 | PR#184 做法 | mmt 是否需要 |
|---|------|-------------|--------------|
| 1 | 引擎硬编码 MTE | `data_op_engine_type` 读 `ASH_ENGINE` env(跨机用 UDMA/ROCE) | 需要(已有 `MOE_ASH_ENGINE` 开关,默认值需注意) |
| 2 | device_id 误用全局 rank | `aclshmem_create_tensor(device_id=LOCAL_RANK)` | **需要(核心改动)** |
| 3 | `init_process_group` 传 rank | 去掉 `rank=`,用 `env://` + 显式 `device_id=npu:{local}` | 需要(goldens 现在传 `rank=local_pe`,双机必错) |
| 4 | master 地址硬编码回环 | `ASH_MASTER_ADDR`/`ASH_MASTER_PORT` env | 需要(已有读取,默认 127.0.0.1) |
| 5 | RDMA 写覆盖本地填充 | `_reset_buffer` 加 `synchronize()+barrier()` | 不适用(mmt 无此形态) |
| 6 | barrier 残留值提前满足等待 | `signal_epoch` 从 `barrier_epoch` 续接 | 不适用(mmt 已用单调 SET epoch 防御) |
| 7 | signal 张量 dtype | int32→int64(`dl.notify` 要求) | 不适用(mmt 不用 `dl.notify`;signal=int32×16/slot 是既有 ABI,库侧 `signal_op/putmem_signal` 只收 int32 指针) |
| 8 | swizzle 变体 | `dist_swizzle2d`→`dist_swizzle2d_Nz` | 不适用(mmt 未使用 swizzle2d 系列) |
| 9 | API 拼写 | `aclshmem_finialize`→`aclshmem_finalize` | 不适用(mmt 已用正确拼写) |

### 2.2 跨机是否一定需要 UDMA

**结论:不是"一定 UDMA",但一定不能纯 MTE;跨机必须选一个能走网络的引擎(UDMA 或 ROCE),本机 950DT 推荐 `MTE|UDMA` 组合。**

证据链:
1. **MTE 不能跨机(权威)**:cann-shmem 头文件 `include/device/gm2gm/engine/shmem_device_mte.h:174`、`include/device/gm2gm/shmem_device_amo.h:189` 等十余处注释明确 *"The MTE transport for this operation does not support cross-PCIe (inter-node) communication"*。MTE 仅节点内搬数。
2. **引擎枚举是 4 值位掩码**:`shmem_common_types.h:78-84` — `MTE=0x01, SDMA=0x02, ROCE=0x04, UDMA=0x08`,可按位或组合。PR#184 注释亦写 *"Inter-node runs need a cross-node engine (UDMA/ROCE)"*。
3. **950 系列上 UDMA 是新主推引擎**(CANN SHMEM v1.6.0 "昇腾950系列 RDMA/UDMA 双引擎");ROCE 是传统跨机 RDMA 路径。
4. **组合模式**:`MTE|UDMA`(testkit `enable_udma` 路径)即节点内走 MTE、跨节点走 UDMA——单机已验证的 MTE 路径不用重验。
5. **正确性告警**:本机纯 UDMA 下 `getmem/putmem_signal` 曾出现数据损坏(见 `tests/_moe_testkit.py:105-111` 注释,源自 08-ascend-transpose-all2all 教程笔记)。**纯 UDMA 启用前必须重跑全套 golden;`MTE|UDMA` 是 MoonEP 单 kernel 前向测试已用配置。**
6. 附加:库侧 `dl.notify` 的 64 位 add 语义限 950 芯片放行(`is_compile_on_910_95`)——本机 950DT 满足,但 mmt 未用 `dl.notify`,不受影响。

### 2.3 算子侧单机假设根源

**kernel 形参 `LOCAL_RANK` 的实际语义 = ACLSHMEM 全局 PE(= `ep_group.rank()`,见 `src/mega_moe/ops/forward.py:106-112`),被同时兼任本机 NPU 设备号。** 单机 8 卡下两者恰好相等;双机 node1 的 PE 8-15 → `npu:8..15` 不存在,必崩。

PE 与设备号混用的全部落点:

| 类别 | 位置 |
|---|---|
| `aclshmem_create_tensor(device_id=<全局rank>)` | `src/mega_moe/runtime/workspace.py:151,160,178,200,209`(peer/routing_weight/signal/planning_counts/metadata_counts 5 处);`src/mega_moe/ops/forward.py:587,691`;`src/mega_moe/kernels/dispatch_fc2_bwd.py:606`;`src/mega_moe/kernels/mega_bwd.py:2371,2395,2415`;`src/mega_moe/runtime/replica_weight_prefetch.py:208-218,~L158(grad_push scratch);`tests/_moe_testkit.py:140,174-176`;`tests/fstage/test_mega_bwd_probes.py`(4 处) |
| `f"npu:{rank}"` 字符串 | `src/mega_moe/ops/backward.py:248,337`;`src/mega_moe/kernels/dispatch_fc2_bwd.py:71,164`;`src/mega_moe/kernels/combine_fc1_bwd.py:86,410,531`;`tests/_moe_testkit.py:137,172`;`tests/layer/test_moe_suite.py`(约 20 处);`benchmark/layer/profile_single_kernel_forward.py:301,309` |
| 反向 saved 链路 | `saved["ep_rank"]` 定义于 `ops/_native_saved.py:443`、`ops/_single_saved_adapter.py:373`,被 kernel 传参(PE,正确)与设备分配(错误)两用 |

**有利条件**:
- `runtime/workspace.py:121-127` 已强制 **EP 组 == 完整 ACLSHMEM world**(`ash.my_pe()==rank` 校验)→ 双机以全局 world 初始化时 PE 编址天然成立,只需拆设备号
- kernel 内 PE 编址/所有权判断(`peer != LOCAL_RANK`、`expert // EPN == LOCAL_RANK` 等)语义全部正确,不用动
- 仓内已有正确写法先例:`ops/_fc1_host_offload.py:174` 用 `torch.npu.current_device()`

**测试启动层的单机假设**:
- `conftest.py:20-32`:`mp.spawn` 启动,rank=spawn 索引兼任设备号;`MASTER_ADDR` 默认 localhost
- `bench_moe_suite.py:46` 明确 "do not wrap in torchrun"
- goldens(`_goldens/bigop_ref.py:153-155`、`_goldens/backward.py:291-293`)虽用 torchrun,但 `dist.init_process_group(rank=local_pe)` 把 LOCAL_RANK 当全局 rank 传——单机恰好相等,双机必错
- 参考写法:`3rdparty/bigop/ut/build_moe.py:63` 的 `set_device(rank % device_count())`

### 2.4 框架侧集成链路与阻塞点

集成链路:Kimi-K3 FSDP2 训练 → `dispatcher: megamoe` → `megamoe_ep_dispatcher.py`(774 行适配器)→ `mega_moe`(本仓,editable 安装,`MEGAMOE_REPO_ROOT` 校验)。

| # | 阻塞点 | 位置 |
|---|--------|------|
| 1 | ★`_get_peer_mem` 把 EP-group rank 当 NPU 设备号(`device=f"npu:{rank}"` + `aclshmem_create_tensor(device_id=rank)`),注释自认"单机 8 卡布局" | `megamoe_ep_dispatcher.py:322-356` |
| 2 | `_AclshmemRuntime.ensure` 硬编码 `OpEngineType.MTE`(MTE 不能跨机)+ `ASH_MASTER_ADDR` 默认 127.0.0.1 + 仅 ip_port 引导 | `megamoe_ep_dispatcher.py:173-224` |
| 3 | 启动脚本 `ASH_MASTER_PORT=$((RANDOM...))` 每节点随机 → 双机端口不一致;未导出 ASH_MASTER_ADDR | `examples/kimi_k3/finetune_kimik3.sh:34` |
| 4 | `NNODES=1 / NODE_RANK=0 / MASTER_ADDR=localhost` 单机参数 | `finetune_kimik3.sh:79-93` |
| 5 | `expert_parallel_size: 8` 需 16;EP=16 下对称堆 `MOE_FUSED_ASH_SIZE_GB=8` 需重估 | `examples/kimi_k3/kimik3_config.yaml:23`、dispatcher L536 |

**有利条件(无需改动)**:
- 框架多机通路已就绪:`trainer.py:140-168` 用 `LOCAL_RANK` set_device + `env://` init_process_group(多机正确);`parallel_state.py:78-84` EP mesh 是 DeviceMesh 最后一维,天然跨节点;README 已写明多机只改 `MASTER_ADDR/NNODES/NODE_RANK`
- 参照实现:同文件旁的 MoonEP 路径 `moonep/buffer.py:463-501`(`_init_shmem_runtime`)已是 **uniqueid 优先引导**(rank0 生成 → HCCL group 广播 → `aclshmem_init_using_unique_id`,天然跨节点),`buffer.py:508,552` 用 `torch.npu.current_device()`

### 2.5 本地库能力核对(cann-shmem 1.6.0 / triton_dist overlay)

- `OpEngineType{MTE,SDMA,ROCE,UDMA}` 四值齐全,位掩码可组合 ✅
- `InitAttr(my_rank/n_ranks/ip_port/local_mem_size/option_attr)` + `aclshmem_init` 新式 API ✅;uniqueid 引导入口存在(`shmem.core` / `aclshmem_init_using_unique_id`)✅
- `aclshmem_create_tensor(shape, dtype, device_id)`、`aclshmem_finalize` ✅
- `dist_swizzle2d_Nz`/`gemm_swizzle2d_Nz` 存在(overlay 追加,mmt 不用)✅
- 设备侧 signal 原语只收 int32 指针;mmt 的 signal 布局与之相容 ✅
- ⚠️ 已安装 triton_dist 无 pip 元数据、与本地 3.4.0 wheel 内容分叉(适配 triton 3.5.0 + Nz 追加);**误装 3.4.0 wheel 会回退 Nz 且 API 与 triton 3.5.0 不兼容**——双机复现性风险,建议留档当前 overlay

---

## 3. 改造设计

### 3.0 总体:单主开关 + 单一设备解析源

- **`MEGAMOE_MULTI_NODE` env**(默认 unset):
  - unset:所有行为与现状逐位一致(设备号=rank、引擎=MTE、引导=ip_port 127.0.0.1:8666、conftest 单节点 spawn)——总回退开关
  - `=1`:设备号取 `torch.npu.current_device()`、引导走 uniqueid(显式设了 `ASH_MASTER_ADDR/PORT` 则仍走 ipport)、引擎 `MTE|UDMA`、测试 spawn 带节点偏移
- **新模块 `src/mega_moe/runtime/device.py`**(唯一设备解析源):

```python
def resolve_local_device(pe_rank: int) -> int:
    """本地 NPU 序号(aclshmem_create_tensor 的 device_id / f"npu:{d}" 用)。
    优先级: MEGAMOE_LOCAL_DEVICE > (MEGAMOE_MULTI_NODE=1 -> torch.npu.current_device()) > pe_rank(现状)"""

def saved_device_id(saved: dict) -> int:
    """反向链路 saved dict 的本地设备号;旧 dict 回落到 ep_rank(兼容)。"""

def device_str(dev: int) -> str:
    """f"npu:{dev}" 统一收口(~30 处)。"""
```

- **kernel 形参 `LOCAL_RANK` 不重命名**(188 处/14 文件、constexpr、改名会失效 Triton 编译缓存、重跑全部 golden 基线、与上游 PR#184 冲突;收益仅可读性)。替代动作:`kernels/common.py` 顶部加命名约定注释——*"kernel 形参 LOCAL_RANK = ACLSHMEM 全局 PE,与 torchrun 的 `$LOCAL_RANK` 无关,勿混用"*,并在 6 个 launch 大站加行内注释。

### 3.1 阶段 1:算子侧设备号拆分

**(1) context 构造链**
- `runtime/workspace.py`:`MoEForwardContext` 加 `local_device: Optional[int] = None`;`create_moe_forward_context(..., local_device: Optional[int] = None)` 新 kw-only 参数;L151/160/178/200/209 五处 `device_id=rank → device_id=dev`(`dev = rank if local_device is None else local_device`)
- `ops/forward.py`:`FusedMoEForward.__init__` 加 `self.local_device = resolve_local_device(self.rank)`;传入 context(L159);L587/L691 两处 `device_id` 改 `self.local_device`;L414/424 replica buffer 透传;**L539/1271/1720 `LOCAL_RANK=self.rank` 不动(PE 语义正确)**

**(2) 反向 saved 链路**
- `ops/_native_saved.py::assemble_native_saved`(L437-454)与 `ops/_single_saved_adapter.py::enrich_single_kernel_saved`(L367-383):新增 `local_device=int(getattr(op, "local_device", op.rank))`,与 `ep_rank` 并列;`ep_rank` 保持 PE 语义
- 消费端改走 `saved_device_id(saved)`:`ops/backward.py:248,337`;`kernels/dispatch_fc2_bwd.py:71,164,611`;`kernels/combine_fc1_bwd.py:86,410,531`;`kernels/mega_bwd.py` 的 `_ensure_mega_combine_buf`/`_ensure_mega_redispatch_buf`/`_ensure_mega_signal_local` 3 处
- kernel launch 的 `LOCAL_RANK=saved["ep_rank"]` 全部不动

**(3) replica 链路**
- `runtime/replica_weight_prefetch.py`:`ReplicaWeightBuffers` 加 `local_device` 字段(覆盖 `ensure_grad_push_scratch` 的 `_alloc`);`allocate/acquire_replica_weight_buffers(..., local_device=None)` 透传;池 key 不含 local_device(同进程内恒定)

**(4) 测试/benchmark 侧**
- `tests/_moe_testkit.py` 的 `make_peer_mem`/`make_moonep_backward_peer_mem` 内部 `dev = resolve_local_device(rank)`(签名不变)
- `tests/fstage/test_mega_bwd_probes.py`(~8 处)、`tests/layer/test_moe_suite.py`(~20 处)、`benchmark/layer/profile_single_kernel_forward.py:301,309` 机械替换;kernel launch 的 `LOCAL_RANK=rank`(PE)不动

### 3.2a 实施记录(2026-09-22,dual-node 分支):两个方案变更

1. **引导:uniqueid → ip_port(必改)**。实施时核实 python API:`aclshmem_init_using_unique_id(mype, npes, mem_size, uid)` **没有 attr 参数**,内部固定默认引擎 MTE——无法带 `MTE|UDMA` 掩码,即无法跨节点(`aclshmemx_set_attr_uniqueid_args`/`aclshmemx_init_attr` 存在于 C 库但未导出到 python)。因此双机主路径 = **ip_port 引导 + 两节点显式一致的 `ASH_MASTER_ADDR=<node0-IP>` + 固定 `ASH_MASTER_PORT`**,与上游 PR#184 的跨机做法一致。防御:`MEGAMOE_MULTI_NODE=1` 且 `ASH_MASTER_ADDR` 为回环时,testkit 与框架 dispatcher 均**快速报错**(否则表现为首个跨节点 kernel 内挂死)。
2. **设备拆分落点比 §2.3 清单更广**。除表列 30 处外,实测还有:`tests/_moe_baselines.py`、`tests/layer/test_{single_kernel_moonep,fwd_phase_timing,debug_moonep_saved}.py`、`tests/fstage/test_f0b_probes.py`(4 处)、`benchmark/layer/bench_moe_suite.py`(5 处)——已全部经 `mega_moe.runtime.device` 收口。`kernels/fc2_combine.py` 的 `local_rank` 形参是 PE 语义(`< world_size` 校验),**不改**。

其余按本方案落地:`src/mega_moe/runtime/device.py`(唯一解析源,`MEGAMOE_LOCAL_DEVICE > MULTI_NODE=1→current_device > PE`)、conftest `MMT_NNODES/MMT_NODE_RANK` 偏移(`MASTER_ADDR/PORT` 双机必须显式一致,parent 侧校验)、goldens `RANK/LOCAL_RANK` 拆分、`kernels/common.py` LOCAL_RANK 命名契约注释、框架 `_local_device`/ensure 守卫/heap 预警/会话日志、`finetune_kimik3.sh` 拓扑参数化 + `kimik3_config_2n.yaml`(EP=16)。

### 3.2 阶段 2:初始化/引导 + 引擎

**引导选型:uniqueid 经 HCCL 广播(推荐),ip_port env 为显式覆盖**

理由:
1. 免配置一致性:finetune 脚本现行 `ASH_MASTER_PORT` 每节点随机,双机端口不一致是现行 bug;uniqueid 不占预定义端口,免疫此类冲突
2. 仓内已验证:同版本 cann-shmem 上 `moonep/buffer.py:463-501` 的 `aclshmem_get_unique_id → _broadcast_bytes(L510-522) → aclshmem_init_using_unique_id` 全链路可用;`_group_device`(L503-508)用 `torch.npu.current_device()` 建广播 device,双机正确
3. 与约束自洽:EP 组==world,`my_rank` 传 `dist.get_rank(ep_group)` 即全局 PE;幂等闸 `aclshmemx_init_status()` 已存在
4. 显式设 `ASH_MASTER_ADDR/PORT` 仍走 `InitAttr+ip_port`(PR#184 风格,兼作 uniqueid 异常时的回退)

算子侧 testkit 改造(`tests/_moe_testkit.py::init_aclshmem(..., bootstrap=None)`):
- `bootstrap=None` 时自动推导:显式传了 `ip_port` 或设了 `ASH_MASTER_ADDR` → ipport(现状);否则 `MEGAMOE_MULTI_NODE=1` → uniqueid
- uniqueid 路径:rank==0 `ash.aclshmem_get_unique_id()` 取字节串 → `dist.group.WORLD` 长度前缀+uint8 广播(照抄 moonep `_broadcast_bytes`)→ `aclshmem_init_using_unique_id(rank, world_size, size_bytes, uid_bytes)`;要求先 `init_process_group`(现有测试顺序已满足)
- 引擎保持三档:`MOE_ASH_ENGINE=udma` 纯 UDMA / `enable_udma` → `MTE|UDMA` / 默认 MTE

**引擎门禁顺序**:单机 MTE 回归 → 单机 `MTE|UDMA`(既有 single-kernel 测试已覆盖)→ 双机 `MTE|UDMA` golden →(可选)纯 UDMA golden 全过后才可信 → 性能对比。ROCE 仅当 UDMA 跨机链路不可用时预研。

### 3.3 阶段 3:框架侧

**(1) `megamoe_ep_dispatcher.py`**
- `_get_peer_mem`(L322-356):`dev = torch.npu.current_device() if MEGAMOE_MULTI_NODE=1 else rank`;`device=f"npu:{dev}"`、`aclshmem_create_tensor(device_id=dev)`;更新"单机8卡布局"注释
- `_AclshmemRuntime.ensure`(L173-224):引导——multi-node 且未显式设 ASH 端点 → uniqueid 广播(照抄 moonep);否则维持 InitAttr+tcp;引擎——默认 MTE 不变,`MEGAMOE_MULTI_NODE=1 → MTE|UDMA`,`MOE_ASH_ENGINE=udma` 显式覆盖;幂等闸/barrier/heap 读取保持
- heap 重估:W=16 时主导项 peer_mem 系随 `receive_capacity_factor`(默认=W)翻倍(signal/metadata 尺寸 W 不变、replica 表减半)→ `MOE_FUSED_ASH_SIZE_GB 8→16`;**勿压 `megamoe_receive_capacity_factor` 换 heap**(cf<W 是静默溢出→前向越界写→单 rank 消失型挂死);加启动期 heap 预测日志(warning only)
- 顺带:`_assert_efsdp_one` 报错文案按 world_size 措辞

**(2) 启动脚本与配置**
- `finetune_kimik3.sh`:`NNODES/NODE_RANK` 参数化(默认 1/0);`NNODES>1` 时 `export MEGAMOE_MULTI_NODE=1`、`MOE_FUSED_ASH_SIZE_GB=16`、`MASTER_ADDR` 必须显式(node0 IP);ASH_MASTER_PORT 随机化仅在单机分支保留;`HCCL_CONNECT_TIMEOUT=1200` 保留
- 新建 `examples/kimi_k3/kimik3_config_2n.yaml`:`expert_parallel_size: 16`(不动单机默认);确认 `num_experts % 16 == 0`、EFSDP=1(megamoe 硬约束)
- 测试侧超时:`DIST_TEST_TIMEOUT_S` 双机建议 3600

---

## 4. 测试与验证

### 4.1 mmt 套件双机跑法:保留 mp.spawn + 节点偏移(不切 torchrun)

理由:conftest 的错误队列/超时/僵尸清理机制对定位"单 rank 消失"至关重要且每节点独立;`bench_moe_suite.py` 明确不兼容 torchrun。

- `conftest.py::_worker_wrapper(local_i, global_world, ..., nproc_per_node, node_rank)`:`rank = node_rank*nproc_per_node + local_i`(全局 PE);`torch.npu.set_device(local_i)`;`init_process_group(rank=rank, world_size=global_world)`
- `run_dist_test`:读 `MMT_NNODES`(默认 1)/`MMT_NODE_RANK`(默认 0)→ `nprocs = world_size // nnodes`;默认路径与现状逐位一致
- 双机运行 = 两节点各起一次 pytest,共享:`MASTER_ADDR=<node0-IP>`、相同 `MASTER_PORT`、`MMT_NNODES=2`、`MMT_NODE_RANK=0/1`、`MEGAMOE_MULTI_NODE=1`
- G2(2×2,W4)可直接复制的命令(两节点均在 mmt 仓根;`<node0-ip>` 替换;`w2` 门形状即 world=4 用例按需换名):
  ```bash
  source <venv>/activate-moe.sh
  export MMT_NNODES=2 MMT_NODE_RANK=<0|1> MEGAMOE_MULTI_NODE=1 \
         MASTER_ADDR=<node0-ip> MASTER_PORT=29511 ASH_MASTER_ADDR=<node0-ip> \
         ASH_MASTER_PORT=41888 DIST_TEST_TIMEOUT_S=3600 MOE_FUSED_ASH_SIZE_GB=2 \
         TRITON_CACHE_DIR=/tmp/triton-moe-g2-$MMT_NODE_RANK \
         MOE_MEGA_GRAD_TRANSPORT=udma MOE_MEGA_REPREFETCH_TRANSPORT=udma MOE_MEGA_HEAP_PROBE=1
  python -m pytest "tests/layer/test_moe_suite.py::test_single_kernel_moonep_autograd_w2" -x -q -s
  ```
  (引擎在 MULTI_NODE=1 下自动 `MTE|UDMA`,无需 MOE_ASH_ENGINE;先 `ping <node0-ip>` 两网段各一次确认路由。)
- goldens 修复(`_goldens/bigop_ref.py`、`backward.py` 的 `_main`):`rank=int(os.environ["RANK"])`、`set_device(LOCAL_RANK)`、`init_process_group(rank=rank)`、`device=f"npu:{local}"`

### 4.2 递进门禁 G0→G5

每步强制 `MOE_MEGA_HEAP_PROBE=1` 审计对称 slab 偏移全 rank 一致(这是"slab 尺寸跨 rank MAX"纪律的直接观测器)。

| 步骤 | 配置 | 通过门禁 |
|---|---|---|
| G0 | 单机 W2/W8,**无新 env** | 全量套件全绿 + heap probe 偏移与改前逐位一致(**单机零变化证明**) |
| G1 | 单机 W8,`MEGAMOE_MULTI_NODE=1` | 全绿(current_device 路径等价性) |
| G2 | **2 节点×2 卡(W4)**,`MTE\|UDMA`,uniqueid | golden 数值对齐;偏移全 rank 一致;无挂死(2×2 已覆盖 PE≥设备数的全部语义,失败面最小) |
| G3 | 2×4(W8) | + `bench_moe_suite`;耗时与单机 W8 同形对比记录 |
| G4 | 2×8(W16) 全量 + MoonEP 路径(`megamoe_enable_moonep`) | + epoch 单调性用例;`MEGAMOE_SINGLE_KERNEL=0` 的 W16 编译 smoke(仅记录) |
| G5 | 框架 2×8 冒烟 | finetune 2 iters→10 iters;loss/grad norm 正常;16 rank 偏移一致;`MEGAMOE_DEBUG=1` 路由跨节点占比≈50% |

框架冒烟步骤:
1. 双节点准备:同 venv、同 `MEGAMOE_REPO_ROOT`、node0 IP 互通、每节点 `HCCL_NPU_SOCKET_PORT_RANGE` 段错开或一致策略确认
2. node0:`NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> bash finetune_kimik3.sh`;node1 同命令 `NODE_RANK=1`
3. 观察顺序:HCCL 16 rank 建网 → `_AclshmemRuntime.ensure` uniqueid 广播日志 → heap 预测日志 → heap probe 偏移一致 → 2 iters loss/grad norm → 10 iters 计时

### 4.3 fc2_combine static_range 风险:不需要为 Kimi-K3 处理

已核实:Kimi 路径(`MEGAMOE_SINGLE_KERNEL=1` 前向 `fused_forward.py` + `MOE_BWD_MEGA=1` 后向 `mega_bwd.py`)对 W 全部动态 `range`,无 static_range;`fc2_combine.py:410` 的 `static_range(0, WORLD_SIZE)` 仅在分段路径(SINGLE_KERNEL=0)使用,W=16 为 16 次线性展开无爆炸。应急项:若 W16 编译超预算,`static_range→range` 局部降级。`balanced_routing.py:216/333` 仅记录。

---

## 5. 风险清单与回退

| 风险 | 特征/定位手段 | 回退 |
|---|---|---|
| 引导失败(aclshmem_init 挂/报错) | 挂在 ensure 内、无 kernel 日志;`ASCEND_SLOG_PRINT_TO_STDOUT=1` 看 aclshmem 日志;先验 HCCL broadcast 是否完成 | 显式 export `ASH_MASTER_ADDR/PORT` 走 ipport 覆盖;仍失败则 `NNODES=1` |
| **signal 错位死锁**(对称 slab 尺寸发散) | 一 rank 在 kernel 内 spin、其余 rank 卡 barrier;**先看 heap probe 偏移是否全 rank 一致**;再查两节点 yaml/env 是否同步 | 修配置;`MOE_BENCH_BWD_WARMUP/ITERS=1` + 分段计时二分 |
| epoch 语义破坏(重置/重叠) | 表现为**静默错值**而非挂死;跑 fstage probes 的 epoch 单调用例 | 本方案不触碰 epoch 计数任何代码 |
| 跨节点 UDMA 数据损坏 | golden 数值错/NaN 而非挂死;单机 `MTE\|UDMA` vs 双机对比二分;dmesg 查设备 SMMU 错误 | 纯 UDMA 实验回 `MTE\|UDMA`;双机无 MTE-only 回退(MTE 不过节点),只能回单机 |
| W16 heap 不足 | `aclshmem_create_tensor` OOM/分配失败(显式,易定位) | `MOE_FUSED_ASH_SIZE_GB` 16→24;或临时压 `megamoe_max_tokens_per_rank`(勿压 cf) |
| HCCL 16 rank 建网超时 | 卡在建网、日志 EI0020/超时 | `HCCL_CONNECT_TIMEOUT`;每节点独立 `SOCKET_PORT_RANGE` 段 |
| 分段路径 W16 编译膨胀 | 首次 launch 前长编译 | `static_range→range` 局部降级(应急) |
| triton_dist overlay 被误替换 | import 报错/Nz 缺失/API 不兼容 | 留档当前 site-packages overlay,勿装 3.4.0 wheel |

**总回退开关**:`MEGAMOE_MULTI_NODE` unset + `NNODES=1` → 算子、testkit、conftest、dispatcher、脚本全部回到现行单机行为(G0 门禁强制验收)。

---

## 6. 关键文件清单

**算子仓(mmt)**
- 新建:`src/mega_moe/runtime/device.py`
- 修改:`src/mega_moe/runtime/workspace.py`、`src/mega_moe/ops/forward.py`、`src/mega_moe/ops/_native_saved.py`、`src/mega_moe/ops/_single_saved_adapter.py`、`src/mega_moe/ops/backward.py`、`src/mega_moe/kernels/{mega_bwd,dispatch_fc2_bwd,combine_fc1_bwd,common}.py`、`src/mega_moe/runtime/replica_weight_prefetch.py`、`tests/_moe_testkit.py`、`conftest.py`、`src/mega_moe/_goldens/{bigop_ref,backward}.py`、`tests/fstage/test_mega_bwd_probes.py`、`tests/layer/test_moe_suite.py`、`benchmark/layer/profile_single_kernel_forward.py`

**框架仓(framework)**
- 修改:`mindspeed_mm/fsdp/distributed/expert_parallel/megamoe_ep_dispatcher.py`、`examples/kimi_k3/finetune_kimik3.sh`
- 新建:`examples/kimi_k3/kimik3_config_2n.yaml`

**实施顺序**:阶段 1(设备号)→ G0/G1 → 阶段 2(引导/引擎)→ G2/G3 → 阶段 3(框架)→ G4/G5 → 收尾(文档、静态审计、可选纯 UDMA 实验)。每阶段独立可验证、可合并。
