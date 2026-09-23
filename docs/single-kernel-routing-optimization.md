# 单 kernel routing 与 wave Scalar 优化

## 范围与限制

修改保持单次 launch、send/recv 布局、稳定 route 顺序、每 core 的 GEMM tile 顺序和现有 phase barrier。不改 MoonEP scatter，不增加配置开关。

新增普通 GM 的紧凑 wave 任务表，并在 pipeline signal slab 尾部增加本地 return-checker epoch slots。host 分配、释放和 kernel 参数已一起更新。FC1、FC2、dispatch、return 改为消费紧凑表；reduce 前改为分片 acquire 后发布本地完成信号。所有原先的 remote return counter 仍被逐波 acquire，不使用“仅等最后 wave”的推断。

本地没有 torch、Triton 或 NPU。Host 测试只能验证整数计算、任务/信号覆盖和调用接口，不能证明 Ascend lowering、UB 分配、跨引擎调度或通信可见性。设备编译、正确性和耗时均待 NPU 验收；README 中历史 routing 10.0 → 3.4 ms 不是本次结果。

## Routing metadata 与稳定 scatter

稳定 send row 的公式为：

```text
send_row(r) = sum(count[e'] for e' < expert[r])
            + sum(core_count[c', expert[r]] for c' < core[r])
            + count(r' < r, same core, same expert)
```

- stable cursor 改为 `[cores, expert_block]` exclusive scan。
- destination 与 wave offsets 复用 source counts 归约。
- pull starts 改向量归约/前缀。
- 去掉 histogram 预清零，将逆映射初始化融合进 histogram 遍历，向量化 signal counter reset。
- E≥128 的非 MoonEP scatter 使用 32-route block、32×32 两两匹配求块内序号，再用 histogram 更新块间 cursor。两个 Vector lane 各拥有互斥 expert 半区；E<128 保留原 dense 路径。

完整 Kimi E=896、R=65536 时，expert-ID 扫描由 28 遍减少到两个 lane 各 1 遍。pairwise 匹配累计约 419 万元素，原 dense 匹配约 5872 万元素。新路径也有 histogram/gather/循环开销，不能把元素数比例当作加速比。128 的分支阈值未做 NPU 调优。

无效 route 的逆映射保持 `-1`；`send_route_indices` 必须是有效 route 的 expert-major 稳定排序；`send_token_indices = send_route_indices // TOPK`；`route_to_send` 是其逆映射。

## FC1 / FC2 入口：紧凑 wave 任务表

此前 FC1、FC2 每个 wave 都遍历全部本地 expert，读表计算区间后才跳过空任务。完整 Kimi 的均匀分布例子中，112 experts、336 个 M tiles、21 waves，每 core 每 FC 做 2352 次 expert 探测，但只有 126 个非空 wave/expert 组合。

现由 `_build_wave_tasks` 在 routing 阶段按 rank 构建：

- `wave_task_offsets[rank, wave]`：每 wave 的任务起点，含 sentinel；未使用尾部填为总任务数。
- `wave_tasks[rank, task, 5]`：`expert, expert_offset, begin, end, first_block`。

非空任务数不超过 `非空 experts + waves - 1`；每 rank 预留 `MAX_WAVES + physical_experts` 个任务，无需稠密的 `waves × experts` 表。生成顺序依旧是 wave-major、expert-major，保持旧调度顺序。builder 对 expert 轴向量化，跨 wave 的 part 循环只用于真正跨多个 wave 的 expert。容量溢出的 rank 不写任务表，现有全局容量判断让所有 rank 跳过 pipeline，避免写越界。

FC1/FC2、dispatch、return 都消费同一表，避免各自重建区间。FC2 在 activation wait 前读取任务区间，仍保留全核 activation acquire。

FC1 另外：

- 直接接收任务表中的 expert offset，不再重新读取 `recv_expert_offs`。
- first-sweep bound 和 input-group offset 外提。
- tile 的运行时除余改为初始化一次 quotient/remainder，再逐步递增 row/n tile；Cube 与 Vector 的一步错位保持不变。FULL_GROUP 的 row parts 显式取 constexpr GROUP_WINDOWS。

FC1 的 dispatch source 区间二分和就绪 wait 仍保留；本次没有增加 M-row-tile readiness 表或改 GEMM/CV 握手。FC2 的 constexpr 除余、GEMM 内部不变量仍由编译器处理。

### 内存与成本

每 rank 的任务存储为 `world × (MAX_WAVES + EPR) × 5 × 4` 字节，offset 存储为 `world × (MAX_WAVES + 1) × 4` 字节。表每次随路由重建，不依赖固定均匀分布。任务构建计入现有 `stable_cursors` timing 段；必须同时比较 routing 与 wave_pipeline，避免只把时间前移。

## reduce 前 Vector 尾部：分片 return 检查

原先每个 Vector lane 都遍历全部 `(destination, wave)` 并 acquire return counter。W=8、各 21 waves、32 cores 时共 10752 次 remote-counter wait 调用。

现在由 `min(2×cores, world)` 个 checker 按 destination 分片：

1. checker 逐波 acquire 自己负责的所有原始 return counters，expected ADD 数仍由原 return-unit 分配推导。
2. 将 acquire token 绑定到原 signal allocation base，经 fence 后向本地 checker slot 发布当前 `signal_epoch`。
3. 每条 reducer lane acquire 整个 checker epoch slab，所得 token 继续传给原 `consume_token(combine_buf)`。

上述 W8 示例中，原 remote counters 共 acquire 168 次；随后每 lane 一次覆盖 8 个 slots 的本地 wait 调用。后端仍可能逐 slot 检查，不能把调用数减少直接当作耗时倍数。远端 FC2/return 的真实晚到延迟也不会因此消失。

新增 slots 接在原 activation/FC2/return slabs 后，不覆盖旧 counters。初始化 reset 同步扩容；epoch 沿用现有每次 forward 递增机制。零波 destination 的 checker 也发布完成；W>2×cores 时一个 checker 负责多个 destination。所有 dispatch 保持在 return waits 之前，避免引入跨 rank 依赖环。

这条 acquire→fence/发布→acquire 接力需要 NPU 验证跨核、跨 rank 可见性及编译后的 consume-token 依赖。Host 模型只检查覆盖和协议结构，不模拟实际缓存、乱序或异步通信。

## 本地验收

2026-09-22：下列完整 host 回归 **375 passed**；Python 语法检查与 `git diff --check` 通过。NPU 编译、correctness 和性能未运行。

测试直接 AST 提取生产 helpers，使用带越界、重复写和 lane 互斥检查的 NumPy shim 执行。覆盖：

- metadata/scatter 的空 core、尾块、非法 route、偏斜、E=1/8/32/33/64/127/128/129/896、完整 Kimi 65536 routes、W=128 与 MoonEP metadata。
- 稠密旧 wave 扫描与紧凑表逐项对齐；EPR=1/4/7/14/33/112，windows=1/4/16/64，全空、热 expert、零行空洞和重复调用。
- 任务容量上界与容量溢出不写表。
- FC1 递增 tile 序列与原除余公式完全一致；实际 FC2 helper 的逐 core tile 顺序与旧调度一致。
- 每个 remote return counter 恰好一个 checker acquire；所有 reducer 等待全部 checker；缺失任意早期 wave 信号时不能发布该 checker 或放行 reduce；旧 epoch 不放行。
- JIT helper 参数完整绑定、未定义名称检查及 host launch 的新 workspace 参数位置检查。

运行：

```bash
python3 -m pytest --noconftest -q \
  tests/function/test_single_kernel_routing_metadata.py \
  tests/function/test_single_kernel_stable_scatter.py \
  tests/function/test_single_kernel_wave_tasks.py \
  tests/function/test_jit_call_binding.py \
  tests/function/test_single_kernel_wide_world.py \
  tests/function/test_fwd_phase_timing.py \
  tests/function/test_pipe_profile_summary.py \
  tests/function/test_npu_occupancy.py
```

设备新增节点只做语法检查，未在本机执行。源码减少除余不保证后端按预期降低 Scalar 开销，仍要对照编译 IR 和 trace。

## NPU 验收入口

先做 compile-only，分别覆盖默认、timing/saved，以及 MoonEP 和宽 world。输出目录须不存在或为空：

```bash
python benchmark/layer/compile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-w8-t4k \
  --save-fc1 --save-dtype fp16 --timing \
  --output-dir /tmp/routing-full-w8-fp16-timing

python benchmark/layer/compile_single_kernel_forward.py \
  --case performance-fwd-kimi-k3-wide-w128-t4k --moonep \
  --output-dir /tmp/routing-wide-w128-moonep
```

正确性：

```bash
python -m pytest tests/layer/test_moe_suite.py \
  -k 'single_kernel_forward or single_kernel_dynamic_waves or single_kernel_routing_large_experts' \
  -v -s
python -m pytest tests/layer/test_single_kernel_moonep.py -v -s
```

新增 E128/E896 节点用小 hidden/FFN、tokens=2051 覆盖大专家 scatter。复用 worker 包含 dropped routes、全丢弃、集中 expert 0、连续调用、saved/FP8/timing、容量溢出；saved 子例逐项核对 send/逆映射。W128 需要对应多机环境另验。

性能：

```bash
MOE_FUSED_ASH_SIZE_GB=6 python -m pytest tests/layer/test_fwd_phase_timing.py \
  -k 'performance-fwd-kimi-k3-w8-t4k and fp16 and not unsaved' -m dist -v -s
```

基线与修改版用相同 case、saved 格式、warmup/samples。完整与 trimmed 分开测。记录各 routing 段原始 ticks、wave_pipeline、return_wait、事件钟 e2e，并看首次 Cube 发射和 Scalar 空洞。`fc1_cube_wall`、`fc2_wave_wall` 都包含等待，不是纯 GEMM 或纯 Scalar 耗时。
