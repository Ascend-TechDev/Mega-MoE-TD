# Kimi K3 fused forward 清理与验收记录（2026-09-26）

> 本文记录本次代码清理、当前最优 fused forward 的硬件复验、源码指纹和遗留项。原始 12 点矩阵仍见 docs/performance/kimi-k3-forward-matrix-2026-09-26.md；本文单点是清理后复验，不覆盖那份 9/12 历史矩阵。

## 结论

- 当前源码在 Ascend950DT 8 卡、E896、每卡 4096 tokens、top-k=16、偏斜路由、MoonEP 关闭下完成 correctness、occupancy、50 次 fused 和 50 次 Torch 计时。
- fused：median 15.129 / mean 15.035 / P95 15.432 / range 14.110-15.542 ms；Torch grouped-GEMM + HCCL：median 21.740 / mean 21.880 / P95 22.139 / range 21.641-24.166 ms；median 加速比 **1.437x**。
- 与历史最佳统一口径结果（fused 15.095 ms、Torch 21.697 ms、1.437x）相比，本次复验为 15.129 ms、21.740 ms、1.437x，未见材料回退；一次复验不能证明每个删除项的独立收益。
- run artifact：/tmp/kimi_forward_cleanup_final_20260926/full_t4k_moonep0。原始 benchmark_result.json、逐卡区间、telemetry.jsonl、occupancy 日志和完整环境均保留。

## 测试输入与计时口径

- case=performance-fwd-kimi-k3-w8-t4k；model=KIMI-K3；world=8；tokens/rank=4096；global tokens=32768；hidden=3584；ffn=3072；E=896；top-k=16；capacity factor=1.25；drop_frac=0.0。
- 输入形状：hidden/output=[4096,3584]；routing indices/weights=[4096,16]；每 rank W1=[112,3584,6144]、W2=[112,3584,3072]；BF16 hidden/weights/output，FP32 routing weights，INT32 expert indices；SwiGLU。
- owner route quota=[19,27,11,11,15,15,15,15]/128；实际 routes_received_per_rank=[65672, 65348, 65442, 65024, 65821, 65733, 65716, 65532]；所有 896 个 global experts active。
- MoonEP=关闭；FC1 block=[128, 512, 128]；FC2 block=[128, 512, 128]；dispatch M=256；wave windows=16；AICore programs=32；AIVector programs=64；symmetric heap=16 GiB。
- benchmark-only forward；计时边界是 router 后的完整 forward，包含 routing metadata、dispatch、FC1、SwiGLU、FC2、combine 和调用内 workspace reset；不含初始化、JIT 首次编译、权重/输入生成和 correctness。
- correctness 在计时前覆盖 normal、zero-receive/empty-expert、negative/out-of-range all-drop，rtol=atol=0.05；warmup=5、iterations=50；NPU event；8 卡取 MAX。
- baseline 是 Torch-NPU grouped-GEMM + HCCL，始终无 MoonEP；加速比=Torch median/fused median。

## 运行环境与审计

- 运行前 HEAD=7a33afb；该 run 发生在清理工作区提交前，当前文件 SHA256 见下表。不能把 run 的 HEAD 字段误读为最终 commit。
- Python=/home/vllm_kimiw/.venv-udma/bin/python / 3.11.10；Torch=2.10.0+cpu；torch_npu=2.10.0.post1.dev20260528；Triton runtime=3.6.0。
- CANN=/usr/local/Ascend/cann-9.2.0-beta.2；compiler=/home/vllm_kimiw/.venv-udma/lib/python3.11/site-packages/ascendnpuir/bin/bishengir-compile；version=bishengir-compile 1.2.0 (https://gitcode.com/Ascend/AscendNPU-IR.git b229bc6ccbcc 2026-09-03) llvm 19.1.7 3254a1b1c59d Release build；TRITON_DISABLE_FFTS=1；cache=/tmp/kimi_forward_cleanup_final_20260926_cache。
- communication init order=hccl_collectives_then_aclshmem；ASCEND_LAUNCH_BLOCKING 未设置；occupancy gate 通过，运行期间未发现外部 NPU 占用。

| 当前源码 | SHA256 |
|---|---|
| src/mega_moe/kernels/fused_forward.py | 99c78cf3063a07a2fb5031c874ff52fcdbb53c11f6aa0caef2123faa861c0bd0 |
| src/mega_moe/kernels/fc2_combine.py | 497200e5ee197764b7ec90d5eca305c1f3da77da431558d1ec0e82e3597cf33b |
| src/mega_moe/ops/forward.py | 090732e13d3af02f4ef8ef3d536f4ff38370adea9e5b73cba33bf52e93e94186 |
| benchmark/layer/compile_single_kernel_forward.py | 4f2d39cbc9aea6c7b43d56ff6bac43f65f2a0172e1655b5d8e65f30dc58b9cad |
| benchmark/layer/profile_single_kernel_forward.py | 1a954ffbf5e70617e82bd57344a1e38b0a73f2bb0e884f03702764671c72bc30 |
| config/_shapes.py | 4a4d40e1518b913abfefe9415b2d2851f84ea5491d7a63722984b7d8562b3b0a |

## 清理内容

- 删除 fused forward 中已停用的 kernel-side _reset_route_to_send 和注释 launch；保留生产 host .fill_(-1)，因为 dropped route 不被 scatter 覆盖，重复调用必须清除旧逆映射。
- 删除 LAST_RETURN_ONLY constexpr、隐藏的 _single_wait_last_return override、不可达的 all-wave wait fallback；生产路径固定为 final-wave counter acquire + fence/epoch relay。
- 删除 FC2 的 FC2_V1_HOST_ITEM_TABLE 环境分支、慢 host item-table builder 和重复 wrapper；保留并重命名唯一 device builder，调用路径不变。
- 更新 compile/profile 调用、AST 回归和 wave-task 回归；没有删除 MoonEP UDMA、saved/FP8/FP16 支持、容量检查或 SYS_CNT 工具。
- 清理目标是无效/未使用/已否决的 forward 分支；没有把历史实验目录或原始矩阵产物删除，便于审计。

## 回归与硬件验证

- host/JIT：428 passed，14 个 torch deprecation warnings；Python touched-files py_compile 通过；git diff --check 通过。
- correctness：normal、zero-receive/empty-expert、negative/out-of-range all-drop 全部通过；occupancy gate device_occupancy.jsonl 通过。
- 计时样本：fused 50 个、Torch 50 个；逐卡 host intervals 在 benchmark/host_call_intervals_rank0.json 至 rank7.json；DCMI v2 telemetry 以各 rank measured interval 并集筛选，排除 warmup 和调用间空隙。

## 逐卡功耗与频率

下表为有效遥测读数的 median [min–max]，括号内是读数数量；频率 MHz，功耗 W。离散 DCMI 采样不等同于能耗积分。

| rank | fused 频率 | fused 功耗 | Torch 频率 | Torch 功耗 |
|---:|---|---|---|---|
| 0 | 1500 [1400–1650] (77) | 895.8 [773.6–900.5] (77) | 1650 [1650–1650] (112) | 817.1 [798.3–845.7] (112) |
| 1 | 1550 [1400–1650] (78) | 896.2 [827.6–900.6] (78) | 1650 [1650–1650] (110) | 815.5 [796.7–843.4] (110) |
| 2 | 1600 [1400–1650] (79) | 894.7 [823.0–901.2] (79) | 1650 [1650–1650] (110) | 806.9 [783.1–845.9] (110) |
| 3 | 1650 [1450–1650] (79) | 892.6 [802.4–899.9] (79) | 1650 [1650–1650] (110) | 797.1 [775.9–820.5] (110) |
| 4 | 1600 [1400–1650] (79) | 894.2 [813.4–900.8] (79) | 1650 [1650–1650] (110) | 803.5 [784.6–849.5] (110) |
| 5 | 1550 [1400–1650] (78) | 896.6 [828.0–901.3] (78) | 1650 [1650–1650] (109) | 811.9 [792.8–841.4] (109) |
| 6 | 1650 [1400–1650] (78) | 893.0 [804.6–901.1] (78) | 1650 [1650–1650] (108) | 807.2 [791.9–839.6] (108) |
| 7 | 1650 [1450–1650] (76) | 890.6 [796.1–899.5] (76) | 1650 [1650–1650] (108) | 793.9 [777.6–829.8] (108) |

## 遗留事项与后续补测

- full_t4k_moonep1、full_t8k_moonep1、full_t16k_moonep1 原始配置均在首次 backend compile 超过 900 s，被 SIGTERM（returncode=-15），没有 correctness 或 timing 结果；这些是待补测点，不是性能 0。
- 历史 9/12 矩阵不能推出 E896 + MoonEP 的加速比，也不能证明 1.6x 目标已达到；补测必须保持同一 frozen source、top-k=16、偏斜路由和 block/wave 配置，任何编译策略改动另建 supplemental artifact。
- saved-FC1 的大 tile compile-time/UB 限制仍保留；本次验证是 unsaved return_saved=False forward。
- kernel 内 reset 的底层 lowering/UB 机制尚未完全证明；当前 host reset 是通过 correctness 并纳入 e2e 计时的生产路径。

## 可复现入口

source /tmp/kimi_forward_matrix_20260926_completion/environment.sh
export TRITON_CACHE_DIR=/tmp/kimi_forward_cleanup_final_20260926_cache
python benchmark/layer/run_kimi_forward_matrix.py --output-dir /tmp/kimi_forward_cleanup_final_20260926 --routing-profile uniform --trimmed-topk 16 --points full_t4k_moonep0 --sample-interval 0.01 --case-timeout 900

原始命令和完整参数以 full_t4k_moonep0/job.json、status.json、benchmark/run_metadata.json 为准；文档同名 JSON 保存完整机器可读审计。
