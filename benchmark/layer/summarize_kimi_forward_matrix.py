# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Build an auditable forward-only report from all planned matrix points."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics

PHASES = {"candidate": "single_kernel", "baseline": "torch_grouped_hccl"}
KERNEL = "src/mega_moe/kernels/fused_forward.py"
OWNER_COUNTS = (19, 27, 11, 11, 15, 15, 15, 15)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stats(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        raise ValueError("Non-finite or non-numeric measurement")
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "p95": p95, "max": max(values)}


def in_intervals(midpoint, intervals, starts):
    index = bisect_right(starts, midpoint) - 1
    return index >= 0 and midpoint <= intervals[index][1]


def summarize_telemetry(point_dir, warmup, iterations, world):
    windows = {}
    for rank in range(world):
        path = point_dir / "benchmark" / f"host_call_intervals_rank{rank}.json"
        recorded = read_json(path)
        if recorded is None:
            raise ValueError(f"Missing per-rank intervals: {path}")
        windows[rank] = {}
        for phase in PHASES:
            intervals = recorded.get(phase, [])
            if len(intervals) != warmup + iterations:
                raise ValueError(f"Wrong interval count: {path}, {phase}: {len(intervals)}")
            measured = intervals[warmup:]
            if any(a is None or b is None or a > b for a, b in measured):
                raise ValueError(f"Invalid interval in {path}")
            if any(measured[i][1] > measured[i + 1][0] for i in range(len(measured) - 1)):
                raise ValueError(f"Overlapping measured intervals in {path}")
            windows[rank][phase] = (measured, [a for a, _ in measured])
    selected = {phase: {str(rank): {"frequency": [], "power": []}
                        for rank in range(world)} for phase in PHASES}
    with (point_dir / "telemetry.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            rank = row["rank"]
            if rank not in windows:
                continue
            midpoint = (row["start_monotonic_ns"] + row["end_monotonic_ns"]) // 2
            for phase in PHASES:
                if not in_intervals(midpoint, *windows[rank][phase]):
                    continue
                for metric, key in (("frequency", "frequency_mhz"), ("power", "power_w")):
                    if row.get(f"{metric}_status") == 0 and row.get(key) is not None:
                        selected[phase][str(rank)][metric].append(row[key])
    summary = {phase: {rank: {metric: stats(values) for metric, values in measurements.items()}
                       for rank, measurements in devices.items()} for phase, devices in selected.items()}
    for phase, devices in summary.items():
        for rank, measurements in devices.items():
            for metric, values in measurements.items():
                if not values["n"]:
                    raise ValueError(f"No measured {metric} samples: {point_dir.name}, {phase}, rank {rank}")
    return summary


def point_record(root, planned, frozen_hash, supplement=False):
    point_dir = root / planned["point"]
    job = read_json(point_dir / "job.json", planned)
    status = read_json(point_dir / "status.json", {})
    result = read_json(point_dir / "benchmark/benchmark_result.json", {})
    meta = read_json(point_dir / "benchmark/run_metadata.json", {})
    record = {"point": planned["point"], "artifact": str(point_dir), "job": job,
              "status": status, "result": result, "metadata": meta, "supplement": supplement,
              "valid": False, "metrics": {}, "telemetry": {}}
    if not result:
        record["state"] = "incomplete" if not status else "failed_before_timing"
        return record
    case = job["case"]
    if status.get("returncode") != 0:
        raise ValueError(f"Result exists but process failed: {point_dir}")
    if result["correctness_gate"]["status"] != "passed_before_timing":
        raise ValueError(f"Correctness gate not passed: {point_dir}")
    if result["occupancy_gate"]["status"] != "passed":
        raise ValueError(f"Device occupancy gate not passed: {point_dir}")
    if result["case_id"] != case["case_id"] or result["tokens_per_rank"] != case["tokens"]:
        raise ValueError(f"Input case mismatch: {point_dir}")
    if result["shape"] != {key: case[key] for key in ("hidden", "ffn", "topk", "num_experts")}:
        raise ValueError(f"Input shape mismatch: {point_dir}")
    if meta["source_sha256"][KERNEL] != frozen_hash:
        raise ValueError(f"Kernel source hash mismatch: {point_dir}")
    protocol = result["protocol"]
    if protocol != {"warmup": 5, "iterations": 50, "clock": "npu_event", "rank_reduction": "MAX"}:
        raise ValueError(f"Unexpected timing protocol: {point_dir}")
    config = result["operator_config"]
    for keys, block in ((("fc1_gemm_block_size_m", "fc1_gemm_block_size_n", "fc1_gemm_block_size_k"), job["fc1_block"]),
                        (("fc2_combine_block_size_m", "fc2_gemm_block_size_n", "fc2_gemm_block_size_k"), job["fc2_block"])):
        if [config[key] for key in keys] != block:
            raise ValueError(f"Block configuration mismatch: {point_dir}")
    if config["enable_moonep"] != job["moonep"] or config["single_kernel_group_windows"] != job["wave_windows"]:
        raise ValueError(f"MoonEP/wave configuration mismatch: {point_dir}")
    for phase, source in PHASES.items():
        samples = result["samples_ms"][source]
        if len(samples) != protocol["iterations"] or any(value <= 0 for value in samples):
            raise ValueError(f"Invalid timing samples: {point_dir}, {phase}")
        record["metrics"][phase] = stats(samples)
    record["speedup"] = record["metrics"]["baseline"]["median"] / record["metrics"]["candidate"]["median"]
    record["telemetry"] = summarize_telemetry(point_dir, protocol["warmup"], protocol["iterations"], case["world_size"])
    expected_routes = [case["tokens"] * case["world_size"] * case["topk"] * quota // 128 for quota in OWNER_COUNTS]
    if result["route_distribution"]["routes_received_per_rank"] != expected_routes:
        raise ValueError(f"Unexpected skewed owner distribution: {point_dir}")
    record.update(valid=True, state="measured")
    return record


def load_matrix(root, supplement=False):
    manifest = read_json(root / "matrix.json")
    if manifest is None:
        raise ValueError(f"No matrix manifest: {root}")
    jobs = manifest["planned_jobs"]
    combinations = [(j["case"]["tokens"], j["case"]["num_experts"], j["moonep"]) for j in jobs]
    if len(set(combinations)) != len(combinations):
        raise ValueError("Duplicate planned combinations")
    if not supplement and set(combinations) != {(t, e, m) for t in (4096, 8192, 16384) for e in (32, 896) for m in (False, True)}:
        raise ValueError("Primary matrix must contain all 12 requested combinations")
    if manifest["routing_profile"] != "skewed" or any(j["case"]["topk"] != 16 for j in jobs):
        raise ValueError("Expected user-selected skewed routing and top-k=16")
    records = [point_record(root, job, manifest["frozen_kernel_sha256"], supplement) for job in jobs]
    return manifest, records


def telemetry_cell(values, digits):
    if not values.get("n"):
        return "—"
    return f"{values['median']:.{digits}f} [{values['min']:.{digits}f}–{values['max']:.{digits}f}], {values['n']}"


def timings(record, phase):
    if not record["valid"]:
        return "—"
    values = record["metrics"][phase]
    return " / ".join(f"{values[key]:.3f}" for key in ("median", "mean", "p95"))


def build_report(root, manifest, records, diagnostics, generated):
    primary = [r for r in records if not r["supplement"]]
    valid = [r for r in primary if r["valid"]]
    pending = len(primary) - len(valid)
    reference = valid[0]["metadata"] if valid else {}
    env = reference.get("environment", {})
    compiler = env.get("npu_compiler", {})
    lines = ["# Kimi K3 8 卡 fused forward 性能矩阵", "",
             f"> 生成时间：{generated}。原始矩阵：`{root}`。",
             f"> **已取得有效性能数据 {len(valid)}/{len(primary)} 点；尚缺 {pending} 点。不能将编译超时视为性能结果或正确性失败。**", "",
             "暂停新优化探索，固定当前已验证的 fused forward 源码。比较每卡 4096/8192/16384 tokens、裁剪 E=32 / 全量 E=896、MoonEP 开/关。两种专家规模都使用 top-k=16 和现有偏斜路由。", "",
             "## 测量与输入口径", "",
             "- **只测 forward**：直接执行 `op.forward(..., return_saved=False)`；输入无需梯度，不调用 backward。`torch.autograd.profiler.record_function` 只是 profiler 标记，此次 `--benchmark-only` 跳过 profiling 分支，不属于计时操作。",
             "- **计时边界**：router 之后的完整 forward，包含路由元数据、dispatch、FC1、weighted SwiGLU、FC2、combine，以及调用中的工作区重置。权重/输入生成、算子初始化、JIT 首次编译和正确性检查均在计时外；不是纯 GEMM 或单个设备 kernel 的裸时间。",
             "- 每点先验证 normal、zero-receive/empty-expert、negative/out-of-range all-drop，容差 rtol=atol=0.05；然后 5 次预热、50 次 NPU event 采样，每样本取 8 卡 MAX。没有额外的 changed-hidden 正确性门。",
             "- median、mean、P95 全部从 50 个原始样本重算。P95 使用排序后位置 `0.95*(n-1)` 的线性插值。加速比 = `Torch grouped-GEMM + HCCL median / fused median`。",
             "- **MoonEP 开关只作用于 fused 候选**；Torch baseline 始终是无 MoonEP 的 grouped-GEMM + HCCL。同一专家规模与 token 档位的开/关使用相同输入种子和路由；wave windows 随开关为 32/16，因此开关差异也包含运行配置变化。",
             "- 偏斜 owner 配额为 `[19,27,11,11,15,15,15,15]/128`，所有全局专家均活跃。每个 token 的 16 个专家互不重复；各 owner 内轮询分配。裁剪/全量的 owner 负载相同，专家身份及权重形状随专家数改变。",
             "- 权重种子 `42+rank`；输入种子 `43+rank*1000`。hidden states 在 routing logits 之前生成。BF16：hidden、W1、W2、output；FP32：routing weights；INT32：selected experts。activation=SwiGLU；capacity factor=1.6875；drop_frac=0；对称堆每卡 16 GiB。",
             "- 每卡有 32 个 AICore programs / 64 个 AIVector programs；dispatch block M=256。MoonEP replica cache 关闭，每次 forward 刷新迁移权重。`save_fc1_dtype=bf16` 是配置字段，本次 `return_saved=False` 不保存训练中间值。",
             "- **逐卡遥测**：DCMI v2 只读采样，目标间隔 10 ms，frequency type=7，功耗原始单位 0.1 W。分别按 rank0…7 各自的 50 个测量调用区间的并集筛选；排除 5 次预热与调用间空隙。区间结束点在 NPU event 完成后、rank-MAX 集合通信前。",
             "- 遥测表显示 `中位数 [最小–最大], 有效样本数`。这是 host/event 区间内离散设备读数，不是每个 kernel 的精确能耗积分；驱动读数刷新率可能低于轮询频率。每次硬件运行前等待 8 卡空闲，并检查运行期间的外部进程占用。", "",
             "## 固定版本与工具链", "",
             f"- worktree HEAD：`{reference.get('git_commit', '—')}`；kernel SHA256：`{manifest['frozen_kernel_sha256']}`。这是包含 PR #68 及后续已验证优化的当前 worktree，不应标为裸 main 或裸 PR #68。",
             f"- Python：`{env.get('python', '—')}`（{env.get('python_version', '—')}）；Torch={env.get('torch', '—')}；torch_npu={env.get('torch_npu', '—')}；运行时 Triton={env.get('triton_runtime', '—')}。",
             f"- CANN：`{env.get('ascend_home_resolved', env.get('ascend_home', '—'))}`；编译器：`{compiler.get('resolved_path', compiler.get('path', '—'))}`。",
             f"- 编译器版本：`{' '.join(compiler.get('version', '—').split())}`；`TRITON_DISABLE_FFTS={compiler.get('triton_disable_ffts', '—')}`。",
             f"- 原始 Triton cache：`{compiler.get('triton_cache_dir', '—')}`。各点 `run_metadata.json` 保存完整环境、包版本、device library 与相关源码 SHA256；不以其他环境的旧结果替代。", "",
             "## 全部 12 个计划数据点", "",
             "时间列均为 median / mean / P95，单位 ms；`—` 表示没有有效计时。", "",
             "| 数据点 | case | tokens/rank | E / top-k | MoonEP | FC1 M,N,K | FC2 M,N,K | wave | fused ms | Torch ms | 加速比 | 状态 |",
             "|---|---|---:|---:|:---:|---|---|---:|---:|---:|---:|---|"]
    for r in records:
        j = r["job"]; c = j["case"]
        speed = f"{r['speedup']:.3f}×" if r["valid"] else "—"
        state = "正确性/占用通过" if r["valid"] else "未取得计时"
        label = r["point"] + ("（补测配置）" if r["supplement"] else "")
        lines.append(f"| `{label}` | `{c['case_id']}` | {c['tokens']} | {c['num_experts']} / {c['topk']} | {'开' if j['moonep'] else '关'} | {','.join(map(str,j['fc1_block']))} | {','.join(map(str,j['fc2_block']))} | {j['wave_windows']} | {timings(r,'candidate')} | {timings(r,'baseline')} | {speed} | {state} |")
    lines.extend(["", "## 当前可支持的比较", ""])
    for tokens in (4096, 8192, 16384):
        points = {(r["job"]["case"]["num_experts"], r["job"]["moonep"]): r for r in valid if r["job"]["case"]["tokens"] == tokens}
        off, on, full = points.get((32, False)), points.get((32, True)), points.get((896, False))
        if off and on and full:
            ratio = off["metrics"]["candidate"]["median"] / on["metrics"]["candidate"]["median"]
            lines.append(f"- {tokens//1024}k：裁剪无 MoonEP / 裁剪有 MoonEP / 全量无 MoonEP 对 Torch 的加速比分别为 {off['speedup']:.3f}× / {on['speedup']:.3f}× / {full['speedup']:.3f}×；裁剪开启 MoonEP 后的 forward 中位数改善 {ratio:.3f}×。")
    lines.extend(["", "在本次 top-k=16、偏斜路由的受控输入中，无 MoonEP 的全量 forward 中位数与裁剪相当或更低。历史 1.6× 比较需要同时核对 top-k、路由和计时边界；不能用本矩阵推断不同口径的历史结论。全量 + MoonEP 缺失，尚不能比较两种专家规模下 MoonEP 的完整收益。", "", "## 每点参数、命令与逐卡遥测", ""])
    for index, r in enumerate(records, 1):
        j = r["job"]; c = j["case"]; status = r["status"]
        t,h,f,e = c["tokens"],c["hidden"],c["ffn"],c["num_experts"]//c["world_size"]
        routes = [t*c["world_size"]*c["topk"]*quota//128 for quota in OWNER_COUNTS]
        lines.extend([f"### {index}. `{r['point']}`" + ("（补测配置）" if r["supplement"] else ""), "",
                      f"- case=`{c['case_id']}`；model={c['model']}；world={c['world_size']}；tokens/rank={t}；global tokens={t*c['world_size']}；E={c['num_experts']}；E/rank={e}；top-k={c['topk']}；capacity={c['capacity_factor']}。",
                      f"- 形状：hidden/output=`[{t},{h}]`；indices/routing weights=`[{t},{c['topk']}]`；W1=`[{e},{h},{2*f}]`；W2=`[{e},{h},{f}]`。",
                      f"- MoonEP={'开' if j['moonep'] else '关'}；FC1 block={j['fc1_block']}；FC2 block={j['fc2_block']}；dispatch M={j['dispatch_block']}；wave={j['wave_windows']}；其余公共配置见测量口径。",
                      f"- 路由计划的原始 owner routes/rank={routes}；active global experts={c['num_experts']}。",
                      f"- 执行时间：{status.get('started_utc','—')} → {status.get('ended_utc','—')}；进程 returncode={status.get('returncode','—')}；总用时={status.get('duration_s','—')} s（包含初始化/编译/验证，不是 forward 延迟）。",
                      f"- 原始目录：`{r['artifact']}`；`job.json`、`status.json`、`benchmark.log`、`telemetry.jsonl` 均保留。"])
        if r["valid"]:
            route = r["result"]["route_distribution"]
            moonep = route.get("moonep", {})
            lines.append(f"- 结果：fused median/mean/P95={timings(r,'candidate')} ms；Torch={timings(r,'baseline')} ms；加速比={r['speedup']:.3f}×；50 样本；正确性与外部占用检查通过。")
            if moonep:
                lines.append(f"- MoonEP 迁移：copies/rank={moonep.get('copies_per_rank')}；每次 forward 合计权重复制 {moonep.get('total_weight_bytes_per_forward')} bytes；transport={moonep.get('transport')}；刷新={moonep.get('replica_refresh')}。")
            lines.append("- 原始计时在 `benchmark/benchmark_result.json`；逐卡区间在 `benchmark/host_call_intervals_rank0.json` 至 `rank7.json`；版本在 `benchmark/run_metadata.json`。")
        else:
            lines.append(f"- **未取得有效 forward 延迟/加速比**：{status.get('stop_reason','无完整结果文件')}。原始三个全量 + MoonEP 点均停在首次正确性调用触发的后端编译，尚未完成正确性门和计时；此处不把编译期功耗冒充 forward 功耗。")
        command = j.get("command", status.get("command", []))
        if command:
            lines.extend(["", "```bash", shlex.join(command), "```", ""])
        if r["valid"]:
            lines.extend(["| rank / device | fused 频率 MHz | fused 功耗 W | Torch 频率 MHz | Torch 功耗 W |", "|---:|---|---|---|---|"])
            for rank in range(c["world_size"]):
                candidate = r["telemetry"]["candidate"][str(rank)]
                baseline = r["telemetry"]["baseline"][str(rank)]
                lines.append(f"| {rank} | {telemetry_cell(candidate['frequency'],0)} | {telemetry_cell(candidate['power'],1)} | {telemetry_cell(baseline['frequency'],0)} | {telemetry_cell(baseline['power'],1)} |")
        lines.append("")
    lines.extend(["## 缺失点的补测记录", "",
                  "原配置全量 + MoonEP 三点均超过 900 s 运行预算，编译阶段未退出。以下尝试只用于补齐基准，没有改动 frozen kernel 源码，也没有可报告的额外性能数据。", ""])
    for diagnostic in diagnostics:
        status = diagnostic["status"]
        lines.append(f"- `{diagnostic['path']}`：returncode={status.get('returncode')}；duration={status.get('duration_s','—')} s；{status.get('stop_reason', diagnostic.get('note','详见对应日志'))}。")
        if diagnostic.get("note") and status.get("stop_reason"):
            lines.append(f"  说明：{diagnostic['note']}")
    lines.extend(["", "这些失败不能证明 CANN 版本不匹配，也不是 `rtsGetHardwareSyncAddr` 的运行时报错。缺失点仍需成功编译、通过三个正确性场景和设备占用检查后，才能补入有效性能表。所有中间产物保留，报告旁的同名 JSON 保存重算统计、逐卡遥测摘要、状态与审计信息。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--diagnostic-status", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = args.matrix.resolve()
    manifest, records = load_matrix(root)
    for supplement in args.supplement:
        _, extra = load_matrix(supplement.resolve(), supplement=True)
        records.extend(extra)
    records.sort(key=lambda r: (r["supplement"],r["job"]["case"]["tokens"],r["job"]["case"]["num_experts"],r["job"]["moonep"]))
    diagnostics = [{"path": str(path.resolve()), "status": read_json(path)} for path in args.diagnostic_status]
    if any(d["status"] is None for d in diagnostics):
        raise ValueError("Missing diagnostic status file")
    generated = datetime.now(timezone.utc).isoformat()
    valid = [r for r in records if r["valid"]]
    source_versions = {key: sorted({r["metadata"]["source_sha256"].get(key,"missing") for r in valid})
                       for key in (KERNEL,"src/mega_moe/ops/forward.py","src/mega_moe/kernels/fused_moonep.py","benchmark/layer/_kimi_routes.py")}
    if any(len(values) != 1 for values in source_versions.values()):
        raise ValueError("Measured points used different kernel/operator/routing sources")
    audit = {"planned_primary_points": len(manifest["planned_jobs"]), "measured_primary_points": sum(r["valid"] and not r["supplement"] for r in records),
             "required_samples_per_phase": 50, "telemetry_ranks_per_phase": 8,
             "telemetry_selection": "union of rank-specific measured intervals; warmup and inter-call gaps excluded",
             "source_versions": source_versions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_report(root,manifest,records,diagnostics,generated))
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps({"generated_utc": generated, "primary_matrix": str(root), "audit": audit,
                                     "records": records, "diagnostics": diagnostics},indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({**audit,"markdown":str(args.output),"json":str(json_path)},ensure_ascii=False))


if __name__ == "__main__":
    main()
