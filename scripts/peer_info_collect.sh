#!/bin/bash
# 双机联调对端信息采集脚本（只读，不改任何配置）。
# 用法：在对端机器任意目录执行  bash peer_info_collect.sh
# 输出带标签的报告，全部信息用于 G2 前置核对；本端期望值见 docs/dual-node-adaptation.md。
set -u

# 如果对端仓路径不同，改这里（默认与本地布局一致）
MMT_DIR="${MMT_DIR:-/home/t0303/project/mmt}"
FW_DIR="${FW_DIR:-/home/t0303/project/framework}"
VENV_ACT="/root/moe-venv/activate-moe.sh"
NODE0_IP="141.61.95.70"   # 本端(node0)业务 IP

section() { echo; echo "===== $* ====="; }

section "1. 机器标识与网络"
echo "hostname: $(uname -n)"
echo "kernel:   $(uname -r)"
if command -v ip >/dev/null 2>&1; then
    ip -br addr 2>/dev/null | grep -v "^lo"
else
    grep -oE "([0-9]{1,3}\.){3}[0-9]{1,3}" /proc/net/fib_trie 2>/dev/null | sort -u | head
fi
echo "--- ping 本端 $NODE0_IP (业务网段) ---"
ping -c 2 -W 2 "$NODE0_IP" 2>&1 | tail -2

section "2. NPU 状态（8x Ascend950DT / 空闲 HBM）"
npu-smi info 2>/dev/null | sed -n '1,12p;/NPU ID/,+40p' | head -60

section "3. 驱动 / CANN 版本"
echo "--- driver ---"; cat /usr/local/Ascend/driver/version.info 2>/dev/null | grep -E "Version|package_version" | head -3
echo "--- /usr/local/Ascend ---"; ls /usr/local/Ascend 2>/dev/null

section "4. python 环境（venv）"
if [ -f "$VENV_ACT" ]; then
    # 子 shell 里 source，避免污染当前环境
    ( source "$VENV_ACT"
      python - <<'PY'
import torch, torch_npu
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
PY
      pip list 2>/dev/null | grep -iE "cann|shmem|triton|nvidia-nvshmem"
    )
else
    echo "!! 未找到 $VENV_ACT —— 请告知对端 venv 激活路径"
fi

section "5. RoCE 平面（UDMA 跨机数据面的关键）"
if command -v hccn_tool >/dev/null 2>&1; then
    for i in 0 1 2 3 4 5 6 7; do
        echo "-- npu$i --"; hccn_tool -i "$i" -ip -g 2>&1 | head -4
    done
else
    echo "!! 容器内无 hccn_tool（本端同样没有）"
    echo "   需在宿主机执行: hccn_tool -i 0 -ip -g （0..7 各一次）"
    echo "   或询问网络管理员: 两机 NPU RoCE 平面是否同子网/已互通"
fi

section "6. G2 相关端口占用 / 防火墙"
echo "--- 监听端口(29511/41888/HCCL段) ---"
ss -tlnu 2>/dev/null | awk 'NR==1 || /29511|41888|41900|45000/' | head -10
echo "--- firewalld/iptables ---"
systemctl is-active firewalld 2>/dev/null || echo "firewalld: n/a"
iptables -L -n 2>/dev/null | head -5 || echo "iptables: 无权限或不存在"

section "7. mmt 仓状态"
if [ -d "$MMT_DIR/.git" ]; then
    git -C "$MMT_DIR" remote -v
    echo "branch: $(git -C "$MMT_DIR" branch --show-current)"
    git -C "$MMT_DIR" log --oneline -3
    echo "--- submodule bigop ---"
    git -C "$MMT_DIR" submodule status 2>/dev/null
else
    echo "!! 未找到 $MMT_DIR —— 请告知对端 mmt 仓路径"
fi

section "8. framework 仓状态（G5 用）"
if [ -d "$FW_DIR/.git" ]; then
    git -C "$FW_DIR" log --oneline -2
else
    echo "!! 未找到 $FW_DIR —— 请告知对端 framework 仓路径"
fi

section "9. 磁盘 / triton cache"
df -h /root /tmp /home 2>/dev/null | grep -v "^Filesystem" | sort -u
echo "triton cache: $(du -sh ~/.triton/cache 2>/dev/null | cut -f1 || echo '空')"

echo
echo "===== 采集完成：请把以上完整输出发回 ====="
