#!/bin/bash
# Local two-process G2a bootstrap repro driver. ~1 min per iteration, no /etc
# swap, no peer node: run on ANY machine to capture what that machine's shmem
# build emits for the step-8 endpoint descriptor (see gwtrace hexdump).
#
# Env:
#   REPRO_MASTER  this host's business IP (default: first of `hostname -I`)
#   REPRO_VENV    path to venv activate script (default /root/moe-venv/activate-moe.sh)
#   STAGGER       seconds between rank0 and rank1 start (default 8)
# Prereq: gcc -shared -fPIC -O1 -o /tmp/gwtrace.so scripts/gwtrace.c -ldl
set -u
source "${REPRO_VENV:-/root/moe-venv/activate-moe.sh}"
cd "$(dirname "$0")/.."

def_ip() { python3 -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0])" 2>/dev/null; }
export LD_PRELOAD=/tmp/gwtrace.so
export MEGAMOE_MULTI_NODE=1 MASTER_ADDR="${REPRO_MASTER:-$(def_ip)}" MASTER_PORT=29521
export ASH_MASTER_ADDR="$MASTER_ADDR" ASH_MASTER_PORT=41889 SHMEM_LOG_LEVEL=INFO
T=$(mktemp -d /tmp/repro.XXXXXX)

if [ "${FIRST:-0}" = "1" ]; then
  # rank1 first: its store client connects to a not-yet-existing server.
  REPRO_RANK=1 timeout 240 python -u scripts/g2_shmem_repro.py > "$T/repro_r1.log" 2>&1 &
  R1PID=$!
  sleep "${STAGGER:-8}"
  REPRO_RANK=0 timeout 240 python -u scripts/g2_shmem_repro.py > "$T/repro_r0.log" 2>&1
  R0=$?
  wait $R1PID; R1=$?
else
  REPRO_RANK=0 timeout 240 python -u scripts/g2_shmem_repro.py > "$T/repro_r0.log" 2>&1 &
  R0PID=$!
  sleep "${STAGGER:-8}"
  REPRO_RANK=1 timeout 240 python -u scripts/g2_shmem_repro.py > "$T/repro_r1.log" 2>&1
  R1=$?
  wait $R0PID; R0=$?
fi
echo "RESULT r0_exit=$R0 r1_exit=$R1  (logs: $T/repro_r{0,1}.log)"
echo "== step-8 descriptor from each rank (the cross-node diff artifact) =="
for r in 0 1; do
  echo "--- r$r ---"
  grep -A13 "Append key=SHM_(0)_S_0_8_GA" "$T/repro_r$r.log" | head -15
done
