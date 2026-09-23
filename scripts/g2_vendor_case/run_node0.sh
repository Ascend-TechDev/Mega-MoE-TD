#!/bin/bash
# Vendor case launcher — node0 (141.61.95.70 / srv_24 / superpod_16383).
# Usage: ./run_node0.sh {push|pull|tunnel} [engine]
set -eu
CASE=${1:-push}
ENGINE=${2:-udma}

source /root/moe-venv/activate-moe.sh
SP=$(python3 -c "import shmem, os; print(os.path.dirname(os.path.dirname(shmem.__file__)))" 2>/dev/null || echo /usr/local/python3.11.10/lib/python3.11/site-packages)
export LD_LIBRARY_PATH=$SP/shmem/backends/950:${LD_LIBRARY_PATH:-}
export ASH_MASTER_ADDR=141.61.95.70 ASH_MASTER_PORT=41921
export MOE_ASH_ENGINE=$ENGINE
cd "$(dirname "$0")"

exec torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
     --master_addr=141.61.95.70 --master_port=29561 repro.py "$CASE"
