#!/usr/bin/env bash
# 安装测试期第三方 golden 依赖（bigop）为 editable 包。
# 在仓库根目录运行：bash scripts/install_3rdparty.sh
#
# bigop 作为 3rdparty/bigop git submodule 接入（pin 到固定 commit，见 .gitmodules）。
# editable 安装使后续 git submodule update 换 commit 后 import bigop 立即生效，无需 pip 重装。
# torch / torch-npu 由 Ascend 环境提供，故 --no-deps（与 mega_moe 自身安装一致）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

git submodule update --init --recursive 3rdparty/bigop
python -m pip install -e 3rdparty/bigop --no-deps
python -c "import bigop; print('[ok] bigop ->', bigop.__file__)"
