#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
if [[ ! -x .venv/bin/python ]]; then
  printf '请先运行 bash scripts/setup.sh\n' >&2
  exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.tools/uv/bin/uv pip check --python .venv/bin/python
.venv/bin/python scripts/doctor.py

