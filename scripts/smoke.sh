#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
.venv/bin/python scripts/smoke.py "$@"

