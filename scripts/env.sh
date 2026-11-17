#!/usr/bin/env bash
# Source from a project script; do not alter the user's system Python or shell config.
STUDIO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PYTHON_INSTALL_DIR="$STUDIO_ROOT/.tools/python"
export UV_CACHE_DIR="$STUDIO_ROOT/.cache/uv"
export HF_HOME="$STUDIO_ROOT/models/cache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$STUDIO_ROOT/.cache/matplotlib"
export NUMBA_CACHE_DIR="$STUDIO_ROOT/.cache/numba"
export PYTHONPYCACHEPREFIX="$STUDIO_ROOT/.cache/pycache"
export PYTHONPATH="$STUDIO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$STUDIO_ROOT"

