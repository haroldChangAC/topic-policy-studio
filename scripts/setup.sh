#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p .tools .cache models/cache outputs
if [[ ! -x .tools/uv/bin/uv ]]; then
  python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --index-url https://pypi.org/simple --target .tools/uv 'uv==0.13.7'
fi
STUDIO_PYTHON_VERSION="$(cat .python-version)"
.tools/uv/bin/uv python install "$STUDIO_PYTHON_VERSION" --no-bin
if [[ -f uv.lock ]]; then
  .tools/uv/bin/uv sync --locked --python "$STUDIO_PYTHON_VERSION"
else
  .tools/uv/bin/uv sync --python "$STUDIO_PYTHON_VERSION"
fi
.tools/uv/bin/uv pip check --python .venv/bin/python
printf '\n环境配置完成。下一步：bash scripts/check.sh\n'

