#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
.venv/bin/python scripts/download_model.py

