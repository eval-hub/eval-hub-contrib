#!/usr/bin/env bash
set -euo pipefail
cd /app
export STABLETOOLBENCH_CONFIG="${STABLETOOLBENCH_CONFIG:-/app/config.yml}"
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"
