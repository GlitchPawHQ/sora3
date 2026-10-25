#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
exec uvicorn sora3_demo.api:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
