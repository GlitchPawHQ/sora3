#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; this script switches to the project root.
cd "$(dirname "$0")"

echo "==> Installing Python dependencies..."
python -m pip install --disable-pip-version-check -r requirements.txt

echo "==> Starting the homepage/API server..."
export SORA3_GENERATION_ENABLED="${SORA3_GENERATION_ENABLED:-false}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m uvicorn sora3_demo.api:app --app-dir src --host 0.0.0.0 --port "${PORT:-8000}"
