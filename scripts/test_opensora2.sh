#!/usr/bin/env bash
set -euo pipefail
ROOT="${OPEN_SORA_DIR:-/opt/Open-Sora}"
cd "$ROOT"
# One official CLI inference test. This writes a real generated sample under samples/.
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py \
  configs/diffusion/inference/256px.py \
  --save-dir samples \
  --prompt "A red ball bouncing on a table in a softly lit room" \
  --offload True
find samples -type f -name '*.mp4' -printf '%T@ %p\n' | sort -nr | head -5
