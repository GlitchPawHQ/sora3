#!/usr/bin/env bash
set -euo pipefail

# Run only on a compatible Linux NVIDIA GPU worker, not the free Render web service.
ROOT="${OPEN_SORA_DIR:-/opt/Open-Sora}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Open-Sora 2.0 needs a compatible NVIDIA CUDA GPU." >&2
  exit 1
fi
if [ ! -d "$ROOT/.git" ]; then
  git clone https://github.com/hpcaitech/Open-Sora.git "$ROOT"
fi
cd "$ROOT"
python -m pip install --upgrade pip
# Follow the upstream installation path; CUDA/PyTorch and xformers must match the worker image.
python -m pip install -v .
# Install xformers only if compatible with the worker's installed PyTorch/CUDA build.
# See the official README for the correct wheel index for your CUDA version.
python -m pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts
# The config expects the model, VAE, T5, and CLIP files under ./ckpts.
for f in ckpts/Open_Sora_v2.safetensors ckpts/hunyuan_vae.safetensors ckpts/google/t5-v1_1-xxl ckpts/openai/clip-vit-large-patch14; do
  if [ ! -e "$f" ]; then
    echo "WARNING: expected checkpoint path not found: $f" >&2
  fi
done
printf '\nOpen-Sora repository: %s\n' "$ROOT"
printf 'Next: configure OPEN_SORA_DIR=%s and run a real inference test before enabling API generation.\n' "$ROOT"
