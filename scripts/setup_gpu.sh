#!/usr/bin/env bash
set -euo pipefail

# Run ONLY on a Linux GPU machine, not in Android/Termux.
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This setup is intended for a Linux GPU machine, not Android/Termux."
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Verify that this machine has a supported NVIDIA GPU and driver."
  echo "Do not continue on the Android phone itself."
  exit 2
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

if [[ ! -d diffusers ]]; then
  git clone https://github.com/huggingface/diffusers.git
fi

python -m pip install -e ./diffusers
python -m pip install -r ./diffusers/examples/cogvideo/requirements.txt

echo
echo "GPU setup complete. Record the exact upstream revision:"
git -C diffusers rev-parse HEAD
echo
echo "Next: validate your dataset and edit scripts/train_cogvideox_lora.sh."
