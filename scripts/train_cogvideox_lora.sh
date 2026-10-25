#!/usr/bin/env bash
set -euo pipefail

# Launch on a supported Linux NVIDIA GPU host, from the project root.
# Update DATA_ROOT if your dataset is stored elsewhere.
DATA_ROOT="${DATA_ROOT:-$PWD/data}"
DIFFUSERS_DIR="${DIFFUSERS_DIR:-$PWD/diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/checkpoints/cogvideox-lora}"

if [[ ! -f "$DATA_ROOT/prompts.txt" || ! -f "$DATA_ROOT/videos.txt" ]]; then
  echo "Expected $DATA_ROOT/prompts.txt and $DATA_ROOT/videos.txt"
  exit 2
fi
if [[ ! -f "$DIFFUSERS_DIR/examples/cogvideo/train_cogvideox_lora.py" ]]; then
  echo "Could not find the official trainer. Run scripts/setup_gpu.sh first."
  exit 2
fi

python scripts/validate_dataset.py --root "$DATA_ROOT"

mkdir -p "$OUTPUT_DIR"
accelerate launch "$DIFFUSERS_DIR/examples/cogvideo/train_cogvideox_lora.py" \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --instance_data_root "$DATA_ROOT" \
  --caption_column prompts.txt \
  --video_column videos.txt \
  --output_dir "$OUTPUT_DIR" \
  --rank 16 \
  --lora_alpha 16 \
  --mixed_precision fp16 \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --skip_frames_start 0 \
  --skip_frames_end 0 \
  --train_batch_size 1 \
  --max_train_steps 100 \
  --checkpointing_steps 50 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing \
  --enable_slicing \
  --enable_tiling \
  --use_8bit_adam \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 10 \
  --max_grad_norm 1.0 \
  --seed 42

echo
echo "Training command finished. Check: $OUTPUT_DIR"
echo "Look for pytorch_lora_weights.safetensors and inspect the trainer logs."
echo "100 steps is only a smoke test; it is not evidence of a useful fine-tune."
