# Open-Sora 2.0 migration (prepared, not yet GPU-validated)

This branch of the project replaces the CogVideoX Diffusers call in `src/sora3_demo/generation.py` with a subprocess call to the official Open-Sora 2.0 inference script.

- Official repo: https://github.com/hpcaitech/Open-Sora
- Model repository: `hpcai-tech/Open-Sora-v2`
- Inference config: `configs/diffusion/inference/256px.py`
- Inference entry point: `scripts/diffusion/inference.py`

## Important

**Do not enable generation on the existing free Render web service.** `render.yaml` intentionally keeps `SORA3_GENERATION_ENABLED=false`. This service does not provide the GPU memory required for Open-Sora 2.0. This migration has not been tested on a GPU and is not proof that a video can already be generated.

The official Open-Sora 2.0 README documents the 11B model, checkpoint download, CUDA-oriented installation, and inference commands. Its reported 256px reference run uses 52.5 GB peak GPU memory on one H100/H800-class GPU; requirements vary by hardware and settings.

## GPU worker preparation

Run `scripts/setup_opensora_gpu.sh` on a compatible Linux NVIDIA GPU machine only. The script clones the official repository and downloads its checkpoint set. Review the official installation instructions to select PyTorch/xformers builds matching the worker CUDA runtime:

https://github.com/hpcaitech/Open-Sora

The API adapter expects:

- `OPEN_SORA_DIR` — path to the cloned official Open-Sora repository (default `/opt/Open-Sora`)
- `OPEN_SORA_CONFIG` — inference config path (default `configs/diffusion/inference/256px.py`)
- `SORA3_GENERATION_ENABLED=false` until an actual MP4 test succeeds

The adapter uses the official `torchrun ... scripts/diffusion/inference.py ... --prompt ... --offload True` route and checks that a new non-empty MP4 was produced. It does not create or return a placeholder video.

## Compatibility notes

The existing API request schema is retained to minimize disruption. Open-Sora 2.0 has its own parameters; `num_frames` is adjusted to the documented `4k+1` convention and limited to 125. `lora_path` and `guidance_scale` remain in the function signature for API compatibility but are not used by this Open-Sora config. Open-Sora settings such as aspect ratio, guidance, and sampling options should be configured with its own config/CLI overrides.

## Before deploying

1. Build and install on the intended GPU worker.
2. Confirm the checkpoint files and dependencies are present.
3. Run one actual generation directly using the official CLI.
4. Verify the MP4 opens and is non-empty.
5. Only then plan a secure Render-to-worker connection. Keep generation disabled until that integration is implemented and tested.

This ZIP is a source-code migration package only. It does not modify your Render service or create GPU resources.
