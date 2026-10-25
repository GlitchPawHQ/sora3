#!/usr/bin/env python3
"""Generate an MP4 using CogVideoX-2B plus a trained Diffusers LoRA adapter."""
import argparse
from pathlib import Path
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora", default="checkpoints/cogvideox-lora")
    parser.add_argument("--base-model", default="THUDM/CogVideoX-2b")
    parser.add_argument("--output", default="outputs/generated.mp4")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. This inference script is intended for a suitable GPU host, not Termux.")

    dtype = torch.float16
    print(f"Loading base model {args.base_model}; this may download several GB.")
    pipe = CogVideoXPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(args.lora, adapter_name="sora3-demo")
    pipe.set_adapters(["sora3-demo"], [args.lora_scale])

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    frames = result.frames[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=args.fps)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
