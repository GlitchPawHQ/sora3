#!/usr/bin/env python3
"""Generate an MP4 from a text prompt using pretrained CogVideoX."""
import argparse
from sora3_demo.generation import generate_video

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="THUDM/CogVideoX-2b")
    p.add_argument("--output", default="outputs/generated.mp4")
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", default=None)
    a = p.parse_args()
    path = generate_video(
        prompt=a.prompt, output_path=a.output, model_id=a.model, lora_path=a.lora,
        num_frames=a.frames, fps=a.fps, num_inference_steps=a.steps,
        guidance_scale=a.guidance_scale, seed=a.seed,
    )
    print(f"Saved generated video: {path}")

if __name__ == "__main__":
    main()
