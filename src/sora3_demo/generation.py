"""Open-Sora 2.0 text-to-video adapter.

This calls the official Open-Sora 2.0 inference CLI. It does not generate
placeholder output. The worker must have the official repository, checkpoints,
and a compatible CUDA GPU installed before generation can be enabled.
"""
from pathlib import Path
import os
import subprocess
import sys


def generate_video(
    prompt: str,
    output_path: str,
    model_id: str = "hpcai-tech/Open-Sora-v2",
    lora_path: str | None = None,
    num_frames: int = 49,
    fps: int = 8,
    num_inference_steps: int = 30,
    guidance_scale: float = 6.0,
    seed: int = 42,
) -> str:
    """Run real Open-Sora 2.0 inference and return the saved MP4 path.

    Open-Sora 2.0 currently uses its own configs/checkpoints rather than the
    Diffusers CogVideoX pipeline. lora_path and guidance_scale are retained in
    the API signature for compatibility but are not passed to Open-Sora 2.0.
    """
    repo = Path(os.getenv("OPEN_SORA_DIR", "/opt/Open-Sora")).resolve()
    script = repo / "scripts" / "diffusion" / "inference.py"
    config = Path(os.getenv(
        "OPEN_SORA_CONFIG", "configs/diffusion/inference/256px.py"
    ))
    if not config.is_absolute():
        config = repo / config
    if not script.is_file():
        raise RuntimeError(
            f"Open-Sora 2.0 is not installed at {repo}. Set OPEN_SORA_DIR to the official Open-Sora repository."
        )
    if not config.is_file():
        raise RuntimeError(f"Open-Sora inference config not found: {config}")

    # Open-Sora 2.0 expects frame counts of 4k+1 and fewer than 129 frames.
    frames = max(9, min(125, int(num_frames)))
    frames = 4 * round((frames - 1) / 4) + 1
    frames = min(frames, 125)
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = output.parent / f"opensora_{output.stem}"
    samples.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in samples.rglob("*.mp4")}

    command = [
        "torchrun", "--nproc_per_node", "1", "--standalone",
        str(script), str(config),
        "--save-dir", str(samples),
        "--prompt", prompt,
        "--num_frames", str(frames),
        "--sampling_option.num_steps", str(max(1, min(50, int(num_inference_steps)))),
        "--sampling_option.seed", str(int(seed)),
        "--seed", str(int(seed)),
        "--fps_save", str(max(1, min(60, int(fps)))),
        "--offload", "True",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(command, cwd=str(repo), env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "No output from Open-Sora").strip()
        raise RuntimeError(f"Open-Sora 2.0 inference failed (exit {completed.returncode}): {details[-5000:]}")

    candidates = [p for p in samples.rglob("*.mp4") if p.resolve() not in before and p.is_file()]
    if not candidates:
        raise RuntimeError(
            f"Open-Sora finished without a new MP4 in {samples}. Check the worker logs and output configuration."
        )
    generated = max(candidates, key=lambda p: p.stat().st_mtime)
    if generated.resolve() != output:
        import shutil
        shutil.copy2(generated, output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Open-Sora returned an empty or missing MP4 file.")
    return str(output)
