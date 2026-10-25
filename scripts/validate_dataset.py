#!/usr/bin/env python3
"""Validate CogVideoX's local prompts.txt/videos.txt dataset format."""
import argparse
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def read_nonempty(path: Path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Folder containing prompts.txt, videos.txt, and video files")
    args = parser.parse_args()
    root = Path(args.root)
    prompts_path = root / "prompts.txt"
    videos_path = root / "videos.txt"

    missing = [str(p) for p in (prompts_path, videos_path) if not p.is_file()]
    if missing:
        raise SystemExit("Missing required files: " + ", ".join(missing))

    prompts = read_nonempty(prompts_path)
    videos = read_nonempty(videos_path)
    errors = []

    if len(prompts) != len(videos):
        errors.append(f"Caption count ({len(prompts)}) differs from video count ({len(videos)}).")

    for index, rel in enumerate(videos, 1):
        p = root / rel
        if not p.is_file():
            errors.append(f"Video line {index}: file not found: {p}")
        elif p.suffix.lower() not in VIDEO_SUFFIXES:
            errors.append(f"Video line {index}: unusual video extension: {p.suffix}")

    if not prompts:
        errors.append("No captions found.")
    if any(not p.strip() for p in prompts):
        errors.append("At least one caption is blank.")

    print(f"Dataset root: {root.resolve()}")
    print(f"Captions: {len(prompts)}")
    print(f"Video paths: {len(videos)}")
    if errors:
        print("\nProblems:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Manifest looks structurally valid. This does not verify codec, resolution, frame count, or license.")
    print("Next, run the official trainer on a supported GPU machine.")


if __name__ == "__main__":
    main()
