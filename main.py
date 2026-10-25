"""Pella-compatible ASGI entry point.

If API dependencies are missing, install requirements.txt using pip, then
import and expose the FastAPI application as `app`.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# Ensure local package imports work regardless of the platform's working directory.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Some hosts provide only Main File + App Entry and no install/start command.
# Install dependencies before importing the FastAPI application.
if importlib.util.find_spec("fastapi") is None:
    if not REQUIREMENTS.is_file():
        raise RuntimeError(f"Missing dependency file: {REQUIREMENTS}")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--disable-pip-version-check", "-r", str(REQUIREMENTS)
    ])

# Keep generation disabled by default on low-memory hosting.
os.environ.setdefault("SORA3_GENERATION_ENABLED", "true")


# Install Open-Sora 2.0 source and package if missing.
import subprocess
from pathlib import Path
import os

OPEN_SORA_DIR = Path(
    os.environ.get("OPEN_SORA_DIR", "/opt/Open-Sora")
).resolve()

if not (OPEN_SORA_DIR / "opensora").is_dir():
    OPEN_SORA_DIR.parent.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([
        "git", "clone",
        "--depth", "1",
        "https://github.com/hpcaitech/Open-Sora.git",
        str(OPEN_SORA_DIR),
    ])

# Install the actual Open-Sora Python package.
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "--disable-pip-version-check",
    "-v", ".",
], cwd=str(OPEN_SORA_DIR))

os.environ["OPEN_SORA_DIR"] = str(OPEN_SORA_DIR)


from sora3_demo.api import app  # noqa: E402,F401
