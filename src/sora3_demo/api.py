
"""FastAPI service for Open-Sora 2.0 video generation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
import json
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from sora3_demo.prompt_library import PROMPTS


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[2]


def configured_path(variable: str, default: Path) -> Path:
    """Resolve a configured path and create its directory."""
    value = os.getenv(variable)
    path = Path(value).expanduser() if value else default
    path = path.resolve()

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create directory for {variable}: {path}. "
            "Check the configured path and its write permissions."
        ) from exc

    if not path.is_dir():
        raise RuntimeError(f"{variable} must point to a directory: {path}")

    return path


OUTPUT_DIR = configured_path(
    "SORA3_OUTPUT_DIR",
    APP_ROOT / "outputs",
)

STATE_DIR = configured_path(
    "SORA3_STATE_DIR",
    APP_ROOT / "job_state",
)

MODEL_ID = os.getenv(
    "SORA3_MODEL_ID",
    "hpcai-tech/Open-Sora-v2",
)

LORA_PATH = os.getenv("SORA3_LORA_PATH") or None

MAX_WORKERS = max(
    1,
    int(os.getenv("SORA3_MAX_GENERATIONS", "1")),
)

GENERATION_ENABLED = os.getenv(
    "SORA3_GENERATION_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

# Set SORA3_CORS_ORIGINS to a comma-separated list of actual
# frontend origins if the frontend is hosted on another domain.
DEFAULT_CORS_ORIGINS = (
    "https://sora-3-l49n.onrender.com,"
    "https://sora3-unreleased-demo.onrender.com"
)

CORS_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "SORA3_CORS_ORIGINS",
        DEFAULT_CORS_ORIGINS,
    ).split(",")
    if origin.strip()
]


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

app = FastAPI(
    title="Sora3 Demo Text-to-Video API",
    description=(
        "Experimental text-to-video API using an existing Open-Sora "
        "generation implementation. This is not an OpenAI/Sora model. "
        "Actual generation requires a compatible GPU and installed "
        "inference dependencies."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

jobs: dict[str, dict] = {}
lock = Lock()


def now() -> str:
    """Return a UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------
# Job persistence
# ---------------------------------------------------------------------

def save_job(job: dict) -> None:
    """Atomically save a job record."""
    job_id = job["job_id"]
    job_file = STATE_DIR / f"{job_id}.json"
    temp_file = STATE_DIR / f"{job_id}.{uuid4().hex}.tmp"

    try:
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(job, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_file, job_file)

    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass


def load_jobs() -> None:
    """Restore saved jobs after an API restart."""
    for job_file in STATE_DIR.glob("*.json"):
        try:
            with job_file.open("r", encoding="utf-8") as file:
                job = json.load(file)

            job_id = job.get("job_id")

            if not isinstance(job_id, str) or not job_id:
                continue

            if job_file.name != f"{job_id}.json":
                continue

            # A process restart cannot resume an in-memory task.
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="failed",
                    error=(
                        "Generation was interrupted by an API restart. "
                        "Submit a new job."
                    ),
                    completed_at=now(),
                )

                save_job(job)

            jobs[job_id] = job

        except (OSError, ValueError, TypeError) as exc:
            print(
                f"Could not restore job {job_file.name}: {exc}",
                flush=True,
            )


load_jobs()


# ---------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str | None = Field(
        default=None,
        min_length=3,
        max_length=2000,
    )

    prompt_id: int | None = Field(
        default=None,
        ge=1,
        le=450,
    )

    num_frames: int = Field(default=49, ge=9, le=49)
    fps: int = Field(default=8, ge=4, le=24)
    steps: int = Field(default=30, ge=1, le=50)

    guidance_scale: float = Field(
        default=6.0,
        ge=1.0,
        le=12.0,
    )

    seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
    )


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    prompt: str
    status_url: str


# ---------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------

def run_job(job_id: str, prompt: str, params: dict) -> None:
    """Run genuine video generation using the project's generation module."""
    try:
        with lock:
            job = jobs.get(job_id)

            if job is None:
                return

            job.update(
                status="running",
                started_at=now(),
            )
            save_job(job)

        # This module must implement actual Open-Sora inference.
        # It must not return a placeholder or sample video.
        from sora3_demo.generation import generate_video

        output_path = (OUTPUT_DIR / f"{job_id}.mp4").resolve()

        if not output_path.is_relative_to(OUTPUT_DIR.resolve()):
            raise RuntimeError("Invalid output path.")

        saved = generate_video(
            prompt=prompt,
            output_path=str(output_path),
            model_id=MODEL_ID,
            lora_path=LORA_PATH,
            num_frames=params["num_frames"],
            fps=params["fps"],
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            seed=params["seed"],
        )

        if not saved:
            raise RuntimeError(
                "The generation module did not return an output path."
            )

        saved_path = Path(saved).resolve()

        if not saved_path.is_file():
            raise RuntimeError(
                "Generation finished without creating a video file."
            )

        if not saved_path.is_relative_to(OUTPUT_DIR.resolve()):
            raise RuntimeError(
                "The generated video is outside the configured output directory."
            )

        if saved_path.stat().st_size == 0:
            raise RuntimeError("The generated video file is empty.")

        with lock:
            job = jobs.get(job_id)

            if job is not None:
                job.update(
                    status="completed",
                    output_path=str(saved_path),
                    completed_at=now(),
                )
                save_job(job)

    except Exception as exc:
        traceback.print_exc()

        error_message = f"{type(exc).__name__}: {exc}"

        with lock:
            job = jobs.get(job_id)

            if job is not None:
                job.update(
                    status="failed",
                    error=error_message[:2000],
                    completed_at=now(),
                )

                try:
                    save_job(job)
                except OSError:
                    traceback.print_exc()


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def root():
    homepage = APP_ROOT / "static" / "index.html"

    if homepage.is_file():
        return HTMLResponse(
            homepage.read_text(encoding="utf-8")
        )

    return HTMLResponse(
        "<h1>Sora3 Demo</h1>"
        "<p>Homepage file is missing. Open <a href='/docs'>API docs</a>.</p>",
        status_code=200,
    )


@app.get("/health")
def health():
    with lock:
        queued = sum(
            1 for job in jobs.values()
            if job.get("status") == "queued"
        )

        running = sum(
            1 for job in jobs.values()
            if job.get("status") == "running"
        )

    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "gpu_generation": "requires a compatible GPU",
        "queued_jobs": queued,
        "running_jobs": running,
        "generation_enabled": GENERATION_ENABLED,
    }


@app.get("/prompts")
def list_prompts(offset: int = 0, limit: int = 50):
    if offset < 0 or limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="offset must be >= 0 and limit must be 1..100",
        )

    items = [
        {"id": int(prompt_id), "prompt": prompt}
        for prompt_id, prompt in PROMPTS.items()
        if offset <= int(prompt_id) - 1 < offset + limit
    ]

    return {
        "total": len(PROMPTS),
        "offset": offset,
        "limit": limit,
        "items": items,
    }


@app.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: int):
    prompt = PROMPTS.get(str(prompt_id))

    if prompt is None:
        raise HTTPException(
            status_code=404,
            detail="Prompt ID was not found.",
        )

    return {"id": prompt_id, "prompt": prompt}


@app.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=202,
)
def submit_generation(request: GenerateRequest):
    if not GENERATION_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generation is disabled on this API. Enable it only "
                "when a compatible GPU and the real Open-Sora inference "
                "implementation are available."
            ),
        )

    if bool(request.prompt) == (request.prompt_id is not None):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'prompt' or 'prompt_id'.",
        )

    if request.prompt is not None:
        prompt = request.prompt.strip()
    else:
        prompt = PROMPTS.get(str(request.prompt_id), "")

    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Prompt cannot be empty or the prompt ID is invalid.",
        )

    job_id = uuid4().hex

    params = {
        "num_frames": request.num_frames,
        "fps": request.fps,
        "steps": request.steps,
        "guidance_scale": request.guidance_scale,
        "seed": request.seed,
    }

    job = {
        "job_id": job_id,
        "status": "queued",
        "prompt": prompt,
        "parameters": params,
        "created_at": now(),
    }

    with lock:
        jobs[job_id] = job

        try:
            save_job(job)
        except OSError as exc:
            jobs.pop(job_id, None)
            raise HTTPException(
                status_code=500,
                detail=f"Could not save the job record: {exc}",
            ) from exc

    try:
        executor.submit(run_job, job_id, prompt, params)

    except Exception as exc:
        with lock:
            jobs[job_id].update(
                status="failed",
                error=f"Could not schedule generation: {exc}",
                completed_at=now(),
            )
            save_job(jobs[job_id])

        raise HTTPException(
            status_code=503,
            detail="Could not schedule the generation job.",
        ) from exc

    return GenerateResponse(
        job_id=job_id,
        status="queued",
        prompt=prompt,
        status_url=f"/jobs/{job_id}",
    )



@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)

    # If the job is missing from memory, try restoring it from disk.
    if job is None:
        job_file = STATE_DIR / f"{job_id}.json"

        try:
            with job_file.open("r", encoding="utf-8") as file:
                stored_job = json.load(file)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Job not found in memory or on disk. "
                    "The job state may have been lost during deployment."
                ),
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not restore job record: {exc}",
            ) from exc

        if stored_job.get("job_id") != job_id:
            raise HTTPException(
                status_code=500,
                detail="Stored job ID does not match the requested ID.",
            )

        # An in-memory task cannot survive an API process restart.
        if stored_job.get("status") in {"queued", "running"}:
            stored_job.update(
                status="failed",
                error=(
                    "Generation was interrupted by an API restart. "
                    "Submit a new job."
                ),
                completed_at=now(),
            )
            save_job(stored_job)

        with lock:
            job = jobs.setdefault(job_id, stored_job)

    result = dict(job)

    if result.get("status") == "completed":
        result["download_url"] = f"/jobs/{job_id}/video"

    return result



@app.get("/jobs/{job_id}/video")
def download_video(job_id: str):
    with lock:
        job = jobs.get(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        if job.get("status") != "completed":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Video is not ready; current status: "
                    f"{job.get('status')}"
                ),
            )

        stored_path = job.get("output_path")

    if not stored_path:
        raise HTTPException(
            status_code=404,
            detail="The job has no output file.",
        )

    path = Path(stored_path).resolve()

    if not path.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(
            status_code=403,
            detail="The output path is not permitted.",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Generated video file is missing.",
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )
