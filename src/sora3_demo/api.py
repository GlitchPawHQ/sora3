"""FastAPI service for Open-Sora 2.0 video generation."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from sora3_demo.prompt_library import PROMPTS

APP_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.getenv("SORA3_OUTPUT_DIR", str(APP_ROOT / "outputs"))).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = os.getenv("SORA3_MODEL_ID", "hpcai-tech/Open-Sora-v2 (11B)")
LORA_PATH = os.getenv("SORA3_LORA_PATH") or None
MAX_WORKERS = max(1, int(os.getenv("SORA3_MAX_GENERATIONS", "1")))
GENERATION_ENABLED = os.getenv("SORA3_GENERATION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

app = FastAPI(
    title="Sora3 Demo Text-to-Video API",
    description=(
        "Experimental text-to-video API using the official Open-Sora 2.0 inference CLI. "
        "This is not an OpenAI/Sora model. Generation requires a compatible GPU worker."
    ),
    version="1.0.0",
)
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Persistent job records.
STATE_DIR = Path(
    os.getenv("SORA3_STATE_DIR", str(APP_ROOT / "job_state"))
).resolve()
STATE_DIR.mkdir(parents=True, exist_ok=True)

jobs: dict[str, dict] = {}
lock = Lock()


def save_job(job: dict) -> None:
    """Atomically save a job without exposing partially written JSON."""
    job_file = STATE_DIR / f"{job['job_id']}.json"
    temp_file = STATE_DIR / f"{job['job_id']}.tmp"

    with temp_file.open("w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_file, job_file)


def load_jobs() -> None:
    """Restore jobs after an API restart."""
    for job_file in STATE_DIR.glob("*.json"):
        try:
            with job_file.open("r", encoding="utf-8") as f:
                job = json.load(f)

            job_id = job.get("job_id")
            if not job_id or job_file.name != f"{job_id}.json":
                continue

            # A job interrupted by a process restart must not remain
            # permanently queued or running.
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="failed",
                    error="Generation was interrupted by an API restart.",
                    completed_at=now(),
                )
                save_job(job)

            jobs[job_id] = job
        except (OSError, ValueError, TypeError) as exc:
            print(f"Could not restore job {job_file.name}: {exc}")


load_jobs()


class GenerateRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=3, max_length=2000)
    prompt_id: int | None = Field(default=None, ge=1, le=450)
    num_frames: int = Field(default=49, ge=9, le=49)
    fps: int = Field(default=8, ge=4, le=24)
    steps: int = Field(default=30, ge=1, le=50)
    guidance_scale: float = Field(default=6.0, ge=1.0, le=12.0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

class GenerateResponse(BaseModel):
    job_id: str
    status: str
    prompt: str
    status_url: str

def now():
    return datetime.now(timezone.utc).isoformat()


def run_job(job_id: str, prompt: str, params: dict):
    try:
        with lock:
            jobs[job_id].update(
                status="running",
                started_at=now(),
            )
            save_job(jobs[job_id])

        from sora3_demo.generation import generate_video

        output = OUTPUT_DIR / f"{job_id}.mp4"
        saved = generate_video(
            prompt=prompt,
            output_path=str(output),
            model_id=MODEL_ID,
            lora_path=LORA_PATH,
            num_frames=params["num_frames"],
            fps=params["fps"],
            num_inference_steps=params["steps"],
            guidance_scale=params["guidance_scale"],
            seed=params["seed"],
        )

        saved_path = Path(saved).resolve()
        if not saved_path.is_file():
            raise RuntimeError("Generation returned without creating a video file.")

        with lock:
            jobs[job_id].update(
                status="completed",
                output_path=str(saved_path),
                completed_at=now(),
            )
            save_job(jobs[job_id])

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

        with lock:
            if job_id in jobs:
                jobs[job_id].update(
                    status="failed",
                    error=error,
                    completed_at=now(),
                )
                save_job(jobs[job_id])


@app.get("/", response_class=HTMLResponse)
def root():
    homepage = APP_ROOT / "static" / "index.html"
    if homepage.exists():
        return HTMLResponse(homepage.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Sora3 Demo</h1><p>Homepage file is missing. Open <a href='/docs'>API docs</a>.</p>",
        status_code=200,
    )

@app.get("/health")
def health():
    with lock:
        queued = sum(1 for j in jobs.values() if j["status"] == "queued")
        running = sum(1 for j in jobs.values() if j["status"] == "running")
    return {"status": "ok", "model_id": MODEL_ID, "gpu_generation": "requires CUDA GPU",
            "queued_jobs": queued, "running_jobs": running,
            "generation_enabled": GENERATION_ENABLED}

@app.get("/prompts")
def list_prompts(offset: int = 0, limit: int = 50):
    if offset < 0 or limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="offset must be >= 0 and limit must be 1..100")
    items = [
        {"id": int(pid), "prompt": prompt}
        for pid, prompt in PROMPTS.items()
        if offset <= int(pid) - 1 < offset + limit
    ]
    return {"total": len(PROMPTS), "offset": offset, "limit": limit, "items": items}

@app.get("/prompts/{prompt_id}")
def get_prompt(prompt_id: int):
    prompt = PROMPTS.get(str(prompt_id))
    if prompt is None:
        raise HTTPException(status_code=404, detail="Prompt ID must be between 1 and 450")
    return {"id": prompt_id, "prompt": prompt}

@app.post("/generate", response_model=GenerateResponse, status_code=202)
def submit_generation(request: GenerateRequest):
    if not GENERATION_ENABLED:
        raise HTTPException(status_code=503, detail="Video generation is disabled on this host. This deployment can serve the API and prompt library, but generation requires a separate compatible GPU worker.")
    if not request.prompt and request.prompt_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'prompt' or 'prompt_id'.")
    if request.prompt and request.prompt_id is not None:
        raise HTTPException(status_code=422, detail="Provide 'prompt' or 'prompt_id', not both.")
    prompt = request.prompt.strip() if request.prompt else PROMPTS[str(request.prompt_id)]
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty.")
    job_id = uuid4().hex
    params = {
        "num_frames": request.num_frames, "fps": request.fps, "steps": request.steps,
        "guidance_scale": request.guidance_scale, "seed": request.seed,
    }
   
    with lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "prompt": prompt,
            "parameters": params,
            "created_at": now(),
        }
        save_job(jobs[job_id])

    try:
        executor.submit(run_job, job_id, prompt, params)
    except Exception as exc:
        with lock:
            jobs[job_id].update(
                status="failed",
                error=f"Could not schedule generation: {type(exc).__name__}: {exc}",
                completed_at=now(),
            )
            save_job(jobs[job_id])

    return GenerateResponse(
        job_id=job_id,
        status=jobs[job_id]["status"],
        prompt=prompt,
        status_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found on this API instance.")
        result = dict(job)
    if result.get("status") == "completed":
        result["download_url"] = f"/jobs/{job_id}/video"
    return result

@app.get("/jobs/{job_id}/video")
def download_video(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.get("status") != "completed":
            raise HTTPException(status_code=409, detail=f"Video is not ready; current status: {job.get('status')}")
        path = Path(job["output_path"])
    if not path.is_file() or not path.resolve().is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=404, detail="Generated file is missing.")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
