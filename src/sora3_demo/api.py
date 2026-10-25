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
jobs: dict[str, dict] = {}
lock = Lock()

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
            jobs[job_id]["status"] = "running"
            jobs[job_id]["started_at"] = now()
        # Heavy model import happens only inside the background worker.
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
        with lock:
            jobs[job_id].update(status="completed", output_path=saved, completed_at=now())
    except Exception as exc:
        with lock:
            jobs[job_id].update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                completed_at=now(),
            )

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
            "job_id": job_id, "status": "queued", "prompt": prompt, "parameters": params,
            "created_at": now(),
        }
    executor.submit(run_job, job_id, prompt, params)
    return GenerateResponse(job_id=job_id, status="queued", prompt=prompt,
                            status_url=f"/jobs/{job_id}")

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
    if not path.is_file() or path.parent != OUTPUT_DIR:
        raise HTTPException(status_code=404, detail="Generated file is missing.")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
