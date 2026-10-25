# Text-to-video API

This API calls the official Open-Sora 2.0 inference CLI through a subprocess adapter. It accepts text, queues generation, returns a job ID, and serves the resulting MP4 when ready. It does not require input video paths for generation.

## Hardware

Run on a compatible Linux machine with an NVIDIA CUDA GPU. It is not a practical native Android/Termux inference server. The API can be started on a CPU-only machine for endpoint inspection, but actual generation will fail with a clear CUDA requirement message.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-gpu.txt
export PYTHONPATH="$PWD/src"
./run_api.sh
```

`requirements-gpu.txt` is the legacy CogVideoX dependency list; Open-Sora 2.0 must be installed separately using the official repository instructions and `scripts/setup_opensora_gpu.sh`. For production GPU deployments, install a PyTorch build matching the CUDA environment from the official PyTorch installation selector before installing the rest of the ML dependencies if needed.

Open interactive API docs at `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /health` — API status
- `GET /prompts?offset=0&limit=50` — browse prompt library (maximum 100 per request)
- `GET /prompts/1` — get prompt 1
- `POST /generate` — submit a text-to-video generation job
- `GET /jobs/{job_id}` — poll generation status
- `GET /jobs/{job_id}/video` — download generated MP4 after completion

### Generate with custom text

```bash
curl -X POST http://127.0.0.1:8000/generate   -H 'Content-Type: application/json'   -d '{"prompt":"A cartoon cat runs toward the camera, stops, and waves.","steps":30,"seed":42}'
```

### Generate using one of the 450 prompts

```bash
curl -X POST http://127.0.0.1:8000/generate   -H 'Content-Type: application/json'   -d '{"prompt_id":1,"steps":30,"seed":42}'
```

The response contains `job_id` and `status_url`. Poll the status:

```bash
curl http://127.0.0.1:8000/jobs/JOB_ID
```

When status is `completed`, download the video:

```bash
curl -L http://127.0.0.1:8000/jobs/JOB_ID/video -o generated.mp4
```

### Python client example

```python
import time
import requests

base = "http://127.0.0.1:8000"
response = requests.post(f"{base}/generate", json={
    "prompt_id": 1,
    "steps": 30,
    "seed": 42,
})
response.raise_for_status()
job_id = response.json()["job_id"]

while True:
    job = requests.get(f"{base}/jobs/{job_id}").json()
    print(job["status"])
    if job["status"] in {"completed", "failed"}:
        break
    time.sleep(5)

if job["status"] == "completed":
    video = requests.get(f"{base}/jobs/{job_id}/video")
    video.raise_for_status()
    open("generated.mp4", "wb").write(video.content)
else:
    print(job.get("error"))
```

## Important notes

- This uses the open-source Open-Sora 2.0 model, not an OpenAI/Sora model. See README_OPENSORA2_MIGRATION.md for the GPU-worker requirements.
- Initial model download and inference require substantial disk space and GPU memory.
- Jobs are stored in memory; restarting the API loses job status. Use a durable queue/database for production.
- Bind to localhost or protect the service with authentication and HTTPS before exposing it to the internet. This starter intentionally has no public authentication.
- Do not put secret tokens into prompts or commit them to source control.
