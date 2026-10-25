# Sora3 Demo — Render deployment

Experimental FastAPI homepage and prompt-library API with an Open-Sora 2.0 inference adapter. **This is not an OpenAI/Sora model.**

## Deploy to Render

1. Upload the contents of this ZIP to a new GitHub repository (the files must be at the repository root).
2. In Render, choose **New → Blueprint** and connect that repository, or create a **Web Service** manually.
3. For a manual service use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy. Render provides an HTTPS URL when the service is ready.

`render.yaml` contains the same basic service configuration.

## Endpoints

- `/` — homepage
- `/health` — health check
- `/docs` — interactive API docs
- `/prompts` — prompt library
- `/prompts/1` — example prompt

## Important limitations

- `SORA3_GENERATION_ENABLED` is set to `false` by default so the free web service can serve the homepage and API without trying to load a large model. The `/generate` endpoint returns HTTP 503 while disabled.
- Actual Open-Sora 2.0 (11B) video generation requires a compatible GPU worker and substantial GPU memory; the free Render web service is not sufficient. See README_OPENSORA2_MIGRATION.md.
- Render free services may sleep when idle.
- Do not commit API keys, passwords, or private tokens.
