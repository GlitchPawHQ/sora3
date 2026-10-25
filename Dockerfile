FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ffmpeg git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN python3 -m pip install --break-system-packages -r requirements-gpu.txt
ENV PYTHONPATH=/app/src SORA3_OUTPUT_DIR=/app/outputs
EXPOSE 8000
CMD ["sh", "-c", "uvicorn sora3_demo.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
