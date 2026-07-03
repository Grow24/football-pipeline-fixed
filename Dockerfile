FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DEVICE=cpu \
    MODE=RADAR \
    DATA_ROOT=/data \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# System deps: ffmpeg (video I/O) + libgl1/libglib (OpenCV runtime) + git (pip VCS install)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Persist uploads/artifacts across restarts if a volume is mounted here.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8080

# $PORT is injected by Zeabur; default 8080 for local runs.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
