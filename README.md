# Football Pipeline - grow24.ai

## Fixes applied
- Phase 1: Correct axis scaling (105/12000, 68/7000)
- Phase 2: OOB detection filtering
- Phase 3: Robust homography (min 6 keypoints, fallback transformer)

## Setup
apt-get update && apt-get install -y python3 python3-pip ffmpeg libgl1
pip3 install -r requirements.txt

## Run
python3 main-2-pose.py \
  --source_video_path video.mp4 \
  --target_video_path output.mp4 \
  --device cpu \
  --mode RADAR \
  --csv_path tracking.csv

## Requirements
- Python 3.10+
- 8GB RAM minimum
- No GPU needed (CPU works, ~30-40 min per video)

## Web API (for Zeabur / container deploy)
`app.py` is a FastAPI wrapper. It accepts a video upload, runs the pipeline in a
background subprocess, and exposes status + downloadable artifacts.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8080
Then open http://localhost:8080 for the upload UI.

Endpoints:
- `GET  /`                              simple upload UI
- `GET  /health`                        health check
- `POST /jobs`                          upload video -> {job_id}
- `GET  /jobs/{job_id}`                 status + artifacts
- `GET  /jobs/{job_id}/download/{name}` name in: video | csv | heatmap | pose

## Deploy on Zeabur
1. Push this folder to a GitHub repo (models in `data/*.pt` are large -> use Git LFS).
2. Zeabur -> New Service -> Deploy from GitHub -> pick the repo.
   If it lives in a subfolder, set the service **Root Directory** to it.
3. Zeabur auto-detects the `Dockerfile` and builds it (ffmpeg + libgl1 included).
4. Zeabur injects `PORT`; the container listens on it automatically.
5. (Recommended) Mount a Volume at `/data` so uploads/outputs survive restarts.

### Environment variables (Zeabur -> Variables tab)
| Key            | Default          | Purpose                                  |
|----------------|------------------|------------------------------------------|
| `PORT`         | (Zeabur sets it) | HTTP port the server binds to            |
| `DEVICE`       | `cpu`            | Inference device (`cpu` / `cuda`)        |
| `MODE`         | `RADAR`          | Pipeline mode                            |
| `DATA_ROOT`    | `/data`          | Where uploads + artifacts are stored     |
| `MAX_UPLOAD_MB`| `500`            | Max upload size in MB                     |

Resources: pick a plan with **8GB+ RAM** and a few CPU cores. Processing a video
takes ~30-40 min on CPU, so use the async job flow (upload -> poll `/jobs/{id}`).
