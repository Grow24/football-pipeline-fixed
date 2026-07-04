"""
FastAPI wrapper around the football video-analysis pipeline (main-2-pose.py).

Designed for Zeabur (or any container host):
  - Listens on $PORT (default 8080).
  - Accepts a video upload, runs the pipeline in a background worker
    (as an isolated subprocess so a crash / OOM cannot take down the API),
    then exposes job status + downloadable artifacts.

Endpoints:
  GET  /                       -> simple HTML upload UI
  GET  /health                 -> health check for Zeabur
  POST /jobs                   -> upload video, start a job, returns {job_id}
  GET  /jobs/{job_id}          -> job status + available artifacts
  GET  /jobs/{job_id}/download/{artifact}
                               -> download an output file
                                  artifact in: video | csv | heatmap | pose
"""

import os
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# --------------------------------------------------------------------------- #
# Configuration (all overridable via Zeabur environment variables)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = BASE_DIR / "main-2-pose.py"

# Where uploads + generated artifacts live. On Zeabur the container fs is
# ephemeral; mount a volume here or swap to object storage for persistence.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/tmp/pipeline-jobs"))
DEVICE = os.environ.get("DEVICE", "cpu")
MODE = os.environ.get("MODE", "RADAR")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Football Pipeline API", version="1.0.0")

# --------------------------------------------------------------------------- #
# In-memory job registry.
# NOTE: single-instance only. For multiple replicas use Redis / a DB instead.
# --------------------------------------------------------------------------- #
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()

ARTIFACT_FILES = {
    "video": "output.mp4",
    "csv": "tracking.csv",
    "heatmap": "heatmap.png",
    "pose": "pose.csv",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _available_artifacts(job_dir: Path) -> Dict[str, bool]:
    return {
        name: (job_dir / filename).exists()
        for name, filename in ARTIFACT_FILES.items()
    }


def _transcode_to_h264(video_path: Path, log_path: Path) -> None:
    """Re-encode a video in place to browser-friendly H.264 (yuv420p, faststart).

    No-op if ffmpeg is missing or the file does not exist; the original mp4v
    file is kept in that case so nothing is lost.
    """
    if not video_path.exists() or shutil.which("ffmpeg") is None:
        return
    tmp_out = video_path.with_name(video_path.stem + "_h264.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        "-an",
        str(tmp_out),
    ]
    try:
        with open(log_path, "a") as log_file:
            log_file.write("\n=== ffmpeg H.264 re-encode ===\n")
            proc = subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False
            )
        if proc.returncode == 0 and tmp_out.exists():
            tmp_out.replace(video_path)
        else:
            tmp_out.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - keep original file on any failure
        tmp_out.unlink(missing_ok=True)


def _run_pipeline(job_id: str, job_dir: Path, source_path: Path) -> None:
    """Runs the pipeline as a subprocess and records the outcome."""
    output_video = job_dir / ARTIFACT_FILES["video"]
    csv_path = job_dir / ARTIFACT_FILES["csv"]
    heatmap_path = job_dir / ARTIFACT_FILES["heatmap"]
    pose_path = job_dir / ARTIFACT_FILES["pose"]
    log_path = job_dir / "run.log"

    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--source_video_path", str(source_path),
        "--target_video_path", str(output_video),
        "--device", DEVICE,
        "--mode", MODE,
        "--csv_path", str(csv_path),
        "--heatmap_path", str(heatmap_path),
        "--pose_path", str(pose_path),
    ]

    _set_job(job_id, status="running", started_at=_now(), command=" ".join(cmd))

    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode == 0:
            # supervision's VideoSink writes mp4v (MPEG-4 Part 2), which most
            # browsers/players refuse to play. Re-encode to H.264 + faststart
            # so the result is playable everywhere and guaranteed finalized.
            _set_job(job_id, status="encoding")
            _transcode_to_h264(output_video, log_path)
            _set_job(
                job_id,
                status="completed",
                finished_at=_now(),
                return_code=0,
                artifacts=_available_artifacts(job_dir),
            )
        else:
            _set_job(
                job_id,
                status="failed",
                finished_at=_now(),
                return_code=proc.returncode,
                error=f"Pipeline exited with code {proc.returncode}. See run.log.",
                artifacts=_available_artifacts(job_dir),
            )
    except Exception as exc:  # noqa: BLE001 - surface any launch failure
        _set_job(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc),
        )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "device": DEVICE, "mode": MODE}


@app.post("/jobs")
async def create_job(file: UploadFile = File(...)) -> JSONResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex
    job_dir = DATA_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source_path = job_dir / f"input{ext}"

    # Stream to disk with a size guard (avoid loading whole video in memory).
    size = 0
    try:
        with open(source_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_MB} MB limit.",
                    )
                out.write(chunk)
    finally:
        await file.close()

    _set_job(
        job_id,
        status="queued",
        created_at=_now(),
        filename=file.filename,
        size_bytes=size,
    )

    worker = threading.Thread(
        target=_run_pipeline,
        args=(job_id, job_dir, source_path),
        daemon=True,
    )
    worker.start()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/jobs/{job_id}",
        },
    )


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = DATA_ROOT / job_id
    # Only expose artifacts once the pipeline has finished. While the job is
    # still "running" the output.mp4 exists on disk but is not finalized yet
    # (VideoSink writes frames incrementally), so downloading it early yields
    # a corrupt file ("no playable streams").
    if job.get("status") == "completed" and job_dir.exists():
        job["artifacts"] = _available_artifacts(job_dir)
    else:
        job["artifacts"] = {name: False for name in ARTIFACT_FILES}
    job["job_id"] = job_id
    return job


@app.get("/jobs/{job_id}/download/{artifact}")
def download_artifact(job_id: str, artifact: str) -> FileResponse:
    if artifact not in ARTIFACT_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown artifact '{artifact}'. Allowed: {sorted(ARTIFACT_FILES)}",
        )
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is '{job.get('status')}'. Artifacts are only downloadable once the job is completed.",
        )
    job_dir = DATA_ROOT / job_id
    file_path = job_dir / ARTIFACT_FILES[artifact]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not available yet")
    return FileResponse(path=str(file_path), filename=ARTIFACT_FILES[artifact])


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Football Pipeline</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto;
           padding: 0 16px; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.5rem; }
    .card { background: #1e293b; border-radius: 12px; padding: 20px; margin-top: 16px; }
    input[type=file] { width: 100%; margin: 12px 0; color: #e2e8f0; }
    button { background: #ec4899; color: #fff; border: 0; border-radius: 8px;
             padding: 10px 18px; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    pre { background: #0f172a; border-radius: 8px; padding: 12px; overflow: auto;
          font-size: .85rem; }
    a { color: #38bdf8; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
    .muted { color: #94a3b8; font-size: .85rem; }
  </style>
</head>
<body>
  <h1>Football Pipeline &mdash; Soccer AI</h1>
  <div class="card">
    <p>Upload a match clip. Processing runs in the background (CPU: ~30&ndash;40 min per video).</p>
    <input id="file" type="file" accept="video/*" />
    <button id="upload">Upload &amp; Analyze</button>
    <p class="muted">Max upload size is set by the server (MAX_UPLOAD_MB).</p>
  </div>
  <div class="card" id="statusCard" style="display:none">
    <h3>Job status</h3>
    <p id="phase" class="muted">&mdash;</p>
    <pre id="status">&mdash;</pre>
    <div class="row" id="downloads"></div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
let pollTimer = null;

async function poll(jobId) {
  const res = await fetch(`/jobs/${jobId}`);
  const data = await res.json();
  $("status").textContent = JSON.stringify(data, null, 2);

  const phase = $("phase");
  if (data.status === "running") {
    phase.textContent = "Processing... please wait (CPU: ~30-40 min). Downloads appear when finished.";
  } else if (data.status === "encoding") {
    phase.textContent = "Finalizing video (H.264 re-encode)... almost done.";
  } else if (data.status === "queued") {
    phase.textContent = "Queued...";
  } else if (data.status === "completed") {
    phase.textContent = "Done! Download your results below.";
  } else if (data.status === "failed") {
    phase.textContent = "Job failed. Check server run.log for details.";
  }

  const dl = $("downloads");
  dl.innerHTML = "";
  const arts = data.artifacts || {};
  for (const [name, ready] of Object.entries(arts)) {
    if (ready) {
      const a = document.createElement("a");
      a.href = `/jobs/${jobId}/download/${name}`;
      a.textContent = `Download ${name}`;
      dl.appendChild(a);
    }
  }
  if (data.status === "completed" || data.status === "failed") {
    clearInterval(pollTimer);
  }
}

$("upload").addEventListener("click", async () => {
  const f = $("file").files[0];
  if (!f) { alert("Choose a video first."); return; }
  $("upload").disabled = true;
  $("statusCard").style.display = "block";
  $("status").textContent = "Uploading...";

  const fd = new FormData();
  fd.append("file", f);
  try {
    const res = await fetch("/jobs", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { $("status").textContent = JSON.stringify(data, null, 2); return; }
    const jobId = data.job_id;
    $("status").textContent = "Queued: " + jobId;
    pollTimer = setInterval(() => poll(jobId), 3000);
    poll(jobId);
  } catch (e) {
    $("status").textContent = "Error: " + e;
  } finally {
    $("upload").disabled = false;
  }
});
</script>
</body>
</html>"""
