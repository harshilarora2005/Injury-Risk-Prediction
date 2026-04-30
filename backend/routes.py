"""
routes.py
All HTTP endpoints for the Phase 7 API.

Endpoints (mounted under /api in main.py):
  POST   /upload                         → start a new job, returns {job_id}
  GET    /jobs/{job_id}                  → poll job status snapshot
  GET    /jobs/{job_id}/stream           → SSE live progress stream
  GET    /jobs/{job_id}/result           → final InferenceResult JSON
  GET    /jobs/{job_id}/artifacts/{name} → download an artifact file
  GET    /healthz                        → liveness + model-loaded flag
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from inference import load_model
from pipeline import JOBS, Job, run_pipeline
from schemas import JobStatus

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = (BASE_DIR.parent / "outputs").resolve()
UPLOADS_DIR = (BASE_DIR.parent / "uploads").resolve()
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_CAMERA_ANGLES = {"front", "sagittal"}

ARTIFACT_MIME = {
    "output_skeleton_overlay.mp4":  "video/mp4",
    "risk_timeline.png":            "image/png",
    "movement_annotations.txt":     "text/plain",
    "per_window_predictions.csv":   "text/csv",
    "summary_report.pdf":           "application/pdf",
}

router = APIRouter()


# ── Health check ──────────────────────────────────────────────────────────────
@router.get("/healthz")
def healthz():
    model_loaded = True
    err = None
    try:
        load_model()
    except Exception as e:
        model_loaded = False
        err = str(e)
    return {"ok": True, "model_loaded": model_loaded, "error": err}


# ── POST /upload ──────────────────────────────────────────────────────────────
@router.post("/upload")
async def upload_clip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_angle: str = Form("sagittal"),
):
    """Accept a video upload, kick off the pipeline, return the job_id."""
    if camera_angle not in ALLOWED_CAMERA_ANGLES:
        raise HTTPException(
            status_code=400,
            detail=f"camera_angle must be one of {sorted(ALLOWED_CAMERA_ANGLES)}. "
                   f"Oblique-angle clips invalidate the 2D angle math.",
        )

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video extension '{ext}'. Allowed: {sorted(ALLOWED_EXTS)}",
        )

    job_id = uuid.uuid4().hex
    job_dir = OUTPUTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    input_path = UPLOADS_DIR / f"{job_id}{ext}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job = Job(
        job_id=job_id,
        filename=safe_name,
        input_path=str(input_path),
        output_dir=str(job_dir),
        camera_angle=camera_angle,
    )
    JOBS[job_id] = job
    job.update(stage="queued", progress=0, message="Upload received, queued for processing")

    background_tasks.add_task(run_pipeline, job_id)
    return {"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job.snapshot()


# ── GET /jobs/{job_id}/stream  (Server-Sent Events) ───────────────────────────
@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    if JOBS.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    async def event_generator():
        last_payload = None
        while True:
            j = JOBS.get(job_id)
            if j is None:
                break
            payload = json.dumps(j.snapshot())
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if j.stage in ("done", "error"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable proxy buffering
        },
    )


# ── GET /jobs/{job_id}/result ─────────────────────────────────────────────────
@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if job.stage == "error":
        raise HTTPException(status_code=500, detail=job.error or "Pipeline failed")
    if job.stage != "done" or job.result is None:
        raise HTTPException(status_code=409, detail=f"Job not finished (stage={job.stage})")
    return JSONResponse(job.result)


# ── GET /jobs/{job_id}/artifacts/{name} ───────────────────────────────────────
@router.get("/jobs/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if name not in ARTIFACT_MIME:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{name}'")
    path = Path(job.output_dir) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not yet generated: {name}")
    return FileResponse(path, media_type=ARTIFACT_MIME[name], filename=name)
