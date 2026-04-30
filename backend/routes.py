"""
routes.py
All HTTP endpoints for the Phase 7 API.

Endpoints (mounted under /api in main.py):
  POST   /upload                         → start a new job, returns {job_id}
  GET    /jobs/{job_id}                  → poll job status snapshot
  GET    /jobs/{job_id}/stream           → SSE live progress stream
  GET    /jobs/{job_id}/result           → final InferenceResult JSON
  GET    /jobs/{job_id}/artifacts/{name} → download / stream an artifact file
  GET    /healthz                        → liveness + model-loaded flag
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, File, Form, HTTPException,
    Request, UploadFile,
)
from fastapi.responses import (
    FileResponse, JSONResponse, Response, StreamingResponse,
)

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

# Artifacts that browsers need to seek (Range request support)
RANGE_SUPPORTED = {"output_skeleton_overlay.mp4"}

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
    with input_path.open("wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

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
            "X-Accel-Buffering": "no",
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
async def get_artifact(name: str, job_id: str, request: Request):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    if name not in ARTIFACT_MIME:
        raise HTTPException(status_code=404, detail=f"Unknown artifact '{name}'")

    path = Path(job.output_dir) / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not yet generated: {name}")

    mime = ARTIFACT_MIME[name]

    # ── Byte-range support for video (required for browser <video> seeking) ───
    if name in RANGE_SUPPORTED:
        return _range_response(path, mime, request)

    # ── Non-video artifacts: simple FileResponse ──────────────────────────────
    return FileResponse(path, media_type=mime, filename=name)


def _range_response(path: Path, mime: str, request: Request) -> Response:
    """
    Serve a file with HTTP 206 Partial Content support.
    Browsers send Range: bytes=0- for video; without this they can't seek.
    """
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        # No Range header → serve the whole file with Accept-Ranges declared
        return _full_video_response(path, mime, file_size)

    # Parse "bytes=start-end"
    try:
        range_val = range_header.strip().replace("bytes=", "")
        start_str, _, end_str = range_val.partition("-")
        start = int(start_str) if start_str else 0
        end   = int(end_str)   if end_str   else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start > end or start >= file_size:
        raise HTTPException(
            status_code=416,
            detail="Range Not Satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    def iter_file():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            buf = 1 << 16  # 64 KB chunks
            while remaining > 0:
                data = f.read(min(buf, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=mime,
        headers={
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk_size),
            "Accept-Ranges":  "bytes",
            "Cache-Control":  "no-cache",
        },
    )


def _full_video_response(path: Path, mime: str, file_size: int) -> Response:
    def iter_file():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 16)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        iter_file(),
        status_code=200,
        media_type=mime,
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges":  "bytes",
            "Cache-Control":  "no-cache",
        },
    )