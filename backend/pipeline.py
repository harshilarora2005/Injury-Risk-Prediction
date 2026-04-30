"""
pipeline.py
Phase 7 pipeline orchestration + in-memory Job registry.

Pipeline (per Phase 7 workflow doc):
  1. Pose extraction        (feature_engineering.extract_features_from_video)
  2. Feature engineering    (same call — returns (N, 8) matrix + landmarks)
  3. Window inference       (inference.run_inference)            → CSV artifact
  4. Frame-level assignment (utils.assign_frame_labels + rolling_mode_smooth)
  5. Skeleton overlay       (overlay.render_overlay_video)       → MP4 artifact
  6. Risk timeline          (timeline.generate_timeline)         → PNG artifact
  7. Annotations            (annotations.generate_annotations)   → TXT artifact
  8. Summary report         (report.generate_report)             → PDF artifact
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from feature_engineering import extract_features_from_video
from inference import load_model, validate_input, run_inference
from overlay import render_overlay_video
from timeline import generate_timeline
from annotations import generate_annotations
from report import generate_report
from schemas import (
    AnnotatedEvent,
    InferenceResult,
    RiskSummary,
    WindowPrediction,
)
from utils import (
    assign_frame_labels,
    build_risk_summary,
    extract_high_events,
    rolling_mode_smooth,
)

log = logging.getLogger("phase7.pipeline")


# ── In-memory job registry ────────────────────────────────────────────────────
@dataclass
class Job:
    job_id: str
    filename: str
    input_path: str
    output_dir: str
    camera_angle: str
    stage: str = "queued"
    progress: int = 0
    message: str = "Job queued"
    result: Optional[dict] = None
    error: Optional[str] = None
    updated_at: float = field(default_factory=time.time)

    def update(self, *, stage: str = None, progress: int = None, message: str = None):
        if stage is not None:
            self.stage = stage
        if progress is not None:
            self.progress = max(0, min(100, int(progress)))
        if message is not None:
            self.message = message
        self.updated_at = time.time()
        log.info(f"[{self.job_id[:8]}] {self.stage} {self.progress}% — {self.message}")

    def snapshot(self) -> dict:
        return {
            "job_id":   self.job_id,
            "stage":    self.stage,
            "progress": self.progress,
            "message":  self.message,
            "result":   self.result,
            "error":    self.error,
        }


JOBS: Dict[str, Job] = {}


# ── Pipeline orchestration ────────────────────────────────────────────────────
def run_pipeline(job_id: str) -> None:
    """Synchronous pipeline executed in a background thread."""
    job = JOBS[job_id]
    out_dir = Path(job.output_dir)

    def progress(pct: int, msg: str, stage: Optional[str] = None):
        job.update(progress=pct, message=msg, stage=stage or job.stage)

    try:
        # ── 0. Load model ──
        progress(2, "Loading model", stage="model")
        model, meta = load_model()
        window_size = meta["window_size"]

        # ── 1. Pose extraction + feature engineering (Phases 3-4 replay) ──
        progress(5, "Extracting pose landmarks", stage="pose")

        def pose_cb(pct, msg):
            progress(pct, msg, stage="pose" if pct < 50 else "features")

        feature_matrix, fps, total_frames, all_landmarks = \
            extract_features_from_video(job.input_path, progress_callback=pose_cb)

        if total_frames == 0 or len(feature_matrix) == 0:
            raise ValueError("No frames decoded from video")

        # ── 2. Validation gates ──
        progress(50, "Validating input", stage="features")
        validate_input(feature_matrix, fps, meta)

        # Reject clips with too many missing keypoints overall (>20% of frames)
        frame_nan_fraction = np.isnan(feature_matrix).any(axis=1).mean()
        if frame_nan_fraction > 0.20:
            raise ValueError(
                f"{frame_nan_fraction*100:.1f}% of frames have missing keypoints "
                f"(threshold 20%). Re-record with the athlete fully in-frame."
            )

        # ── 3. Window inference ──
        progress(52, "Running BiLSTM inference", stage="inference")
        csv_path = out_dir / "per_window_predictions.csv"
        predictions = run_inference(
            feature_matrix=feature_matrix,
            model=model,
            meta=meta,
            output_csv_path=str(csv_path),
            progress_callback=lambda p, m: progress(p, m, stage="inference"),
        )

        # ── 4. Frame-level risk assignment + smoothing ──
        progress(86, "Assigning frame-level risk", stage="overlay")
        frame_labels, frame_phigh = assign_frame_labels(
            predictions, n_frames=total_frames, window_size=window_size,
        )
        frame_labels = rolling_mode_smooth(
            frame_labels, frame_phigh, window=3, high_override_threshold=0.85,
        )

        # ── 5. Identify contiguous High Risk events ──
        high_events = extract_high_events(frame_labels, min_duration=3)

        # ── 6. Annotations ──
        progress(88, "Generating biomechanical annotations", stage="annotations")
        ann_path = out_dir / "movement_annotations.txt"
        annotated_events = generate_annotations(
            high_events=high_events,
            feature_matrix=feature_matrix,
            output_txt_path=str(ann_path),
        )

        # ── 7. Skeleton overlay ──
        progress(89, "Rendering skeleton overlay video", stage="overlay")
        overlay_path = out_dir / "output_skeleton_overlay.mp4"
        render_overlay_video(
            input_video_path=job.input_path,
            output_video_path=str(overlay_path),
            frame_labels=frame_labels,
            frame_phigh=frame_phigh,
            all_landmarks=all_landmarks,
            event_annotations=annotated_events,
            progress_callback=lambda p, m: progress(p, m, stage="overlay"),
        )

        # ── 8. Risk timeline figure ──
        progress(96, "Generating risk timeline figure", stage="timeline")
        timeline_path = out_dir / "risk_timeline.png"
        generate_timeline(
            predictions=predictions,
            feature_matrix=feature_matrix,
            fps=fps,
            output_path=str(timeline_path),
            high_events=high_events,
        )

        # ── 9. Build risk summary ──
        risk_summary = build_risk_summary(predictions)

        # ── 10. Summary PDF report ──
        progress(98, "Building PDF summary report", stage="report")
        report_path = out_dir / "summary_report.pdf"
        clip_meta = {
            "filename":     job.filename,
            "duration_sec": (total_frames / fps) if fps > 0 else 0.0,
            "fps":          fps,
            "total_frames": total_frames,
            "camera_angle": job.camera_angle,
        }
        generate_report(
            output_pdf_path=str(report_path),
            timeline_img_path=str(timeline_path),
            clip_meta=clip_meta,
            risk_summary=risk_summary,
            annotated_events=annotated_events,
        )

        # ── 11. Build final result payload ──
        result = InferenceResult(
            job_id=job.job_id,
            filename=job.filename,
            duration_sec=clip_meta["duration_sec"],
            fps=fps,
            total_frames=total_frames,
            camera_angle=job.camera_angle,
            risk_summary=RiskSummary(
                total_windows=risk_summary["total_windows"],
                low_count=risk_summary["low_count"],
                medium_count=risk_summary["medium_count"],
                high_count=risk_summary["high_count"],
                low_pct=risk_summary["low_pct"],
                medium_pct=risk_summary["medium_pct"],
                high_pct=risk_summary["high_pct"],
                peak_high_window=(
                    WindowPrediction(**risk_summary["peak_high_window"])
                    if risk_summary["peak_high_window"] else None
                ),
            ),
            annotated_events=[
                AnnotatedEvent(
                    start_frame=ev["start_frame"],
                    end_frame=ev["end_frame"],
                    dominant_subscore=ev["dominant_subscore"],
                    annotation=ev["annotation"],
                    peak_frame=ev.get("peak_frame"),
                )
                for ev in annotated_events
            ],
            artifacts={
                "overlay_video":   f"/api/jobs/{job.job_id}/artifacts/output_skeleton_overlay.mp4",
                "timeline_image":  f"/api/jobs/{job.job_id}/artifacts/risk_timeline.png",
                "annotations_txt": f"/api/jobs/{job.job_id}/artifacts/movement_annotations.txt",
                "predictions_csv": f"/api/jobs/{job.job_id}/artifacts/per_window_predictions.csv",
                "summary_pdf":     f"/api/jobs/{job.job_id}/artifacts/summary_report.pdf",
            },
        )

        job.result = json.loads(result.json())
        job.update(stage="done", progress=100, message="Pipeline complete")

    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"[{job_id[:8]}] Pipeline failed: {e}\n{tb}")
        job.error = str(e)
        job.update(stage="error", progress=job.progress, message=f"Failed: {e}")
