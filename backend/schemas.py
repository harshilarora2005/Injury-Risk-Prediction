from pydantic import BaseModel
from typing import Optional, List


class WindowPrediction(BaseModel):
    start_frame: int
    end_frame: int
    P_low: float
    P_medium: float
    P_high: float
    label: str  # "Low" | "Medium" | "High"


class AnnotatedEvent(BaseModel):
    start_frame: int
    end_frame: int
    dominant_subscore: str
    annotation: str
    peak_frame: Optional[int] = None


class RiskSummary(BaseModel):
    total_windows: int
    low_count: int
    medium_count: int
    high_count: int
    low_pct: float
    medium_pct: float
    high_pct: float
    peak_high_window: Optional[WindowPrediction]


class InferenceResult(BaseModel):
    job_id: str
    filename: str
    duration_sec: float
    fps: float
    total_frames: int
    camera_angle: str
    risk_summary: RiskSummary
    annotated_events: List[AnnotatedEvent]
    artifacts: dict  # relative URLs to each output file


class JobStatus(BaseModel):
    job_id: str
    stage: str          # "uploading" | "pose" | "features" | "inference" | "overlay" | "timeline" | "report" | "done" | "error"
    progress: int       # 0–100
    message: str
    result: Optional[InferenceResult] = None
    error: Optional[str] = None