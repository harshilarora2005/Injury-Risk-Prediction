"""
overlay.py
Draws color-coded MediaPipe skeleton + HUD on each frame of the input video.
Outputs output_skeleton_overlay.mp4.
"""

import cv2
import numpy as np
import subprocess
import shutil
import os
import tempfile
from pathlib import Path
from typing import List, Optional

# Risk → colors (BGR for OpenCV)
COLORS = {
    "Low":    {"joint": (0, 200, 0),    "bone": (144, 238, 144)},
    "Medium": {"joint": (0, 200, 255),  "bone": (0, 215, 255)},   # Yellow in BGR
    "High":   {"joint": (20, 20, 220),  "bone": (0, 69, 255)},    # Red in BGR
}

# MediaPipe Pose connections (subset for clarity)
POSE_CONNECTIONS = [
    (11, 12),  # shoulders
    (11, 23),  # L shoulder→hip
    (12, 24),  # R shoulder→hip
    (23, 24),  # hips
    (23, 25),  # L hip→knee
    (24, 26),  # R hip→knee
    (25, 27),  # L knee→ankle
    (26, 28),  # R knee→ankle
    (11, 13),  # L shoulder→elbow
    (12, 14),  # R shoulder→elbow
    (13, 15),  # L elbow→wrist
    (14, 16),  # R elbow→wrist
]


def _draw_skeleton(frame, landmarks, color_joint, color_bone, frame_w, frame_h):
    """Draw joints and bones on `frame` in-place."""
    if landmarks is None:
        return

    # Collect pixel coords
    pts = {}
    for idx, lm in enumerate(landmarks):
        if lm.visibility >= 0.5:
            pts[idx] = (int(lm.x * frame_w), int(lm.y * frame_h))

    # Draw bones
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color_bone, 2, cv2.LINE_AA)

    # Draw joints
    for idx, pt in pts.items():
        cv2.circle(frame, pt, 5, color_joint, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)  # white ring


def _draw_hud(frame, label: str, p_high: float, frame_idx: int, total_frames: int):
    """Draw top-left HUD overlay."""
    color_map = {
        "Low": (0, 200, 0),
        "Medium": (0, 200, 255),
        "High": (20, 20, 220),
    }
    label_color = color_map.get(label, (200, 200, 200))

    # Semi-transparent black background for HUD
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (420, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Risk Level: {label}", (14, 32),
                font, 0.75, label_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx} / {total_frames}", (14, 82),
                font, 0.60, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_event_annotation(frame, annotation: str, frame_w: int, frame_h: int):
    """Overlay annotation text at bottom of frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = frame.copy()
    text_y = frame_h - 20
    cv2.rectangle(overlay, (0, frame_h - 40), (frame_w, frame_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, annotation, (10, text_y),
                font, 0.55, (255, 255, 100), 1, cv2.LINE_AA)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _reencode_h264(raw_path: str, final_path: str) -> None:
    """
    Re-encode an OpenCV-written mp4v file to browser-compatible H.264/AAC.
    Falls back silently if ffmpeg is unavailable (video will still exist,
    just may not play in all browsers).
    """
    if not _has_ffmpeg():
        shutil.move(raw_path, final_path)
        return
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",           
        "-pix_fmt", "yuv420p", 
        "-movflags", "+faststart",  
        "-an",                 
        final_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(raw_path)
    except subprocess.CalledProcessError as e:
        # ffmpeg failed — fall back to the raw file
        shutil.move(raw_path, final_path)


def render_overlay_video(
    input_video_path: str,
    output_video_path: str,
    frame_labels: List[str],
    frame_phigh: List[float],
    all_landmarks: list,
    event_annotations: List[dict],  
    progress_callback=None,
) -> None:
    """
    Read input video frame-by-frame, draw skeleton + HUD, write to output.
    OpenCV writes a temp file with mp4v; FFmpeg re-encodes to H.264 so every
    browser can play it without a plugin.
    """
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp_fd, tmp_path = tempfile.mkstemp(suffix="_raw.mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (frame_w, frame_h))

    # Build active annotations per frame
    frame_annotation = {}
    for ev in event_annotations:
        for f in range(ev["start_frame"], ev["end_frame"] + 1):
            frame_annotation[f] = ev["annotation"]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = frame_labels[frame_idx] if frame_idx < len(frame_labels) else "Low"
        p_high = frame_phigh[frame_idx] if frame_idx < len(frame_phigh) else 0.0
        lms = all_landmarks[frame_idx] if frame_idx < len(all_landmarks) else None

        colors = COLORS.get(label, COLORS["Low"])
        _draw_skeleton(frame, lms, colors["joint"], colors["bone"], frame_w, frame_h)
        _draw_hud(frame, label, p_high, frame_idx, total_frames)

        if frame_idx in frame_annotation:
            _draw_event_annotation(frame, frame_annotation[frame_idx], frame_w, frame_h)

        out.write(frame)

        if progress_callback and frame_idx % 30 == 0:
            pct = 85 + int((frame_idx / max(total_frames, 1)) * 10)
            progress_callback(pct, f"Rendering overlay — frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out.release()

    if progress_callback:
        progress_callback(95, "Re-encoding to H.264 for browser playback")
    _reencode_h264(tmp_path, output_video_path)