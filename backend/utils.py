"""
utils.py
Shared helpers for Phase 7:
  - assign_frame_labels: latest-window assignment policy (Step 3 of workflow)
  - rolling_mode_smooth: 3-window rolling mode filter to suppress flicker
  - extract_high_events:  contiguous (start, end) frame ranges of High Risk
  - build_risk_summary:   distribution + peak-window stats for the report
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ── Step 3: Frame-level risk assignment ───────────────────────────────────────

def assign_frame_labels(
    predictions: List[Optional[dict]],
    n_frames: int,
    window_size: int,
) -> Tuple[List[str], List[float]]:
    """
    Latest-window assignment: each frame f gets the label of the most recently
    completed window that contains f (window ending at or before f).

    Returns:
        frame_labels: List[str]   length n_frames  ("Low" | "Medium" | "High")
        frame_phigh:  List[float] length n_frames  (P_high of that window)

    Frames before the first complete window default to "Low" / 0.0.
    """
    frame_labels: List[str] = ["Low"] * n_frames
    frame_phigh: List[float] = [0.0] * n_frames

    # Map end_frame -> prediction (skip None / corrupted)
    end_to_pred: Dict[int, dict] = {}
    for p in predictions:
        if p is None:
            continue
        end_to_pred[p["end_frame"]] = p

    last_pred: Optional[dict] = None
    for f in range(n_frames):
        if f in end_to_pred:
            last_pred = end_to_pred[f]
        if last_pred is not None:
            frame_labels[f] = last_pred["label"]
            frame_phigh[f] = float(last_pred["P_high"])

    return frame_labels, frame_phigh


def rolling_mode_smooth(
    frame_labels: List[str],
    frame_phigh: List[float],
    window: int = 3,
    high_override_threshold: float = 0.85,
) -> List[str]:
    """
    Rolling-mode filter to suppress single-window High flickers.

    A 'High' frame surrounded by non-High frames is downgraded to 'Medium'
    UNLESS its P_high exceeds `high_override_threshold` (default 0.85).
    """
    n = len(frame_labels)
    if n == 0:
        return frame_labels

    half = window // 2
    smoothed = list(frame_labels)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = frame_labels[lo:hi]
        mode_label, _ = Counter(local).most_common(1)[0]

        if frame_labels[i] == "High" and mode_label != "High":
            if frame_phigh[i] < high_override_threshold:
                smoothed[i] = "Medium"
        elif frame_labels[i] != mode_label and mode_label != "High":
            # Light smoothing for Low/Medium flicker only — never invent High.
            smoothed[i] = mode_label

    return smoothed


# ── High Risk event extraction ────────────────────────────────────────────────

def extract_high_events(
    frame_labels: List[str],
    min_duration: int = 3,
) -> List[Tuple[int, int]]:
    """
    Return list of (start_frame, end_frame) for contiguous runs of 'High'.
    Runs shorter than `min_duration` frames are dropped.
    """
    events: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    for i, lbl in enumerate(frame_labels):
        if lbl == "High" and not in_event:
            in_event = True
            start = i
        elif lbl != "High" and in_event:
            in_event = False
            end = i - 1
            if end - start + 1 >= min_duration:
                events.append((start, end))
    if in_event:
        end = len(frame_labels) - 1
        if end - start + 1 >= min_duration:
            events.append((start, end))
    return events


# ── Risk distribution summary ─────────────────────────────────────────────────

def build_risk_summary(predictions: List[Optional[dict]]) -> Dict[str, Any]:
    """Build the risk_summary dict consumed by report.py."""
    valid = [p for p in predictions if p is not None]
    total = len(valid)
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for p in valid:
        counts[p["label"]] = counts.get(p["label"], 0) + 1

    def pct(n: int) -> float:
        return (n / total * 100.0) if total > 0 else 0.0

    high_preds = [p for p in valid if p["label"] == "High"]
    peak: Optional[dict] = None
    if high_preds:
        peak = max(high_preds, key=lambda p: p["P_high"])
    elif valid:
        # fall back to overall highest P_high window
        peak = max(valid, key=lambda p: p["P_high"])

    return {
        "total_windows": total,
        "low_count":     counts["Low"],
        "medium_count":  counts["Medium"],
        "high_count":    counts["High"],
        "low_pct":       pct(counts["Low"]),
        "medium_pct":    pct(counts["Medium"]),
        "high_pct":      pct(counts["High"]),
        "peak_high_window": peak,
    }
