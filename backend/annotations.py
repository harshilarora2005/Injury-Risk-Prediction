"""
annotations.py
Re-runs Phase 5 sub-score logic on flagged High Risk windows to identify
the dominant biomechanical signal and produce human-readable annotation strings.
"""

import numpy as np
from typing import List, Tuple, Dict

# ── Phase 5 sub-score thresholds (directional references, not clinical cutoffs) ─
KNEE_FLEXION_LOW_THRESHOLD = 30.0    # degrees — reduced flexion at landing
ASYM_THRESHOLD             = 15.0    # degrees — L-R difference
TRUNK_LEAN_THRESHOLD       = 20.0    # degrees from vertical
VELOCITY_THRESHOLD         = 8.0     # degrees/frame — high angular velocity


def compute_knee_subscore(window: np.ndarray) -> float:
    """High score = reduced knee flexion (straighter leg at landing)."""
    kf = np.nanmean(window[:, 0:2])   # mean of L+R knee flexion
    if np.isnan(kf):
        return 0.0
    # Lower knee flexion → higher risk score (inverted)
    return max(0.0, (KNEE_FLEXION_LOW_THRESHOLD - kf) / KNEE_FLEXION_LOW_THRESHOLD)


def compute_asym_subscore(window: np.ndarray) -> float:
    """High score = large L-R knee asymmetry."""
    asym = np.nanmean(window[:, 4])   # column 4: lr_knee_asymmetry
    if np.isnan(asym):
        return 0.0
    return min(1.0, asym / ASYM_THRESHOLD)


def compute_trunk_subscore(window: np.ndarray) -> float:
    """High score = excessive trunk lean."""
    trunk = np.nanmean(window[:, 5])  # column 5: trunk_lean
    if np.isnan(trunk):
        return 0.0
    return min(1.0, trunk / TRUNK_LEAN_THRESHOLD)


def compute_velocity_subscore(window: np.ndarray) -> float:
    """High score = high angular velocity of knee."""
    vel = np.nanmean(np.abs(window[:, 6:8]))   # cols 6,7: angular velocity L+R
    if np.isnan(vel):
        return 0.0
    return min(1.0, vel / VELOCITY_THRESHOLD)


ANNOTATION_MAP = {
    "asym":  "Asymmetric loading detected — frames {start}–{end}",
    "knee":  "Reduced knee flexion at landing — frame {peak_frame}",
    "trunk": "Excessive trunk lean detected — frames {start}–{end}",
    "vel":   "High angular velocity at ground contact — frame {peak_frame}",
}


def annotate_event(
    window_features: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> dict:
    """
    Identify dominant sub-score for a High Risk window and return annotation dict.
    NOTE: This gives the highest single-signal contributor, which is an approximation —
    the BiLSTM may have flagged a temporal co-occurrence not captured by per-frame scores.
    """
    scores = {
        "knee":  compute_knee_subscore(window_features),
        "asym":  compute_asym_subscore(window_features),
        "trunk": compute_trunk_subscore(window_features),
        "vel":   compute_velocity_subscore(window_features),
    }

    dominant = max(scores, key=scores.get)

    # Peak frame: frame within window with highest absolute velocity (for knee/vel)
    # or highest asymmetry (for asym/trunk)
    peak_col = {
        "knee":  0,   # left knee flexion (look for minimum)
        "asym":  4,
        "trunk": 5,
        "vel":   6,
    }[dominant]

    col_vals = window_features[:, peak_col]
    valid = ~np.isnan(col_vals)
    if valid.any():
        if dominant == "knee":
            rel_idx = int(np.nanargmin(col_vals))  # most straight = most risk
        else:
            rel_idx = int(np.nanargmax(np.abs(col_vals)))
        peak_frame = start_frame + rel_idx
    else:
        peak_frame = (start_frame + end_frame) // 2

    template = ANNOTATION_MAP[dominant]
    annotation = template.format(
        start=start_frame,
        end=end_frame,
        peak_frame=peak_frame,
    )

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "dominant_subscore": dominant,
        "annotation": annotation,
        "peak_frame": peak_frame,
        "scores": scores,
    }


def generate_annotations(
    high_events: List[Tuple[int, int]],
    feature_matrix: np.ndarray,
    output_txt_path: str,
) -> List[dict]:
    """Annotate all High Risk events and write movement_annotations.txt."""
    results = []
    for start, end in high_events:
        window = feature_matrix[start: end + 1]
        ev = annotate_event(window, start, end)
        results.append(ev)

    # Write human-readable text file
    with open(output_txt_path, "w") as f:
        f.write("ACL Risk Movement Annotations\n")
        f.write("=" * 50 + "\n")
        f.write("SCREENING ONLY — NOT A CLINICAL DIAGNOSIS\n")
        f.write("=" * 50 + "\n\n")
        if not results:
            f.write("No High Risk events detected in this clip.\n")
        for i, ev in enumerate(results, 1):
            f.write(f"Event {i}: {ev['annotation']}\n")
            f.write(f"  Dominant signal : {ev['dominant_subscore'].upper()}\n")
            scores = ev.get("scores", {})
            for k, v in scores.items():
                f.write(f"  {k:8s} sub-score: {v:.3f}\n")
            f.write("\n")

    return results