"""
timeline.py
Generates risk_timeline.png — dual-panel figure:
  Top:    P_high per window + threshold bands + High Risk shaded regions
  Bottom: L/R smoothed knee flexion + asymmetry shaded area + ground contact markers
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from typing import List, Optional


def _find_local_minima(arr: np.ndarray, min_distance: int = 10) -> List[int]:
    """Simple local minima detection for ground contact events."""
    minima = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if not minima or i - minima[-1] >= min_distance:
                minima.append(i)
    return minima


def generate_timeline(
    predictions: List[Optional[dict]],
    feature_matrix: np.ndarray,
    fps: float,
    output_path: str,
    high_events: List[tuple],   # list of (start_frame, end_frame)
) -> None:
    """
    Generate and save the dual-panel risk timeline figure.

    feature_matrix columns:
      0: knee_flexion_left
      1: knee_flexion_right
      4: lr_knee_asymmetry
    """
    # ── Build window-level P_high series ──────────────────────────────────────
    valid_preds = [p for p in predictions if p is not None]
    if not valid_preds:
        return

    window_end_frames = [p["end_frame"] for p in valid_preds]
    p_high_series = [p["P_high"] for p in valid_preds]
    x_frames = np.array(window_end_frames)

    # ── Frame-level knee angles ────────────────────────────────────────────────
    n_frames = len(feature_matrix)
    frame_axis = np.arange(n_frames)
    kf_l = feature_matrix[:, 0]
    kf_r = feature_matrix[:, 1]
    asym = np.abs(kf_l - kf_r)

    # Ground contact = local minima in knee flexion (combined)
    combined_kf = np.nanmean(np.stack([kf_l, kf_r], axis=1), axis=1)
    ground_contacts = _find_local_minima(combined_kf)

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1]},
    )
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax_top, ax_bot):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color("#333344")
        ax.spines["left"].set_color("#333344")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── Top panel: P_high ──────────────────────────────────────────────────────
    # Background color bands
    ax_top.axhspan(0.00, 0.33, color="#00c853", alpha=0.10)
    ax_top.axhspan(0.33, 0.66, color="#ffd600", alpha=0.10)
    ax_top.axhspan(0.66, 1.00, color="#ff1744", alpha=0.10)

    # Threshold dashed lines
    ax_top.axhline(0.33, color="#ffd600", lw=0.8, ls="--", alpha=0.7, label="Low→Medium (0.33)")
    ax_top.axhline(0.66, color="#ff5252", lw=0.8, ls="--", alpha=0.7, label="Medium→High (0.66)")

    # P_high line
    ax_top.plot(x_frames, p_high_series, color="#ef5350", lw=1.8, label="P(High)", zorder=3)
    ax_top.fill_between(x_frames, 0, p_high_series, color="#ef5350", alpha=0.18)

    # Vertical shading for High Risk events
    for start, end in high_events:
        ax_top.axvspan(start, end, color="#ff1744", alpha=0.18, zorder=2)

    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("P(High Risk)", color="#cccccc", fontsize=10)
    ax_top.set_title("ACL Risk Score Timeline",
                     color="#dddddd", fontsize=11, pad=8)
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.3,
                  labelcolor="#cccccc", facecolor="#1a1d27")

    # ── Bottom panel: Knee flexion ─────────────────────────────────────────────
    ax_bot.plot(frame_axis, kf_l, color="#42a5f5", lw=1.5, label="Left knee flexion")
    ax_bot.plot(frame_axis, kf_r, color="#ff7043", lw=1.5, label="Right knee flexion")

    # Asymmetry shaded area
    ax_bot.fill_between(
        frame_axis,
        np.nanmin(np.stack([kf_l, kf_r], axis=1), axis=1),
        np.nanmax(np.stack([kf_l, kf_r], axis=1), axis=1),
        color="#ab47bc", alpha=0.20, label="L-R asymmetry",
    )

    # Ground contact tick marks
    for gc in ground_contacts:
        ax_bot.axvline(gc, color="#80cbc4", lw=0.6, ls=":", alpha=0.7)

    # High event vertical bands
    for start, end in high_events:
        ax_bot.axvspan(start, end, color="#ff1744", alpha=0.12, zorder=2)

    ax_bot.set_ylabel("Knee Flexion (°)", color="#cccccc", fontsize=10)
    ax_bot.set_xlabel("Frame number", color="#cccccc", fontsize=10)
    ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.3,
                  labelcolor="#cccccc", facecolor="#1a1d27")

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)