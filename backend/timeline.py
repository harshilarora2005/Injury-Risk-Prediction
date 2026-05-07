import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_local_minima(arr, min_distance=10):
    minima = []
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if not minima or i - minima[-1] >= min_distance:
                minima.append(i)
    return minima


def generate_timeline(predictions, feature_matrix, fps, output_path, high_events):

    valid_preds = [p for p in predictions if p is not None]
    if not valid_preds:
        return

    x_frames = np.array([p["end_frame"] for p in valid_preds])
    p_high_series = [p["P_high"] for p in valid_preds]

    n_frames = len(feature_matrix)
    frame_axis = np.arange(n_frames)

    kf_l = feature_matrix[:, 0]
    kf_r = feature_matrix[:, 1]
    trunk = feature_matrix[:, 5]

    combined_kf = np.nanmean(np.stack([kf_l, kf_r], axis=1), axis=1)
    ground_contacts = _find_local_minima(combined_kf)

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1, 1]}
    )

    fig.patch.set_facecolor("#0f1117")

    for ax in (ax_top, ax_mid, ax_bot):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        ax.spines["bottom"].set_color("#333344")
        ax.spines["left"].set_color("#333344")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ── TOP: RISK ─────────────────────────────
    ax_top.axhspan(0.00, 0.33, color="#00c853", alpha=0.10, label="Low")
    ax_top.axhspan(0.33, 0.66, color="#ffd600", alpha=0.10, label="Medium")
    ax_top.axhspan(0.66, 1.00, color="#ff1744", alpha=0.10, label="High")

    ax_top.axhline(0.33, color="#ffd600", lw=0.8, ls="--", alpha=0.7)
    ax_top.axhline(0.66, color="#ff5252", lw=0.8, ls="--", alpha=0.7)

    ax_top.plot(x_frames, p_high_series, color="#ef5350", lw=1.8, label="P(High)")
    ax_top.fill_between(x_frames, 0, p_high_series, color="#ef5350", alpha=0.18)

    for start, end in high_events:
        ax_top.axvspan(start, end, color="#ff1744", alpha=0.18)

    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("Risk", color="#cccccc")
    ax_top.legend(loc="upper right", fontsize=8, framealpha=0.3)

    # ── MID: KNEES ────────────────────────────
    ax_mid.plot(frame_axis, kf_l, color="#42a5f5", lw=1.5, label="Left Knee")
    ax_mid.plot(frame_axis, kf_r, color="#ff7043", lw=1.5, label="Right Knee")

    ax_mid.fill_between(
        frame_axis,
        np.nanmin(np.stack([kf_l, kf_r], axis=1), axis=1),
        np.nanmax(np.stack([kf_l, kf_r], axis=1), axis=1),
        color="#ab47bc", alpha=0.20, label="Asymmetry"
    )

    for gc in ground_contacts:
        ax_mid.axvline(gc, color="#80cbc4", lw=0.6, ls=":", alpha=0.7)

    for start, end in high_events:
        ax_mid.axvspan(start, end, color="#ff1744", alpha=0.12)

    ax_mid.set_ylabel("Knee Flexion (°)", color="#cccccc")
    ax_mid.legend(loc="upper right", fontsize=8, framealpha=0.3)

    # ── BOT: TRUNK ────────────────────────────
    ax_bot.plot(frame_axis, trunk, color="#66bb6a", lw=1.8, label="Trunk Lean")

    for start, end in high_events:
        ax_bot.axvspan(start, end, color="#ff1744", alpha=0.10)

    ax_bot.set_ylabel("Trunk Lean (°)", color="#cccccc")
    ax_bot.set_xlabel("Frame", color="#cccccc")
    ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.3)

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)