import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions
from feature_extraction import extract_all_features

ASF_PATH = "data/raw/cmu_mocap/subject_01/skeleton.asf"
AMC_PATH = "data/raw/cmu_mocap/subject_01/motions/01_01.amc"


def load():
    joints    = parse_asf(ASF_PATH)
    motions   = parse_amc(AMC_PATH)
    positions = compute_joint_positions(motions, joints)
    return extract_all_features(positions)


def plot_flexion_curves(f):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Joint Flexion Curves", fontweight="bold")

    pairs = [
        ("left_knee_flexion_deg",       "right_knee_flexion_deg",       "Knee Flexion (°)"),
        ("left_hip_flexion_deg",        "right_hip_flexion_deg",        "Hip Flexion (°)"),
        ("left_ankle_dorsiflexion_deg", "right_ankle_dorsiflexion_deg", "Ankle Dorsiflexion (°)"),
    ]

    for ax, (l, r, ylabel) in zip(axes, pairs):
        ax.plot(f[l], color="#2563EB", lw=1.6, label="Left")
        ax.plot(f[r], color="#DC2626", lw=1.6, linestyle="--", label="Right")
        ax.axhline(0, color="#9CA3AF", lw=0.8, linestyle=":")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()


def plot_asymmetry(f):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Left-Right Asymmetry Index (%)", fontweight="bold")

    pairs = [
        ("knee_flexion_asymmetry_index",      "Knee Flexion AI (%)"),
        ("hip_flexion_asymmetry_index",        "Hip Flexion AI (%)"),
        ("ankle_dorsiflexion_asymmetry_index", "Ankle Dorsiflexion AI (%)"),
    ]

    for ax, (key, ylabel) in zip(axes, pairs):
        if key not in f:
            ax.set_title(f"{ylabel} — not available")
            continue
        ax.plot(f[key], color="#7C3AED", lw=1.4)
        ax.axhline(0,   color="#9CA3AF", lw=0.8, linestyle=":")
        ax.axhline( 15, color="#F59E0B", lw=1.0, linestyle="--", label="+15% threshold")
        ax.axhline(-15, color="#F59E0B", lw=1.0, linestyle="--", label="-15% threshold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()


def plot_knee_valgus(f):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Knee Valgus Proxy (cm)", fontweight="bold")

    for ax, side, color in zip(axes, ["left", "right"], ["#2563EB", "#DC2626"]):
        ax.plot(f[f"{side}_knee_valgus_proxy_cm"], color=color, lw=1.6)
        ax.axhline(0, color="#9CA3AF", lw=0.8, linestyle=":")
        ax.fill_between(
            range(len(f[f"{side}_knee_valgus_proxy_cm"])),
            f[f"{side}_knee_valgus_proxy_cm"], 0,
            where=[v > 0 for v in f[f"{side}_knee_valgus_proxy_cm"]],
            alpha=0.15, color="#DC2626", label="Valgus"
        )
        ax.set_ylabel(f"{side.title()} (cm)")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()


def plot_hip_knee_ratio(f):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Hip-Knee Flexion Ratio", fontweight="bold")

    for ax, side, color in zip(axes, ["left", "right"], ["#2563EB", "#DC2626"]):
        ratio = f[f"{side}_hip_knee_ratio"]
        ax.plot(ratio, color=color, lw=1.6)
        ax.axhline(0.5, color="#F59E0B", lw=1.0, linestyle="--", label="Lower bound (0.5)")
        ax.axhline(0.8, color="#10B981", lw=1.0, linestyle="--", label="Upper bound (0.8)")
        ax.set_ylabel(f"{side.title()} ratio")
        ax.set_ylim(-1, 5)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()


def main():
    print("Loading data...")
    features = load()
    print("Feature keys:", sorted(features.keys()))
    print("Plotting...")

    plot_flexion_curves(features)
    plot_asymmetry(features)
    plot_knee_valgus(features)
    plot_hip_knee_ratio(features)

    plt.show()


if __name__ == "__main__":
    main()