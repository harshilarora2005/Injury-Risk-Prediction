import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions
from feature_extraction import extract_all_features, summarise_features


# ── Paths ─────────────────────────────────────────────────────────────────────
ASF_PATH = 'data/raw/cmu_mocap/subject_01/skeleton.asf'
AMC_PATH = 'data/raw/cmu_mocap/subject_01/motions/01_01.amc'


# ── Load ──────────────────────────────────────────────────────────────────────
def load_motion_data():
    print('Parsing ASF ...')
    joints = parse_asf(ASF_PATH)

    print('Parsing AMC ...')
    motions = parse_amc(AMC_PATH)
    print(f'  {len(motions)} frames loaded')

    print('Computing joint positions ...')
    positions = compute_joint_positions(motions, joints)
    print('  Done.\n')

    return motions, positions


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_flexion_curves(features):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    pairs = [
        ('left_knee_flexion_deg',       'right_knee_flexion_deg',       'Knee Flexion'),
        ('left_hip_flexion_deg',        'right_hip_flexion_deg',        'Hip Flexion'),
        ('left_ankle_dorsiflexion_deg', 'right_ankle_dorsiflexion_deg', 'Ankle Dorsiflexion'),
    ]
    for ax, (l_key, r_key, title) in zip(axes, pairs):
        ax.plot(features[l_key],  label='Left')
        ax.plot(features[r_key],  label='Right', linestyle='--')
        ax.set_ylabel('Degrees')
        ax.set_title(title)
        ax.legend()
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Joint Flexion Curves')
    plt.tight_layout()
    plt.show()


def plot_velocity_profiles(features):
    joints = ['root', 'lfoot', 'rfoot', 'lfemur', 'rfemur']
    fig, axes = plt.subplots(len(joints), 1, figsize=(12, 10), sharex=True)
    for ax, j in zip(axes, joints):
        ax.plot(features[f'{j}_speed'])
        ax.set_ylabel('cm/s')
        ax.set_title(f'{j} speed')
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Joint Velocity Profiles')
    plt.tight_layout()
    plt.show()


def plot_deceleration(features):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, joint in zip(axes, ['root', 'lfoot', 'rfoot']):
        accel = features[f'{joint}_acceleration']
        ax.plot(accel, label='acceleration')
        events = features[f'{joint}_decel_event_frames']
        ax.scatter(events, accel[events], color='red', zorder=5, label='decel event')
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel('cm/s²')
        ax.set_title(f'{joint} deceleration')
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Deceleration Patterns')
    plt.tight_layout()
    plt.show()


def plot_asymmetry(features):
    pairs = [
        'knee_flexion_asymmetry_index',
        'hip_flexion_asymmetry_index',
        'ankle_dorsiflexion_asymmetry_index',
    ]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes, pairs):
        if key in features:
            ax.plot(features[key])
            ax.axhline(0,   color='black', linewidth=0.5, linestyle=':')
            ax.axhline(15,  color='red',   linewidth=0.8, linestyle='--', label='+15% threshold')
            ax.axhline(-15, color='red',   linewidth=0.8, linestyle='--', label='-15% threshold')
            ax.set_ylabel('AI (%)')
            ax.set_title(key.replace('_', ' ').title())
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Left-Right Asymmetry Index')
    plt.tight_layout()
    plt.show()


def plot_knee_valgus(features):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, side in zip(axes, ['left', 'right']):
        key = f'{side}_knee_valgus_proxy_cm'
        ax.plot(features[key])
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel('cm')
        ax.set_title(f'{side.title()} knee valgus proxy (+ = valgus)')
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Knee Valgus Proxy')
    plt.tight_layout()
    plt.show()


def print_summary(summary):
    print('── Feature Summary (per-trial scalars) ──────────────────────────')
    for k, v in sorted(summary.items()):
        print(f'  {k:<55} {v:>10.3f}')
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    motions, positions = load_motion_data()

    print('Extracting features ...')
    features = extract_all_features(positions, motions, fps=120)
    print(f'  {len(features)} feature arrays extracted.\n')

    summary = summarise_features(features)
    print_summary(summary)

    plot_flexion_curves(features)
    plot_velocity_profiles(features)
    plot_deceleration(features)
    plot_asymmetry(features)
    plot_knee_valgus(features)


if __name__ == '__main__':
    main()