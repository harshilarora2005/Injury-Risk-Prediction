import numpy as np
import matplotlib.pyplot as plt

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions
from feature_extraction import extract_all_features, summarise_features
from motion_segmentation import get_category, iter_motions, RUNNING, JUMPING, CUTTING


# ── Paths ─────────────────────────────────────────────────────────────────────
ASF_PATH  = 'data/raw/cmu_mocap/subject_01/skeleton.asf'
AMC_PATH  = 'data/raw/cmu_mocap/subject_01/motions/01_01.amc'
DATA_ROOT = 'data/raw/cmu_mocap'


# ── Load one trial ────────────────────────────────────────────────────────────
def load_single_trial():
    joints    = parse_asf(ASF_PATH)
    motions   = parse_amc(AMC_PATH)
    positions = compute_joint_positions(motions, joints)
    print(f'Loaded {len(positions)} frames.\n')
    return motions, positions


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_knee_flexion(features):
    plt.figure(figsize=(12, 4))
    plt.plot(features['left_knee_flexion_deg'],  label='Left')
    plt.plot(features['right_knee_flexion_deg'], label='Right', linestyle='--')
    plt.axhline(30, color='red', linewidth=0.8, linestyle=':', label='30° risk threshold')
    plt.ylabel('Degrees'); plt.xlabel('Frame')
    plt.title('Knee Flexion Angle'); plt.legend(); plt.tight_layout(); plt.show()


def plot_knee_valgus(features):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, side in zip(axes, ['left', 'right']):
        ax.plot(features[f'{side}_knee_valgus_cm'])
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel('cm'); ax.set_title(f'{side.title()} knee valgus  (+ = valgus)')
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Knee Valgus Proxy'); plt.tight_layout(); plt.show()


def plot_hip_knee_ratio(features):
    plt.figure(figsize=(12, 4))
    plt.plot(features['left_hip_knee_ratio'],  label='Left')
    plt.plot(features['right_hip_knee_ratio'], label='Right', linestyle='--')
    plt.axhline(0.5, color='red', linewidth=0.8, linestyle=':', label='0.5 risk threshold')
    plt.ylabel('Ratio'); plt.xlabel('Frame')
    plt.title('Hip / Knee Flexion Ratio'); plt.legend(); plt.tight_layout(); plt.show()


def plot_hip(features):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, key, title in zip(
        axes,
        ['left_hip_flexion_deg', 'left_hip_adduction_deg'],
        ['Hip Flexion (Left)', 'Hip Adduction (Left)  (+ = adduction)'],
    ):
        ax.plot(features[key])
        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Degrees'); ax.set_title(title)
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Hip Kinematics'); plt.tight_layout(); plt.show()


def plot_asymmetry(features):
    keys = ['knee_flexion_asymmetry_pct', 'knee_valgus_asymmetry_pct', 'hip_flexion_asymmetry_pct']
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, key in zip(axes, keys):
        ax.plot(features[key])
        ax.axhline(0,   color='black', linewidth=0.5, linestyle=':')
        ax.axhline( 15, color='red',   linewidth=0.8, linestyle='--', label='+15%')
        ax.axhline(-15, color='red',   linewidth=0.8, linestyle='--', label='-15%')
        ax.set_ylabel('AI (%)'); ax.set_title(key.replace('_', ' '))
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Frame')
    plt.suptitle('Left-Right Asymmetry Index'); plt.tight_layout(); plt.show()


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(summary):
    print('── Feature Summary ───────────────────────────────────────────────')
    for k, v in sorted(summary.items()):
        print(f'  {k:<50} {v:>10.3f}')
    print()


# ── Catalog test ──────────────────────────────────────────────────────────────
def test_catalog():
    print('── Catalog lookup ────────────────────────────────────────────────')
    cases = [
        (1,  1, JUMPING),
        (9,  1, RUNNING),
        (15, 1, CUTTING),
        (99, 1, 'other'),
    ]
    for subj, mot, expected in cases:
        result = get_category(subj, mot)
        mark   = '✓' if result == expected else '✗'
        print(f'  {mark}  subject={subj:03d}  motion={mot:02d}  expected={expected:<8}  got={result}')
    print()


# ── Multi-trial table ─────────────────────────────────────────────────────────
def test_multi_trial():
    import os
    if not os.path.isdir(DATA_ROOT):
        print(f'── multi-trial skipped (DATA_ROOT not found)\n')
        return

    print('── Multi-trial feature summary ───────────────────────────────────')
    rows = []
    for trial in iter_motions(DATA_ROOT, categories=[RUNNING, JUMPING, CUTTING], verbose=False):
        s = summarise_features(extract_all_features(trial.positions, trial.motions))
        s['subject']  = float(trial.subject)
        s['category'] = trial.category
        rows.append(s)
        if len(rows) >= 6:
            break

    if not rows:
        print('  No labeled trials found.\n')
        return

    cols   = ['left_knee_flexion_deg_mean', 'left_knee_valgus_cm_mean',
              'left_hip_knee_ratio_mean',   'knee_flexion_asymmetry_pct_mean']
    labels = ['L.knee_flex', 'L.valgus', 'hip/knee', 'asym_pct']

    header = f'  {"subj":>5}  {"category":<10}' + ''.join(f'  {l:>12}' for l in labels)
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for row in rows:
        line = f'  {int(row["subject"]):>5}  {row["category"]:<10}'
        for c in cols:
            line += f'  {row.get(c, float("nan")):>12.2f}'
        print(line)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    motions, positions = load_single_trial()

    features = extract_all_features(positions, motions)
    print_summary(summarise_features(features))

    plot_knee_flexion(features)
    plot_knee_valgus(features)
    plot_hip_knee_ratio(features)
    plot_hip(features)
    plot_asymmetry(features)

    test_catalog()
    test_multi_trial()


if __name__ == '__main__':
    main()