import os
import csv
import sys

sys.path.insert(0, os.path.dirname(__file__))

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions
from feature_extraction import extract_all_features, summarise_features


DATA_ROOT = "data/raw/cmu_mocap"
OUTPUT_CSV = "data/processed/phase1_features.csv"


def run_subject(subject_dir):
    asf_path = os.path.join(subject_dir, "skeleton.asf")
    motions_dir = os.path.join(subject_dir, "motions")

    if not os.path.exists(asf_path) or not os.path.exists(motions_dir):
        return []

    joints = parse_asf(asf_path)
    rows = []

    for fname in sorted(os.listdir(motions_dir)):
        if not fname.endswith(".amc"):
            continue

        amc_path = os.path.join(motions_dir, fname)
        subject_id = os.path.basename(subject_dir)
        trial_id = os.path.splitext(fname)[0]

        try:
            motions = parse_amc(amc_path)
            positions = compute_joint_positions(motions, joints)
            features = extract_all_features(positions)
            summary = summarise_features(features)
        except Exception as e:
            print(f"  SKIP {subject_id}/{trial_id}: {e}")
            continue

        row = {"subject": subject_id, "trial": trial_id}
        row.update(summary)
        rows.append(row)
        print(f"  OK   {subject_id}/{trial_id}  ({len(summary)} features)")

    return rows


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_rows = []

    for subject in sorted(os.listdir(DATA_ROOT)):
        subject_dir = os.path.join(DATA_ROOT, subject)
        if not os.path.isdir(subject_dir):
            continue
        print(f"Subject: {subject}")
        all_rows.extend(run_subject(subject_dir))

    if not all_rows:
        print("No data found.")
        return

    fieldnames = ["subject", "trial"] + sorted(
        k for k in all_rows[0] if k not in ("subject", "trial")
    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} trials → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()