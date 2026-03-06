import os
import shutil

ARCHIVE_ROOT = "archive"
DEST_ROOT = "data/raw/cmu_mocap"

C3D_DATASETS = [
    "allc3d_0",
    "allc3d_1a",
    "allc3d_1b",
    "allc3d_234",
]

os.makedirs(DEST_ROOT, exist_ok=True)

def ensure_c3d_folder(subject_id):
    subject_dir = os.path.join(DEST_ROOT, f"subject_{subject_id}")
    c3d_dir = os.path.join(subject_dir, "c3d")
    os.makedirs(c3d_dir, exist_ok=True)
    return c3d_dir

for dataset in C3D_DATASETS:
    subjects_root = os.path.join(ARCHIVE_ROOT, dataset, "subjects")
    if not os.path.exists(subjects_root):
        continue

    for subject in os.listdir(subjects_root):
        subject_path = os.path.join(subjects_root, subject)
        if not os.path.isdir(subject_path):
            continue

        c3d_dest = ensure_c3d_folder(subject)

        for file in os.listdir(subject_path):
            if file.lower().endswith(".c3d"):
                src = os.path.join(subject_path, file)
                dst = os.path.join(c3d_dest, file)

                if not os.path.exists(dst):
                    shutil.move(src, dst)

print("C3D files moved successfully.")