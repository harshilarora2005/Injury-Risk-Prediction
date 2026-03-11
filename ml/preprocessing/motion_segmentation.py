"""
motion_segmentation.py
──────────────────────
Labels CMU MoCap files by activity type using a hardcoded catalog and
provides a multi-file loader that yields only the requested categories.

No kinematic auto-detection — labels come from catalog only.
Unlabeled files are skipped unless OTHER is explicitly requested.

Usage
-----
    from motion_segmentation import iter_motions, get_category

    for trial in iter_motions('data/raw/cmu_mocap', categories=['running', 'cutting']):
        features = extract_all_features(trial.positions, trial.motions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

from parse_asf import parse_asf
from parse_amc import parse_amc
from compute_joint_positions import compute_joint_positions


# ─────────────────────────────────────────────────────────────────────────────
# Categories
# ─────────────────────────────────────────────────────────────────────────────

RUNNING = 'running'
JUMPING = 'jumping'
CUTTING = 'cutting'
OTHER   = 'other'

CATEGORIES = {RUNNING, JUMPING, CUTTING, OTHER}


# ─────────────────────────────────────────────────────────────────────────────
# CMU catalog
# Source: http://mocap.cs.cmu.edu/search.php
# Format: { subject_id: { motion_num: category } }
# motion_num 0 = applies to all motions for that subject
# ─────────────────────────────────────────────────────────────────────────────

CMU_CATALOG: dict[int, dict[int, str]] = {

    # ── Jumping ───────────────────────────────────────────────────────────────
    1:  {1: JUMPING, 7: JUMPING, 11: JUMPING, 13: JUMPING},  # playground jumps
    13: {0: JUMPING},   # jumping jacks + jumps
    18: {0: JUMPING},   # jump + land sequences
    36: {1: JUMPING, 2: JUMPING},
    43: {1: JUMPING, 3: JUMPING},
    86: {0: JUMPING},

    # ── Running ───────────────────────────────────────────────────────────────
    2:  {0: RUNNING},   # walk/run transitions
    9:  {0: RUNNING},   # running various speeds
    16: {0: RUNNING},   # run + direction changes
    35: {1: RUNNING, 2: RUNNING, 3: RUNNING},
    56: {0: RUNNING},
    91: {0: RUNNING},

    # ── Cutting / agility / direction change ──────────────────────────────────
    15: {0: CUTTING},   # side-step, cut, pivot
    39: {0: CUTTING},   # agility drills
    49: {1: CUTTING, 2: CUTTING},
    105:{0: CUTTING},

}


def get_category(subject: int, motion_num: int) -> str:
    """
    Returns the catalog label for a subject + motion number.
    Checks exact motion number first, then subject-wide default (key 0).
    Returns OTHER if not found.
    """
    subj = CMU_CATALOG.get(subject, {})
    return subj.get(motion_num, subj.get(0, OTHER))


# ─────────────────────────────────────────────────────────────────────────────
# Trial dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionTrial:
    subject:    int
    motion_num: int
    category:   str
    asf_path:   Path
    amc_path:   Path
    motions:    list = field(default_factory=list, repr=False)
    positions:  list = field(default_factory=list, repr=False)

    def __str__(self):
        return (f'Subject {self.subject:03d} | Motion {self.motion_num:02d} | '
                f'{self.category:<8} | {self.amc_path.name}')


# ─────────────────────────────────────────────────────────────────────────────
# Multi-file loader
# ─────────────────────────────────────────────────────────────────────────────

def iter_motions(
    data_root:  str | Path,
    categories: Optional[List[str]] = None,
    verbose:    bool = True,
) -> Iterator[MotionTrial]:
    """
    Walk data_root, yield MotionTrial objects for files matching the
    requested categories. Unlabeled files (OTHER) are skipped unless
    OTHER is explicitly included in categories.

    Expected layout
    ───────────────
    data_root/
      subject_01/
        skeleton.asf
        motions/
          01_01.amc
          ...

    Parameters
    ----------
    data_root  : root directory of CMU data
    categories : categories to yield; None → all labeled (no OTHER)
    verbose    : print one line per yielded trial
    """
    data_root  = Path(data_root)
    categories = set(categories) if categories else {RUNNING, JUMPING, CUTTING}

    subject_dirs = sorted(
        d for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith('subject_')
    )

    if verbose:
        print(f'Found {len(subject_dirs)} subject directories.')

    for subj_dir in subject_dirs:
        asf_files = list(subj_dir.glob('*.asf'))
        if not asf_files:
            continue

        subject_num = _subject_num(subj_dir.name)
        asf_path    = asf_files[0]
        motions_dir = subj_dir / 'motions'
        if not motions_dir.exists():
            continue

        for amc_path in sorted(motions_dir.glob('*.amc')):
            motion_num = _motion_num(amc_path.name)
            category   = get_category(subject_num, motion_num)

            if category not in categories:
                continue

            try:
                joints      = parse_asf(str(asf_path))
                raw_motions = parse_amc(str(amc_path))
                positions   = compute_joint_positions(raw_motions, joints)
            except Exception as e:
                if verbose:
                    print(f'  [skip] {amc_path.name} — {e}')
                continue

            trial = MotionTrial(
                subject    = subject_num,
                motion_num = motion_num,
                category   = category,
                asf_path   = asf_path,
                amc_path   = amc_path,
                motions    = raw_motions,
                positions  = positions,
            )

            if verbose:
                print(f'  ✓ {trial}  ({len(positions)} frames)')

            yield trial


# ─────────────────────────────────────────────────────────────────────────────
# Filename helpers
# ─────────────────────────────────────────────────────────────────────────────

def _subject_num(dirname: str) -> int:
    try:
        return int(dirname.split('_')[-1])
    except ValueError:
        return 0


def _motion_num(filename: str) -> int:
    try:
        return int(Path(filename).stem.split('_')[-1])
    except ValueError:
        return 0