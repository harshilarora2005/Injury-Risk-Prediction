"""
feature_extraction.py
─────────────────────
Extracts the five biomechanically validated risk features from CMU MoCap
joint positions. Feature selection is grounded in:

    Belkhelladi et al. (2025) — systematic review, 28 studies / 2819 athletes
    Powers (2025)             — hip-knee mechanics and ACL loading

Feature groups
--------------
1. Knee flexion angle          — most consistent ACL predictor across all studies
2. Knee valgus proxy           — present in 100% of knee-loading studies
3. Hip-knee flexion ratio      — explicitly named in cutting + Powers literature
4. Hip flexion / adduction     — significant in 83% of hip biomechanics studies
5. Left-right asymmetry index  — flagged at knee and ankle level in cutting studies
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(positions, joint):
    """Extract (n_frames, 3) array for one joint. Missing frames → NaN row."""
    out = np.full((len(positions), 3), np.nan)
    for i, frame in enumerate(positions):
        if joint in frame:
            out[i] = frame[joint]
    return out


def _angle_between(v1, v2):
    """Angle in degrees between two (n, 3) arrays of vectors."""
    v1n = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(v1n * v2n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _asymmetry_index(left, right):
    """
    Asymmetry Index (%) = 100 * (L - R) / (0.5 * (|L| + |R|))
    NaN-masked when both sides < 1° to prevent division blow-up.
    Clamped to ±200%.
    """
    denom = 0.5 * (np.abs(left) + np.abs(right))
    ai = 100.0 * (left - right) / (denom + 1e-9)
    ai = np.where(denom < 1.0, np.nan, ai)
    return np.clip(ai, -200.0, 200.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Knee flexion angle
#    Belkhelladi et al. 2025: decreased knee flexion → increased ACL loading
#    Risk threshold: < 30° at initial contact (Powers 2025)
# ─────────────────────────────────────────────────────────────────────────────

def knee_flexion(positions):
    """
    Knee flexion (degrees) from the hip–knee–ankle segment angle.
    0° = fully extended, 90° = right angle.
    """
    results = {}
    for side, hip_j, knee_j, ankle_j in [
        ('left',  'lfemur', 'ltibia', 'lfoot'),
        ('right', 'rfemur', 'rtibia', 'rfoot'),
    ]:
        hip   = _get(positions, hip_j)
        knee  = _get(positions, knee_j)
        ankle = _get(positions, ankle_j)

        v_thigh = hip   - knee
        v_shank = ankle - knee
        results[f'{side}_knee_flexion_deg'] = 180.0 - _angle_between(v_thigh, v_shank)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Knee valgus proxy
#    Belkhelladi et al. 2025: increased medial knee alignment → increased ACL load
#    Risk threshold: > 8° dynamic valgus (Powers 2025)
# ─────────────────────────────────────────────────────────────────────────────

def knee_valgus(positions):
    """
    Signed medial-lateral knee deviation from the hip–ankle line in the
    frontal plane (XZ). Positive = valgus (knee collapses inward).
    """
    results = {}
    for side, hip_j, knee_j, ankle_j in [
        ('left',  'lfemur', 'ltibia', 'lfoot'),
        ('right', 'rfemur', 'rtibia', 'rfoot'),
    ]:
        hip_xz   = _get(positions, hip_j)[:,   [0, 2]]
        knee_xz  = _get(positions, knee_j)[:,  [0, 2]]
        ankle_xz = _get(positions, ankle_j)[:, [0, 2]]

        leg_vec = ankle_xz - hip_xz
        hk_vec  = knee_xz  - hip_xz
        leg_len = np.linalg.norm(leg_vec, axis=1, keepdims=True) + 1e-9

        # 2D cross product = signed lateral deviation
        cross = (leg_vec[:, 0] * hk_vec[:, 1]
                 - leg_vec[:, 1] * hk_vec[:, 0]) / leg_len[:, 0]

        results[f'{side}_knee_valgus_cm'] = cross

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hip-knee flexion ratio
#    Powers 2025 + cutting study: low ratio at initial contact = knee-dominant
#    loading pattern. Normal range ~0.5–0.8; < 0.5 = elevated risk
# ─────────────────────────────────────────────────────────────────────────────

def hip_knee_ratio(positions):
    """
    Hip flexion angle / knee flexion angle per frame.
    Requires knee_flexion and hip_flexion to already be computed.
    """
    knee = knee_flexion(positions)
    hip  = hip_flexion(positions)

    results = {}
    for side in ('left', 'right'):
        h = hip[f'{side}_hip_flexion_deg']
        k = knee[f'{side}_knee_flexion_deg']
        results[f'{side}_hip_knee_ratio'] = h / (k + 1e-9)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hip flexion and adduction
#    Belkhelladi et al. 2025: increased hip abduction/internal rotation
#    significant in 83% of hip biomechanics studies
#    Risk threshold: hip adduction > 10° (Powers 2025)
# ─────────────────────────────────────────────────────────────────────────────

def hip_flexion(positions):
    """
    Hip flexion (degrees): angle between the thigh vector and a downward
    vertical reference [0, -1, 0].
    """
    ref = np.array([0.0, -1.0, 0.0])
    results = {}
    for side, femur_j in [('left', 'lfemur'), ('right', 'rfemur')]:
        root  = _get(positions, 'root')
        femur = _get(positions, femur_j)
        thigh = femur - root
        refs  = np.tile(ref, (len(thigh), 1))
        results[f'{side}_hip_flexion_deg'] = _angle_between(thigh, refs)
    return results


def hip_adduction(positions):
    """
    Hip adduction (degrees): lateral deviation of the thigh from the
    pelvis midline in the frontal plane (XZ). Positive = adduction.
    """
    results = {}
    for side, femur_j in [('left', 'lfemur'), ('right', 'rfemur')]:
        root_xz  = _get(positions, 'root')[:,  [0, 2]]
        femur_xz = _get(positions, femur_j)[:, [0, 2]]
        thigh_xz = femur_xz - root_xz

        # Angle from vertical (0, 1) in frontal plane
        ref = np.tile([0.0, 1.0], (len(thigh_xz), 1))
        norm_t = thigh_xz / (np.linalg.norm(thigh_xz, axis=1, keepdims=True) + 1e-9)
        dot    = np.clip(np.sum(norm_t * ref, axis=1), -1.0, 1.0)
        angle  = np.degrees(np.arccos(dot))

        # Sign: left adduction = thigh moves right (+X), right adduction = moves left (-X)
        sign = np.sign(thigh_xz[:, 0]) * (-1 if side == 'left' else 1)
        results[f'{side}_hip_adduction_deg'] = angle * sign

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Left-right asymmetry index
#    Cutting study 2025: asymmetries most pronounced at knee and ankle;
#    > 15% asymmetry index flagged as clinically meaningful
# ─────────────────────────────────────────────────────────────────────────────

def asymmetry(positions):
    """
    Asymmetry index (%) for knee flexion, knee valgus, and hip flexion.
    """
    kf = knee_flexion(positions)
    kv = knee_valgus(positions)
    hf = hip_flexion(positions)

    results = {}
    for label, l_key, r_key in [
        ('knee_flexion', 'left_knee_flexion_deg',  'right_knee_flexion_deg'),
        ('knee_valgus',  'left_knee_valgus_cm',    'right_knee_valgus_cm'),
        ('hip_flexion',  'left_hip_flexion_deg',   'right_hip_flexion_deg'),
    ]:
        src = {**kf, **kv, **hf}
        ai  = _asymmetry_index(src[l_key], src[r_key])
        results[f'{label}_asymmetry_pct'] = ai

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(positions, motions=None, fps=120):
    """
    Run all five feature groups. Returns a flat dict of named arrays.
    motions and fps are accepted for API compatibility but not used here.
    """
    features = {}
    features.update(knee_flexion(positions))
    features.update(knee_valgus(positions))
    features.update(hip_knee_ratio(positions))
    features.update(hip_flexion(positions))
    features.update(hip_adduction(positions))
    features.update(asymmetry(positions))
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics  (per-trial scalars for ML feature matrix)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_features(features):
    """
    Reduce per-frame arrays → per-trial scalars: mean, std, min, max, range.
    """
    summary = {}
    for key, val in features.items():
        arr   = np.asarray(val, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            continue
        summary[f'{key}_mean']  = float(np.mean(valid))
        summary[f'{key}_std']   = float(np.std(valid))
        summary[f'{key}_min']   = float(np.min(valid))
        summary[f'{key}_max']   = float(np.max(valid))
        summary[f'{key}_range'] = float(np.ptp(valid))
    return summary