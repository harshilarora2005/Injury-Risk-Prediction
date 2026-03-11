import numpy as np

def get(positions, joint_name):
    out = np.full((len(positions), 3), np.nan)
    for i, frame in enumerate(positions):
        if joint_name in frame:
            out[i] = frame[joint_name]
    return out


def angle_between(v1, v2):
    v1n = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(v1n * v2n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def asymmetry_index(left, right):
    return 100.0 * (left - right) / (0.5 * (np.abs(left) + np.abs(right)) + 1e-9)


def knee_flexion(positions):
    """
    Knee flexion (degrees) from the hip–knee–ankle segment angle.
    0° = fully extended. Returns left and right per-frame arrays.
    """
    results = {}
    for side, (hip_j, knee_j, ankle_j) in [
        ('left',  ('lfemur', 'ltibia', 'lfoot')),
        ('right', ('rfemur', 'rtibia', 'rfoot')),
    ]:
        v_thigh = get(positions, hip_j)   - get(positions, knee_j)
        v_shank = get(positions, ankle_j) - get(positions, knee_j)
        results[f'{side}_knee_flexion_deg'] = 180.0 - angle_between(v_thigh, v_shank)
    return results

def hip_flexion(positions):
    """
    Hip flexion (degrees): angle between downward reference [0, -1, 0]
    and the root→femur (thigh) vector.
    """
    results = {}
    ref = np.array([0.0, -1.0, 0.0])
    for side, femur_j in [('left', 'lfemur'), ('right', 'rfemur')]:
        thigh_vec = get(positions, femur_j) - get(positions, 'root')
        refs      = np.tile(ref, (len(thigh_vec), 1))
        results[f'{side}_hip_flexion_deg'] = angle_between(thigh_vec, refs)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Ankle Dorsiflexion
# ─────────────────────────────────────────────────────────────────────────────

def ankle_dorsiflexion(positions):
    """Ankle dorsiflexion (degrees): angle between shank and foot vectors."""
    results = {}
    for side, (tibia_j, foot_j, toes_j) in [
        ('left',  ('ltibia', 'lfoot', 'ltoes')),
        ('right', ('rtibia', 'rfoot', 'rtoes')),
    ]:
        v_shank = get(positions, tibia_j) - get(positions, foot_j)
        v_foot  = get(positions, toes_j)  - get(positions, foot_j)
        results[f'{side}_ankle_dorsiflexion_deg'] = angle_between(v_shank, v_foot)
    return results

def asymmetry_features(flexion_features):
    """
    Asymmetry Index (%) for knee, hip, and ankle between left and right.
    Positive = left side dominant; negative = right side dominant.
    """
    results = {}
    pairs = [
        ('left_knee_flexion_deg',       'right_knee_flexion_deg',       'knee_flexion'),
        ('left_hip_flexion_deg',        'right_hip_flexion_deg',        'hip_flexion'),
        ('left_ankle_dorsiflexion_deg', 'right_ankle_dorsiflexion_deg', 'ankle_dorsiflexion'),
    ]
    for l_key, r_key, label in pairs:
        if l_key in flexion_features and r_key in flexion_features:
            ai = asymmetry_index(flexion_features[l_key], flexion_features[r_key])
            results[f'{label}asymmetry_index'] = ai
    return results

def knee_valgus_proxy(positions):
    """
    Medial-lateral knee deviation relative to the hip-ankle line,
    projected onto the frontal plane (X-Z in CMU coordinates, Y = vertical).

    Positive = knee inside the hip-ankle line (valgus).
    Negative = knee outside (varus).
    """
    results = {}
    for side, (hip_j, knee_j, ankle_j) in [
        ('left',  ('lfemur', 'ltibia', 'lfoot')),
        ('right', ('rfemur', 'rtibia', 'rfoot')),
    ]:
        hip_xz   = get(positions, hip_j)[:,   [0, 2]]
        knee_xz  = get(positions, knee_j)[:,  [0, 2]]
        ankle_xz = get(positions, ankle_j)[:, [0, 2]]

        leg_vec = ankle_xz - hip_xz
        hk_vec  = knee_xz  - hip_xz
        leg_len = np.linalg.norm(leg_vec, axis=1, keepdims=True) + 1e-9

        # 2D signed cross product: positive = medial (valgus) deviation
        cross = (leg_vec[:, 0] * hk_vec[:, 1]
                 - leg_vec[:, 1] * hk_vec[:, 0]) / leg_len[:, 0]

        results[f'{side}_knee_valgus_proxy_cm'] = cross
    return results

def hip_knee_flexion_ratio(flexion_features):
    """
    Hip flexion / knee flexion ratio per frame.
    Low ratio at initial contact → knee-dominant loading → ACL risk proxy.
    (Literature normal range: ~0.5–0.8; Powers 2025.)
    """
    results = {}
    for side in ('left', 'right'):
        hip_key  = f'{side}_hip_flexion_deg'
        knee_key = f'{side}_knee_flexion_deg'
        if hip_key in flexion_features and knee_key in flexion_features:
            ratio = flexion_features[hip_key] / (flexion_features[knee_key] + 1e-9)
            results[f'{side}_hip_knee_ratio'] = ratio
    return results


def extract_all_features(positions):
    knee  = knee_flexion(positions)
    hip   = hip_flexion(positions)
    ankle = ankle_dorsiflexion(positions)
    flexion = {**knee, **hip, **ankle}

    features = {}
    features.update(flexion)                      
    features.update(hip)                           
    features.update(ankle)                          
    features.update(asymmetry_features(flexion))    
    features.update(knee_valgus_proxy(positions))   
    features.update(hip_knee_flexion_ratio(flexion))
    return features

def summarise_features(features):
    """
    Reduce per-frame arrays to per-trial scalars:
    mean, std, min, max, range for each time-series feature.

    Returns
    -------
    summary : dict { feature_stat_name -> float }
    """
    summary = {}
    for key, val in features.items():
        arr   = np.asarray(val, dtype=float)
        if arr.ndim != 1 or len(arr) < 2:
            continue
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            continue
        summary[f'{key}_mean']  = float(np.mean(valid))
        summary[f'{key}_std']   = float(np.std(valid))
        summary[f'{key}_min']   = float(np.min(valid))
        summary[f'{key}_max']   = float(np.max(valid))
        summary[f'{key}_range'] = float(np.ptp(valid))
    return summary