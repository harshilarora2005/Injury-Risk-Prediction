"""
feature_extraction.py
─────────────────────
Extracts biomechanically relevant features from CMU MoCap joint positions
and raw motion angle data.

Input
-----
positions : list of dicts  { joint_name -> np.array([x, y, z]) }   (from compute_joint_positions)
motions   : list of dicts  { joint_name -> [float, ...] }           (from parse_amc)

Output
------
features : dict  { feature_name -> np.array (n_frames,) }

Feature Groups
--------------
1. Joint Flexion Curves        — knee, hip, ankle angles over time
2. Velocity Profiles           — per-joint speed (magnitude of positional derivative)
3. Deceleration Patterns       — acceleration signal, highlights landing / cut events
4. Left-Right Asymmetry        — signed and absolute asymmetry index per feature pair
5. Knee Valgus Proxy           — medial-lateral knee deviation relative to hip-foot line
6. Hip-Knee Flexion Ratio      — hip flexion / knee flexion (ACL risk proxy)
7. Ground Contact Events       — heel-strike and toe-off detection via foot velocity minima
"""

import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _positions_to_array(positions, joint_name):
    """Extract (n_frames, 3) array for one joint. Missing frames → NaN row."""
    out = np.full((len(positions), 3), np.nan)
    for i, frame in enumerate(positions):
        if joint_name in frame:
            out[i] = frame[joint_name]
    return out


def _angle_between(v1, v2):
    """
    Angle (degrees) between two vectors or arrays of vectors.
    v1, v2 : (..., 3)
    """
    v1n = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2, axis=-1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(v1n * v2n, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def _asymmetry_index(left, right):
    """
    Asymmetry Index (%) = 100 * (left - right) / (0.5 * (|left| + |right|) + 1e-9)
    Positive → left > right.
    """
    return 100.0 * (left - right) / (0.5 * (np.abs(left) + np.abs(right)) + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Joint Flexion Curves
# ─────────────────────────────────────────────────────────────────────────────

def knee_flexion(positions):
    """
    Knee flexion angle (degrees) computed from the hip–knee–ankle segment angle.
    0° = fully extended leg.
    Returns left and right arrays of shape (n_frames,).
    """
    results = {}
    for side, (hip, knee, ankle) in [
        ('left',  ('lfemur',  'ltibia', 'lfoot')),
        ('right', ('rfemur',  'rtibia', 'rfoot')),
    ]:
        hip_pos   = _positions_to_array(positions, hip)
        knee_pos  = _positions_to_array(positions, knee)
        ankle_pos = _positions_to_array(positions, ankle)

        v_thigh  = hip_pos   - knee_pos   # thigh vector (knee→hip)
        v_shank  = ankle_pos - knee_pos   # shank vector (knee→ankle)

        angle = _angle_between(v_thigh, v_shank)
        # 180° = straight leg; convert so 0° = straight
        results[f'{side}_knee_flexion_deg'] = 180.0 - angle

    return results


def hip_flexion(positions):
    """
    Hip flexion angle (degrees): angle between pelvis-vertical and thigh vector.
    Uses root→femur as thigh and a downward reference [0, -1, 0].
    """
    results = {}
    ref = np.array([0.0, -1.0, 0.0])

    for side, (femur_joint,) in [
        ('left',  ('lfemur',)),
        ('right', ('rfemur',)),
    ]:
        root_pos  = _positions_to_array(positions, 'root')
        femur_pos = _positions_to_array(positions, femur_joint)

        thigh_vec = femur_pos - root_pos   # hip→knee direction
        refs      = np.tile(ref, (len(thigh_vec), 1))
        angle     = _angle_between(thigh_vec, refs)

        results[f'{side}_hip_flexion_deg'] = angle

    return results


def ankle_dorsiflexion(positions):
    """
    Ankle dorsiflexion (degrees): angle between shank and foot vectors.
    """
    results = {}
    for side, (tibia, foot, toes) in [
        ('left',  ('ltibia', 'lfoot', 'ltoes')),
        ('right', ('rtibia', 'rfoot', 'rtoes')),
    ]:
        tibia_pos = _positions_to_array(positions, tibia)
        foot_pos  = _positions_to_array(positions, foot)
        toes_pos  = _positions_to_array(positions, toes)

        v_shank = tibia_pos - foot_pos
        v_foot  = toes_pos  - foot_pos
        angle   = _angle_between(v_shank, v_foot)

        results[f'{side}_ankle_dorsiflexion_deg'] = angle

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Velocity Profiles
# ─────────────────────────────────────────────────────────────────────────────

def joint_velocity(positions, joint_names=None, fps=120):
    """
    Speed (cm/s at CMU scale) for each joint as magnitude of frame-to-frame
    positional derivative.

    Parameters
    ----------
    fps : assumed frame rate (CMU mocap default 120 Hz)
    """
    if joint_names is None:
        joint_names = [
            'root', 'lfemur', 'rfemur', 'ltibia', 'rtibia',
            'lfoot', 'rfoot', 'lhumerus', 'rhumerus',
        ]

    results = {}
    for name in joint_names:
        pos = _positions_to_array(positions, name)          # (n, 3)
        vel = np.gradient(pos, axis=0) * fps                # cm/s
        speed = np.linalg.norm(vel, axis=1)                 # scalar per frame
        results[f'{name}_speed'] = speed
        results[f'{name}_velocity_xyz'] = vel               # keep 3-axis too

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deceleration Patterns
# ─────────────────────────────────────────────────────────────────────────────

def deceleration_profile(positions, fps=120):
    """
    Signed deceleration of root (COM proxy) and each foot.
    Negative values = deceleration events (landing, cutting).

    Returns scalar deceleration magnitude and event frames.
    """
    results = {}

    for joint in ['root', 'lfoot', 'rfoot']:
        pos   = _positions_to_array(positions, joint)
        vel   = np.gradient(pos, axis=0) * fps
        speed = np.linalg.norm(vel, axis=1)
        accel = np.gradient(speed) * fps                    # dv/dt

        results[f'{joint}_acceleration'] = accel

        # Detect deceleration peaks (negative acceleration > 1 std below mean)
        threshold = accel.mean() - accel.std()
        peaks, _ = find_peaks(-accel, height=-threshold)    # find troughs
        results[f'{joint}_decel_event_frames'] = peaks

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Left-Right Asymmetry
# ─────────────────────────────────────────────────────────────────────────────

def asymmetry_features(flexion_features):
    """
    Computes asymmetry index (%) for knee, hip, and ankle between left and right.
    Input: dict containing left/right flexion curve arrays (from above functions).
    """
    results = {}
    pairs = [
        ('left_knee_flexion_deg',        'right_knee_flexion_deg',        'knee_flexion'),
        ('left_hip_flexion_deg',         'right_hip_flexion_deg',         'hip_flexion'),
        ('left_ankle_dorsiflexion_deg',  'right_ankle_dorsiflexion_deg',  'ankle_dorsiflexion'),
    ]
    for l_key, r_key, label in pairs:
        if l_key in flexion_features and r_key in flexion_features:
            left  = flexion_features[l_key]
            right = flexion_features[r_key]
            ai    = _asymmetry_index(left, right)
            results[f'{label}_asymmetry_index']     = ai
            results[f'{label}_asymmetry_abs_mean']  = np.nanmean(np.abs(ai))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Knee Valgus Proxy
# ─────────────────────────────────────────────────────────────────────────────

def knee_valgus_proxy(positions):
    """
    Approximates dynamic knee valgus as the medial-lateral displacement of the
    knee relative to the hip-ankle line (projected onto the frontal plane, XZ).

    Positive = knee inside the hip-ankle line (valgus).
    Negative = knee outside (varus).

    Returns left and right arrays (n_frames,).
    """
    results = {}
    for side, (hip_j, knee_j, ankle_j) in [
        ('left',  ('lfemur',  'ltibia', 'lfoot')),
        ('right', ('rfemur',  'rtibia', 'rfoot')),
    ]:
        hip_pos   = _positions_to_array(positions, hip_j)
        knee_pos  = _positions_to_array(positions, knee_j)
        ankle_pos = _positions_to_array(positions, ankle_j)

        # Project onto frontal plane (X and Z axes; Y = vertical in CMU)
        hip_xz   = hip_pos[:,   [0, 2]]
        knee_xz  = knee_pos[:,  [0, 2]]
        ankle_xz = ankle_pos[:, [0, 2]]

        # Vector from hip to ankle (the "ideal" leg line)
        leg_vec = ankle_xz - hip_xz
        leg_len = np.linalg.norm(leg_vec, axis=1, keepdims=True) + 1e-9

        # Vector from hip to knee
        hk_vec = knee_xz - hip_xz

        # Signed lateral deviation = cross product (2D) of leg_vec and hk_vec
        # cross2d(a, b) = a[0]*b[1] - a[1]*b[0]
        cross = (leg_vec[:, 0] * hk_vec[:, 1]
                 - leg_vec[:, 1] * hk_vec[:, 0]) / leg_len[:, 0]

        results[f'{side}_knee_valgus_proxy_cm'] = cross

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hip-Knee Flexion Ratio
# ─────────────────────────────────────────────────────────────────────────────

def hip_knee_flexion_ratio(flexion_features):
    """
    Hip flexion / knee flexion ratio per frame.
    Low ratio (<0.5) at initial contact → knee-dominant loading → ACL risk proxy
    (Powers 2010 pattern; threshold literature: ~0.5–0.8 normal range).
    """
    results = {}
    for side in ('left', 'right'):
        hip_key  = f'{side}_hip_flexion_deg'
        knee_key = f'{side}_knee_flexion_deg'
        if hip_key in flexion_features and knee_key in flexion_features:
            hip   = flexion_features[hip_key]
            knee  = flexion_features[knee_key]
            ratio = hip / (knee + 1e-9)
            results[f'{side}_hip_knee_ratio'] = ratio

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ground Contact Events
# ─────────────────────────────────────────────────────────────────────────────

def ground_contact_events(positions, fps=120, min_gap_frames=10):
    """
    Detect heel-strike and toe-off events from foot joint vertical (Y) velocity.

    Heel-strike  : downward velocity of foot crosses zero (foot stopping)
    Toe-off      : upward velocity peak of foot (foot leaving ground)

    Returns frame indices for each event per side.
    """
    results = {}

    for side, foot_joint, toe_joint in [
        ('left',  'lfoot', 'ltoes'),
        ('right', 'rfoot', 'rtoes'),
    ]:
        foot_pos = _positions_to_array(positions, foot_joint)
        foot_y   = foot_pos[:, 1]                           # vertical axis
        foot_vel = np.gradient(foot_y) * fps

        # Heel-strike: velocity goes from negative → positive (foot hitting ground)
        hs_frames = np.where(
            (foot_vel[:-1] < 0) & (foot_vel[1:] >= 0)
        )[0]

        # Toe-off: velocity goes from positive → negative (foot leaving ground)
        to_frames = np.where(
            (foot_vel[:-1] > 0) & (foot_vel[1:] <= 0)
        )[0]

        # Filter out events too close together (noise)
        def _filter_events(frames):
            if len(frames) == 0:
                return frames
            keep = [frames[0]]
            for f in frames[1:]:
                if f - keep[-1] >= min_gap_frames:
                    keep.append(f)
            return np.array(keep)

        results[f'{side}_heel_strike_frames'] = _filter_events(hs_frames)
        results[f'{side}_toe_off_frames']     = _filter_events(to_frames)

        # Stride time (frames between consecutive heel strikes)
        hs = results[f'{side}_heel_strike_frames']
        if len(hs) > 1:
            stride_times = np.diff(hs) / fps  # seconds
            results[f'{side}_stride_time_s_mean'] = float(np.mean(stride_times))
            results[f'{side}_stride_time_s']       = stride_times

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Master extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_features(positions, motions=None, fps=120):
    """
    Run the full feature extraction pipeline on one motion sequence.

    Parameters
    ----------
    positions : list of dicts  { joint_name -> np.array([x, y, z]) }
    motions   : list of dicts  (optional, reserved for raw-angle features)
    fps       : frame rate (default 120 Hz for CMU)

    Returns
    -------
    features : dict { feature_name -> np.array or scalar }
    """
    features = {}

    # 1. Flexion curves
    knee  = knee_flexion(positions)
    hip   = hip_flexion(positions)
    ankle = ankle_dorsiflexion(positions)
    features.update(knee)
    features.update(hip)
    features.update(ankle)

    # 2. Velocity profiles
    features.update(joint_velocity(positions, fps=fps))

    # 3. Deceleration patterns
    features.update(deceleration_profile(positions, fps=fps))

    # 4. Asymmetry
    flexion_dict = {**knee, **hip, **ankle}
    features.update(asymmetry_features(flexion_dict))

    # 5. Knee valgus proxy
    features.update(knee_valgus_proxy(positions))

    # 6. Hip-knee flexion ratio
    features.update(hip_knee_flexion_ratio(flexion_dict))

    # 7. Ground contact events
    features.update(ground_contact_events(positions, fps=fps))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics helper (for feature analysis / ML input)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_features(features):
    """
    Reduce per-frame arrays to per-trial scalar statistics:
    mean, std, min, max, peak-to-peak for each numeric time-series feature.

    Skips event-index arrays and pre-computed scalars.

    Returns
    -------
    summary : dict { feature_stat_name -> float }
    """
    summary = {}
    skip_suffixes = ('_frames', '_xyz')  # keep raw arrays out of summary

    for key, val in features.items():
        if any(key.endswith(s) for s in skip_suffixes):
            continue
        arr = np.asarray(val, dtype=float)
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