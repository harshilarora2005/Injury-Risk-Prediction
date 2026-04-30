import numpy as np
import cv2
from scipy.signal import savgol_filter
from typing import Tuple, Optional

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions


# ── Landmark indices ──────────────────────────────────────────
MP_LEFT_SHOULDER  = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_HIP       = 23
MP_RIGHT_HIP      = 24
MP_LEFT_KNEE      = 25
MP_RIGHT_KNEE     = 26
MP_LEFT_ANKLE     = 27
MP_RIGHT_ANKLE    = 28

VISIBILITY_THRESHOLD = 0.5


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b - a


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> Optional[float]:
    v1 = _vec(vertex, a)
    v2 = _vec(vertex, b)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan

    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _lm(landmarks, idx, frame_w, frame_h):
    lm = landmarks[idx]

    vis = getattr(lm, "visibility", 1.0)

    if vis < VISIBILITY_THRESHOLD:
        return None, vis

    return np.array([lm.x * frame_w, lm.y * frame_h], dtype=np.float32), vis


def extract_features_from_video(video_path: str, progress_callback=None):
    model_path = "pose_landmarker.task"

    base_options = BaseOptions(model_asset_path=model_path)

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    pose = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_features = []
    all_landmarks = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((frame_idx / fps) * 1000)

        result = pose.detect_for_video(mp_image, timestamp_ms)

        if progress_callback and frame_idx % 30 == 0:
            pct = int((frame_idx / max(total_frames, 1)) * 50)
            progress_callback(pct, f"Extracting pose — frame {frame_idx}/{total_frames}")

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            all_landmarks.append(lms)

            def get(idx):
                return _lm(lms, idx, frame_w, frame_h)

            ls, _ = get(MP_LEFT_SHOULDER)
            rs, _ = get(MP_RIGHT_SHOULDER)
            lh, _ = get(MP_LEFT_HIP)
            rh, _ = get(MP_RIGHT_HIP)
            lk, _ = get(MP_LEFT_KNEE)
            rk, _ = get(MP_RIGHT_KNEE)
            la, _ = get(MP_LEFT_ANKLE)
            ra, _ = get(MP_RIGHT_ANKLE)

            kf_l = _angle_deg(lh, lk, la) if lh is not None and lk is not None and la is not None else np.nan
            kf_r = _angle_deg(rh, rk, ra) if rh is not None and rk is not None and ra is not None else np.nan

            hf_l = _angle_deg(ls, lh, lk) if ls is not None and lh is not None and lk is not None else np.nan
            hf_r = _angle_deg(rs, rh, rk) if rs is not None and rh is not None and rk is not None else np.nan

            asym = abs(kf_l - kf_r) if not (np.isnan(kf_l) or np.isnan(kf_r)) else np.nan

            trunk = np.nan
            if ls is not None and rs is not None and lh is not None and rh is not None:
                mid_s = (ls + rs) / 2
                mid_h = (lh + rh) / 2

                vec = mid_s - mid_h
                vertical = np.array([0, -1], dtype=np.float32)

                n = np.linalg.norm(vec)

                if n > 1e-6:
                    cos_t = np.clip(np.dot(vec / n, vertical), -1.0, 1.0)
                    trunk = float(np.degrees(np.arccos(cos_t)))

            raw_features.append([kf_l, kf_r, hf_l, hf_r, asym, trunk])

        else:
            all_landmarks.append(None)
            raw_features.append([np.nan] * 6)

        frame_idx += 1

    cap.release()
    pose.close()

    raw = np.array(raw_features, dtype=np.float32)

    sg_win = min(7, len(raw) if len(raw) % 2 == 1 else len(raw) - 1)
    sg_win = max(sg_win, 3)

    smoothed = raw.copy()

    for col in range(raw.shape[1]):
        col_data = raw[:, col]
        valid_mask = ~np.isnan(col_data)

        if valid_mask.sum() > sg_win:
            smoothed[valid_mask, col] = savgol_filter(col_data[valid_mask], sg_win, 2)

    vel_l = np.gradient(smoothed[:, 0])
    vel_r = np.gradient(smoothed[:, 1])

    vel_l = np.where(np.isnan(smoothed[:, 0]), np.nan, vel_l)
    vel_r = np.where(np.isnan(smoothed[:, 1]), np.nan, vel_r)

    feature_matrix = np.column_stack([smoothed, vel_l, vel_r])

    return feature_matrix, fps, total_frames, all_landmarks