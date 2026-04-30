"""
inference.py
Loads acl_risk_model_phase6.pt and runs window-by-window inference.
Returns a list of WindowPrediction dicts + raw numpy predictions.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

MODEL_PATH = Path(__file__).parent / "models" / "acl_risk_model_phase6.pt"

# ── Model Architecture (must match training exactly) ──────────────────────────

class ACLRiskLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.4, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last timestep
        return self.fc(out)


# ── Singleton model cache ──────────────────────────────────────────────────────

_model = None
_checkpoint_meta = {}


def load_model(model_path: str = None) -> Tuple[ACLRiskLSTM, dict]:
    global _model, _checkpoint_meta
    if _model is not None:
        return _model, _checkpoint_meta

    path = model_path or str(MODEL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Place acl_risk_model_phase6.pt in backend/models/"
        )

    checkpoint = torch.load(path, map_location="cpu")
    meta = {
        "feature_schema": checkpoint.get("feature_schema", [
            "knee_flexion_left", "knee_flexion_right",
            "hip_flexion_left", "hip_flexion_right",
            "lr_knee_asymmetry", "trunk_lean",
            "angular_velocity_left", "angular_velocity_right",
        ]),
        "label_map": checkpoint.get("label_map", {"Low": 0, "Medium": 1, "High": 2}),
        "window_size": checkpoint.get("window_size", 30),
        "known_limitations": checkpoint.get("known_limitations", [
            "2D pose estimates carry up to 18° absolute error vs. 3D clinical systems; relative temporal trends are reliable, absolute angle values are not",
            "Literature thresholds used as directional references, not exact clinical cutoffs",
            "Proxy labels used in training; no verified injury outcome data",
            "System outputs movement quality indicators only — not a medical diagnosis",
        ]),
    }

    model = ACLRiskLSTM(
        input_size=len(meta["feature_schema"]),
        hidden_size=128,
        num_layers=2,
        dropout=0.4,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _model = model
    _checkpoint_meta = meta
    return _model, _checkpoint_meta


def validate_input(feature_matrix: np.ndarray, fps: float, meta: dict) -> None:
    """Raise ValueError with a clear message if any validation gate fails."""
    expected_cols = len(meta["feature_schema"])
    if feature_matrix.shape[1] != expected_cols:
        raise ValueError(
            f"Feature matrix has {feature_matrix.shape[1]} columns; "
            f"model expects {expected_cols} ({meta['feature_schema']})"
        )
    if fps < 25:
        raise ValueError(
            f"Video FPS is {fps:.1f}. Minimum required is 25 FPS. "
            "Lower frame rates produce unreliable angular velocity derivatives."
        )


def run_inference(
    feature_matrix: np.ndarray,
    model: ACLRiskLSTM,
    meta: dict,
    output_csv_path: str,
    progress_callback=None,
) -> List[Optional[dict]]:
    """
    Slide a window of size `window_size` over `feature_matrix` and run the BiLSTM.
    Returns list of dicts (one per window) or None for corrupted windows.
    Also saves per_window_predictions.csv to output_csv_path.
    """
    window_size = meta["window_size"]
    int_to_label = {v: k for k, v in meta["label_map"].items()}
    n_frames = len(feature_matrix)
    predictions = []

    total_windows = max(0, n_frames - window_size + 1)

    for i in range(total_windows):
        window = feature_matrix[i: i + window_size]          # (30, 8)
        nan_fraction = np.isnan(window).mean()

        if progress_callback and i % 10 == 0:
            pct = 50 + int((i / max(total_windows, 1)) * 35)  # inference = 50–85%
            progress_callback(pct, f"Running inference — window {i}/{total_windows}")

        if nan_fraction > 0.2:
            predictions.append(None)
            continue

        # NaN imputation: fill remaining NaNs with column mean of non-NaN values
        window_clean = window.copy()
        for col in range(window.shape[1]):
            col_vals = window_clean[:, col]
            col_mean = np.nanmean(col_vals) if not np.all(np.isnan(col_vals)) else 0.0
            window_clean[:, col] = np.where(np.isnan(col_vals), col_mean, col_vals)

        tensor = torch.FloatTensor(window_clean).unsqueeze(0)   # (1, 30, 8)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()

        pred_idx = int(probs.argmax())
        predictions.append({
            "start_frame": i,
            "end_frame": i + window_size - 1,
            "P_low": float(probs[0]),
            "P_medium": float(probs[1]),
            "P_high": float(probs[2]),
            "label": int_to_label.get(pred_idx, "Low"),
        })

    # Save reproducibility artifact
    rows = []
    for p in predictions:
        if p is None:
            rows.append({"start_frame": None, "end_frame": None,
                         "P_low": None, "P_medium": None, "P_high": None,
                         "label": "SKIPPED_NaN"})
        else:
            rows.append(p)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)

    return predictions