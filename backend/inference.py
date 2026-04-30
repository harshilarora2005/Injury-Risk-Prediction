"""
inference.py — ACL Risk BiLSTM model + Phase 7 inference utilities.

Public API expected by pipeline.py:
    load_model()      -> (model, meta)
    validate_input(feature_matrix, fps, meta) -> None  (raises on bad input)
    run_inference(feature_matrix, model, meta,
                  output_csv_path, progress_callback=None) -> List[dict]

Architecture is reconstructed from the trained checkpoint's state_dict:
    - Bidirectional LSTM, num_layers inferred from `lstm.weight_ih_l{n}` keys
    - Hidden size inferred from `lstm.weight_ih_l0` shape
    - Attention: Linear -> Tanh -> Linear  (keys attention.0 / attention.2)
    - Classifier: Dropout -> Linear -> ReLU -> Dropout -> Linear
                  (keys classifier.1 / classifier.4)
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("phase7.inference")

# --------------------------------------------------------------------------- #
# Defaults — override via env vars if needed
# --------------------------------------------------------------------------- #
DEFAULT_CHECKPOINT = os.environ.get(
    "ACL_MODEL_PATH",
    str(Path(__file__).parent / "models" / "acl_risk_model_phase6.pt"),
)
DEFAULT_WINDOW_SIZE = int(os.environ.get("ACL_WINDOW_SIZE", "30"))   # frames
DEFAULT_STRIDE      = int(os.environ.get("ACL_WINDOW_STRIDE", "15")) # frames
DEFAULT_FPS         = float(os.environ.get("ACL_TARGET_FPS", "30"))
LABELS              = ["Low", "Medium", "High"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ACLRiskLSTM(nn.Module):
    """BiLSTM + additive attention pooling + MLP classifier."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        attn_dim: int = 64,
        classifier_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # attention.0, attention.1=Tanh (no params), attention.2
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )
        # classifier.0=Dropout, .1=Linear, .2=ReLU, .3=Dropout, .4=Linear
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        lstm_out, _ = self.lstm(x)                       # (B, T, 2H)
        attn_logits = self.attention(lstm_out)           # (B, T, 1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)   # (B, 2H)
        logits = self.classifier(context)                # (B, C)
        if return_attention:
            return logits, attn_weights.squeeze(-1)
        return logits


# --------------------------------------------------------------------------- #
# Checkpoint loading with auto-inferred dims
# --------------------------------------------------------------------------- #
def _infer_dims_from_state_dict(sd: dict) -> dict:
    required = [
        "lstm.weight_ih_l0",
        "attention.0.weight",
        "attention.2.weight",
        "classifier.1.weight",
        "classifier.4.weight",
    ]
    missing = [k for k in required if k not in sd]
    if missing:
        raise KeyError(
            f"Checkpoint is missing expected keys: {missing}. "
            "It was likely trained with a different architecture."
        )

    w_ih0 = sd["lstm.weight_ih_l0"]
    hidden_size = w_ih0.shape[0] // 4
    input_size = w_ih0.shape[1]

    bidirectional = "lstm.weight_ih_l0_reverse" in sd
    if not bidirectional:
        logger.warning("Checkpoint is unidirectional; expected bidirectional.")

    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in sd:
        num_layers += 1

    attn_dim          = sd["attention.0.weight"].shape[0]
    classifier_hidden = sd["classifier.1.weight"].shape[0]
    num_classes       = sd["classifier.4.weight"].shape[0]

    return dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        attn_dim=attn_dim,
        classifier_hidden=classifier_hidden,
        num_classes=num_classes,
    )


def load_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[ACLRiskLSTM, dict]:
    """
    Load the ACLRiskLSTM checkpoint and return (model, meta).

    `meta` is the dict expected by pipeline.py / utils.py:
        window_size, stride, target_fps, input_size, num_classes, labels, device
    """
    ckpt_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            "Set ACL_MODEL_PATH env var or place the .pt file there."
        )

    device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    # Phase 6 saves under "model_state_dict"; some pipelines use "state_dict";
    # others save the raw state_dict directly.
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        ckpt_meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    elif isinstance(ckpt, dict) and "lstm.weight_ih_l0" in ckpt:
        sd = ckpt
        ckpt_meta = {}
    else:
        sd = ckpt
        ckpt_meta = {}

    # Strip "module." prefix from DataParallel checkpoints
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    dims = _infer_dims_from_state_dict(sd)
    logger.info("Building ACLRiskLSTM with inferred dims: %s", dims)

    model = ACLRiskLSTM(**dims)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        logger.warning("Missing keys: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys: %s", unexpected)

    model.to(device).eval()
    model._device = device  # type: ignore[attr-defined]

    # Resolve labels: prefer trained label_map ({"Low":0,...}) over defaults.
    label_map = ckpt_meta.get("label_map")
    if isinstance(label_map, dict) and label_map:
        labels_sorted = sorted(label_map.items(), key=lambda kv: kv[1])
        labels = [k for k, _ in labels_sorted][: dims["num_classes"]]
    else:
        labels = list(ckpt_meta.get("labels", LABELS))[: dims["num_classes"]]

    meta = {
        "window_size":  int(ckpt_meta.get("window_size", DEFAULT_WINDOW_SIZE)),
        "stride":       int(ckpt_meta.get("stride", DEFAULT_STRIDE)),
        "target_fps":   float(ckpt_meta.get("target_fps", DEFAULT_FPS)),
        "input_size":   dims["input_size"],
        "num_classes":  dims["num_classes"],
        "labels":       labels,
        "feature_cols": list(ckpt_meta.get("feature_cols", [
            "l_knee_angle", "r_knee_angle", "l_hip_angle", "r_hip_angle",
            "trunk_lean",   "asymmetry",    "l_knee_vel", "r_knee_vel",
        ])),
        "architecture": ckpt_meta.get(
            "architecture", "BiLSTM_AttentionPool_2layer_hidden128"
        ),
        "device":       str(device),
        "checkpoint":   str(ckpt_path),
    }
    # Alias for callers that expect "feature_schema" instead of "feature_cols".
    meta["feature_schema"] = meta["feature_cols"]
    logger.info("Model loaded: %s", meta)
    return model, meta


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_input(feature_matrix: np.ndarray, fps: float, meta: dict) -> None:
    """
    Validate that the extracted feature matrix is usable for inference.
    Raises ValueError with an actionable message on failure.
    """
    if feature_matrix is None or not isinstance(feature_matrix, np.ndarray):
        raise ValueError("feature_matrix must be a numpy array")

    if feature_matrix.ndim != 2:
        raise ValueError(
            f"feature_matrix must be 2D (frames, features); got shape "
            f"{feature_matrix.shape}"
        )

    n_frames, n_feat = feature_matrix.shape
    expected_feat = meta["input_size"]
    if n_feat != expected_feat:
        raise ValueError(
            f"Feature dimension mismatch: got {n_feat}, model expects "
            f"{expected_feat}. Re-run feature engineering for this checkpoint."
        )

    win = meta["window_size"]
    if n_frames < win:
        raise ValueError(
            f"Clip too short: {n_frames} frames < window size {win}. "
            f"Need at least {win / max(fps, 1):.1f}s of usable footage."
        )

    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")

    if not np.isfinite(feature_matrix[~np.isnan(feature_matrix)]).all():
        raise ValueError("feature_matrix contains non-finite values (inf)")


# --------------------------------------------------------------------------- #
# Window inference
# --------------------------------------------------------------------------- #
def _make_windows(
    feature_matrix: np.ndarray, window_size: int, stride: int
) -> List[Tuple[int, int, np.ndarray]]:
    """Return list of (start_frame, end_frame_inclusive, window_array)."""
    n = feature_matrix.shape[0]
    out = []
    start = 0
    while start + window_size <= n:
        end = start + window_size  # exclusive
        out.append((start, end - 1, feature_matrix[start:end]))
        start += stride
    return out


@torch.no_grad()
def run_inference(
    feature_matrix: np.ndarray,
    model: ACLRiskLSTM,
    meta: dict,
    output_csv_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    batch_size: int = 32,
) -> List[dict]:
    """
    Slide a window across `feature_matrix`, run BiLSTM inference per window,
    and return a list of per-window prediction dicts.

    Each dict contains:
        window_index, start_frame, end_frame, label,
        P_low, P_medium, P_high, confidence
    """
    window_size = meta["window_size"]
    stride      = meta["stride"]
    labels      = meta.get("labels", LABELS)

    windows = _make_windows(feature_matrix, window_size, stride)
    if not windows:
        raise ValueError("No complete windows could be extracted from the clip.")

    device = getattr(model, "_device", next(model.parameters()).device)

    # Replace NaNs with column means (per window) so model never sees NaN.
    def _fill(arr: np.ndarray) -> np.ndarray:
        if not np.isnan(arr).any():
            return arr
        col_mean = np.nanmean(arr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(arr))
        out = arr.copy()
        out[idx] = np.take(col_mean, idx[1])
        return out

    predictions: List[dict] = []
    n_windows = len(windows)

    for batch_start in range(0, n_windows, batch_size):
        batch = windows[batch_start: batch_start + batch_size]
        x = np.stack([_fill(w[2]).astype(np.float32) for w in batch])
        x_t = torch.from_numpy(x).to(device)
        logits = model(x_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # (B, C)

        for i, (start_f, end_f, _) in enumerate(batch):
            p = probs[i]
            cls = int(np.argmax(p))
            label = labels[cls] if cls < len(labels) else str(cls)

            # Pad probs to 3 classes for the schema (Low/Medium/High)
            p3 = np.zeros(3, dtype=np.float32)
            p3[: len(p)] = p[:3]

            predictions.append({
                "window_index": batch_start + i,
                "start_frame":  int(start_f),
                "end_frame":    int(end_f),
                "label":        label,
                "P_low":        float(p3[0]),
                "P_medium":     float(p3[1]),
                "P_high":       float(p3[2]),
                "confidence":   float(p.max()),
            })

        if progress_callback is not None:
            done = min(batch_start + len(batch), n_windows)
            pct = 52 + int(34 * done / n_windows)  # 52→86 in pipeline budget
            progress_callback(pct, f"Inference: window {done}/{n_windows}")

    if output_csv_path:
        _write_csv(predictions, output_csv_path)

    return predictions


def _write_csv(predictions: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index", "start_frame", "end_frame", "label",
        "P_low", "P_medium", "P_high", "confidence",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in predictions:
            w.writerow(row)
