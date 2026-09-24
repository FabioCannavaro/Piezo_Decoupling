# realtime_server.py
#
# Final real-time inference server for the 3-channel ConvNeXtV2 model.
#
# Folder structure:
#   real_time/
#   ├─ main.py              <- main_paper_reproduction_strain_centered.py renamed to main.py
#   ├─ realtime_server.py   <- this file
#   └─ best_model.pth       <- one selected 3-channel 10-fold checkpoint
#
# Run:
#   uvicorn realtime_server:app --host 0.0.0.0 --port 8000
#
# Android -> POST /predict
# {
#   "ch1": [...],
#   "ch2": [...],
#   "temp": [...]
# }

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import timm

from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT:
# Use the exact preprocessing functions used to generate processed_centered_3ch.
# Rename main_paper_reproduction_strain_centered.py -> main.py
from main import (
    TARGET_FS,
    STRAIN_BAND,
    TEMP_LP,
    butter_bandpass,
    butter_lowpass,
    remove_strain_baseline,
    cwt_channel_morse,
)


# =========================================================
# Configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# You can override this with an environment variable:
#   set MODEL_CKPT=C:\...\foldX\best_model.pth
MODEL_CKPT = Path(
    os.environ.get("MODEL_CKPT", str(BASE_DIR / "best_model.pth"))
)

USE_GPU_IF_AVAILABLE = True
USE_AMP_ON_CUDA = True

# 6 s x 50 Hz = 300 samples in the paper dataset.
# We do not hard-fail when the length differs because the CWT function
# resizes the time axis to 256, but 300 samples is recommended.
RECOMMENDED_N_SAMPLES = 300


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(title="Realtime Multimodal Object Classification Server")


class WindowInput(BaseModel):
    ch1: List[float]
    ch2: List[float]
    temp: List[float]


# =========================================================
# Model
# =========================================================

def build_model(
    num_classes: int,
    in_chans: int = 3,
    head_dropout: float = 0.3,
) -> nn.Module:
    """
    Exact architecture used by the final 10-fold training code:
    ConvNeXtV2-Tiny FCMAE + custom classification head.
    """
    # pretrained=False is intentional here.
    # The checkpoint contains the full trained state_dict, so there is
    # no need to download/load ImageNet/FCMAE weights again at runtime.
    model = timm.create_model(
        "convnextv2_tiny.fcmae",
        pretrained=False,
        num_classes=0,
        in_chans=in_chans,
    )

    in_features = model.num_features

    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(in_features),
        nn.Dropout(p=head_dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_model_and_stats(
    ckpt_path: Path,
    device: torch.device,
):
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Place a selected 3-channel fold best_model.pth next to "
            f"realtime_server.py, or set MODEL_CKPT."
        )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = ["model_state", "stats", "class_to_idx"]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint is missing required key(s): {missing}"
        )

    stats = checkpoint["stats"]
    class_to_idx = checkpoint["class_to_idx"]

    if not isinstance(class_to_idx, dict) or len(class_to_idx) == 0:
        raise ValueError(
            "checkpoint['class_to_idx'] is empty or invalid."
        )

    if "mean" not in stats or "std" not in stats:
        raise KeyError("checkpoint['stats'] must contain 'mean' and 'std'.")

    in_chans = len(stats["mean"])
    if in_chans != 3:
        raise ValueError(
            f"This real-time server is configured for the final 3-channel model, "
            f"but checkpoint stats indicate {in_chans} channel(s)."
        )

    if len(stats["std"]) != in_chans:
        raise ValueError("stats['mean'] and stats['std'] have different lengths.")

    head_dropout = float(checkpoint.get("head_dropout", 0.3))
    num_classes = len(class_to_idx)

    model = build_model(
        num_classes=num_classes,
        in_chans=in_chans,
        head_dropout=head_dropout,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )
    model.eval()

    idx_to_class = {
        int(idx): str(cls)
        for cls, idx in class_to_idx.items()
    }

    # Basic consistency check.
    expected_indices = set(range(num_classes))
    if set(idx_to_class.keys()) != expected_indices:
        raise ValueError(
            "class_to_idx indices are not contiguous from 0 to num_classes-1: "
            f"{class_to_idx}"
        )

    metadata = {
        "best_val_acc": checkpoint.get("best_val_acc"),
        "fold": checkpoint.get("fold", checkpoint.get("test_fold")),
        "test_day": checkpoint.get("test_day"),
        "validation_day": checkpoint.get(
            "validation_day",
            checkpoint.get("val_block"),
        ),
        "head_dropout": head_dropout,
    }

    return model, stats, idx_to_class, metadata


# =========================================================
# Realtime preprocessing
# =========================================================

def process_window(
    ch1: np.ndarray,
    ch2: np.ndarray,
    temp: np.ndarray,
    fs: float = TARGET_FS,
) -> np.ndarray:
    """
    Match the final processed_centered_3ch preprocessing:

    strain 1/2:
        median baseline removal
        -> Butterworth band-pass 0.1-15 Hz
        -> Morse CWT 0.2-15 Hz

    temperature:
        Butterworth low-pass 1 Hz
        -> Morse CWT 0.05-15 Hz

    output:
        (3, 96, 256), float32
    """

    n1, n2, n3 = len(ch1), len(ch2), len(temp)

    if n1 == 0:
        raise ValueError("Received an empty sensor window.")

    if not (n1 == n2 == n3):
        raise ValueError(
            f"Length mismatch: ch1={n1}, ch2={n2}, temp={n3}"
        )

    ch1 = np.nan_to_num(
        np.asarray(ch1, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ch2 = np.nan_to_num(
        np.asarray(ch2, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    temp = np.nan_to_num(
        np.asarray(temp, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # ----- Strain channels -----
    ch1_centered = remove_strain_baseline(ch1)
    ch2_centered = remove_strain_baseline(ch2)

    ch1_f = butter_bandpass(
        ch1_centered,
        fs,
        *STRAIN_BAND,
    )
    ch2_f = butter_bandpass(
        ch2_centered,
        fs,
        *STRAIN_BAND,
    )

    s1_spec = cwt_channel_morse(
        ch1_f,
        fs,
    )
    s2_spec = cwt_channel_morse(
        ch2_f,
        fs,
    )

    # ----- Temperature channel -----
    # NOTE:
    # No detrend_poly here.
    # This matches main_paper_reproduction_strain_centered.py.
    temp_f = butter_lowpass(
        temp,
        fs,
        TEMP_LP,
    )

    temp_spec = cwt_channel_morse(
        temp_f,
        fs,
        freq_min=0.05,
    )

    arr = np.stack(
        [s1_spec, s2_spec, temp_spec],
        axis=0,
    ).astype(np.float32)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if arr.shape != (3, 96, 256):
        raise ValueError(
            f"Unexpected processed tensor shape: {arr.shape}; "
            f"expected (3, 96, 256)."
        )

    return arr


# =========================================================
# Device / checkpoint initialization
# =========================================================

if USE_GPU_IF_AVAILABLE and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

amp_enabled = (
    device.type == "cuda"
    and USE_AMP_ON_CUDA
)

print("=" * 60)
print("[RealtimeServer] Starting")
print(f"[RealtimeServer] Device      : {device}")
print(f"[RealtimeServer] AMP         : {amp_enabled}")
print(f"[RealtimeServer] Checkpoint  : {MODEL_CKPT}")

model, stats, idx_to_class, checkpoint_meta = load_model_and_stats(
    MODEL_CKPT,
    device,
)

num_channels = len(stats["mean"])

mean = torch.tensor(
    stats["mean"],
    dtype=torch.float32,
    device=device,
).view(1, num_channels, 1, 1)

std = torch.tensor(
    stats["std"],
    dtype=torch.float32,
    device=device,
).view(1, num_channels, 1, 1)

print(f"[RealtimeServer] Classes     : {idx_to_class}")
print(f"[RealtimeServer] Mean        : {stats['mean']}")
print(f"[RealtimeServer] Std         : {stats['std']}")
print(f"[RealtimeServer] Metadata    : {checkpoint_meta}")
print("=" * 60)


# =========================================================
# Endpoints
# =========================================================

@app.get("/ping")
def ping():
    return {
        "status": "ok",
        "device": str(device),
        "amp": amp_enabled,
        "model": "convnextv2_tiny.fcmae",
        "checkpoint": MODEL_CKPT.name,
        "classes": idx_to_class,
        "recommended_n_samples": RECOMMENDED_N_SAMPLES,
    }


@app.post("/predict")
def predict(window: WindowInput):
    """
    Android raw 3-channel window
      -> centered strain preprocessing
      -> Morse CWT
      -> checkpoint training-stat normalization
      -> ConvNeXtV2 inference
      -> JSON result
    """

    total_start = time.perf_counter()

    try:
        ch1 = np.asarray(window.ch1, dtype=np.float32)
        ch2 = np.asarray(window.ch2, dtype=np.float32)
        temp = np.asarray(window.temp, dtype=np.float32)

        n_samples = len(ch1)

        # -------------------------
        # Preprocessing
        # -------------------------
        prep_start = time.perf_counter()

        arr = process_window(
            ch1=ch1,
            ch2=ch2,
            temp=temp,
            fs=TARGET_FS,
        )

        prep_ms = (
            time.perf_counter() - prep_start
        ) * 1000.0

        # -------------------------
        # Training-stat z-score
        # -------------------------
        x = torch.from_numpy(arr).unsqueeze(0).to(
            device,
            non_blocking=True,
        )

        x = (x - mean) / (std + 1e-8)

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # -------------------------
        # Inference timing
        # -------------------------
        if device.type == "cuda":
            torch.cuda.synchronize()

        infer_start = time.perf_counter()

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(x)

            probs = torch.softmax(
                logits.float(),
                dim=1,
            )[0]

        if device.type == "cuda":
            torch.cuda.synchronize()

        inference_ms = (
            time.perf_counter() - infer_start
        ) * 1000.0

        pred_idx = int(
            torch.argmax(probs).item()
        )
        pred_label = idx_to_class[pred_idx]
        confidence = float(
            probs[pred_idx].item()
        )

        prob_list = (
            probs.detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        # Sorted top-3 predictions for debugging/demo display.
        k = min(3, len(prob_list))
        top_values, top_indices = torch.topk(
            probs,
            k=k,
        )

        top_k = [
            {
                "class_idx": int(idx.item()),
                "class_label": idx_to_class[int(idx.item())],
                "probability": float(val.item()),
            }
            for val, idx in zip(top_values, top_indices)
        ]

        total_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        print(
            f"[Predict] n={n_samples} | "
            f"{pred_label} ({confidence * 100:.2f}%) | "
            f"prep={prep_ms:.1f} ms | "
            f"infer={inference_ms:.1f} ms | "
            f"total={total_ms:.1f} ms"
        )

        # Existing fields are retained so the Android app can keep using
        # class_idx / class_label / probs / idx_to_class.
        return {
            "class_idx": pred_idx,
            "class_label": pred_label,
            "confidence": confidence,
            "confidence_percent": confidence * 100.0,
            "probs": prob_list,
            "idx_to_class": idx_to_class,
            "top_k": top_k,
            "n_samples": n_samples,
            "recommended_n_samples": RECOMMENDED_N_SAMPLES,
            "preprocessing_ms": prep_ms,
            "inference_ms": inference_ms,
            "total_server_ms": total_ms,
            "device": str(device),
        }

    except Exception as e:
        print(f"[Predict][ERROR] {type(e).__name__}: {e}")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
        }
