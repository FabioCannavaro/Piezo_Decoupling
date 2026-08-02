"""
Raw CSV-domain augmentation for Piezo_Decoupling.

Takes original raw sensor CSVs (data_study/<class>/T{n}.csv, columns Time,W,X,Y)
and oversamples under-represented classes up to TARGET_PER_CLASS by augmenting
the RAW signal (time_shift, time_stretch, gain_jitter, noise) BEFORE the
CWT/spectrogram step in main.py.

Design choices (agreed in planning):
- Output filenames continue the existing sequential numbering (T{n+1}.csv, ...)
  with NO "__aug" marker in the name, so downstream code that only looks at the
  npy folder sees a clean, natural-looking dataset.
- A separate augmentation_log.csv records, for every file (original AND
  augmented), whether it is synthetic, which original it was derived from, and
  the exact augmentation parameters used. This log is copied alongside the
  generated npy datasets so train.py can consult it instead of guessing from
  filenames.
- One random draw (shift / stretch scale / gain / noise) per augmented sample
  is applied identically to every raw column present (Time is left untouched,
  W/X/Y all get the same shift & stretch & gain; noise is added per-channel
  since W (temperature) and X/Y (strain) live on very different scales).
- Because the same augmented CSV is later fed through main.py twice
  (include_temp=False -> 2ch, include_temp=True -> 3ch), the 2ch and 3ch
  datasets are guaranteed to originate from identical raw augmented signals.
"""

import os
import math
import random
import argparse
import shutil
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import resample

from main import detect_columns, TIME_CANDIDATES, TEMP_CANDIDATES  # reuse official column-detection logic

# --------------------- config ---------------------

TARGET_PER_CLASS = 300
PER_SOURCE_LIMIT = 8
SEED = 42

# --------------------- raw augmentation primitives ---------------------

def raw_time_shift(x: np.ndarray, k: int) -> np.ndarray:
    return np.roll(x, k)

def raw_time_stretch(x: np.ndarray, scale: float) -> np.ndarray:
    n = len(x)
    new_n = max(8, int(round(n * scale)))
    stretched = resample(x, new_n)
    back = resample(stretched, n)  # re-fit to original sample count / duration
    return back.astype(np.float64)

def raw_gain_jitter(x: np.ndarray, gain: float) -> np.ndarray:
    return x * gain

def raw_add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    sig_pow = float(np.mean(x ** 2) + 1e-8)
    snr = 10 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr
    n = np.random.normal(0.0, math.sqrt(max(noise_pow, 0.0)), size=x.shape)
    return x + n

def augment_row_set(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Apply ONE consistent draw of (shift, stretch, gain) across all `cols`,
    plus independent per-channel noise. Returns a new DataFrame (copy).
    Also returns the params used (attached as df.attrs for logging).
    """
    out = df.copy()
    n = len(df)

    do_shift = np.random.rand() < 0.8
    do_stretch = np.random.rand() < 0.6
    do_gain = np.random.rand() < 0.7
    do_noise = np.random.rand() < 0.7

    k = int(np.random.uniform(-0.15, 0.15) * n) if do_shift else 0
    scale = float(np.random.uniform(0.9, 1.1)) if do_stretch else 1.0
    gain = float(1.0 + np.random.uniform(-0.12, 0.12)) if do_gain else 1.0
    snr_db = float(np.random.uniform(18, 30)) if do_noise else None

    for c in cols:
        x = out[c].to_numpy(dtype=float)
        if do_shift:
            x = raw_time_shift(x, k)
        if do_stretch:
            x = raw_time_stretch(x, scale)
        if do_gain:
            x = raw_gain_jitter(x, gain)
        if do_noise:
            x = raw_add_noise(x, snr_db)
        out[c] = x

    out.attrs["aug_params"] = {
        "shift_samples": k if do_shift else 0,
        "stretch_scale": scale if do_stretch else 1.0,
        "gain": gain if do_gain else 1.0,
        "noise_snr_db": snr_db if do_noise else "",
    }
    return out

# --------------------- core ---------------------

def augment_class(in_dir: str, out_dir: str, target: int, per_source_limit: int,
                   log_rows: List[dict], cls_name: str,
                   day_map: Optional[Dict[str, int]] = None,
                   target_per_day: Optional[int] = None):
    """
    If day_map is None: oversample the whole class up to `target` (original behavior).
    If day_map is provided (filename -> day id): oversample WITHIN each day bucket
    up to `target_per_day`, so every synthetic sample's day == its source's day.
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_files = sorted(Path(in_dir).glob("T*.csv"),
                       key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0)
    n_orig = len(csv_files)

    # 1) copy originals unchanged, log them (with day, if known)
    for src in csv_files:
        dst = Path(out_dir) / src.name
        shutil.copy2(src, dst)
        day = day_map.get(src.name, "") if day_map else ""
        log_rows.append({
            "class": cls_name, "filename": src.name, "is_aug": 0,
            "source_original": "", "day": day,
            "shift_samples": "", "stretch_scale": "", "gain": "", "noise_snr_db": "",
        })

    next_idx = n_orig + 1

    def do_one_augmentation(candidates: List[Path], per_source_counter: Dict[str, int],
                            day_value):
        nonlocal next_idx
        avail = [p for p in candidates if per_source_counter.get(p.name, 0) < per_source_limit]
        if not avail:
            for p in candidates:
                per_source_counter[p.name] = 0
            avail = candidates[:]
        src = random.choice(avail)
        per_source_counter[src.name] = per_source_counter.get(src.name, 0) + 1

        df = pd.read_csv(src)
        _, s1, s2, tcol = detect_columns(df, include_temp=False)
        try:
            _, _, _, tcol_probe = detect_columns(df, include_temp=True)
            if tcol_probe:
                tcol = tcol_probe
        except ValueError:
            tcol = None

        aug_cols = [c for c in [s1, s2, tcol] if c is not None]
        aug_df = augment_row_set(df, aug_cols)

        out_name = f"T{next_idx}.csv"
        aug_df.to_csv(Path(out_dir) / out_name, index=False)

        p = aug_df.attrs["aug_params"]
        log_rows.append({
            "class": cls_name, "filename": out_name, "is_aug": 1,
            "source_original": src.name, "day": day_value,
            "shift_samples": p["shift_samples"], "stretch_scale": p["stretch_scale"],
            "gain": p["gain"], "noise_snr_db": p["noise_snr_db"],
        })
        next_idx += 1

    if day_map is None:
        need = max(0, target - n_orig)
        if need == 0:
            print(f"[{cls_name}] already has {n_orig} >= target {target}, no augmentation needed")
            return
        print(f"[{cls_name}] {n_orig} originals -> need {need} more (target {target})")
        counter: Dict[str, int] = {}
        for _ in range(need):
            do_one_augmentation(csv_files, counter, day_value="")
        return

    # --- day-grouped oversampling ---
    assert target_per_day is not None, "target_per_day required when day_map is given"
    by_day: Dict[int, List[Path]] = {}
    for src in csv_files:
        d = day_map.get(src.name)
        if d is None:
            print(f"[WARN] {cls_name}/{src.name} missing from day_assignment; skipped in day-grouping")
            continue
        by_day.setdefault(d, []).append(src)

    for day, files_in_day in sorted(by_day.items()):
        have = len(files_in_day)
        need = max(0, target_per_day - have)
        if need == 0:
            print(f"[{cls_name}][day {day}] already has {have} >= target {target_per_day}")
            continue
        print(f"[{cls_name}][day {day}] {have} originals -> need {need} more (target/day {target_per_day})")
        counter: Dict[str, int] = {}
        for _ in range(need):
            do_one_augmentation(files_in_day, counter, day_value=day)


def load_day_map(day_assignment_csv: str, cls_name: str) -> Dict[str, int]:
    m = {}
    with open(day_assignment_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["class"] == cls_name:
                m[row["filename"]] = int(row["day"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data_study")
    ap.add_argument("--output_dir", type=str, default="data_study_aug_300")
    ap.add_argument("--target_per_class", type=int, default=TARGET_PER_CLASS)
    ap.add_argument("--per_source_limit", type=int, default=PER_SOURCE_LIMIT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--day_assignment", type=str, default=None,
                    help="Optional day_assignment.csv from assign_days.py. When given, "
                         "oversampling happens within each (class, day) bucket instead of per-class.")
    ap.add_argument("--target_per_day", type=int, default=30,
                    help="Per-day target count, used only when --day_assignment is given.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    class_dirs = sorted([d for d in os.listdir(args.input_dir)
                         if os.path.isdir(os.path.join(args.input_dir, d))])

    log_rows: List[dict] = []
    for cls in class_dirs:
        day_map = load_day_map(args.day_assignment, cls) if args.day_assignment else None
        augment_class(
            in_dir=os.path.join(args.input_dir, cls),
            out_dir=os.path.join(args.output_dir, cls),
            target=args.target_per_class,
            per_source_limit=args.per_source_limit,
            log_rows=log_rows,
            cls_name=cls,
            day_map=day_map,
            target_per_day=args.target_per_day if day_map else None,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "augmentation_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "class", "filename", "is_aug", "source_original", "day",
            "shift_samples", "stretch_scale", "gain", "noise_snr_db",
        ])
        w.writeheader()
        w.writerows(log_rows)

    n_aug = sum(1 for r in log_rows if r["is_aug"] == 1)
    print(f"\nDone. {len(log_rows)} total files ({n_aug} synthetic) written to {args.output_dir}")
    print(f"Log: {log_path}")
    if args.day_assignment:
        print("⚠️  day field comes from a PLACEHOLDER assignment (see assign_days.py) — "
              "not a real collection-date record.")


if __name__ == "__main__":
    main()
