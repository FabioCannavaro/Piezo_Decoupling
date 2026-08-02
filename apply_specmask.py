"""
Bakes a strong SpecAugment-style mask into the AUGMENTED (is_aug==1) npy files
produced from a given augmentation_log.csv.

Run once per channel-count dataset (2ch, 3ch, ...). Because the mask
coordinates are derived from a hash of the filename (not from a fresh random
draw), running this against both the 2ch and 3ch npy folders for the same
augmented CSV set produces IDENTICAL time/freq mask placement in both -> the
two channel-count datasets stay directly comparable.

Only touches files flagged is_aug==1 in the log; originals are left untouched.
"""

import os
import csv
import hashlib
import argparse

import numpy as np


def seeded_rng(cls_name: str, filename: str) -> np.random.RandomState:
    key = f"{cls_name}/{filename}".encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.RandomState(seed)


def strong_spec_mask(x: np.ndarray, rng: np.random.RandomState,
                     time_frac=(0.08, 0.25), freq_frac=(0.08, 0.25), p=0.8) -> np.ndarray:
    if rng.rand() > p:
        return x
    y = x.copy()
    _, H, W = y.shape
    if rng.rand() < 0.8:
        w = int(rng.uniform(time_frac[0], time_frac[1]) * W)
        w = max(1, min(W, w))
        t0 = rng.randint(0, max(1, W - w + 1))
        y[:, :, t0:t0 + w] = 0
    if rng.rand() < 0.8:
        h = int(rng.uniform(freq_frac[0], freq_frac[1]) * H)
        h = max(1, min(H, h))
        f0 = rng.randint(0, max(1, H - h + 1))
        y[:, f0:f0 + h, :] = 0
    if rng.rand() < 0.4:
        g = 1.0 + rng.uniform(-0.12, 0.12)
        y = y * g
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_root", type=str, required=True, help="e.g. input_data_2ch_300")
    ap.add_argument("--log_csv", type=str, required=True, help="augmentation_log.csv from raw_augment_csv.py")
    args = ap.parse_args()

    with open(args.log_csv, "r") as f:
        rows = list(csv.DictReader(f))

    n_done = 0
    for row in rows:
        if row["is_aug"] != "1":
            continue
        cls, csv_name = row["class"], row["filename"]
        npy_name = os.path.splitext(csv_name)[0] + ".npy"
        path = os.path.join(args.npy_root, cls, npy_name)
        if not os.path.exists(path):
            print(f"[WARN] missing {path}, skipping")
            continue

        arr = np.load(path)
        rng = seeded_rng(cls, csv_name)  # keyed on the ORIGINAL csv filename -> shared across 2ch/3ch
        arr = strong_spec_mask(arr, rng)
        np.save(path, arr)
        n_done += 1

    print(f"Applied spec-mask to {n_done} synthetic npy files under {args.npy_root}")


if __name__ == "__main__":
    main()
