"""
PLACEHOLDER day assignment for pipeline validation.

IMPORTANT: This does NOT recover real collection dates. The original
filenames were bulk-renamed to T{n}.csv on 2025-08-18 (confirmed), so no
reliable per-file date survives in data_study/. This script exists ONLY to
let us build and sanity-check the Leave-One-Day-Out (LODO) training pipeline
before real per-day data is available. Every row in the output log is tagged
note="PLACEHOLDER_ARBITRARY_DAY" so it can never be silently mistaken for a
real experimental record later.

Do NOT report results from this placeholder split as evidence of the
device/session/day generalization the manuscript needs -- swap in
day_assignment.csv derived from real timestamps once that data exists, then
rerun steps 2-4 unchanged.
"""

import os
import csv
import random
import argparse
from pathlib import Path
from typing import List


def assign_days_for_class(csv_files: List[Path], n_days: int, rng: random.Random) -> dict:
    files = csv_files[:]
    rng.shuffle(files)  # break any residual ordering so the placeholder split has no hidden structure
    assignment = {}
    for i, f in enumerate(files):
        assignment[f.name] = (i % n_days) + 1  # day ids 1..n_days, round-robin over the shuffled list
    return assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="data_study")
    ap.add_argument("--output_csv", type=str, default="day_assignment.csv")
    ap.add_argument("--n_days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    class_dirs = sorted([d for d in os.listdir(args.input_dir)
                         if os.path.isdir(os.path.join(args.input_dir, d))])

    rows = []
    for cls in class_dirs:
        cdir = Path(args.input_dir) / cls
        csv_files = sorted(cdir.glob("*.csv"))
        assignment = assign_days_for_class(csv_files, args.n_days, rng)
        counts = {}
        for name, day in assignment.items():
            rows.append({
                "class": cls, "filename": name, "day": day,
                "note": "PLACEHOLDER_ARBITRARY_DAY",
            })
            counts[day] = counts.get(day, 0) + 1
        print(f"[{cls}] {len(csv_files)} files -> day counts: "
              f"{ {d: counts.get(d, 0) for d in range(1, args.n_days + 1)} }")

    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "filename", "day", "note"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n⚠️  PLACEHOLDER day assignment written to {args.output_csv}")
    print("    This is NOT a real collection-date record. Pipeline validation only.")


if __name__ == "__main__":
    main()
