"""
Leave-One-Day-Out (LODO) training for Piezo_Decoupling.

Reuses almost everything from train.py (model, optimizer/scheduler, epoch
loop, plotting) and only replaces the fold-construction logic:

- Folds are defined by the 'day' column in augmentation_log.csv (see
  assign_days.py + raw_augment_csv.py --day_assignment), NOT by a random
  StratifiedKFold split.
- For each held-out day D:
    * test set  = ORIGINAL (non-synthetic) samples whose day == D
    * train set = ORIGINAL + synthetic samples whose day != D
    * val set   = a stratified slice of ORIGINAL samples carved out of the
                  training days (synthetic samples never enter val/test,
                  same purity rule as train.py's holdout test)
- A synthetic sample's day always equals its source original's day (enforced
  upstream by raw_augment_csv.py), so held-out-day test data is never
  near-duplicated inside the training set.

⚠️ If the 'day' field in augmentation_log.csv comes from assign_days.py's
placeholder assignment, this LODO run is a PIPELINE VALIDATION ONLY — it does
NOT demonstrate real day/session generalization. Swap in a log with real
per-file collection dates before using these numbers in the manuscript.
"""

import os
import csv
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from spectrogram_dataset import SpectrogramDataset
from train import (
    nowstamp, load_aug_log, _is_aug_path, build_idx_to_class,
    compute_stats_from_paths, make_subset_from_indices, build_model,
    build_optimizer_and_scheduler, unfreeze_backbone_and_reset_opt,
    label_smoothing_for_epoch, epoch_loop, save_val_confmat_png, plot_fold_curves,
)


def load_day_lookup(root: str) -> Dict[str, int]:
    """path key 'class/filename.npy' -> day (int). Missing entries -> not in dict."""
    log_path = os.path.join(root, "augmentation_log.csv")
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"{log_path} not found. LODO needs augmentation_log.csv with a 'day' column "
            f"(copy it into the npy dataset root, see raw_augment_csv.py)."
        )
    lookup = {}
    with open(log_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("day"):
                continue
            key = f"{row['class']}/{os.path.splitext(row['filename'])[0]}.npy"
            lookup[key] = int(row["day"])
    if not lookup:
        raise ValueError(f"{log_path} has no 'day' values — was --day_assignment used in raw_augment_csv.py?")
    return lookup


def day_of_path(p: str, day_lookup: Dict[str, int]):
    key = "/".join(os.path.normpath(p).split(os.sep)[-2:])
    return day_lookup.get(key)


def train_lodo(root: str, batch_size: int, epochs: int, lr: float, freeze_epochs: int,
              val_ratio: float, out_root: str, seed: int, head_dropout: float,
              label_smoothing_max: float, noise_std: float, strong_specaugment: bool,
              mixup_alpha: float, use_mixup: bool):

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    import train as _train_mod
    _train_mod._AUG_LOOKUP = load_aug_log(root)
    day_lookup = load_day_lookup(root)
    print("⚠️  Using 'day' field from augmentation_log.csv for LODO splits — "
          "confirm this is a REAL date record, not the assign_days.py placeholder, "
          "before reporting these numbers.")

    ds_full = SpectrogramDataset(root, augment=False, stats=None, index_json=None)
    all_paths = ds_full.image_paths
    labels = ds_full.labels.clone()
    num_classes = int(labels.max()) + 1
    idx_to_class = build_idx_to_class(ds_full, labels)

    days = np.array([day_of_path(p, day_lookup) for p in all_paths])
    if any(d is None for d in days):
        n_missing = sum(1 for d in days if d is None)
        print(f"[WARN] {n_missing} samples have no day assignment and will be excluded from LODO entirely")
    valid_mask = np.array([d is not None for d in days])
    unique_days = sorted(set(int(d) for d in days[valid_mask]))

    dataset_name = os.path.basename(os.path.normpath(root))
    run_dir = out_root or os.path.join("runs", f"{nowstamp()}_convnext_tiny_{dataset_name}_LODO")
    print("📁 LODO results will be saved under:", run_dir)
    os.makedirs(run_dir, exist_ok=True)

    is_orig = np.array([not _is_aug_path(p) for p in all_paths])

    meta = {
        "root": root, "mode": "leave_one_day_out", "days": unique_days, "seed": seed,
        "val_ratio": val_ratio, "num_classes": int(num_classes), "device": str(device),
        "regularization": {
            "head_dropout": head_dropout, "label_smoothing_max": label_smoothing_max,
            "noise_std": noise_std, "strong_specaugment": strong_specaugment,
            "mixup_alpha": mixup_alpha, "use_mixup": use_mixup, "freeze_epochs": freeze_epochs,
        },
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    rows = []
    for held_out_day in unique_days:
        print(f"\n========== LODO: holding out day {held_out_day} ==========")

        test_day_mask = valid_mask & (days == held_out_day)
        train_days_mask = valid_mask & (days != held_out_day)

        test_idx = np.where(test_day_mask & is_orig)[0]           # test = ORIGINAL only, held-out day
        train_pool_idx = np.where(train_days_mask)[0]             # train pool = orig+aug, all other days

        pool_is_orig = is_orig[train_pool_idx]
        orig_pool = train_pool_idx[pool_is_orig]
        aug_pool = train_pool_idx[~pool_is_orig]

        y_orig_pool = labels[orig_pool].numpy()
        # carve val out of ORIGINAL training-day samples only (never synthetic, never the held-out day)
        if len(set(y_orig_pool.tolist())) > 1 and len(orig_pool) > 10:
            tr_loc, va_loc = train_test_split(
                np.arange(len(orig_pool)), test_size=val_ratio, random_state=seed,
                stratify=y_orig_pool,
            )
        else:
            split = max(1, int(len(orig_pool) * (1 - val_ratio)))
            tr_loc, va_loc = np.arange(split), np.arange(split, len(orig_pool))

        train_idx = np.concatenate([orig_pool[tr_loc], aug_pool]).astype(int)
        val_idx = orig_pool[va_loc].astype(int)

        if len(test_idx) == 0:
            print(f"[day {held_out_day}] no ORIGINAL samples on this day, skipping")
            continue

        train_paths = [all_paths[i] for i in train_idx]
        stats = compute_stats_from_paths(train_paths)

        train_ds = make_subset_from_indices(root, stats, train_idx, augment=True,
                                            noise_std=noise_std, strong_specaugment=strong_specaugment)
        val_ds = make_subset_from_indices(root, stats, val_idx, augment=False,
                                          noise_std=0.0, strong_specaugment=False)

        cls_counts = torch.bincount(train_ds.labels, minlength=num_classes)
        class_weights = (cls_counts.sum() / (cls_counts + 1e-6)).float()
        sample_weights = class_weights[train_ds.labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)

        input_channels = len(stats["mean"])
        model = build_model(num_classes, in_chans=input_channels, head_dropout=head_dropout).to(device)
        optimizer, scheduler, warm_freeze = build_optimizer_and_scheduler(
            model, base_lr=lr, epochs=epochs, freeze_epochs=freeze_epochs, weight_decay=1e-4
        )

        day_dir = os.path.join(run_dir, f"day{held_out_day}")
        os.makedirs(day_dir, exist_ok=True)

        epoch_csv = os.path.join(day_dir, "epoch_metrics.csv")
        with open(epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "label_smoothing"])

        best_val_acc = -1.0
        history = []

        for epoch in range(epochs):
            if epoch == warm_freeze:
                optimizer, scheduler = unfreeze_backbone_and_reset_opt(
                    model, current_epoch=epoch, epochs=epochs, base_lr=lr, weight_decay=1e-4
                )

            ls = label_smoothing_for_epoch(epoch, epochs, ls_max=label_smoothing_max, decay_until=0.8)
            train_crit = nn.CrossEntropyLoss(label_smoothing=0.05)
            val_crit = nn.CrossEntropyLoss()

            tr_loss, tr_acc = epoch_loop(model, train_loader, train_crit, device, train_mode=True,
                                         optimizer=optimizer, max_grad=5.0, use_mixup=use_mixup, mixup_alpha=mixup_alpha)
            va_loss, va_acc = epoch_loop(model, val_loader, val_crit, device, train_mode=False,
                                         optimizer=None, use_mixup=False, mixup_alpha=0.0)

            scheduler.step()
            cur_lr = optimizer.param_groups[0]["lr"]

            print(f"[Day {held_out_day} held out] [Epoch {epoch+1}/{epochs}] "
                  f"Train {tr_loss:.4f}/{tr_acc:.3f} | Val {va_loss:.4f}/{va_acc:.3f} | LR {cur_lr:.2e} | LS {ls:.3f}")

            with open(epoch_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch + 1, tr_loss, tr_acc, va_loss, va_acc, cur_lr, ls])
            history.append({"epoch": epoch + 1, "train_loss": float(tr_loss), "train_acc": float(tr_acc),
                            "val_loss": float(va_loss), "val_acc": float(va_acc), "lr": float(cur_lr),
                            "label_smoothing": float(ls)})

            if (va_acc > best_val_acc) and torch.isfinite(torch.tensor(va_loss)):
                best_val_acc = va_acc
                torch.save({
                    "model_state": model.state_dict(), "best_val_acc": float(best_val_acc),
                    "stats": stats, "class_to_idx": getattr(train_ds, "class_to_idx", None),
                }, os.path.join(day_dir, "best_model.pth"))

        pd.DataFrame(history).to_csv(os.path.join(day_dir, "training_metrics.csv"), index=False)
        try:
            plot_fold_curves(epoch_csv, os.path.join(day_dir, "training_curves_acc.png"),
                             os.path.join(day_dir, "training_curves_loss.png"))
        except Exception as e:
            print("[WARN] plot_fold_curves failed:", e)

        # ---- evaluate on the held-out day (ORIGINAL samples only) ----
        test_stats = compute_stats_from_paths(train_paths)  # normalize with train-side stats, never test stats
        test_ds = make_subset_from_indices(root, test_stats, test_idx, augment=False,
                                           noise_std=0.0, strong_specaugment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        ckpt = torch.load(os.path.join(day_dir, "best_model.pth"), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                pred = model(x).argmax(1)
                ys.append(y.numpy()); ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        rep = classification_report(y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0)

        np.save(os.path.join(day_dir, "test_confusion_matrix.npy"), cm)
        with open(os.path.join(day_dir, "test_classification_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        with open(os.path.join(day_dir, "test_acc.txt"), "w") as f:
            f.write(f"test_acc={acc:.6f}\n")

        print(f"[Day {held_out_day} held out] test_acc={acc:.4f} (n_test={len(test_idx)}, original-only)")
        rows.append({
            "held_out_day": held_out_day, "n_test_original": int(len(test_idx)),
            "best_val_acc": ckpt.get("best_val_acc"), "test_acc": acc,
            "test_macro_f1": rep.get("macro avg", {}).get("f1-score"),
        })

    df = pd.DataFrame(rows).sort_values("held_out_day")
    df.to_csv(os.path.join(run_dir, "lodo_summary.csv"), index=False)
    print("\n===== LODO summary =====")
    print(df.to_string(index=False))
    print("Saved to:", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Leave-One-Day-Out training for Sensor Spectrograms")
    ap.add_argument("--root", type=str, required=True, help="npy dataset root, must contain augmentation_log.csv with a 'day' column")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--out_root", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--head_dropout", type=float, default=0.3)
    ap.add_argument("--label_smoothing_max", type=float, default=0.1)
    ap.add_argument("--noise_std", type=float, default=0.03)
    ap.add_argument("--no_strong_specaug", action="store_true")
    ap.add_argument("--mixup_alpha", type=float, default=0.4)
    ap.add_argument("--no_mixup", action="store_true")
    args = ap.parse_args()

    train_lodo(
        root=args.root, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        freeze_epochs=args.freeze_epochs, val_ratio=args.val_ratio, out_root=args.out_root,
        seed=args.seed, head_dropout=args.head_dropout, label_smoothing_max=args.label_smoothing_max,
        noise_std=args.noise_std, strong_specaugment=(not args.no_strong_specaug),
        mixup_alpha=args.mixup_alpha, use_mixup=(not args.no_mixup),
    )
