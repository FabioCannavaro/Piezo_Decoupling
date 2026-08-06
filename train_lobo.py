"""
Five-fold grouped day-level cross-validation.

Data structure for each object class:
- 30 actual measurement days
- 10 original measurements per day
- 300 original measurements per class (T1 ... T300)

First, the 30 days are assigned to 10 temporally distributed groups:
- group 1  = days 1, 11, 21
- group 2  = days 2, 12, 22
- ...
- group 10 = days 10, 20, 30

Then, two distributed groups are combined into each test fold:
- fold 1 = groups 1 and 6
- fold 2 = groups 2 and 7
- fold 3 = groups 3 and 8
- fold 4 = groups 4 and 9
- fold 5 = groups 5 and 10

Validation groups:
- fold 1 -> group 2
- fold 2 -> group 3
- fold 3 -> group 4
- fold 4 -> group 5
- fold 5 -> group 1

Within every fold, training, validation, and test days are mutually exclusive.
Use --folds 1 4 for the fold-1/fold-4 pilot run.
Omit --folds for the final complete five-fold run.
"""

import argparse
import csv
import json
import os
import re
import random
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from spectrogram_dataset import SpectrogramDataset
from train import (
    nowstamp,
    build_idx_to_class,
    compute_stats_from_paths,
    make_subset_from_indices,
    build_model,
    build_optimizer_and_scheduler,
    unfreeze_backbone_and_reset_opt,
    epoch_loop,
    plot_fold_curves,
)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def path_key(path: str) -> str:
    """Return class/filename.npy for a dataset path."""
    return "/".join(os.path.normpath(path).split(os.sep)[-2:])


def parse_trial_number(path_or_filename: str) -> Optional[int]:
    """Extract trial number from T<number>.npy or T<number>.csv."""
    m = re.search(r"(?:^|[/\\])T(\d+)\.(?:npy|csv)$", path_or_filename, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def original_day_and_block(
    trial_number: int,
    samples_per_day: int,
    expected_blocks: int,
) -> Tuple[int, int]:
    """Map each trial to its actual day and temporally distributed group."""
    day = (trial_number - 1) // samples_per_day + 1
    block = (day - 1) % expected_blocks + 1
    return day, block


def load_aug_provenance(
    root: str,
    expected_blocks: int,
    expected_days: int,
) -> Dict[str, dict]:
    """
    Load provenance for synthetic samples.

    Preferred columns in augmentation_log.csv:
      class, filename, is_aug, day, block

    - day must mean the actual measurement day (1..30), not a pre-grouped block.
    - block may be supplied directly. If absent, it is derived from actual day.
    """
    log_path = os.path.join(root, "augmentation_log.csv")
    if not os.path.exists(log_path):
        return {}

    rows: Dict[str, dict] = {}
    with open(log_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("class") or not row.get("filename"):
                continue

            key = f"{row['class']}/{os.path.splitext(row['filename'])[0]}.npy"
            is_aug = str(row.get("is_aug", "0")).strip() == "1"

            day = None
            block = None

            if str(row.get("day", "")).strip():
                day = int(row["day"])
                if not 1 <= day <= expected_days:
                    raise ValueError(
                        f"Invalid actual day={day} for {key}. Expected 1..{expected_days}. "
                        "The day column must contain actual measurement days, not 1..10 grouped blocks."
                    )

            derived_block = None
            if day is not None:
                derived_block = (day - 1) % expected_blocks + 1

            if str(row.get("block", "")).strip():
                block = int(row["block"])
                if not 1 <= block <= expected_blocks:
                    raise ValueError(
                        f"Invalid block={block} for {key}. Expected 1..{expected_blocks}."
                    )
                if derived_block is not None and block != derived_block:
                    raise ValueError(
                        f"Block mismatch for {key}: log block={block}, "
                        f"distributed block={derived_block}, day={day}."
                    )
            elif derived_block is not None:
                block = derived_block

            rows[key] = {"is_aug": is_aug, "day": day, "block": block}

    return rows


def build_sample_metadata(
    all_paths,
    root: str,
    original_trials_per_class: int,
    samples_per_day: int,
    expected_blocks: int,
    expected_days: int,
):
    """
    Build original/day/block metadata.

    Original identity is anchored to the raw-data naming convention T1..T300.
    This avoids losing original samples because of an incorrect is_aug flag.
    Synthetic files require valid provenance in augmentation_log.csv.
    """
    aug_provenance = load_aug_provenance(root, expected_blocks, expected_days)

    n = len(all_paths)
    is_original = np.zeros(n, dtype=bool)
    days = np.full(n, -1, dtype=int)
    blocks = np.full(n, -1, dtype=int)

    missing_synthetic_provenance = []

    for i, p in enumerate(all_paths):
        key = path_key(p)
        trial = parse_trial_number(p)

        # T1..T300 are the 300 original measurements in each class.
        if trial is not None and 1 <= trial <= original_trials_per_class:
            day, block = original_day_and_block(trial, samples_per_day, expected_blocks)
            is_original[i] = True
            days[i] = day
            blocks[i] = block
            continue

        # Anything outside the original trial range is treated as synthetic.
        prov = aug_provenance.get(key)
        if prov is None or prov.get("block") is None:
            missing_synthetic_provenance.append(key)
            continue

        is_original[i] = False
        if prov.get("day") is not None:
            days[i] = int(prov["day"])
        blocks[i] = int(prov["block"])

    return is_original, days, blocks, missing_synthetic_provenance


def validate_original_counts(
    all_paths,
    labels,
    is_original,
    blocks,
    idx_to_class,
    expected_blocks: int,
    expected_per_class_per_block: int,
    run_dir: str,
):
    """Require exactly 30 original samples per class in every distributed 3-day group."""
    records = []
    errors = []
    num_classes = int(labels.max()) + 1

    for block in range(1, expected_blocks + 1):
        for cls in range(num_classes):
            count = int(np.sum(is_original & (blocks == block) & (labels.numpy() == cls)))
            records.append(
                {
                    "block": block,
                    "class_index": cls,
                    "class_name": idx_to_class.get(cls, str(cls)),
                    "n_original": count,
                }
            )
            if count != expected_per_class_per_block:
                errors.append(
                    f"block {block}, class {idx_to_class.get(cls, cls)}: "
                    f"expected {expected_per_class_per_block}, found {count}"
                )

    count_path = os.path.join(run_dir, "original_count_by_block_and_class.csv")
    pd.DataFrame(records).to_csv(count_path, index=False)

    expected_total = expected_blocks * expected_per_class_per_block * num_classes
    actual_total = int(is_original.sum())
    if actual_total != expected_total:
        errors.append(f"total originals: expected {expected_total}, found {actual_total}")

    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(
            "Original sample-count validation failed. Training was stopped before model fitting.\n"
            f"Count table: {count_path}\n{preview}"
        )


def save_confusion_matrix(cm, class_names, out_png, normalize=True, title="Confusion matrix"):
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title + (" (row normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = float(mat.max()) / 2 if mat.size else 0
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            text = f"{mat[r, c] * 100:.1f}%" if normalize else str(int(mat[r, c]))
            ax.text(c, r, text, ha="center", va="center", fontsize=8,
                    color="white" if mat[r, c] > threshold else "black")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_mean_curves(histories, run_dir):
    """Aggregate the 10 fold training curves by epoch and save mean ± SD plots."""
    all_hist = pd.concat(histories, ignore_index=True)
    all_hist.to_csv(os.path.join(run_dir, "all_folds_epoch_metrics.csv"), index=False)

    summary = (
        all_hist.groupby("epoch")
        .agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_sd=("train_loss", "std"),
            val_loss_mean=("val_loss", "mean"),
            val_loss_sd=("val_loss", "std"),
            train_acc_mean=("train_acc", "mean"),
            train_acc_sd=("train_acc", "std"),
            val_acc_mean=("val_acc", "mean"),
            val_acc_sd=("val_acc", "std"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(run_dir, "mean_training_curves.csv"), index=False)
    summary.to_csv(os.path.join(run_dir, "all_folds_epoch_mean_sd.csv"), index=False)

    x = summary["epoch"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label in [("train_loss", "Training loss"), ("val_loss", "Validation loss")]:
        mean = summary[f"{prefix}_mean"].to_numpy()
        sd = summary[f"{prefix}_sd"].fillna(0).to_numpy()
        line = ax.plot(x, mean, label=label)[0]
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mean loss across selected grouped folds")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "mean_loss_10fold.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label in [("train_acc", "Training accuracy"), ("val_acc", "Validation accuracy")]:
        mean = summary[f"{prefix}_mean"].to_numpy()
        sd = summary[f"{prefix}_sd"].fillna(0).to_numpy()
        line = ax.plot(x, mean, label=label)[0]
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.2, color=line.get_color())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean accuracy across selected grouped folds")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "mean_accuracy_10fold.png"), dpi=300)
    plt.close(fig)


def train_lobo(
    root: str,
    batch_size: int,
    epochs: int,
    lr: float,
    freeze_epochs: int,
    out_root: Optional[str],
    seed: int,
    head_dropout: float,
    noise_std: float,
    strong_specaugment: bool,
    mixup_alpha: float,
    use_mixup: bool,
    label_smoothing: float,
    include_synthetic: bool,
    samples_per_day: int,
    days_per_block: int,
    expected_days: int,
    expected_blocks: int,
    original_trials_per_class: int,
    folds_to_run,
    eval_only: bool = False,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if original_trials_per_class != samples_per_day * expected_days:
        raise ValueError(
            "original_trials_per_class must equal samples_per_day × expected_days: "
            f"{original_trials_per_class} != {samples_per_day} × {expected_days}"
        )
    if expected_days != days_per_block * expected_blocks:
        raise ValueError(
            "expected_days must equal days_per_block × expected_blocks: "
            f"{expected_days} != {days_per_block} × {expected_blocks}"
        )

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다. "
            "CPU용 PyTorch가 설치되었거나 NVIDIA 드라이버를 인식하지 못하고 있습니다."
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print(f"Using device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    amp_enabled = device.type == "cuda"

    print(f"AMP FP16 enabled: {amp_enabled}")
    print("TF32 enabled: True")

    ds_full = SpectrogramDataset(root, augment=False, stats=None, index_json=None)
    all_paths = ds_full.image_paths
    labels = ds_full.labels.clone()
    num_classes = int(labels.max()) + 1
    idx_to_class = build_idx_to_class(ds_full, labels)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    dataset_name = os.path.basename(os.path.normpath(root))
    if eval_only:
        if out_root is None:
            raise ValueError(
                "--eval_only 사용 시 기존 실행 결과 폴더를 --out_root로 지정해야 합니다."
            )
        run_dir = os.path.abspath(out_root)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"기존 결과 폴더를 찾을 수 없습니다: {run_dir}")
        print("Evaluation-only results folder:", run_dir)
    else:
        run_dir = out_root or os.path.join(
            "runs", f"{nowstamp()}_convnextv2_tiny_{dataset_name}_LOBO"
        )
        os.makedirs(run_dir, exist_ok=True)
        print("LOBO results:", run_dir)

    is_original, days, blocks, missing_synth = build_sample_metadata(
        all_paths=all_paths,
        root=root,
        original_trials_per_class=original_trials_per_class,
        samples_per_day=samples_per_day,
        expected_blocks=expected_blocks,
        expected_days=expected_days,
    )

    mapping_rows = []
    for day in range(1, expected_days + 1):
        assigned_group = (day - 1) % expected_blocks + 1
        mapping_rows.append(
            {
                "actual_day": day,
                "assigned_group": assigned_group,
                "first_trial": (day - 1) * samples_per_day + 1,
                "last_trial": day * samples_per_day,
            }
        )
    pd.DataFrame(mapping_rows).to_csv(
        os.path.join(run_dir, "day_to_distributed_group_mapping.csv"),
        index=False,
    )

    expected_per_class_per_block = samples_per_day * days_per_block
    validate_original_counts(
        all_paths=all_paths,
        labels=labels,
        is_original=is_original,
        blocks=blocks,
        idx_to_class=idx_to_class,
        expected_blocks=expected_blocks,
        expected_per_class_per_block=expected_per_class_per_block,
        run_dir=run_dir,
    )

    if include_synthetic and missing_synth:
        preview = "\n".join(missing_synth[:20])
        raise ValueError(
            f"{len(missing_synth)} synthetic samples lack valid day/block provenance. "
            "Cannot include them without leakage risk. Examples:\n" + preview
        )

    meta = {
        "root": root,
        "mode": "five-fold grouped temporally distributed day-level cross-validation",
        "actual_days": expected_days,
        "samples_per_day_per_class": samples_per_day,
        "days_per_block": days_per_block,
        "blocks": expected_blocks,
        "validation_rule": "fold-specific independent three-day validation group",
        "include_synthetic_training_only": include_synthetic,
        "num_classes": num_classes,
        "seed": seed,
        "device": str(device),
        "folds_to_run": list(folds_to_run),

        "test_fold_groups": {
            fold : [fold]
            for fold in range(1,11)
        },
        "validation_group_by_fold": {
            fold: (fold%10) + 1
            for fold in range(1,11)
        },
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "freeze_epochs": freeze_epochs,
            "head_dropout": head_dropout,
            "noise_std": noise_std,
            "strong_specaugment": strong_specaugment,
            "mixup_alpha": mixup_alpha,
            "use_mixup": use_mixup,
            "label_smoothing": label_smoothing,
            "weight_decay": 1e-4,
            "gradient_value_clip": 5.0,
            "mixed_precision": "AMP FP16",
            "tf32": True,
            "num_workers": 8,
            "persistent_workers": True,
            "prefetch_factor": 4,
        },
    }
    if not eval_only:
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    summary_rows = []
    all_true = []
    all_pred = []
    histories = []

    ten_fold_test_blocks = {
        fold: [fold]
        for fold in range(1,11)
    }
    ten_fold_val_blocks = {
        fold: (fold%10)+1
        for fold in range(1,11)
    }

    five_fold_test_blocks = {
        1: [1, 6],
        2: [2, 7],
        3: [3, 8],
        4: [4, 9],
        5: [5, 10],
    }
    five_fold_val_block = {
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 1,
    }

    folds_to_run = list(folds_to_run)
    invalid_folds = sorted(set(folds_to_run) - set(ten_fold_test_blocks))
    if invalid_folds:
        raise ValueError(f"Invalid folds: {invalid_folds}. Allowed folds are 1..10.")

    for test_fold in folds_to_run:
        test_blocks = ten_fold_test_blocks[test_fold]
        val_block = ten_fold_val_blocks[test_fold]
        excluded_blocks = set(test_blocks) | {val_block}
        train_blocks = [
            b for b in range(1, expected_blocks + 1)
            if b not in excluded_blocks
        ]

        print(f"\n========== Fold {test_fold}/10 ==========")
        print(
            f"train groups={train_blocks}, "
            f"val group={val_block}, test groups={test_blocks}"
        )

        test_idx = np.where(is_original & np.isin(blocks, test_blocks))[0]
        val_idx = np.where(is_original & (blocks == val_block))[0]

        train_mask = np.isin(blocks, train_blocks)
        if not include_synthetic:
            train_mask &= is_original
        train_idx = np.where(train_mask)[0]

        expected_test_n = (
            expected_per_class_per_block * len(test_blocks) * num_classes
        )
        expected_val_n = expected_per_class_per_block * num_classes
        expected_train_orig_n = (
            expected_per_class_per_block * len(train_blocks) * num_classes
        )

        n_train_orig = int(np.sum(is_original[train_idx]))
        if (
            len(test_idx) != expected_test_n
            or len(val_idx) != expected_val_n
            or n_train_orig != expected_train_orig_n
        ):
            raise RuntimeError(
                f"Fold {test_fold} count mismatch: "
                f"train originals={n_train_orig} (expected {expected_train_orig_n}), "
                f"val={len(val_idx)} (expected {expected_val_n}), "
                f"test={len(test_idx)} (expected {expected_test_n})"
            )

        fold_dir = os.path.join(run_dir, f"fold{test_fold}")
        checkpoint_path = os.path.join(fold_dir, "best_model.pth")

        if eval_only:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Fold {test_fold} checkpoint를 찾을 수 없습니다: {checkpoint_path}"
                )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            stats = checkpoint.get("stats")
            if not stats:
                raise KeyError(
                    f"Checkpoint에 stats가 없습니다: {checkpoint_path}"
                )
            print(
                f"[test fold {test_fold}] training skipped; "
                f"loaded checkpoint: {checkpoint_path}"
            )
            print(
                f"[test fold {test_fold}] best_val_acc="
                f"{checkpoint.get('best_val_acc')}"
            )
        else:
            train_paths = [all_paths[i] for i in train_idx]
            stats = compute_stats_from_paths(train_paths)

            train_ds = make_subset_from_indices(
                root, stats, train_idx, augment=True,
                noise_std=noise_std, strong_specaugment=strong_specaugment,
            )
            val_ds = make_subset_from_indices(
                root, stats, val_idx, augment=False,
                noise_std=0.0, strong_specaugment=False,
            )

            train_generator = torch.Generator()
            train_generator.manual_seed(seed + int(test_fold))

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True,
                worker_init_fn=seed_worker,
                generator=train_generator,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True,
                worker_init_fn=seed_worker,
            )

            model = build_model(
                num_classes,
                in_chans=len(stats["mean"]),
                head_dropout=head_dropout,
            ).to(device)
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=amp_enabled,
            )
            optimizer, scheduler, warm_freeze = build_optimizer_and_scheduler(
                model,
                base_lr=lr,
                epochs=epochs,
                freeze_epochs=freeze_epochs,
                weight_decay=1e-4,
            )

            os.makedirs(fold_dir, exist_ok=True)
            epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
            with open(epoch_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "train_acc", "val_loss",
                    "val_acc", "lr", "label_smoothing"
                ])

            best_val_acc = -1.0
            history = []

            for epoch in range(epochs):
                if epoch == warm_freeze:
                    optimizer, scheduler = unfreeze_backbone_and_reset_opt(
                        model,
                        current_epoch=epoch,
                        epochs=epochs,
                        base_lr=lr,
                        weight_decay=1e-4,
                    )

                ls = label_smoothing
                train_crit = nn.CrossEntropyLoss(label_smoothing=ls)
                val_crit = nn.CrossEntropyLoss()

                tr_loss, tr_acc = epoch_loop(
                    model,
                    train_loader,
                    train_crit,
                    device,
                    train_mode=True,
                    optimizer=optimizer,
                    max_grad=5.0,
                    use_mixup=use_mixup,
                    mixup_alpha=mixup_alpha,
                    scaler=scaler,
                    use_amp=amp_enabled,
                )
                va_loss, va_acc = epoch_loop(
                    model,
                    val_loader,
                    val_crit,
                    device,
                    train_mode=False,
                    optimizer=None,
                    use_mixup=False,
                    mixup_alpha=0.0,
                    scaler=None,
                    use_amp=amp_enabled,
                )

                scheduler.step()
                cur_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"[test fold {test_fold}] epoch {epoch + 1}/{epochs} | "
                    f"train {tr_loss:.4f}/{tr_acc:.3f} | "
                    f"val {va_loss:.4f}/{va_acc:.3f} | lr {cur_lr:.2e}"
                )

                row = {
                    "fold": test_fold,
                    "epoch": epoch + 1,
                    "train_loss": float(tr_loss),
                    "train_acc": float(tr_acc),
                    "val_loss": float(va_loss),
                    "val_acc": float(va_acc),
                    "lr": float(cur_lr),
                    "label_smoothing": ls,
                }
                history.append(row)
                with open(epoch_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch + 1, tr_loss, tr_acc, va_loss, va_acc, cur_lr, ls
                    ])

                if va_acc > best_val_acc and np.isfinite(va_loss):
                    best_val_acc = float(va_acc)
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "best_val_acc": best_val_acc,
                            "stats": stats,
                            "class_to_idx": getattr(train_ds, "class_to_idx", None),
                            "test_fold": test_fold,
                            "test_blocks": test_blocks,
                            "val_block": val_block,
                            "head_dropout": head_dropout,
                        },
                        checkpoint_path,
                    )

            history_df = pd.DataFrame(history)
            history_df.to_csv(
                os.path.join(fold_dir, "training_metrics.csv"), index=False
            )
            histories.append(history_df)
            plot_fold_curves(
                epoch_csv,
                os.path.join(fold_dir, "training_curves_acc.png"),
                os.path.join(fold_dir, "training_curves_loss.png"),
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

        # Held-out test evaluation. Always use the checkpoint's train statistics.
        checkpoint_stats = checkpoint.get("stats", stats)
        checkpoint_dropout = float(checkpoint.get("head_dropout", head_dropout))

        model = build_model(
            num_classes,
            in_chans=len(checkpoint_stats["mean"]),
            head_dropout=checkpoint_dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        test_ds = make_subset_from_indices(
            root, checkpoint_stats, test_idx, augment=False,
            noise_std=0.0, strong_specaugment=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=4,
            persistent_workers=True,
            worker_init_fn=seed_worker,
        )

        y_true_parts = []
        y_pred_parts = []
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.nan_to_num(
                    x.to(device, non_blocking=True),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(x)
                    pred = logits.argmax(dim=1)

                y_true_parts.append(y.numpy())
                y_pred_parts.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            output_dict=True,
            zero_division=0,
        )

        expected_row_sum = expected_per_class_per_block * len(test_blocks)
        row_sums = cm.sum(axis=1)
        if not np.all(row_sums == expected_row_sum):
            raise RuntimeError(
                f"Fold {test_fold} confusion-matrix row sums are {row_sums.tolist()}, "
                f"expected {expected_row_sum} samples per class."
            )

        np.save(os.path.join(fold_dir, "test_confusion_matrix.npy"), cm)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            os.path.join(fold_dir, "test_confusion_matrix_count.csv")
        )
        cm_normalized = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(cm_normalized, index=class_names, columns=class_names).to_csv(
            os.path.join(fold_dir, "test_confusion_matrix_normalized.csv")
        )
        save_confusion_matrix(
            cm,
            class_names,
            os.path.join(fold_dir, "test_confusion_matrix_count.png"),
            normalize=False,
            title=f"Grouped fold {test_fold} test confusion matrix",
        )
        save_confusion_matrix(
            cm,
            class_names,
            os.path.join(fold_dir, "test_confusion_matrix_normalized.png"),
            normalize=True,
            title=f"Grouped fold {test_fold} test confusion matrix",
        )
        pd.DataFrame(
            {
                "true_index": y_true,
                "true_class": [class_names[int(i)] for i in y_true],
                "pred_index": y_pred,
                "pred_class": [class_names[int(i)] for i in y_pred],
            }
        ).to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)

        with open(os.path.join(fold_dir, "test_classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        all_true.append(y_true)
        all_pred.append(y_pred)
        summary_rows.append(
            {
                "held_out_fold": test_fold,
                "held_out_groups": ",".join(map(str, test_blocks)),
                "validation_block": val_block,
                "n_train_total": int(len(train_idx)),
                "n_train_original": n_train_orig,
                "n_val_original": int(len(val_idx)),
                "n_test_original": int(len(test_idx)),
                "best_val_acc": checkpoint.get("best_val_acc"),
                "test_acc": float(acc),
                "test_macro_f1": float(report["macro avg"]["f1-score"]),
            }
        )
        print(f"[test fold {test_fold}] test_acc={acc:.4f}, n_test={len(test_idx)}")

    summary_df = pd.DataFrame(summary_rows).sort_values("held_out_fold")
    summary_name = (
        "grouped_10fold_eval_summary.csv"
        if eval_only else "grouped_10fold_summary.csv"
    )
    summary_df.to_csv(os.path.join(run_dir, summary_name), index=False)

    aggregate_prefix = "eval_" if eval_only else ""

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    aggregate_cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(num_classes)))
    expected_aggregate_row_sum = (
        expected_per_class_per_block
        * sum(len(ten_fold_test_blocks[f]) for f in folds_to_run)
    )
    aggregate_row_sums = aggregate_cm.sum(axis=1)
    if not np.all(aggregate_row_sums == expected_aggregate_row_sum):
        raise RuntimeError(
            f"Aggregate confusion-matrix row sums are {aggregate_row_sums.tolist()}, "
            f"expected {expected_aggregate_row_sum} samples per class."
        )

    np.save(os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix.npy"), aggregate_cm)
    aggregate_cm_df = pd.DataFrame(
        aggregate_cm,
        index=class_names,
        columns=class_names,
    )
    aggregate_cm_df.to_csv(
        os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix_count.csv")
    )
    # Backward-compatible alias.
    aggregate_cm_df.to_csv(
        os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix.csv")
    )
    aggregate_cm_normalized = (
        aggregate_cm.astype(float)
        / np.maximum(aggregate_cm.sum(axis=1, keepdims=True), 1)
    )
    pd.DataFrame(
        aggregate_cm_normalized,
        index=class_names,
        columns=class_names,
    ).to_csv(os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix_normalized.csv"))

    save_confusion_matrix(
        aggregate_cm,
        class_names,
        os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix_count.png"),
        normalize=False,
        title="Aggregated grouped-fold test confusion matrix",
    )
    save_confusion_matrix(
        aggregate_cm,
        class_names,
        os.path.join(run_dir, aggregate_prefix + "aggregate_confusion_matrix_normalized.png"),
        normalize=True,
        title="Aggregated grouped-fold test confusion matrix",
    )

    if histories and not eval_only:
        save_mean_curves(histories, run_dir)

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    overall_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    final_metrics = {
        "n_unique_test_predictions": int(len(y_true_all)),
        "expected_n_unique_test_predictions": int(
            expected_per_class_per_block
            * sum(len(ten_fold_test_blocks[f]) for f in folds_to_run)
            * num_classes
        ),
        "aggregate_accuracy": float(overall_acc),
        "aggregate_macro_f1": float(overall_report["macro avg"]["f1-score"]),
        "fold_accuracy_mean": float(summary_df["test_acc"].mean()),
        "fold_accuracy_sd": float(summary_df["test_acc"].std(ddof=1)),
        "fold_macro_f1_mean": float(summary_df["test_macro_f1"].mean()),
        "fold_macro_f1_sd": float(summary_df["test_macro_f1"].std(ddof=1)),
    }
    final_metrics_name = "eval_final_metrics.json" if eval_only else "final_metrics.json"
    with open(os.path.join(run_dir, final_metrics_name), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n===== LOBO summary =====")
    print(summary_df.to_string(index=False))
    print(json.dumps(final_metrics, indent=2))
    print("Saved to:", run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ten-fold grouped temporally distributed day-level training"
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200) #200
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--out_root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--head_dropout", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument(
        "--strong_specaug",
        action="store_true",
        help="Enable strong time/frequency masking. Disabled by default.",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument(
        "--include_synthetic",
        action="store_true",
        help="Include synthetic samples only from training blocks. Disabled by default for clean evaluation.",
    )

    parser.add_argument("--samples_per_day", type=int, default=10)
    parser.add_argument("--days_per_block", type=int, default=3)
    parser.add_argument("--expected_days", type=int, default=30)
    parser.add_argument("--expected_blocks", type=int, default=10)
    parser.add_argument("--original_trials_per_class", type=int, default=300)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1,11)),
        help="Folds to run. Use --folds 1 4 for the pilot; omit for all ten folds.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and evaluate existing fold*/best_model.pth checkpoints.",
    )

    args = parser.parse_args()
    train_lobo(
        root=args.root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        out_root=args.out_root,
        seed=args.seed,
        head_dropout=args.head_dropout,
        noise_std=args.noise_std,
        strong_specaugment=args.strong_specaug,
        mixup_alpha=args.mixup_alpha,
        use_mixup=not args.no_mixup,
        label_smoothing=args.label_smoothing,
        include_synthetic=args.include_synthetic,
        samples_per_day=args.samples_per_day,
        days_per_block=args.days_per_block,
        expected_days=args.expected_days,
        expected_blocks=args.expected_blocks,
        original_trials_per_class=args.original_trials_per_class,
        folds_to_run=args.folds,
        eval_only=args.eval_only,
    )
