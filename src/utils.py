import os
import random
import re
from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_idx_to_class(ds_full, labels_tensor):
    if hasattr(ds_full, "class_names") and ds_full.class_names:
        return {
            i: ds_full.class_names[i]
            for i in range(len(ds_full.class_names))
        }

    if hasattr(ds_full, "class_to_idx") and ds_full.class_to_idx:
        inverse = {
            v: k
            for k, v in ds_full.class_to_idx.items()
        }
        return {
            i: inverse.get(i, str(i))
            for i in range(int(labels_tensor.max()) + 1)
        }

    return {
        i: str(i)
        for i in range(int(labels_tensor.max()) + 1)
    }


def parse_day(
    path_or_filename: str,
    samples_per_day: int,
) -> Optional[int]:

    normalized = path_or_filename.replace("\\", "/")

    new_match = re.search(
        r"(?:^|/)Day(\d+)_D\d+_T\d+\.(?:npy|csv)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if new_match:
        return int(new_match.group(1))

    old_match = re.search(
        r"(?:^|/)T(\d+)\.(?:npy|csv)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if old_match:
        trial_number = int(old_match.group(1))
        return (trial_number - 1) // samples_per_day + 1

    return None


def build_sample_days(
    all_paths,
    samples_per_day: int,
    expected_days: int,
) -> np.ndarray:
    days = np.full(len(all_paths), -1, dtype=int)
    invalid_files = []

    for i, path in enumerate(all_paths):
        day = parse_day(
            path,
            samples_per_day=samples_per_day,
        )

        if day is None or not 1 <= day <= expected_days:
            invalid_files.append(path)
            continue

        days[i] = day

    if invalid_files:
        preview = "\n".join(invalid_files[:20])
        raise ValueError(
            f"{len(invalid_files)} files have invalid day information.\n"
            f"Examples:\n{preview}"
        )

    return days


def validate_counts(
    labels,
    days,
    idx_to_class,
    expected_days: int,
    samples_per_day: int,
    run_dir: str,
) -> None:
    """Check the expected number of samples for every class and day."""
    labels_np = labels.cpu().numpy()
    num_classes = int(labels.max()) + 1

    records = []
    errors = []

    for day in range(1, expected_days + 1):
        for cls in range(num_classes):
            count = int(
                np.sum(
                    (days == day)
                    & (labels_np == cls)
                )
            )

            records.append(
                {
                    "day": day,
                    "class_index": cls,
                    "class_name": idx_to_class.get(cls, str(cls)),
                    "n_samples": count,
                }
            )

            if count != samples_per_day:
                errors.append(
                    f"day {day}, class {idx_to_class.get(cls, cls)}: "
                    f"expected {samples_per_day}, found {count}"
                )

    count_path = os.path.join(
        run_dir,
        "sample_count_by_day_and_class.csv",
    )
    pd.DataFrame(records).to_csv(
        count_path,
        index=False,
    )

    expected_total = (
        expected_days
        * samples_per_day
        * num_classes
    )
    actual_total = len(labels_np)

    if actual_total != expected_total:
        errors.append(
            f"total samples: expected {expected_total}, "
            f"found {actual_total}"
        )

    if errors:
        preview = "\n".join(errors[:30])
        raise ValueError(
            "Sample-count validation failed. "
            "Training was stopped before model fitting.\n"
            f"Count table: {count_path}\n"
            f"{preview}"
        )


def get_fold_indices(
    days: np.ndarray,
    test_day: int,
    expected_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, int]:
    """
    Return train/validation/test indices for one day-based fold.

    test       = test_day
    validation = next day (cyclic)
    training   = all remaining days
    """
    val_day = (test_day % expected_days) + 1

    train_days = [
        day
        for day in range(1, expected_days + 1)
        if day not in {test_day, val_day}
    ]

    train_idx = np.where(
        np.isin(days, train_days)
    )[0]
    val_idx = np.where(
        days == val_day
    )[0]
    test_idx = np.where(
        days == test_day
    )[0]

    return (
        train_idx,
        val_idx,
        test_idx,
        train_days,
        val_day,
    )


def validate_fold_counts(
    fold: int,
    train_idx,
    val_idx,
    test_idx,
    train_days,
    samples_per_day: int,
    num_classes: int,
) -> None:
    """Validate sample counts for one fold."""
    expected_train_n = (
        samples_per_day
        * len(train_days)
        * num_classes
    )
    expected_val_n = (
        samples_per_day
        * num_classes
    )
    expected_test_n = (
        samples_per_day
        * num_classes
    )

    if (
        len(train_idx) != expected_train_n
        or len(val_idx) != expected_val_n
        or len(test_idx) != expected_test_n
    ):
        raise RuntimeError(
            f"Fold {fold} count mismatch: "
            f"train={len(train_idx)} "
            f"(expected {expected_train_n}), "
            f"val={len(val_idx)} "
            f"(expected {expected_val_n}), "
            f"test={len(test_idx)} "
            f"(expected {expected_test_n})"
        )


def plot_fold_curves(
    csv_path: str,
    out_acc_png: str,
    out_loss_png: str,
) -> None:
    """Save training/validation accuracy and loss curves for one fold."""
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df["train_acc"],
        label="train_acc",
    )
    plt.plot(
        df["epoch"],
        df["val_acc"],
        label="val_acc",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curves (Accuracy)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_acc_png,
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="train_loss",
    )
    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="val_loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves (Loss)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_loss_png,
        dpi=200,
    )
    plt.close()


def save_mean_curves(
    histories,
    run_dir: str,
) -> None:
    """Save epoch-wise mean ± SD curves across folds."""
    all_hist = pd.concat(
        histories,
        ignore_index=True,
    )

    all_hist.to_csv(
        os.path.join(
            run_dir,
            "all_folds_epoch_metrics.csv",
        ),
        index=False,
    )

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

    summary.to_csv(
        os.path.join(
            run_dir,
            "all_folds_epoch_mean_sd.csv",
        ),
        index=False,
    )

    x = summary["epoch"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label in [
        ("train_loss", "Training loss"),
        ("val_loss", "Validation loss"),
    ]:
        mean = summary[f"{prefix}_mean"].to_numpy()
        sd = summary[f"{prefix}_sd"].fillna(0).to_numpy()

        line = ax.plot(
            x,
            mean,
            label=label,
        )[0]
        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.2,
            color=line.get_color(),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mean loss across folds")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            run_dir,
            "mean_loss_10fold.png",
        ),
        dpi=300,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, label in [
        ("train_acc", "Training accuracy"),
        ("val_acc", "Validation accuracy"),
    ]:
        mean = summary[f"{prefix}_mean"].to_numpy()
        sd = summary[f"{prefix}_sd"].fillna(0).to_numpy()

        line = ax.plot(
            x,
            mean,
            label=label,
        )[0]
        ax.fill_between(
            x,
            mean - sd,
            mean + sd,
            alpha=0.2,
            color=line.get_color(),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean accuracy across folds")
    ax.legend()
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            run_dir,
            "mean_accuracy_10fold.png",
        ),
        dpi=300,
    )
    plt.close(fig)
