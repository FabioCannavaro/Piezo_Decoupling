"""
Training-data-volume experiment using the same day-based 10-fold protocol.

For fold k:
- test       = day k
- validation = next day (cyclic)
- training   = remaining eight days

Only the TRAINING SET is reduced to the requested fraction.
Validation and test sets are unchanged.

To avoid confounding training-data volume with day/class imbalance,
sampling is performed independently within each (training day, class)
stratum.

Example:
python train_data_fraction.py --root "day15_3ch"

Default fractions:
10%, 25%, 50%, 75%, 100%
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.dataset import SpectrogramDataset
from src.evaluate import (
    evaluate_model,
    save_aggregate_results,
    save_fold_results,
)
from src.train import (
    build_model,
    build_optimizer_and_scheduler,
    compute_stats_from_paths,
    epoch_loop,
    make_subset_from_indices,
    unfreeze_backbone_and_reset_opt,
)
from src.utils import (
    build_idx_to_class,
    build_sample_days,
    get_fold_indices,
    nowstamp,
    plot_fold_curves,
    save_mean_curves,
    seed_worker,
    validate_counts,
    validate_fold_counts,
)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    sampler=None,
):
    """Create a DataLoader using the common paper settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )



def select_training_fraction(
    train_idx,
    labels,
    days,
    fraction: float,
    seed: int,
):
    """
    Select an EXACT fraction of the full training set while preserving
    the training-day x class distribution as closely as mathematically
    possible.

    Example for 1920 training samples:
        10%  -> 192 samples exactly
        25%  -> 480 samples exactly
        50%  -> 960 samples exactly
        75%  -> 1440 samples exactly
        100% -> 1920 samples exactly

    For fractions such as 25% or 75%, an identical number cannot always
    be selected from every day x class stratum (e.g., 30 * 0.25 = 7.5).
    In that case, the remaining samples are distributed across strata
    using a largest-remainder allocation with deterministic seeded
    tie-breaking. This keeps the allocation as balanced as possible
    while matching the requested TOTAL sample count exactly.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"train fraction must be in (0, 1], got {fraction}"
        )

    train_idx = np.asarray(
        train_idx,
        dtype=np.int64,
    )

    # Preserve the exact original training set and order for the 100% case.
    if fraction >= 1.0:
        return train_idx.copy()

    labels_np = np.asarray(
        labels.cpu().numpy()
        if torch.is_tensor(labels)
        else labels
    )
    days_np = np.asarray(days)

    rng = np.random.default_rng(seed)

    # Exact target number for the whole fold.
    target_total = int(
        round(
            len(train_idx)
            * fraction
        )
    )
    target_total = max(
        1,
        min(
            target_total,
            len(train_idx),
        ),
    )

    # Build day x class strata.
    strata = []
    train_days = np.unique(
        days_np[train_idx]
    )
    train_classes = np.unique(
        labels_np[train_idx]
    )

    for day in train_days:
        for cls in train_classes:
            stratum = train_idx[
                (
                    days_np[train_idx] == day
                )
                & (
                    labels_np[train_idx] == cls
                )
            ]

            if len(stratum) == 0:
                continue

            ideal = len(stratum) * fraction
            base = int(np.floor(ideal))
            remainder = float(ideal - base)

            strata.append(
                {
                    "day": int(day),
                    "class": int(cls),
                    "indices": stratum,
                    "n_total": int(len(stratum)),
                    "ideal": float(ideal),
                    "n_keep": int(base),
                    "remainder": remainder,
                    # Seeded random tie-breaker so equal remainders
                    # are distributed without a systematic bias.
                    "tie": float(rng.random()),
                }
            )

    if not strata:
        raise RuntimeError(
            "No training strata were found."
        )

    # If the target is large enough, preserve representation from every
    # existing day x class stratum.
    if target_total >= len(strata):
        for s in strata:
            if s["n_keep"] == 0:
                s["n_keep"] = 1

    # Current total after floor/minimum allocation.
    current_total = sum(
        s["n_keep"]
        for s in strata
    )

    # If minimum-one adjustment made us exceed the exact target,
    # remove from the smallest-remainder strata first.
    if current_total > target_total:
        removable = sorted(
            strata,
            key=lambda s: (
                s["remainder"],
                s["tie"],
            ),
        )

        excess = current_total - target_total

        for s in removable:
            min_keep = (
                1
                if target_total >= len(strata)
                else 0
            )

            while (
                excess > 0
                and s["n_keep"] > min_keep
            ):
                s["n_keep"] -= 1
                excess -= 1

            if excess == 0:
                break

        if excess != 0:
            raise RuntimeError(
                "Could not match the requested training size exactly."
            )

    # Distribute any remaining quota by largest remainder.
    elif current_total < target_total:
        candidates = sorted(
            strata,
            key=lambda s: (
                -s["remainder"],
                s["tie"],
            ),
        )

        remaining = target_total - current_total

        while remaining > 0:
            progressed = False

            for s in candidates:
                if remaining == 0:
                    break

                if s["n_keep"] < s["n_total"]:
                    s["n_keep"] += 1
                    remaining -= 1
                    progressed = True

            if not progressed:
                raise RuntimeError(
                    "Could not allocate the requested training size."
                )

    # Sample within every stratum.
    selected = []

    for s in strata:
        n_keep = s["n_keep"]

        if n_keep <= 0:
            continue

        chosen = rng.choice(
            s["indices"],
            size=n_keep,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    selected = np.asarray(
        sorted(selected),
        dtype=np.int64,
    )

    if len(selected) != target_total:
        raise RuntimeError(
            "Exact-size sampling failed: "
            f"selected={len(selected)}, "
            f"target={target_total}"
        )

    return selected


def train_10fold_fraction(
    root: str,
    batch_size: int,
    epochs: int,
    lr: float,
    freeze_epochs: int,
    out_root: str | None,
    seed: int,
    head_dropout: float,
    noise_std: float,
    strong_specaugment: bool,
    mixup_alpha: float,
    use_mixup: bool,
    label_smoothing: float,
    samples_per_day: int,
    expected_days: int,
    train_fraction: float,
    folds_to_run=None,
) -> None:
    # ---------------------------------------------------------
    # Reproducibility / GPU
    # ---------------------------------------------------------
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU를 사용할 수 없습니다."
        )

    device = torch.device(
        "cuda:0"
    )
    torch.cuda.set_device(0)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision(
        "high"
    )

    amp_enabled = True

    print(
        f"Using GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        "Training settings: "
        f"noise_std={noise_std}, "
        f"strong_specaugment={strong_specaugment}, "
        f"mixup={use_mixup}, "
        f"mixup_alpha={mixup_alpha}, "
        f"label_smoothing={label_smoothing}"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    ds_full = SpectrogramDataset(
        root,
        augment=False,
        stats=None,
        index_json=None,
    )

    all_paths = ds_full.image_paths
    labels = ds_full.labels.clone()

    num_classes = (
        int(labels.max()) + 1
    )

    idx_to_class = (
        build_idx_to_class(
            ds_full,
            labels,
        )
    )

    class_names = [
        idx_to_class[i]
        for i in range(num_classes)
    ]

    # ---------------------------------------------------------
    # Output folder
    # ---------------------------------------------------------
    dataset_name = os.path.basename(
        os.path.normpath(root)
    )

    run_dir = (
        out_root
        or os.path.join(
            "outputs",
            (
                f"{nowstamp()}_"
                f"convnextv2_tiny_"
                f"{dataset_name}_"
                f"TRAINFRAC_{int(round(train_fraction * 100)):03d}"
            ),
        )
    )

    os.makedirs(
        run_dir,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Day information
    # ---------------------------------------------------------
    days = build_sample_days(
        all_paths=all_paths,
        samples_per_day=samples_per_day,
        expected_days=expected_days,
    )

    validate_counts(
        labels=labels,
        days=days,
        idx_to_class=idx_to_class,
        expected_days=expected_days,
        samples_per_day=samples_per_day,
        run_dir=run_dir,
    )

    if folds_to_run is None:
        folds_to_run = list(
            range(
                1,
                expected_days + 1,
            )
        )
    else:
        folds_to_run = list(
            folds_to_run
        )

    invalid_folds = sorted(
        set(folds_to_run)
        - set(
            range(
                1,
                expected_days + 1,
            )
        )
    )

    if invalid_folds:
        raise ValueError(
            f"Invalid folds: "
            f"{invalid_folds}"
        )

    # ---------------------------------------------------------
    # Save configuration
    # ---------------------------------------------------------
    meta = {
        "root": root,
        "expected_days": expected_days,
        "samples_per_day_per_class": samples_per_day,
        "num_classes": num_classes,
        "class_names": class_names,
        "seed": seed,
        "train_fraction": float(train_fraction),
        "train_percent": float(train_fraction * 100.0),
        "sampling_strategy": "stratified within each training day x class",
        "folds_to_run": folds_to_run,
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
        },
    }

    with open(
        os.path.join(
            run_dir,
            "meta.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------
    summary_rows = []
    all_true = []
    all_pred = []
    all_prob = []
    histories = []

    for fold in folds_to_run:
        test_day = fold

        (
            train_idx,
            val_idx,
            test_idx,
            train_days,
            val_day,
        ) = get_fold_indices(
            days=days,
            test_day=test_day,
            expected_days=expected_days,
        )

        validate_fold_counts(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            train_days=train_days,
            samples_per_day=samples_per_day,
            num_classes=num_classes,
        )

        print(
            f"\n========== "
            f"Fold {fold}/{expected_days} "
            f"=========="
        )
        print(
            f"train days={train_days}, "
            f"val day={val_day}, "
            f"test day={test_day}"
        )

        fold_dir = os.path.join(
            run_dir,
            f"fold{fold}",
        )

        checkpoint_path = os.path.join(
            fold_dir,
            "best_model.pth",
        )

        # =====================================================
        # TRAIN
        # =====================================================
        os.makedirs(
            fold_dir,
            exist_ok=True,
        )

        full_train_idx = np.asarray(
            train_idx,
            dtype=np.int64,
        )

        train_idx = select_training_fraction(
            train_idx=full_train_idx,
            labels=labels,
            days=days,
            fraction=train_fraction,
            seed=(
                seed
                + fold * 1000
                + int(
                    round(
                        train_fraction
                        * 1000
                    )
                )
            ),
        )

        print(
            f"training fraction="
            f"{train_fraction:.2f} | "
            f"selected {len(train_idx)}/"
            f"{len(full_train_idx)} "
            f"training samples"
        )

        train_paths = [
            all_paths[i]
            for i in train_idx
        ]

        stats = compute_stats_from_paths(
            train_paths
        )

        train_ds = make_subset_from_indices(
            root,
            stats,
            train_idx,
            augment=True,
            noise_std=noise_std,
            strong_specaugment=strong_specaugment,
        )

        val_ds = make_subset_from_indices(
            root,
            stats,
            val_idx,
            augment=False,
        )

        # Class-balanced sampler
        class_counts = torch.bincount(
            train_ds.labels,
            minlength=num_classes,
        )

        class_weights = (
            class_counts.sum()
            / (
                class_counts
                + 1e-6
            )
        ).float()

        sample_weights = (
            class_weights[
                train_ds.labels
            ]
        )

        sampler_generator = (
            torch.Generator()
        )
        sampler_generator.manual_seed(
            seed + fold
        )

        train_sampler = (
            WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(
                    train_ds
                ),
                replacement=True,
                generator=sampler_generator,
            )
        )

        train_loader = make_loader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
        )

        val_loader = make_loader(
            val_ds,
            batch_size=batch_size,
        )

        model = build_model(
            num_classes=num_classes,
            in_chans=len(
                stats["mean"]
            ),
            head_dropout=head_dropout,
        ).to(device)

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

        (
            optimizer,
            scheduler,
            warm_freeze,
        ) = build_optimizer_and_scheduler(
            model,
            base_lr=lr,
            epochs=epochs,
            freeze_epochs=freeze_epochs,
            weight_decay=1e-4,
        )

        epoch_csv = os.path.join(
            fold_dir,
            "epoch_metrics.csv",
        )

        with open(
            epoch_csv,
            "w",
            newline="",
        ) as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "lr",
                    "label_smoothing",
                ]
            )

        best_val_acc = -1.0
        history = []

        for epoch in range(
            epochs
        ):
            if epoch == warm_freeze:
                (
                    optimizer,
                    scheduler,
                ) = (
                    unfreeze_backbone_and_reset_opt(
                        model,
                        current_epoch=epoch,
                        epochs=epochs,
                        base_lr=lr,
                        weight_decay=1e-4,
                    )
                )

            train_criterion = (
                nn.CrossEntropyLoss(
                    label_smoothing=label_smoothing
                )
            )

            val_criterion = (
                nn.CrossEntropyLoss()
            )

            tr_loss, tr_acc = epoch_loop(
                model,
                train_loader,
                train_criterion,
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
                val_criterion,
                device,
                train_mode=False,
                use_mixup=False,
                scaler=None,
                use_amp=amp_enabled,
            )

            scheduler.step()

            current_lr = (
                optimizer.param_groups[
                    0
                ]["lr"]
            )

            print(
                f"[fold {fold}] "
                f"epoch {epoch + 1}/{epochs} | "
                f"train "
                f"{tr_loss:.4f}/{tr_acc:.3f} | "
                f"val "
                f"{va_loss:.4f}/{va_acc:.3f} | "
                f"lr {current_lr:.2e}"
            )

            row = {
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": float(
                    tr_loss
                ),
                "train_acc": float(
                    tr_acc
                ),
                "val_loss": float(
                    va_loss
                ),
                "val_acc": float(
                    va_acc
                ),
                "lr": float(
                    current_lr
                ),
                "label_smoothing": (
                    label_smoothing
                ),
            }

            history.append(
                row
            )

            with open(
                epoch_csv,
                "a",
                newline="",
            ) as f:
                csv.writer(f).writerow(
                    [
                        epoch + 1,
                        tr_loss,
                        tr_acc,
                        va_loss,
                        va_acc,
                        current_lr,
                        label_smoothing,
                    ]
                )

            if (
                va_acc > best_val_acc
                and np.isfinite(
                    va_loss
                )
            ):
                best_val_acc = float(
                    va_acc
                )

                torch.save(
                    {
                        "model_state": (
                            model.state_dict()
                        ),
                        "best_val_acc": (
                            best_val_acc
                        ),
                        "stats": stats,
                        "class_to_idx": getattr(
                            train_ds,
                            "class_to_idx",
                            None,
                        ),
                        "fold": fold,
                        "test_day": test_day,
                        "validation_day": val_day,
                        "head_dropout": (
                            head_dropout
                        ),
                    },
                    checkpoint_path,
                )

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            os.path.join(
                fold_dir,
                "training_metrics.csv",
            ),
            index=False,
        )

        histories.append(
            history_df
        )

        plot_fold_curves(
            epoch_csv,
            os.path.join(
                fold_dir,
                "training_curves_acc.png",
            ),
            os.path.join(
                fold_dir,
                "training_curves_loss.png",
            ),
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        # =====================================================
        # TEST
        # =====================================================
        checkpoint_stats = checkpoint[
            "stats"
        ]

        checkpoint_dropout = float(
            checkpoint.get(
                "head_dropout",
                head_dropout,
            )
        )

        model = build_model(
            num_classes=num_classes,
            in_chans=len(
                checkpoint_stats["mean"]
            ),
            head_dropout=checkpoint_dropout,
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state"]
        )

        test_ds = make_subset_from_indices(
            root,
            checkpoint_stats,
            test_idx,
            augment=False,
        )

        test_loader = make_loader(
            test_ds,
            batch_size=batch_size,
        )

        (
            y_true,
            y_pred,
            y_prob,
            test_acc,
            cm,
            report,
            test_macro_auroc_ovr,
            test_per_class_auroc,
        ) = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=num_classes,
            use_amp=amp_enabled,
        )

        save_fold_results(
            fold_dir=fold_dir,
            fold=fold,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            cm=cm,
            report=report,
            macro_auroc=test_macro_auroc_ovr,
            per_class_auroc=test_per_class_auroc,
            class_names=class_names,
            samples_per_day=samples_per_day,
        )

        all_true.append(
            y_true
        )
        all_pred.append(
            y_pred
        )
        all_prob.append(
            y_prob
        )

        summary_rows.append(
            {
                "fold": fold,
                "test_day": test_day,
                "validation_day": val_day,
                "train_fraction": float(
                    train_fraction
                ),
                "train_percent": float(
                    train_fraction * 100.0
                ),
                "n_train_full": len(
                    full_train_idx
                ),
                "n_train": len(
                    train_idx
                ),
                "n_val": len(
                    val_idx
                ),
                "n_test": len(
                    test_idx
                ),
                "best_val_acc": (
                    checkpoint.get(
                        "best_val_acc"
                    )
                ),
                "test_acc": test_acc,
                "test_macro_f1": float(
                    report[
                        "macro avg"
                    ]["f1-score"]
                ),
                "test_macro_auroc_ovr": float(
                    test_macro_auroc_ovr
                ),
            }
        )

        print(
            f"[fold {fold}] "
            f"test_acc="
            f"{test_acc:.4f} | "
            f"macro_F1="
            f"{report['macro avg']['f1-score']:.4f} | "
            f"macro_AUROC(OvR)="
            f"{test_macro_auroc_ovr:.4f}"
        )

    # ---------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------
    if histories:
        save_mean_curves(
            histories,
            run_dir,
        )

    (
        summary_df,
        final_metrics,
    ) = save_aggregate_results(
        run_dir=run_dir,
        summary_rows=summary_rows,
        all_true=all_true,
        all_pred=all_pred,
        all_prob=all_prob,
        class_names=class_names,
        num_classes=num_classes,
        samples_per_day=samples_per_day,
        folds_to_run=folds_to_run,
        eval_only=False,
    )

    print(
        "\n===== Training-data fraction 10-fold summary ====="
    )
    print(
        summary_df.to_string(
            index=False
        )
    )
    print(
        json.dumps(
            final_metrics,
            indent=2,
        )
    )
    print(
        "Saved to:",
        run_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Training-data-volume experiment "
            "with day-based 10-fold evaluation."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Processed npy dataset root.",
    )

    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[
            0.10,
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        help=(
            "Training fractions. "
            "Default: 0.10 0.25 0.50 0.75 1.00"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help=(
            "Base output directory. "
            "Each fraction is saved in its own subfolder."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--no_strong_specaug",
        action="store_true",
    )

    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--no_mixup",
        action="store_true",
    )

    parser.add_argument(
        "--samples_per_day",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--expected_days",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--full_run_dir",
        type=str,
        default=None,
        help=(
            "Optional existing main 10-fold run directory. "
            "When provided and 100%% is requested, the 100%% point is "
            "reused from <full_run_dir>/final_metrics.csv instead of "
            "retraining. This keeps the training-volume 100%% point "
            "identical to the main 10-fold result."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Example: --folds 1. "
            "Omit to run all folds."
        ),
    )

    args = parser.parse_args()

    for fraction in args.fractions:
        if not (
            0.0 < fraction <= 1.0
        ):
            raise ValueError(
                f"Invalid fraction: {fraction}"
            )

    dataset_name = os.path.basename(
        os.path.normpath(
            args.root
        )
    )

    base_out_root = (
        args.out_root
        or os.path.join(
            "outputs",
            (
                f"{nowstamp()}_"
                f"convnextv2_tiny_"
                f"{dataset_name}_"
                f"TRAINING_VOLUME"
            ),
        )
    )

    os.makedirs(
        base_out_root,
        exist_ok=True,
    )

    experiment_rows = []

    for fraction in args.fractions:
        percent = int(
            round(
                fraction * 100
            )
        )

        fraction_dir = os.path.join(
            base_out_root,
            f"fraction_{percent:03d}",
        )

        print(
            "\n\n########################################"
        )
        print(
            f"Training-data volume: {percent}%"
        )
        print(
            "########################################"
        )

        # -----------------------------------------------------
        # 100%: optionally reuse the main 10-fold result.
        # -----------------------------------------------------
        if (
            fraction >= 1.0
            and args.full_run_dir is not None
        ):
            source_final_csv = os.path.join(
                args.full_run_dir,
                "final_metrics.csv",
            )

            if not os.path.isfile(
                source_final_csv
            ):
                raise FileNotFoundError(
                    "Could not find the main 10-fold final metrics: "
                    f"{source_final_csv}"
                )

            os.makedirs(
                fraction_dir,
                exist_ok=True,
            )

            row = pd.read_csv(
                source_final_csv
            ).iloc[0].to_dict()

            row[
                "train_fraction"
            ] = 1.0

            row[
                "train_percent"
            ] = 100.0

            row[
                "result_source"
            ] = "reused_main_10fold"

            experiment_rows.append(
                row
            )

            # Keep a local copy of the reused final metrics.
            reused_df = pd.DataFrame(
                [row]
            )

            reused_df.to_csv(
                os.path.join(
                    fraction_dir,
                    "final_metrics.csv",
                ),
                index=False,
            )

            reuse_meta = {
                "train_fraction": 1.0,
                "train_percent": 100.0,
                "result_source": "reused_main_10fold",
                "source_run_dir": os.path.abspath(
                    args.full_run_dir
                ),
                "source_final_metrics": os.path.abspath(
                    source_final_csv
                ),
                "note": (
                    "The 100% training-volume point was reused from the "
                    "main 10-fold experiment to ensure an identical full-data "
                    "reference rather than a second stochastic retraining."
                ),
            }

            with open(
                os.path.join(
                    fraction_dir,
                    "reuse_info.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    reuse_meta,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                "Reused 100% result from:",
                source_final_csv,
            )

            continue

        # -----------------------------------------------------
        # Fractions below 100%, or 100% when no existing
        # main-run directory is provided.
        # -----------------------------------------------------
        train_10fold_fraction(
            root=args.root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            out_root=fraction_dir,
            seed=args.seed,
            head_dropout=args.head_dropout,
            noise_std=args.noise_std,
            strong_specaugment=(
                not args.no_strong_specaug
            ),
            mixup_alpha=args.mixup_alpha,
            use_mixup=(
                not args.no_mixup
            ),
            label_smoothing=args.label_smoothing,
            samples_per_day=args.samples_per_day,
            expected_days=args.expected_days,
            train_fraction=fraction,
            folds_to_run=args.folds,
        )

        final_csv = os.path.join(
            fraction_dir,
            "final_metrics.csv",
        )

        if os.path.isfile(
            final_csv
        ):
            row = pd.read_csv(
                final_csv
            ).iloc[0].to_dict()

            row[
                "train_fraction"
            ] = float(
                fraction
            )

            row[
                "train_percent"
            ] = float(
                percent
            )

            row[
                "result_source"
            ] = "trained_fraction_run"

            experiment_rows.append(
                row
            )

    if experiment_rows:
        summary_df = pd.DataFrame(
            experiment_rows
        ).sort_values(
            "train_percent"
        )

        summary_path = os.path.join(
            base_out_root,
            "training_volume_summary.csv",
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

        print(
            "\nSaved Origin-ready summary:",
            summary_path,
        )
