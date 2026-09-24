"""
Day-based 10-fold training and evaluation.

For fold k:
- test       = day k
- validation = next day (cyclic)
- training   = remaining eight days
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


def train_10fold(
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
    folds_to_run=None,
    eval_only: bool = False,
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

    if eval_only:
        if out_root is None:
            raise ValueError(
                "--eval_only 사용 시 "
                "--out_root를 지정해야 합니다."
            )

        run_dir = os.path.abspath(
            out_root
        )

        if not os.path.isdir(
            run_dir
        ):
            raise FileNotFoundError(
                run_dir
            )

    else:
        run_dir = (
            out_root
            or os.path.join(
                "outputs",
                (
                    f"{nowstamp()}_"
                    f"convnextv2_tiny_"
                    f"{dataset_name}_"
                    f"10FOLD_DAY"
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
    if not eval_only:
        meta = {
            "root": root,
            "expected_days": expected_days,
            "samples_per_day_per_class": samples_per_day,
            "num_classes": num_classes,
            "class_names": class_names,
            "seed": seed,
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
        if eval_only:
            if not os.path.isfile(
                checkpoint_path
            ):
                raise FileNotFoundError(
                    checkpoint_path
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            stats = checkpoint[
                "stats"
            ]

        else:
            os.makedirs(
                fold_dir,
                exist_ok=True,
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
    if histories and not eval_only:
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
        eval_only=eval_only,
    )

    print(
        "\n===== 10-fold summary ====="
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
            "Day-based 10-fold "
            "training and evaluation"
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
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
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Example: --folds 1. "
            "Omit to run all folds."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
    )

    args = parser.parse_args()

    train_10fold(
        root=args.root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        out_root=args.out_root,
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
        folds_to_run=args.folds,
        eval_only=args.eval_only,
    )