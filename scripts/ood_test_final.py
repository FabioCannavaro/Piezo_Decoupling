"""
Evaluate trained fold checkpoints on position-shift OOD data.

Expected OOD directory structure
--------------------------------
<ood_root>/
    <class_name>/
        center/
            *.npy
        left/
            *.npy
        right/
            *.npy

IMPORTANT
---------
The OOD npy files must have the SAME channel configuration as the
trained model:
- 1ch-Ch1 checkpoint -> 1ch-Ch1 OOD npy
- 1ch-Ch2 checkpoint -> 1ch-Ch2 OOD npy
- 2ch checkpoint     -> 2ch OOD npy
- 3ch checkpoint     -> 3ch OOD npy

No training or fine-tuning is performed.

Outputs are CSV-first so figures can be redrawn in Origin:
- ood_predictions.csv
- ood_fold_metrics.csv
- ood_position_summary.csv
- ood_class_position_fold_metrics.csv
- ood_class_position_summary.csv

Example
-------
python ood_test.py ^
    --ood_root "test_data_3_3ch" ^
    --run_dir "outputs/...day15_3ch...10FOLD_DAY" ^
    --train_root "day15_3ch"
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from src.dataset import SpectrogramDataset
from src.train import build_model


class OODDataset(Dataset):
    def __init__(
        self,
        records,
        stats,
    ):
        self.records = records

        self.mean = np.asarray(
            stats["mean"],
            dtype=np.float32,
        )[:, None, None]

        self.std = np.asarray(
            stats["std"],
            dtype=np.float32,
        )[:, None, None]

        self.std = np.maximum(
            self.std,
            1e-8,
        )

    def __len__(self):
        return len(
            self.records
        )

    def __getitem__(
        self,
        idx,
    ):
        rec = self.records[idx]

        arr = np.load(
            rec["path"]
        ).astype(
            np.float32
        )

        if arr.ndim == 2:
            arr = arr[
                None,
                ...,
            ]

        if arr.ndim != 3:
            raise ValueError(
                f"Expected (C,H,W), got "
                f"{arr.shape}: {rec['path']}"
            )

        if (
            arr.shape[0]
            != self.mean.shape[0]
        ):
            raise ValueError(
                "OOD/checkpoint channel mismatch: "
                f"file={arr.shape[0]}, "
                f"checkpoint={self.mean.shape[0]} | "
                f"{rec['path']}"
            )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        arr = (
            arr
            - self.mean
        ) / self.std

        x = torch.from_numpy(
            arr
        ).float()

        y = int(
            rec["label"]
        )

        return (
            x,
            y,
            rec["path"],
            rec["class_name"],
            rec["position"],
        )


def infer_class_mapping(
    checkpoint,
    train_root=None,
):
    """
    Prefer the class_to_idx saved inside the checkpoint.
    If unavailable, recover it from the original training dataset.
    """
    class_to_idx = checkpoint.get(
        "class_to_idx"
    )

    if class_to_idx:
        class_to_idx = {
            str(k): int(v)
            for k, v in class_to_idx.items()
        }

        return class_to_idx

    if train_root is None:
        raise RuntimeError(
            "Checkpoint has no class_to_idx. "
            "Provide --train_root so the original "
            "training class mapping can be recovered."
        )

    ds = SpectrogramDataset(
        train_root,
        augment=False,
        stats=None,
        index_json=None,
    )

    mapping = getattr(
        ds,
        "class_to_idx",
        None,
    )

    if not mapping:
        raise RuntimeError(
            "Could not recover class_to_idx "
            "from the training dataset."
        )

    return {
        str(k): int(v)
        for k, v in mapping.items()
    }


def discover_folds(
    run_dir,
    requested_folds=None,
):
    if requested_folds:
        return sorted(
            set(
                int(x)
                for x in requested_folds
            )
        )

    fold_ids = []

    for path in glob.glob(
        os.path.join(
            run_dir,
            "fold*",
        )
    ):
        if not os.path.isdir(
            path
        ):
            continue

        match = re.fullmatch(
            r"fold(\d+)",
            os.path.basename(
                path
            ),
        )

        if match:
            fold_ids.append(
                int(
                    match.group(1)
                )
            )

    fold_ids = sorted(
        set(fold_ids)
    )

    if not fold_ids:
        raise RuntimeError(
            f"No fold directories found under: {run_dir}"
        )

    return fold_ids


def build_ood_records(
    ood_root,
    class_to_idx,
    positions,
):
    records = []

    available_classes = []

    for class_name in sorted(
        os.listdir(
            ood_root
        )
    ):
        class_dir = os.path.join(
            ood_root,
            class_name,
        )

        if not os.path.isdir(
            class_dir
        ):
            continue

        if (
            class_name
            not in class_to_idx
        ):
            raise ValueError(
                f"OOD class '{class_name}' is not "
                "present in the training class mapping."
            )

        available_classes.append(
            class_name
        )

        for position in positions:
            pos_dir = os.path.join(
                class_dir,
                position,
            )

            if not os.path.isdir(
                pos_dir
            ):
                continue

            for npy_path in sorted(
                glob.glob(
                    os.path.join(
                        pos_dir,
                        "*.npy",
                    )
                )
            ):
                records.append(
                    {
                        "path": npy_path,
                        "class_name": class_name,
                        "label": int(
                            class_to_idx[
                                class_name
                            ]
                        ),
                        "position": position,
                    }
                )

    if not records:
        raise RuntimeError(
            f"No OOD npy files found under: {ood_root}"
        )

    return (
        records,
        sorted(
            set(
                available_classes
            )
        ),
    )


@torch.no_grad()
def infer_checkpoint(
    model,
    loader,
    device,
    idx_to_class,
    fold,
    use_amp=True,
):
    model.eval()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    rows = []

    for (
        x,
        y,
        paths,
        class_names,
        positions,
    ) in loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(
                x
            )

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        pred = probs.argmax(
            dim=1
        )

        probs_np = probs.cpu().numpy()
        pred_np = pred.cpu().numpy()
        y_np = y.numpy()

        for i in range(
            len(y_np)
        ):
            row = {
                "fold": int(
                    fold
                ),
                "position": positions[i],
                "path": paths[i],
                "true_index": int(
                    y_np[i]
                ),
                "true_class": class_names[i],
                "pred_index": int(
                    pred_np[i]
                ),
                "pred_class": idx_to_class[
                    int(
                        pred_np[i]
                    )
                ],
                "correct": int(
                    pred_np[i]
                    == y_np[i]
                ),
            }

            for c in sorted(
                idx_to_class
            ):
                row[
                    f"prob_{idx_to_class[c]}"
                ] = float(
                    probs_np[i, c]
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def summarize_predictions(
    predictions_df,
    available_class_names,
    class_to_idx,
    reference_position,
):
    available_labels = [
        class_to_idx[
            name
        ]
        for name in available_class_names
    ]

    fold_rows = []

    for (
        fold,
        position,
    ), group in predictions_df.groupby(
        [
            "fold",
            "position",
        ],
        sort=True,
    ):
        y_true = group[
            "true_index"
        ].to_numpy()

        y_pred = group[
            "pred_index"
        ].to_numpy()

        fold_rows.append(
            {
                "fold": int(
                    fold
                ),
                "position": position,
                "distribution": (
                    "reference"
                    if position
                    == reference_position
                    else "OOD"
                ),
                "n_samples": int(
                    len(group)
                ),
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),
                "macro_f1_available_classes": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=available_labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    fold_metrics = pd.DataFrame(
        fold_rows
    ).sort_values(
        [
            "position",
            "fold",
        ]
    )

    position_summary = (
        fold_metrics
        .groupby(
            [
                "position",
                "distribution",
            ],
            as_index=False,
        )
        .agg(
            n_folds=(
                "fold",
                "nunique",
            ),
            n_samples_per_fold=(
                "n_samples",
                "first",
            ),
            accuracy_mean=(
                "accuracy",
                "mean",
            ),
            accuracy_sd=(
                "accuracy",
                lambda x: (
                    x.std(ddof=1)
                    if len(x) > 1
                    else 0.0
                ),
            ),
            macro_f1_mean=(
                "macro_f1_available_classes",
                "mean",
            ),
            macro_f1_sd=(
                "macro_f1_available_classes",
                lambda x: (
                    x.std(ddof=1)
                    if len(x) > 1
                    else 0.0
                ),
            ),
        )
    )

    class_fold_rows = []

    for (
        fold,
        position,
        class_name,
    ), group in predictions_df.groupby(
        [
            "fold",
            "position",
            "true_class",
        ],
        sort=True,
    ):
        class_fold_rows.append(
            {
                "fold": int(
                    fold
                ),
                "position": position,
                "distribution": (
                    "reference"
                    if position
                    == reference_position
                    else "OOD"
                ),
                "class_name": class_name,
                "n_samples": int(
                    len(group)
                ),
                "class_accuracy": float(
                    group[
                        "correct"
                    ].mean()
                ),
            }
        )

    class_fold = pd.DataFrame(
        class_fold_rows
    ).sort_values(
        [
            "position",
            "class_name",
            "fold",
        ]
    )

    class_summary = (
        class_fold
        .groupby(
            [
                "position",
                "distribution",
                "class_name",
            ],
            as_index=False,
        )
        .agg(
            n_folds=(
                "fold",
                "nunique",
            ),
            n_samples_per_fold=(
                "n_samples",
                "first",
            ),
            class_accuracy_mean=(
                "class_accuracy",
                "mean",
            ),
            class_accuracy_sd=(
                "class_accuracy",
                lambda x: (
                    x.std(ddof=1)
                    if len(x) > 1
                    else 0.0
                ),
            ),
        )
    )

    return (
        fold_metrics,
        position_summary,
        class_fold,
        class_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained fold checkpoints "
            "on center/left/right position data "
            "without retraining."
        )
    )

    parser.add_argument(
        "--ood_root",
        required=True,
        help=(
            "Channel-matched OOD npy root: "
            "<class>/<position>/*.npy"
        ),
    )

    parser.add_argument(
        "--run_dir",
        required=True,
        help=(
            "Training output directory containing "
            "foldN/best_model.pth"
        ),
    )

    parser.add_argument(
        "--train_root",
        default=None,
        help=(
            "Original training npy root. "
            "Needed only if class_to_idx was not "
            "saved in the checkpoint."
        ),
    )

    parser.add_argument(
        "--positions",
        nargs="+",
        default=[
            "center",
            "left",
            "right",
        ],
    )

    parser.add_argument(
        "--reference_position",
        default="center",
        help=(
            "Position treated as the reference/ID condition. "
            "Other positions are labeled OOD."
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Specific checkpoint folds. "
            "Omit to evaluate every fold found."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--out_dir",
        default=None,
        help=(
            "Default: <run_dir>/ood_evaluation"
        ),
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable."
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

    fold_ids = discover_folds(
        args.run_dir,
        args.folds,
    )

    out_dir = (
        args.out_dir
        or os.path.join(
            args.run_dir,
            "ood_evaluation",
        )
    )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    # Use the first checkpoint to recover the global class mapping.
    first_checkpoint_path = os.path.join(
        args.run_dir,
        f"fold{fold_ids[0]}",
        "best_model.pth",
    )

    first_checkpoint = torch.load(
        first_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    class_to_idx = infer_class_mapping(
        first_checkpoint,
        train_root=args.train_root,
    )

    idx_to_class = {
        idx: name
        for name, idx
        in class_to_idx.items()
    }

    (
        records,
        available_class_names,
    ) = build_ood_records(
        ood_root=args.ood_root,
        class_to_idx=class_to_idx,
        positions=args.positions,
    )

    print(
        "OOD classes:",
        available_class_names,
    )

    print(
        "Positions:",
        sorted(
            set(
                r["position"]
                for r in records
            )
        ),
    )

    print(
        "Total OOD samples:",
        len(records),
    )

    all_prediction_frames = []

    for fold in fold_ids:
        checkpoint_path = os.path.join(
            args.run_dir,
            f"fold{fold}",
            "best_model.pth",
        )

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

        checkpoint_mapping = infer_class_mapping(
            checkpoint,
            train_root=args.train_root,
        )

        if (
            checkpoint_mapping
            != class_to_idx
        ):
            raise RuntimeError(
                f"Class mapping differs in fold {fold}."
            )

        stats = checkpoint[
            "stats"
        ]

        in_chans = len(
            stats["mean"]
        )

        num_classes = len(
            class_to_idx
        )

        head_dropout = float(
            checkpoint.get(
                "head_dropout",
                0.3,
            )
        )

        model = build_model(
            num_classes=num_classes,
            in_chans=in_chans,
            head_dropout=head_dropout,
        ).to(
            device
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        dataset = OODDataset(
            records=records,
            stats=stats,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pred_df = infer_checkpoint(
            model=model,
            loader=loader,
            device=device,
            idx_to_class=idx_to_class,
            fold=fold,
            use_amp=(
                not args.no_amp
            ),
        )

        all_prediction_frames.append(
            pred_df
        )

        print(
            f"[fold {fold}] "
            f"overall OOD-set accuracy="
            f"{pred_df['correct'].mean():.4f}"
        )

    predictions_df = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    predictions_path = os.path.join(
        out_dir,
        "ood_predictions.csv",
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    (
        fold_metrics,
        position_summary,
        class_fold,
        class_summary,
    ) = summarize_predictions(
        predictions_df=predictions_df,
        available_class_names=available_class_names,
        class_to_idx=class_to_idx,
        reference_position=args.reference_position,
    )

    fold_metrics.to_csv(
        os.path.join(
            out_dir,
            "ood_fold_metrics.csv",
        ),
        index=False,
    )

    position_summary.to_csv(
        os.path.join(
            out_dir,
            "ood_position_summary.csv",
        ),
        index=False,
    )

    class_fold.to_csv(
        os.path.join(
            out_dir,
            "ood_class_position_fold_metrics.csv",
        ),
        index=False,
    )

    class_summary.to_csv(
        os.path.join(
            out_dir,
            "ood_class_position_summary.csv",
        ),
        index=False,
    )

    config = {
        "ood_root": os.path.abspath(
            args.ood_root
        ),
        "run_dir": os.path.abspath(
            args.run_dir
        ),
        "train_root": (
            os.path.abspath(
                args.train_root
            )
            if args.train_root
            else None
        ),
        "folds": fold_ids,
        "positions": args.positions,
        "reference_position": (
            args.reference_position
        ),
        "available_ood_classes": (
            available_class_names
        ),
        "class_to_idx": class_to_idx,
        "note": (
            "No retraining or fine-tuning was performed. "
            "Each trained fold checkpoint was evaluated "
            "on the same external position dataset."
        ),
    }

    with open(
        os.path.join(
            out_dir,
            "ood_config.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\\n===== OOD position summary ====="
    )

    print(
        position_summary.to_string(
            index=False
        )
    )

    print(
        "\\nSaved to:",
        out_dir,
    )


if __name__ == "__main__":
    main()
