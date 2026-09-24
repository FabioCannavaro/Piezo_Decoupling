"""
Motorized external test for the existing 10-fold object-recognition models.

Expected processed data:
    motorized_processed/
        eraser/*.npy
        hotpeltier/*.npy
        peltier/*.npy

Each .npy must use the SAME preprocessing/CWT format as the training data.

Example:
    python scripts/test_motorized.py \
        --motor_root "motorized_processed" \
        --run_dir "outputs/YOUR_10FOLD_RUN"
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# scripts/test_motorized.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_multiclass_auroc, evaluate_model
from src.train import build_model


DEFAULT_TEST_CLASSES = ["sponge", "hotpeltier", "peltier"]
CLASS_ALIASES = {"hotpeliter": "hotpeltier"}  # typo-safe


class MotorizedDataset(Dataset):
    """External .npy dataset normalized with each checkpoint's train stats."""

    def __init__(self, records, stats):
        self.records = records
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(-1, 1, 1)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = np.load(rec["path"])
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if arr.ndim != 3:
            raise ValueError(f"Expected (C,H,W), got {arr.shape}: {rec['path']}")
        if arr.shape[0] != len(self.mean):
            raise ValueError(
                f"Channel mismatch: {rec['path']} has {arr.shape[0]} ch, "
                f"checkpoint expects {len(self.mean)} ch"
            )

        x = torch.from_numpy(arr).float()
        x = (x - self.mean) / (self.std + 1e-8)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.tensor(rec["true_idx"], dtype=torch.long)
        return x, y


def discover_folds(run_dir, requested=None):
    if requested:
        folds = sorted(set(requested))
    else:
        folds = []
        for name in os.listdir(run_dir):
            m = re.fullmatch(r"fold(\d+)", name, flags=re.IGNORECASE)
            if m and os.path.isfile(os.path.join(run_dir, name, "best_model.pth")):
                folds.append(int(m.group(1)))
        folds = sorted(set(folds))

    if not folds:
        raise FileNotFoundError(f"No fold*/best_model.pth found in {run_dir}")

    for fold in folds:
        ckpt = os.path.join(run_dir, f"fold{fold}", "best_model.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)

    return folds


def load_mapping(checkpoint):
    mapping = checkpoint.get("class_to_idx")
    if not mapping:
        raise KeyError("Checkpoint has no class_to_idx")
    return {str(k): int(v) for k, v in mapping.items()}


def build_records(motor_root, class_to_idx, test_classes, expected_per_class):
    records = []
    counts = {}

    folder_lookup = {}
    for folder in os.listdir(motor_root):
        path = os.path.join(motor_root, folder)
        if not os.path.isdir(path):
            continue
        canonical = CLASS_ALIASES.get(folder.lower(), folder.lower())
        folder_lookup[canonical] = path

    for class_name in test_classes:
        if class_name not in class_to_idx:
            raise ValueError(
                f"'{class_name}' not found in model classes: {list(class_to_idx.keys())}"
            )
        if class_name not in folder_lookup:
            raise FileNotFoundError(
                f"Motorized class folder not found: {class_name} under {motor_root}"
            )

        class_dir = folder_lookup[class_name]
        paths = sorted(
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(".npy")
        )
        counts[class_name] = len(paths)

        if expected_per_class > 0 and len(paths) != expected_per_class:
            raise ValueError(
                f"{class_name}: expected {expected_per_class} files, found {len(paths)}"
            )

        for path in paths:
            records.append(
                {
                    "path": path,
                    "file": os.path.relpath(path, motor_root),
                    "true_class": class_name,
                    "true_idx": class_to_idx[class_name],
                }
            )

    print("Motorized counts:", counts)
    return records


def selected_metrics(y_true, y_pred, y_prob, test_indices, idx_to_class):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=test_indices,
        average=None,
        zero_division=0,
    )

    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=test_indices,
        average="macro",
        zero_division=0,
    )

    macro_auroc, per_class_auroc = compute_multiclass_auroc(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=y_prob.shape[1],
    )

    per_class = []
    for i, class_idx in enumerate(test_indices):
        per_class.append(
            {
                "class": idx_to_class[class_idx],
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "auroc_ovr": float(per_class_auroc[class_idx]),
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_3class": float(macro_f1),
        "macro_auroc_ovr_3class": float(macro_auroc),
    }
    return metrics, pd.DataFrame(per_class)


def get_rectangular_cm(y_true, y_pred, test_indices, num_classes):
    row_map = {class_idx: r for r, class_idx in enumerate(test_indices)}
    cm = np.zeros((len(test_indices), num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[row_map[int(t)], int(p)] += 1
    return cm


def save_rectangular_cm(cm, row_names, col_names, out_dir, prefix):
    count_df = pd.DataFrame(cm, index=row_names, columns=col_names)
    count_df.to_csv(os.path.join(out_dir, f"{prefix}_confusion_count.csv"))

    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    pd.DataFrame(norm, index=row_names, columns=col_names).to_csv(
        os.path.join(out_dir, f"{prefix}_confusion_normalized.csv")
    )

    for matrix, suffix, normalized in [
        (cm, "count", False),
        (norm, "normalized", True),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=45, ha="right")
        ax.set_yticks(range(len(row_names)))
        ax.set_yticklabels(row_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True motorized class")
        ax.set_title("Motorized external test" + (" (row normalized)" if normalized else ""))
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

        threshold = matrix.max() / 2 if matrix.size else 0
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                text = f"{matrix[r, c] * 100:.1f}%" if normalized else str(int(matrix[r, c]))
                ax.text(
                    c, r, text,
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[r, c] > threshold else "black",
                )

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_{suffix}.png"), dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motor_root", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_TEST_CLASSES)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--expected_per_class", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    motor_root = os.path.abspath(args.motor_root)
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(
        args.out_dir or os.path.join(run_dir, "motorized_evaluation")
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    folds = discover_folds(run_dir, args.folds)

    first_ckpt = torch.load(
        os.path.join(run_dir, f"fold{folds[0]}", "best_model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    class_to_idx = load_mapping(first_ckpt)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    test_classes = [CLASS_ALIASES.get(x.lower(), x.lower()) for x in args.classes]
    test_indices = [class_to_idx[x] for x in test_classes]

    records = build_records(
        motor_root,
        class_to_idx,
        test_classes,
        args.expected_per_class,
    )

    pd.DataFrame(records)[["file", "true_class", "true_idx"]].to_csv(
        os.path.join(out_dir, "motorized_manifest.csv"), index=False
    )

    fold_rows = []
    all_pred_frames = []
    all_fold_probs = []
    reference_true = None

    print("\nModel classes:", class_to_idx)
    print("Folds:", folds)
    print("External samples:", len(records))

    for fold in folds:
        checkpoint_path = os.path.join(run_dir, f"fold{fold}", "best_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if load_mapping(checkpoint) != class_to_idx:
            raise RuntimeError(f"Class mapping differs in fold {fold}")

        stats = checkpoint["stats"]
        head_dropout = float(
            checkpoint.get(
                "head_dropout",
                checkpoint.get("meta", {}).get("head_dropout", 0.3),
            )
        )

        model = build_model(
            num_classes=len(class_to_idx),
            in_chans=len(stats["mean"]),
            head_dropout=head_dropout,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        dataset = MotorizedDataset(records, stats)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        (
            y_true,
            y_pred,
            y_prob,
            _,
            _,
            _,
            _,
            _,
        ) = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            num_classes=len(class_to_idx),
            use_amp=not args.no_amp,
        )

        if reference_true is None:
            reference_true = y_true.copy()
        elif not np.array_equal(reference_true, y_true):
            raise RuntimeError("Sample order changed between folds")

        metrics, per_class_df = selected_metrics(
            y_true, y_pred, y_prob, test_indices, idx_to_class
        )
        metrics["fold"] = fold
        metrics["best_val_acc"] = checkpoint.get("best_val_acc", np.nan)
        fold_rows.append(metrics)
        all_fold_probs.append(y_prob)

        pred_df = pd.DataFrame(
            {
                "fold": fold,
                "file": [r["file"] for r in records],
                "true_class": [idx_to_class[int(x)] for x in y_true],
                "pred_class": [idx_to_class[int(x)] for x in y_pred],
                "correct": (y_true == y_pred).astype(int),
                "confidence": y_prob.max(axis=1),
            }
        )
        for c, name in enumerate(class_names):
            pred_df[f"prob_{name}"] = y_prob[:, c]
        all_pred_frames.append(pred_df)

        print(
            f"fold {fold}: acc={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1_3class']:.4f}, "
            f"AUROC={metrics['macro_auroc_ovr_3class']:.4f}"
        )

    fold_summary = pd.DataFrame(fold_rows).sort_values("fold")
    fold_summary.to_csv(os.path.join(out_dir, "fold_summary.csv"), index=False)
    pd.concat(all_pred_frames, ignore_index=True).to_csv(
        os.path.join(out_dir, "all_fold_predictions.csv"), index=False
    )

    # Probability-mean ensemble: still N=150 unique external samples.
    ensemble_prob = np.stack(all_fold_probs, axis=0).mean(axis=0)
    ensemble_pred = ensemble_prob.argmax(axis=1)
    y_true = reference_true

    ensemble_metrics, per_class_df = selected_metrics(
        y_true, ensemble_pred, ensemble_prob, test_indices, idx_to_class
    )
    per_class_df.to_csv(
        os.path.join(out_dir, "ensemble_per_class_metrics.csv"), index=False
    )

    ensemble_df = pd.DataFrame(
        {
            "file": [r["file"] for r in records],
            "true_class": [idx_to_class[int(x)] for x in y_true],
            "pred_class": [idx_to_class[int(x)] for x in ensemble_pred],
            "correct": (y_true == ensemble_pred).astype(int),
            "confidence": ensemble_prob.max(axis=1),
        }
    )
    for c, name in enumerate(class_names):
        ensemble_df[f"prob_{name}"] = ensemble_prob[:, c]
    ensemble_df.to_csv(
        os.path.join(out_dir, "ensemble_predictions.csv"), index=False
    )

    cm = get_rectangular_cm(
        y_true, ensemble_pred, test_indices, len(class_to_idx)
    )
    save_rectangular_cm(
        cm, test_classes, class_names, out_dir, prefix="ensemble"
    )

    # Reviewer-relevant matched pair: peltier vs hotpeltier.
    pair_mask = np.isin(
        y_true,
        [class_to_idx["peltier"], class_to_idx["hotpeltier"]],
    )
    pair_accuracy = float(
        np.mean(y_true[pair_mask] == ensemble_pred[pair_mask])
    )
    pair_mutual_confusions = int(
        np.sum(
            (y_true == class_to_idx["peltier"])
            & (ensemble_pred == class_to_idx["hotpeltier"])
        )
        + np.sum(
            (y_true == class_to_idx["hotpeltier"])
            & (ensemble_pred == class_to_idx["peltier"])
        )
    )

    final = {
        "n_unique_external_samples": len(records),
        "n_folds": len(folds),
        "ensemble_accuracy": ensemble_metrics["accuracy"],
        "ensemble_macro_f1_3class": ensemble_metrics["macro_f1_3class"],
        "ensemble_macro_auroc_ovr_3class": ensemble_metrics["macro_auroc_ovr_3class"],
        "fold_accuracy_mean": float(fold_summary["accuracy"].mean()),
        "fold_accuracy_sd": float(fold_summary["accuracy"].std(ddof=1)) if len(fold_summary) > 1 else 0.0,
        "fold_macro_f1_mean": float(fold_summary["macro_f1_3class"].mean()),
        "fold_macro_f1_sd": float(fold_summary["macro_f1_3class"].std(ddof=1)) if len(fold_summary) > 1 else 0.0,
        "matched_pair_accuracy_full_8class": pair_accuracy,
        "peltier_hotpeltier_mutual_confusions": pair_mutual_confusions,
        "peltier_hotpeltier_mutual_confusion_rate": float(pair_mutual_confusions / pair_mask.sum()),
    }

    pd.DataFrame([final]).to_csv(
        os.path.join(out_dir, "final_metrics.csv"), index=False
    )
    with open(os.path.join(out_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL =====")
    print(f"Ensemble accuracy = {final['ensemble_accuracy']:.4f}")
    print(f"Ensemble macro-F1 = {final['ensemble_macro_f1_3class']:.4f}")
    print(f"Ensemble AUROC = {final['ensemble_macro_auroc_ovr_3class']:.4f}")
    print(
        f"Fold accuracy = {final['fold_accuracy_mean']:.4f} "
        f"± {final['fold_accuracy_sd']:.4f}"
    )
    print(f"Peltier/hot-Peltier pair accuracy = {pair_accuracy:.4f}")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
