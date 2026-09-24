import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def compute_multiclass_auroc(
    y_true,
    y_prob,
    num_classes: int,
):
    """
    Multiclass AUROC using one-vs-rest (OvR).

    Returns
    -------
    macro_auroc : float
        Macro-average AUROC across classes.
    per_class_auroc : np.ndarray, shape (num_classes,)
        One-vs-rest AUROC for each class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim != 2 or y_prob.shape[1] != num_classes:
        raise ValueError(
            f"y_prob shape={y_prob.shape}, "
            f"expected (N, {num_classes})."
        )

    per_class = np.full(
        num_classes,
        np.nan,
        dtype=np.float64,
    )

    for c in range(num_classes):
        y_binary = (
            y_true == c
        ).astype(np.int32)

        if np.unique(y_binary).size < 2:
            continue

        per_class[c] = roc_auc_score(
            y_binary,
            y_prob[:, c],
        )

    valid = np.isfinite(
        per_class
    )

    macro_auroc = (
        float(np.mean(per_class[valid]))
        if np.any(valid)
        else float("nan")
    )

    return (
        macro_auroc,
        per_class,
    )


def save_roc_curve_points(
    y_true,
    y_prob,
    class_names,
    out_csv: str,
) -> None:
    """
    Save Origin-ready ROC coordinates in long format.

    Columns:
    class_index, class_name, point_index, fpr, tpr, threshold, auroc_ovr
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    rows = []

    for c, class_name in enumerate(class_names):
        y_binary = (
            y_true == c
        ).astype(np.int32)

        if np.unique(y_binary).size < 2:
            continue

        fpr, tpr, thresholds = roc_curve(
            y_binary,
            y_prob[:, c],
        )

        auc_value = roc_auc_score(
            y_binary,
            y_prob[:, c],
        )

        for i, (
            fpr_i,
            tpr_i,
            threshold_i,
        ) in enumerate(
            zip(
                fpr,
                tpr,
                thresholds,
            )
        ):
            rows.append(
                {
                    "class_index": c,
                    "class_name": class_name,
                    "point_index": i,
                    "fpr": float(fpr_i),
                    "tpr": float(tpr_i),
                    "threshold": float(
                        threshold_i
                    ),
                    "auroc_ovr": float(
                        auc_value
                    ),
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        out_csv,
        index=False,
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    num_classes: int,
    use_amp: bool = True,
):
    """
    Run inference and return predictions, probabilities,
    classification metrics, and macro OvR AUROC.
    """
    model.eval()

    amp_enabled = bool(
        use_amp
        and device.type == "cuda"
    )

    y_true_parts = []
    y_pred_parts = []
    y_prob_parts = []

    for x, y in loader:
        x = torch.nan_to_num(
            x.to(
                device,
                non_blocking=True,
            ),
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

        probs = torch.softmax(
            logits.float(),
            dim=1,
        )

        pred = probs.argmax(
            dim=1
        )

        y_true_parts.append(
            y.numpy()
        )
        y_pred_parts.append(
            pred.cpu().numpy()
        )
        y_prob_parts.append(
            probs.cpu().numpy()
        )

    y_true = np.concatenate(
        y_true_parts
    )

    y_pred = np.concatenate(
        y_pred_parts
    )

    y_prob = np.concatenate(
        y_prob_parts,
        axis=0,
    )

    acc = accuracy_score(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        ),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(num_classes)
        ),
        output_dict=True,
        zero_division=0,
    )

    macro_auroc, per_class_auroc = (
        compute_multiclass_auroc(
            y_true=y_true,
            y_prob=y_prob,
            num_classes=num_classes,
        )
    )

    return (
        y_true,
        y_pred,
        y_prob,
        float(acc),
        cm,
        report,
        float(macro_auroc),
        per_class_auroc,
    )


def save_confusion_matrix(
    cm,
    class_names,
    out_png: str,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> None:
    """Save a count or row-normalized confusion-matrix image."""
    mat = cm.astype(float)

    if normalize:
        mat = mat / np.maximum(
            mat.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        mat,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(class_names))
    )
    ax.set_yticks(
        range(len(class_names))
    )
    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        class_names
    )
    ax.set_xlabel(
        "Predicted class"
    )
    ax.set_ylabel(
        "True class"
    )
    ax.set_title(
        title
        + (
            " (row normalized)"
            if normalize
            else ""
        )
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    threshold = (
        float(mat.max()) / 2
        if mat.size
        else 0.0
    )

    for r in range(
        mat.shape[0]
    ):
        for c in range(
            mat.shape[1]
        ):
            text = (
                f"{mat[r, c] * 100:.1f}%"
                if normalize
                else str(
                    int(mat[r, c])
                )
            )

            ax.text(
                c,
                r,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if mat[r, c] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()
    fig.savefig(
        out_png,
        dpi=300,
    )
    plt.close(fig)


def save_fold_results(
    fold_dir: str,
    fold: int,
    y_true,
    y_pred,
    y_prob,
    cm,
    report,
    macro_auroc: float,
    per_class_auroc,
    class_names,
    samples_per_day: int,
) -> None:
    """Save all test outputs for one fold."""
    row_sums = cm.sum(axis=1)

    if not np.all(
        row_sums == samples_per_day
    ):
        raise RuntimeError(
            f"Fold {fold} confusion-matrix row sums "
            f"are {row_sums.tolist()}, "
            f"expected {samples_per_day} samples per class."
        )

    np.save(
        os.path.join(
            fold_dir,
            "test_confusion_matrix.npy",
        ),
        cm,
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        os.path.join(
            fold_dir,
            "test_confusion_matrix_count.csv",
        )
    )

    cm_normalized = (
        cm.astype(float)
        / np.maximum(
            cm.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    pd.DataFrame(
        cm_normalized,
        index=class_names,
        columns=class_names,
    ).to_csv(
        os.path.join(
            fold_dir,
            "test_confusion_matrix_normalized.csv",
        )
    )

    save_confusion_matrix(
        cm,
        class_names,
        os.path.join(
            fold_dir,
            "test_confusion_matrix_count.png",
        ),
        normalize=False,
        title=f"Fold {fold} test confusion matrix",
    )

    save_confusion_matrix(
        cm,
        class_names,
        os.path.join(
            fold_dir,
            "test_confusion_matrix_normalized.png",
        ),
        normalize=True,
        title=f"Fold {fold} test confusion matrix",
    )

    prediction_data = {
        "true_index": y_true,
        "true_class": [
            class_names[int(i)]
            for i in y_true
        ],
        "pred_index": y_pred,
        "pred_class": [
            class_names[int(i)]
            for i in y_pred
        ],
    }

    for c, class_name in enumerate(
        class_names
    ):
        prediction_data[
            f"prob_{class_name}"
        ] = y_prob[:, c]

    pd.DataFrame(
        prediction_data
    ).to_csv(
        os.path.join(
            fold_dir,
            "test_predictions.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            fold_dir,
            "test_classification_report.json",
        ),
        "w",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    pd.DataFrame(
        {
            "class_index": list(
                range(len(class_names))
            ),
            "class_name": class_names,
            "auroc_ovr": per_class_auroc,
        }
    ).to_csv(
        os.path.join(
            fold_dir,
            "test_auroc_per_class.csv",
        ),
        index=False,
    )

    save_roc_curve_points(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        out_csv=os.path.join(
            fold_dir,
            "test_roc_curves.csv",
        ),
    )

    with open(
        os.path.join(
            fold_dir,
            "test_auroc.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "definition": (
                    "multiclass one-vs-rest "
                    "macro-average AUROC"
                ),
                "macro_auroc_ovr": float(
                    macro_auroc
                ),
            },
            f,
            indent=2,
        )


def save_aggregate_results(
    run_dir: str,
    summary_rows,
    all_true,
    all_pred,
    all_prob,
    class_names,
    num_classes: int,
    samples_per_day: int,
    folds_to_run,
    eval_only: bool = False,
):
    """
    Save fold summary, aggregate confusion matrices,
    aggregate AUROC, and final metrics.
    """
    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values("fold")

    summary_name = (
        "10fold_eval_summary.csv"
        if eval_only
        else "10fold_summary.csv"
    )

    summary_df.to_csv(
        os.path.join(
            run_dir,
            summary_name,
        ),
        index=False,
    )

    prefix = (
        "eval_"
        if eval_only
        else ""
    )

    y_true_all = np.concatenate(
        all_true
    )

    y_pred_all = np.concatenate(
        all_pred
    )

    y_prob_all = np.concatenate(
        all_prob,
        axis=0,
    )

    aggregate_cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=list(
            range(num_classes)
        ),
    )

    expected_row_sum = (
        samples_per_day
        * len(folds_to_run)
    )

    row_sums = aggregate_cm.sum(
        axis=1
    )

    if not np.all(
        row_sums == expected_row_sum
    ):
        raise RuntimeError(
            "Aggregate confusion-matrix row sums "
            f"are {row_sums.tolist()}, "
            f"expected {expected_row_sum} samples per class."
        )

    np.save(
        os.path.join(
            run_dir,
            prefix
            + "aggregate_confusion_matrix.npy",
        ),
        aggregate_cm,
    )

    pd.DataFrame(
        aggregate_cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        os.path.join(
            run_dir,
            prefix
            + "aggregate_confusion_matrix_count.csv",
        )
    )

    aggregate_cm_normalized = (
        aggregate_cm.astype(float)
        / np.maximum(
            aggregate_cm.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    pd.DataFrame(
        aggregate_cm_normalized,
        index=class_names,
        columns=class_names,
    ).to_csv(
        os.path.join(
            run_dir,
            prefix
            + "aggregate_confusion_matrix_normalized.csv",
        )
    )

    save_confusion_matrix(
        aggregate_cm,
        class_names,
        os.path.join(
            run_dir,
            prefix
            + "aggregate_confusion_matrix_count.png",
        ),
        normalize=False,
        title="Aggregated 10-fold test confusion matrix",
    )

    save_confusion_matrix(
        aggregate_cm,
        class_names,
        os.path.join(
            run_dir,
            prefix
            + "aggregate_confusion_matrix_normalized.png",
        ),
        normalize=True,
        title="Aggregated 10-fold test confusion matrix",
    )

    overall_acc = accuracy_score(
        y_true_all,
        y_pred_all,
    )

    overall_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=list(
            range(num_classes)
        ),
        output_dict=True,
        zero_division=0,
    )

    (
        aggregate_macro_auroc,
        aggregate_per_class_auroc,
    ) = compute_multiclass_auroc(
        y_true=y_true_all,
        y_prob=y_prob_all,
        num_classes=num_classes,
    )

    pd.DataFrame(
        {
            "class_index": list(
                range(num_classes)
            ),
            "class_name": class_names,
            "auroc_ovr": (
                aggregate_per_class_auroc
            ),
        }
    ).to_csv(
        os.path.join(
            run_dir,
            prefix
            + "aggregate_auroc_per_class.csv",
        ),
        index=False,
    )

    save_roc_curve_points(
        y_true=y_true_all,
        y_prob=y_prob_all,
        class_names=class_names,
        out_csv=os.path.join(
            run_dir,
            prefix
            + "aggregate_roc_curves.csv",
        ),
    )

    def safe_sd(series):
        if len(series) <= 1:
            return 0.0
        return float(
            series.std(ddof=1)
        )

    final_metrics = {
        "n_test_predictions": int(
            len(y_true_all)
        ),
        "expected_n_test_predictions": int(
            samples_per_day
            * len(folds_to_run)
            * num_classes
        ),
        "aggregate_accuracy": float(
            overall_acc
        ),
        "aggregate_macro_f1": float(
            overall_report[
                "macro avg"
            ]["f1-score"]
        ),
        "aggregate_macro_auroc_ovr": float(
            aggregate_macro_auroc
        ),
        "fold_accuracy_mean": float(
            summary_df[
                "test_acc"
            ].mean()
        ),
        "fold_accuracy_sd": safe_sd(
            summary_df["test_acc"]
        ),
        "fold_macro_f1_mean": float(
            summary_df[
                "test_macro_f1"
            ].mean()
        ),
        "fold_macro_f1_sd": safe_sd(
            summary_df[
                "test_macro_f1"
            ]
        ),
        "fold_macro_auroc_ovr_mean": float(
            summary_df[
                "test_macro_auroc_ovr"
            ].mean()
        ),
        "fold_macro_auroc_ovr_sd": safe_sd(
            summary_df[
                "test_macro_auroc_ovr"
            ]
        ),
    }

    final_metrics_csv_name = (
        "eval_final_metrics.csv"
        if eval_only
        else "final_metrics.csv"
    )

    pd.DataFrame(
        [final_metrics]
    ).to_csv(
        os.path.join(
            run_dir,
            final_metrics_csv_name,
        ),
        index=False,
    )

    final_metrics_name = (
        "eval_final_metrics.json"
        if eval_only
        else "final_metrics.json"
    )

    with open(
        os.path.join(
            run_dir,
            final_metrics_name,
        ),
        "w",
    ) as f:
        json.dump(
            final_metrics,
            f,
            indent=2,
        )

    return (
        summary_df,
        final_metrics,
    )
