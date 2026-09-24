import os
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import models

from spectrogram_dataset import SpectrogramDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import csv
import matplotlib.pyplot as plt
import pandas as pd


# --------------------- utils ---------------------

def nowstamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_aug_path(p: str) -> bool:
    return "__aug" in os.path.basename(p)


@torch.no_grad()
def compute_stats_from_paths(train_paths):
    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sqsum = torch.zeros(3, dtype=torch.float64)
    n_px = 0
    for p in train_paths:
        arr = np.load(p)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.from_numpy(arr)  # (3,H,W)
        _, h, w = t.shape
        n = h * w
        ch = t.reshape(3, -1).double()
        ch_sum += ch.sum(dim=1)
        ch_sqsum += (ch ** 2).sum(dim=1)
        n_px += n
    mean = (ch_sum / max(1, n_px)).float()
    var = (ch_sqsum / max(1, n_px)).float() - mean**2
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    return {"mean": mean.tolist(), "std": std.tolist()}


def build_idx_to_class(ds_full, labels_tensor):
    if hasattr(ds_full, "class_names") and ds_full.class_names:
        names = ds_full.class_names
        return {i: names[i] for i in range(len(names))}
    if hasattr(ds_full, "class_to_idx") and ds_full.class_to_idx:
        inv = {v: k for k, v in ds_full.class_to_idx.items()}
        return {i: inv.get(i, str(i)) for i in range(int(labels_tensor.max()) + 1)}
    return {i: str(i) for i in range(int(labels_tensor.max()) + 1)}


def print_per_class_counts(all_paths, labels, num_classes, idx_to_class, title_prefix=""):
    per_class_orig = defaultdict(int); per_class_aug = defaultdict(int)
    for i, p in enumerate(all_paths):
        (per_class_aug if _is_aug_path(p) else per_class_orig)[int(labels[i])] += 1
    print(f"{title_prefix}Per-class originals:",
          {idx_to_class[c]: per_class_orig[c] for c in range(num_classes)})
    print(f"{title_prefix}Per-class aug      :",
          {idx_to_class[c]: per_class_aug[c]  for c in range(num_classes)})


def build_holdout_test(all_paths, all_labels, per_class=40, include_aug=False, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(all_labels)
    paths = np.asarray(all_paths)

    classes = sorted(np.unique(labels).tolist())

    cls_to_orig = {c: [] for c in classes}
    cls_to_aug  = {c: [] for c in classes}
    for i, (p, y) in enumerate(zip(paths, labels)):
        (cls_to_aug if _is_aug_path(p) else cls_to_orig)[int(y)].append(i)

    test_idx = []
    for c in classes:
        idxs = cls_to_orig[c][:]
        rng.shuffle(idxs)
        pick = idxs[:per_class]
        if len(pick) < per_class:
            print(f"[WARN] Class {c} has only {len(pick)} originals (requested {per_class}).")
        test_idx.extend(pick)

    if include_aug:
        for c in classes:
            test_idx.extend(cls_to_aug[c])

    test_set = set(test_idx)
    remain_idx = [i for i in range(len(paths)) if i not in test_set]
    return test_idx, remain_idx


def make_subset_from_indices(root, base_stats, indices, augment=False):
    ds = SpectrogramDataset(root, stats=base_stats, augment=augment, index_json=None)
    ds.image_paths = [ds.image_paths[i] for i in indices]
    ds.labels = ds.labels[indices]
    if hasattr(ds, "groups"):
        ds.groups = [ds.groups[i] for i in indices]
    return ds


def build_model(num_classes: int):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model


def build_optimizer_and_scheduler(model, base_lr, epochs, freeze_epochs=5, weight_decay=1e-4):
    for p in model.features.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=base_lr, weight_decay=weight_decay)

    warmup_epochs = max(1, min(5, freeze_epochs))
    main_epochs = max(1, epochs - warmup_epochs)

    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0,
                                               total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_epochs)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine],
                                                  milestones=[warmup_epochs])
    return opt, sched, warmup_epochs


def unfreeze_backbone_and_reset_opt(model, current_epoch, epochs, base_lr, weight_decay=1e-4):
    for p in model.features.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    remain = max(1, epochs - current_epoch - 1)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=1)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1])
    return opt, sched


def make_criteria(epoch_idx, total_epochs):
    sharp_phase = (epoch_idx >= int(total_epochs * 0.7))
    ls = 0.0 if sharp_phase else 0.01
    train_criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    val_criterion   = nn.CrossEntropyLoss()
    return train_criterion, val_criterion


def epoch_loop(model, loader, criterion, device, train_mode=True, optimizer=None, max_grad=5.0):
    model.train(mode=train_mode)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                        p.grad.clamp_(-max_grad, max_grad)
                optimizer.step()

        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_seen += y.numel()
        total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / max(1, total_seen)
    avg_acc = total_correct / max(1, total_seen)
    return avg_loss, avg_acc


@torch.no_grad()
def save_val_confmat_png(model, val_loader, num_classes, out_png, device):
    model.eval()
    ys, ps = [], []
    for x, y in val_loader:
        x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.to(device)
        pred = model(x).argmax(1)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure(figsize=(6,5))
    im = plt.imshow(cmn, aspect='auto')
    plt.title('Val Confusion Matrix (normalized)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()


def plot_fold_curves(csv_path, out_acc_png, out_loss_png):
    df = pd.read_csv(csv_path)
    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training Curves (Accuracy)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_acc_png, dpi=200); plt.close()

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Curves (Loss)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_loss_png, dpi=200); plt.close()


def plot_aggregate_curves(run_dir, k_folds, out_acc_png, out_loss_png):
    dfs = []
    for fold_id in range(1, k_folds+1):
        csv_path = os.path.join(run_dir, f"fold{fold_id}", "epoch_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            dfs.append(df)
    if not dfs:
        return
    # Align by min length
    min_len = min(len(df) for df in dfs)
    acc_tr = np.stack([df['train_acc'].values[:min_len] for df in dfs], axis=0)
    acc_va = np.stack([df['val_acc'].values[:min_len] for df in dfs], axis=0)
    ls_tr  = np.stack([df['train_loss'].values[:min_len] for df in dfs], axis=0)
    ls_va  = np.stack([df['val_loss'].values[:min_len] for df in dfs], axis=0)
    epochs = np.arange(1, min_len+1)

    # Accuracy (train & val on same fig)
    plt.figure(figsize=(8,5))
    for y, label in [(acc_tr, 'train_acc'), (acc_va, 'val_acc')]:
        mean = y.mean(axis=0); std = y.std(axis=0)
        plt.plot(epochs, mean, label=f'{label} (mean)')
        plt.fill_between(epochs, mean-std, mean+std, alpha=0.15)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training Curves (Accuracy, mean ± std)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_acc_png, dpi=200); plt.close()

    # Loss (train & val)
    plt.figure(figsize=(8,5))
    for y, label in [(ls_tr, 'train_loss'), (ls_va, 'val_loss')]:
        mean = y.mean(axis=0); std = y.std(axis=0)
        plt.plot(epochs, mean, label=f'{label} (mean)')
        plt.fill_between(epochs, mean-std, mean+std, alpha=0.15)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curves (Loss, mean ± std)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_loss_png, dpi=200); plt.close()


# --------------------- training (as function) ---------------------

def train_kfold(root: str,
                batch_size: int = 32,
                epochs: int = 200,
                lr: float = 3e-5,
                freeze_epochs: int = 5,
                augment: bool = True,
                k_folds: int = 5,
                test_per_class: int = 40,
                test_with_aug: bool = False,
                out_root: str = None,
                seed: int = 42):

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load once without augment to get full paths/labels
    ds_full = SpectrogramDataset(root, augment=False, stats=None, index_json=None)
    all_paths = ds_full.image_paths
    labels = ds_full.labels.clone()
    num_classes = int(labels.max()) + 1
    idx_to_class = build_idx_to_class(ds_full, labels)

    print("📁 K-Fold results will be saved under:", end=" ")
    run_dir = out_root or os.path.join("runs", f"{nowstamp()}_convnext_tiny_kfold_hard_input_all_300")
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    # Print per-class counts (overall)
    print_per_class_counts(all_paths, labels, num_classes, idx_to_class, title_prefix="   ")

    # Build holdout test
    test_idx, remain_idx = build_holdout_test(all_paths, labels,
                                              per_class=test_per_class,
                                              include_aug=test_with_aug,
                                              seed=seed)
    
    holdout_items = []
    groups = getattr(ds_full, "groups", None)
    for i in test_idx:
        cls_idx = int(labels[i])
        cls_name = idx_to_class[cls_idx]       # ex) "Idle", "Writing" ...
        rel_path = os.path.relpath(all_paths[i], root)
        group = groups[i] if groups is not None else "g0"
        holdout_items.append({
            "path": rel_path,
            "class": cls_name,
            "group": group,
            "label_idx": cls_idx,
        })

    holdout_json = os.path.join(run_dir, "index_holdout_test.json")
    with open(holdout_json, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "per_class": test_per_class,
            "include_aug": test_with_aug,
            "items": holdout_items,
        }, f, ensure_ascii=False, indent=2)
    print(f"📄 Saved holdout test index -> {holdout_json}")

    # Prepare remain pool
    rem_paths = [all_paths[i] for i in remain_idx]
    rem_labels = labels[remain_idx]
    is_orig_mask = np.array([not _is_aug_path(p) for p in rem_paths], dtype=bool)
    local_idx = np.arange(len(rem_paths))

    orig_local = local_idx[is_orig_mask]
    aug_local  = local_idx[~is_orig_mask]

    y_orig = rem_labels[orig_local].numpy()
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_val_accs = []
    meta = {
        "root": root,
        "k_folds": k_folds,
        "seed": seed,
        "test_per_class": test_per_class,
        "test_with_aug": test_with_aug,
        "num_classes": int(num_classes),
        "device": str(device),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for fold_id, (tr_loc, va_loc) in enumerate(skf.split(orig_local, y_orig), start=1):
        print(f"\n========== Fold {fold_id}/{k_folds} ==========")

        # Map back to global remaining indices
        tr_global_orig = [remain_idx[orig_local[i]] for i in tr_loc]
        va_global_orig = [remain_idx[orig_local[i]] for i in va_loc]
        tr_global_aug  = [remain_idx[i] for i in aug_local.tolist()]
        train_idx = np.array(tr_global_orig + tr_global_aug, dtype=int)
        val_idx   = np.array(va_global_orig, dtype=int)

        # Sizes
        n_tr_orig = len(tr_global_orig)
        n_tr_aug  = len(tr_global_aug)
        n_va_orig = len(val_idx)
        print("📊 Fold dataset sizes:")
        print(f"   Train: orig={n_tr_orig}, aug={n_tr_aug}, total={n_tr_orig + n_tr_aug}")
        print(f"   Val  : orig={n_va_orig}, aug=0, total={n_va_orig}  (val_aug는 항상 0)")

        # Stats from TRAIN paths only
        train_paths = [all_paths[i] for i in train_idx]
        stats = compute_stats_from_paths(train_paths)

        # Datasets / Loaders
        train_ds = make_subset_from_indices(root, stats, train_idx, augment=True)
        val_ds   = make_subset_from_indices(root, stats, val_idx,   augment=False)

        # Weighted sampler
        cls_counts = torch.bincount(train_ds.labels, minlength=num_classes)
        class_weights = (cls_counts.sum() / (cls_counts + 1e-6)).float()
        sample_weights = class_weights[train_ds.labels]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(train_ds),
                                        replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, drop_last=False)

        # Model / Opt / Sched
        model = build_model(num_classes).to(device)
        optimizer, scheduler, warm_freeze = build_optimizer_and_scheduler(
            model, base_lr=lr, epochs=epochs, freeze_epochs=freeze_epochs, weight_decay=1e-4
        )

        # History + Epoch CSV (epoch_metrics) + also your training_metrics.csv
        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
        with open(epoch_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])
        history = []

        best_val_acc = -1.0

        for epoch in range(epochs):
            if epoch == warm_freeze:
                optimizer, scheduler = unfreeze_backbone_and_reset_opt(
                    model, current_epoch=epoch, epochs=epochs, base_lr=lr, weight_decay=1e-4
                )

            train_crit, val_crit = make_criteria(epoch, epochs)
            tr_loss, tr_acc = epoch_loop(model, train_loader, train_crit, device,
                                         train_mode=True, optimizer=optimizer, max_grad=5.0)
            va_loss, va_acc = epoch_loop(model, val_loader,   val_crit,   device,
                                         train_mode=False, optimizer=None)

            scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']

            print(f"[KFold {fold_id}] [Epoch {epoch+1}/{epochs}] "
                  f"Train {tr_loss:.4f}/{tr_acc:.3f} | Val {va_loss:.4f}/{va_acc:.3f} "
                  f"| LR {cur_lr:.2e}")

            with open(epoch_csv, "a", newline="") as f:
                w = csv.writer(f); w.writerow([epoch+1, tr_loss, tr_acc, va_loss, va_acc, cur_lr])

            history.append({
                'epoch': epoch + 1,
                'train_loss': float(tr_loss),
                'train_acc': float(tr_acc),
                'val_loss': float(va_loss),
                'val_acc': float(va_acc),
                'lr': float(cur_lr),
            })

            if (va_acc > best_val_acc) and torch.isfinite(torch.tensor(va_loss)):
                best_val_acc = va_acc
                save_path = os.path.join(fold_dir, 'best_model.pth')
                torch.save({
                    'model_state': model.state_dict(),
                    'class_to_idx': getattr(train_ds, 'class_to_idx', None),
                    'best_val_acc': float(best_val_acc),
                    'stats': stats,
                }, save_path)
                print(f"✅ Saved best model to {fold_dir} (acc={best_val_acc:.3f})")

            # Small late-stage tweak (optional)
            if epoch == int(epochs * 0.9):
                for g in optimizer.param_groups:
                    g['lr'] = min(g['lr'], 1e-6)
                    g['weight_decay'] = 0.0

        # Save fold-level history in your filename too
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "training_metrics.csv"), index=False)

        # Per-fold curve plots from epoch_metrics.csv
        try:
            plot_fold_curves(epoch_csv,
                             os.path.join(fold_dir, "training_curves_acc.png"),
                             os.path.join(fold_dir, "training_curves_loss.png"))
        except Exception as e:
            print("[WARN] plot_fold_curves failed:", e)

        # Also save val confusion matrix PNG using the best model already in memory
        try:
            save_val_confmat_png(model, val_loader, num_classes,
                                 os.path.join(fold_dir, "confusion_matrix_val.png"), device)
        except Exception as e:
            print("[WARN] save_val_confmat_png failed:", e)

    # --------- Evaluate per-fold models on HOLDOUT TEST ----------
    print("\n===== EVAL: HOLDOUT TEST =====")
    rem_train_paths = [all_paths[i] for i in remain_idx]
    test_stats = compute_stats_from_paths(rem_train_paths)

    test_ds = make_subset_from_indices(root, test_stats, test_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    per_fold_test = {}
    for fold_id in range(1, k_folds + 1):
        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location=device)

        model = build_model(num_classes).to(device)
        model.load_state_dict(ckpt['model_state'])  # use your key
        model.eval()

        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                y = y.to(device)
                pred = model(x).argmax(1)
                ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys); y_pred = np.concatenate(ps)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        rep = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                    output_dict=True, zero_division=0)
        np.save(os.path.join(fold_dir, "test_confusion_matrix.npy"), cm)
        with open(os.path.join(fold_dir, "test_classification_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        with open(os.path.join(fold_dir, "test_acc.txt"), "w") as f:
            f.write(f"test_acc={acc:.6f}\n")
        per_fold_test[f"fold{fold_id}"] = {"acc": acc}

    # --------- Aggregate reports at top-level ----------
    # 1) summary CSV
    rows = []
    for fold_id in range(1, k_folds + 1):
        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        # read best and test
        def read_float_from_txt(path, key):
            if not os.path.exists(path): return None
            with open(path,'r') as f:
                for line in f:
                    if key in line:
                        try: return float(line.strip().split('=')[-1])
                        except: pass
            return None
        best_val = read_float_from_txt(os.path.join(fold_dir, "best.txt"), "best_val_acc")
        # v3.1 no longer writes best.txt; recover from checkpoint
        if best_val is None:
            ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location='cpu')
            best_val = ckpt.get('best_val_acc', None)
        test_acc = read_float_from_txt(os.path.join(fold_dir, "test_acc.txt"), "test_acc")
        # macro F1
        rep_path = os.path.join(fold_dir, "test_classification_report.json")
        macro_f1 = None
        if os.path.exists(rep_path):
            with open(rep_path, "r") as f:
                rep = json.load(f)
                if 'macro avg' in rep:
                    macro_f1 = rep['macro avg'].get('f1-score', None)
        rows.append({"fold": fold_id, "best_val_acc": best_val, "test_acc": test_acc, "test_macro_f1": macro_f1})
    df = pd.DataFrame(rows).sort_values('fold')
    df.to_csv(os.path.join(run_dir, "training_metrics.csv"), index=False)

    # 2) aggregate curves (mean±std across folds)
    try:
        # Rebuild curves from per-fold epoch_metrics.csv
        dfs = []
        for fold_id in range(1, k_folds+1):
            csv_path = os.path.join(run_dir, f"fold{fold_id}", "epoch_metrics.csv")
            if os.path.exists(csv_path):
                df_fold = pd.read_csv(csv_path)
                dfs.append(df_fold)
        if dfs:
            min_len = min(len(df) for df in dfs)
            acc_tr = np.stack([df['train_acc'].values[:min_len] for df in dfs], axis=0)
            acc_va = np.stack([df['val_acc'].values[:min_len] for df in dfs], axis=0)
            ls_tr  = np.stack([df['train_loss'].values[:min_len] for df in dfs], axis=0)
            ls_va  = np.stack([df['val_loss'].values[:min_len] for df in dfs], axis=0)
            epochs = np.arange(1, min_len+1)

            # acc
            plt.figure(figsize=(8,5))
            for y, label in [(acc_tr, 'train_acc'), (acc_va, 'val_acc')]:
                mean = y.mean(axis=0); std = y.std(axis=0)
                plt.plot(epochs, mean, label=f'{label} (mean)')
                plt.fill_between(epochs, mean-std, mean+std, alpha=0.15)
            plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Training Curves (Accuracy, mean ± std)')
            plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "training_curves_acc.png"), dpi=200); plt.close()

            # loss
            plt.figure(figsize=(8,5))
            for y, label in [(ls_tr, 'train_loss'), (ls_va, 'val_loss')]:
                mean = y.mean(axis=0); std = y.std(axis=0)
                plt.plot(epochs, mean, label=f'{label} (mean)')
                plt.fill_between(epochs, mean-std, mean+std, alpha=0.15)
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Curves (Loss, mean ± std)')
            plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "training_curves_loss.png"), dpi=200); plt.close()
    except Exception as e:
        print("[WARN] aggregate curves failed:", e)

    print("Done.")
    print("Saved to:", run_dir)


# --------------------- CLI entry ---------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='input_data_300')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--freeze_epochs', type=int, default=5)
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--test_per_class', type=int, default=40)
    ap.add_argument('--test_with_aug', action='store_true')
    ap.add_argument('--out_root', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train_kfold(root=args.root,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                freeze_epochs=args.freeze_epochs,
                augment=args.augment,
                k_folds=args.k_folds,
                test_per_class=args.test_per_class,
                test_with_aug=args.test_with_aug,
                out_root=args.out_root,
                seed=args.seed)
