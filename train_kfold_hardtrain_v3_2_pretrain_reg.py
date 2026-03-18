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

from torch.utils.data import Dataset

import timm
import torch.nn as nn

class AugmentWrapper(Dataset):
    def __init__(self, base_ds, noise_std=0.0, strong_specaugment=False):
        # base는 반드시 object.__setattr__로 넣어서 attribute lookup 꼬임 방지
        object.__setattr__(self, "base", base_ds)
        self.noise_std = float(noise_std)
        self.strong_specaugment = bool(strong_specaugment)

        # ✅ 학습 코드에서 필요한 속성은 "명시적으로" 노출
        self.labels = getattr(base_ds, "labels", None)
        self.image_paths = getattr(base_ds, "image_paths", None)
        self.groups = getattr(base_ds, "groups", None)
        self.class_to_idx = getattr(base_ds, "class_to_idx", None)
        self.root = getattr(base_ds, "root", None)
        self.stats = getattr(base_ds, "stats", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.strong_specaugment:
            x = _strong_spec_mask(x)
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y


# --------------------- utils ---------------------

def nowstamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_aug_path(p: str) -> bool:
    return "__aug" in os.path.basename(p)


@torch.no_grad()
def compute_stats_from_paths(train_paths):
    if not train_paths:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

    # ✅ 첫 번째 파일을 읽어 채널 수(C) 자동 확인
    first_arr = np.load(train_paths[0])
    # npy shape: (C, H, W)
    C = first_arr.shape[0] if first_arr.ndim == 3 else 1

    ch_sum = torch.zeros(C, dtype=torch.float64)
    ch_sqsum = torch.zeros(C, dtype=torch.float64)
    n_px = 0

    for p in train_paths:
        arr = np.load(p)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.from_numpy(arr)  # (C,H,W)
        
        # 채널별 합계 계산
        # t가 (C, H, W)이므로 (C, -1)로 펼침
        ch = t.reshape(C, -1).double()
        
        ch_sum += ch.sum(dim=1)
        ch_sqsum += (ch ** 2).sum(dim=1)
        n_px += t.shape[1] * t.shape[2] # H * W

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


# --------------------- dataset wrappers (no file edit) ---------------------

def _strong_spec_mask(x: torch.Tensor, time_frac=(0.08, 0.25), freq_frac=(0.08, 0.25), p=0.8):
    """
    Stronger SpecAugment on already z-scored x (3,H,W).
    - time/freq masks bigger than your default.
    """
    if np.random.rand() > p:
        return x
    _, H, W = x.shape
    # time mask
    if np.random.rand() < 0.8:
        w = int(np.random.uniform(time_frac[0], time_frac[1]) * W)
        w = max(1, min(W, w))
        t0 = np.random.randint(0, max(1, W - w + 1))
        x[:, :, t0:t0+w] = 0
    # freq mask
    if np.random.rand() < 0.8:
        h = int(np.random.uniform(freq_frac[0], freq_frac[1]) * H)
        h = max(1, min(H, h))
        f0 = np.random.randint(0, max(1, H - h + 1))
        x[:, f0:f0+h, :] = 0
    # mild gain jitter
    if np.random.rand() < 0.4:
        g = 1.0 + np.random.uniform(-0.12, 0.12)
        x = x * g
    return x


def make_subset_from_indices(root, base_stats, indices, augment=False,
                            noise_std=0.0, strong_specaugment=False):
    ds = SpectrogramDataset(root, stats=base_stats, augment=augment, index_json=None)
    ds.image_paths = [ds.image_paths[i] for i in indices]
    ds.labels = ds.labels[indices]
    if hasattr(ds, "groups"):
        ds.groups = [ds.groups[i] for i in indices]

    if augment and (noise_std > 0.0 or strong_specaugment):
        ds = AugmentWrapper(ds, noise_std=noise_std, strong_specaugment=strong_specaugment)

    return ds




# --------------------- model (pretrained 유지) ---------------------



def build_model(num_classes: int, in_chans: int = 3, head_dropout: float = 0.3):
    model = timm.create_model(
        "convnextv2_tiny.fcmae",
        pretrained=True,
        num_classes=0,
        in_chans=in_chans,  # ✅ 입력 채널 수 설정 추가
    )

    in_features = model.num_features

    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.LayerNorm(in_features),
        nn.Dropout(p=head_dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# --------------------- mixup ---------------------

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], float(lam)


def mixup_loss(criterion, logits, y_a, y_b, lam: float):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


@torch.no_grad()
def mixup_expected_acc(pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float):
    pa = (pred == y_a).float()
    pb = (pred == y_b).float()
    return float((lam * pa + (1 - lam) * pb).mean().item())


# --------------------- criteria / schedule ---------------------

def label_smoothing_for_epoch(epoch: int, total_epochs: int,
                              ls_max=0.1, decay_until=0.8):
    """
    Strong LS early -> 0 near the end.
    """
    t = epoch / max(1, total_epochs - 1)
    return ls_max if t < decay_until else 0.0


def build_optimizer_and_scheduler(model, base_lr, epochs, freeze_epochs=5, weight_decay=1e-4):
    # freeze backbone first
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head.parameters():
        p.requires_grad = True

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
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    remain = max(1, epochs - current_epoch - 1)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.2, end_factor=1.0, total_iters=1)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remain)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1])
    return opt, sched


# --------------------- loops ---------------------

def epoch_loop(model, loader, criterion, device,
               train_mode=True, optimizer=None, max_grad=5.0,
               use_mixup=True, mixup_alpha=0.4):
    model.train(mode=train_mode)
    total_loss = 0.0
    total_seen = 0
    acc_sum = 0.0

    for x, y in loader:
        x = torch.nan_to_num(x.to(device, non_blocking=True), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            if train_mode and use_mixup and mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
                logits = model(x)
                loss = mixup_loss(criterion, logits, y_a, y_b, lam)
                pred = logits.argmax(dim=1)
                batch_acc = mixup_expected_acc(pred, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)
                pred = logits.argmax(dim=1)
                batch_acc = float((pred == y).float().mean().item())

            if train_mode:
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
                        p.grad.clamp_(-max_grad, max_grad)
                optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        acc_sum += batch_acc * bs
        total_seen += bs

    return total_loss / max(1, total_seen), acc_sum / max(1, total_seen)


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
    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training Curves (Accuracy)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_acc_png, dpi=200); plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training Curves (Loss)')
    plt.grid(True, linestyle=':'); plt.legend(); plt.tight_layout()
    plt.savefig(out_loss_png, dpi=200); plt.close()


# --------------------- training ---------------------

def train_kfold(root: str,
                batch_size: int = 16,
                epochs: int = 300,
                lr: float = 3e-5,
                freeze_epochs: int = 5,
                augment: bool = True,
                k_folds: int = 5,
                test_per_class: int = 40,
                test_with_aug: bool = False,
                out_root: str = None,
                seed: int = 42,
                # regularization knobs
                head_dropout: float = 0.3,
                label_smoothing_max: float = 0.05,
                noise_std: float = 0.01,
                strong_specaugment: bool = True,
                mixup_alpha: float = 0.4,
                use_mixup: bool = True):

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ds_full = SpectrogramDataset(root, augment=False, stats=None, index_json=None)
    all_paths = ds_full.image_paths
    labels = ds_full.labels.clone()
    num_classes = int(labels.max()) + 1
    idx_to_class = build_idx_to_class(ds_full, labels)

    run_dir = out_root or os.path.join("runs", f"{nowstamp()}_convnext_tiny_input_xy")
    print("📁 K-Fold results will be saved under:", run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print_per_class_counts(all_paths, labels, num_classes, idx_to_class, title_prefix="   ")

    # holdout test
    test_idx, remain_idx = build_holdout_test(all_paths, labels,
                                              per_class=test_per_class,
                                              include_aug=test_with_aug,
                                              seed=seed)
    holdout_items = []
    groups = getattr(ds_full, "groups", None)
    for i in test_idx:
        cls_idx = int(labels[i])
        cls_name = idx_to_class[cls_idx]
        rel_path = os.path.relpath(all_paths[i], root)
        group = groups[i] if groups is not None else "g0"
        holdout_items.append({"path": rel_path, "class": cls_name, "group": group, "label_idx": cls_idx})

    holdout_json = os.path.join(run_dir, "index_holdout_test.json")
    with open(holdout_json, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "per_class": test_per_class, "include_aug": test_with_aug, "items": holdout_items},
                  f, ensure_ascii=False, indent=2)
    print(f"📄 Saved holdout test index -> {holdout_json}")

    # remain pool (orig-only stratify)
    rem_paths = [all_paths[i] for i in remain_idx]
    rem_labels = labels[remain_idx]
    is_orig_mask = np.array([not _is_aug_path(p) for p in rem_paths], dtype=bool)
    local_idx = np.arange(len(rem_paths))
    orig_local = local_idx[is_orig_mask]
    aug_local  = local_idx[~is_orig_mask]
    y_orig = rem_labels[orig_local].numpy()

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    meta = {
        "root": root,
        "k_folds": k_folds,
        "seed": seed,
        "test_per_class": test_per_class,
        "test_with_aug": test_with_aug,
        "num_classes": int(num_classes),
        "device": str(device),
        "regularization": {
            "head_dropout": head_dropout,
            "label_smoothing_max": label_smoothing_max,
            "noise_std": noise_std,
            "strong_specaugment": strong_specaugment,
            "mixup_alpha": mixup_alpha,
            "use_mixup": use_mixup,
            "freeze_epochs": freeze_epochs
        }
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for fold_id, (tr_loc, va_loc) in enumerate(skf.split(orig_local, y_orig), start=1):
        print(f"\n========== Fold {fold_id}/{k_folds} ==========")

        tr_global_orig = [remain_idx[orig_local[i]] for i in tr_loc]
        va_global_orig = [remain_idx[orig_local[i]] for i in va_loc]
        tr_global_aug  = [remain_idx[i] for i in aug_local.tolist()]
        train_idx = np.array(tr_global_orig + tr_global_aug, dtype=int)
        val_idx   = np.array(va_global_orig, dtype=int)

        print("📊 Fold dataset sizes:")
        print(f"   Train: orig={len(tr_global_orig)}, aug={len(tr_global_aug)}, total={len(train_idx)}")
        print(f"   Val  : orig={len(val_idx)}, aug=0, total={len(val_idx)}")

        # stats from TRAIN only
        train_paths = [all_paths[i] for i in train_idx]
        stats = compute_stats_from_paths(train_paths)

        # datasets/loaders
        train_ds = make_subset_from_indices(
            root, stats, train_idx,
            augment=True,
            noise_std=noise_std,
            strong_specaugment=strong_specaugment
        )
        val_ds = make_subset_from_indices(
            root, stats, val_idx,
            augment=False,
            noise_std=0.0,
            strong_specaugment=False
        )

        cls_counts = torch.bincount(train_ds.labels, minlength=num_classes)
        class_weights = (cls_counts.sum() / (cls_counts + 1e-6)).float()
        sample_weights = class_weights[train_ds.labels]
        sampler = WeightedRandomSampler(weights=sample_weights,
                                        num_samples=len(train_ds),
                                        replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, drop_last=False, prefetch_factor=2)

        # model (pretrained)
        input_channels = len(stats['mean'])
        model = build_model(num_classes, in_chans=input_channels, head_dropout=head_dropout).to(device)
        optimizer, scheduler, warm_freeze = build_optimizer_and_scheduler(
            model, base_lr=lr, epochs=epochs, freeze_epochs=freeze_epochs, weight_decay=1e-4
        )

        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
        with open(epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr","label_smoothing"])

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

            tr_loss, tr_acc = epoch_loop(
                model, train_loader, train_crit, device,
                train_mode=True, optimizer=optimizer, max_grad=5.0,
                use_mixup=use_mixup, mixup_alpha=mixup_alpha
            )
            va_loss, va_acc = epoch_loop(
                model, val_loader, val_crit, device,
                train_mode=False, optimizer=None,
                use_mixup=False, mixup_alpha=0.0
            )

            scheduler.step()
            cur_lr = optimizer.param_groups[0]['lr']

            print(f"[KFold {fold_id}] [Epoch {epoch+1}/{epochs}] "
                  f"Train {tr_loss:.4f}/{tr_acc:.3f} | Val {va_loss:.4f}/{va_acc:.3f} "
                  f"| LR {cur_lr:.2e} | LS {ls:.3f}")

            with open(epoch_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch+1, tr_loss, tr_acc, va_loss, va_acc, cur_lr, ls])

            history.append({
                "epoch": epoch+1,
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
                "lr": float(cur_lr),
                "label_smoothing": float(ls),
            })

            if (va_acc > best_val_acc) and torch.isfinite(torch.tensor(va_loss)):
                best_val_acc = va_acc
                save_path = os.path.join(fold_dir, "best_model.pth")
                torch.save({
                    "model_state": model.state_dict(),
                    "best_val_acc": float(best_val_acc),
                    "stats": stats,
                    "class_to_idx": getattr(train_ds, "class_to_idx", None),
                    "meta": {
                        "head_dropout": head_dropout,
                        "label_smoothing_max": label_smoothing_max,
                        "noise_std": noise_std,
                        "strong_specaugment": strong_specaugment,
                        "mixup_alpha": mixup_alpha,
                        "use_mixup": use_mixup,
                        "freeze_epochs": freeze_epochs
                    }
                }, save_path)
                print(f"✅ Saved best model to {fold_dir} (acc={best_val_acc:.3f})")

        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "training_metrics.csv"), index=False)

        try:
            plot_fold_curves(epoch_csv,
                             os.path.join(fold_dir, "training_curves_acc.png"),
                             os.path.join(fold_dir, "training_curves_loss.png"))
        except Exception as e:
            print("[WARN] plot_fold_curves failed:", e)

        try:
            save_val_confmat_png(model, val_loader, num_classes,
                                 os.path.join(fold_dir, "confusion_matrix_val.png"), device)
        except Exception as e:
            print("[WARN] save_val_confmat_png failed:", e)

    # ------------------ HOLDOUT TEST EVAL ------------------
    print("\n===== EVAL: HOLDOUT TEST =====")
    rem_train_paths = [all_paths[i] for i in remain_idx]
    test_stats = compute_stats_from_paths(rem_train_paths)

    test_ds = make_subset_from_indices(root, test_stats, test_idx, augment=False,
                                       noise_std=0.0, strong_specaugment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    for fold_id in range(1, k_folds + 1):
        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location=device)

        model = build_model(num_classes, head_dropout=head_dropout).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        ys, ps = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = torch.nan_to_num(x.to(device), nan=0.0, posinf=0.0, neginf=0.0)
                y = y.to(device)
                pred = model(x).argmax(1)
                ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
        y_true = np.concatenate(ys); y_pred = np.concatenate(ps)

        acc = accuracy_score(y_true, y_pred)  # Top-1 acc
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        rep = classification_report(y_true, y_pred, labels=list(range(num_classes)),
                                    output_dict=True, zero_division=0)

        np.save(os.path.join(fold_dir, "test_confusion_matrix.npy"), cm)
        with open(os.path.join(fold_dir, "test_classification_report.json"), "w") as f:
            json.dump(rep, f, indent=2)
        with open(os.path.join(fold_dir, "test_acc.txt"), "w") as f:
            f.write(f"test_acc={acc:.6f}\n")

        print(f"[Fold {fold_id}] test_acc={acc:.4f}")

    # Aggregate summary CSV
    rows = []
    for fold_id in range(1, k_folds + 1):
        fold_dir = os.path.join(run_dir, f"fold{fold_id}")
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), map_location="cpu")
        best_val = ckpt.get("best_val_acc", None)

        test_acc = None
        p = os.path.join(fold_dir, "test_acc.txt")
        if os.path.exists(p):
            with open(p, "r") as f:
                for line in f:
                    if "test_acc=" in line:
                        test_acc = float(line.strip().split("=")[-1])

        macro_f1 = None
        rep_path = os.path.join(fold_dir, "test_classification_report.json")
        if os.path.exists(rep_path):
            with open(rep_path, "r") as f:
                rep = json.load(f)
            if "macro avg" in rep:
                macro_f1 = rep["macro avg"].get("f1-score", None)

        rows.append({"fold": fold_id, "best_val_acc": best_val, "test_acc": test_acc, "test_macro_f1": macro_f1})

    df = pd.DataFrame(rows).sort_values("fold")
    df.to_csv(os.path.join(run_dir, "training_metrics.csv"), index=False)

    print("Done.")
    print("Saved to:", run_dir)


# --------------------- CLI entry ---------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='input_data_300_xy_2026')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--freeze_epochs', type=int, default=5)
    ap.add_argument('--augment', action='store_true')
    ap.add_argument('--k_folds', type=int, default=5)
    ap.add_argument('--test_per_class', type=int, default=40)
    ap.add_argument('--test_with_aug', action='store_true')
    ap.add_argument('--out_root', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)

    # reg knobs
    ap.add_argument('--head_dropout', type=float, default=0.3)
    ap.add_argument('--label_smoothing_max', type=float, default=0.1)
    ap.add_argument('--noise_std', type=float, default=0.03)
    ap.add_argument('--no_strong_specaug', action='store_true')
    ap.add_argument('--mixup_alpha', type=float, default=0.4)
    ap.add_argument('--no_mixup', action='store_true')

    args = ap.parse_args()

    train_kfold(
        root=args.root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        augment=args.augment,
        k_folds=args.k_folds,
        test_per_class=args.test_per_class,
        test_with_aug=args.test_with_aug,
        out_root=args.out_root,
        seed=args.seed,
        head_dropout=args.head_dropout,
        label_smoothing_max=args.label_smoothing_max,
        noise_std=args.noise_std,
        strong_specaugment=(not args.no_strong_specaug),
        mixup_alpha=args.mixup_alpha,
        use_mixup=(not args.no_mixup),
    )
