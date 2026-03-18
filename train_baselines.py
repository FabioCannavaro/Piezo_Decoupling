"""
Baseline Training Script
Trains baseline classification models (ResNet, EfficientNet, MobileNet, ViT) 
ensuring the identical Holdout Test Set used by 'Ours' is strictly excluded to prevent data leakage.
"""

import os
import json
import argparse
import traceback
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from sklearn.model_selection import StratifiedKFold

# Import utilities from the main training script
from spectrogram_dataset import SpectrogramDataset
from train import epoch_loop, compute_stats_from_paths, make_subset_from_indices

MODELS_TO_TRAIN = [
    "resnet18",             
    "efficientnet_b0",      
    "mobilenetv3_large_100",
    "regnety_008",          
    "vit_base_patch16_224"  
]

def get_classifier_layer(model):
    if isinstance(model, nn.Sequential):
        model = model[-1]

    if hasattr(model, 'fc'): return model.fc
    elif hasattr(model, 'classifier'): return model.classifier
    elif hasattr(model, 'head'):
        if hasattr(model.head, 'fc'): return model.head.fc
        return model.head
    return None

def build_optimizer_for_baseline(model, lr, epochs, freeze_epochs=0):
    for p in model.parameters(): p.requires_grad = False
    
    classifier = get_classifier_layer(model)
    if classifier:
        for p in classifier.parameters(): p.requires_grad = True
    else:
        for p in model.parameters(): p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    warmup_epochs = max(1, freeze_epochs)
    main_epochs = max(1, epochs - warmup_epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=main_epochs)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_epochs])
    return opt, sched, warmup_epochs

def unfreeze_all_params(model, base_lr):
    for p in model.parameters(): p.requires_grad = True
    return torch.optim.AdamW(model.parameters(), lr=base_lr)

def train_single_fold_baseline(model_name, root, test_json_path, epochs, batch_size, lr, seed, device):
    print(f"\n========================================================")
    print(f"Training Baseline: {model_name}")
    print(f"========================================================")
    
    # Save directory
    save_dir = os.path.join("runs", "baselines", model_name, "fold1")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Load full dataset
    ds_full = SpectrogramDataset(root, augment=False)
    all_paths = ds_full.image_paths
    
    # 2. Exclude Holdout Test Set completely (Strict matching)
    with open(test_json_path, 'r') as f:
        test_info = json.load(f)
    
    abs_test_paths = set(os.path.normpath(os.path.abspath(os.path.join(root, item['path']))) for item in test_info['items'])
        
    train_candidate_indices = []
    excluded_count = 0
    
    for i, p in enumerate(all_paths):
        norm_p = os.path.normpath(os.path.abspath(p))
        if norm_p not in abs_test_paths:
            train_candidate_indices.append(i)
        else:
            excluded_count += 1
            
    if excluded_count == 0:
        print("Warning: No files excluded. Check your JSON path matching.")
        
    print(f"   Original Total: {len(all_paths)} | Excluded Test: {excluded_count} | Training Pool: {len(train_candidate_indices)}")
    
    candidate_indices = np.array(train_candidate_indices)
    candidate_paths = [all_paths[i] for i in candidate_indices]
    candidate_labels = ds_full.labels[candidate_indices]

    # 3. Stratified Split (Training / Validation)
    is_orig = np.array([not "__aug" in os.path.basename(p) for p in candidate_paths])
    local_indices = np.arange(len(candidate_indices))
    orig_local_idx = local_indices[is_orig]
    orig_labels = candidate_labels[orig_local_idx]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_loc, va_loc = list(skf.split(orig_local_idx, orig_labels))[0] # Only running the 1st Fold
    
    tr_orig_global = candidate_indices[orig_local_idx[tr_loc]]
    va_orig_global = candidate_indices[orig_local_idx[va_loc]]
    aug_global = candidate_indices[local_indices[~is_orig]]
    
    train_idx = np.concatenate([tr_orig_global, aug_global])
    val_idx = va_orig_global
    
    # 4. Dataset & Dataloaders
    train_paths_final = [all_paths[i] for i in train_idx]
    stats = compute_stats_from_paths(train_paths_final)
    
    train_ds = make_subset_from_indices(root, stats, train_idx, augment=True)
    val_ds = make_subset_from_indices(root, stats, val_idx, augment=False)
    
    cls_counts = torch.bincount(train_ds.labels, minlength=8)
    weights = (cls_counts.sum() / (cls_counts + 1e-6)).float()
    sampler = WeightedRandomSampler(weights[train_ds.labels], len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 5. Model Initialization
    model = timm.create_model(model_name, pretrained=True, num_classes=8, in_chans=len(stats['mean']))
    
    if "vit" in model_name:
        print(f"   👉 [Info] Adding Resize((224,224)) wrapper for ViT")
        model = nn.Sequential(T.Resize((224, 224), antialias=True), model)
    
    model.to(device)
    
    # 6. Optimization Loop
    freeze_epochs = 5
    optimizer, scheduler, warm_freeze = build_optimizer_for_baseline(model, lr, epochs, freeze_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(epochs):
        if epoch == warm_freeze:
            print(f"🔓 Unfreezing backbone at epoch {epoch+1}...")
            optimizer = unfreeze_all_params(model, lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch)

        tr_loss, tr_acc = epoch_loop(model, train_loader, criterion, device, train_mode=True, optimizer=optimizer)
        va_loss, va_acc = epoch_loop(model, val_loader, criterion, device, train_mode=False)
        
        scheduler.step()
        print(f"[{model_name}] Ep {epoch+1}/{epochs} | Tr: {tr_acc:.3f} | Va: {va_acc:.3f} | Best: {best_acc:.3f}")
        
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                'model_state': model.state_dict(),
                'best_acc': best_acc,
                'model_name': model_name
            }, os.path.join(save_dir, "best_model.pth"))
            
    print(f"✅ {model_name} Finished. Best Val Acc: {best_acc:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument('--root_data', type=str, required=True, help="Path to the training dataset directory")
    parser.add_argument('--test_json', type=str, required=True, help="Path to the holdout test index JSON file used by the main model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for m_name in MODELS_TO_TRAIN:
        save_path = os.path.join("runs", "baselines", m_name, "fold1", "best_model.pth")
        if os.path.exists(save_path):
            print(f"⏭️  Skipping {m_name} (Already exists at {save_path})")
            continue
        
        try:
            train_single_fold_baseline(
                model_name=m_name, root=args.root_data, test_json_path=args.test_json, 
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=device
            )
        except Exception as e:
            print(f"❌ Error training {m_name}: {e}")
            traceback.print_exc()