"""
Evaluate and Compare Models (ROC-AUC) with K-Fold Ensemble
Averages the prediction probabilities across all K folds for a fair and robust evaluation.
Generates ROC curve plots and extracts FPR/TPR data to CSV for external plotting tools (e.g., Origin).
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

from spectrogram_dataset import SpectrogramDataset

# ==========================================
# Configuration: Models to Compare
# ==========================================
MODELS = [
    {
        "name": "Ours (ConvNeXt-V2)",
        "dir": "runs/ours_model",  # change
        "arch": "convnextv2_tiny.fcmae",
        "in_chans": 3,
        "is_ours": True 
    },
    {
        "name": "ResNet-18",
        "dir": "runs/baselines/resnet18",
        "arch": "resnet18",
        "in_chans": 3,
        "is_ours": False
    },
    {
        "name": "EfficientNet-B0",
        "dir": "runs/baselines/efficientnet_b0",
        "arch": "efficientnet_b0",
        "in_chans": 3,
        "is_ours": False
    },
    {
        "name": "MobileNetV3",
        "dir": "runs/baselines/mobilenetv3_large_100",
        "arch": "mobilenetv3_large_100",
        "in_chans": 3,
        "is_ours": False
    },
]

CLASS_NAMES = ['clipper', 'earbud', 'eraser', 'finger', 'hotpeltier', 'peltier', 'pen', 'sponge']


def load_trained_model(info, fold_idx, num_classes, device):
    ckpt_path = os.path.join(info['dir'], f"fold{fold_idx}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        return None

    if info['is_ours']:
        model = timm.create_model(info['arch'], pretrained=False, num_classes=0, in_chans=info['in_chans'])
        in_features = model.num_features
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
    else:
        model = timm.create_model(info['arch'], pretrained=False, num_classes=num_classes, in_chans=info['in_chans'])


    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.input_channels = info['in_chans']
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Models using K-Fold Ensemble (ROC-AUC)")
    parser.add_argument('--root_data', type=str, required=True, help="Path to the test dataset directory")
    parser.add_argument('--test_json', type=str, required=True, help="Path to the holdout test index JSON file")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds to ensemble")
    parser.add_argument('--out_png', type=str, default="AUROC_Comparison_Ensemble.png", help="Path to save the ROC curve plot")
    parser.add_argument('--out_csv', type=str, default="AUROC_Origin_Data_Ensemble.csv", help="Path to save FPR/TPR data for Origin")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.test_json):
        print(f"Holdout Test JSON not found: {args.test_json}")
        return

    ds_full = SpectrogramDataset(args.root_data, augment=False)
    
    with open(args.test_json, 'r') as f:
        test_info = json.load(f)
    
    test_paths_set = set([item['path'].replace("\\", "/") for item in test_info['items']])
    test_indices = [i for i, p in enumerate(ds_full.image_paths) 
                    if os.path.relpath(p, args.root_data).replace("\\", "/") in test_paths_set]
            
    if not test_indices:
        print("No matching test files found in the dataset.")
        return

    ds_test = Subset(ds_full, test_indices)
    loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test Set Size: {len(ds_test)} samples")
    
    y_true = np.array([y.numpy() for _, y in loader for y in y])
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))

    plt.figure(figsize=(10, 8))
    origin_data = {}
    max_len = 0

    for info in MODELS:
        print(f"\n--- Evaluating {info['name']} (Ensembling {args.k_folds} Folds) ---")
        
        all_folds_probs = []
        
        for fold_idx in range(1, args.k_folds + 1):
            model = load_trained_model(info, fold_idx, len(CLASS_NAMES), device)
            if model is None:
                print(f" Fold {fold_idx} checkpoint not found. Skipping this fold.")
                continue
            
            fold_probs = []
            with torch.no_grad():
                for x, _ in loader:
                    x = x.to(device).float()
                    in_chans = getattr(model, 'input_channels', 3)
                    if in_chans == 2 and x.shape[1] == 3:
                        x = x[:, :2, :, :]
                        
                    probs = torch.softmax(model(x), dim=1)
                    fold_probs.extend(probs.cpu().numpy())
            
            all_folds_probs.append(fold_probs)
            print(f" Fold {fold_idx} evaluated.")

        if not all_folds_probs:
            print(f"No trained folds found for {info['name']}. Skipping model.")
            continue
            
        ensembled_probs = np.mean(all_folds_probs, axis=0)
        
        auc_score = roc_auc_score(y_true_bin, ensembled_probs, average='micro', multi_class='ovr')
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), ensembled_probs.ravel())
        
        plt.plot(fpr, tpr, lw=2, label=f"{info['name']} (AUC = {auc_score:.4f})")
        print(f" {info['name']} Ensemble Complete (AUC: {auc_score:.4f})")

        name_clean = info['name'].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        origin_data[f"{name_clean}_FPR"] = fpr
        origin_data[f"{name_clean}_TPR"] = tpr
        max_len = max(max_len, len(fpr))

    padded_data = {k: np.concatenate([v, np.full(max_len - len(v), np.nan)]) for k, v in origin_data.items()}
    pd.DataFrame(padded_data).to_csv(args.out_csv, index=False)
    print(f"\n Extracted Origin CSV data saved to: {args.out_csv}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison (5-Fold Ensemble)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"Plotted Graph saved to: {args.out_png}")


if __name__ == "__main__":
    main()