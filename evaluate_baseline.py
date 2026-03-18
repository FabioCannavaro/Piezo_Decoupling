"""
Evaluate and Compare Models (ROC-AUC)
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
# Update 'path' with the actual checkpoint paths for your experiments
MODELS = [
    {
        "name": "Ours (ConvNeXt-V2)",
        "path": "runs/ours_model/fold3/best_model.pth", 
        "arch": "convnextv2_tiny.fcmae",
        "in_chans": 3,
        "is_ours": True 
    },
    {
        "name": "ResNet-18",
        "path": "runs/baselines/resnet18/fold1/best_model.pth",
        "arch": "resnet18",
        "in_chans": 3,
        "is_ours": False
    },
    {
        "name": "EfficientNet-B0",
        "path": "runs/baselines/efficientnet_b0/fold1/best_model.pth",
        "arch": "efficientnet_b0",
        "in_chans": 3,
        "is_ours": False
    },
    {
        "name": "MobileNetV3",
        "path": "runs/baselines/mobilenetv3_large_100/fold1/best_model.pth",
        "arch": "mobilenetv3_large_100",
        "in_chans": 3,
        "is_ours": False
    },
    {
        "name": "ViT-Base",
        "path": "runs/baselines/vit_base_patch16_224/fold1/best_model.pth",
        "arch": "vit_base_patch16_224",
        "in_chans": 3,
        "is_ours": False
    },
]

CLASS_NAMES = ['clipper', 'earbud', 'eraser', 'finger', 'hotpeltier', 'peltier', 'pen', 'sponge']


def load_trained_model(info, num_classes, device):
    print(f"Loading {info['name']}...")
    
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

        if "vit" in info['arch']:
            print(f"   👉 Wrapping {info['name']} with Resize((224,224)) to match checkpoint...")
            model = nn.Sequential(
                T.Resize((224, 224), antialias=True),
                model
            )

    if not os.path.exists(info['path']):
        print(f"❌ Error: Model path not found -> {info['path']}")
        return None

    ckpt = torch.load(info['path'], map_location=device)
    state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"⚠️ Strict load failed for {info['name']}, trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    model.input_channels = info['in_chans']
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate and Compare Models (ROC-AUC)")
    parser.add_argument('--root_data', type=str, required=True, help="Path to the test dataset directory")
    parser.add_argument('--test_json', type=str, required=True, help="Path to the holdout test index JSON file")
    parser.add_argument('--out_png', type=str, default="AUROC_Comparison.png", help="Path to save the ROC curve plot")
    parser.add_argument('--out_csv', type=str, default="AUROC_Origin_Data.csv", help="Path to save FPR/TPR data for Origin")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.test_json):
        print(f"❌ Holdout Test JSON not found: {args.test_json}")
        return

    ds_full = SpectrogramDataset(args.root_data, augment=False)
    
    with open(args.test_json, 'r') as f:
        test_info = json.load(f)
    
    test_paths_set = set([item['path'].replace("\\", "/") for item in test_info['items']])
    test_indices = [i for i, p in enumerate(ds_full.image_paths) 
                    if os.path.relpath(p, args.root_data).replace("\\", "/") in test_paths_set]
            
    if not test_indices:
        print("❌ No matching test files found in the dataset.")
        return

    ds_test = Subset(ds_full, test_indices)
    loader = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=4)
    print(f"📊 Test Set Size: {len(ds_test)} samples")
    
    y_true = np.array([y.numpy() for _, y in loader for y in y])
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))

    plt.figure(figsize=(10, 8))
    origin_data = {}
    max_len = 0

    for info in MODELS:
        model = load_trained_model(info, len(CLASS_NAMES), device)
        if model is None: continue
        
        y_scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device).float()
                
                # Handle channel mismatch gracefully
                in_chans = getattr(model, 'input_channels', 3)
                if in_chans == 2 and x.shape[1] == 3:
                    x = x[:, :2, :, :]
                    
                probs = torch.softmax(model(x), dim=1)
                y_scores.extend(probs.cpu().numpy())
        
        y_scores = np.array(y_scores)
        auc_score = roc_auc_score(y_true_bin, y_scores, average='micro', multi_class='ovr')
        
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        plt.plot(fpr, tpr, lw=2, label=f"{info['name']} (AUC = {auc_score:.4f})")
        print(f"   👉 {info['name']} Evaluation Complete.")

        # Data collection for CSV
        name_clean = info['name'].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        origin_data[f"{name_clean}_FPR"] = fpr
        origin_data[f"{name_clean}_TPR"] = tpr
        max_len = max(max_len, len(fpr))

    # Save to CSV
    padded_data = {k: np.concatenate([v, np.full(max_len - len(v), np.nan)]) for k, v in origin_data.items()}
    pd.DataFrame(padded_data).to_csv(args.out_csv, index=False)
    print(f"✅ Extracted Origin CSV data saved to: {args.out_csv}")

    # Save Plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison on Holdout Test Set', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=300)
    print(f"✅ Plotted Graph saved to: {args.out_png}")


if __name__ == "__main__":
    main()