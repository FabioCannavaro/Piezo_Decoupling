import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None, stats=None, augment=False, index_json=None):
        self.root = image_dir
        self.transform = transform  
        self.augment = augment

        self.image_paths = []
        self.labels = []
        self.groups = []
        self.class_to_idx = {}

        idx_path = index_json or os.path.join(image_dir, 'index.json')
        if os.path.exists(idx_path):
            with open(idx_path, 'r') as f:
                items = json.load(f)
            classes = sorted(list({it['class'] for it in items}))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for it in items:
                self.image_paths.append(os.path.join(image_dir, it['path']))
                self.labels.append(self.class_to_idx[it['class']])
                self.groups.append(it.get('group', 'g0'))
        else:
            classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
            for c in classes:
                cdir = os.path.join(image_dir, c)
                for name in os.listdir(cdir):
                    if name.endswith('.npy'):
                        self.image_paths.append(os.path.join(cdir, name))
                        self.labels.append(self.class_to_idx[c])
                        self.groups.append(name.split('_')[0])

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        if stats is None and len(self.image_paths) > 0:
            first_arr = np.load(self.image_paths[0])
            C = first_arr.shape[0] if first_arr.ndim == 3 else 1

            ch_sum = torch.zeros(C, dtype=torch.float64)
            ch_sqsum = torch.zeros(C, dtype=torch.float64)
            n_px = 0
            
            for p in self.image_paths:
                arr = np.load(p)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.from_numpy(arr)
                n = t.shape[1] * t.shape[2]
                
                ch_sum += t.view(C, -1).double().sum(dim=1)
                ch_sqsum += (t.view(C, -1).double() ** 2).sum(dim=1)
                n_px += n
                
            mean = (ch_sum / max(1, n_px)).float()
            var = (ch_sqsum / max(1, n_px)).float() - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            self.stats = {'mean': mean.tolist(), 'std': std.tolist()}
        elif stats is not None:
            self.stats = stats
        else:
            self.stats = {'mean': [0.0], 'std': [1.0]}

        C = len(self.stats['mean'])
        self.mean = torch.tensor(self.stats['mean']).float().view(C, 1, 1)
        self.std  = torch.tensor(self.stats['std']).float().view(C, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def _spec_augment(self, x: torch.Tensor):
        # x shape: (C, H, W)
        H, W = x.shape[1], x.shape[2]
        
        # Time mask
        if random.random() < 0.5:
            w = max(1, random.randint(W // 32, max(2, W // 8)))
            t0 = random.randint(0, max(0, W - w))
            x[:, :, t0:t0+w] = 0
            
        # Frequency mask
        if random.random() < 0.5:
            h = max(1, random.randint(H // 32, max(2, H // 8)))
            f0 = random.randint(0, max(0, H - h))
            x[:, f0:f0+h, :] = 0
            
        # Small gain jitter
        if random.random() < 0.3:
            g = 1.0 + random.uniform(-0.1, 0.1)
            x *= g
            
        return x

    def __getitem__(self, idx):
        arr = np.load(self.image_paths[idx])      # (C, H, W)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(arr)                 # float32
        
        # Z-score normalization (dataset-wide)
        x = (x - self.mean) / (self.std + 1e-8)   
        
        if self.augment:
            x = self._spec_augment(x)
            
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, self.labels[idx]

# -------------------------------------------------------------
# Strong SpecAugment 및 AugmentWrapper
# -------------------------------------------------------------

def _strong_spec_mask(x: torch.Tensor, time_frac=(0.08, 0.25), freq_frac=(0.08, 0.25), p=0.8):
    if np.random.rand() > p:
        return x
    _, H, W = x.shape
    if np.random.rand() < 0.8:
        w = int(np.random.uniform(time_frac[0], time_frac[1]) * W)
        w = max(1, min(W, w))
        t0 = np.random.randint(0, max(1, W - w + 1))
        x[:, :, t0:t0+w] = 0
    if np.random.rand() < 0.8:
        h = int(np.random.uniform(freq_frac[0], freq_frac[1]) * H)
        h = max(1, min(H, h))
        f0 = np.random.randint(0, max(1, H - h + 1))
        x[:, f0:f0+h, :] = 0
    if np.random.rand() < 0.4:
        g = 1.0 + np.random.uniform(-0.12, 0.12)
        x = x * g
    return x

class AugmentWrapper(Dataset):
    def __init__(self, base_ds, noise_std=0.0, strong_specaugment=False):
        object.__setattr__(self, "base", base_ds)
        self.noise_std = float(noise_std)
        self.strong_specaugment = bool(strong_specaugment)

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