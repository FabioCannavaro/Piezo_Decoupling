import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None, stats=None, augment=False, index_json=None):
        self.root = image_dir
        self.transform = transform  # (미사용) torchvision 변환 대신 z-score/증강 사용
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
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
            for it in items:
                self.image_paths.append(os.path.join(image_dir, it['path']))
                self.labels.append(self.class_to_idx[it['class']])
                self.groups.append(it.get('group', 'g0'))
        else:
            classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
            for c in classes:
                cdir = os.path.join(image_dir, c)
                for name in os.listdir(cdir):
                    if name.endswith('.npy'):
                        self.image_paths.append(os.path.join(cdir, name))
                        self.labels.append(self.class_to_idx[c])
                        self.groups.append(name.split('_')[0])

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # dataset-wide stats
        if stats is None:
            ch_sum = torch.zeros(3)
            ch_sqsum = torch.zeros(3)
            n_px = 0
            for p in self.image_paths:
                arr = np.load(p)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                t = torch.from_numpy(arr)
                n = t.shape[1]*t.shape[2]
                ch_sum += t.view(3, -1).sum(dim=1)
                ch_sqsum += (t.view(3, -1)**2).sum(dim=1)
                n_px += n
            mean = ch_sum / max(1, n_px)
            var = (ch_sqsum / max(1, n_px)) - mean**2
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            self.stats = {'mean': mean.tolist(), 'std': std.tolist()}
        else:
            self.stats = stats
        C = len(self.stats['mean'])
        self.mean = torch.tensor(self.stats['mean']).float().view(C,1,1)
        self.std  = torch.tensor(self.stats['std']).float().view(C,1,1)

    def __len__(self):
        return len(self.image_paths)

    def _spec_augment(self, x: torch.Tensor):
        # x: 3×H×W
        H, W = x.shape[1], x.shape[2]
        # time mask
        if random.random() < 0.5:
            w = max(1, random.randint(W//32, max(2, W//8)))
            t0 = random.randint(0, max(0, W-w))
            x[:,:,t0:t0+w] = 0
        # freq mask
        if random.random() < 0.5:
            h = max(1, random.randint(H//32, max(2, H//8)))
            f0 = random.randint(0, max(0, H-h))
            x[:,f0:f0+h,:] = 0
        # small gain jitter
        if random.random() < 0.3:
            g = 1.0 + random.uniform(-0.1, 0.1)
            x *= g
        return x

    def __getitem__(self, idx):
        arr = np.load(self.image_paths[idx])      # 3×H×W
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(arr)                 # float32
        x = (x - self.mean) / (self.std + 1e-8)   # z-score (dataset-wide)
        if self.augment:
            x = self._spec_augment(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, self.labels[idx]
