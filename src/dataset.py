"""PyTorch dataset and augmentation utilities for CWT spectrogram tensors."""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Load preprocessed ``.npy`` spectrograms.

    Expected sample shape:
        (C, H, W)

    The dataset applies dataset-level Z-score normalization using supplied
    statistics. During training, the original lightweight augmentation stage
    can be enabled with ``augment=True``.
    """

    def __init__(
        self,
        image_dir,
        transform=None,
        stats=None,
        augment=False,
        index_json=None,
    ):
        self.root = image_dir
        self.transform = transform  # Kept for compatibility; not used in paper pipeline.
        self.augment = augment

        self.image_paths = []
        self.labels = []
        self.groups = []
        self.class_to_idx = {}

        self._load_index(index_json)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        self.stats = stats if stats is not None else self._compute_dataset_stats()

        channels = len(self.stats["mean"])
        self.mean = torch.tensor(self.stats["mean"]).float().view(channels, 1, 1)
        self.std = torch.tensor(self.stats["std"]).float().view(channels, 1, 1)

    def _load_index(self, index_json=None):
        """Load file paths/classes from index.json or scan class directories."""
        idx_path = index_json or os.path.join(self.root, "index.json")

        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            classes = sorted({item["class"] for item in items})
            self.class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

            for item in items:
                self.image_paths.append(os.path.join(self.root, item["path"]))
                self.labels.append(self.class_to_idx[item["class"]])
                self.groups.append(item.get("group", "g0"))
            return

        classes = sorted(
            directory
            for directory in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, directory))
        )
        self.class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            for name in os.listdir(class_dir):
                if not name.endswith(".npy"):
                    continue

                self.image_paths.append(os.path.join(class_dir, name))
                self.labels.append(self.class_to_idx[class_name])
                self.groups.append(name.split("_")[0])

    def _compute_dataset_stats(self):
        """Compute per-channel statistics when no external stats are supplied.

        ``train_paper_10fold.py`` later recomputes statistics using training-only
        paths for each fold. This fallback behavior is kept identical to the
        previous implementation.
        """
        if not self.image_paths:
            return {"mean": [0.0], "std": [1.0]}

        first_arr = np.load(self.image_paths[0])
        channels = first_arr.shape[0] if first_arr.ndim == 3 else 1

        ch_sum = torch.zeros(channels, dtype=torch.float64)
        ch_sqsum = torch.zeros(channels, dtype=torch.float64)
        n_px = 0

        for path in self.image_paths:
            arr = np.load(path)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            tensor = torch.from_numpy(arr)
            flat = tensor.view(channels, -1).double()

            ch_sum += flat.sum(dim=1)
            ch_sqsum += (flat**2).sum(dim=1)
            n_px += tensor.shape[1] * tensor.shape[2]

        mean = (ch_sum / max(1, n_px)).float()
        var = (ch_sqsum / max(1, n_px)).float() - mean**2
        std = torch.sqrt(torch.clamp(var, min=1e-8))

        return {"mean": mean.tolist(), "std": std.tolist()}

    def __len__(self):
        return len(self.image_paths)

    def _spec_augment(self, x: torch.Tensor):
        """Original lightweight time/frequency masking + gain jitter."""
        _, height, width = x.shape

        # Time mask
        if random.random() < 0.5:
            mask_width = max(
                1,
                random.randint(width // 32, max(2, width // 8)),
            )
            start = random.randint(0, max(0, width - mask_width))
            x[:, :, start : start + mask_width] = 0

        # Frequency mask
        if random.random() < 0.5:
            mask_height = max(
                1,
                random.randint(height // 32, max(2, height // 8)),
            )
            start = random.randint(0, max(0, height - mask_height))
            x[:, start : start + mask_height, :] = 0

        # Small gain jitter
        if random.random() < 0.3:
            gain = 1.0 + random.uniform(-0.1, 0.1)
            x *= gain

        return x

    def __getitem__(self, idx):
        arr = np.load(self.image_paths[idx])
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.from_numpy(arr)

        # Dataset-wide Z-score normalization using fold-specific stats when supplied.
        x = (x - self.mean) / (self.std + 1e-8)

        if self.augment:
            x = self._spec_augment(x)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, self.labels[idx]


# -----------------------------------------------------------------------------
# Strong augmentation used by paper training
# -----------------------------------------------------------------------------


def _strong_spec_mask(
    x: torch.Tensor,
    time_frac=(0.08, 0.25),
    freq_frac=(0.08, 0.25),
    p=0.8,
):
    """Apply the original strong SpecAugment configuration."""
    if np.random.rand() > p:
        return x

    _, height, width = x.shape

    if np.random.rand() < 0.8:
        mask_width = int(np.random.uniform(time_frac[0], time_frac[1]) * width)
        mask_width = max(1, min(width, mask_width))
        start = np.random.randint(0, max(1, width - mask_width + 1))
        x[:, :, start : start + mask_width] = 0

    if np.random.rand() < 0.8:
        mask_height = int(np.random.uniform(freq_frac[0], freq_frac[1]) * height)
        mask_height = max(1, min(height, mask_height))
        start = np.random.randint(0, max(1, height - mask_height + 1))
        x[:, start : start + mask_height, :] = 0

    if np.random.rand() < 0.4:
        gain = 1.0 + np.random.uniform(-0.12, 0.12)
        x = x * gain

    return x


class AugmentWrapper(Dataset):
    """Add strong SpecAugment and Gaussian noise around a base dataset."""

    def __init__(self, base_ds, noise_std=0.0, strong_specaugment=False):
        object.__setattr__(self, "base", base_ds)
        self.noise_std = float(noise_std)
        self.strong_specaugment = bool(strong_specaugment)

        # Expose metadata expected by training code.
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
