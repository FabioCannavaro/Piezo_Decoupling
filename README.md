# Piezo Paper Classification Pipeline — Clean Version

This folder keeps only the current paper-reproduction pipeline.

## Files

### `main_paper_reproduction.py`
Raw CSV -> filtered Morse CWT spectrogram -> `.npy`

- target sampling rate: 50 Hz
- strain band-pass: 0.1–15 Hz
- temperature low-pass: 1 Hz
- CWT output: 96 frequency bins × 256 time bins
- default: 2 channels (strain X, strain Y)
- `--include_temp`: 3 channels (strain X, strain Y, temperature)

Example:

```bash
python main_paper_reproduction.py \
    --input_dir data300 \
    --output_dir output_300 \
    --include_temp
```

---

### `spectrogram_dataset.py`
PyTorch dataset + normalization + augmentation.

Training behavior is preserved:

1. Z-score normalization using fold-specific training statistics
2. built-in lightweight time/frequency masking when `augment=True`
3. optional strong SpecAugment
4. optional Gaussian noise

---

### `train.py`
Shared functions used by `train_paper_10fold.py` only.

Contains:

- training-only normalization statistics
- dataset subset construction
- ConvNeXtV2-Tiny model
- MixUp
- AdamW + warmup/cosine scheduler
- backbone freeze/unfreeze
- AMP training loop
- training curve plotting

The old random StratifiedKFold + fixed holdout training code was removed.

---

### `train_paper_10fold.py`
Current paper-matched day-based 10-fold training/evaluation runner.

Default split for 10 measurement days:

- Fold 1: test Day 1, validation Day 2, train Day 3–10
- Fold 2: test Day 2, validation Day 3, train remaining days
- ...
- Fold 10: test Day 10, validation Day 1, train remaining days

Default paper training settings are unchanged:

- ConvNeXtV2 Tiny (`convnextv2_tiny.fcmae`)
- epochs: 200
- batch size: 16
- learning rate: 3e-5
- head dropout: 0.3
- noise std: 0.03
- strong SpecAugment: ON
- MixUp: ON, alpha 0.4
- label smoothing: 0.05
- WeightedRandomSampler
- AMP FP16
- TF32

Full 10-fold:

```bash
python train_paper_10fold.py --root output_300
```

Pilot Fold 1 only:

```bash
python train_paper_10fold.py --root output_300 --folds 1
```

Evaluate existing checkpoints only:

```bash
python train_paper_10fold.py \
    --root output_300 \
    --out_root runs/YOUR_EXISTING_RUN \
    --eval_only
```

## Recommended workflow

```text
Raw CSV
   ↓
main_paper_reproduction.py
   ↓
.npy CWT spectrograms
   ↓
spectrogram_dataset.py
   ↓
train.py (shared model/training functions)
   ↓
train_paper_10fold.py
   ↓
fold checkpoints + confusion matrices + metrics
```

## What was removed

From the old `train.py`:

- random `StratifiedKFold` training entry point
- random fixed holdout-test construction
- old augmentation provenance helpers used only by that pipeline
- duplicate strong SpecAugment definition
- unused validation-confusion helper
- old CLI / `__main__` training runner

These were not used by the current paper-matched day-based 10-fold script.
