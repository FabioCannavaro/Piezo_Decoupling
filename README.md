# Hardware-encoded modality separation: analysis code

This repository contains the classification, external evaluation, benchmarking, and real-time Python code associated with the article **“Hardware-encoded modality separation for multimodal sensing and artificial perception.”** Experimental data are archived on Zenodo at **[10.5281/zenodo.22972999](https://doi.org/10.5281/zenodo.22972999)**. Place the downloaded dataset under `data/` in the repository root. A software release DOI will be added when assigned.

## Dataset layout

```text
data/
  raw/                    # raw measurements by class
  MOTORIZED/              # motorized measurements
  odd/                    # contact-position measurements
  processed_centered_3ch/ # three-channel CWT .npy inputs
  processed_center_2ch/   # two-channel CWT .npy inputs
  processed_1ch/          # one-channel CWT .npy inputs
  processed_MOT/          # motorized processed .npy inputs
```

Keep the original class folders and filenames. `src/dataset.py` loads `index.json` when present, or scans class directories and infers a group from each filename prefix. Each processed sample has shape `(channels, 96, 256)`. The code below consumes the processed arrays directly; do not remove them from the archived dataset until raw-to-processed reproduction has been verified.

## Install and run

Use Python 3.10+ and install a CUDA-enabled PyTorch build suitable for your system. Then install the direct Python dependencies listed in `requirements.txt`. Package versions are not pinned to the original experimental environment. Run these commands from the repository root.

```bash
python -m pip install -r requirements.txt
```

## Preprocess raw CSV files

`scripts/preprocess_raw.py` converts the raw CSV measurements into CWT tensors with shape `(channels, 96, 256)` and writes an `index.json` file. It detects `25X` and `25Y` as the two strain channels and `25W` as the temperature channel. The strain channels are median-centered and band-pass filtered before the Morse CWT; the temperature channel is low-pass filtered without median centering.

Generate the three-channel inputs used for the main classification analysis:

```bash
python -m scripts.preprocess_raw \
  --input_dir data/raw \
  --output_dir data/processed_centered_3ch \
  --include_temp
```

Generate the two-channel strain-only inputs:

```bash
python -m scripts.preprocess_raw \
  --input_dir data/raw \
  --output_dir data/processed_center_2ch
```

Generate the one-channel ablation inputs using the first strain channel:

```bash
python -m scripts.preprocess_raw_1ch \
  --input_dir data/raw \
  --output_dir data/processed_1ch
```

The processed motorized and contact-position test inputs used for the reported analyses are included in the archived dataset. Until exact raw-to-processed equivalence has been verified for every analysis branch, use the archived processed arrays to reproduce the reported results.

## Train and evaluate

```bash
python -m scripts.train_10fold --root data/processed_centered_3ch --no_strong_specaug --no_mixup
python -m scripts.train_10fold --root data/processed_center_2ch --no_strong_specaug --no_mixup
python -m scripts.train_10fold --root data/processed_1ch --no_strong_specaug --no_mixup
```

`scripts/train_10fold.py` holds out one measurement day for testing and the next day for validation. The defaults enable strong SpecAugment and MixUp; the commands above disable both as in the later experiments. Verify which run and flags produced each result in the final manuscript. For a single fold add `--folds 1`; to reevaluate saved checkpoints use `--out_root PATH_TO_RUN --eval_only`. Training requires CUDA.

```bash
python -m scripts.train_data_fraction --root data/processed_centered_3ch --no_strong_specaug --no_mixup
python -m scripts.test_motorized --motor_root data/processed_MOT --run_dir PATH_TO_RUN
python -m scripts.benchmark_model --root data/processed_centered_3ch --run_dir PATH_TO_RUN --fold 1
```

`PATH_TO_RUN` is a training output directory containing `foldN/best_model.pth`. Checkpoints are not included in this repository. The position test expects processed `<class>/<position>/*.npy` under `--ood_root`. If `data/odd` holds raw CSVs, preprocess and validate them before running:

```bash
python -m scripts.ood_test_final --ood_root PATH_TO_PROCESSED_POSITION_DATA --run_dir PATH_TO_RUN --train_root data/processed_centered_3ch
```

`real_time/main.py` provides a CSV-to-CWT utility for the real-time demonstration. For the paper data pipeline, use `scripts/preprocess_raw.py`. `real_time/realtime_server.py` needs a compatible trained checkpoint. Firmware and Android app source are not included here.

## Archiving

Before making a GitHub Release for Zenodo, verify the paper's reported results against saved run outputs and record the corresponding tag, run settings, and data DOI. Archive raw and processed inputs together unless you have verified exact regeneration; include a dataset README specifying formats, units, filenames, and the relationship of each folder to the figures and tables.
