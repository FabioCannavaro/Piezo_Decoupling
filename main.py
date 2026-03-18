"""
Data Preprocessing Pipeline for Piezoelectric Sensor Decoupling
Converts raw multichannel CSV sensor data (Strain, Temperature) into spectrogram tensors.

This script supports an ablation study setup:
- 3-channel output: Includes Strain X, Strain Y, and Temperature.
- 2-channel output: Includes Strain X, Strain Y only (Temperature excluded).
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample

from ssqueezepy import cwt as ssq_cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq

# ==========================================
# Signal Processing Configuration
# ==========================================
TARGET_FS = 50.0  # Target sampling rate (Hz)

# Butterworth filter settings
STRAIN_BAND = (0.1, 15.0)  # Hz
TEMP_LP = 1.0              # Hz
FILTER_ORDER = 4

# Morse CWT settings
FREQ_MIN, FREQ_MAX = 0.2, 15.0  # Hz
N_FREQ = 96                     # Number of frequency bins (Height)
TARGET_WIDTH = 256              # Target time steps (Width)

MORSE_GAMMA = 3.0
MORSE_TIME_BANDWIDTH = 60.0
MORSE_BETA = MORSE_TIME_BANDWIDTH / MORSE_GAMMA
MORSE_WAVELET = ('gmw', {'beta': MORSE_BETA, 'gamma': MORSE_GAMMA})
_MORSE_WAVELET_OBJ = Wavelet(MORSE_WAVELET)

# Column detection heuristics
STRAIN_CANDIDATES = [('X', 'Y'), ('25X', '25Y'), ('strain1', 'strain2'), ('S1', 'S2')]
TEMP_CANDIDATES = ['W', '25W', 'temp', 'temperature', 'Temp', 'Temperature', 'Y_temp']
TIME_CANDIDATES = ['time', 'Time', 'timestamp', 'Timestamp']


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------- Filtering ---------------------

def butter_bandpass(sig: np.ndarray, fs: float, low: float, high: float, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < 1 and 0 < high_n < 1 and low_n < high_n):
        return sig
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, sig, method='gust')


def butter_lowpass(sig: np.ndarray, fs: float, cutoff: float, order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    c = cutoff / nyq
    if not (0 < c < 1):
        return sig
    b, a = butter(order, c, btype='low')
    return filtfilt(b, a, sig, method='gust')


# --------------------- Preprocessing ---------------------

def detect_columns(df: pd.DataFrame, include_temp: bool) -> Tuple[Optional[str], str, str, Optional[str]]:
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    
    s1 = s2 = None
    for a, b in STRAIN_CANDIDATES:
        if a in df.columns and b in df.columns:
            s1, s2 = a, b
            break
            
    if s1 is None or s2 is None:
        raise ValueError(f"Could not detect strain columns. Found: {df.columns.tolist()}")

    tcol = None
    if include_temp:
        tcol = next((t for t in TEMP_CANDIDATES if t in df.columns), None)
        if tcol is None:
            raise ValueError(f"Could not detect temp column but --include_temp is True. Found: {df.columns.tolist()}")

    return time_col, s1, s2, tcol


def unify_fs(time: Optional[np.ndarray], x: np.ndarray, target_fs: float) -> Tuple[np.ndarray, float]:
    if time is None:
        return resample(x, len(x)), target_fs
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return resample(x, len(x)), target_fs
    
    fs_raw = 1.0 / np.mean(dt)
    n_new = max(8, int(round(len(x) * (target_fs / fs_raw))))
    return resample(x, n_new), target_fs


# --------------------- CWT ---------------------

def cwt_channel_morse(sig: np.ndarray, fs: float, freq_min: float = FREQ_MIN,
                      freq_max: float = FREQ_MAX, n_freq: int = N_FREQ, target_width: int = TARGET_WIDTH) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    Wx, scales = ssq_cwt(sig, wavelet=MORSE_WAVELET, fs=fs)
    freqs = scale_to_freq(scales, _MORSE_WAVELET_OBJ, len(sig), fs=fs)
    
    freqs = np.asarray(freqs, dtype=float)
    Wx = np.asarray(Wx)

    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    freqs_sel = freqs[mask]
    Wx_sel = Wx[mask, :]
    
    sort_idx = np.argsort(freqs_sel)
    freqs_sel = freqs_sel[sort_idx]
    Wx_sel = Wx_sel[sort_idx, :]

    target_freqs = np.linspace(freq_min, freq_max, n_freq)
    mag = np.abs(Wx_sel)
    n_sel, T = mag.shape

    mag_interp = np.empty((n_freq, T), dtype=np.float32)
    for t in range(T):
        mag_interp[:, t] = np.interp(target_freqs, freqs_sel, mag[:, t])

    if T != target_width:
        t_old = np.arange(T)
        t_new = np.linspace(0, T - 1, target_width)
        mag_resized = np.empty((n_freq, target_width), dtype=np.float32)
        for fi in range(n_freq):
            mag_resized[fi, :] = np.interp(t_new, t_old, mag_interp[fi, :])
    else:
        mag_resized = mag_interp

    mag_resized = np.nan_to_num(mag_resized, nan=0.0, posinf=0.0, neginf=0.0)
    mag_log = np.log1p(mag_resized)
    mag_log = np.nan_to_num(mag_log, nan=0.0, posinf=0.0, neginf=0.0)

    return mag_log.astype(np.float32)


# --------------------- Processing Pipeline ---------------------

def process_csv(csv_path: str, include_temp: bool) -> np.ndarray:
    df = pd.read_csv(csv_path)
    time_col, s1, s2, tcol = detect_columns(df, include_temp)

    time = df[time_col].to_numpy(dtype=float) if time_col is not None else None
    ch1 = df[s1].to_numpy(dtype=float)
    ch2 = df[s2].to_numpy(dtype=float)

    ch1_rs, fs = unify_fs(time, ch1, TARGET_FS)
    ch2_rs, _ = unify_fs(time, ch2, TARGET_FS)

    ch1_f = butter_bandpass(ch1_rs, fs, *STRAIN_BAND)
    ch2_f = butter_bandpass(ch2_rs, fs, *STRAIN_BAND)

    S1 = cwt_channel_morse(ch1_f, fs)
    S2 = cwt_channel_morse(ch2_f, fs)

    channels = [S1, S2]

    if include_temp and tcol:
        temp = df[tcol].to_numpy(dtype=float)
        temp_rs, _ = unify_fs(time, temp, TARGET_FS)
        temp_f = butter_lowpass(temp_rs, fs, TEMP_LP)
        
        T_spec = cwt_channel_morse(temp_f, fs, freq_min=0.05)
        channels.append(T_spec)

    tensor = np.stack(channels, axis=0).astype(np.float32)
    if not np.isfinite(tensor).all():
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
    return tensor


def infer_group_name(csv_path: str) -> str:
    stem = Path(csv_path).stem
    return stem.split('_')[0]


def main():
    parser = argparse.ArgumentParser(description="Generate Spectrograms from Sensor CSV Data")
    parser.add_argument('--input_dir', type=str, default='data_study', help='Path to raw data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save processed .npy files')
    parser.add_argument('--include_temp', action='store_true', help='Include temperature data (generates 3-channel instead of 2-channel)')
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    index_json_path = os.path.join(args.output_dir, 'index.json')
    index = []

    class_dirs = [d for d in sorted(os.listdir(args.input_dir)) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    for cls in class_dirs:
        in_dir = os.path.join(args.input_dir, cls)
        out_dir = os.path.join(args.output_dir, cls)
        ensure_dir(out_dir)

        csv_files = sorted(glob.glob(os.path.join(in_dir, '*.csv')))
        for csv_path in csv_files:
            try:
                arr = process_csv(csv_path, args.include_temp)
                base = Path(csv_path).stem + '.npy'
                out_path = os.path.join(out_dir, base)
                np.save(out_path, arr)

                index.append({
                    'path': os.path.relpath(out_path, args.output_dir),
                    'class': cls,
                    'group': infer_group_name(csv_path)
                })
                print(f"✅ Saved {out_path} shape={arr.shape}")
            except Exception as e:
                print(f"❌ Error {csv_path}: {e}")

    with open(index_json_path, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"📄 Index written: {index_json_path} ({len(index)} samples)")


if __name__ == '__main__':
    main()