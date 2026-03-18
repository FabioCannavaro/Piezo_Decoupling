import os
import json
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, savgol_filter, resample

from ssqueezepy import cwt as ssq_cwt, Wavelet
from ssqueezepy.experimental import scale_to_freq


# ---- Paths ----
INPUT_ROOT = 'data_study'          # class별 폴더 안에 CSV
OUTPUT_ROOT = 'input_data_2026'             # 3×H×W float .npy 저장
INDEX_JSON = 'input_data_xy_2026/index.json'   # 샘플 인덱스


# ---- Config ----
TARGET_FS = 50.0  # Hz (통일)

# Butterworth filters
STRAIN_BAND = (0.1, 15.0)  # Hz
TEMP_LP = 1.0              # Hz
FILTER_ORDER = 4

# CWT / Frequency settings
FREQ_MIN, FREQ_MAX = 0.2, 15.0  # Hz
N_FREQ = 96

# Morse wavelet parameters (MATLAB: gamma=3, TimeBandwidth=60 → beta=20)
MORSE_GAMMA = 3.0
MORSE_TIME_BANDWIDTH = 60.0
MORSE_BETA = MORSE_TIME_BANDWIDTH / MORSE_GAMMA

MORSE_WAVELET = ('gmw', {'beta': MORSE_BETA, 'gamma': MORSE_GAMMA})

# Image size
H, W = N_FREQ, 256  # (freq × time)


# Column name preference: time / temp=W / strain1=X / strain2=Y
STRAIN_CANDIDATES = [
    ('X', 'Y'),
    ('25X', '25Y'),
    ('strain1', 'strain2'),
    ('S1', 'S2'),
]
TEMP_CANDIDATES = ['W', '25W', 'temp', 'temperature', 'Temp', 'Temperature', 'Y_temp']
TIME_CANDIDATES = ['time', 'Time', 'timestamp', 'Timestamp']


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ---------- Filtering & preprocessing ----------

def butter_bandpass(sig: np.ndarray, fs: float, low: float, high: float,
                    order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    if not (0 < low_n < 1 and 0 < high_n < 1 and low_n < high_n):
        return sig
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, sig, method='gust')


def butter_lowpass(sig: np.ndarray, fs: float, cutoff: float,
                   order: int = FILTER_ORDER) -> np.ndarray:
    nyq = 0.5 * fs
    c = cutoff / nyq
    if not (0 < c < 1):
        return sig
    b, a = butter(order, c, btype='low')
    return filtfilt(b, a, sig, method='gust')


def detrend_poly(sig: np.ndarray, window: int = 101, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay 기반 polynomial detrend."""
    if len(sig) < 11:
        return sig - np.median(sig)
    # 데이터 길이에 맞게 window / polyorder 자동 조정
    window = min(window, max(7, (len(sig) // 5) * 2 + 1))
    polyorder = min(polyorder, max(2, window // 10))
    trend = savgol_filter(sig, window_length=window, polyorder=polyorder)
    return sig - trend


# ---------- Column detection ----------

def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    # time
    time_col = None
    for c in TIME_CANDIDATES:
        if c in df.columns:
            time_col = c
            break

    # Hard preference: W=temp, X/Y=strain
    if 'X' in df.columns and 'Y' in df.columns:
        s1, s2 = 'X', 'Y'
    elif '25X' in df.columns and '25Y' in df.columns:
        s1, s2 = '25X', '25Y'
    else:
        s1 = s2 = None
        for a, b in STRAIN_CANDIDATES:
            if a in df.columns and b in df.columns:
                s1, s2 = a, b
                break

    if 'W' in df.columns:
        tcol = 'W'
    elif '25W' in df.columns:
        tcol = '25W'
    else:
        tcol = None
        for t in TEMP_CANDIDATES:
            if t in df.columns:
                tcol = t
                break

    # Fallbacks (최후의 수단)
    if s1 is None or s2 is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        for c in [time_col, tcol]:
            if c in num_cols:
                try:
                    num_cols.remove(c)
                except ValueError:
                    pass
        if len(num_cols) >= 2:
            s1, s2 = num_cols[0], num_cols[1]

    if tcol is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        for c in [time_col, s1, s2]:
            if c in num_cols:
                try:
                    num_cols.remove(c)
                except ValueError:
                    pass
        if len(num_cols) >= 1:
            tcol = num_cols[0]

    if s1 is None or s2 is None or tcol is None:
        raise ValueError(f"Could not detect sensor columns. Found: {df.columns.tolist()}")
    return time_col, s1, s2, tcol


def unify_fs(time: np.ndarray, x: np.ndarray, target_fs: float):
    """불규칙 샘플링 → target_fs로 resample."""
    if time is None:
        # time 정보 없으면 길이 유지 (어차피 상대 time만 중요)
        return resample(x, len(x)), target_fs
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return resample(x, len(x)), target_fs
    fs_raw = 1.0 / np.mean(dt)
    n_new = int(round(len(x) * (target_fs / fs_raw)))
    n_new = max(8, n_new)
    x_rs = resample(x, n_new)
    return x_rs, target_fs


# ---------- Morse CWT ----------

# 미리 wavelet 객체 생성 (scale_to_freq에서 사용)
_MORSE_WAVELET_OBJ = Wavelet(MORSE_WAVELET)


def cwt_channel_morse(sig: np.ndarray, fs: float,
                      freq_min: float = FREQ_MIN,
                      freq_max: float = FREQ_MAX,
                      n_freq: int = N_FREQ,
                      target_width: int = W) -> np.ndarray:
    """
    하나의 1D 채널에 대해 analytic Morse wavelet(CWT) → (n_freq × target_width) 스펙트로그램 반환.
    """
    sig = np.asarray(sig, dtype=float)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) CWT (ssqueezepy)
    # Wx: (n_scales, T), scales: (n_scales,)
    Wx, scales = ssq_cwt(sig, wavelet=MORSE_WAVELET, fs=fs)

    # 2) scales → frequency (Hz)
    #    scale_to_freq는 각 scale에 대응하는 중심 주파수 반환
    freqs = scale_to_freq(scales, _MORSE_WAVELET_OBJ, len(sig), fs=fs)

    freqs = np.asarray(freqs, dtype=float)
    Wx = np.asarray(Wx)

    # 3) 관심 주파수 범위 [freq_min, freq_max]만 선택
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not np.any(mask):
        # 범위 안에 아무것도 없으면 전체를 쓰고 나중에 interpolate
        mask = np.ones_like(freqs, dtype=bool)

    freqs_sel = freqs[mask]
    Wx_sel = Wx[mask, :]       # (n_sel, T)

    # 4) freqs_sel이 단조 증가하도록 정렬
    sort_idx = np.argsort(freqs_sel)
    freqs_sel = freqs_sel[sort_idx]
    Wx_sel = Wx_sel[sort_idx, :]

    # 5) 원하는 주파수 grid로 보간 (n_freq)
    target_freqs = np.linspace(freq_min, freq_max, n_freq)

    # magnitude 기준으로 interpolation
    mag = np.abs(Wx_sel)  # (n_sel, T)
    n_sel, T = mag.shape

    # freq 방향 보간: (n_freq, T)
    mag_interp = np.empty((n_freq, T), dtype=np.float32)
    for t in range(T):
        mag_interp[:, t] = np.interp(
            target_freqs,
            freqs_sel,
            mag[:, t]
        )

    # 6) time 축을 target_width로 리샘플링
    if T != target_width:
        t_old = np.arange(T)
        t_new = np.linspace(0, T - 1, target_width)
        mag_resized = np.empty((n_freq, target_width), dtype=np.float32)
        for fi in range(n_freq):
            mag_resized[fi, :] = np.interp(
                t_new,
                t_old,
                mag_interp[fi, :]
            )
    else:
        mag_resized = mag_interp

    # 7) log compression + NaN/Inf 처리
    mag_resized = np.nan_to_num(mag_resized, nan=0.0, posinf=0.0, neginf=0.0)
    mag_log = np.log1p(mag_resized)  # log(1 + x)로 dynamic range 줄이기
    mag_log = np.nan_to_num(mag_log, nan=0.0, posinf=0.0, neginf=0.0)

    return mag_log.astype(np.float32)  # (n_freq, target_width)


# ---------- CSV → 3채널 Morse CWT 변환 ----------

def process_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    time_col, s1, s2, tcol = detect_columns(df)

    time = df[time_col].to_numpy(dtype=float) if time_col is not None else None
    ch1 = df[s1].to_numpy(dtype=float)
    ch2 = df[s2].to_numpy(dtype=float)
    temp = df[tcol].to_numpy(dtype=float)

    # unify fs
    chs = []
    for sig in (ch1, ch2, temp):
        sig_rs, fs = unify_fs(time, sig, TARGET_FS)
        chs.append(sig_rs)
    ch1, ch2, temp = chs[0], chs[1], chs[2]

    # filtering
    ch1 = butter_bandpass(ch1, fs, *STRAIN_BAND)
    ch2 = butter_bandpass(ch2, fs, *STRAIN_BAND)
    temp_f = butter_lowpass(temp, fs, TEMP_LP)
    #temp_f = detrend_poly(temp_f)

    # CWT (Morse)
    S1 = cwt_channel_morse(ch1, fs)
    S2 = cwt_channel_morse(ch2, fs)
    T  = cwt_channel_morse(temp_f, fs, freq_min=0.05)

    T = T * 5.0  # 값을 5배 뻥튀기

    tensor = np.stack([S1, S2, T], axis=0).astype(np.float32)  # 3×H×W
    if not np.isfinite(tensor).all():
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def infer_group_name(csv_path: str) -> str:
    stem = Path(csv_path).stem
    return stem.split('_')[0]


def main():
    ensure_dir(OUTPUT_ROOT)
    index = []

    class_dirs = [d for d in sorted(os.listdir(INPUT_ROOT))
                  if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    for cls in class_dirs:
        in_dir = os.path.join(INPUT_ROOT, cls)
        out_dir = os.path.join(OUTPUT_ROOT, cls)
        ensure_dir(out_dir)

        csv_files = sorted(glob.glob(os.path.join(in_dir, '*.csv')))
        for csv_path in csv_files:
            try:
                arr = process_csv(csv_path)
                base = Path(csv_path).stem + '.npy'
                out_path = os.path.join(out_dir, base)
                np.save(out_path, arr)

                if not np.isfinite(arr).all():
                    print(f"⚠️ Non-finite values after save in {out_path}.")

                index.append({
                    'path': os.path.relpath(out_path, OUTPUT_ROOT),
                    'class': cls,
                    'group': infer_group_name(csv_path)
                })
                print(f"✅ Saved {out_path} shape={arr.shape}")
            except Exception as e:
                print(f"❌ Error {csv_path}: {e}")

    ensure_dir(os.path.dirname(INDEX_JSON))
    with open(INDEX_JSON, 'w') as f:
        json.dump(index, f, indent=2)
    print(f"📄 Index written: {INDEX_JSON} ({len(index)} samples)")


if __name__ == '__main__':
    main()
