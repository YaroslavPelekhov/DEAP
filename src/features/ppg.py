"""
PPG (BVP) feature extraction — HRV features.
Moved from peripheral_features.py, unchanged logic.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch


FEATURE_NAMES = [
    'mean_hr', 'sdnn', 'rmssd', 'pnn50',
    'lf_power', 'hf_power', 'lf_hf_ratio',
    'mean_amp', 'std_amp', 'ibi_cv',
]
N_PPG_FEATURES = len(FEATURE_NAMES)


def _ppg_sos(fs: int = 128) -> np.ndarray:
    nyq = fs / 2.0
    return butter(3, [0.5 / nyq, 8.0 / nyq], btype='bandpass', output='sos')


def extract_ppg_features(ppg: np.ndarray, fs: int = 128) -> np.ndarray:
    """Extract 10 HRV features from a PPG trial."""
    ppg_f = sosfiltfilt(_ppg_sos(fs), ppg)
    p_min, p_max = ppg_f.min(), ppg_f.max()
    if p_max - p_min > 1e-8:
        ppg_f = 2 * (ppg_f - p_min) / (p_max - p_min) - 1

    peaks, _ = find_peaks(ppg_f, distance=int(0.4 * fs), prominence=0.1)
    feats = np.zeros(N_PPG_FEATURES, dtype=np.float32)
    if len(peaks) < 3:
        return feats

    ibi   = np.diff(peaks) / fs * 1000.0
    valid = (ibi > 300) & (ibi < 2000)
    if valid.sum() < 2:
        return feats
    ibi = ibi[valid]

    mean_hr = 60000.0 / ibi.mean()
    sdnn    = ibi.std(ddof=1)
    diffs   = np.diff(ibi)
    rmssd   = np.sqrt(np.mean(diffs ** 2))
    pnn50   = 100.0 * np.mean(np.abs(diffs) > 50)
    ibi_cv  = sdnn / (ibi.mean() + 1e-8)

    t_ibi     = np.cumsum(ibi) / 1000.0
    t_uniform = np.arange(t_ibi[0], t_ibi[-1], 0.25)
    if len(t_uniform) < 8:
        lf_power, hf_power, lf_hf = 0.0, 0.0, 0.0
    else:
        ibi_u = np.interp(t_uniform, t_ibi, ibi)
        freqs, psd = welch(ibi_u, fs=4.0, nperseg=min(len(ibi_u), 64))
        lf_power = float(np.log1p(psd[(freqs >= 0.04) & (freqs < 0.15)].sum()))
        hf_power = float(np.log1p(psd[(freqs >= 0.15) & (freqs < 0.40)].sum()))
        lf_hf    = lf_power / (hf_power + 1e-8)

    amps = ppg_f[peaks]
    feats[:] = [mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf,
                float(amps.mean()), float(amps.std(ddof=1)) if len(amps) > 1 else 0.0,
                ibi_cv]
    return feats.astype(np.float32)


def extract_ppg_subject(ppg_trials: np.ndarray, n_windows: int | None = None,
                        fs: int = 128) -> np.ndarray:
    """
    (n_trials, n_samples) -> (n_trials [* n_windows], 10)
    If n_windows given, repeats each trial's features n_windows times.
    """
    per_trial = np.stack([extract_ppg_features(ppg_trials[i], fs)
                          for i in range(len(ppg_trials))])
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)
