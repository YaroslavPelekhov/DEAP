"""
GSR (EDA) feature extraction.

Features (8 core — no DEAP-specific consensus labels):
  0  scl_mean      - mean tonic level
  1  scl_std       - std tonic level
  2  eda_mean      - mean raw EDA
  3  eda_std       - std raw EDA
  4  scr_n_peaks   - number of SCR peaks
  5  scr_mean_amp  - mean SCR amplitude
  6  scr_rise_rate - mean SCR rise rate
  7  eda_slope     - linear trend slope

Note: DEAP-specific consensus labels (cons_val, cons_ar) are intentionally
excluded — they are stimulus-specific and do not transfer to new devices/stimuli.
FAA and FTA are computed in eeg.py as they require EEG frontal channels.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


FEATURE_NAMES = [
    'scl_mean', 'scl_std',
    'eda_mean', 'eda_std',
    'scr_n_peaks', 'scr_mean_amp', 'scr_rise_rate',
    'eda_slope',
]
N_GSR_FEATURES = len(FEATURE_NAMES)


def extract_gsr_features(gsr: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Extract EDA features from a single trial.

    Args:
        gsr: (n_samples,) raw EDA signal
        fs:  sampling frequency

    Returns:
        features: (8,) float32
    """
    gsr = gsr.astype(np.float64)
    feats = np.zeros(N_GSR_FEATURES, dtype=np.float32)

    if gsr.std() > 1e3:
        gsr = (gsr - gsr.mean()) / (gsr.std() + 1e-8)

    tonic_win = int(4 * fs)
    scl = uniform_filter1d(gsr, size=tonic_win)
    phasic = gsr - scl

    min_dist = int(0.5 * fs)
    peaks, _ = find_peaks(phasic, distance=min_dist, prominence=0.01)
    n_peaks  = len(peaks)
    mean_amp = float(phasic[peaks].mean()) if n_peaks > 0 else 0.0

    rise_rates = []
    for pk in peaks:
        onset = max(0, pk - int(0.5 * fs))
        rise  = phasic[pk] - phasic[onset]
        dt    = (pk - onset) / fs + 1e-8
        rise_rates.append(rise / dt)
    mean_rise = float(np.mean(rise_rates)) if rise_rates else 0.0

    t = np.arange(len(gsr))
    slope = float(np.polyfit(t, gsr, 1)[0])

    feats[:] = [
        scl.mean(), scl.std(ddof=1),
        gsr.mean(), gsr.std(ddof=1),
        n_peaks, mean_amp, mean_rise, slope,
    ]
    return feats.astype(np.float32)


def extract_gsr_subject(gsr_trials: np.ndarray, n_windows: int | None = None,
                        fs: int = 128) -> np.ndarray:
    """
    (n_trials, n_samples) -> (n_trials [* n_windows], 8)
    If n_windows given, repeats each trial's features n_windows times.
    """
    per_trial = np.stack([extract_gsr_features(gsr_trials[i], fs)
                          for i in range(len(gsr_trials))])
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)
