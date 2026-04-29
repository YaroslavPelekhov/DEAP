"""
EEG feature extraction: Differential Entropy (DE).

DE for a Gaussian signal in band [f1, f2] is:
    DE = 0.5 * log(2*pi*e * sigma^2)

This is the most reproducible and effective EEG feature for emotion recognition
on DEAP (see: Li et al. 2018, SEED; Zheng et al. 2015 IEEE Trans Affect Comput).

Pipeline per trial:
  1. Bandpass filter EEG for each of 5 bands
  2. Split into 1-s non-overlapping windows (60 windows per 60-s trial)
  3. Compute DE per window, channel, band  ->  (n_windows, 160) per trial

Each window is kept as a separate sample (standard SOTA approach).
This gives 40 trials x 60 windows = 2400 samples/subject vs 40 if averaged.
GroupKFold at trial level prevents data leakage between folds.
"""
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.signal import butter, sosfiltfilt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SFREQ, EEG_BANDS, EEG_WINDOW_SEC, N_EEG_FEATURES


# ─── Filter bank ─────────────────────────────────────────────────────────────

def _bandpass_sos(low: float, high: float, fs: int = SFREQ, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")


_FILTER_BANK: Dict[str, np.ndarray] = {
    name: _bandpass_sos(lo, hi)
    for name, (lo, hi) in EEG_BANDS.items()
}


# ─── Core DE computation ─────────────────────────────────────────────────────

def _de_windows(eeg: np.ndarray, win: int) -> np.ndarray:
    """
    Compute DE for all windows and bands at once.

    Args:
        eeg : (n_channels, n_samples)
        win : samples per window

    Returns:
        (n_windows, n_channels * n_bands) = (n_windows, 160) float32
    """
    n_ch, n_samp = eeg.shape
    n_wins = n_samp // win
    n_bands = len(EEG_BANDS)

    de = np.zeros((n_wins, n_ch, n_bands), dtype=np.float32)

    for b_idx, sos in enumerate(_FILTER_BANK.values()):
        filtered = sosfiltfilt(sos, eeg, axis=1)        # (n_ch, n_samp)
        for w in range(n_wins):
            seg = filtered[:, w * win : (w + 1) * win]  # (n_ch, win)
            var = np.var(seg, axis=1, ddof=1)            # (n_ch,)
            var = np.maximum(var, 1e-12)
            de[w, :, b_idx] = 0.5 * np.log(2 * np.pi * np.e * var)

    return de.reshape(n_wins, n_ch * n_bands)            # (n_windows, 160)


# ─── Subject-level extraction ────────────────────────────────────────────────

def extract_de_subject(
    eeg_trials: np.ndarray,
    window_sec: float = EEG_WINDOW_SEC,
    fs: int = SFREQ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-window DE features for all trials of one subject.

    Args:
        eeg_trials : (n_trials, n_channels, n_samples)
        window_sec : window length in seconds (default 1 s)
        fs         : sampling frequency

    Returns:
        feats  : (n_trials * n_windows, 160)  float32
        groups : (n_trials * n_windows,)       int  — trial index for GroupKFold
    """
    n_trials = eeg_trials.shape[0]
    win = int(window_sec * fs)
    n_wins = eeg_trials.shape[2] // win

    all_feats  = np.zeros((n_trials * n_wins, N_EEG_FEATURES), dtype=np.float32)
    all_groups = np.zeros(n_trials * n_wins, dtype=np.int32)

    for i in range(n_trials):
        w_feats = _de_windows(eeg_trials[i], win)          # (n_wins, 160)
        start = i * n_wins
        all_feats[start : start + n_wins]  = w_feats
        all_groups[start : start + n_wins] = i

    return all_feats, all_groups
