"""
PPG (BVP) feature extraction — HRV features.

Low-level functions (numpy-only)
─────────────────────────────────
  extract_ppg_features(ppg, fs)        – 10 HRV features from one trial
  extract_ppg_subject(trials, ...)     – all trials, optional window repeat

MNE-aware extractor class
─────────────────────────
  HRVExtractor(EpochExtractor)         – one feature vector per epoch
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch

import mne

from .base import EpochExtractor


FEATURE_NAMES = [
    'mean_hr', 'sdnn', 'rmssd', 'pnn50',
    'lf_power', 'hf_power', 'lf_hf_ratio',
    'mean_amp', 'std_amp', 'ibi_cv',
]
N_PPG_FEATURES = len(FEATURE_NAMES)


# ── Low-level helpers ──────────────────────────────────────────────────────

def _ppg_sos(fs: int = 128) -> np.ndarray:
    nyq = fs / 2.0
    return butter(3, [0.5 / nyq, 8.0 / nyq], btype='bandpass', output='sos')


def extract_ppg_features(ppg: np.ndarray, fs: int = 128) -> np.ndarray:
    """Extract 10 HRV features from a single PPG trial (1-D array)."""
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

    amps    = ppg_f[peaks]
    feats[:] = [mean_hr, sdnn, rmssd, pnn50, lf_power, hf_power, lf_hf,
                float(amps.mean()),
                float(amps.std(ddof=1)) if len(amps) > 1 else 0.0,
                ibi_cv]
    return feats.astype(np.float32)


def extract_ppg_subject(ppg_trials: np.ndarray,
                         n_windows: int | None = None,
                         fs: int = 128) -> np.ndarray:
    """
    (n_trials, n_samples) → (n_trials [* n_windows], 10)
    If n_windows given, repeats each trial's features n_windows times.
    """
    per_trial = np.stack([extract_ppg_features(ppg_trials[i], fs)
                          for i in range(len(ppg_trials))])
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)


# ── MNE-aware extractor ────────────────────────────────────────────────────

class HRVExtractor(EpochExtractor):
    """
    Heart-Rate Variability features from PPG channel.

    Returns (n_epochs, 10) — one feature vector per epoch.
    The pipeline repeats this n_wins times to align with EEG windows.

    Parameters
    ----------
    ppg_ch : name of PPG channel in the Epochs object (default 'PPG')
    """

    def __init__(self, ppg_ch: str = 'PPG'):
        self.ppg_ch = ppg_ch

    @property
    def feature_names(self):
        return [f'PPG_{n}' for n in FEATURE_NAMES]

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        sfreq = int(epochs.info['sfreq'])
        try:
            data = self._get_channel(epochs, self.ppg_ch)  # (n_ep, n_samp)
        except (ValueError, KeyError):
            return np.zeros((len(epochs), N_PPG_FEATURES), dtype=np.float32)
        return np.stack([extract_ppg_features(data[i], sfreq)
                         for i in range(len(data))]).astype(np.float32)
