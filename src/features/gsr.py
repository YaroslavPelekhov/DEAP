"""
GSR (EDA) feature extraction.

Low-level functions (numpy-only)
─────────────────────────────────
  extract_gsr_features(gsr, fs)        – 8 EDA features from one trial
  extract_gsr_subject(trials, ...)     – all trials, optional window repeat

MNE-aware extractor class
─────────────────────────
  EDAExtractor(EpochExtractor)         – one feature vector per epoch

Features (8 core — no DEAP-specific consensus labels):
  scl_mean         mean tonic (skin conductance level)
  scl_std          std tonic level
  gsr_mean         mean raw EDA
  gsr_std          std raw EDA
  scr_n_peaks      number of SCR peaks
  scr_mean_amp     mean SCR amplitude
  scr_peak_auc     area under clipped phasic curve (µS·s)
  scr_peak_density SCR peaks per minute

Note: DEAP-specific consensus labels (cons_val, cons_ar) are intentionally
excluded — they are stimulus-specific and do not transfer to new devices/stimuli.
FAA and FTA are in eeg.py as they require EEG frontal channels.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

import mne

from .base import EpochExtractor


FEATURE_NAMES = [
    'scl_mean', 'scl_std',
    'gsr_mean', 'gsr_std',
    'scr_n_peaks', 'scr_mean_amp', 'scr_peak_auc',
    'scr_peak_density',
]
N_GSR_FEATURES = len(FEATURE_NAMES)


# ── Low-level helpers ──────────────────────────────────────────────────────

def extract_gsr_features(gsr: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Extract 8 EDA features from a single trial.

    Parameters
    ----------
    gsr : (n_samples,) raw EDA signal
    fs  : sampling frequency

    Returns
    -------
    features : (8,) float32
    """
    gsr   = gsr.astype(np.float64)
    feats = np.zeros(N_GSR_FEATURES, dtype=np.float32)

    # Normalize if wildly scaled
    if gsr.std() > 1e3:
        gsr = (gsr - gsr.mean()) / (gsr.std() + 1e-8)

    tonic_win = int(4 * fs)
    scl    = uniform_filter1d(gsr, size=tonic_win)
    phasic = gsr - scl

    # Adaptive prominence: at least 0.005 µS or 10% of phasic std
    prominence = max(0.005, 0.1 * phasic.std())
    min_dist = int(0.5 * fs)
    peaks, _ = find_peaks(phasic, distance=min_dist, prominence=prominence)
    n_peaks  = len(peaks)
    mean_amp = float(phasic[peaks].mean()) if n_peaks > 0 else 0.0

    # Peak AUC: area under clipped phasic curve (µS·s)
    peak_auc = float(np.trapz(np.clip(phasic, 0.0, None)) / fs)

    # Peak density: peaks per minute
    duration_min = len(gsr) / fs / 60.0
    peak_density = float(n_peaks / (duration_min + 1e-8))

    feats[:] = [
        scl.mean(), scl.std(ddof=1),
        gsr.mean(), gsr.std(ddof=1),
        n_peaks, mean_amp, peak_auc, peak_density,
    ]
    return feats.astype(np.float32)


def extract_gsr_subject(gsr_trials: np.ndarray,
                         n_windows: int | None = None,
                         fs: int = 128) -> np.ndarray:
    """
    (n_trials, n_samples) → (n_trials [* n_windows], 8)
    If n_windows given, repeats each trial's features n_windows times.
    """
    per_trial = np.stack([extract_gsr_features(gsr_trials[i], fs)
                          for i in range(len(gsr_trials))])
    if n_windows is None:
        return per_trial
    return np.repeat(per_trial, n_windows, axis=0)


# ── MNE-aware extractor ────────────────────────────────────────────────────

class EDAExtractor(EpochExtractor):
    """
    Electrodermal Activity (EDA/GSR) features from GSR channel.

    Returns (n_epochs, 8) — one feature vector per epoch.
    The pipeline repeats this n_wins times to align with EEG windows.

    Parameters
    ----------
    gsr_ch : name of GSR channel in the Epochs object (default 'GSR')
    """

    def __init__(self, gsr_ch: str = 'GSR'):
        self.gsr_ch = gsr_ch

    @property
    def feature_names(self):
        return [f'GSR_{n}' for n in FEATURE_NAMES]

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        sfreq = int(epochs.info['sfreq'])
        try:
            data = self._get_channel(epochs, self.gsr_ch)  # (n_ep, n_samp)
        except (ValueError, KeyError):
            return np.zeros((len(epochs), N_GSR_FEATURES), dtype=np.float32)
        return np.stack([extract_gsr_features(data[i], sfreq)
                         for i in range(len(data))]).astype(np.float32)
