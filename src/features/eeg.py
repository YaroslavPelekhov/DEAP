"""
EEG feature extraction.

Features per window per channel:
  - Differential Entropy (DE) for each frequency band  →  n_ch × n_bands
  - Hjorth Mobility and Complexity                     →  n_ch × 2

Total per window: n_ch × (n_bands + 2)
For 32 channels, 4 bands (theta/alpha/beta/gamma): 32 × 6 = 192

Frontal Asymmetry indices (per trial, repeated across windows):
  - FAA: log(alpha_F4) - log(alpha_F3)   (Frontal Alpha Asymmetry)
  - FTA: log(theta_F4) - log(theta_F3)   (Frontal Theta Asymmetry)

References:
  - Li et al. (2018) SEED dataset
  - Hjorth (1970) EEG descriptors
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Dict, List, Optional, Tuple


# ── Default band definitions ───────────────────────────────────────────────
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    'theta': (5, 7),
    'alpha': (8, 13),
    'beta':  (14, 30),
    'gamma': (31, 45),
}

SFREQ_DEFAULT = 128


def _bandpass_sos(low: float, high: float, fs: int, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')


def hjorth_params(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hjorth Mobility and Complexity for a multi-channel segment.

    Args:
        seg: (n_ch, n_samples)

    Returns:
        mobility: (n_ch,)
        complexity: (n_ch,)
    """
    d1 = np.diff(seg, axis=1)
    d2 = np.diff(d1, axis=1)
    var_x  = np.maximum(np.var(seg, axis=1, ddof=1), 1e-12)
    var_d1 = np.maximum(np.var(d1,  axis=1, ddof=1), 1e-12)
    var_d2 = np.maximum(np.var(d2,  axis=1, ddof=1), 1e-12)
    mob  = np.sqrt(var_d1 / var_x)
    comp = np.sqrt(var_d2 / var_d1) / (mob + 1e-12)
    return mob.astype(np.float32), comp.astype(np.float32)


class EEGExtractor:
    """
    Extract DE + Hjorth features from EEG windows.

    Args:
        bands:      frequency band dict {name: (lo, hi)}
        fs:         sampling frequency
        win_sec:    window length in seconds
        stride_sec: stride between windows (= win_sec for non-overlapping)
        channel_names: list of channel names (10-20 system)
    """

    def __init__(
        self,
        bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
        fs: int = SFREQ_DEFAULT,
        win_sec: float = 1.0,
        stride_sec: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
    ):
        self.bands = bands
        self.fs = fs
        self.win = int(win_sec * fs)
        self.stride = int((stride_sec or win_sec) * fs)
        self.channel_names = channel_names

        # Pre-build filter bank
        self._sos: Dict[str, np.ndarray] = {
            name: _bandpass_sos(lo, hi, fs)
            for name, (lo, hi) in bands.items()
        }

        self.n_bands = len(bands)
        self.feature_names_per_ch: List[str] = (
            list(bands.keys()) + ['Hjorth_mob', 'Hjorth_comp']
        )
        self.n_feat_per_ch = len(self.feature_names_per_ch)

    @property
    def feature_names(self) -> List[str]:
        """Full list: EEG_Fp1_theta, EEG_Fp1_alpha, ..., EEG_Fp1_Hjorth_mob, ..."""
        ch_names = self.channel_names or [f'ch{i}' for i in range(32)]
        return [
            f'EEG_{ch}_{feat}'
            for ch in ch_names
            for feat in self.feature_names_per_ch
        ]

    def extract_trial(
        self,
        eeg: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract features from a single trial.

        Args:
            eeg:      (n_ch, n_samples) — already preprocessed EEG
            baseline: (n_ch, baseline_samples) or None — subtracted from each window

        Returns:
            feats: (n_windows, n_ch * n_feat_per_ch)
        """
        n_ch, n_samp = eeg.shape
        n_wins = (n_samp - self.win) // self.stride + 1

        # Baseline mean per channel per band (for DE baseline correction)
        base_de: Optional[Dict[str, np.ndarray]] = None
        if baseline is not None:
            base_de = {}
            for name, sos in self._sos.items():
                bl_filt = sosfiltfilt(sos, baseline, axis=1)
                bl_var  = np.maximum(np.var(bl_filt, axis=1, ddof=1), 1e-12)
                base_de[name] = 0.5 * np.log(2 * np.pi * np.e * bl_var)

        de_all = np.zeros((n_wins, n_ch, self.n_bands), dtype=np.float32)

        for b_idx, (name, sos) in enumerate(self._sos.items()):
            filtered = sosfiltfilt(sos, eeg, axis=1)
            for w in range(n_wins):
                s = w * self.stride
                seg = filtered[:, s: s + self.win]
                var = np.maximum(np.var(seg, axis=1, ddof=1), 1e-12)
                de  = 0.5 * np.log(2 * np.pi * np.e * var)
                if base_de is not None:
                    de = de - base_de[name]
                de_all[w, :, b_idx] = de

        # Hjorth per window
        hjorth = np.zeros((n_wins, n_ch * 2), dtype=np.float32)
        for w in range(n_wins):
            s = w * self.stride
            seg = eeg[:, s: s + self.win]
            mob, comp = hjorth_params(seg)
            hjorth[w, :n_ch] = mob
            hjorth[w, n_ch:] = comp

        # Interleave: for each channel → [band0, band1, ..., mob, comp]
        de_flat = de_all.reshape(n_wins, n_ch * self.n_bands)
        # de_flat layout: [ch0_b0, ch0_b1, ..., ch0_bN, ch1_b0, ...]
        # hjorth layout:  [mob_ch0, mob_ch1, ..., comp_ch0, comp_ch1, ...]
        # We want: [ch0_b0, ..., ch0_bN, ch0_mob, ch0_comp, ch1_b0, ...]

        de_per_ch  = de_flat.reshape(n_wins, n_ch, self.n_bands)      # (W, C, B)
        mob_per_ch = hjorth[:, :n_ch, np.newaxis]                       # (W, C, 1)
        cmp_per_ch = hjorth[:, n_ch:, np.newaxis]                       # (W, C, 1)

        combined = np.concatenate([de_per_ch, mob_per_ch, cmp_per_ch], axis=2)  # (W, C, B+2)
        return combined.reshape(n_wins, n_ch * self.n_feat_per_ch)

    def extract_subject(
        self,
        eeg_trials: np.ndarray,
        baselines: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for all trials of a subject.

        Args:
            eeg_trials: (n_trials, n_ch, n_samples)
            baselines:  (n_trials, n_ch, baseline_samples) or None

        Returns:
            feats:  (n_trials * n_windows, n_ch * n_feat_per_ch)
            groups: (n_trials * n_windows,)  trial index for GroupKFold
        """
        n_trials = eeg_trials.shape[0]
        all_feats, all_groups = [], []

        for i in range(n_trials):
            bl = baselines[i] if baselines is not None else None
            w_feats = self.extract_trial(eeg_trials[i], baseline=bl)
            all_feats.append(w_feats)
            all_groups.append(np.full(len(w_feats), i, dtype=np.int32))

        return np.vstack(all_feats), np.concatenate(all_groups)

    def compute_faa_fta(
        self,
        eeg_trials: np.ndarray,
        left_idx: int,
        right_idx: int,
    ) -> np.ndarray:
        """
        Frontal Alpha/Theta Asymmetry.

        Returns (n_trials, 2): [FAA, FTA] per trial.
        FAA = log(alpha_right) - log(alpha_left)
        FTA = log(theta_right) - log(theta_left)
        """
        bands = list(self.bands.keys())
        alpha_idx = bands.index('alpha') if 'alpha' in bands else None
        theta_idx = bands.index('theta') if 'theta' in bands else None

        results = []
        for trial in eeg_trials:
            row = [0.0, 0.0]
            for band_i, band_name in enumerate(['alpha', 'theta']):
                if band_name not in self._sos:
                    continue
                filt = sosfiltfilt(self._sos[band_name], trial, axis=1)
                var_l = np.maximum(np.var(filt[left_idx], ddof=1), 1e-12)
                var_r = np.maximum(np.var(filt[right_idx], ddof=1), 1e-12)
                asym  = 0.5 * np.log(var_r) - 0.5 * np.log(var_l)
                if band_name == 'alpha': row[0] = asym
                if band_name == 'theta': row[1] = asym
            results.append(row)
        return np.array(results, dtype=np.float32)
