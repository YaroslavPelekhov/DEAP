"""
EEG feature extraction.

Low-level functions (numpy-only)
─────────────────────────────────
  hjorth_params(seg)          – Hjorth mobility + complexity
  EEGExtractor                – old DEAP-specific class (kept for backward compat)

MNE-aware extractor classes (inherit from base.py)
───────────────────────────────────────────────────
  DEExtractor(WindowExtractor)        – Differential Entropy per band
  HjorthExtractor(WindowExtractor)    – Hjorth mobility + complexity
  DEHjorthExtractor(WindowExtractor)  – combined DE + Hjorth (v13 standard)
  FAAExtractor(EpochExtractor)        – Frontal Alpha/Theta Asymmetry per epoch

Features per window per channel (DEHjorthExtractor):
  DE theta, DE alpha, DE beta, DE gamma, Hjorth_mob, Hjorth_comp
  → n_ch × 6  (32 ch DEAP → 192, 19 ch Barometer → 114)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Dict, List, Optional, Tuple

import mne

from .base import WindowExtractor, EpochExtractor


# ── Default band definitions ───────────────────────────────────────────────
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    'theta': (5, 7),
    'alpha': (8, 13),
    'beta':  (14, 30),
    'gamma': (31, 45),
}

SFREQ_DEFAULT = 128


# ── Low-level helpers ──────────────────────────────────────────────────────

def _bandpass_sos(low: float, high: float, fs: int = SFREQ_DEFAULT,
                  order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')


def hjorth_params(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hjorth Mobility and Complexity for a multi-channel segment.

    Parameters
    ----------
    seg : (n_ch, n_samples)

    Returns
    -------
    mobility   : (n_ch,)
    complexity : (n_ch,)
    """
    d1 = np.diff(seg, axis=1)
    d2 = np.diff(d1,  axis=1)
    var_x  = np.maximum(np.var(seg, axis=1, ddof=1), 1e-12)
    var_d1 = np.maximum(np.var(d1,  axis=1, ddof=1), 1e-12)
    var_d2 = np.maximum(np.var(d2,  axis=1, ddof=1), 1e-12)
    mob    = np.sqrt(var_d1 / var_x)
    comp   = np.sqrt(var_d2 / var_d1) / (mob + 1e-12)
    return mob.astype(np.float32), comp.astype(np.float32)


# ── Backward-compatible DEAP class ────────────────────────────────────────

class EEGExtractor:
    """
    Extract DE + Hjorth features from raw numpy arrays.

    Kept for backward compatibility with the old FeaturePipeline.
    New code should use DEHjorthExtractor (MNE-aware).
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
        ch_names = self.channel_names or [f'ch{i}' for i in range(32)]
        return [f'EEG_{ch}_{feat}'
                for ch in ch_names
                for feat in self.feature_names_per_ch]

    def extract_trial(
        self,
        eeg: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract features from a single trial.

        Parameters
        ----------
        eeg      : (n_ch, n_samples)
        baseline : (n_ch, baseline_samples) or None

        Returns
        -------
        (n_windows, n_ch * n_feat_per_ch)
        """
        n_ch, n_samp = eeg.shape
        n_wins = (n_samp - self.win) // self.stride + 1

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
                s   = w * self.stride
                seg = filtered[:, s: s + self.win]
                var = np.maximum(np.var(seg, axis=1, ddof=1), 1e-12)
                de  = 0.5 * np.log(2 * np.pi * np.e * var)
                if base_de is not None:
                    de = de - base_de[name]
                de_all[w, :, b_idx] = de

        hjorth = np.zeros((n_wins, n_ch * 2), dtype=np.float32)
        for w in range(n_wins):
            s   = w * self.stride
            seg = eeg[:, s: s + self.win]
            mob, comp = hjorth_params(seg)
            hjorth[w, :n_ch] = mob
            hjorth[w, n_ch:] = comp

        de_per_ch  = de_all.reshape(n_wins, n_ch, self.n_bands)
        mob_per_ch = hjorth[:, :n_ch, np.newaxis]
        cmp_per_ch = hjorth[:, n_ch:, np.newaxis]
        combined   = np.concatenate([de_per_ch, mob_per_ch, cmp_per_ch], axis=2)
        return combined.reshape(n_wins, n_ch * self.n_feat_per_ch)

    def extract_subject(
        self,
        eeg_trials: np.ndarray,
        baselines: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for all trials.

        Parameters
        ----------
        eeg_trials : (n_trials, n_ch, n_samples)
        baselines  : (n_trials, n_ch, baseline_samples) or None

        Returns
        -------
        feats  : (n_trials * n_windows, n_ch * n_feat_per_ch)
        groups : (n_trials * n_windows,)
        """
        all_feats, all_groups = [], []
        for i in range(len(eeg_trials)):
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
        """Frontal Alpha/Theta Asymmetry. Returns (n_trials, 2)."""
        results = []
        for trial in eeg_trials:
            row = [0.0, 0.0]
            for band_name in ('alpha', 'theta'):
                if band_name not in self._sos:
                    continue
                filt  = sosfiltfilt(self._sos[band_name], trial, axis=1)
                var_l = np.maximum(np.var(filt[left_idx],  ddof=1), 1e-12)
                var_r = np.maximum(np.var(filt[right_idx], ddof=1), 1e-12)
                asym  = 0.5 * np.log(var_r) - 0.5 * np.log(var_l)
                if band_name == 'alpha': row[0] = asym
                if band_name == 'theta': row[1] = asym
            results.append(row)
        return np.array(results, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# MNE-aware extractor classes
# ═══════════════════════════════════════════════════════════════════════════

class DEExtractor(WindowExtractor):
    """
    Differential Entropy per frequency band per EEG channel.

    Returns (n_epochs * n_wins, n_ch * n_bands) per call to transform().
    Supports baseline correction when epochs contain tmin < 0 (e.g. DEAP).
    """

    def __init__(
        self,
        bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
        window_sec: float = 1.0,
        stride_sec: Optional[float] = None,
        baseline_correct: bool = True,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__(window_sec, stride_sec)
        self.bands   = bands
        self.baseline_correct = baseline_correct
        self.channel_names    = channel_names
        self._sos = {name: _bandpass_sos(lo, hi)
                     for name, (lo, hi) in bands.items()}

    @property
    def feature_names(self) -> List[str]:
        ch = self.channel_names or [f'ch{i}' for i in range(32)]
        return [f'EEG_{c}_de_{b}' for c in ch for b in self.bands]

    # ── override full transform to handle per-epoch baseline ──────────────

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        eeg_ep = epochs.copy()
        with mne.utils.use_log_level('WARNING'):
            eeg_ep.pick_types(eeg=True)

        data, sfreq = self._get_signal(eeg_ep)    # (n_ep, n_ch, n_sig)
        bl_de = self._compute_baseline_de(epochs, sfreq)

        results = []
        for i, ep_data in enumerate(data):
            ep_bl = {k: v[i] for k, v in bl_de.items()} if bl_de else None
            results.append(self._extract_de(ep_data, sfreq, ep_bl))
        return np.vstack(results)

    def _transform_epoch(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        return self._extract_de(data, sfreq, None)

    def _compute_baseline_de(self, epochs: mne.Epochs, sfreq: float
                              ) -> Optional[Dict[str, np.ndarray]]:
        if not self.baseline_correct or epochs.tmin >= 0:
            return None
        tmax_bl = -1.0 / sfreq
        if epochs.tmin >= tmax_bl:
            return None
        bl_ep = epochs.copy()
        with mne.utils.use_log_level('WARNING'):
            bl_ep.pick_types(eeg=True)
            bl_ep = bl_ep.crop(tmax=tmax_bl)
        bl_data = bl_ep.get_data()   # (n_ep, n_ch, n_bl)
        result: Dict[str, np.ndarray] = {}
        for name, sos in self._sos.items():
            bl_filt = sosfiltfilt(sos, bl_data, axis=2)
            var     = np.maximum(np.var(bl_filt, axis=2, ddof=1), 1e-12)
            result[name] = 0.5 * np.log(2 * np.pi * np.e * var)  # (n_ep, n_ch)
        return result

    def _extract_de(self, data: np.ndarray, sfreq: float,
                    baseline: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """data: (n_ch, n_samples) → (n_wins, n_ch * n_bands)"""
        wins   = self._slice_windows(data, sfreq)   # (n_wins, n_ch, win)
        n_wins, n_ch, _ = wins.shape
        n_bands = len(self._sos)
        de_all  = np.zeros((n_wins, n_ch, n_bands), dtype=np.float32)

        for b_idx, (name, sos) in enumerate(self._sos.items()):
            filt = sosfiltfilt(sos, data, axis=1)   # filter full epoch first
            wins_filt = self._slice_windows(filt, sfreq)
            var = np.maximum(np.var(wins_filt, axis=2, ddof=1), 1e-12)
            de  = 0.5 * np.log(2 * np.pi * np.e * var)   # (n_wins, n_ch)
            if baseline is not None:
                de = de - baseline[name][np.newaxis, :]
            de_all[:, :, b_idx] = de

        return de_all.reshape(n_wins, n_ch * n_bands)


class HjorthExtractor(WindowExtractor):
    """
    Hjorth Mobility and Complexity per EEG channel per window.

    Returns (n_epochs * n_wins, n_ch * 2).
    """

    def __init__(
        self,
        window_sec: float = 1.0,
        stride_sec: Optional[float] = None,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__(window_sec, stride_sec)
        self.channel_names = channel_names

    @property
    def feature_names(self) -> List[str]:
        ch = self.channel_names or [f'ch{i}' for i in range(32)]
        return [f'EEG_{c}_{p}' for c in ch for p in ('Hjorth_mob', 'Hjorth_comp')]

    def _transform_epoch(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """data: (n_ch, n_samples) → (n_wins, n_ch * 2)"""
        wins = self._slice_windows(data, sfreq)   # (n_wins, n_ch, win)
        n_wins, n_ch, _ = wins.shape
        out = np.zeros((n_wins, n_ch * 2), dtype=np.float32)
        for w, seg in enumerate(wins):
            mob, comp       = hjorth_params(seg)
            out[w, :n_ch]   = mob
            out[w,  n_ch:]  = comp
        return out


class DEHjorthExtractor(WindowExtractor):
    """
    Combined DE + Hjorth extractor — the standard EEG feature set (v13).

    Features per channel: [de_theta, de_alpha, de_beta, de_gamma,
                           Hjorth_mob, Hjorth_comp]
    Total: n_ch × 6   (192 for DEAP 32-ch, 114 for Barometer 19-ch)

    Works with both DEAP (has baseline tmin < 0) and NeuroBarometer (tmin = 0).
    """

    def __init__(
        self,
        bands: Dict[str, Tuple[float, float]] = DEFAULT_BANDS,
        window_sec: float = 1.0,
        stride_sec: Optional[float] = None,
        baseline_correct: bool = True,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__(window_sec, stride_sec)
        self.bands            = bands
        self.baseline_correct = baseline_correct
        self.channel_names    = channel_names
        self._sos  = {name: _bandpass_sos(lo, hi)
                      for name, (lo, hi) in bands.items()}
        self._feat_per_ch = list(bands.keys()) + ['Hjorth_mob', 'Hjorth_comp']

    @property
    def feature_names(self) -> List[str]:
        ch = self.channel_names or [f'ch{i}' for i in range(32)]
        return [f'EEG_{c}_{f}' for c in ch for f in self._feat_per_ch]

    # ── override transform to pick EEG channels + handle baseline ─────────

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        eeg_ep = epochs.copy()
        with mne.utils.use_log_level('WARNING'):
            eeg_ep.pick_types(eeg=True)

        data, sfreq = self._get_signal(eeg_ep)    # (n_ep, n_ch, n_sig)
        bl_de = self._compute_baseline_de(epochs, sfreq)

        # Lazily update channel_names from epochs (for feature_names property)
        if self.channel_names is None:
            self.channel_names = eeg_ep.ch_names

        results = []
        for i, ep_data in enumerate(data):
            ep_bl = {k: v[i] for k, v in bl_de.items()} if bl_de else None
            results.append(self._extract_epoch(ep_data, sfreq, ep_bl))
        return np.vstack(results)

    def _transform_epoch(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        return self._extract_epoch(data, sfreq, None)

    def _compute_baseline_de(self, epochs: mne.Epochs, sfreq: float
                              ) -> Optional[Dict[str, np.ndarray]]:
        if not self.baseline_correct or epochs.tmin >= 0:
            return None
        tmax_bl = -1.0 / sfreq
        if epochs.tmin >= tmax_bl:
            return None
        bl_ep = epochs.copy()
        with mne.utils.use_log_level('WARNING'):
            bl_ep.pick_types(eeg=True)
            bl_ep = bl_ep.crop(tmax=tmax_bl)
        bl_data = bl_ep.get_data()   # (n_ep, n_ch, n_bl)
        result: Dict[str, np.ndarray] = {}
        for name, sos in self._sos.items():
            filt = sosfiltfilt(sos, bl_data, axis=2)
            var  = np.maximum(np.var(filt, axis=2, ddof=1), 1e-12)
            result[name] = 0.5 * np.log(2 * np.pi * np.e * var)  # (n_ep, n_ch)
        return result

    def _extract_epoch(self, data: np.ndarray, sfreq: float,
                       baseline_de: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        data: (n_ch, n_samples) — signal portion only (t ≥ 0)
        Returns (n_wins, n_ch * n_feat_per_ch)
        """
        wins = self._slice_windows(data, sfreq)   # (n_wins, n_ch, win)
        n_wins, n_ch, _ = wins.shape
        n_bands  = len(self._sos)
        n_feat   = n_bands + 2

        result = np.zeros((n_wins, n_ch, n_feat), dtype=np.float32)

        # DE per band — filter full epoch, then slice
        for b_idx, (name, sos) in enumerate(self._sos.items()):
            filt  = sosfiltfilt(sos, data, axis=1)
            fwins = self._slice_windows(filt, sfreq)
            var   = np.maximum(np.var(fwins, axis=2, ddof=1), 1e-12)
            de    = 0.5 * np.log(2 * np.pi * np.e * var)    # (n_wins, n_ch)
            if baseline_de is not None:
                de = de - baseline_de[name][np.newaxis, :]
            result[:, :, b_idx] = de

        # Hjorth per window
        for w, seg in enumerate(wins):
            mob, comp           = hjorth_params(seg)
            result[w, :, n_bands]     = mob
            result[w, :, n_bands + 1] = comp

        return result.reshape(n_wins, n_ch * n_feat)


class FAAExtractor(EpochExtractor):
    """
    Frontal Alpha Asymmetry (FAA) and Frontal Theta Asymmetry (FTA).

    Computed over the full signal portion (t ≥ 0) — one value per epoch.
    Returns (n_epochs, 2): col 0 = FAA, col 1 = FTA.

    FAA = log(alpha power F4) − log(alpha power F3)
    FTA = log(theta power F4) − log(theta power F3)
    """

    def __init__(
        self,
        left_ch:  str = 'F3',
        right_ch: str = 'F4',
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.left_ch  = left_ch
        self.right_ch = right_ch
        self.bands = bands or {'alpha': (8, 13), 'theta': (5, 7)}
        self._sos  = {name: _bandpass_sos(lo, hi)
                      for name, (lo, hi) in self.bands.items()}

    @property
    def feature_names(self) -> List[str]:
        return ['FAA', 'FTA']

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        n_ep = len(epochs)
        out  = np.zeros((n_ep, 2), dtype=np.float32)

        # Pick the two frontal channels (gracefully skip if absent)
        try:
            ep = epochs.copy()
            with mne.utils.use_log_level('WARNING'):
                ep.pick_channels([self.left_ch, self.right_ch])
        except (ValueError, KeyError):
            return out   # channels not in this dataset

        if ep.tmin < 0:
            with mne.utils.use_log_level('WARNING'):
                ep = ep.crop(tmin=0.0)

        data = ep.get_data()   # (n_ep, 2, n_samples)
        # channel order: left=0, right=1 (we picked in that order)
        idx = {ch: i for i, ch in enumerate(ep.ch_names)}
        l, r = idx[self.left_ch], idx[self.right_ch]

        for name, sos in self._sos.items():
            filt  = sosfiltfilt(sos, data, axis=2)   # (n_ep, 2, n_samp)
            var_l = np.maximum(np.var(filt[:, l, :], axis=1, ddof=1), 1e-12)
            var_r = np.maximum(np.var(filt[:, r, :], axis=1, ddof=1), 1e-12)
            asym  = 0.5 * np.log(var_r) - 0.5 * np.log(var_l)
            if name == 'alpha':
                out[:, 0] = asym
            elif name == 'theta':
                out[:, 1] = asym

        return out
