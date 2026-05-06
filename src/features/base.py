"""
Base class for all feature extractors.

Every extractor follows the same contract:
  - takes mne.Epochs as input
  - returns np.ndarray of shape (n_samples, n_features)
  - exposes feature_names list

Two flavours:
  WindowExtractor  — splits each epoch into windows, returns (n_epochs*n_wins, n_feats)
  EpochExtractor   — one feature vector per epoch,   returns (n_epochs, n_feats)

The pipeline aligns them by repeating EpochExtractor output n_wins times.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import mne


class BaseExtractor(ABC):
    """Abstract base for all extractors."""

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Names of the output features (length == n_features)."""

    @abstractmethod
    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        """
        Compute features.

        Parameters
        ----------
        epochs : mne.Epochs

        Returns
        -------
        np.ndarray  shape depends on flavour:
          WindowExtractor → (n_epochs * n_wins, n_features)
          EpochExtractor  → (n_epochs, n_features)
        """

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self.feature_names)} features)'


class WindowExtractor(BaseExtractor):
    """
    Base for extractors that operate on sliding windows within each epoch.

    Subclasses implement _transform_epoch(eeg, sfreq) → (n_wins, n_feats)
    where eeg is (n_ch, n_samples) for a single epoch (baseline excluded).
    """

    def __init__(self, window_sec: float = 1.0, stride_sec: float | None = None):
        self.window_sec = window_sec
        self.stride_sec = stride_sec or window_sec

    def _get_signal(self, epochs: mne.Epochs) -> tuple[np.ndarray, float]:
        """
        Extract signal data from epochs, excluding baseline (t < 0).
        Returns (n_epochs, n_ch, n_samples), sfreq.
        """
        sfreq = epochs.info['sfreq']
        # Crop to signal portion only (t >= 0)
        ep = epochs.copy()
        if ep.tmin < 0:
            ep = ep.crop(tmin=0.0)
        return ep.get_data(), sfreq

    @abstractmethod
    def _transform_epoch(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Parameters
        ----------
        data  : (n_ch, n_samples) single epoch signal
        sfreq : sampling frequency

        Returns
        -------
        (n_wins, n_features)
        """

    def transform(self, epochs: mne.Epochs) -> np.ndarray:
        data, sfreq = self._get_signal(epochs)
        results = []
        for ep_data in data:
            results.append(self._transform_epoch(ep_data, sfreq))
        return np.vstack(results)   # (n_epochs * n_wins, n_features)

    def n_windows(self, epochs: mne.Epochs) -> int:
        """Number of windows per epoch."""
        data, sfreq = self._get_signal(epochs)
        win = int(self.window_sec * sfreq)
        stride = int(self.stride_sec * sfreq)
        n_samp = data.shape[2]
        return (n_samp - win) // stride + 1

    def _slice_windows(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Slice a single epoch into windows.

        Parameters
        ----------
        data  : (n_ch, n_samples)
        sfreq : sampling frequency

        Returns
        -------
        (n_wins, n_ch, win_samples)
        """
        win    = int(self.window_sec * sfreq)
        stride = int(self.stride_sec * sfreq)
        n_samp = data.shape[1]
        n_wins = (n_samp - win) // stride + 1
        return np.stack([data[:, w * stride: w * stride + win]
                         for w in range(n_wins)])


class EpochExtractor(BaseExtractor):
    """
    Base for extractors that produce one feature vector per epoch
    (e.g. PPG HRV, GSR EDA — computed on the full trial signal).
    """

    def _get_channel(self, epochs: mne.Epochs, ch_name: str) -> np.ndarray:
        """
        Extract one channel from epochs as (n_epochs, n_samples).
        Baseline (t < 0) is excluded.
        """
        ep = epochs.copy().pick_channels([ch_name])
        if ep.tmin < 0:
            ep = ep.crop(tmin=0.0)
        return ep.get_data()[:, 0, :]   # (n_epochs, n_samples)
