"""
FeaturePipeline: full feature extraction for DEAP subjects with caching.

Usage:
    pipeline = FeaturePipeline(data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    features = pipeline.run(subject_ids=[1, 2, ..., 32])
    # Returns dict: {sid: {'eeg': ndarray, 'ppg': ndarray, 'gsr': ndarray,
    #                      'labels': ndarray, 'groups': ndarray}}
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .eeg import EEGExtractor, DEFAULT_BANDS
from .ppg import extract_ppg_subject
from .gsr import extract_gsr_subject

# DEAP channel indices
EEG_CH  = list(range(32))
GSR_CH  = 36
PPG_CH  = 38
BASELINE_SAMPLES = 384   # first 3 s at 128 Hz
SFREQ = 128


def _load_subject(data_dir: Path, sid: int) -> dict:
    """Load raw .dat file for subject sid (1-indexed)."""
    import pickle as pkl
    path = data_dir / f's{sid:02d}.dat'
    with open(path, 'rb') as f:
        raw = pkl.load(f, encoding='latin1')
    # data: (40, 40, 8064), labels: (40, 4)
    return {'data': raw['data'], 'labels': raw['labels']}


class FeaturePipeline:
    """
    Full feature extraction + caching.

    cache_version: increment when feature logic changes
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        cache_version: str = 'v13',
        bands: dict = DEFAULT_BANDS,
        win_sec: float = 1.0,
        stride_sec: Optional[float] = None,
        channel_subset: Optional[List[int]] = None,   # e.g. DEAP_BAROMETER_INDICES
        fs: int = SFREQ,
    ):
        self.data_dir  = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version   = cache_version
        self.fs        = fs
        self.ch_subset = channel_subset  # None = use all 32

        from ..data.channels import DEAP_CHANNELS
        ch_names = ([DEAP_CHANNELS[i] for i in channel_subset]
                    if channel_subset else DEAP_CHANNELS)
        self.eeg_extractor = EEGExtractor(
            bands=bands, fs=fs,
            win_sec=win_sec, stride_sec=stride_sec,
            channel_names=ch_names,
        )

    def _cache_path(self, subject_ids: List[int]) -> Path:
        ids_str = '_'.join(str(s) for s in subject_ids)
        return self.cache_dir / f'features_{self.version}_{ids_str}.pkl'

    def _extract_one(self, sid: int) -> dict:
        raw = _load_subject(self.data_dir, sid)
        data   = raw['data'].astype(np.float32)    # (40, 40, 8064)
        labels = raw['labels']                      # (40, 4)

        ch = self.ch_subset if self.ch_subset else list(range(32))
        eeg_trials = data[:, ch, BASELINE_SAMPLES:]                # (40, n_ch, 7680)
        baselines  = data[:, ch, :BASELINE_SAMPLES]                # (40, n_ch, 384)
        ppg_trials = data[:, PPG_CH, BASELINE_SAMPLES:]            # (40, 7680)
        gsr_trials = data[:, GSR_CH, BASELINE_SAMPLES:]            # (40, 7680)

        eeg_feats, groups = self.eeg_extractor.extract_subject(eeg_trials, baselines)
        n_wins = eeg_feats.shape[0] // len(eeg_trials)

        ppg_feats = extract_ppg_subject(ppg_trials, n_windows=n_wins, fs=self.fs)
        gsr_feats = extract_gsr_subject(gsr_trials, n_windows=n_wins, fs=self.fs)

        # FAA / FTA (frontal asymmetry) — append to GSR as extra features
        from ..data.channels import DEAP_CHANNELS, FAA_LEFT, FAA_RIGHT
        # Map global DEAP indices to local ch subset indices
        def _local(global_idx):
            return ch.index(global_idx) if global_idx in ch else None
        l_idx = _local(FAA_LEFT[0])
        r_idx = _local(FAA_RIGHT[0])
        if l_idx is not None and r_idx is not None:
            faa_fta = self.eeg_extractor.compute_faa_fta(eeg_trials, l_idx, r_idx)
            faa_fta_wins = np.repeat(faa_fta, n_wins, axis=0)
        else:
            faa_fta_wins = np.zeros((eeg_feats.shape[0], 2), dtype=np.float32)
        gsr_feats = np.concatenate([gsr_feats, faa_fta_wins], axis=1)

        # Binary labels (threshold at 5)
        val_bin = (labels[:, 0] >= 5).astype(np.int64)
        ar_bin  = (labels[:, 1] >= 5).astype(np.int64)
        raw_labels = np.stack([val_bin, ar_bin], axis=1)     # (40, 2)
        labels_win = np.repeat(raw_labels, n_wins, axis=0)   # (40*n_wins, 2)

        return {
            'eeg':        eeg_feats,
            'ppg':        ppg_feats,
            'gsr':        gsr_feats,
            'labels':     raw_labels,
            'labels_win': labels_win,
            'groups':     groups,
        }

    def run(
        self,
        subject_ids: List[int],
        force: bool = False,
    ) -> Dict[int, dict]:
        cache = self._cache_path(subject_ids)
        if not force and cache.exists():
            print(f'[FeaturePipeline] Loading from cache: {cache.name}')
            with open(cache, 'rb') as f:
                return pickle.load(f)

        features = {}
        for sid in subject_ids:
            print(f'  Extracting s{sid:02d}...', end=' ', flush=True)
            features[sid] = self._extract_one(sid)
            print(f"EEG {features[sid]['eeg'].shape}  PPG {features[sid]['ppg'].shape}  GSR {features[sid]['gsr'].shape}")

        with open(cache, 'wb') as f:
            pickle.dump(features, f, protocol=4)
        print(f'[FeaturePipeline] Saved: {cache}')
        return features
