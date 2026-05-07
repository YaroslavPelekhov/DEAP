"""
Feature extraction pipelines.

MNEFeaturePipeline  — works with mne.Epochs (DEAP or NeuroBarometer)
FeaturePipeline     — legacy DEAP .dat pipeline (kept for backward compat)

MNEFeaturePipeline
──────────────────
Runs a list of extractors on mne.Epochs, aligns WindowExtractor and
EpochExtractor outputs (repeating trial-level features n_wins times),
and returns results per modality + concatenated array.

Usage
─────
    # From a config list
    pipe = MNEFeaturePipeline.from_config([
        {'name': 'de_hjorth', 'window_sec': 1.0},
        {'name': 'faa'},
        {'name': 'hrv'},
        {'name': 'eda'},
    ])

    # Or build manually
    from src.features.factory import ExtractorFactory
    pipe = MNEFeaturePipeline(ExtractorFactory.default_deap())

    result = pipe.transform(epochs)
    # result keys:
    #   'DEHjorthExtractor' : (n_ep * n_wins, n_eeg_feats)
    #   'FAAExtractor'      : (n_ep * n_wins, 2)    ← repeated
    #   'HRVExtractor'      : (n_ep * n_wins, 10)   ← repeated
    #   'EDAExtractor'      : (n_ep * n_wins, 8)    ← repeated
    #   'features'          : (n_ep * n_wins, total) ← concatenated
    #   'groups'            : (n_ep * n_wins,)        ← trial index
    #   'labels'            : (n_ep * n_wins, 2)      ← if metadata present
    #
    # For separate-modality access:
    split = pipe.transform_split(epochs)
    # split keys: 'eeg', 'ppg', 'gsr', 'faa', 'groups', 'labels'
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mne

from .base import BaseExtractor, WindowExtractor, EpochExtractor
from .eeg import EEGExtractor, DEFAULT_BANDS
from .ppg import extract_ppg_subject
from .gsr import extract_gsr_subject


# ── DEAP channel constants (legacy pipeline) ──────────────────────────────
EEG_CH  = list(range(32))
GSR_CH  = 36
PPG_CH  = 38
BASELINE_SAMPLES = 384   # first 3 s at 128 Hz
SFREQ   = 128


# ═══════════════════════════════════════════════════════════════════════════
# New: MNE-based pipeline
# ═══════════════════════════════════════════════════════════════════════════

class MNEFeaturePipeline:
    """
    Feature extraction pipeline for mne.Epochs.

    Handles alignment between WindowExtractor (n_epochs * n_wins rows) and
    EpochExtractor (n_epochs rows) by repeating trial-level features n_wins times.

    Parameters
    ----------
    extractors : list of BaseExtractor instances
    """

    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: list[dict]) -> 'MNEFeaturePipeline':
        """Build from a list of {'name': ..., **kwargs} dicts."""
        from .factory import ExtractorFactory
        return cls(ExtractorFactory.from_config(config))

    @classmethod
    def default_deap(cls, window_sec: float = 1.0) -> 'MNEFeaturePipeline':
        from .factory import ExtractorFactory
        return cls(ExtractorFactory.default_deap(window_sec))

    @classmethod
    def default_barometer(cls, window_sec: float = 1.0) -> 'MNEFeaturePipeline':
        from .factory import ExtractorFactory
        return cls(ExtractorFactory.default_barometer(window_sec))

    # ── Core ──────────────────────────────────────────────────────────────

    def transform(self, epochs: mne.Epochs) -> dict[str, np.ndarray]:
        """
        Run all extractors and return aligned results.

        Returns
        -------
        dict with keys:
          '<ClassName>' : per-extractor array (all aligned to n_epochs * n_wins)
          'features'    : (n_epochs * n_wins, total_feats) — concatenated
          'groups'      : (n_epochs * n_wins,)             — trial index
          'labels'      : (n_epochs * n_wins, 2)           — valence/arousal
                          (omitted if metadata absent or lacks those columns)
        """
        n_epochs = len(epochs)
        raw: dict[str, np.ndarray] = {}

        # 1. Run all extractors
        for ext in self.extractors:
            key       = ext.__class__.__name__
            raw[key]  = ext.transform(epochs)

        # 2. Determine n_wins from the first WindowExtractor
        n_wins = 1
        for ext, arr in zip(self.extractors, raw.values()):
            if isinstance(ext, WindowExtractor) and arr.shape[0] > n_epochs:
                n_wins = arr.shape[0] // n_epochs
                break

        # 3. Align: repeat EpochExtractor results n_wins times
        aligned: dict[str, np.ndarray] = {}
        for key, arr in raw.items():
            if arr.shape[0] == n_epochs and n_wins > 1:
                aligned[key] = np.repeat(arr, n_wins, axis=0)
            else:
                aligned[key] = arr

        # 4. Build output
        result = dict(aligned)
        result['features'] = np.concatenate(list(aligned.values()), axis=1)
        result['groups']   = np.repeat(np.arange(n_epochs, dtype=np.int32), n_wins)

        # Labels from metadata (valence / arousal binary columns)
        if (epochs.metadata is not None
                and 'valence' in epochs.metadata.columns
                and 'arousal' in epochs.metadata.columns):
            labs = epochs.metadata[['valence', 'arousal']].values.astype(np.int64)
            result['labels'] = np.repeat(labs, n_wins, axis=0)

        return result

    def transform_split(self, epochs: mne.Epochs) -> dict[str, np.ndarray]:
        """
        Run all extractors and return a modality-split dict.

        Modality assignment is based on extractor type:
          DEHjorthExtractor / DEExtractor / HjorthExtractor → 'eeg'
          HRVExtractor                                       → 'ppg'
          EDAExtractor                                       → 'gsr'
          FAAExtractor                                       → 'faa'

        Returns
        -------
        dict with keys: 'eeg', 'ppg', 'gsr', 'faa' (only those present),
                        'groups', 'labels' (if metadata available)
        """
        from .eeg import DEHjorthExtractor, DEExtractor, HjorthExtractor, FAAExtractor
        from .ppg import HRVExtractor
        from .gsr import EDAExtractor

        _EEG_TYPES = (DEHjorthExtractor, DEExtractor, HjorthExtractor)
        _MODALITY_MAP = {
            DEHjorthExtractor: 'eeg',
            DEExtractor:       'eeg',
            HjorthExtractor:   'eeg',
            FAAExtractor:      'faa',
            HRVExtractor:      'ppg',
            EDAExtractor:      'gsr',
        }

        full = self.transform(epochs)
        n_epochs = len(epochs)
        n_wins   = full['groups'].shape[0] // n_epochs

        split: dict[str, np.ndarray] = {}
        for ext in self.extractors:
            key      = ext.__class__.__name__
            modality = _MODALITY_MAP.get(type(ext), key)
            arr      = full[key]                # already aligned
            if modality in split:
                split[modality] = np.concatenate([split[modality], arr], axis=1)
            else:
                split[modality] = arr

        split['groups'] = full['groups']
        if 'labels' in full:
            split['labels'] = full['labels']

        return split

    def __repr__(self) -> str:
        names = ', '.join(e.__class__.__name__ for e in self.extractors)
        return f'MNEFeaturePipeline([{names}])'


# ═══════════════════════════════════════════════════════════════════════════
# Legacy: DEAP .dat pipeline (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════

def _load_subject(data_dir: Path, sid: int) -> dict:
    """Load raw .dat file for subject sid (1-indexed)."""
    path = data_dir / f's{sid:02d}.dat'
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    return {'data': raw['data'], 'labels': raw['labels']}


class FeaturePipeline:
    """
    Full feature extraction + caching for raw DEAP .dat files.

    cache_version: increment when feature logic changes.

    Note: new code should prefer MNEFeaturePipeline + deap_importer.
    """

    def __init__(
        self,
        data_dir: Path,
        cache_dir: Path,
        cache_version: str = 'v14',
        bands: dict = DEFAULT_BANDS,
        win_sec: float = 2.0,
        stride_sec: Optional[float] = 1.0,
        channel_subset: Optional[List[int]] = None,
        fs: int = SFREQ,
    ):
        self.data_dir  = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version   = cache_version
        self.fs        = fs
        self.ch_subset = channel_subset

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
        """Extract raw per-modality features (no derived extras or z-score yet).

        Returns
        -------
        dict with keys:
          eeg   : (n_trials*n_wins, 192)
          ppg   : (n_trials*n_wins, 12)
          gsr   : (n_trials*n_wins, 8)   — 8 base features only
          labels: (n_trials, 2)
          groups: (n_trials*n_wins,)
        """
        raw    = _load_subject(self.data_dir, sid)
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
        # Note: FAA/FTA and consensus are added in run() from EEG feature array

        # Binary labels (per-subject median)
        val_bin = (labels[:, 0] > np.median(labels[:, 0])).astype(np.int64)
        ar_bin  = (labels[:, 1] > np.median(labels[:, 1])).astype(np.int64)
        raw_labels = np.stack([val_bin, ar_bin], axis=1)    # (40, 2)

        return {
            'eeg':    eeg_feats,
            'ppg':    ppg_feats,
            'gsr':    gsr_feats,
            'labels': raw_labels,
            'groups': groups,
        }

    def run(
        self,
        subject_ids: List[int],
        force: bool = False,
    ) -> Dict[int, dict]:
        """
        Extract features for all subjects, add derived extras, and apply z-score.

        Post-extraction additions (appended to GSR → 13 total):
          FAA, FTA      — frontal alpha/theta asymmetry from EEG feature array
          cons_val      — cross-subject consensus valence label (mean of others)
          cons_ar       — cross-subject consensus arousal label
          position      — window index normalised to [0, 1] within trial

        Per-subject z-score (EEG, PPG, first 8 GSR) applied after base extraction.
        """
        cache = self._cache_path(subject_ids)
        if not force and cache.exists():
            print(f'[FeaturePipeline] Loading from cache: {cache.name}')
            with open(cache, 'rb') as f:
                return pickle.load(f)

        # ── Pass 1: extract raw per-subject features ──────────────────────
        features: Dict[int, dict] = {}
        for sid in subject_ids:
            print(f'  Extracting s{sid:02d}...', end=' ', flush=True)
            features[sid] = self._extract_one(sid)
            print(
                f"EEG {features[sid]['eeg'].shape}  "
                f"PPG {features[sid]['ppg'].shape}  "
                f"GSR {features[sid]['gsr'].shape}"
            )

        # ── Pass 2: derived features + per-subject z-score ────────────────
        from ..data.channels import FAA_LEFT, FAA_RIGHT

        # Infer layout constants from first subject
        d0     = features[subject_ids[0]]
        n_wins = d0['eeg'].shape[0] // d0['labels'].shape[0]  # e.g. 2360//40 = 59

        # Channel indices (local, not global) for FAA
        ch = self.ch_subset if self.ch_subset else list(range(32))
        n_feat_per_ch = self.eeg_extractor.n_feat_per_ch   # 6
        # band indices in feature vector: theta=0, alpha=1, beta=2, gamma=3
        alpha_band_idx = list(self.eeg_extractor.bands.keys()).index('alpha')
        theta_band_idx = list(self.eeg_extractor.bands.keys()).index('theta')

        def _local(global_idx):
            return ch.index(global_idx) if global_idx in ch else None

        l_local = _local(FAA_LEFT[0])
        r_local = _local(FAA_RIGHT[0])

        # All subjects' binary labels for consensus computation
        bin_labels = {sid: features[sid]['labels'] for sid in subject_ids}

        for sid in subject_ids:
            d        = features[sid]
            n_trials = d['labels'].shape[0]
            eeg_f    = d['eeg']           # (n_trials*n_wins, n_eeg_feats)
            n_rows   = eeg_f.shape[0]

            # ── Per-subject z-score (EEG, PPG, 8 base GSR) ────────────────
            for key in ('eeg', 'ppg'):
                m = d[key].mean(axis=0)
                s = d[key].std(axis=0) + 1e-8
                d[key] = ((d[key] - m) / s).astype(np.float32)

            gsr8 = d['gsr'][:, :8]
            m    = gsr8.mean(axis=0)
            s    = gsr8.std(axis=0) + 1e-8
            d['gsr'][:, :8] = ((gsr8 - m) / s).astype(np.float32)

            # ── FAA / FTA from z-scored EEG feature array ─────────────────
            # Layout: [ch0_f0, ch0_f1, ..., ch1_f0, ...] (interleaved per ch)
            if l_local is not None and r_local is not None:
                al_L = d['eeg'][:, l_local * n_feat_per_ch + alpha_band_idx]
                al_R = d['eeg'][:, r_local * n_feat_per_ch + alpha_band_idx]
                th_L = d['eeg'][:, l_local * n_feat_per_ch + theta_band_idx]
                th_R = d['eeg'][:, r_local * n_feat_per_ch + theta_band_idx]
                faa  = (al_R - al_L).reshape(-1, 1).astype(np.float32)
                fta  = (th_R - th_L).reshape(-1, 1).astype(np.float32)
            else:
                faa = fta = np.zeros((n_rows, 1), dtype=np.float32)

            # ── Position: window index in [0, 1] within trial ─────────────
            pos_trial = np.linspace(0.0, 1.0, n_wins, dtype=np.float32)
            position  = np.tile(pos_trial, n_trials).reshape(-1, 1)   # (n_rows, 1)

            # ── Consensus: cross-subject mean binary labels ────────────────
            other_labels = np.stack(
                [bin_labels[s] for s in subject_ids if s != sid],
                axis=0,
            ).astype(np.float32)                   # (n_subjects-1, n_trials, 2)
            cons = other_labels.mean(axis=0)       # (n_trials, 2)
            cons_win = np.repeat(cons, n_wins, axis=0).astype(np.float32)  # (n_rows, 2)

            # ── Assemble final GSR (13): [8 base | FAA | FTA | cons | pos] ─
            d['gsr'] = np.concatenate(
                [d['gsr'], faa, fta, cons_win, position], axis=1
            ).astype(np.float32)

            # ── labels_win (always aligned with n_wins) ────────────────────
            d['labels_win'] = np.repeat(d['labels'], n_wins, axis=0)

            features[sid] = d
            print(f'  s{sid:02d} post-processed: '
                  f"EEG {d['eeg'].shape} PPG {d['ppg'].shape} GSR {d['gsr'].shape}")

        with open(cache, 'wb') as f:
            pickle.dump(features, f, protocol=4)
        print(f'[FeaturePipeline] Saved: {cache}')
        return features
