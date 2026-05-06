"""
DEAP dataset → mne.Epochs

Converts raw .dat files into standard MNE Epochs objects so that all
downstream feature extractors work with a single unified format.

Structure of returned Epochs
────────────────────────────
  n_epochs  : 40 (one per video trial)
  n_channels: 34  (32 EEG + GSR + PPG)
  tmin      : -3.0 s  (pre-stimulus baseline kept)
  tmax      :  60.0 s - 1 sample

Channel types
  EEG  → 'eeg'   (32 channels, 10-20 names, standard_1020 montage)
  GSR  → 'misc'
  PPG  → 'misc'

Metadata (pandas DataFrame, one row per epoch)
  trial       : 0-indexed trial number
  valence_raw : continuous rating 1–9
  arousal_raw : continuous rating 1–9
  dominance   : continuous rating 1–9
  liking      : continuous rating 1–9
  valence     : binary 0/1 (above subject's own median)
  arousal     : binary 0/1 (above subject's own median)
  event_id    : 1-4 encoding (val*2 + ar + 1)

Event mapping
  1 = LV_LA  (low valence, low arousal)
  2 = LV_HA
  3 = HV_LA
  4 = HV_HA
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import mne

from .channels import DEAP_CHANNELS

# ── Constants ────────────────────────────────────────────────────────────────
SFREQ            = 128
BASELINE_SAMPLES = 384          # first 3 s (pre-stimulus)
TOTAL_SAMPLES    = 8064         # 63 s total
SIGNAL_SAMPLES   = TOTAL_SAMPLES - BASELINE_SAMPLES  # 7680 = 60 s

TMIN = -BASELINE_SAMPLES / SFREQ   # -3.0 s
TMAX = SIGNAL_SAMPLES  / SFREQ - 1 / SFREQ   # 59.992 s

EEG_IDX  = list(range(32))
GSR_IDX  = 36
PPG_IDX  = 38

EVENT_ID = {
    'LV_LA': 1,
    'LV_HA': 2,
    'HV_LA': 3,
    'HV_HA': 4,
}

# ── MNE channel info ─────────────────────────────────────────────────────────

def _make_info() -> mne.Info:
    ch_names = DEAP_CHANNELS + ['GSR', 'PPG']
    ch_types = ['eeg'] * 32 + ['misc', 'misc']
    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    with mne.utils.use_log_level('WARNING'):
        info.set_montage(montage, on_missing='ignore')
    return info


_INFO = _make_info()   # built once, reused


# ── Binary label helper ───────────────────────────────────────────────────────

def _binarize(ratings: np.ndarray) -> np.ndarray:
    """
    Binarise valence & arousal using each subject's own median.
    Returns (n_trials, 2) int array: col0=valence, col1=arousal.
    Using per-subject median keeps class balance ~50/50 regardless
    of individual rating biases.
    """
    val = ratings[:, 0]
    ar  = ratings[:, 1]
    return np.stack([
        (val > np.median(val)).astype(np.int64),
        (ar  > np.median(ar )).astype(np.int64),
    ], axis=1)


# ── Core loader ───────────────────────────────────────────────────────────────

def load_subject_epochs(
    subject_id: int,
    data_dir: Path | str,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Load one DEAP subject as mne.Epochs.

    Parameters
    ----------
    subject_id : 1-indexed subject number (1–32)
    data_dir   : folder with s01.dat … s32.dat
    verbose    : MNE verbosity

    Returns
    -------
    mne.Epochs  with 40 epochs, metadata DataFrame, standard_1020 montage
    """
    data_dir = Path(data_dir)
    path = data_dir / f's{subject_id:02d}.dat'
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    data   = raw['data'].astype(np.float32)    # (40, 40, 8064)
    labels = raw['labels'].astype(np.float32)  # (40, 4)
    n_trials = data.shape[0]

    # ── Build data array: (40, 34, 8064) ─────────────────────────────────────
    eeg = data[:, EEG_IDX, :]            # (40, 32, 8064)
    gsr = data[:, GSR_IDX, :][:, None]  # (40,  1, 8064)
    ppg = data[:, PPG_IDX, :][:, None]  # (40,  1, 8064)
    epochs_data = np.concatenate([eeg, gsr, ppg], axis=1)  # (40, 34, 8064)

    # ── Binary labels & event array ───────────────────────────────────────────
    bin_labels = _binarize(labels)  # (40, 2)
    event_codes = bin_labels[:, 0] * 2 + bin_labels[:, 1] + 1  # 1-4

    # MNE events: (n_events, 3) — [sample, 0, event_id]
    # We fake one event per epoch at sample 0 (EpochsArray doesn't need real times)
    events = np.column_stack([
        np.arange(n_trials) * TOTAL_SAMPLES,
        np.zeros(n_trials, dtype=int),
        event_codes,
    ])

    # ── Metadata DataFrame ────────────────────────────────────────────────────
    metadata = pd.DataFrame({
        'subject':      subject_id,
        'trial':        np.arange(n_trials),
        'valence_raw':  labels[:, 0],
        'arousal_raw':  labels[:, 1],
        'dominance':    labels[:, 2],
        'liking':       labels[:, 3],
        'valence':      bin_labels[:, 0],
        'arousal':      bin_labels[:, 1],
        'event_id':     event_codes,
    })

    # ── Create EpochsArray ────────────────────────────────────────────────────
    log_level = 'WARNING' if not verbose else 'INFO'
    with mne.utils.use_log_level(log_level):
        epochs = mne.EpochsArray(
            data      = epochs_data,
            info      = _INFO,
            events    = events,
            tmin      = TMIN,
            event_id  = EVENT_ID,
            metadata  = metadata,
            baseline  = None,   # baseline stored in tmin:0, applied in extractors
            verbose   = verbose,
        )
    return epochs


def load_subjects(
    subject_ids: List[int],
    data_dir: Path | str,
    verbose: bool = False,
) -> dict[int, mne.Epochs]:
    """
    Load multiple subjects.

    Returns {subject_id: mne.Epochs}
    """
    data_dir = Path(data_dir)
    result = {}
    for sid in subject_ids:
        print(f'  Loading s{sid:02d} ...', end='\r')
        result[sid] = load_subject_epochs(sid, data_dir, verbose=verbose)
    print(f'  Loaded {len(result)} subjects.          ')
    return result
