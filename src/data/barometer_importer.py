"""
NeuroBarometer → mne.Epochs

Converts NeuroBarometer recordings into the same mne.Epochs format
as deap_importer.py, so all feature extractors work identically.

NeuroBarometer specs
────────────────────
  EEG  : 20 channels (see BAROMETER_CHANNELS in channels.py)
  GSR  : 1 channel
  PPG  : 1 channel
  sfreq: configurable (default 250 Hz, resampled to 128 Hz to match DEAP)

Supported input formats
───────────────────────
  1. EDF / BDF           → mne.io.read_raw_edf / read_raw_bdf
  2. NumPy arrays        → from_arrays()
  3. CSV                 → from_csv()

Metadata / labels
─────────────────
  Labels must be provided externally (questionnaire responses or annotations).
  Pass as a DataFrame with columns: valence, arousal (binary 0/1).
  If not provided, epochs are created without event labels (useful for inference).

Usage
─────
  # From EDF file
  epochs = load_barometer_edf('recording.edf', labels_df=df)

  # From numpy arrays (shape: n_channels × n_samples)
  epochs = from_arrays(eeg=arr, gsr=gsr, ppg=ppg, sfreq=250, labels_df=df)

  # Inference mode (no labels)
  epochs = from_arrays(eeg=arr, gsr=gsr, ppg=ppg, sfreq=250)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import mne

from .channels import BAROMETER_IN_DEAP

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_SFREQ = 128      # resample to match DEAP
N_BARO_CH    = len(BAROMETER_IN_DEAP)  # 19 EEG channels

EVENT_ID = {
    'LV_LA': 1,
    'LV_HA': 2,
    'HV_LA': 3,
    'HV_HA': 4,
    'unknown': 0,
}


# ── Channel info builder ──────────────────────────────────────────────────────

def _make_info(sfreq: float = TARGET_SFREQ) -> mne.Info:
    ch_names = BAROMETER_IN_DEAP + ['GSR', 'PPG']
    ch_types = ['eeg'] * N_BARO_CH + ['misc', 'misc']
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    montage = mne.channels.make_standard_montage('standard_1020')
    with mne.utils.use_log_level('WARNING'):
        info.set_montage(montage, on_missing='ignore')
    return info


# ── Labels helper ─────────────────────────────────────────────────────────────

def _events_from_labels(labels_df: Optional[pd.DataFrame], n_epochs: int,
                        n_samples_per_epoch: int) -> tuple[np.ndarray, dict]:
    """Build MNE events array from labels DataFrame."""
    if labels_df is None:
        codes = np.ones(n_epochs, dtype=int)   # code=1, unnamed
        event_id = {'unknown': 1}
    else:
        val = labels_df['valence'].values.astype(int)
        ar  = labels_df['arousal'].values.astype(int)
        codes = val * 2 + ar + 1
        # Only include event names that actually appear
        present = set(codes.tolist())
        event_id = {k: v for k, v in EVENT_ID.items() if v in present}

    events = np.column_stack([
        np.arange(n_epochs) * n_samples_per_epoch,
        np.zeros(n_epochs, dtype=int),
        codes,
    ])
    return events, event_id


# ── Public API ────────────────────────────────────────────────────────────────

def from_arrays(
    eeg: np.ndarray,
    gsr: np.ndarray,
    ppg: np.ndarray,
    sfreq: float,
    labels_df: Optional[pd.DataFrame] = None,
    epoch_len_sec: Optional[float] = None,
    tmin: float = 0.0,
    resample: bool = True,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Create mne.Epochs from raw numpy arrays.

    Parameters
    ----------
    eeg        : (n_eeg_ch, n_samples) or (n_epochs, n_eeg_ch, n_samples)
    gsr        : (n_samples,)          or (n_epochs, n_samples)
    ppg        : (n_samples,)          or (n_epochs, n_samples)
    sfreq      : recording sample rate
    labels_df  : DataFrame with columns [valence, arousal] per epoch, or None
    epoch_len_sec : if eeg is 2D (continuous), split into epochs of this length
    tmin       : epoch start time relative to event
    resample   : resample to TARGET_SFREQ (128 Hz) if sfreq differs
    """
    # ── Normalise to 3D: (n_epochs, n_ch, n_samples) ─────────────────────────
    if eeg.ndim == 2:
        # Continuous recording — split into epochs
        if epoch_len_sec is None:
            raise ValueError('epoch_len_sec required for continuous (2D) input')
        n_samples_ep = int(epoch_len_sec * sfreq)
        n_epochs = eeg.shape[1] // n_samples_ep
        eeg = eeg[:, :n_epochs * n_samples_ep].reshape(
            eeg.shape[0], n_epochs, n_samples_ep).transpose(1, 0, 2)
        gsr = gsr[:n_epochs * n_samples_ep].reshape(n_epochs, n_samples_ep)
        ppg = ppg[:n_epochs * n_samples_ep].reshape(n_epochs, n_samples_ep)
    else:
        n_epochs = eeg.shape[0]

    assert eeg.shape[1] == N_BARO_CH, (
        f'Expected {N_BARO_CH} EEG channels ({BAROMETER_IN_DEAP}), '
        f'got {eeg.shape[1]}'
    )

    # ── Concatenate channels: (n_epochs, n_eeg+2, n_samples) ─────────────────
    data = np.concatenate([
        eeg.astype(np.float32),
        gsr[:, None, :].astype(np.float32),
        ppg[:, None, :].astype(np.float32),
    ], axis=1)

    # ── Build events & metadata ───────────────────────────────────────────────
    events, event_id = _events_from_labels(labels_df, n_epochs, data.shape[2])
    metadata = labels_df.copy() if labels_df is not None else None

    # ── Create EpochsArray ────────────────────────────────────────────────────
    info = _make_info(sfreq=sfreq)
    log_level = 'WARNING' if not verbose else 'INFO'
    with mne.utils.use_log_level(log_level):
        epochs = mne.EpochsArray(
            data=data, info=info, events=events,
            tmin=tmin, event_id=event_id,
            metadata=metadata, baseline=None,
        )
        if resample and abs(sfreq - TARGET_SFREQ) > 1:
            epochs = epochs.resample(TARGET_SFREQ, verbose=verbose)

    return epochs


def load_barometer_edf(
    path: Path | str,
    eeg_ch_names: list[str],
    gsr_ch_name: str,
    ppg_ch_name: str,
    labels_df: Optional[pd.DataFrame] = None,
    epoch_len_sec: float = 60.0,
    tmin: float = 0.0,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Load a NeuroBarometer EDF recording as mne.Epochs.

    Parameters
    ----------
    path         : path to .edf file
    eeg_ch_names : list of EEG channel names in the EDF (must match BAROMETER_IN_DEAP order)
    gsr_ch_name  : name of GSR channel in EDF
    ppg_ch_name  : name of PPG channel in EDF
    labels_df    : DataFrame with valence/arousal labels per epoch
    epoch_len_sec: length of each epoch in seconds
    """
    path = Path(path)
    with mne.utils.use_log_level('WARNING'):
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=verbose)

    sfreq = raw.info['sfreq']

    # Extract channels
    all_picks = eeg_ch_names + [gsr_ch_name, ppg_ch_name]
    raw.pick_channels(all_picks)

    data = raw.get_data()   # (n_ch, n_samples)
    eeg = data[:len(eeg_ch_names)]
    gsr = data[len(eeg_ch_names)]
    ppg = data[len(eeg_ch_names) + 1]

    return from_arrays(
        eeg=eeg, gsr=gsr, ppg=ppg, sfreq=sfreq,
        labels_df=labels_df, epoch_len_sec=epoch_len_sec,
        tmin=tmin, verbose=verbose,
    )
