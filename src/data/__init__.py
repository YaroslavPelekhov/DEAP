"""
Data loading package.

Importers (MNE-based, recommended)
───────────────────────────────────
  deap_importer.load_subject_epochs(sid, data_dir) → mne.Epochs
  deap_importer.load_subjects(sids, data_dir)       → {sid: mne.Epochs}

  barometer_importer.from_arrays(eeg, gsr, ppg, sfreq, ...) → mne.Epochs
  barometer_importer.load_barometer_edf(path, ...)           → mne.Epochs

Channel definitions
───────────────────
  channels.DEAP_CHANNELS         – 32-ch list in DEAP order
  channels.BAROMETER_IN_DEAP     – 19 shared channels (new names)
  channels.DEAP_BAROMETER_INDICES – indices of shared channels in DEAP array
"""

from .channels import (
    DEAP_CHANNELS,
    BAROMETER_CHANNELS,
    BAROMETER_IN_DEAP,
    DEAP_BAROMETER_INDICES,
)

from . import deap_importer, barometer_importer

__all__ = [
    'DEAP_CHANNELS',
    'BAROMETER_CHANNELS',
    'BAROMETER_IN_DEAP',
    'DEAP_BAROMETER_INDICES',
    'deap_importer',
    'barometer_importer',
]
