"""
DEAP dataset loader.

Loads raw .dat files, strips baseline, returns raw signals split by modality.
Does NOT perform feature extraction — that lives in features/.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR, N_SUBJECTS, N_TRIALS, EEG_CHANNELS,
    GSR_CHANNEL, PPG_CHANNEL,
    BASELINE_SAMPLES, LABEL_THRESHOLD, LABEL_NAMES,
)


def load_subject(subject_id: int) -> Dict:
    """
    Load a single subject .dat file.

    Returns dict with keys:
      'eeg'   : (40, 32, 7680)  float64  — EEG, baseline removed
      'ppg'   : (40, 7680)      float64  — PPG/BVP, baseline removed
      'gsr'   : (40, 7680)      float64  — GSR/EDA, baseline removed
      'labels': (40, 4)         float32  — valence, arousal, dominance, liking
    """
    path = DATA_DIR / f"s{subject_id:02d}.dat"
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    data = raw["data"].astype(np.float64)   # (40, 40, 8064)
    labels = raw["labels"].astype(np.float32)  # (40, 4)

    return {
        "eeg":          data[:, EEG_CHANNELS, BASELINE_SAMPLES:],   # (40, 32, 7680)
        "baseline_eeg": data[:, EEG_CHANNELS, :BASELINE_SAMPLES],   # (40, 32, 384)
        "ppg":          data[:, PPG_CHANNEL,  BASELINE_SAMPLES:],   # (40, 7680)
        "gsr":          data[:, GSR_CHANNEL,  BASELINE_SAMPLES:],   # (40, 7680)
        "labels": labels,
    }


def get_binary_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert continuous ratings to binary labels using subject-specific median.

    Using the subject's own median (instead of a fixed threshold of 5) ensures
    ~50/50 class balance per subject, which is critical for DEAP where rating
    distributions vary widely across subjects.

    Columns 0=valence, 1=arousal.
    Returns (N, 2) int64 array: 1 = High (above median), 0 = Low.
    """
    val = labels[:, 0]
    ar  = labels[:, 1]
    return np.stack([
        (val > np.median(val)).astype(np.int64),
        (ar  > np.median(ar )).astype(np.int64),
    ], axis=1)


def load_all_subjects(
    subject_ids: List[int] | None = None,
    verbose: bool = True,
) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
    """
    Load raw signals for all (or selected) subjects.

    Returns:
      signals_by_subject : {subj_id: {eeg, ppg, gsr, labels}}
      labels_by_subject  : {subj_id: (40, 2) binary labels}
    """
    if subject_ids is None:
        subject_ids = list(range(1, N_SUBJECTS + 1))

    signals, bin_labels = {}, {}
    for sid in subject_ids:
        if verbose:
            print(f"  Loading s{sid:02d}.dat ...", end="\r")
        subj = load_subject(sid)
        signals[sid] = subj
        bin_labels[sid] = get_binary_labels(subj["labels"])

    if verbose:
        print(f"  Loaded {len(subject_ids)} subjects.           ")
    return signals, bin_labels
