"""
Evaluation metrics for DEAP binary classification.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Args:
        y_true: (N, 2) — [valence, arousal]
        y_pred: (N, 2) — [valence, arousal]

    Returns dict with keys: valence_acc, arousal_acc, valence_f1, arousal_f1, mean_acc
    """
    va = accuracy_score(y_true[:, 0], y_pred[:, 0]) * 100
    aa = accuracy_score(y_true[:, 1], y_pred[:, 1]) * 100
    vf = f1_score(y_true[:, 0], y_pred[:, 0], average='weighted', zero_division=0) * 100
    af = f1_score(y_true[:, 1], y_pred[:, 1], average='weighted', zero_division=0) * 100
    return {
        'valence_acc': round(va, 2),
        'arousal_acc': round(aa, 2),
        'valence_f1':  round(vf, 2),
        'arousal_f1':  round(af, 2),
        'mean_acc':    round((va + aa) / 2, 2),
    }


def majority_vote(
    preds: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Aggregate window-level predictions to trial-level by majority vote.

    Args:
        preds:  (N,) predicted class per window
        groups: (N,) trial index per window

    Returns:
        voted: (n_trials,) trial-level predictions
    """
    trial_ids = np.unique(groups)
    voted = np.array([
        int(np.bincount(preds[groups == tid]).argmax())
        for tid in trial_ids
    ])
    return voted


def aggregate_subject_results(per_fold: list[dict]) -> dict:
    """
    Average fold metrics for a single subject.
    per_fold: list of dicts from compute_metrics()
    """
    keys = list(per_fold[0].keys())
    return {k: float(np.mean([f[k] for f in per_fold])) for k in keys}
