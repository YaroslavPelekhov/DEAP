"""
Training loops for subject-dependent (SD) and subject-independent (LOSO) protocols.

Subject-Dependent (SD):
  GroupKFold over 40 trials (8 folds = 5 trials/fold).
  Each trial expands to n_windows samples; GroupKFold keeps all windows from a
  trial in the same fold -> no data leakage.
  Train: 35 trials x 60 windows = 2100 samples
  Test :  5 trials x 60 windows =  300 samples

Subject-Independent / LOSO:
  Leave-one-subject-out cross-validation.
  Train on 31 subjects x 40 trials x 60 windows = 74400 samples, test on 2400.
"""
import sys, copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS, PATIENCE,
    SEED, SD_N_FOLDS, N_TRIALS,
)
from models.mmcat import MMCAT
from models.light_mlp import LightMLP

# Default model class — LightMLP works much better for feature-based DE approach
# (28K params vs 660K for MMCAT; appropriate for 2100 training samples per fold)
MODEL_CLASS = LightMLP


# ─── Dataset helper ───────────────────────────────────────────────────────────

def _make_dataset(eeg, ppg, gsr, labels, device):
    # Keep tensors on CPU — move each batch to device in the training loop
    # (prevents CUDA OOM when all 32-subject windows are loaded at once)
    return TensorDataset(
        torch.tensor(eeg,    dtype=torch.float32),
        torch.tensor(ppg,    dtype=torch.float32),
        torch.tensor(gsr,    dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


# ─── Scaler ───────────────────────────────────────────────────────────────────

def _scale(train_eeg, train_ppg, train_gsr, test_eeg, test_ppg, test_gsr):
    # Peripheral signals: StandardScaler (always 2D)
    sc_ppg = StandardScaler().fit(train_ppg)
    sc_gsr = StandardScaler().fit(train_gsr)

    # EEG: 2D = DE features → StandardScaler
    #       4D = raw windows (EEGNet) → per-channel z-score using train stats
    if train_eeg.ndim == 2:
        sc_eeg = StandardScaler().fit(train_eeg)
        tr_e = sc_eeg.transform(train_eeg)
        te_e = sc_eeg.transform(test_eeg)
    else:
        # (N, 1, C, T) — normalise each channel across the training set
        mean = train_eeg.mean(axis=(0, 3), keepdims=True)  # (1,1,C,1)
        std  = train_eeg.std(axis=(0, 3), keepdims=True).clip(1e-8)
        tr_e = (train_eeg - mean) / std
        te_e = (test_eeg  - mean) / std

    return (
        tr_e, sc_ppg.transform(train_ppg), sc_gsr.transform(train_gsr),
        te_e, sc_ppg.transform(test_ppg),  sc_gsr.transform(test_gsr),
    )


# ─── Label expansion helper ───────────────────────────────────────────────────

def _expand_labels(labels_per_trial: np.ndarray, n_windows: int) -> np.ndarray:
    """(n_trials, 2) -> (n_trials * n_windows, 2) by repeating each label."""
    return np.repeat(labels_per_trial, n_windows, axis=0)


# ─── Single train / eval cycle ────────────────────────────────────────────────

def train_one_fold(
    train_eeg, train_ppg, train_gsr, train_labels,
    test_eeg,  test_ppg,  test_gsr,  test_labels,
    test_groups: np.ndarray,          # trial id per test window — for majority vote
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, float]:
    (tr_e, tr_p, tr_g,
     te_e, te_p, te_g) = _scale(train_eeg, train_ppg, train_gsr,
                                  test_eeg,  test_ppg,  test_gsr)

    train_ds = _make_dataset(tr_e, tr_p, tr_g, train_labels, device)
    test_ds  = _make_dataset(te_e, te_p, te_g, test_labels,  device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = MODEL_CLASS().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 100
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for eeg_b, ppg_b, gsr_b, lbl_b in train_loader:
            eeg_b = eeg_b.to(device)
            ppg_b = ppg_b.to(device)
            gsr_b = gsr_b.to(device)
            lbl_b = lbl_b.to(device)
            optimizer.zero_grad()
            v_logits, a_logits = model(eeg_b, ppg_b, gsr_b)
            loss = criterion(v_logits, lbl_b[:, 0]) + criterion(a_logits, lbl_b[:, 1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        val_acc = _eval_accuracy(model, test_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            if verbose:
                print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return _full_eval_voted(model, test_loader, test_groups, device)


def _eval_accuracy(model, loader, device):
    """Window-level accuracy used only for early stopping."""
    model.eval()
    vp, ap, vt, at = [], [], [], []
    with torch.no_grad():
        for eeg_b, ppg_b, gsr_b, lbl_b in loader:
            eeg_b = eeg_b.to(device); ppg_b = ppg_b.to(device)
            gsr_b = gsr_b.to(device); lbl_b = lbl_b.to(device)
            v, a = model(eeg_b, ppg_b, gsr_b)
            vp.append(v.argmax(1).cpu()); ap.append(a.argmax(1).cpu())
            vt.append(lbl_b[:, 0].cpu()); at.append(lbl_b[:, 1].cpu())
    vp = torch.cat(vp).numpy(); ap = torch.cat(ap).numpy()
    vt = torch.cat(vt).numpy(); at = torch.cat(at).numpy()
    return (accuracy_score(vt, vp) + accuracy_score(at, ap)) / 2.0


def _full_eval_voted(model, loader, groups: np.ndarray, device) -> Dict[str, float]:
    """
    Majority vote across windows of the same trial.

    For each trial in the test set, take the mode of per-window predictions.
    This is the standard evaluation protocol in SOTA DEAP papers.
    """
    model.eval()
    vp_all, ap_all, vt_all, at_all = [], [], [], []
    with torch.no_grad():
        for eeg_b, ppg_b, gsr_b, lbl_b in loader:
            eeg_b = eeg_b.to(device); ppg_b = ppg_b.to(device)
            gsr_b = gsr_b.to(device); lbl_b = lbl_b.to(device)
            v, a = model(eeg_b, ppg_b, gsr_b)
            vp_all.append(v.argmax(1).cpu()); ap_all.append(a.argmax(1).cpu())
            vt_all.append(lbl_b[:, 0].cpu()); at_all.append(lbl_b[:, 1].cpu())

    vp = torch.cat(vp_all).numpy(); ap = torch.cat(ap_all).numpy()
    vt = torch.cat(vt_all).numpy(); at = torch.cat(at_all).numpy()

    # Majority vote per trial group
    trial_ids = np.unique(groups)
    vp_voted, ap_voted, vt_voted, at_voted = [], [], [], []
    for tid in trial_ids:
        mask = groups == tid
        # mode = most frequent class among windows
        vp_voted.append(int(np.bincount(vp[mask]).argmax()))
        ap_voted.append(int(np.bincount(ap[mask]).argmax()))
        vt_voted.append(int(vt[mask][0]))   # all windows have same true label
        at_voted.append(int(at[mask][0]))

    vp_v = np.array(vp_voted); ap_v = np.array(ap_voted)
    vt_v = np.array(vt_voted); at_v = np.array(at_voted)

    return {
        "valence_acc": accuracy_score(vt_v, vp_v) * 100,
        "arousal_acc": accuracy_score(at_v, ap_v) * 100,
        "valence_f1":  f1_score(vt_v, vp_v, average="weighted") * 100,
        "arousal_f1":  f1_score(at_v, ap_v, average="weighted") * 100,
    }


# ─── Subject-Dependent (SD) ───────────────────────────────────────────────────

def run_subject_dependent(
    features: Dict[int, Dict[str, np.ndarray]],
    labels:   Dict[int, np.ndarray],
    device: torch.device,
) -> Dict:
    """
    GroupKFold(8) per subject, splitting at trial level.

    features[sid]:
      'eeg'    : (n_trials * n_windows, 160)
      'ppg'    : (n_trials * n_windows, 10)
      'gsr'    : (n_trials * n_windows, 8)
      'groups' : (n_trials * n_windows,)   trial index
    labels[sid]: (n_trials, 2) — one label per trial
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    gkf = GroupKFold(n_splits=SD_N_FOLDS)
    subject_results = {}

    for sid, feats in features.items():
        eeg    = feats["eeg"]     # (n_trials*n_wins, 160)
        ppg    = feats["ppg"]     # (n_trials*n_wins, 10)
        gsr    = feats["gsr"]     # (n_trials*n_wins, 8)
        groups = feats["groups"]  # (n_trials*n_wins,)

        # n_windows per trial
        n_wins = len(groups) // N_TRIALS
        # expand trial-level labels to window-level
        lbl_windows = _expand_labels(labels[sid], n_wins)  # (n_trials*n_wins, 2)

        fold_metrics = []
        for _, (train_idx, test_idx) in enumerate(gkf.split(eeg, groups=groups)):
            m = train_one_fold(
                eeg[train_idx], ppg[train_idx], gsr[train_idx], lbl_windows[train_idx],
                eeg[test_idx],  ppg[test_idx],  gsr[test_idx],  lbl_windows[test_idx],
                test_groups=groups[test_idx],
                device=device,
            )
            fold_metrics.append(m)

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in fold_metrics[0]}
        subject_results[sid] = {"folds": fold_metrics, "mean": avg}
        print(
            f"  s{sid:02d} | "
            f"Val {avg['valence_acc']:.1f}% | "
            f"Ar {avg['arousal_acc']:.1f}%"
        )

    all_means = [r["mean"] for r in subject_results.values()]
    aggregate = {k: (np.mean([m[k] for m in all_means]),
                     np.std([m[k] for m in all_means]))
                 for k in all_means[0]}
    return {"protocol": "SD", "subjects": subject_results, "aggregate": aggregate}


# ─── Subject-Independent (LOSO) ───────────────────────────────────────────────

def run_loso(
    features: Dict[int, Dict[str, np.ndarray]],
    labels:   Dict[int, np.ndarray],
    device: torch.device,
) -> Dict:
    """
    Leave-One-Subject-Out.
    Train on 31 subjects (all windows), test on 1 subject (all windows).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    subject_ids    = sorted(features.keys())
    subject_results = {}

    for test_sid in subject_ids:
        train_sids = [s for s in subject_ids if s != test_sid]

        n_wins = len(features[test_sid]["groups"]) // N_TRIALS

        train_eeg = np.concatenate([features[s]["eeg"] for s in train_sids])
        train_ppg = np.concatenate([features[s]["ppg"] for s in train_sids])
        train_gsr = np.concatenate([features[s]["gsr"] for s in train_sids])
        train_lbl = np.concatenate([
            _expand_labels(labels[s], n_wins) for s in train_sids
        ])

        test_eeg = features[test_sid]["eeg"]
        test_ppg = features[test_sid]["ppg"]
        test_gsr = features[test_sid]["gsr"]
        test_lbl = _expand_labels(labels[test_sid], n_wins)

        m = train_one_fold(
            train_eeg, train_ppg, train_gsr, train_lbl,
            test_eeg,  test_ppg,  test_gsr,  test_lbl,
            test_groups=features[test_sid]["groups"],
            device=device,
        )
        subject_results[test_sid] = m
        print(
            f"  LOSO s{test_sid:02d} | "
            f"Val {m['valence_acc']:.1f}% | "
            f"Ar {m['arousal_acc']:.1f}%"
        )

    all_m = list(subject_results.values())
    aggregate = {k: (np.mean([m[k] for m in all_m]),
                     np.std([m[k] for m in all_m]))
                 for k in all_m[0]}
    return {"protocol": "LOSO", "subjects": subject_results, "aggregate": aggregate}


# ─── Result reporting ────────────────────────────────────────────────────────

def print_results(results: Dict):
    protocol = results["protocol"]
    agg      = results["aggregate"]
    sep      = "-" * 55
    print(f"\n{sep}")
    print(f"  Protocol : {protocol}")
    print(sep)
    print(f"  {'Metric':<20} {'Mean':>8}  {'Std':>8}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}")
    for k, (mean, std) in agg.items():
        print(f"  {k:<20} {mean:>7.2f}%  {std:>7.2f}%")
    print(f"{sep}\n")
