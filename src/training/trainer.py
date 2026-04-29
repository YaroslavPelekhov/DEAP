"""
Training loops: Subject-Dependent (GroupKFold) and LOSO.

Usage:
    from src.training.trainer import train_sd, train_loso

    sd_results = train_sd(features, model_name='multimodal')
    loso_results = train_loso(features, model_name='multimodal')
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_metrics, majority_vote, aggregate_subject_results
from ..models.factory import create_model, count_params

# ── Defaults ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS       = 80
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 15
SD_N_FOLDS   = 8


def _make_loader(e, p, g, l, batch_size, shuffle):
    ds = TensorDataset(
        torch.tensor(e, dtype=torch.float32),
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(g, dtype=torch.float32),
        torch.tensor(l, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _scale(tr_e, tr_p, tr_g, te_e, te_p, te_g):
    def fit_transform(tr, te):
        sc = StandardScaler().fit(tr)
        return sc.transform(tr), sc.transform(te)
    tr_e, te_e = fit_transform(tr_e, te_e)
    tr_p, te_p = fit_transform(tr_p, te_p)
    tr_g, te_g = fit_transform(tr_g, te_g)
    return tr_e, tr_p, tr_g, te_e, te_p, te_g


def _class_weights(labels, device):
    n = labels.shape[0]
    wv = torch.tensor(n / (2 * np.bincount(labels[:, 0], minlength=2).clip(1)),
                      dtype=torch.float32).to(device)
    wa = torch.tensor(n / (2 * np.bincount(labels[:, 1], minlength=2).clip(1)),
                      dtype=torch.float32).to(device)
    return nn.CrossEntropyLoss(weight=wv), nn.CrossEntropyLoss(weight=wa)


def _eval_window(model, loader, device, mask_eeg=True, mask_ppg=True, mask_gsr=True):
    model.eval()
    all_vp, all_ap, all_vt, all_at = [], [], [], []
    with torch.no_grad():
        for e, p, g, l in loader:
            e, p, g, l = e.to(device), p.to(device), g.to(device), l.to(device)
            if not mask_eeg: e = torch.zeros_like(e)
            if not mask_ppg: p = torch.zeros_like(p)
            if not mask_gsr: g = torch.zeros_like(g)
            v, a = model(e, p, g)
            all_vp.append(v.argmax(1).cpu())
            all_ap.append(a.argmax(1).cpu())
            all_vt.append(l[:, 0].cpu())
            all_at.append(l[:, 1].cpu())
    vp = torch.cat(all_vp).numpy(); ap = torch.cat(all_ap).numpy()
    vt = torch.cat(all_vt).numpy(); at = torch.cat(all_at).numpy()
    return np.stack([vp, ap], axis=1), np.stack([vt, at], axis=1)


def _train_one_fold(
    tr_e, tr_p, tr_g, tr_l,
    te_e, te_p, te_g, te_l, te_groups,
    model_name: str, device,
    save_path: Optional[Path] = None,
) -> tuple[dict, nn.Module]:
    tr_e, tr_p, tr_g, te_e, te_p, te_g = _scale(tr_e, tr_p, tr_g, te_e, te_p, te_g)
    tr_ld = _make_loader(tr_e, tr_p, tr_g, tr_l, BATCH_SIZE, shuffle=True)
    te_ld = _make_loader(te_e, te_p, te_g, te_l, BATCH_SIZE, shuffle=False)

    model = create_model(model_name, in_eeg=tr_e.shape[1],
                         in_ppg=tr_p.shape[1], in_gsr=tr_g.shape[1]).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/100)
    crit_val, crit_ar = _class_weights(tr_l, device)

    best_acc, best_state, patience_ctr = -1.0, None, 0
    for _ in range(1, EPOCHS + 1):
        model.train()
        for e, p, g, l in tr_ld:
            e, p, g, l = e.to(device), p.to(device), g.to(device), l.to(device)
            opt.zero_grad()
            v, a = model(e, p, g)
            (crit_val(v, l[:, 0]) + crit_ar(a, l[:, 1])).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        preds, truths = _eval_window(model, te_ld, device)
        acc = (compute_metrics(truths, preds)['mean_acc'])
        if acc > best_acc:
            best_acc, best_state, patience_ctr = acc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    model.load_state_dict(best_state)

    # Majority vote metrics
    preds, truths = _eval_window(model, te_ld, device)
    vp_voted = majority_vote(preds[:, 0], te_groups)
    ap_voted = majority_vote(preds[:, 1], te_groups)
    vt_voted = majority_vote(truths[:, 0], te_groups)
    at_voted = majority_vote(truths[:, 1], te_groups)
    metrics = compute_metrics(
        np.stack([vt_voted, at_voted], axis=1),
        np.stack([vp_voted, ap_voted], axis=1),
    )

    if save_path:
        from ..utils.io import save_model
        save_model(model, save_path,
                   meta={'in_eeg': tr_e.shape[1], 'in_ppg': tr_p.shape[1],
                         'in_gsr': tr_g.shape[1], 'model': model_name})
    return metrics, model


def train_sd(
    features: Dict[int, dict],
    model_name: str = 'multimodal',
    n_folds: int = SD_N_FOLDS,
    device: torch.device = DEVICE,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Subject-dependent training with GroupKFold.

    Returns:
        {'aggregate': {...}, 'subjects': {sid: {...}}}
    """
    subject_results = {}
    gkf = GroupKFold(n_splits=n_folds)

    for sid, data in features.items():
        eeg = data['eeg']
        ppg = data['ppg']
        gsr = data['gsr']
        lbl = data['labels_win']
        grp = data['groups']

        fold_metrics = []
        for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(eeg, lbl, grp)):
            save_path = (models_dir / f's{sid:02d}_fold{fold_i}_{model_name}.pt'
                         if models_dir else None)
            metrics, _ = _train_one_fold(
                eeg[tr_idx], ppg[tr_idx], gsr[tr_idx], lbl[tr_idx],
                eeg[te_idx], ppg[te_idx], gsr[te_idx], lbl[te_idx],
                grp[te_idx], model_name, device, save_path,
            )
            fold_metrics.append(metrics)
            if verbose:
                print(f'  s{sid:02d} fold{fold_i+1}/{n_folds}: '
                      f'Val {metrics["valence_acc"]:.1f}%  Ar {metrics["arousal_acc"]:.1f}%')

        subject_results[sid] = aggregate_subject_results(fold_metrics)

    # Aggregate across subjects
    agg_keys = list(next(iter(subject_results.values())).keys())
    aggregate = {k: float(np.mean([v[k] for v in subject_results.values()])) for k in agg_keys}

    return {'aggregate': aggregate, 'subjects': subject_results}


def train_loso(
    features: Dict[int, dict],
    model_name: str = 'multimodal',
    device: torch.device = DEVICE,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Subject-Out training.

    Returns:
        {'aggregate': {...}, 'subjects': {sid: {...}}}
    """
    subject_ids = sorted(features.keys())
    subject_results = {}

    for sid in subject_ids:
        train_ids = [s for s in subject_ids if s != sid]

        def _pool(key):
            return np.vstack([features[s][key] for s in train_ids])
        def _pool1d(key):
            return np.concatenate([features[s][key] for s in train_ids])

        tr_e = _pool('eeg'); tr_p = _pool('ppg'); tr_g = _pool('gsr')
        tr_l = _pool('labels_win')
        te_e = features[sid]['eeg']; te_p = features[sid]['ppg']
        te_g = features[sid]['gsr']; te_l = features[sid]['labels_win']
        te_grp = features[sid]['groups']

        save_path = (models_dir / f'loso_s{sid:02d}_{model_name}.pt'
                     if models_dir else None)
        metrics, _ = _train_one_fold(
            tr_e, tr_p, tr_g, tr_l,
            te_e, te_p, te_g, te_l, te_grp,
            model_name, device, save_path,
        )
        subject_results[sid] = metrics
        if verbose:
            print(f'  LOSO s{sid:02d}: Val {metrics["valence_acc"]:.1f}%  Ar {metrics["arousal_acc"]:.1f}%')

    agg_keys = list(next(iter(subject_results.values())).keys())
    aggregate = {k: float(np.mean([v[k] for v in subject_results.values()])) for k in agg_keys}

    return {'aggregate': aggregate, 'subjects': subject_results}
