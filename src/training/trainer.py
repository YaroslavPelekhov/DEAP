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
from ..models.dann import DANNNet, dann_alpha

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


# ── DANN helpers ───────────────────────────────────────────────────────────────

def _fit_scale(tr: np.ndarray, te: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr)
    return sc.transform(tr), sc.transform(te)


def _make_loader_with_subjects(e, p, g, l, s, batch_size, shuffle):
    """Like _make_loader but includes per-sample subject labels."""
    from torch.utils.data import TensorDataset
    ds = TensorDataset(
        torch.tensor(e, dtype=torch.float32),
        torch.tensor(p, dtype=torch.float32),
        torch.tensor(g, dtype=torch.float32),
        torch.tensor(l, dtype=torch.long),
        torch.tensor(s, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_loso_dann(
    features: Dict[int, dict],
    device: torch.device = DEVICE,
    models_dir: Optional[Path] = None,
    domain_weight: float = 0.3,
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Subject-Out with Domain-Adversarial Neural Network adaptation.

    The shared encoder is regularised via gradient reversal to produce
    subject-invariant features, improving cross-subject generalisation.

    Parameters
    ----------
    features      : output of FeaturePipeline.run()
    domain_weight : weight of subject classification loss (λ).
                    0 = pure emotion loss (= regular LOSO).
                    Recommended sweep: [0.1, 0.2, 0.3, 0.5].
    verbose       : print per-subject results

    Returns
    -------
    {'aggregate': {...}, 'subjects': {sid: {...}}}
    """
    subject_ids     = sorted(features.keys())
    subject_results = {}

    for sid in subject_ids:
        train_ids  = [s for s in subject_ids if s != sid]
        n_subjects = len(train_ids)

        # ── Pool training data, assign local subject index 0…n-1 ──────────
        parts = {k: [] for k in ('e', 'p', 'g', 'l', 's')}
        for local_idx, tsid in enumerate(train_ids):
            d = features[tsid]
            n = d['eeg'].shape[0]
            parts['e'].append(d['eeg'])
            parts['p'].append(d['ppg'])
            parts['g'].append(d['gsr'])
            parts['l'].append(d['labels_win'])
            parts['s'].append(np.full(n, local_idx, dtype=np.int64))

        tr_e = np.vstack(parts['e']); tr_p = np.vstack(parts['p'])
        tr_g = np.vstack(parts['g']); tr_l = np.vstack(parts['l'])
        tr_s = np.concatenate(parts['s'])

        te_e = features[sid]['eeg']; te_p = features[sid]['ppg']
        te_g = features[sid]['gsr']; te_l = features[sid]['labels_win']
        te_grp = features[sid]['groups']

        # ── Normalise ──────────────────────────────────────────────────────
        tr_e, te_e = _fit_scale(tr_e, te_e)
        tr_p, te_p = _fit_scale(tr_p, te_p)
        tr_g, te_g = _fit_scale(tr_g, te_g)

        # ── Loaders ───────────────────────────────────────────────────────
        tr_ld = _make_loader_with_subjects(tr_e, tr_p, tr_g, tr_l, tr_s,
                                           BATCH_SIZE, shuffle=True)
        te_ld = _make_loader(te_e, te_p, te_g, te_l, BATCH_SIZE, shuffle=False)

        # ── Model ─────────────────────────────────────────────────────────
        model = DANNNet(
            in_eeg=tr_e.shape[1], in_ppg=tr_p.shape[1], in_gsr=tr_g.shape[1],
            n_subjects=n_subjects,
        ).to(device)

        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/100)
        crit_val, crit_ar = _class_weights(tr_l, device)
        crit_subj = nn.CrossEntropyLoss()

        total_steps = EPOCHS * len(tr_ld)
        global_step = 0
        best_acc, best_state, patience_ctr = -1.0, None, 0

        for _ in range(1, EPOCHS + 1):
            model.train()
            for e_b, p_b, g_b, l_b, s_b in tr_ld:
                e_b, p_b, g_b, l_b, s_b = (x.to(device)
                                            for x in (e_b, p_b, g_b, l_b, s_b))
                alpha = dann_alpha(global_step, total_steps)
                global_step += 1

                opt.zero_grad()
                val_out, ar_out, subj_out = model(e_b, p_b, g_b, alpha=alpha)
                loss = (crit_val(val_out, l_b[:, 0])
                        + crit_ar(ar_out, l_b[:, 1])
                        + domain_weight * crit_subj(subj_out, s_b))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            preds, truths = _eval_window(model, te_ld, device)
            acc = compute_metrics(truths, preds)['mean_acc']
            if acc > best_acc:
                best_acc, best_state, patience_ctr = acc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        model.load_state_dict(best_state)

        # ── Majority-vote evaluation ───────────────────────────────────────
        preds, truths = _eval_window(model, te_ld, device)
        vp = majority_vote(preds[:, 0],  te_grp)
        ap = majority_vote(preds[:, 1],  te_grp)
        vt = majority_vote(truths[:, 0], te_grp)
        at = majority_vote(truths[:, 1], te_grp)
        metrics = compute_metrics(
            np.stack([vt, at], axis=1),
            np.stack([vp, ap], axis=1),
        )
        subject_results[sid] = metrics

        if verbose:
            print(f'  DANN s{sid:02d}: '
                  f'Val {metrics["valence_acc"]:.1f}%  Ar {metrics["arousal_acc"]:.1f}%  '
                  f'(lam={domain_weight}, alpha_final={dann_alpha(global_step, total_steps):.2f})')

        if models_dir:
            from ..utils.io import save_model
            save_model(model, models_dir / f'loso_dann_s{sid:02d}.pt',
                       meta={'in_eeg': tr_e.shape[1], 'in_ppg': tr_p.shape[1],
                             'in_gsr': tr_g.shape[1], 'n_subjects': n_subjects,
                             'domain_weight': domain_weight})

    agg_keys  = list(next(iter(subject_results.values())).keys())
    aggregate = {k: float(np.mean([v[k] for v in subject_results.values()]))
                 for k in agg_keys}
    return {'aggregate': aggregate, 'subjects': subject_results}


# ═══════════════════════════════════════════════════════════════════════════
# Temporal LOSO — TemporalNet processes full trial sequences (B, T, feats)
# ═══════════════════════════════════════════════════════════════════════════

def _reshape_trials(arr: np.ndarray, n_wins: int) -> np.ndarray:
    """(n_trials * n_wins, n_feats) → (n_trials, n_wins, n_feats)"""
    n_trials = arr.shape[0] // n_wins
    return arr.reshape(n_trials, n_wins, arr.shape[1])


def _scale_3d(tr: np.ndarray, te: np.ndarray):
    """Scale (n_trials, n_wins, n_feats) by fitting on flattened train data."""
    shape_tr, shape_te = tr.shape, te.shape
    tr_flat, te_flat = _fit_scale(tr.reshape(-1, shape_tr[-1]),
                                  te.reshape(-1, shape_te[-1]))
    return tr_flat.reshape(shape_tr), te_flat.reshape(shape_te)


def _eval_temporal(model, loader, device):
    """Evaluate TemporalNet; returns (preds, truths) both (n_trials, 2)."""
    model.eval()
    vp_list, ap_list, vt_list, at_list = [], [], [], []
    with torch.no_grad():
        for e, p, g, l in loader:
            e, p, g = e.to(device), p.to(device), g.to(device)
            v, a = model(e, p, g)
            vp_list.append(v.argmax(1).cpu())
            ap_list.append(a.argmax(1).cpu())
            vt_list.append(l[:, 0])
            at_list.append(l[:, 1])
    vp = torch.cat(vp_list).numpy(); ap = torch.cat(ap_list).numpy()
    vt = torch.cat(vt_list).numpy(); at = torch.cat(at_list).numpy()
    return np.stack([vp, ap], axis=1), np.stack([vt, at], axis=1)


def train_loso_temporal(
    features: Dict[int, dict],
    device: torch.device = DEVICE,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    LOSO training with TemporalNet (Bi-GRU over full trial sequences).

    Key difference from train_loso():
      • Data reshaped to (n_trials, n_wins, n_feats) — one row per trial
      • TemporalNet outputs one prediction per trial directly
      • No majority vote needed; batch size = n_trials (not n_wins)

    Expected: ~78% Val / ~65% Ar on DEAP (matches v13 notebook).
    """
    # ── Notebook-matched hyperparameters ──────────────────────────────────────
    LR_T         = 1e-3    # notebook: 1e-3
    BATCH_SIZE_T = 16      # notebook: 16 (trial batches)
    EPOCHS_T     = 100     # notebook: 100
    PATIENCE_T   = 15      # notebook: 15

    subject_ids = sorted(features.keys())
    subject_results = {}

    # Infer n_wins from first subject
    d0 = features[subject_ids[0]]
    n_wins = d0['eeg'].shape[0] // d0['labels'].shape[0]   # e.g. 2360 // 40 = 59

    for sid in subject_ids:
        train_ids = [s for s in subject_ids if s != sid]

        # ── Pool training trials ───────────────────────────────────────────
        tr_e = np.vstack([_reshape_trials(features[s]['eeg'], n_wins)
                          for s in train_ids])   # (31*40, n_wins, 192)
        tr_p = np.vstack([_reshape_trials(features[s]['ppg'], n_wins)
                          for s in train_ids])
        tr_g = np.vstack([_reshape_trials(features[s]['gsr'], n_wins)
                          for s in train_ids])
        tr_l = np.vstack([features[s]['labels'] for s in train_ids])   # (1240, 2)

        te_e = _reshape_trials(features[sid]['eeg'], n_wins)   # (40, n_wins, 192)
        te_p = _reshape_trials(features[sid]['ppg'], n_wins)
        te_g = _reshape_trials(features[sid]['gsr'], n_wins)
        te_l = features[sid]['labels']                          # (40, 2)

        # ── Normalise per-feature across training windows ──────────────────
        tr_e, te_e = _scale_3d(tr_e, te_e)
        tr_p, te_p = _scale_3d(tr_p, te_p)
        tr_g, te_g = _scale_3d(tr_g, te_g)

        # ── Loaders (batch = trials, not windows) ──────────────────────────
        tr_ld = DataLoader(
            TensorDataset(torch.tensor(tr_e, dtype=torch.float32),
                          torch.tensor(tr_p, dtype=torch.float32),
                          torch.tensor(tr_g, dtype=torch.float32),
                          torch.tensor(tr_l, dtype=torch.long)),
            batch_size=BATCH_SIZE_T, shuffle=True, drop_last=False,
        )
        te_ld = DataLoader(
            TensorDataset(torch.tensor(te_e, dtype=torch.float32),
                          torch.tensor(te_p, dtype=torch.float32),
                          torch.tensor(te_g, dtype=torch.float32),
                          torch.tensor(te_l, dtype=torch.long)),
            batch_size=40, shuffle=False, drop_last=False,
        )

        # ── Model ─────────────────────────────────────────────────────────
        model = create_model('temporal',
                             in_eeg=tr_e.shape[-1],
                             in_ppg=tr_p.shape[-1],
                             in_gsr=tr_g.shape[-1]).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=LR_T, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_T,
                                                           eta_min=LR_T / 100)
        crit_val, crit_ar = _class_weights(tr_l, device)

        best_acc, best_state, patience_ctr = -1.0, None, 0

        for _ in range(1, EPOCHS_T + 1):
            model.train()
            for e_b, p_b, g_b, l_b in tr_ld:
                e_b, p_b, g_b, l_b = (x.to(device)
                                      for x in (e_b, p_b, g_b, l_b))
                opt.zero_grad()
                v, a = model(e_b, p_b, g_b)
                (crit_val(v, l_b[:, 0]) + crit_ar(a, l_b[:, 1])).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            preds, truths = _eval_temporal(model, te_ld, device)
            acc = compute_metrics(truths, preds)['mean_acc']
            if acc > best_acc:
                best_acc, best_state, patience_ctr = acc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE_T:
                break

        model.load_state_dict(best_state)
        preds, truths = _eval_temporal(model, te_ld, device)
        metrics = compute_metrics(truths, preds)
        subject_results[sid] = metrics

        if verbose:
            print(f'  Temporal s{sid:02d}: '
                  f'Val {metrics["valence_acc"]:.1f}%  Ar {metrics["arousal_acc"]:.1f}%')

        if models_dir:
            from ..utils.io import save_model
            save_model(model, models_dir / f'loso_temporal_s{sid:02d}.pt',
                       meta={'in_eeg': tr_e.shape[-1], 'in_ppg': tr_p.shape[-1],
                             'in_gsr': tr_g.shape[-1], 'n_wins': n_wins})

    agg_keys  = list(next(iter(subject_results.values())).keys())
    aggregate = {k: float(np.mean([v[k] for v in subject_results.values()]))
                 for k in agg_keys}
    return {'aggregate': aggregate, 'subjects': subject_results}


# ═══════════════════════════════════════════════════════════════════════════
# TemporalDANN — Bi-GRU trial embedding + GRL at trial level
# ═══════════════════════════════════════════════════════════════════════════

def train_loso_dann_temporal(
    features: Dict[int, dict],
    device: torch.device = DEVICE,
    models_dir: Optional[Path] = None,
    domain_weight: float = 0.3,
    verbose: bool = True,
) -> dict:
    """
    LOSO with TemporalNet + Domain-Adversarial adaptation at trial level.

    Architecture: same Bi-GRU as TemporalNet, GRL on trial embedding h (B,128).
    This is the semantically correct level: subject-specific physiology
    (baseline HR, resting EEG power) manifests as a consistent offset across
    the whole trial — captured in h, not individual 1-second windows.

    Parameters
    ----------
    features      : output of FeaturePipeline.run()
    domain_weight : weight of subject classification loss (lambda).
    verbose       : print per-subject results

    Returns
    -------
    {'aggregate': {...}, 'subjects': {sid: {...}}}
    """
    from ..models.temporal_dann import TemporalDANNNet

    subject_ids = sorted(features.keys())
    subject_results: dict = {}

    # Infer n_wins from first subject (e.g. 2400 windows / 40 trials = 60)
    d0     = features[subject_ids[0]]
    n_wins = d0['eeg'].shape[0] // d0['labels'].shape[0]

    for sid in subject_ids:
        train_ids  = [s for s in subject_ids if s != sid]
        n_subjects = len(train_ids)

        # ── Pool training trials, assign local subject index 0…n-1 ──────────
        tr_e_list: list = []
        tr_p_list: list = []
        tr_g_list: list = []
        tr_l_list: list = []
        tr_s_list: list = []

        for local_idx, tsid in enumerate(train_ids):
            d        = features[tsid]
            n_trials = d['labels'].shape[0]
            tr_e_list.append(_reshape_trials(d['eeg'], n_wins))
            tr_p_list.append(_reshape_trials(d['ppg'], n_wins))
            tr_g_list.append(_reshape_trials(d['gsr'], n_wins))
            tr_l_list.append(d['labels'])
            tr_s_list.append(np.full(n_trials, local_idx, dtype=np.int64))

        tr_e = np.vstack(tr_e_list)         # (31*40, 60, 192)
        tr_p = np.vstack(tr_p_list)
        tr_g = np.vstack(tr_g_list)
        tr_l = np.vstack(tr_l_list)         # (31*40, 2)
        tr_s = np.concatenate(tr_s_list)    # (31*40,)

        te_e = _reshape_trials(features[sid]['eeg'], n_wins)   # (40, 60, 192)
        te_p = _reshape_trials(features[sid]['ppg'], n_wins)
        te_g = _reshape_trials(features[sid]['gsr'], n_wins)
        te_l = features[sid]['labels']                          # (40, 2)

        # ── Normalise ──────────────────────────────────────────────────────
        tr_e, te_e = _scale_3d(tr_e, te_e)
        tr_p, te_p = _scale_3d(tr_p, te_p)
        tr_g, te_g = _scale_3d(tr_g, te_g)

        # ── Loaders (trial-level batching) ────────────────────────────────
        tr_ld = DataLoader(
            TensorDataset(
                torch.tensor(tr_e, dtype=torch.float32),
                torch.tensor(tr_p, dtype=torch.float32),
                torch.tensor(tr_g, dtype=torch.float32),
                torch.tensor(tr_l, dtype=torch.long),
                torch.tensor(tr_s, dtype=torch.long),
            ),
            batch_size=32, shuffle=True, drop_last=False,
        )
        te_ld = DataLoader(
            TensorDataset(
                torch.tensor(te_e, dtype=torch.float32),
                torch.tensor(te_p, dtype=torch.float32),
                torch.tensor(te_g, dtype=torch.float32),
                torch.tensor(te_l, dtype=torch.long),
            ),
            batch_size=40, shuffle=False, drop_last=False,
        )

        # ── Model ─────────────────────────────────────────────────────────
        model = TemporalDANNNet(
            in_eeg=tr_e.shape[-1],
            in_ppg=tr_p.shape[-1],
            in_gsr=tr_g.shape[-1],
            n_subjects=n_subjects,
        ).to(device)

        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR/100)
        crit_val, crit_ar = _class_weights(tr_l, device)
        crit_subj = nn.CrossEntropyLoss()

        total_steps = EPOCHS * len(tr_ld)
        global_step = 0
        best_acc, best_state, patience_ctr = -1.0, None, 0

        for _ in range(1, EPOCHS + 1):
            model.train()
            for e_b, p_b, g_b, l_b, s_b in tr_ld:
                e_b, p_b, g_b, l_b, s_b = (x.to(device)
                                            for x in (e_b, p_b, g_b, l_b, s_b))
                alpha = dann_alpha(global_step, total_steps)
                global_step += 1

                opt.zero_grad()
                val_out, ar_out, subj_out = model(e_b, p_b, g_b, alpha=alpha)
                loss = (crit_val(val_out, l_b[:, 0])
                        + crit_ar(ar_out, l_b[:, 1])
                        + domain_weight * crit_subj(subj_out, s_b))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            preds, truths = _eval_temporal(model, te_ld, device)
            acc = compute_metrics(truths, preds)['mean_acc']
            if acc > best_acc:
                best_acc, best_state, patience_ctr = acc, copy.deepcopy(model.state_dict()), 0
            else:
                patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        model.load_state_dict(best_state)
        preds, truths = _eval_temporal(model, te_ld, device)
        metrics = compute_metrics(truths, preds)
        subject_results[sid] = metrics

        if verbose:
            print(f'  TempDANN s{sid:02d}: '
                  f'Val {metrics["valence_acc"]:.1f}%  Ar {metrics["arousal_acc"]:.1f}%  '
                  f'(lam={domain_weight})')

        if models_dir:
            from ..utils.io import save_model
            save_model(model, models_dir / f'loso_tdann_s{sid:02d}.pt',
                       meta={'in_eeg': tr_e.shape[-1], 'in_ppg': tr_p.shape[-1],
                             'in_gsr': tr_g.shape[-1], 'n_subjects': n_subjects,
                             'domain_weight': domain_weight, 'n_wins': n_wins})

    agg_keys  = list(next(iter(subject_results.values())).keys())
    aggregate = {k: float(np.mean([v[k] for v in subject_results.values()]))
                 for k in agg_keys}
    return {'aggregate': aggregate, 'subjects': subject_results}
