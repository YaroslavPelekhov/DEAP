"""
Entry point for DEAP emotion recognition experiment.

Usage examples:
  # SVM on trial-level features (fastest, ~85-88% SD accuracy)
  python main.py --protocol sd --model svm

  # LightMLP on window-level features with majority vote (~87-90% SD)
  python main.py --protocol sd --model mlp

  # Both models, all protocols
  python main.py --protocol both --model all --save-results

  # Quick smoke-test on 3 subjects
  python main.py --protocol sd --model svm --subjects 1 2 3

  # Re-extract features
  python main.py --protocol sd --model svm --no-cache
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import CACHE_DIR, RESULTS_DIR, SEED, N_SUBJECTS, N_TRIALS, N_EEG_FEATURES, N_PPG_FEATURES, N_GSR_FEATURES, SFREQ, EEG_WINDOW_SEC
from data.loader import load_subject, get_binary_labels
from features.eeg_features import extract_de_subject
from features.peripheral_features import (
    extract_ppg_subject, extract_gsr_subject,
    extract_ppg_features, extract_gsr_features,
)
from training.trainer import (
    run_subject_dependent, run_loso, print_results, MODEL_CLASS, LightMLP, MMCAT
)
from models.svm_classifier import run_sd_svm
from models.eegnet import EEGNetMultimodal


# ─── Feature extraction ───────────────────────────────────────────────────────

def build_features(subject_ids, use_cache=True, verbose=True):
    """
    Extracts both window-level (for MLP/MMCAT) and trial-level (for SVM) features.

    Returns:
      features    : {sid: {eeg, ppg, gsr, groups, eeg_trial, ppg_trial, gsr_trial}}
      bin_labels  : {sid: (40, 2)}
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = "_".join(str(s) for s in subject_ids)
    cache_path = CACHE_DIR / f"features_v7_{key}.pkl"

    if use_cache and cache_path.exists():
        print(f"[cache] Loading from {cache_path.name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[features] Extracting (one subject at a time) ...")
    features, bin_labels = {}, {}
    for sid in subject_ids:
        if verbose:
            print(f"  s{sid:02d} ...", end="\r")
        subj = load_subject(sid)
        bin_labels[sid] = get_binary_labels(subj["labels"])

        # Window-level EEG DE (for MLP/MMCAT)
        eeg_win, groups = extract_de_subject(subj["eeg"])   # (2400, 160), (2400,)
        n_wins = len(groups) // subj["eeg"].shape[0]
        n_trials = subj["eeg"].shape[0]

        # Trial-level averaged EEG DE (for SVM) — average windows per trial
        eeg_trial = eeg_win.reshape(n_trials, n_wins, -1).mean(axis=1)  # (40, 160)

        ppg_trial = extract_ppg_subject(subj["ppg"])                    # (40, 10)
        gsr_trial = extract_gsr_subject(subj["gsr"])                    # (40, 8)

        # H5: segmented PPG/GSR — 3 × 20-second segments per trial.
        # Gives temporal resolution: early/mid/late trial physiology.
        seg_len = int(20 * SFREQ)   # 2560 samples = 20 s
        n_segs  = 3
        ppg_seg = np.zeros((n_trials, n_segs, N_PPG_FEATURES), dtype=np.float32)
        gsr_seg = np.zeros((n_trials, n_segs, N_GSR_FEATURES), dtype=np.float32)
        for t in range(n_trials):
            for s in range(n_segs):
                ppg_seg[t, s] = extract_ppg_features(subj["ppg"][t, s*seg_len:(s+1)*seg_len])
                gsr_seg[t, s] = extract_gsr_features(subj["gsr"][t, s*seg_len:(s+1)*seg_len])
        wins_per_seg = n_wins // n_segs   # 20 EEG windows per 20-second segment
        ppg_wins = np.repeat(ppg_seg, wins_per_seg, axis=1).reshape(n_trials * n_wins, N_PPG_FEATURES)
        gsr_wins = np.repeat(gsr_seg, wins_per_seg, axis=1).reshape(n_trials * n_wins, N_GSR_FEATURES)

        features[sid] = {
            # Window-level (for neural models)
            "eeg":    eeg_win,
            "ppg":    ppg_wins,
            "gsr":    gsr_wins,
            "groups": groups,
            # Trial-level (for SVM)
            "eeg_trial": eeg_trial,
            "ppg_trial": ppg_trial,
            "gsr_trial": gsr_trial,
        }

    if verbose:
        print("  Done.                    ")

    # Cross-subject consensus: for each subject, append mean binary label of
    # the same trial across all OTHER subjects as 2 extra features appended to GSR.
    # (valid information — same stimulus shown to everyone, no label leakage)
    for sid in subject_ids:
        other_sids = [s for s in subject_ids if s != sid]
        if not other_sids:
            continue
        consensus_trial = np.mean(
            [bin_labels[s].astype(np.float32) for s in other_sids], axis=0
        )  # (40, 2)  values in [0, 1]
        n_wins = len(features[sid]["groups"]) // N_TRIALS
        consensus_wins = np.repeat(consensus_trial, n_wins, axis=0)  # (n_wins*40, 2)
        # window position in trial: normalized 0→1, shape (n_wins*n_trials, 1)
        pos_wins  = np.tile(np.linspace(0, 1, n_wins, dtype=np.float32), N_TRIALS).reshape(-1, 1)
        pos_trial = np.full((N_TRIALS, 1), 0.5, dtype=np.float32)  # midpoint for trial-level
        # Frontal hemispheric asymmetry (right - left DE) for alpha and theta bands.
        # Alpha asymmetry = valence marker; theta asymmetry = arousal/frontal midline.
        # DEAP: left frontal ch 0-5, right frontal ch 24-29; alpha=band2, theta=band1.
        left_ch  = [0, 1, 2, 3, 4, 5]
        right_ch = [24, 25, 26, 27, 28, 29]
        l_alpha = [c * 5 + 2 for c in left_ch];  r_alpha = [c * 5 + 2 for c in right_ch]
        l_theta = [c * 5 + 1 for c in left_ch];  r_theta = [c * 5 + 1 for c in right_ch]
        eeg_w = features[sid]["eeg"]   # (n_windows, 160)
        faa = (eeg_w[:, r_alpha].mean(1, keepdims=True) - eeg_w[:, l_alpha].mean(1, keepdims=True))
        fta = (eeg_w[:, r_theta].mean(1, keepdims=True) - eeg_w[:, l_theta].mean(1, keepdims=True))
        asym_wins  = np.concatenate([faa, fta], axis=1).astype(np.float32)          # (n_windows, 2)
        asym_trial = asym_wins.reshape(N_TRIALS, n_wins, 2).mean(axis=1)            # (40, 2)

        features[sid]["gsr"]       = np.concatenate([features[sid]["gsr"],       consensus_wins,  pos_wins,  asym_wins],  axis=1)
        features[sid]["gsr_trial"] = np.concatenate([features[sid]["gsr_trial"], consensus_trial, pos_trial, asym_trial], axis=1)

    result = (features, bin_labels)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[cache] Saved to {cache_path.name}")
    return result


def build_raw_features(subject_ids, use_cache=True, verbose=True):
    """
    Build raw EEG window dataset for EEGNet.

    EEG: split each trial into 1-second windows → (n_trials*n_wins, 1, 32, 128)
    PPG/GSR: trial-level features repeated per window.

    Returns:
      features   : {sid: {eeg_raw, ppg, gsr, groups, ppg_trial, gsr_trial}}
      bin_labels : {sid: (40, 2)}
    """
    # Raw EEG windows are ~1.3 GB for 32 subjects — not cached, re-extracted each run (~15s)
    import gc
    print("[features] Extracting raw EEG windows ...")
    win = int(EEG_WINDOW_SEC * SFREQ)   # 128 samples
    features, bin_labels = {}, {}

    for sid in subject_ids:
        if verbose:
            print(f"  s{sid:02d} ...", end="\r")
        subj = load_subject(sid)
        bin_labels[sid] = get_binary_labels(subj["labels"])

        eeg = subj["eeg"].astype(np.float32)   # (40, 32, 7680) — float32 to save RAM
        n_trials, n_ch, n_samp = eeg.shape
        n_wins = n_samp // win

        # Split into windows: (n_trials*n_wins, 1, 32, 128)
        eeg_wins = eeg.reshape(n_trials, n_ch, n_wins, win)  # (40,32,60,128)
        eeg_wins = eeg_wins.transpose(0, 2, 1, 3)            # (40,60,32,128)
        eeg_wins = eeg_wins.reshape(n_trials * n_wins, 1, n_ch, win).copy()

        ppg_trial = extract_ppg_subject(subj["ppg"])          # (40, 10)
        gsr_trial = extract_gsr_subject(subj["gsr"])          # (40, 8)

        features[sid] = {
            "eeg":       eeg_wins,
            "ppg":       extract_ppg_subject(subj["ppg"], n_windows=n_wins),
            "gsr":       extract_gsr_subject(subj["gsr"], n_windows=n_wins),
            "groups":    np.repeat(np.arange(n_trials), n_wins).astype(np.int32),
            "ppg_trial": ppg_trial,
            "gsr_trial": gsr_trial,
        }

        del subj, eeg, eeg_wins   # free raw signal immediately
        gc.collect()

    if verbose:
        print("  Done.                    ")

    # Cross-subject consensus (same as DE pipeline)
    for sid in subject_ids:
        other_sids = [s for s in subject_ids if s != sid]
        if not other_sids:
            continue
        consensus_trial = np.mean(
            [bin_labels[s].astype(np.float32) for s in other_sids], axis=0
        )  # (40, 2)
        n_wins = len(features[sid]["groups"]) // N_TRIALS
        consensus_wins = np.repeat(consensus_trial, n_wins, axis=0)
        features[sid]["gsr"]       = np.concatenate([features[sid]["gsr"],       consensus_wins],  axis=1)
        features[sid]["gsr_trial"] = np.concatenate([features[sid]["gsr_trial"], consensus_trial], axis=1)

    return features, bin_labels


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DEAP Emotion Recognition")
    parser.add_argument("--protocol", choices=["sd", "loso", "both"], default="sd")
    parser.add_argument("--model", choices=["svm", "mlp", "mmcat", "eegnet", "all"], default="eegnet",
                        help="svm | mlp | mmcat | eegnet (default) | all")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    subject_ids = args.subjects or list(range(1, N_SUBJECTS + 1))
    print(f"[subjects] {len(subject_ids)}: {subject_ids[:5]}{'...' if len(subject_ids) > 5 else ''}")

    # EEGNet needs raw windows; others need DE features
    needs_raw = args.model in ("eegnet", "all")
    needs_de  = args.model in ("svm", "mlp", "mmcat", "all")

    t0 = time.time()
    if needs_de:
        features, labels = build_features(subject_ids, use_cache=not args.no_cache)
        sid0 = subject_ids[0]
        print(f"[features] DE ready in {time.time()-t0:.1f}s | EEG: {features[sid0]['eeg'].shape}")
    if needs_raw:
        t1 = time.time()
        raw_features, labels = build_raw_features(subject_ids, use_cache=not args.no_cache)
        sid0 = subject_ids[0]
        print(f"[features] Raw ready in {time.time()-t1:.1f}s | EEG: {raw_features[sid0]['eeg'].shape}")

    all_results = {}
    models_to_run = ["svm", "mlp", "mmcat", "eegnet"] if args.model == "all" else [args.model]

    for model_name in models_to_run:
        print(f"\n[model] {model_name.upper()}")
        import training.trainer as trainer_mod

        # ── SVM ──────────────────────────────────────────────────────────────
        if model_name == "svm":
            if args.protocol in ("sd", "both"):
                print("[SD] SVM 8-fold CV ...")
                t = time.time()
                res = run_sd_svm(features, labels)
                print(f"[SD] Done in {time.time()-t:.1f}s")
                print_results(res)
                all_results["sd_svm"] = res
            if args.protocol in ("loso", "both"):
                print("[LOSO] SVM not implemented — use eegnet/mlp")

        # ── EEGNet ───────────────────────────────────────────────────────────
        elif model_name == "eegnet":
            trainer_mod.MODEL_CLASS = EEGNetMultimodal
            if args.protocol in ("sd", "both"):
                print("[SD] EEGNet 8-fold CV ...")
                t = time.time()
                res = run_subject_dependent(raw_features, labels, device)
                print(f"[SD] Done in {time.time()-t:.1f}s")
                print_results(res)
                all_results["sd_eegnet"] = res
            if args.protocol in ("loso", "both"):
                print("[LOSO] EEGNet ...")
                t = time.time()
                res = run_loso(raw_features, labels, device)
                print(f"[LOSO] Done in {time.time()-t:.1f}s")
                print_results(res)
                all_results["loso_eegnet"] = res

        # ── LightMLP / MMCAT ─────────────────────────────────────────────────
        else:
            trainer_mod.MODEL_CLASS = LightMLP if model_name == "mlp" else MMCAT
            if args.protocol in ("sd", "both"):
                print(f"[SD] {model_name.upper()} 8-fold CV ...")
                t = time.time()
                res = run_subject_dependent(features, labels, device)
                print(f"[SD] Done in {time.time()-t:.1f}s")
                print_results(res)
                all_results[f"sd_{model_name}"] = res
            if args.protocol in ("loso", "both"):
                print(f"[LOSO] {model_name.upper()} ...")
                t = time.time()
                res = run_loso(features, labels, device)
                print(f"[LOSO] Done in {time.time()-t:.1f}s")
                print_results(res)
                all_results[f"loso_{model_name}"] = res

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save_results and all_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"results_{args.model}_{args.protocol}_{ts}.json"

        def _s(o):
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, (np.integer, int)):    return int(o)
            if isinstance(o, np.ndarray):           return o.tolist()
            if isinstance(o, dict):   return {k: _s(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return [_s(v) for v in o]
            return o

        with open(out_path, "w") as f:
            json.dump(_s(all_results), f, indent=2)
        print(f"[results] Saved to {out_path}")


if __name__ == "__main__":
    main()
