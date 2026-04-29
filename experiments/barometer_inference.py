"""
Barometer inference: run a trained model on new EEG+GSR data from the NeuroBarometer.

The NeuroBarometer has 20 EEG channels (a subset of DEAP's 32).
This script:
  1. Loads a trained model from data/models/
  2. Reads raw Barometer data (numpy arrays)
  3. Extracts features using only the 19 shared channels
  4. Returns valence / arousal predictions

Usage:
    python experiments/barometer_inference.py --model data/models/loso_s01_multimodal.pt
                                               --eeg  barometer_eeg.npy
                                               --gsr  barometer_gsr.npy
                                               --ppg  barometer_ppg.npy

    Or import and call predict() directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.channels import BAROMETER_IN_DEAP, DEAP_BAROMETER_INDICES
from src.features.eeg import EEGExtractor, DEFAULT_BANDS
from src.features.ppg import extract_ppg_features
from src.features.gsr import extract_gsr_features
from src.utils.io import load_model
import config as cfg


N_BAROMETER_CH = len(BAROMETER_IN_DEAP)   # 19 channels

print(f'[barometer] Using {N_BAROMETER_CH} EEG channels: {BAROMETER_IN_DEAP}')


def build_extractor() -> EEGExtractor:
    """Feature extractor configured for Barometer channel subset."""
    return EEGExtractor(
        bands=DEFAULT_BANDS,
        fs=cfg.SFREQ,
        win_sec=1.0,
        channel_names=BAROMETER_IN_DEAP,
    )


def extract_features(
    eeg: np.ndarray,      # (n_ch, n_samples) or (n_trials, n_ch, n_samples)
    ppg: np.ndarray,      # (n_samples,) or (n_trials, n_samples)
    gsr: np.ndarray,      # (n_samples,) or (n_trials, n_samples)
    fs: int = cfg.SFREQ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from Barometer data.

    Returns (eeg_feats, ppg_feats, gsr_feats).
    """
    extractor = build_extractor()

    # Support both single-trial and multi-trial input
    if eeg.ndim == 2:
        eeg = eeg[np.newaxis]   # (1, n_ch, n_samples)
        ppg = ppg[np.newaxis]
        gsr = gsr[np.newaxis]

    assert eeg.shape[1] == N_BAROMETER_CH, (
        f'Expected {N_BAROMETER_CH} EEG channels, got {eeg.shape[1]}. '
        f'Channels should be (in order): {BAROMETER_IN_DEAP}'
    )

    eeg_feats, _ = extractor.extract_subject(eeg)
    n_wins = eeg_feats.shape[0] // len(eeg)

    ppg_feats = np.vstack([
        np.tile(extract_ppg_features(ppg[i], fs), (n_wins, 1))
        for i in range(len(ppg))
    ])
    gsr_feats = np.vstack([
        np.tile(extract_gsr_features(gsr[i], fs), (n_wins, 1))
        for i in range(len(gsr))
    ])

    # Pad GSR to include FAA/FTA slots (zeros — no frontal channels guarantee on Barometer)
    # Compute FAA/FTA if F3/F4 are available
    faa_fta = np.zeros((eeg_feats.shape[0], 2), dtype=np.float32)
    try:
        f3_idx = BAROMETER_IN_DEAP.index('F3')
        f4_idx = BAROMETER_IN_DEAP.index('F4')
        faa_fta_per_trial = extractor.compute_faa_fta(eeg, f3_idx, f4_idx)
        faa_fta = np.repeat(faa_fta_per_trial, n_wins, axis=0)
    except ValueError:
        pass
    gsr_feats = np.concatenate([gsr_feats, faa_fta], axis=1)

    return eeg_feats, ppg_feats, gsr_feats


def predict(
    eeg: np.ndarray,
    ppg: np.ndarray,
    gsr: np.ndarray,
    model_path: Path | str,
    model_name: str = 'multimodal',
    device: torch.device | None = None,
) -> dict:
    """
    Run inference on Barometer data.

    Returns:
        {
            'valence': np.ndarray (n_windows,) predicted class [0=low, 1=high],
            'arousal': np.ndarray (n_windows,),
            'valence_prob': np.ndarray (n_windows, 2),
            'arousal_prob': np.ndarray (n_windows, 2),
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eeg_f, ppg_f, gsr_f = extract_features(eeg, ppg, gsr)

    model = load_model(
        model_path, model_name,
        in_eeg=eeg_f.shape[1], in_ppg=ppg_f.shape[1], in_gsr=gsr_f.shape[1],
    ).to(device)
    model.eval()

    with torch.no_grad():
        te = torch.tensor(eeg_f, dtype=torch.float32).to(device)
        tp = torch.tensor(ppg_f, dtype=torch.float32).to(device)
        tg = torch.tensor(gsr_f, dtype=torch.float32).to(device)
        val_logits, ar_logits = model(te, tp, tg)

    import torch.nn.functional as F
    val_prob = F.softmax(val_logits, dim=-1).cpu().numpy()
    ar_prob  = F.softmax(ar_logits,  dim=-1).cpu().numpy()

    return {
        'valence':      val_prob.argmax(1),
        'arousal':      ar_prob.argmax(1),
        'valence_prob': val_prob,
        'arousal_prob': ar_prob,
    }


def main():
    parser = argparse.ArgumentParser(description='NeuroBarometer inference')
    parser.add_argument('--model', required=True, help='Path to .pt model file')
    parser.add_argument('--model-name', default='multimodal')
    parser.add_argument('--eeg',  required=True, help='Path to EEG .npy (n_ch, n_samples)')
    parser.add_argument('--ppg',  required=True, help='Path to PPG .npy (n_samples,)')
    parser.add_argument('--gsr',  required=True, help='Path to GSR .npy (n_samples,)')
    args = parser.parse_args()

    eeg = np.load(args.eeg)
    ppg = np.load(args.ppg)
    gsr = np.load(args.gsr)

    results = predict(eeg, ppg, gsr, args.model, args.model_name)
    print(f"Predicted valence: {results['valence']}  (0=Low, 1=High)")
    print(f"Predicted arousal: {results['arousal']}  (0=Low, 1=High)")


if __name__ == '__main__':
    main()
