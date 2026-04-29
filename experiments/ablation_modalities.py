"""
Ablation study: compare accuracy across modality subsets.

Loads saved model weights (or trains fresh) and evaluates all 7 non-empty
combinations of {EEG, PPG, GSR}.

Results are printed as a table and saved to results/ablation_modalities.json
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_sd, DEVICE
from src.utils.io import save_results, timestamped_path
import config as cfg

SUBJECT_IDS = list(range(1, cfg.N_SUBJECTS + 1))
MODELS_DIR  = ROOT / 'data' / 'models' / 'ablation'

COMBINATIONS = [
    ('EEG+PPG+GSR', True,  True,  True),
    ('EEG+PPG',     True,  True,  False),
    ('EEG+GSR',     True,  False, True),
    ('PPG+GSR',     False, True,  True),
    ('EEG only',    True,  False, False),
    ('PPG only',    False, True,  False),
    ('GSR only',    False, False, True),
]


def zero_modality(e, p, g, use_eeg, use_ppg, use_gsr):
    if not use_eeg: e = np.zeros_like(e)
    if not use_ppg: p = np.zeros_like(p)
    if not use_gsr: g = np.zeros_like(g)
    return e, p, g


def main():
    pipeline = FeaturePipeline(
        data_dir=cfg.DATA_DIR,
        cache_dir=cfg.CACHE_DIR,
        cache_version='v13',
    )
    features = pipeline.run(SUBJECT_IDS)

    print('\n=== ABLATION: MODALITY COMBINATIONS ===\n')
    print(f'{"Combination":<16}  {"Val Acc":>8}  {"Ar Acc":>8}  {"Mean":>8}')
    print('-' * 50)

    all_results = {}
    for name, use_e, use_p, use_g in COMBINATIONS:
        # Mask features
        masked_features = {}
        for sid, data in features.items():
            e, p, g = data['eeg'].copy(), data['ppg'].copy(), data['gsr'].copy()
            e, p, g = zero_modality(e, p, g, use_e, use_p, use_g)
            masked_features[sid] = {**data, 'eeg': e, 'ppg': p, 'gsr': g}

        results = train_sd(masked_features, model_name='multimodal',
                           device=DEVICE, verbose=False)
        agg = results['aggregate']
        print(f'{name:<16}  {agg["valence_acc"]:>7.1f}%  {agg["arousal_acc"]:>7.1f}%  {agg["mean_acc"]:>7.1f}%')
        all_results[name] = agg

    out = timestamped_path(ROOT / 'results', 'ablation_modalities')
    save_results(all_results, out)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
