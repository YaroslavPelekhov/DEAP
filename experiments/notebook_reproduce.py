"""
Reproduce notebook LOSO results with v14 features.

Key changes vs v13:
  • win_sec=2.0, stride_sec=1.0  → 59 windows/trial
  • PPG: 12 features (+resp_rate_bpm, +resp_power)
  • GSR: 13 features (base 8 + FAA + FTA + cons_val + cons_ar + position)
  • Per-subject z-score applied before caching
  • LR=1e-3, batch_size=16, EPOCHS=100  (match notebook train_fold_temporal)

Expected: ~78% Val / ~65% Ar (LOSO, majority vote)

Run:
    python -u experiments/notebook_reproduce.py
"""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, CACHE_DIR
from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_loso_temporal

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUT = RESULTS_DIR / 'notebook_reproduce.json'

ALL_SIDS = list(range(1, 33))

# ── Features (v14, new defaults) ──────────────────────────────────────────────
print('Loading / extracting v14 features...')
pipe  = FeaturePipeline(data_dir=DATA_DIR, cache_dir=CACHE_DIR)  # v14 by default
feats = pipe.run(ALL_SIDS)

d0 = feats[1]
print(f'\nFeature shapes per subject:')
print(f'  EEG: {d0["eeg"].shape}  PPG: {d0["ppg"].shape}  GSR: {d0["gsr"].shape}')
print(f'  Labels: {d0["labels"].shape}  n_wins: '
      f'{d0["eeg"].shape[0] // d0["labels"].shape[0]}')

# ── LOSO training (notebook-matched hyperparameters) ──────────────────────────
print('\nRunning LOSO with TemporalNet (LR=1e-3, batch=16, EPOCHS=100)...\n')
t0  = time.time()
res = train_loso_temporal(feats, verbose=True)
elapsed = time.time() - t0

v = res['aggregate']['valence_acc']
a = res['aggregate']['arousal_acc']
print(f'\n>> Valence {v:.2f}%  Arousal {a:.2f}%  [{elapsed:.0f}s]')

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUT, 'w') as f:
    json.dump({'aggregate': res['aggregate'], 'subjects': res['subjects'],
               'elapsed_s': elapsed}, f, indent=2)
print(f'Full results -> {OUT}')
