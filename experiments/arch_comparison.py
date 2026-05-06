"""
Architecture comparison — LOSO on all 32 subjects.

Models tested:
  multimodal  — flat MLP per window + majority vote  (baseline)
  temporal    — Bi-GRU over 60-window trial sequence  (expected best)
  mmcat       — cross-attention EEG <-> PPG+GSR       (unknown)

Results saved incrementally to results/arch_comparison.json.
Run: python -u experiments/arch_comparison.py
"""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, CACHE_DIR
from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_loso, train_loso_temporal

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUT = RESULTS_DIR / 'arch_comparison.json'

ALL_SIDS = list(range(1, 33))

# ── Features (cached) ──────────────────────────────────────────────────────
print('Loading features...')
pipe  = FeaturePipeline(data_dir=DATA_DIR, cache_dir=CACHE_DIR, cache_version='v13')
feats = pipe.run(ALL_SIDS)
print(f'  EEG dim: {feats[1]["eeg"].shape[1]}   '
      f'PPG: {feats[1]["ppg"].shape[1]}   GSR: {feats[1]["gsr"].shape[1]}')

results = {}


def save():
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)


def run(name, fn, **kw):
    print(f'\n=== {name} LOSO (32 subj) ===')
    t0 = time.time()
    res = fn(feats, verbose=True, **kw)
    elapsed = time.time() - t0
    v = res['aggregate']['valence_acc']
    a = res['aggregate']['arousal_acc']
    print(f'>> Val {v:.2f}%  Ar {a:.2f}%  [{elapsed:.0f}s]')
    results[name] = {**res['aggregate'], 'subjects': res['subjects'],
                     'elapsed_s': elapsed}
    save()
    return res


# ── 1. MultiModalNet (flat windows) ───────────────────────────────────────
run('multimodal', train_loso, model_name='multimodal')

# ── 2. TemporalNet (full trial sequences) ─────────────────────────────────
run('temporal', train_loso_temporal)

# ── 3. MMCAT (cross-attention, flat windows) ──────────────────────────────
run('mmcat', train_loso, model_name='mmcat')

# ── Summary ────────────────────────────────────────────────────────────────
print('\n' + '=' * 52)
print(f'{"Model":<15} {"Val LOSO":>10} {"Ar LOSO":>10}')
print('-' * 52)
for k, v in results.items():
    print(f'{k:<15} {v["valence_acc"]:>9.2f}% {v["arousal_acc"]:>9.2f}%')
print('=' * 52)

print(f'\nFull results -> {OUT}')
