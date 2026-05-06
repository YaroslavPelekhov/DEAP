"""
TemporalNet vs TemporalNet+DANN — LOSO on all 32 subjects.

Hypothesis: applying GRL at the trial-embedding level (after the Bi-GRU)
produces subject-invariant features → higher arousal LOSO.

Run:
    python -u experiments/temporal_dann_comparison.py

Results saved incrementally to results/temporal_dann_comparison.json
"""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, CACHE_DIR
from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_loso_temporal, train_loso_dann_temporal

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUT = RESULTS_DIR / 'temporal_dann_comparison.json'

ALL_SIDS = list(range(1, 33))

# ── Features (cached) ──────────────────────────────────────────────────────
print('Loading features...')
pipe  = FeaturePipeline(data_dir=DATA_DIR, cache_dir=CACHE_DIR, cache_version='v13')
feats = pipe.run(ALL_SIDS)
print(f'  EEG: {feats[1]["eeg"].shape}  PPG: {feats[1]["ppg"].shape}  '
      f'GSR: {feats[1]["gsr"].shape}  Labels: {feats[1]["labels"].shape}')

results = {}


def save():
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)


def run(name, fn, **kw):
    print(f'\n=== {name} (32 subj LOSO) ===')
    t0  = time.time()
    res = fn(feats, verbose=True, **kw)
    elapsed = time.time() - t0
    v = res['aggregate']['valence_acc']
    a = res['aggregate']['arousal_acc']
    print(f'>> Val {v:.2f}%  Ar {a:.2f}%  [{elapsed:.0f}s]')
    results[name] = {**res['aggregate'], 'subjects': res['subjects'],
                     'elapsed_s': elapsed}
    save()
    return res


# ── 1. TemporalNet baseline ────────────────────────────────────────────────
run('temporal', train_loso_temporal)

# ── 2. TemporalDANN  lam=0.1 ──────────────────────────────────────────────
run('temporal_dann_lam0.1', train_loso_dann_temporal, domain_weight=0.1)

# ── 3. TemporalDANN  lam=0.3 ──────────────────────────────────────────────
run('temporal_dann_lam0.3', train_loso_dann_temporal, domain_weight=0.3)

# ── 4. TemporalDANN  lam=0.5 ──────────────────────────────────────────────
run('temporal_dann_lam0.5', train_loso_dann_temporal, domain_weight=0.5)

# ── Summary ────────────────────────────────────────────────────────────────
print('\n' + '=' * 58)
print(f'{"Model":<28} {"Val LOSO":>10} {"Ar LOSO":>10}')
print('-' * 58)
for k, v in results.items():
    print(f'{k:<28} {v["valence_acc"]:>9.2f}% {v["arousal_acc"]:>9.2f}%')
print('=' * 58)
print(f'\nFull results -> {OUT}')
