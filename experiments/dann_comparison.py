"""
DANN vs baseline LOSO comparison — all 32 subjects.

Saves results to results/dann_comparison.json.
Run: python experiments/dann_comparison.py
"""
import json, time, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR, CACHE_DIR
from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_loso, train_loso_dann

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUT = RESULTS_DIR / 'dann_comparison.json'

ALL_SIDS = list(range(1, 33))

# ── Feature extraction (cached after first run) ────────────────────────────
print('Loading features...')
pipe  = FeaturePipeline(data_dir=DATA_DIR, cache_dir=CACHE_DIR, cache_version='v13')
feats = pipe.run(ALL_SIDS)

results = {}

# ── Baseline LOSO ──────────────────────────────────────────────────────────
print('\n=== Baseline LOSO (32 subj) ===')
t0 = time.time()
base = train_loso(feats, verbose=True)
elapsed = time.time() - t0
print(f'>> Val {base["aggregate"]["valence_acc"]:.2f}%  '
      f'Ar  {base["aggregate"]["arousal_acc"]:.2f}%  [{elapsed:.0f}s]')
results['baseline'] = {**base['aggregate'],
                       'subjects': base['subjects'],
                       'elapsed_s': elapsed}

# ── DANN with lambda sweep ─────────────────────────────────────────────────
for lam in [0.1, 0.2, 0.3, 0.5]:
    key = f'dann_lam{lam}'
    print(f'\n=== DANN LOSO lam={lam} (32 subj) ===')
    t0 = time.time()
    res = train_loso_dann(feats, domain_weight=lam, verbose=True)
    elapsed = time.time() - t0
    print(f'>> Val {res["aggregate"]["valence_acc"]:.2f}%  '
          f'Ar  {res["aggregate"]["arousal_acc"]:.2f}%  [{elapsed:.0f}s]')
    results[key] = {**res['aggregate'],
                    'subjects': res['subjects'],
                    'elapsed_s': elapsed}

    # Save incrementally so partial results survive early termination
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved -> {OUT}')

# ── Summary table ──────────────────────────────────────────────────────────
print('\n' + '='*55)
print(f'{"Method":<20} {"Val Acc":>10} {"Ar Acc":>10}')
print('-'*55)
for k, v in results.items():
    print(f'{k:<20} {v["valence_acc"]:>9.2f}% {v["arousal_acc"]:>9.2f}%')
print('='*55)

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nFull results saved -> {OUT}')
