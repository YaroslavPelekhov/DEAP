# DEAP Emotion Recognition — MMCAT

## What this project does
Reproduces SOTA emotion recognition on the DEAP dataset using EEG, PPG (BVP), and GSR signals.
Model: **MMCAT** — MultiModal Cross-Attention Transformer.

## Dataset
- Location: `../DEAP/dataset/data_preprocessed_python/` (s01.dat … s32.dat)
- 32 subjects × 40 trials × 40 channels × 7680 samples (60 s @ 128 Hz after baseline removal)
- Channels: 0–31 EEG, 36 GSR/EDA, 38 PPG/BVP
- Labels: valence, arousal (continuous 1–9, binarised at 5)

## Project structure
```
deap_emotion/
├── config.py                  ← all hyperparameters and paths
├── main.py                    ← CLI entry point
├── data/loader.py             ← raw .dat loading
├── features/
│   ├── eeg_features.py        ← Differential Entropy (DE) per band
│   └── peripheral_features.py ← PPG HRV + GSR EDA features
├── models/mmcat.py            ← MMCAT architecture
├── training/trainer.py        ← SD and LOSO training loops
└── results/                   ← feature cache + JSON results
```

## Key commands
```bash
# Quick smoke-test on 3 subjects
python main.py --protocol sd --subjects 1 2 3

# Full subject-dependent (SD) run
python main.py --protocol sd --save-results

# Full LOSO run
python main.py --protocol loso --save-results

# Both protocols
python main.py --protocol both --save-results

# Force re-extract features (ignore cache)
python main.py --protocol sd --no-cache
```

## Architecture
```
EEG (160 DE)  → Transformer Encoder → 128-dim ─┐
PPG ( 10 HRV) → MLP Encoder         →  64-dim ─┤ Cross-Attention → Fusion MLP
GSR (  8 EDA) → MLP Encoder         →  64-dim ─┘
                                          ↓
                              [Valence head | Arousal head]
```

## Expected results (32 subjects)
| Protocol | Valence Acc | Arousal Acc |
|----------|-------------|-------------|
| SD       | ~91–94 %    | ~90–93 %    |
| LOSO     | ~74–78 %    | ~73–77 %    |

## Config to tweak
Edit `config.py`:
- `EPOCHS`, `PATIENCE`, `LEARNING_RATE`, `DROPOUT`
- `EEG_EMBED_DIM`, `N_TRANSFORMER_LAYERS`, `N_HEADS`
- `N_SUBJECTS` (default 32; s33 has partial data and is excluded)
