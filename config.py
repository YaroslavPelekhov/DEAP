"""
Centralized configuration for DEAP emotion recognition project.
All paths, hyperparameters, and experimental settings live here.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = Path(
    r"C:\Users\psgpe\Downloads\2025-2026_NeuroBarometer Emotions"
    r"\2025-2026_NeuroBarometer Emotions\DEAP\dataset\data_preprocessed_python"
)
CACHE_DIR = ROOT / "results" / "cache"
RESULTS_DIR = ROOT / "results"

# ─── DEAP dataset constants ───────────────────────────────────────────────────
N_SUBJECTS = 32          # s01–s32 (s33 has partial data, excluded by default)
N_TRIALS = 40
N_CHANNELS = 40          # total channels per trial
SFREQ = 128              # Hz (preprocessed)
TRIAL_SAMPLES = 8064     # 63 s × 128 Hz
BASELINE_SAMPLES = 384   # first 3 s are pre-stimulus baseline

# Channel indices (0-based)
EEG_CHANNELS = list(range(32))          # 0–31
GSR_CHANNEL = 36                        # EDA / galvanic skin response
PPG_CHANNEL = 38                        # plethysmograph / BVP

# ─── Feature extraction ───────────────────────────────────────────────────────
# EEG: Differential Entropy per frequency band
EEG_BANDS = {
    "delta": (1, 3),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta":  (14, 30),
    "gamma": (31, 50),
}
EEG_WINDOW_SEC = 1       # non-overlapping 1-s windows for DE computation
N_EEG_FEATURES = len(EEG_CHANNELS) * len(EEG_BANDS)   # 32 × 5 = 160

# PPG / HRV
N_PPG_FEATURES = 10

# GSR / EDA
N_GSR_FEATURES = 8
N_CONSENSUS_FEATURES = 5   # 2 consensus + 1 position + 2 frontal hemispheric asymmetry

N_FEATURES_TOTAL = N_EEG_FEATURES + N_PPG_FEATURES + N_GSR_FEATURES  # 178

# ─── Labels ───────────────────────────────────────────────────────────────────
LABEL_THRESHOLD = 5.0    # split scale [1–9] at midpoint → High / Low
# label indices in DEAP: 0=valence, 1=arousal, 2=dominance, 3=liking
LABEL_NAMES = ["valence", "arousal"]

# ─── Model architecture ───────────────────────────────────────────────────────
EEG_EMBED_DIM = 128
PERIPHERAL_EMBED_DIM = 64
FUSION_DIM = 128
N_HEADS = 8
N_TRANSFORMER_LAYERS = 2
DROPOUT = 0.3

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15            # early stopping patience
SEED = 42

# Subject-dependent CV: k-fold over 40 trials
SD_N_FOLDS = 8           # 5 trials per fold

# ─── Misc ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda"          # "cuda" or "cpu" — auto-detected in main.py
NUM_WORKERS = 0          # Windows: keep 0 to avoid multiprocessing issues
