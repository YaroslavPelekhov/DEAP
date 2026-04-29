"""
SVM classifier for DEAP — trial-level features.

Uses trial-averaged DE + PPG HRV + GSR EDA features with RBF SVM.
This is the most common and reproducible baseline in the literature,
typically achieving 83-89% accuracy on DEAP SD protocol.

Reference: Zheng & Lu (2015), Li et al. (2018).
"""
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEED, SD_N_FOLDS, N_TRIALS


def run_sd_svm(
    features: dict,
    labels: dict,
) -> dict:
    """
    Subject-dependent SVM with 8-fold GroupKFold.

    features[sid]:
      'eeg_trial': (40, 160)  — trial-averaged DE (one vector per trial)
      'ppg_trial': (40,  10)
      'gsr_trial': (40,   8)
    labels[sid]: (40, 2)
    """
    subject_results = {}
    gkf = GroupKFold(n_splits=SD_N_FOLDS)

    for sid in sorted(features.keys()):
        eeg = features[sid]["eeg_trial"]   # (40, 160)
        ppg = features[sid]["ppg_trial"]   # (40,  10)
        gsr = features[sid]["gsr_trial"]   # (40,   8)
        X   = np.concatenate([eeg, ppg, gsr], axis=1)  # (40, 178)
        groups = np.arange(N_TRIALS)      # each trial is its own group

        val_preds = np.zeros(N_TRIALS, dtype=int)
        ar_preds  = np.zeros(N_TRIALS, dtype=int)
        val_true  = labels[sid][:, 0]
        ar_true   = labels[sid][:, 1]

        for train_idx, test_idx in gkf.split(X, groups=groups):
            sc = StandardScaler().fit(X[train_idx])
            X_tr = sc.transform(X[train_idx])
            X_te = sc.transform(X[test_idx])

            # Linear SVM — less prone to overfitting with 35 samples / 178 features
            clf_v = SVC(kernel="linear", C=0.05, random_state=SEED)
            clf_a = SVC(kernel="linear", C=0.05, random_state=SEED)
            clf_v.fit(X_tr, val_true[train_idx])
            clf_a.fit(X_tr, ar_true[train_idx])

            val_preds[test_idx] = clf_v.predict(X_te)
            ar_preds[test_idx]  = clf_a.predict(X_te)

        subject_results[sid] = {
            "valence_acc": accuracy_score(val_true, val_preds) * 100,
            "arousal_acc": accuracy_score(ar_true,  ar_preds)  * 100,
            "valence_f1":  f1_score(val_true, val_preds, average="weighted") * 100,
            "arousal_f1":  f1_score(ar_true,  ar_preds,  average="weighted") * 100,
        }
        m = subject_results[sid]
        print(
            f"  s{sid:02d} | "
            f"Val {m['valence_acc']:.1f}% | "
            f"Ar {m['arousal_acc']:.1f}%"
        )

    all_m = list(subject_results.values())
    aggregate = {k: (np.mean([m[k] for m in all_m]),
                     np.std([m[k] for m in all_m]))
                 for k in all_m[0]}
    return {"protocol": "SD-SVM", "subjects": subject_results, "aggregate": aggregate}
