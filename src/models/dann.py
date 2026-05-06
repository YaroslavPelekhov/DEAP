"""
DANNNet — Domain-Adversarial Neural Network for cross-subject emotion recognition.

Problem
───────
LOSO arousal accuracy (~65%) lags behind SD (~73%) because subjects differ
in their physiological baseline. The model learns subject-specific patterns
that don't transfer.

Solution (Ganin et al., 2016)
──────────────────────────────
Add a subject classifier attached through a Gradient Reversal Layer (GRL).

  During forward pass : GRL is identity
  During backward pass: GRL negates gradients (× −alpha)

Effect: the shared encoder is pushed toward features that are
  • useful for emotion classification  (emotion loss ↓)
  • useless for identifying the subject (domain loss ↑ → GRL → encoder updates ↓)

Result: subject-invariant feature space → better LOSO.

Architecture
─────────────
  EEG(B,192) ──►─┐
  PPG(B, 10) ──►─┤── Shared Encoder ──► h(B,32) ──► val_head → (B,2)
  GSR(B, 10) ──►─┘                          │       ar_head  → (B,2)
                                             │
                                            GRL(alpha)
                                             │
                                        subj_head → (B, n_subj)  [only during training]

alpha scheduling: 0 → 1 over training steps (sigmoid ramp, Ganin et al.)
Modality Dropout retained for inference robustness on new devices.

References
──────────
  Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks"
  Li et al. (2019)    "Cross-Subject EEG Emotion Recognition with DANN"
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .multimodal import ModalityDropout


# ── Gradient Reversal ─────────────────────────────────────────────────────

class _GradReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.alpha * grad, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with coefficient alpha."""
    return _GradReversalFn.apply(x, alpha)


def dann_alpha(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """
    Schedule alpha from 0 → 1 using the sigmoid ramp from Ganin et al.

        p     = step / total_steps           (0 → 1)
        alpha = 2 / (1 + exp(-gamma * p)) - 1
    """
    p = min(step / max(total_steps, 1), 1.0)
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


# ── Model ─────────────────────────────────────────────────────────────────

class DANNNet(nn.Module):
    """
    Domain-Adversarial Network for emotion recognition.

    Same shared encoder as MultiModalNet, plus a subject classifier
    connected through a Gradient Reversal Layer.

    Parameters
    ----------
    in_eeg, in_ppg, in_gsr : feature dimensions per window
    n_subjects             : number of training subjects
                             (LOSO with 32 subjects → 31)
    dropout                : neuron dropout
    modality_dropout       : per-modality dropout (Modality Dropout)
    """

    def __init__(
        self,
        in_eeg: int = 192,
        in_ppg: int = 10,
        in_gsr: int = 10,
        n_subjects: int = 31,
        dropout: float = 0.30,
        modality_dropout: float = 0.20,
    ):
        super().__init__()

        # ── Modality Dropout ──────────────────────────────────────────────
        self.eeg_drop = ModalityDropout(modality_dropout)
        self.ppg_drop = ModalityDropout(modality_dropout)
        self.gsr_drop = ModalityDropout(modality_dropout)

        # ── Shared encoder (mirrors MultiModalNet exactly) ────────────────
        self.eeg_enc = nn.Sequential(
            nn.Linear(in_eeg, 96), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 64), nn.GELU(),
        )
        self.ppg_enc = nn.Sequential(
            nn.Linear(in_ppg, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 32), nn.GELU(),
        )
        self.gsr_enc = nn.Sequential(
            nn.Linear(in_gsr, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 32), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(),
        )

        # ── Emotion heads ─────────────────────────────────────────────────
        self.val_head = nn.Linear(32, 2)
        self.ar_head  = nn.Linear(32, 2)

        # ── Domain (subject) classifier — used only during training ───────
        self.subj_head = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, n_subjects),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Shared feature extraction ─────────────────────────────────────────

    def encode(self, eeg: torch.Tensor, ppg: torch.Tensor,
               gsr: torch.Tensor) -> torch.Tensor:
        """Forward through shared encoder. Returns (B, 32) representation."""
        eeg = self.eeg_drop(eeg)
        ppg = self.ppg_drop(ppg)
        gsr = self.gsr_drop(gsr)
        cat = torch.cat(
            [self.eeg_enc(eeg), self.ppg_enc(ppg), self.gsr_enc(gsr)],
            dim=1,
        )
        return self.fusion(cat)   # (B, 32)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        eeg: torch.Tensor,
        ppg: torch.Tensor,
        gsr: torch.Tensor,
        alpha: float = 0.0,
    ):
        """
        Parameters
        ----------
        alpha : GRL coefficient — 0 at start of training, rises to 1.
                Pass alpha=0 (default) for inference → subj_head not computed.

        Returns
        -------
        Training (alpha > 0):
            (val_logits, ar_logits, subj_logits)
        Inference (alpha == 0):
            (val_logits, ar_logits)
        """
        h = self.encode(eeg, ppg, gsr)

        val_out  = self.val_head(h)
        ar_out   = self.ar_head(h)

        if self.training and alpha > 0.0:
            h_rev    = grad_reverse(h, alpha)
            subj_out = self.subj_head(h_rev)
            return val_out, ar_out, subj_out

        return val_out, ar_out
