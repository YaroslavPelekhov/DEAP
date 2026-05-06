"""
TemporalDANNNet — TemporalNet + Domain-Adversarial adaptation.

Why trial-level GRL is better than window-level:
  TemporalNet's GRU aggregates 60 windows into one trial embedding h (B, 128).
  Subject-specific physiology (baseline HR, resting EEG power) manifests as
  a consistent offset across the whole trial — captured in h, not individual
  windows. Applying GRL at this level is semantically correct.

Architecture:
  EEG (B,T,192) ─┐
  PPG (B,T, 10) ─┤─ window MLPs → Bi-GRU → h (B,128) ──► val_head → (B,2)
  GSR (B,T, 10) ─┘                                  │    ar_head  → (B,2)
                                                     │
                                                   GRL(alpha)
                                                     │
                                               subj_head → (B, n_subj)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .dann import grad_reverse, dann_alpha   # reuse GRL from dann.py


class TemporalDANNNet(nn.Module):
    """
    TemporalNet with a Gradient Reversal subject classifier attached
    to the GRU's final hidden state.

    Parameters
    ----------
    in_eeg, in_ppg, in_gsr : feature dims per window
    n_subjects             : number of training subjects (LOSO: 31)
    gru_hidden             : GRU hidden size (bidirectional → 2×)
    gru_layers             : GRU layers
    dropout                : dropout rate
    """

    def __init__(
        self,
        in_eeg: int = 192,
        in_ppg: int = 10,
        in_gsr: int = 10,
        n_subjects: int = 31,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        D = dropout

        # ── Window encoders (identical to TemporalNet) ────────────────────
        self.eeg_enc = nn.Sequential(
            nn.Linear(in_eeg, 96), nn.GELU(), nn.Dropout(D),
            nn.Linear(96, 64), nn.GELU(),
        )
        self.ppg_enc = nn.Sequential(
            nn.Linear(in_ppg, 32), nn.GELU(), nn.Dropout(D),
            nn.Linear(32, 32), nn.GELU(),
        )
        self.gsr_enc = nn.Sequential(
            nn.Linear(in_gsr, 32), nn.GELU(), nn.Dropout(D),
            nn.Linear(32, 32), nn.GELU(),
        )

        win_emb   = 64 + 32 + 32   # 128
        self.gru  = nn.GRU(
            win_emb, gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if gru_layers > 1 else 0.0,
        )
        trial_dim = gru_hidden * 2   # 128

        # ── Emotion heads ─────────────────────────────────────────────────
        self.val_head  = nn.Linear(trial_dim, 2)
        self.ar_head   = nn.Linear(trial_dim, 2)

        # ── Subject classifier (applied via GRL during training) ──────────
        self.subj_head = nn.Sequential(
            nn.Linear(trial_dim, 64), nn.GELU(), nn.Dropout(D),
            nn.Linear(64, n_subjects),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, eeg, ppg, gsr) -> torch.Tensor:
        """Returns trial embedding h: (B, gru_hidden*2)."""
        B, T, _ = eeg.shape
        e = self.eeg_enc(eeg.reshape(B * T, -1)).reshape(B, T, -1)
        p = self.ppg_enc(ppg.reshape(B * T, -1)).reshape(B, T, -1)
        g = self.gsr_enc(gsr.reshape(B * T, -1)).reshape(B, T, -1)
        h_seq = torch.cat([e, p, g], dim=2)       # (B, T, 128)
        _, hn  = self.gru(h_seq)                   # hn: (layers*2, B, hidden)
        return torch.cat([hn[-2], hn[-1]], dim=1)  # (B, trial_dim)

    def forward(self, eeg, ppg, gsr, alpha: float = 0.0):
        """
        Training (self.training=True):  returns (val, ar, subj_logits)
        Inference (self.training=False): returns (val, ar)
        """
        h = self.encode(eeg, ppg, gsr)
        val_out = self.val_head(h)
        ar_out  = self.ar_head(h)

        if self.training:
            h_rev    = grad_reverse(h, alpha)
            subj_out = self.subj_head(h_rev)
            return val_out, ar_out, subj_out

        return val_out, ar_out
