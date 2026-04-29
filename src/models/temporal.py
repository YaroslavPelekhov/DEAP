"""
TemporalNet — trial-level emotion classifier using bidirectional GRU.

Processes a sequence of T windows per trial:
  1. Each window encoded by per-modality MLPs  → 128-dim
  2. Bidirectional 2-layer GRU over T windows  → final hidden state
  3. Dual classification heads (valence / arousal)

Input:  eeg (B, T, in_eeg),  ppg (B, T, in_ppg),  gsr (B, T, in_gsr)
Output: val_logits (B, 2),   ar_logits (B, 2)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalNet(nn.Module):
    """
    Args:
        in_eeg: EEG feature dimension per window
        in_ppg: PPG feature dimension per window
        in_gsr: GSR feature dimension per window
        gru_hidden: GRU hidden size (bidirectional → 2× this)
        gru_layers: number of GRU layers
        dropout: dropout rate
    """

    def __init__(
        self,
        in_eeg: int = 192,
        in_ppg: int = 10,
        in_gsr: int = 10,
        gru_hidden: int = 64,
        gru_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        D = dropout

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

        win_emb = 64 + 32 + 32   # 128
        self.gru = nn.GRU(
            win_emb, gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True,
            dropout=0.2 if gru_layers > 1 else 0.0,
        )
        fused_dim = gru_hidden * 2   # bidirectional

        self.val_head = nn.Linear(fused_dim, 2)
        self.ar_head  = nn.Linear(fused_dim, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        eeg: torch.Tensor,   # (B, T, in_eeg)
        ppg: torch.Tensor,   # (B, T, in_ppg)
        gsr: torch.Tensor,   # (B, T, in_gsr)
    ):
        B, T, _ = eeg.shape
        e = self.eeg_enc(eeg.reshape(B * T, -1)).reshape(B, T, -1)
        p = self.ppg_enc(ppg.reshape(B * T, -1)).reshape(B, T, -1)
        g = self.gsr_enc(gsr.reshape(B * T, -1)).reshape(B, T, -1)

        h_seq = torch.cat([e, p, g], dim=2)      # (B, T, 128)
        _, hn = self.gru(h_seq)                   # hn: (layers*2, B, hidden)
        # Take last forward + last backward hidden states
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, hidden*2)
        return self.val_head(hn), self.ar_head(hn)
