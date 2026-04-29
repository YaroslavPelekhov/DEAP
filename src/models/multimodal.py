"""
MultiModalNet — window-level multimodal emotion classifier.

Separate encoders per modality + Modality Dropout (p=0.2).
Variable input sizes: pass in_eeg/in_ppg/in_gsr to __init__.

Usage:
    model = MultiModalNet(in_eeg=192, in_ppg=10, in_gsr=10)
    val_logits, ar_logits = model(eeg_tensor, ppg_tensor, gsr_tensor)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityDropout(nn.Module):
    """Zero out an entire modality with probability p during training."""
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        keep = (torch.rand(x.size(0), 1, device=x.device) > self.p).float()
        return x * keep


class MultiModalNet(nn.Module):
    """
    Args:
        in_eeg: EEG feature dimension (e.g. 192 for 32ch × 6)
        in_ppg: PPG feature dimension (default 10)
        in_gsr: GSR feature dimension (default 10, includes FAA/FTA)
        dropout: dropout rate
        modality_dropout: per-modality dropout probability
    """

    def __init__(
        self,
        in_eeg: int = 192,
        in_ppg: int = 10,
        in_gsr: int = 10,
        dropout: float = 0.30,
        modality_dropout: float = 0.20,
    ):
        super().__init__()
        D = dropout

        self.eeg_drop = ModalityDropout(modality_dropout)
        self.ppg_drop = ModalityDropout(modality_dropout)
        self.gsr_drop = ModalityDropout(modality_dropout)

        self.eeg_enc = nn.Sequential(
            nn.Linear(in_eeg, 96), nn.BatchNorm1d(96), nn.GELU(), nn.Dropout(D),
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
        self.fusion = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(D),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.val_head = nn.Linear(32, 2)
        self.ar_head  = nn.Linear(32, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        eeg: torch.Tensor,
        ppg: torch.Tensor,
        gsr: torch.Tensor,
    ):
        eeg = self.eeg_drop(eeg)
        ppg = self.ppg_drop(ppg)
        gsr = self.gsr_drop(gsr)
        emb = torch.cat([self.eeg_enc(eeg), self.ppg_enc(ppg), self.gsr_enc(gsr)], dim=1)
        h   = self.fusion(emb)
        return self.val_head(h), self.ar_head(h)
