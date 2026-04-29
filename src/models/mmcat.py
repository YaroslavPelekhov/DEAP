"""
MMCAT — MultiModal Cross-Attention Transformer for emotion recognition.

Architecture:
  EEG  (160-dim DE features) ──► EEG Encoder  (Linear + Transformer) ──► 128-dim ─┐
  PPG  ( 10-dim HRV features) ──► PPG Encoder  (MLP)                  ──►  64-dim ─┤
  GSR  (  8-dim EDA features) ──► GSR Encoder  (MLP)                  ──►  64-dim ─┘
                                                                                    │
                                         Cross-Attention Fusion (EEG ← PPG+GSR) ◄──┘
                                                    │
                                           Fusion MLP (256 → 128)
                                                    │
                                     ┌──────────────┴──────────────┐
                                  Valence head               Arousal head
                                   Linear(128, 2)             Linear(128, 2)

References:
  - Zheng & Lu (2015) "Investigating Critical Frequency Bands ..."
  - Song et al. (2018) FCAN with cross-modal attention
  - Yin et al. (2021) EEG + peripheral fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    N_EEG_FEATURES, N_PPG_FEATURES, N_GSR_FEATURES, N_CONSENSUS_FEATURES,
    EEG_EMBED_DIM, PERIPHERAL_EMBED_DIM, FUSION_DIM,
    N_HEADS, N_TRANSFORMER_LAYERS, DROPOUT,
)


class MLPEncoder(nn.Module):
    """Small MLP for peripheral signals."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGEncoder(nn.Module):
    """
    Project 160-dim DE features → EEG_EMBED_DIM, then apply
    a shallow Transformer (treating each frequency band as a token).
    """
    N_BANDS = 5
    N_CHANNELS = 32          # 32 × 5 = 160

    def __init__(self):
        super().__init__()
        band_in = self.N_CHANNELS   # one token per band: 32 channels as features

        # Project each band-vector to embedding space
        self.band_proj = nn.Linear(band_in, EEG_EMBED_DIM)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.N_BANDS, EEG_EMBED_DIM)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EEG_EMBED_DIM,
            nhead=N_HEADS,
            dim_feedforward=EEG_EMBED_DIM * 4,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,    # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_TRANSFORMER_LAYERS,
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(EEG_EMBED_DIM * self.N_BANDS, EEG_EMBED_DIM),
            nn.LayerNorm(EEG_EMBED_DIM),
            nn.GELU(),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, eeg_de: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_de: (B, 160)  — flattened DE features [ch0_b0, ch0_b1, ..., ch31_b4]
                    layout: reshape to (B, 32, 5) then transpose to (B, 5, 32)
        """
        B = eeg_de.shape[0]
        # Reshape: (B, 32, 5) — channels first, then bands
        x = eeg_de.view(B, self.N_CHANNELS, self.N_BANDS)
        # Transpose to (B, 5, 32) — 5 tokens of dim 32
        x = x.permute(0, 2, 1)
        # Project each band-token to embedding dim
        x = self.band_proj(x)               # (B, 5, EEG_EMBED_DIM)
        x = x + self.pos_embed
        x = self.transformer(x)             # (B, 5, EEG_EMBED_DIM)
        x = x.reshape(B, -1)               # (B, 5 * EEG_EMBED_DIM)
        return self.out_proj(x)             # (B, EEG_EMBED_DIM)


class CrossModalAttention(nn.Module):
    """
    EEG features attend to concatenated peripheral features.
    Query = EEG,  Key/Value = [PPG; GSR]
    """
    def __init__(self, query_dim: int, kv_dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=n_heads,
            kdim=kv_dim, vdim=kv_dim,
            dropout=DROPOUT, batch_first=True,
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, query_dim)
            kv:    (B, kv_dim)
        Returns:
            (B, query_dim)
        """
        # Add seq-len dim of 1
        q = query.unsqueeze(1)  # (B, 1, query_dim)
        k = kv.unsqueeze(1)     # (B, 1, kv_dim)
        out, _ = self.attn(q, k, k)         # (B, 1, query_dim)
        out = out.squeeze(1)                # (B, query_dim)
        return self.norm(query + out)       # residual + LN


class MMCAT(nn.Module):
    """
    MultiModal Cross-Attention Transformer for DEAP emotion recognition.

    Input:
      eeg_de : (B, 160) — differential entropy features
      ppg    : (B,  10) — HRV features
      gsr    : (B,   8) — EDA features

    Output:
      valence_logits: (B, 2)
      arousal_logits: (B, 2)
    """
    def __init__(self):
        super().__init__()
        self.eeg_enc = EEGEncoder()
        self.ppg_enc = MLPEncoder(N_PPG_FEATURES, PERIPHERAL_EMBED_DIM)
        self.gsr_enc = MLPEncoder(N_GSR_FEATURES + N_CONSENSUS_FEATURES, PERIPHERAL_EMBED_DIM)

        # Peripheral combined dim
        periph_dim = PERIPHERAL_EMBED_DIM * 2   # 64 + 64 = 128

        self.cross_attn = CrossModalAttention(
            query_dim=EEG_EMBED_DIM,
            kv_dim=periph_dim,
            n_heads=4,
        )

        # Fusion: concat EEG-attended + raw peripheral → project
        self.fusion = nn.Sequential(
            nn.Linear(EEG_EMBED_DIM + periph_dim, FUSION_DIM * 2),
            nn.BatchNorm1d(FUSION_DIM * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_DIM * 2, FUSION_DIM),
            nn.BatchNorm1d(FUSION_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),
        )

        self.valence_head = nn.Linear(FUSION_DIM, 2)
        self.arousal_head = nn.Linear(FUSION_DIM, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        eeg_de: torch.Tensor,
        ppg: torch.Tensor,
        gsr: torch.Tensor,
    ):
        eeg_emb   = self.eeg_enc(eeg_de)       # (B, 128)
        ppg_emb   = self.ppg_enc(ppg)          # (B,  64)
        gsr_emb   = self.gsr_enc(gsr)          # (B,  64)
        periph    = torch.cat([ppg_emb, gsr_emb], dim=1)  # (B, 128)

        eeg_fused = self.cross_attn(eeg_emb, periph)  # (B, 128)
        fused     = self.fusion(torch.cat([eeg_fused, periph], dim=1))  # (B, 128)

        return self.valence_head(fused), self.arousal_head(fused)

    def predict_proba(
        self,
        eeg_de: torch.Tensor,
        ppg: torch.Tensor,
        gsr: torch.Tensor,
    ):
        val_logits, ar_logits = self.forward(eeg_de, ppg, gsr)
        return F.softmax(val_logits, dim=-1), F.softmax(ar_logits, dim=-1)
