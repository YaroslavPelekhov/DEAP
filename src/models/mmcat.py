"""
MMCAT — MultiModal Cross-Attention Transformer for emotion recognition.

Architecture:
  EEG  (n_eeg-dim features) ──► EEG Encoder  (Linear + Transformer) ──► eeg_dim  ─┐
  PPG  (n_ppg-dim features) ──► PPG Encoder  (MLP)                  ──► periph_dim ─┤
  GSR  (n_gsr-dim features) ──► GSR Encoder  (MLP)                  ──► periph_dim ─┘
                                                                                    │
                                     Cross-Attention Fusion (EEG ← PPG+GSR) ◄──────┘
                                                    │
                                           Fusion MLP (eeg_dim + 2*periph_dim → fusion_dim)
                                                    │
                                     ┌──────────────┴──────────────┐
                                  Valence head               Arousal head
                                   Linear(fusion_dim, 2)     Linear(fusion_dim, 2)

All dimensions are configurable — no hardcoded constants.

Default values match the v13 feature set:
  in_eeg=192  (32 ch × 6: DE×4 + Hjorth×2)
  in_ppg=10
  in_gsr=10   (8 EDA + 2 FAA/FTA)

References:
  - Song et al. (2018) FCAN with cross-modal attention
  - Yin et al. (2021) EEG + peripheral fusion
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Small MLP encoder for peripheral signals."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGTransformerEncoder(nn.Module):
    """
    EEG encoder: treats each feature-group (band or descriptor) as a token.

    The input vector of shape (B, n_ch * n_feat_per_ch) is reshaped to
    (B, n_feat_per_ch, n_ch) — one token per feature type, each of dimension n_ch.
    A shallow Transformer then mixes across feature types.

    Args:
        in_eeg         : total EEG feature dim (e.g. 192 = 32 × 6)
        n_ch           : number of EEG channels  (e.g. 32)
        n_feat_per_ch  : features per channel     (e.g. 6)
        out_dim        : output embedding dim
        n_heads        : Transformer attention heads
        n_layers       : Transformer encoder layers
        dropout        : dropout rate
    """

    def __init__(
        self,
        in_eeg: int = 192,
        n_ch: int = 32,
        n_feat_per_ch: int = 6,
        out_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert in_eeg == n_ch * n_feat_per_ch, (
            f'in_eeg={in_eeg} must equal n_ch × n_feat_per_ch = {n_ch}×{n_feat_per_ch}'
        )
        self.n_ch          = n_ch
        self.n_feat_per_ch = n_feat_per_ch
        self.n_tokens      = n_feat_per_ch   # one token per feature type
        token_dim          = n_ch            # channel values as token features

        self.band_proj = nn.Linear(token_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens, out_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=n_heads,
            dim_feedforward=out_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,   # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim * self.n_tokens, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, n_ch * n_feat_per_ch)  →  (B, out_dim)"""
        B = eeg.shape[0]
        # (B, n_ch, n_feat_per_ch) → (B, n_feat_per_ch, n_ch)
        x = eeg.view(B, self.n_ch, self.n_feat_per_ch).permute(0, 2, 1)
        x = self.band_proj(x)           # (B, n_tokens, out_dim)
        x = x + self.pos_embed
        x = self.transformer(x)         # (B, n_tokens, out_dim)
        x = x.reshape(B, -1)           # (B, n_tokens * out_dim)
        return self.out_proj(x)         # (B, out_dim)


class CrossModalAttention(nn.Module):
    """
    EEG features attend to concatenated peripheral features.
    Query = EEG,  Key/Value = [PPG; GSR]
    """
    def __init__(self, query_dim: int, kv_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim, num_heads=n_heads,
            kdim=kv_dim, vdim=kv_dim,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q   = query.unsqueeze(1)           # (B, 1, query_dim)
        k   = kv.unsqueeze(1)             # (B, 1, kv_dim)
        out, _ = self.attn(q, k, k)
        out = out.squeeze(1)              # (B, query_dim)
        return self.norm(query + out)     # residual + LN


class MMCAT(nn.Module):
    """
    MultiModal Cross-Attention Transformer.

    Parameters
    ----------
    in_eeg         : EEG feature dim (default 192 = 32 ch × 6 feats)
    in_ppg         : PPG feature dim (default 10)
    in_gsr         : GSR feature dim (default 10)
    n_ch           : EEG channels    (default 32; 19 for NeuroBarometer)
    n_feat_per_ch  : feats/channel   (default 6: DE×4 + Hjorth×2)
    eeg_dim        : EEG encoder output dim
    periph_dim     : PPG/GSR encoder output dim each
    fusion_dim     : fusion MLP output dim
    n_heads        : attention heads
    n_layers       : Transformer layers
    dropout        : dropout rate

    Input:
      eeg : (B, in_eeg)
      ppg : (B, in_ppg)
      gsr : (B, in_gsr)

    Output:
      valence_logits : (B, 2)
      arousal_logits : (B, 2)
    """

    def __init__(
        self,
        in_eeg: int = 192,
        in_ppg: int = 10,
        in_gsr: int = 10,
        n_ch: int = 32,
        n_feat_per_ch: int = 6,
        eeg_dim: int = 128,
        periph_dim: int = 64,
        fusion_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.eeg_enc = EEGTransformerEncoder(
            in_eeg=in_eeg, n_ch=n_ch, n_feat_per_ch=n_feat_per_ch,
            out_dim=eeg_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.ppg_enc = MLPEncoder(in_ppg, periph_dim, dropout=dropout)
        self.gsr_enc = MLPEncoder(in_gsr, periph_dim, dropout=dropout)

        periph_total = periph_dim * 2
        self.cross_attn = CrossModalAttention(
            query_dim=eeg_dim, kv_dim=periph_total, n_heads=n_heads,
        )

        self.fusion = nn.Sequential(
            nn.Linear(eeg_dim + periph_total, fusion_dim * 2),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.valence_head = nn.Linear(fusion_dim, 2)
        self.arousal_head = nn.Linear(fusion_dim, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        eeg: torch.Tensor,
        ppg: torch.Tensor,
        gsr: torch.Tensor,
    ):
        eeg_emb   = self.eeg_enc(eeg)                              # (B, eeg_dim)
        ppg_emb   = self.ppg_enc(ppg)                             # (B, periph_dim)
        gsr_emb   = self.gsr_enc(gsr)                             # (B, periph_dim)
        periph    = torch.cat([ppg_emb, gsr_emb], dim=1)          # (B, periph_dim*2)

        eeg_fused = self.cross_attn(eeg_emb, periph)              # (B, eeg_dim)
        fused     = self.fusion(
            torch.cat([eeg_fused, periph], dim=1)
        )                                                          # (B, fusion_dim)

        return self.valence_head(fused), self.arousal_head(fused)

    def predict_proba(self, eeg, ppg, gsr):
        val_logits, ar_logits = self.forward(eeg, ppg, gsr)
        return F.softmax(val_logits, dim=-1), F.softmax(ar_logits, dim=-1)
