"""
EEGNet + Peripheral Fusion for DEAP emotion recognition.

EEGNet (Lawhern et al., 2018) — compact CNN for EEG classification.
Paper: "EEGNet: A Compact Convolutional Neural Network for
        EEG-based Brain-Computer Interfaces"

Architecture:
  Block 1: Temporal conv → Depthwise spatial conv → AvgPool
  Block 2: Separable conv → AvgPool
  → EEG embedding (64-dim)

Multimodal fusion:
  EEG (1, 32, 128) → EEGNet       → 64-dim  ─┐
  PPG (10,)        → MLP Encoder  → 32-dim  ─┤→ FC → [Val, Ar]
  GSR (8,)         → MLP Encoder  → 32-dim  ─┘

Total params: ~15K (vs 660K MMCAT) — appropriate for 2100 training samples.

Expected DEAP SD accuracy: ~78-82% (Lawhern et al. 2018, Table I)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_PPG_FEATURES, N_GSR_FEATURES, N_CONSENSUS_FEATURES, SFREQ


class EEGNet(nn.Module):
    """
    Pure EEGNet backbone.

    Input : (B, 1, n_channels, n_times)  e.g. (B, 1, 32, 128)
    Output: (B, eeg_embed_dim)
    """
    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = 128,          # samples per window = SFREQ * window_sec
        F1: int = 8,                 # temporal filters
        D: int = 2,                  # depth multiplier (spatial)
        F2: int = 16,                # separable conv filters (= F1 * D)
        kernel_length: int = 64,     # temporal kernel = SFREQ // 2
        dropout_rate: float = 0.5,
        eeg_embed_dim: int = 64,
    ):
        super().__init__()
        self.F2 = F2
        self.eeg_embed_dim = eeg_embed_dim

        # ── Block 1 ────────────────────────────────────────────────────────
        # Temporal convolution: learn frequency-specific filters
        self.conv1 = nn.Conv2d(
            1, F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution: learn channel-specific spatial filters
        self.depthwise = nn.Conv2d(
            F1, F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ── Block 2 ────────────────────────────────────────────────────────
        # Separable conv = depthwise + pointwise
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                      padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute flattened size after both blocks
        # After temporal conv + pad: (1, F1, C, T)
        # After depthwise: (1, F1*D, 1, T)
        # After pool1: (1, F1*D, 1, T//4)
        # After separable + pool2: (1, F2, 1, T//32)
        flat_size = F2 * (n_times // 32)

        self.proj = nn.Sequential(
            nn.Linear(flat_size, eeg_embed_dim),
            nn.BatchNorm1d(eeg_embed_dim),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_channels, n_times)"""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten + project
        x = x.flatten(1)
        return self.proj(x)


class EEGNetMultimodal(nn.Module):
    """
    EEGNet backbone + PPG + GSR for multimodal emotion recognition.

    Inputs:
      eeg : (B, 1, 32, 128)  raw 1-second EEG window
      ppg : (B, 10)           HRV features (trial-level, repeated per window)
      gsr : (B,  8)           EDA features (trial-level, repeated per window)

    Outputs:
      valence_logits : (B, 2)
      arousal_logits : (B, 2)
    """
    def __init__(
        self,
        n_channels: int = 32,
        n_times: int = SFREQ,       # 128 samples = 1 second
        eeg_embed_dim: int = 64,
        periph_embed_dim: int = 32,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            eeg_embed_dim=eeg_embed_dim,
            dropout_rate=dropout_rate,
        )

        # Small MLP encoders for peripheral signals
        self.ppg_enc = nn.Sequential(
            nn.Linear(N_PPG_FEATURES, periph_embed_dim),
            nn.BatchNorm1d(periph_embed_dim),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.gsr_enc = nn.Sequential(
            nn.Linear(N_GSR_FEATURES + N_CONSENSUS_FEATURES, periph_embed_dim),
            nn.BatchNorm1d(periph_embed_dim),
            nn.ELU(),
            nn.Dropout(0.3),
        )

        fusion_dim = eeg_embed_dim + 2 * periph_embed_dim  # 64+32+32=128

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
        )
        self.valence_head = nn.Linear(64, 2)
        self.arousal_head = nn.Linear(64, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, eeg, ppg, gsr):
        eeg_emb  = self.eegnet(eeg)           # (B, 64)
        ppg_emb  = self.ppg_enc(ppg)          # (B, 32)
        gsr_emb  = self.gsr_enc(gsr)          # (B, 32)
        fused    = torch.cat([eeg_emb, ppg_emb, gsr_emb], dim=1)  # (B, 128)
        h        = self.classifier(fused)     # (B, 64)
        return self.valence_head(h), self.arousal_head(h)
