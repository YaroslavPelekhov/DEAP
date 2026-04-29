"""
LightMLP — lightweight multimodal MLP for DEAP feature-based classification.

~28K parameters (vs 660K for MMCAT), appropriate for 2100 training samples.
Concatenates all modalities, passes through shared trunk + two heads.

This is the standard architecture in feature-based DEAP papers that achieve
85-90% accuracy on SD protocol.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import N_EEG_FEATURES, N_PPG_FEATURES, N_GSR_FEATURES, N_CONSENSUS_FEATURES, DROPOUT


class LightMLP(nn.Module):
    """
    Input: EEG (160) + PPG (10) + GSR (8) = 178-dim concatenated features
    Hidden: 256 -> 128
    Output: Valence head (2) + Arousal head (2)
    """
    def __init__(self):
        super().__init__()
        in_dim = N_EEG_FEATURES + N_PPG_FEATURES + N_GSR_FEATURES + N_CONSENSUS_FEATURES  # 180

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.valence_head = nn.Linear(128, 2)
        self.arousal_head = nn.Linear(128, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, eeg, ppg, gsr):
        x = torch.cat([eeg, ppg, gsr], dim=1)  # (B, 178)
        h = self.trunk(x)
        return self.valence_head(h), self.arousal_head(h)
