"""
Model factory.

Usage:
    model = create_model('multimodal', in_eeg=192, in_ppg=10, in_gsr=10)
    model = create_model('temporal',   in_eeg=192, in_ppg=10, in_gsr=10)
    model = create_model('mmcat')      # original MMCAT transformer
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def create_model(
    name: str,
    in_eeg: int = 192,
    in_ppg: int = 10,
    in_gsr: int = 10,
    **kwargs,
) -> nn.Module:
    """
    Create a model by name.

    Args:
        name:   'multimodal' | 'temporal' | 'mmcat'
        in_eeg: EEG input dimension
        in_ppg: PPG input dimension
        in_gsr: GSR input dimension
        **kwargs: passed to model constructor

    Returns:
        nn.Module
    """
    name = name.lower()

    if name == 'multimodal':
        from .multimodal import MultiModalNet
        return MultiModalNet(in_eeg=in_eeg, in_ppg=in_ppg, in_gsr=in_gsr, **kwargs)

    if name == 'temporal':
        from .temporal import TemporalNet
        return TemporalNet(in_eeg=in_eeg, in_ppg=in_ppg, in_gsr=in_gsr, **kwargs)

    if name == 'mmcat':
        from .mmcat import MMCAT
        return MMCAT(**kwargs)

    raise ValueError(f"Unknown model: '{name}'. Choose from: multimodal, temporal, mmcat")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
