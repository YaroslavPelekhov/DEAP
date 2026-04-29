"""
Model weight persistence and results I/O.

Usage:
    save_model(model, path)
    model = load_model(path, 'multimodal', in_eeg=192)

    save_results(results_dict, path)
    results = load_results(path)
"""
from __future__ import annotations

import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Any


def save_model(model: nn.Module, path: Path | str, meta: dict | None = None) -> None:
    """Save model weights (+ optional metadata) to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'state_dict': model.state_dict()}
    if meta:
        payload['meta'] = meta
    torch.save(payload, path)
    print(f'[io] Saved model → {path}')


def load_model(
    path: Path | str,
    model_name: str,
    **model_kwargs,
) -> nn.Module:
    """Load model weights from a .pt file."""
    from ..models.factory import create_model
    path = Path(path)
    payload  = torch.load(path, map_location='cpu')
    model    = create_model(model_name, **model_kwargs)
    model.load_state_dict(payload['state_dict'])
    meta = payload.get('meta', {})
    print(f'[io] Loaded model from {path}  meta={meta}')
    return model


def save_results(results: dict, path: Path | str) -> None:
    """Save results dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    print(f'[io] Results → {path}')


def load_results(path: Path | str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def timestamped_path(base_dir: Path | str, prefix: str, ext: str = 'json') -> Path:
    """E.g. results/results_multimodal_20260427_183000.json"""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path(base_dir) / f'{prefix}_{ts}.{ext}'
