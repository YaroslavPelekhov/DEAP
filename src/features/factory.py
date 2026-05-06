"""
Extractor factory — mirrors models/factory.py in structure.

Usage
─────
  # Single extractor by name
  ext = create_extractor('de_hjorth', window_sec=1.0, baseline_correct=True)
  ext = create_extractor('hrv', ppg_ch='PPG')
  ext = create_extractor('eda', gsr_ch='GSR')
  ext = create_extractor('faa')

  # List from config dict (used by MNEFeaturePipeline.from_config)
  extractors = ExtractorFactory.from_config([
      {'name': 'de_hjorth', 'window_sec': 1.0},
      {'name': 'faa'},
      {'name': 'hrv'},
      {'name': 'eda'},
  ])

  # Preset bundles
  extractors = ExtractorFactory.default_deap()
  extractors = ExtractorFactory.default_barometer()

Available names
───────────────
  'de_hjorth'  DEHjorthExtractor   DE + Hjorth per channel  (WindowExtractor)
  'de'         DEExtractor         DE only                  (WindowExtractor)
  'hjorth'     HjorthExtractor     Hjorth only              (WindowExtractor)
  'faa'        FAAExtractor        Frontal asymmetry        (EpochExtractor)
  'hrv'        HRVExtractor        PPG HRV features         (EpochExtractor)
  'eda'        EDAExtractor        GSR EDA features         (EpochExtractor)
"""
from __future__ import annotations

from typing import Any

from .base import BaseExtractor


# ── Registry ───────────────────────────────────────────────────────────────

def _registry() -> dict[str, type[BaseExtractor]]:
    """Lazy import to avoid circular deps."""
    from .eeg import DEHjorthExtractor, DEExtractor, HjorthExtractor, FAAExtractor
    from .ppg import HRVExtractor
    from .gsr import EDAExtractor
    return {
        'de_hjorth': DEHjorthExtractor,
        'de':        DEExtractor,
        'hjorth':    HjorthExtractor,
        'faa':       FAAExtractor,
        'hrv':       HRVExtractor,
        'eda':       EDAExtractor,
    }


# ── Public API ─────────────────────────────────────────────────────────────

def create_extractor(name: str, **kwargs: Any) -> BaseExtractor:
    """
    Create a feature extractor by name.

    Parameters
    ----------
    name   : extractor key (see module docstring for full list)
    **kwargs : passed directly to the extractor constructor

    Returns
    -------
    BaseExtractor instance
    """
    reg = _registry()
    name = name.lower()
    if name not in reg:
        raise ValueError(
            f"Unknown extractor '{name}'. "
            f"Available: {list(reg)}"
        )
    return reg[name](**kwargs)


class ExtractorFactory:
    """
    Build extractor lists from config dicts.

    Mirrors the style of models/factory.py — same pattern, different domain.
    """

    @classmethod
    def from_config(cls, config: list[dict]) -> list[BaseExtractor]:
        """
        Build a list of extractors from a config list.

        Each element is a dict with required key 'name' and optional kwargs.

        Example
        -------
        ExtractorFactory.from_config([
            {'name': 'de_hjorth', 'window_sec': 1.0, 'baseline_correct': True},
            {'name': 'faa', 'left_ch': 'F3', 'right_ch': 'F4'},
            {'name': 'hrv', 'ppg_ch': 'PPG'},
            {'name': 'eda', 'gsr_ch': 'GSR'},
        ])
        """
        extractors = []
        for cfg in config:
            cfg  = dict(cfg)                 # don't mutate caller's dict
            name = cfg.pop('name')
            extractors.append(create_extractor(name, **cfg))
        return extractors

    @classmethod
    def default_deap(cls, window_sec: float = 1.0) -> list[BaseExtractor]:
        """
        Standard extractor bundle for DEAP (32-ch EEG + PPG + GSR, tmin=-3 s).

        DE + Hjorth with baseline correction, FAA/FTA, HRV, EDA.
        """
        return cls.from_config([
            {'name': 'de_hjorth', 'window_sec': window_sec,
             'baseline_correct': True},
            {'name': 'faa'},
            {'name': 'hrv', 'ppg_ch': 'PPG'},
            {'name': 'eda', 'gsr_ch': 'GSR'},
        ])

    @classmethod
    def default_barometer(cls, window_sec: float = 1.0) -> list[BaseExtractor]:
        """
        Standard extractor bundle for NeuroBarometer (19-ch EEG + PPG + GSR,
        tmin=0 s — no baseline segment).

        Same feature types, baseline correction disabled.
        """
        return cls.from_config([
            {'name': 'de_hjorth', 'window_sec': window_sec,
             'baseline_correct': False},
            {'name': 'faa'},
            {'name': 'hrv', 'ppg_ch': 'PPG'},
            {'name': 'eda', 'gsr_ch': 'GSR'},
        ])
