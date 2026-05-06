"""
Feature extraction package.

Extractor classes
─────────────────
  DEHjorthExtractor   – DE + Hjorth per EEG channel per window   (MNE-aware)
  DEExtractor         – Differential Entropy only                 (MNE-aware)
  HjorthExtractor     – Hjorth Mobility + Complexity              (MNE-aware)
  FAAExtractor        – Frontal Alpha/Theta Asymmetry             (MNE-aware)
  HRVExtractor        – PPG heart-rate variability                (MNE-aware)
  EDAExtractor        – GSR electrodermal activity                (MNE-aware)

Factory
───────
  create_extractor(name, **kwargs)  – build one extractor by name
  ExtractorFactory                  – build lists from config dicts;
                                      default_deap() / default_barometer()

Pipelines
─────────
  MNEFeaturePipeline  – Epochs → aligned feature dict  (new, recommended)
  FeaturePipeline     – .dat files → cached arrays     (legacy DEAP)
"""

from .base import BaseExtractor, WindowExtractor, EpochExtractor

from .eeg import (
    DEHjorthExtractor,
    DEExtractor,
    HjorthExtractor,
    FAAExtractor,
    EEGExtractor,          # legacy
    DEFAULT_BANDS,
    hjorth_params,
)

from .ppg import HRVExtractor, extract_ppg_features, extract_ppg_subject
from .gsr import EDAExtractor, extract_gsr_features, extract_gsr_subject

from .factory import create_extractor, ExtractorFactory
from .pipeline import MNEFeaturePipeline, FeaturePipeline

__all__ = [
    # Base
    'BaseExtractor', 'WindowExtractor', 'EpochExtractor',
    # EEG
    'DEHjorthExtractor', 'DEExtractor', 'HjorthExtractor', 'FAAExtractor',
    'EEGExtractor', 'DEFAULT_BANDS', 'hjorth_params',
    # PPG
    'HRVExtractor', 'extract_ppg_features', 'extract_ppg_subject',
    # GSR
    'EDAExtractor', 'extract_gsr_features', 'extract_gsr_subject',
    # Factory
    'create_extractor', 'ExtractorFactory',
    # Pipelines
    'MNEFeaturePipeline', 'FeaturePipeline',
]
