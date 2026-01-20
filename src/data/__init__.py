"""
Data loading utilities for neural decoding.

Provides loaders for various neural data formats and benchmark datasets:
- NLB: Neural Latents Benchmark datasets
- DANDI: DANDI Archive neural data
- NWB: Neurodata Without Borders format

Also includes data augmentation utilities for improving decoder robustness.
"""

from src.data.augmentation import NeuralDataAugmenter, augment_trials, cutmix_temporal, mixup
from src.data.dandi_loader import DANDIDataLoader
from src.data.nlb_loader import NLBDataLoader

__all__ = [
    "NLBDataLoader",
    "DANDIDataLoader",
    "NeuralDataAugmenter",
    "augment_trials",
    "mixup",
    "cutmix_temporal",
]
