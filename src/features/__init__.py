"""
Feature extraction module for neural decoding.

Provides firing rate computation, spectral feature extraction,
and unified feature extraction pipeline.
"""

from src.features.extractor import FeatureExtractor
from src.features.firing_rates import (
    bin_spikes,
    causal_smooth_firing_rate,
    compute_firing_rates,
    compute_instantaneous_firing_rate,
    gaussian_smooth_firing_rate,
    spike_train_to_firing_rate_multichannel,
)
from src.features.spectral import (
    FREQUENCY_BANDS,
    compute_band_power,
    compute_band_power_timeseries,
    compute_psd,
    compute_spectrogram,
    extract_spectral_features,
)

__all__ = [
    # Extractor
    "FeatureExtractor",
    # Firing rates
    "compute_firing_rates",
    "bin_spikes",
    "gaussian_smooth_firing_rate",
    "causal_smooth_firing_rate",
    "spike_train_to_firing_rate_multichannel",
    "compute_instantaneous_firing_rate",
    # Spectral
    "FREQUENCY_BANDS",
    "compute_psd",
    "compute_band_power",
    "compute_spectrogram",
    "compute_band_power_timeseries",
    "extract_spectral_features",
]
