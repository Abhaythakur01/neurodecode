"""
Neural data preprocessing module.

Provides filtering, artifact removal, normalization, and spike detection
functionality for neural recordings.
"""

from src.preprocessing.artifacts import (
    detect_outliers,
    detect_saturation,
    interpolate_bad_segments,
    remove_outliers,
)
from src.preprocessing.filters import (
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    notch_filter,
    remove_line_noise,
)
from src.preprocessing.normalization import Normalizer, soft_normalize, zscore_normalize
from src.preprocessing.pipeline import (
    LFPPreprocessingPipeline,
    PreprocessingPipeline,
    SpikePreprocessingPipeline,
)
from src.preprocessing.spike_detection import (
    binary_to_spike_times,
    compute_spike_features,
    detect_spikes_multichannel,
    extract_waveforms,
    spike_times_to_binary,
    threshold_crossing,
)

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "LFPPreprocessingPipeline",
    "SpikePreprocessingPipeline",
    # Filters
    "bandpass_filter",
    "lowpass_filter",
    "highpass_filter",
    "notch_filter",
    "remove_line_noise",
    # Normalization
    "Normalizer",
    "zscore_normalize",
    "soft_normalize",
    # Artifacts
    "detect_outliers",
    "remove_outliers",
    "detect_saturation",
    "interpolate_bad_segments",
    # Spike detection
    "threshold_crossing",
    "detect_spikes_multichannel",
    "extract_waveforms",
    "compute_spike_features",
    "spike_times_to_binary",
    "binary_to_spike_times",
]
