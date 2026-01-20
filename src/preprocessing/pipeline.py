"""
Neural data preprocessing pipeline.

Provides a unified interface for applying preprocessing steps
to neural data in the correct order.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.artifacts import detect_outliers, interpolate_bad_segments, remove_outliers
from src.preprocessing.filters import (
    bandpass_filter,
    lowpass_filter,
    notch_filter,
    remove_line_noise,
)
from src.preprocessing.normalization import Normalizer, zscore_normalize
from src.preprocessing.spike_detection import detect_spikes_multichannel


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for neural data.

    Applies filtering, artifact removal, and normalization in sequence.
    Parameters are stored for consistent application to new data.
    """

    def __init__(
        self,
        fs: float = 30000.0,
        bandpass: Optional[Tuple[float, float]] = (300.0, 6000.0),
        notch_freq: Optional[float] = 60.0,
        remove_artifacts: bool = True,
        artifact_threshold: float = 5.0,
        normalize: bool = True,
        normalize_method: str = "zscore",
    ):
        """
        Initialize preprocessing pipeline.

        Args:
            fs: Sampling frequency in Hz.
            bandpass: Tuple of (low_freq, high_freq) for bandpass filter.
                Set to None to skip bandpass filtering.
            notch_freq: Frequency for notch filter (e.g., 60 Hz line noise).
                Set to None to skip notch filtering.
            remove_artifacts: Whether to detect and remove artifacts.
            artifact_threshold: Threshold (in std) for artifact detection.
            normalize: Whether to normalize the data.
            normalize_method: Normalization method ('zscore', 'minmax', 'robust').
        """
        self.fs = fs
        self.bandpass = bandpass
        self.notch_freq = notch_freq
        self.remove_artifacts = remove_artifacts
        self.artifact_threshold = artifact_threshold
        self.normalize = normalize
        self.normalize_method = normalize_method

        self._normalizer: Optional[Normalizer] = None
        self.is_fitted = False
        self._artifact_mask: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "PreprocessingPipeline":
        """
        Fit pipeline parameters on training data.

        Currently only fits the normalizer. Filtering parameters are fixed.

        Args:
            data: Training data of shape (n_samples, n_channels).

        Returns:
            self: Fitted pipeline.
        """
        # Apply filtering before fitting normalizer
        processed = self._apply_filters(data)

        if self.remove_artifacts:
            processed, _ = remove_outliers(
                processed,
                threshold=self.artifact_threshold,
                replacement="interpolate",
            )

        if self.normalize:
            self._normalizer = Normalizer(method=self.normalize_method)
            self._normalizer.fit(processed)

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to data.

        Args:
            data: Data of shape (n_samples, n_channels).

        Returns:
            Preprocessed data.
        """
        # Apply filters
        processed = self._apply_filters(data)

        # Remove artifacts
        if self.remove_artifacts:
            processed, self._artifact_mask = remove_outliers(
                processed,
                threshold=self.artifact_threshold,
                replacement="interpolate",
            )

        # Normalize
        if self.normalize and self._normalizer is not None:
            processed = self._normalizer.transform(processed)

        return processed

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit pipeline and transform data in one step.

        Args:
            data: Data of shape (n_samples, n_channels).

        Returns:
            Preprocessed data.
        """
        self.fit(data)
        return self.transform(data)

    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply configured filters to data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        processed = data.copy().astype(float)

        # Bandpass filter
        if self.bandpass is not None:
            low, high = self.bandpass
            processed = bandpass_filter(processed, low, high, self.fs, axis=0)

        # Notch filter for line noise
        if self.notch_freq is not None:
            processed = remove_line_noise(processed, self.fs, self.notch_freq, axis=0)

        return processed

    def get_artifact_mask(self) -> Optional[np.ndarray]:
        """Get mask of detected artifacts from last transform."""
        return self._artifact_mask

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {
            "fs": self.fs,
            "bandpass": self.bandpass,
            "notch_freq": self.notch_freq,
            "remove_artifacts": self.remove_artifacts,
            "artifact_threshold": self.artifact_threshold,
            "normalize": self.normalize,
            "normalize_method": self.normalize_method,
            "is_fitted": self.is_fitted,
        }


class LFPPreprocessingPipeline(PreprocessingPipeline):
    """
    Preprocessing pipeline optimized for Local Field Potentials (LFP).

    Uses appropriate frequency bands for LFP analysis.
    """

    def __init__(
        self,
        fs: float = 1000.0,
        bandpass: Optional[Tuple[float, float]] = (0.5, 300.0),
        notch_freq: Optional[float] = 60.0,
        **kwargs,
    ):
        super().__init__(fs=fs, bandpass=bandpass, notch_freq=notch_freq, **kwargs)


class SpikePreprocessingPipeline(PreprocessingPipeline):
    """
    Preprocessing pipeline optimized for spike detection.

    Uses appropriate frequency bands for spike analysis and includes
    spike detection functionality.
    """

    def __init__(
        self,
        fs: float = 30000.0,
        bandpass: Optional[Tuple[float, float]] = (300.0, 6000.0),
        spike_threshold_std: float = 4.0,
        **kwargs,
    ):
        super().__init__(fs=fs, bandpass=bandpass, **kwargs)
        self.spike_threshold_std = spike_threshold_std
        self._spike_times: Optional[List[np.ndarray]] = None

    def transform_with_spikes(self, data: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Transform data and detect spikes.

        Args:
            data: Data of shape (n_samples, n_channels).

        Returns:
            Tuple of (preprocessed_data, list_of_spike_times_per_channel).
        """
        processed = self.transform(data)

        # Detect spikes on filtered but not normalized data
        filtered = self._apply_filters(data)
        self._spike_times = detect_spikes_multichannel(
            filtered,
            self.fs,
            threshold_std=self.spike_threshold_std,
        )

        return processed, self._spike_times

    def get_spike_times(self) -> Optional[List[np.ndarray]]:
        """Get detected spike times from last transform."""
        return self._spike_times
