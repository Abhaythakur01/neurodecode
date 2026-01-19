"""
Feature extraction pipeline for neural decoding.

Provides a unified interface for extracting features from neural data,
including firing rates, spectral features, and temporal features.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.features.firing_rates import (
    causal_smooth_firing_rate,
    compute_firing_rates,
    gaussian_smooth_firing_rate,
    spike_train_to_firing_rate_multichannel,
)
from src.features.spectral import compute_band_power, extract_spectral_features


class FeatureExtractor:
    """
    Feature extractor for neural decoding.

    Extracts and combines multiple feature types from neural data
    into a format suitable for decoder input.
    """

    def __init__(
        self,
        bin_size: float = 0.02,
        fs: float = 1000.0,
        include_firing_rates: bool = True,
        include_spectral: bool = False,
        smooth_firing_rates: bool = True,
        smooth_sigma: float = 2.0,
        causal_smoothing: bool = False,
        lag_bins: int = 0,
        history_bins: int = 1,
    ):
        """
        Initialize feature extractor.

        Args:
            bin_size: Time bin size in seconds (default 20ms).
            fs: Sampling frequency in Hz.
            include_firing_rates: Whether to include firing rate features.
            include_spectral: Whether to include spectral features.
            smooth_firing_rates: Whether to smooth firing rates.
            smooth_sigma: Gaussian smoothing width in bins.
            causal_smoothing: Use causal (exponential) smoothing for real-time.
            lag_bins: Number of bins to lag features (for accounting for neural delay).
            history_bins: Number of history bins to include as features.
        """
        self.bin_size = bin_size
        self.fs = fs
        self.include_firing_rates = include_firing_rates
        self.include_spectral = include_spectral
        self.smooth_firing_rates = smooth_firing_rates
        self.smooth_sigma = smooth_sigma
        self.causal_smoothing = causal_smoothing
        self.lag_bins = lag_bins
        self.history_bins = history_bins

        self._n_features: Optional[int] = None
        self._n_neurons: Optional[int] = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "FeatureExtractor":
        """
        Fit extractor parameters (determine feature dimensions).

        Args:
            data: Neural data of shape (n_samples, n_neurons) or
                (n_samples, n_neurons, n_timebins).

        Returns:
            self: Fitted extractor.
        """
        # Handle 3D data (samples, neurons, timebins)
        if data.ndim == 3:
            self._n_neurons = data.shape[1]
        elif data.ndim == 2:
            self._n_neurons = data.shape[1]
        else:
            self._n_neurons = 1

        # Compute number of features
        n_features = 0

        if self.include_firing_rates:
            n_features += self._n_neurons * self.history_bins

        if self.include_spectral:
            # 6 standard frequency bands per channel
            n_features += self._n_neurons * 6

        self._n_features = n_features
        self.is_fitted = True

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from neural data.

        Args:
            data: Neural data of shape (n_samples, n_neurons) or
                (n_samples, n_neurons, n_timebins) for already-binned data.

        Returns:
            Feature array of shape (n_samples, n_features).
        """
        if data.ndim == 3:
            # Data is already binned: (n_samples, n_neurons, n_timebins)
            # Average over timebins or use last bin
            features = self._extract_from_binned(data)
        elif data.ndim == 2:
            # Continuous data: (n_samples, n_neurons)
            features = self._extract_from_continuous(data)
        else:
            raise ValueError(f"Expected 2D or 3D data, got shape {data.shape}")

        return features

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit extractor and transform data."""
        return self.fit(data).transform(data)

    def _extract_from_binned(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from pre-binned data.

        Args:
            data: Shape (n_samples, n_neurons, n_timebins).

        Returns:
            Features of shape (n_samples, n_features).
        """
        n_samples, n_neurons, n_timebins = data.shape

        features_list = []

        if self.include_firing_rates:
            # Use firing rates (mean over timebins or specific bins)
            if self.history_bins == 1:
                # Just use mean firing rate
                firing_rates = np.mean(data, axis=2)
            else:
                # Use last history_bins
                bins_to_use = min(self.history_bins, n_timebins)
                firing_rates = data[:, :, -bins_to_use:].reshape(n_samples, -1)

            if self.smooth_firing_rates:
                if self.causal_smoothing:
                    firing_rates = causal_smooth_firing_rate(
                        firing_rates, self.smooth_sigma, axis=0
                    )
                else:
                    firing_rates = gaussian_smooth_firing_rate(
                        firing_rates, self.smooth_sigma, axis=0
                    )

            features_list.append(firing_rates)

        if self.include_spectral:
            # Compute spectral features per sample
            spectral = np.zeros((n_samples, n_neurons * 6))
            for i in range(n_samples):
                spectral[i] = extract_spectral_features(
                    data[i].T,  # Shape (n_timebins, n_neurons)
                    fs=1.0 / self.bin_size,
                )
            features_list.append(spectral)

        features = np.hstack(features_list) if len(features_list) > 1 else features_list[0]

        # Apply lag
        if self.lag_bins > 0 and features.shape[0] > self.lag_bins:
            features = features[: -self.lag_bins]

        return features

    def _extract_from_continuous(self, data: np.ndarray) -> np.ndarray:
        """
        Extract features from continuous data.

        Args:
            data: Shape (n_samples, n_neurons).

        Returns:
            Features of shape (n_bins, n_features).
        """
        bin_samples = int(self.bin_size * self.fs)
        n_samples, n_neurons = data.shape
        n_bins = n_samples // bin_samples

        features_list = []

        if self.include_firing_rates:
            # Bin the data
            firing_rates = spike_train_to_firing_rate_multichannel(
                data,
                bin_samples,
                smooth=self.smooth_firing_rates,
                smooth_sigma=self.smooth_sigma,
            )

            # Add history if needed
            if self.history_bins > 1:
                firing_rates = self._add_history(firing_rates, self.history_bins)

            features_list.append(firing_rates)

        if self.include_spectral:
            # Compute spectral features per bin
            spectral = []
            for i in range(n_bins):
                start = i * bin_samples
                end = start + bin_samples
                spec_feat = extract_spectral_features(data[start:end], self.fs)
                spectral.append(spec_feat)
            spectral = np.array(spectral)

            # Trim to match firing rates if history was added
            if self.history_bins > 1 and self.include_firing_rates:
                spectral = spectral[self.history_bins - 1 :]

            features_list.append(spectral)

        features = np.hstack(features_list) if len(features_list) > 1 else features_list[0]

        # Apply lag
        if self.lag_bins > 0 and features.shape[0] > self.lag_bins:
            features = features[: -self.lag_bins]

        return features

    def _add_history(self, data: np.ndarray, history_bins: int) -> np.ndarray:
        """Add history bins as additional features."""
        n_bins, n_features = data.shape
        n_output = n_bins - history_bins + 1

        output = np.zeros((n_output, n_features * history_bins))

        for i in range(n_output):
            for h in range(history_bins):
                start_col = h * n_features
                end_col = (h + 1) * n_features
                output[i, start_col:end_col] = data[i + history_bins - 1 - h]

        return output

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        names = []

        if self.include_firing_rates:
            for h in range(self.history_bins):
                lag_str = f"_lag{h}" if h > 0 else ""
                for n in range(self._n_neurons or 0):
                    names.append(f"firing_rate_n{n}{lag_str}")

        if self.include_spectral:
            bands = ["alpha", "beta", "delta", "gamma", "high_gamma", "theta"]
            for band in bands:
                for n in range(self._n_neurons or 0):
                    names.append(f"{band}_power_n{n}")

        return names

    def get_params(self) -> Dict[str, Any]:
        """Get extractor parameters."""
        return {
            "bin_size": self.bin_size,
            "fs": self.fs,
            "include_firing_rates": self.include_firing_rates,
            "include_spectral": self.include_spectral,
            "smooth_firing_rates": self.smooth_firing_rates,
            "smooth_sigma": self.smooth_sigma,
            "causal_smoothing": self.causal_smoothing,
            "lag_bins": self.lag_bins,
            "history_bins": self.history_bins,
            "n_features": self._n_features,
            "is_fitted": self.is_fitted,
        }
