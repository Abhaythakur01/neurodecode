"""
Unit tests for feature extraction module.
"""

import numpy as np
import pytest

from src.features import (
    FeatureExtractor,
    bin_spikes,
    causal_smooth_firing_rate,
    compute_band_power,
    compute_firing_rates,
    compute_psd,
    gaussian_smooth_firing_rate,
)


@pytest.fixture
def sample_spike_times():
    """Generate sample spike times for multiple neurons."""
    np.random.seed(42)
    n_neurons = 5
    duration = 10.0  # seconds
    spike_times = []
    for _ in range(n_neurons):
        # Random spike times with ~20 Hz firing rate
        n_spikes = int(20 * duration)
        times = np.sort(np.random.uniform(0, duration, n_spikes))
        spike_times.append(times)
    return spike_times


@pytest.fixture
def sample_binary_spikes():
    """Generate sample binary spike trains."""
    np.random.seed(42)
    n_samples = 10000  # 10 seconds at 1kHz
    n_neurons = 10
    # Sparse binary spikes (~2% of samples have spikes)
    spikes = (np.random.rand(n_samples, n_neurons) < 0.02).astype(float)
    return spikes


@pytest.fixture
def sample_continuous_signal():
    """Generate sample LFP-like continuous signal."""
    np.random.seed(42)
    fs = 1000
    duration = 5
    n_channels = 4
    t = np.arange(0, duration, 1 / fs)

    # Create signal with different frequency components per channel
    signal = np.zeros((len(t), n_channels))
    for ch in range(n_channels):
        freq = 10 + ch * 5  # Different dominant frequency per channel
        signal[:, ch] = np.sin(2 * np.pi * freq * t) + 0.5 * np.random.randn(len(t))

    return signal, fs


@pytest.mark.unit
class TestFiringRates:
    """Tests for firing rate computation."""

    def test_compute_firing_rates_single_neuron(self, sample_spike_times):
        """Test firing rate computation for single neuron."""
        spike_times = sample_spike_times[0]
        rates = compute_firing_rates(spike_times, bin_size=0.1, duration=10.0)

        assert rates.shape == (100,)  # 10s / 0.1s = 100 bins
        assert np.mean(rates) > 0  # Should have non-zero firing

    def test_compute_firing_rates_multiple_neurons(self, sample_spike_times):
        """Test firing rate computation for multiple neurons."""
        rates = compute_firing_rates(sample_spike_times, bin_size=0.02, duration=10.0)

        n_bins = int(10.0 / 0.02)
        assert rates.shape == (n_bins, len(sample_spike_times))

    def test_bin_spikes(self, sample_binary_spikes):
        """Test spike binning."""
        single_neuron = sample_binary_spikes[:, 0]
        binned = bin_spikes(single_neuron, bin_size_samples=100)

        expected_bins = len(single_neuron) // 100
        assert len(binned) == expected_bins
        assert binned.sum() == single_neuron[: expected_bins * 100].sum()

    def test_gaussian_smooth_firing_rate(self):
        """Test Gaussian smoothing."""
        rates = np.zeros(100)
        rates[50] = 100  # Spike in middle

        smoothed = gaussian_smooth_firing_rate(rates, sigma_bins=3)

        assert smoothed.shape == rates.shape
        assert smoothed[50] < 100  # Peak should be reduced
        assert smoothed[45] > 0  # Smoothing spreads to neighbors

    def test_causal_smooth_firing_rate(self):
        """Test causal (exponential) smoothing."""
        rates = np.zeros(100)
        rates[50] = 100

        smoothed = causal_smooth_firing_rate(rates, tau_bins=5)

        assert smoothed.shape == rates.shape
        # Causal smoothing should only affect future samples
        assert smoothed[49] == 0  # Before spike
        assert smoothed[50] > 0  # At spike
        assert smoothed[55] > 0  # After spike (exponential decay)


@pytest.mark.unit
class TestSpectralFeatures:
    """Tests for spectral feature extraction."""

    def test_compute_psd(self, sample_continuous_signal):
        """Test power spectral density computation."""
        signal, fs = sample_continuous_signal
        freqs, psd = compute_psd(signal[:, 0], fs)

        assert len(freqs) == len(psd)
        assert freqs[0] >= 0
        assert np.all(psd >= 0)  # PSD should be non-negative

    def test_compute_band_power(self, sample_continuous_signal):
        """Test band power computation."""
        signal, fs = sample_continuous_signal
        band_powers = compute_band_power(signal[:, 0], fs)

        assert "alpha" in band_powers
        assert "beta" in band_powers
        assert "delta" in band_powers
        assert all(p >= 0 for p in band_powers.values())

    def test_compute_band_power_relative(self, sample_continuous_signal):
        """Test relative band power computation."""
        signal, fs = sample_continuous_signal
        band_powers = compute_band_power(signal[:, 0], fs, relative=True)

        # Each relative power should be between 0 and 1
        for band, power in band_powers.items():
            assert 0 <= power <= 1, f"Band {band} power {power} out of range"

        # Relative powers should sum to less than or equal to 1
        # (they won't sum to exactly 1 because predefined bands don't cover all frequencies)
        total = sum(band_powers.values())
        assert total <= 1.1

    def test_compute_band_power_multichannel(self, sample_continuous_signal):
        """Test band power for multi-channel data."""
        signal, fs = sample_continuous_signal
        band_powers = compute_band_power(signal, fs)

        assert band_powers["alpha"].shape == (signal.shape[1],)


@pytest.mark.unit
class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_extractor_init(self):
        """Test extractor initialization."""
        extractor = FeatureExtractor(bin_size=0.02, fs=1000)
        assert extractor.bin_size == 0.02
        assert extractor.fs == 1000
        assert not extractor.is_fitted

    def test_extractor_fit(self, sample_binary_spikes):
        """Test extractor fitting."""
        extractor = FeatureExtractor(bin_size=0.02, fs=1000)
        extractor.fit(sample_binary_spikes)

        assert extractor.is_fitted
        assert extractor._n_neurons == sample_binary_spikes.shape[1]

    def test_extractor_transform_continuous(self, sample_binary_spikes):
        """Test extracting features from continuous data."""
        extractor = FeatureExtractor(
            bin_size=0.1,  # 100ms bins
            fs=1000,
            include_firing_rates=True,
            include_spectral=False,
        )
        extractor.fit(sample_binary_spikes)
        features = extractor.transform(sample_binary_spikes)

        expected_bins = sample_binary_spikes.shape[0] // 100  # 100 samples per bin
        assert features.shape[0] == expected_bins

    def test_extractor_transform_binned(self, sample_neural_data):
        """Test extracting features from pre-binned data."""
        X, _ = sample_neural_data  # Shape: (100, 50, 20)
        extractor = FeatureExtractor(include_firing_rates=True, include_spectral=False)
        extractor.fit(X)
        features = extractor.transform(X)

        assert features.shape[0] == X.shape[0]

    def test_extractor_with_history(self, sample_binary_spikes):
        """Test extractor with history bins."""
        extractor = FeatureExtractor(
            bin_size=0.1,
            fs=1000,
            history_bins=3,
            include_firing_rates=True,
            include_spectral=False,
        )
        extractor.fit(sample_binary_spikes)
        features = extractor.transform(sample_binary_spikes)

        # Features should include history
        n_neurons = sample_binary_spikes.shape[1]
        assert features.shape[1] == n_neurons * 3

    def test_extractor_get_params(self):
        """Test get_params method."""
        extractor = FeatureExtractor(bin_size=0.05, lag_bins=2)
        params = extractor.get_params()

        assert params["bin_size"] == 0.05
        assert params["lag_bins"] == 2

    def test_extractor_feature_names(self, sample_binary_spikes):
        """Test feature name generation."""
        extractor = FeatureExtractor(
            include_firing_rates=True,
            include_spectral=False,
            history_bins=2,
        )
        extractor.fit(sample_binary_spikes)
        names = extractor.get_feature_names()

        assert len(names) > 0
        assert any("firing_rate" in name for name in names)
