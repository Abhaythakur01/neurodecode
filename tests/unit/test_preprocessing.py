"""
Unit tests for preprocessing module.
"""

import numpy as np
import pytest

from src.preprocessing import (
    Normalizer,
    PreprocessingPipeline,
    bandpass_filter,
    detect_outliers,
    highpass_filter,
    lowpass_filter,
    notch_filter,
    remove_outliers,
    threshold_crossing,
    zscore_normalize,
)


@pytest.fixture
def sample_signal():
    """Generate sample continuous signal."""
    np.random.seed(42)
    fs = 1000  # 1 kHz
    t = np.arange(0, 1, 1 / fs)
    # Signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 10 * t)  # 10 Hz
        + 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz
        + 0.3 * np.sin(2 * np.pi * 200 * t)  # 200 Hz
        + 0.1 * np.random.randn(len(t))  # Noise
    )
    return signal, fs


@pytest.fixture
def multichannel_signal():
    """Generate multi-channel signal."""
    np.random.seed(42)
    n_samples, n_channels = 1000, 10
    return np.random.randn(n_samples, n_channels)


@pytest.mark.unit
class TestFilters:
    """Tests for filter functions."""

    def test_bandpass_filter(self, sample_signal):
        """Test bandpass filter."""
        signal, fs = sample_signal
        filtered = bandpass_filter(signal, 5, 100, fs)

        assert filtered.shape == signal.shape
        assert not np.allclose(filtered, signal)

    def test_lowpass_filter(self, sample_signal):
        """Test lowpass filter."""
        signal, fs = sample_signal
        filtered = lowpass_filter(signal, 50, fs)

        assert filtered.shape == signal.shape

    def test_highpass_filter(self, sample_signal):
        """Test highpass filter."""
        signal, fs = sample_signal
        filtered = highpass_filter(signal, 20, fs)

        assert filtered.shape == signal.shape

    def test_notch_filter(self, sample_signal):
        """Test notch filter."""
        signal, fs = sample_signal
        filtered = notch_filter(signal, 50, fs)

        assert filtered.shape == signal.shape

    def test_bandpass_invalid_freqs(self, sample_signal):
        """Test bandpass with invalid frequency range."""
        signal, fs = sample_signal
        with pytest.raises(ValueError):
            bandpass_filter(signal, 100, 50, fs)  # low > high

    def test_filter_multichannel(self, multichannel_signal):
        """Test filtering multi-channel data."""
        filtered = bandpass_filter(multichannel_signal, 1, 100, 1000, axis=0)
        assert filtered.shape == multichannel_signal.shape


@pytest.mark.unit
class TestNormalization:
    """Tests for normalization functions."""

    def test_zscore_normalize(self, multichannel_signal):
        """Test z-score normalization."""
        normalized, mean, std = zscore_normalize(multichannel_signal)

        assert normalized.shape == multichannel_signal.shape
        # Check normalized data has zero mean and unit variance
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)

    def test_normalizer_zscore(self, multichannel_signal):
        """Test Normalizer class with zscore."""
        normalizer = Normalizer(method="zscore")
        normalizer.fit(multichannel_signal)
        normalized = normalizer.transform(multichannel_signal)

        assert normalizer.is_fitted
        assert normalized.shape == multichannel_signal.shape

    def test_normalizer_minmax(self, multichannel_signal):
        """Test Normalizer class with minmax."""
        normalizer = Normalizer(method="minmax")
        normalized = normalizer.fit_transform(multichannel_signal)

        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1

    def test_normalizer_robust(self, multichannel_signal):
        """Test Normalizer class with robust scaling."""
        normalizer = Normalizer(method="robust")
        normalized = normalizer.fit_transform(multichannel_signal)

        assert normalized.shape == multichannel_signal.shape

    def test_normalizer_inverse_transform(self, multichannel_signal):
        """Test inverse transform recovers original data."""
        normalizer = Normalizer(method="zscore")
        normalizer.fit(multichannel_signal)
        normalized = normalizer.transform(multichannel_signal)
        recovered = normalizer.inverse_transform(normalized)

        assert np.allclose(recovered, multichannel_signal)

    def test_normalizer_invalid_method(self):
        """Test invalid normalization method raises error."""
        with pytest.raises(ValueError):
            Normalizer(method="invalid")

    def test_normalizer_transform_not_fitted(self, multichannel_signal):
        """Test transform raises error when not fitted."""
        normalizer = Normalizer()
        with pytest.raises(RuntimeError):
            normalizer.transform(multichannel_signal)


@pytest.mark.unit
class TestArtifacts:
    """Tests for artifact detection and removal."""

    def test_detect_outliers_zscore(self, multichannel_signal):
        """Test outlier detection with zscore method."""
        # Add outliers
        data = multichannel_signal.copy()
        data[10, 0] = 100  # Clear outlier

        outliers = detect_outliers(data, threshold=3.0, method="zscore")

        assert outliers.dtype == bool
        assert outliers[10]  # Should detect the outlier

    def test_detect_outliers_mad(self, multichannel_signal):
        """Test outlier detection with MAD method."""
        data = multichannel_signal.copy()
        data[10, 0] = 100

        outliers = detect_outliers(data, threshold=3.0, method="mad")
        assert outliers[10]

    def test_detect_outliers_iqr(self, multichannel_signal):
        """Test outlier detection with IQR method."""
        data = multichannel_signal.copy()
        data[10, 0] = 100

        outliers = detect_outliers(data, threshold=1.5, method="iqr")
        assert outliers[10]

    def test_remove_outliers_interpolate(self, multichannel_signal):
        """Test outlier removal with interpolation."""
        data = multichannel_signal.copy()
        data[50, 0] = 100

        cleaned, mask = remove_outliers(data, replacement="interpolate")

        assert cleaned.shape == data.shape
        assert mask[50]
        assert cleaned[50, 0] != 100

    def test_remove_outliers_clip(self, multichannel_signal):
        """Test outlier removal with clipping."""
        data = multichannel_signal.copy()
        data[50, 0] = 100

        cleaned, mask = remove_outliers(data, replacement="clip")

        assert cleaned[50, 0] < 100


@pytest.mark.unit
class TestSpikeDetection:
    """Tests for spike detection functions."""

    def test_threshold_crossing(self):
        """Test threshold crossing detection."""
        # Create signal with clear spikes
        signal = np.zeros(1000)
        spike_positions = [100, 300, 500, 700]
        for pos in spike_positions:
            signal[pos] = -5  # Negative spike

        spikes = threshold_crossing(signal, threshold=-2, fs=1000, direction="negative")

        assert len(spikes) == len(spike_positions)

    def test_threshold_crossing_refractory(self):
        """Test refractory period enforcement."""
        signal = np.zeros(1000)
        # Spikes too close together
        signal[100] = -5
        signal[101] = -5  # Within refractory period

        spikes = threshold_crossing(signal, threshold=-2, fs=1000, refractory_ms=2)

        assert len(spikes) == 1  # Only first spike detected


@pytest.mark.unit
class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline."""

    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = PreprocessingPipeline(fs=30000, bandpass=(300, 6000))
        assert pipeline.fs == 30000
        assert pipeline.bandpass == (300, 6000)
        assert not pipeline.is_fitted

    def test_pipeline_fit_transform(self, multichannel_signal):
        """Test pipeline fit_transform."""
        pipeline = PreprocessingPipeline(
            fs=1000,
            bandpass=(1, 100),
            normalize=True,
        )
        processed = pipeline.fit_transform(multichannel_signal)

        assert pipeline.is_fitted
        assert processed.shape == multichannel_signal.shape

    def test_pipeline_transform_not_fitted(self, multichannel_signal):
        """Test transform works even when normalizer not fitted."""
        pipeline = PreprocessingPipeline(fs=1000, normalize=False)
        # Should work without fitting when normalize=False
        processed = pipeline.transform(multichannel_signal)
        assert processed.shape == multichannel_signal.shape

    def test_pipeline_get_params(self):
        """Test get_params method."""
        pipeline = PreprocessingPipeline(fs=30000)
        params = pipeline.get_params()

        assert params["fs"] == 30000
        assert "bandpass" in params
        assert "normalize" in params
