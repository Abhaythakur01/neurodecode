"""
Unit tests for evaluation module.
"""

import time

import numpy as np
import pytest

from src.evaluation import (
    LatencyStats,
    LatencyTracker,
    compute_all_metrics,
    compute_metrics_per_dimension,
    correlation,
    cross_validate,
    mae,
    measure_latency,
    mse,
    r2_score,
    rmse,
    snr,
    temporal_split,
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions and ground truth."""
    np.random.seed(42)
    n_samples = 100
    n_outputs = 2

    y_true = np.random.randn(n_samples, n_outputs)
    # Predictions with some error
    y_pred = y_true + 0.1 * np.random.randn(n_samples, n_outputs)

    return y_true, y_pred


@pytest.fixture
def perfect_predictions():
    """Generate perfect predictions (no error)."""
    np.random.seed(42)
    y = np.random.randn(100, 2)
    return y, y.copy()


@pytest.fixture
def random_predictions():
    """Generate random predictions (no correlation)."""
    np.random.seed(42)
    y_true = np.random.randn(100, 2)
    y_pred = np.random.randn(100, 2)
    return y_true, y_pred


@pytest.mark.unit
class TestMetrics:
    """Tests for evaluation metrics."""

    def test_r2_score_good(self, sample_predictions):
        """Test R² score with good predictions."""
        y_true, y_pred = sample_predictions
        r2 = r2_score(y_true, y_pred)

        assert 0 < r2 <= 1
        assert r2 > 0.9  # Good predictions should have high R²

    def test_r2_score_perfect(self, perfect_predictions):
        """Test R² score with perfect predictions."""
        y_true, y_pred = perfect_predictions
        r2 = r2_score(y_true, y_pred)

        assert np.isclose(r2, 1.0)

    def test_r2_score_random(self, random_predictions):
        """Test R² score with random predictions."""
        y_true, y_pred = random_predictions
        r2 = r2_score(y_true, y_pred)

        # Random predictions should have R² near or below 0
        assert r2 < 0.5

    def test_r2_score_raw_values(self, sample_predictions):
        """Test R² score returning raw values per output."""
        y_true, y_pred = sample_predictions
        r2 = r2_score(y_true, y_pred, multioutput="raw_values")

        assert len(r2) == y_true.shape[1]

    def test_mse(self, sample_predictions):
        """Test MSE computation."""
        y_true, y_pred = sample_predictions
        error = mse(y_true, y_pred)

        assert error >= 0
        assert error < 0.1  # Good predictions have small MSE

    def test_mse_perfect(self, perfect_predictions):
        """Test MSE with perfect predictions."""
        y_true, y_pred = perfect_predictions
        error = mse(y_true, y_pred)

        assert np.isclose(error, 0.0)

    def test_rmse(self, sample_predictions):
        """Test RMSE computation."""
        y_true, y_pred = sample_predictions
        error = rmse(y_true, y_pred)

        assert error >= 0
        assert np.isclose(error, np.sqrt(mse(y_true, y_pred)))

    def test_mae(self, sample_predictions):
        """Test MAE computation."""
        y_true, y_pred = sample_predictions
        error = mae(y_true, y_pred)

        assert error >= 0

    def test_correlation(self, sample_predictions):
        """Test correlation computation."""
        y_true, y_pred = sample_predictions
        corr = correlation(y_true, y_pred)

        assert -1 <= corr <= 1
        assert corr > 0.9  # Good predictions have high correlation

    def test_correlation_perfect(self, perfect_predictions):
        """Test correlation with perfect predictions."""
        y_true, y_pred = perfect_predictions
        corr = correlation(y_true, y_pred)

        assert np.isclose(corr, 1.0)

    def test_snr(self, sample_predictions):
        """Test SNR computation."""
        y_true, y_pred = sample_predictions
        signal_to_noise = snr(y_true, y_pred)

        assert signal_to_noise > 0  # Good predictions have positive SNR in dB

    def test_compute_all_metrics(self, sample_predictions):
        """Test computing all metrics at once."""
        y_true, y_pred = sample_predictions
        metrics = compute_all_metrics(y_true, y_pred)

        assert "r2" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "correlation" in metrics
        assert "snr_db" in metrics

    def test_compute_metrics_per_dimension(self, sample_predictions):
        """Test computing metrics per dimension."""
        y_true, y_pred = sample_predictions
        metrics = compute_metrics_per_dimension(y_true, y_pred, dim_names=["x", "y"])

        assert "x" in metrics
        assert "y" in metrics
        assert "r2" in metrics["x"]


@pytest.mark.unit
class TestLatency:
    """Tests for latency measurement."""

    def test_latency_stats(self):
        """Test LatencyStats class."""
        stats = LatencyStats()
        stats.add(10.0)
        stats.add(20.0)
        stats.add(15.0)

        assert stats.count == 3
        assert stats.mean == 15.0
        assert stats.min == 10.0
        assert stats.max == 20.0

    def test_latency_stats_percentiles(self):
        """Test percentile calculations."""
        stats = LatencyStats()
        for i in range(100):
            stats.add(float(i))

        assert stats.median == 49.5
        assert stats.p95 > 90
        assert stats.p99 > 95

    def test_latency_stats_to_dict(self):
        """Test converting stats to dictionary."""
        stats = LatencyStats()
        stats.add(10.0)
        d = stats.to_dict()

        assert "mean_ms" in d
        assert "p95_ms" in d

    def test_latency_tracker(self):
        """Test LatencyTracker class."""
        tracker = LatencyTracker()

        tracker.start("component_a")
        time.sleep(0.01)
        elapsed = tracker.stop("component_a")

        assert elapsed > 0
        assert "component_a" in tracker.components

    def test_latency_tracker_context_manager(self):
        """Test tracker context manager."""
        tracker = LatencyTracker()

        with tracker.track("my_component"):
            time.sleep(0.01)

        stats = tracker.get_stats("my_component")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean > 0

    def test_measure_latency(self):
        """Test measure_latency function."""

        def slow_function():
            time.sleep(0.001)
            return 42

        stats = measure_latency(slow_function, n_iterations=10, warmup=2)

        assert stats.count == 10
        assert stats.mean > 0


@pytest.mark.unit
class TestCrossValidation:
    """Tests for cross-validation."""

    def test_temporal_split(self):
        """Test temporal cross-validation splits."""
        splits = list(temporal_split(100, n_splits=5))

        assert len(splits) <= 5
        for train_idx, test_idx in splits:
            # Train should come before test
            assert train_idx.max() < test_idx.min()
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_temporal_split_indices(self):
        """Test that temporal split preserves temporal order."""
        splits = list(temporal_split(100, n_splits=3))

        for train_idx, test_idx in splits:
            # All training indices should be less than all test indices
            assert np.all(train_idx < test_idx.min())

    def test_cross_validate(self, sample_firing_rates, sample_neural_data):
        """Test cross_validate function with a simple decoder."""
        from src.decoders.base import BaseDecoder

        class SimpleDecoder(BaseDecoder):
            def fit(self, X, y):
                self.n_features = X.shape[1]
                self.n_outputs = y.shape[1]
                self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
                self.is_fitted = True
                return self

            def predict(self, X):
                return X @ self._coef

        X = sample_firing_rates  # (100, 50)
        _, y = sample_neural_data  # Use kinematics (100, 2)

        results = cross_validate(SimpleDecoder, X, y, cv_method="temporal", n_splits=3)

        assert "r2_mean" in results
        assert "r2_std" in results
        assert "fold_metrics" in results
        assert len(results["fold_metrics"]) <= 3
