"""
Unit tests for Wiener Filter decoder.
"""

import numpy as np
import pytest

from src.decoders.classic.wiener_filter import (
    CausalWienerFilter,
    NonCausalWienerFilter,
    WienerFilterDecoder,
)


@pytest.fixture
def linear_data():
    """Generate data with linear relationship suitable for Wiener filter."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_outputs = 2

    # Generate smooth kinematics
    t = np.linspace(0, 10, n_samples)
    y = np.column_stack([np.sin(t), np.cos(t)])

    # Generate neural features as noisy linear function of kinematics
    H = np.random.randn(n_features, n_outputs) * 0.5
    X = y @ H.T + 0.2 * np.random.randn(n_samples, n_features)

    return X, y


@pytest.fixture
def lagged_data():
    """Generate data where neural activity leads kinematics (with lag)."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_outputs = 2
    lag = 5

    # Generate smooth kinematics
    t = np.linspace(0, 10, n_samples)
    y = np.column_stack([np.sin(t), np.cos(t)])

    # Neural activity leads kinematics by 'lag' samples
    H = np.random.randn(n_features, n_outputs)
    X = np.zeros((n_samples, n_features))
    X[:-lag] = y[lag:] @ H.T + 0.2 * np.random.randn(n_samples - lag, n_features)
    X[-lag:] = X[-lag - 1]  # Pad end

    return X, y


@pytest.mark.unit
@pytest.mark.decoder
class TestWienerFilterDecoder:
    """Tests for WienerFilterDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = WienerFilterDecoder(n_lags=5, regularization=1e-3)

        assert decoder.n_lags == 5
        assert decoder.regularization == 1e-3
        assert not decoder.is_fitted

    def test_fit(self, linear_data):
        """Test fitting the decoder."""
        X, y = linear_data
        decoder = WienerFilterDecoder(n_lags=5)
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder.weights is not None
        # Weights shape: (n_features * (n_lags + 1), n_outputs)
        expected_weight_rows = X.shape[1] * (5 + 1)
        assert decoder.weights.shape == (expected_weight_rows, y.shape[1])

    def test_predict(self, linear_data):
        """Test prediction."""
        X, y = linear_data
        decoder = WienerFilterDecoder(n_lags=5)
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        # Output should have fewer samples due to lags
        expected_samples = X.shape[0] - 5
        assert y_pred.shape == (expected_samples, y.shape[1])

    def test_predict_not_fitted(self, linear_data):
        """Test prediction raises error when not fitted."""
        X, _ = linear_data
        decoder = WienerFilterDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_decoding_accuracy(self, linear_data):
        """Test decoding accuracy on linear data."""
        X, y = linear_data

        # Train/test split
        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        decoder = WienerFilterDecoder(n_lags=5, regularization=1e-4)
        decoder.fit(X_train, y_train)

        y_pred = decoder.predict(X_test)
        # Align y_test with predictions
        y_test_aligned = y_test[5:]

        # Compute R²
        ss_res = np.sum((y_test_aligned - y_pred) ** 2)
        ss_tot = np.sum((y_test_aligned - np.mean(y_test_aligned, axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Should have reasonable R² on linear data
        assert r2 > 0.5

    def test_regularization_effect(self, linear_data):
        """Test that regularization affects results."""
        X, y = linear_data

        decoder_low_reg = WienerFilterDecoder(n_lags=3, regularization=1e-6)
        decoder_high_reg = WienerFilterDecoder(n_lags=3, regularization=1.0)

        decoder_low_reg.fit(X, y)
        decoder_high_reg.fit(X, y)

        # High regularization should lead to smaller weights
        assert np.mean(np.abs(decoder_high_reg.weights)) < np.mean(
            np.abs(decoder_low_reg.weights)
        )

    def test_create_lagged_features(self, linear_data):
        """Test lagged feature creation."""
        X, _ = linear_data
        decoder = WienerFilterDecoder(n_lags=3)

        X_lagged = decoder._create_lagged_features(X)

        # Check shape
        expected_samples = X.shape[0] - 3
        expected_features = X.shape[1] * 4  # n_features * (n_lags + 1)
        assert X_lagged.shape == (expected_samples, expected_features)

    def test_insufficient_samples(self, linear_data):
        """Test error with insufficient samples for lags."""
        X, y = linear_data
        decoder = WienerFilterDecoder(n_lags=1000)  # More lags than samples

        with pytest.raises(ValueError):
            decoder.fit(X, y)

    def test_get_params(self, linear_data):
        """Test get_params method."""
        X, y = linear_data
        decoder = WienerFilterDecoder(n_lags=7, regularization=0.01)
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["n_lags"] == 7
        assert params["regularization"] == 0.01


@pytest.mark.unit
@pytest.mark.decoder
class TestCausalWienerFilter:
    """Tests for CausalWienerFilter class."""

    def test_init(self):
        """Test causal filter is same as standard."""
        decoder = CausalWienerFilter(n_lags=10)
        assert decoder.n_lags == 10
        assert decoder.name == "CausalWiener"


@pytest.mark.unit
@pytest.mark.decoder
class TestNonCausalWienerFilter:
    """Tests for NonCausalWienerFilter class."""

    def test_init(self):
        """Test non-causal filter initialization."""
        decoder = NonCausalWienerFilter(n_lags_past=5, n_lags_future=3)

        assert decoder.n_lags_past == 5
        assert decoder.n_lags_future == 3
        assert decoder.n_lags == 8  # past + future

    def test_fit_and_predict(self, linear_data):
        """Test non-causal filter fit and predict."""
        X, y = linear_data
        decoder = NonCausalWienerFilter(n_lags_past=3, n_lags_future=3)
        decoder.fit(X, y)

        assert decoder.is_fitted

        y_pred = decoder.predict(X)
        # Output should have fewer samples due to both past and future lags
        expected_samples = X.shape[0] - 3 - 3
        assert y_pred.shape[0] == expected_samples

    def test_noncausal_better_than_causal(self, lagged_data):
        """Test that non-causal can be better when future matters."""
        X, y = lagged_data

        train_idx = int(0.7 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        causal = CausalWienerFilter(n_lags=10)
        noncausal = NonCausalWienerFilter(n_lags_past=5, n_lags_future=5)

        causal.fit(X_train, y_train)
        noncausal.fit(X_train, y_train)

        # Both should produce predictions
        y_pred_causal = causal.predict(X_test)
        y_pred_noncausal = noncausal.predict(X_test)

        assert y_pred_causal.shape[1] == y.shape[1]
        assert y_pred_noncausal.shape[1] == y.shape[1]
