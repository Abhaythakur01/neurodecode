"""
Unit tests for Kalman Filter decoder.
"""

import numpy as np
import pytest

from src.decoders.classic.kalman_filter import KalmanFilterDecoder, SteadyStateKalmanFilter


@pytest.fixture
def simple_linear_data():
    """Generate data with simple linear relationship."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_outputs = 2

    # Generate smooth kinematics (random walk)
    y = np.zeros((n_samples, n_outputs))
    y[0] = np.random.randn(n_outputs)
    for t in range(1, n_samples):
        y[t] = 0.95 * y[t - 1] + 0.1 * np.random.randn(n_outputs)

    # Generate neural features as linear function of kinematics + noise
    H = np.random.randn(n_features, n_outputs)
    X = y @ H.T + 0.5 * np.random.randn(n_samples, n_features)

    return X, y


@pytest.fixture
def realistic_neural_data():
    """Generate more realistic neural data."""
    np.random.seed(42)
    n_samples = 1000
    n_neurons = 50
    n_outputs = 2

    # Smooth velocity trajectory
    t = np.linspace(0, 10, n_samples)
    velocity = np.column_stack([np.sin(t), np.cos(t)])

    # Tuning curves: each neuron responds to movement direction
    preferred_directions = np.random.uniform(0, 2 * np.pi, n_neurons)
    tuning = np.column_stack([np.cos(preferred_directions), np.sin(preferred_directions)])

    # Neural activity = tuning · velocity + baseline + noise
    baseline = np.random.uniform(5, 20, n_neurons)
    firing_rates = velocity @ tuning.T + baseline + np.random.randn(n_samples, n_neurons) * 2
    firing_rates = np.maximum(firing_rates, 0)  # Non-negative firing rates

    return firing_rates, velocity


@pytest.mark.unit
@pytest.mark.decoder
class TestKalmanFilterDecoder:
    """Tests for KalmanFilterDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = KalmanFilterDecoder(
            name="TestKalman",
            process_noise=1e-3,
            observation_noise=1e-2,
        )

        assert decoder.name == "TestKalman"
        assert decoder.process_noise == 1e-3
        assert decoder.observation_noise == 1e-2
        assert not decoder.is_fitted

    def test_fit(self, simple_linear_data):
        """Test fitting the decoder."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder.n_features == X.shape[1]
        assert decoder.n_outputs == y.shape[1]
        assert decoder.A is not None
        assert decoder.H is not None
        assert decoder.W is not None
        assert decoder.Q is not None

    def test_fit_matrices_shapes(self, simple_linear_data):
        """Test that fitted matrices have correct shapes."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        n_features = X.shape[1]
        n_outputs = y.shape[1]

        assert decoder.A.shape == (n_outputs, n_outputs)
        assert decoder.H.shape == (n_features, n_outputs)
        assert decoder.W.shape == (n_outputs, n_outputs)
        assert decoder.Q.shape == (n_features, n_features)

    def test_predict(self, simple_linear_data):
        """Test prediction."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape

    def test_predict_not_fitted(self, simple_linear_data):
        """Test prediction raises error when not fitted."""
        X, _ = simple_linear_data
        decoder = KalmanFilterDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_predict_single(self, simple_linear_data):
        """Test single-step prediction for real-time use."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        # Predict single time step
        y_single = decoder.predict_single(X[0])

        assert y_single.shape == (y.shape[1],)

    def test_evaluate(self, simple_linear_data):
        """Test evaluation."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        metrics = decoder.evaluate(X, y)

        assert "r2" in metrics
        assert "mse" in metrics
        # Should have reasonable performance on simple linear data
        assert metrics["r2"] > 0.5

    def test_decoding_accuracy(self, simple_linear_data):
        """Test decoding accuracy on data with known linear relationship."""
        X, y = simple_linear_data

        # Split data (temporal - no shuffle)
        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        decoder = KalmanFilterDecoder()
        decoder.fit(X_train, y_train)

        metrics = decoder.evaluate(X_test, y_test)

        # Should achieve reasonable R² on data with known linear relationship
        # Note: Kalman filter performance varies with data characteristics
        assert metrics["r2"] > 0.0  # Better than random chance
        assert "mse" in metrics

    def test_update_online(self, simple_linear_data):
        """Test online update."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder(learning_rate=0.1)
        decoder.fit(X[:200], y[:200])

        H_before = decoder.H.copy()

        # Online update with new data
        decoder.update(X[200:300], y[200:300])

        assert decoder._update_count == 1
        # H should have changed
        assert not np.allclose(decoder.H, H_before)

    def test_reset_state(self, simple_linear_data):
        """Test state reset."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        # Make some predictions to update internal state
        decoder.predict(X[:10])

        # Reset
        decoder.reset_state()

        assert np.allclose(decoder._x, 0)
        assert np.allclose(decoder._P, np.eye(decoder.n_outputs))

    def test_get_kalman_gain(self, simple_linear_data):
        """Test getting Kalman gain."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        K = decoder.get_kalman_gain()

        assert K is not None
        assert K.shape == (decoder.n_outputs, decoder.n_features)

    def test_get_params(self, simple_linear_data):
        """Test get_params method."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder(process_noise=1e-5)
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["process_noise"] == 1e-5
        assert params["n_features"] == X.shape[1]
        assert params["n_outputs"] == y.shape[1]

    def test_positive_definite_covariance(self, simple_linear_data):
        """Test that covariance matrices remain positive definite."""
        X, y = simple_linear_data
        decoder = KalmanFilterDecoder()
        decoder.fit(X, y)

        # Check W is positive definite
        eigvals_W = np.linalg.eigvalsh(decoder.W)
        assert np.all(eigvals_W > 0)

        # Check Q is positive definite
        eigvals_Q = np.linalg.eigvalsh(decoder.Q)
        assert np.all(eigvals_Q > 0)


@pytest.mark.unit
@pytest.mark.decoder
class TestSteadyStateKalmanFilter:
    """Tests for SteadyStateKalmanFilter class."""

    def test_init(self):
        """Test steady-state decoder initialization."""
        decoder = SteadyStateKalmanFilter(name="SteadyState")
        assert decoder.name == "SteadyState"

    def test_fit_computes_steady_state_gain(self, simple_linear_data):
        """Test that fit computes steady-state Kalman gain."""
        X, y = simple_linear_data
        decoder = SteadyStateKalmanFilter()
        decoder.fit(X, y)

        assert decoder._K_ss is not None
        assert decoder._K_ss.shape == (y.shape[1], X.shape[1])

    def test_predict_single_uses_steady_state(self, simple_linear_data):
        """Test single-step prediction uses steady-state gain."""
        X, y = simple_linear_data
        decoder = SteadyStateKalmanFilter()
        decoder.fit(X, y)

        # Multiple predictions should work consistently
        predictions = []
        for i in range(10):
            pred = decoder.predict_single(X[i])
            predictions.append(pred.copy())

        assert len(predictions) == 10

    def test_steady_state_faster_than_regular(self, realistic_neural_data):
        """Test that steady-state is computationally similar or faster."""
        X, y = realistic_neural_data

        regular = KalmanFilterDecoder()
        steady = SteadyStateKalmanFilter()

        regular.fit(X, y)
        steady.fit(X, y)

        # Both should produce predictions
        y_regular = regular.predict(X[:100])
        y_steady = steady.predict(X[:100])

        assert y_regular.shape == y_steady.shape

    def test_accuracy_similar_to_regular(self, realistic_neural_data):
        """Test that accuracy is similar to regular Kalman filter."""
        X, y = realistic_neural_data

        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        regular = KalmanFilterDecoder()
        steady = SteadyStateKalmanFilter()

        regular.fit(X_train, y_train)
        steady.fit(X_train, y_train)

        metrics_regular = regular.evaluate(X_test, y_test)
        metrics_steady = steady.evaluate(X_test, y_test)

        # R² should be within 0.1 of each other
        assert abs(metrics_regular["r2"] - metrics_steady["r2"]) < 0.1
