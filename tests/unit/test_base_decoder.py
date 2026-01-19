"""
Unit tests for base decoder classes.
"""

import numpy as np
import pytest

from src.decoders.base import BaseDecoder, OnlineDecoder


class ConcreteDecoder(BaseDecoder):
    """Concrete implementation of BaseDecoder for testing."""

    def __init__(self, name: str = "ConcreteDecoder"):
        super().__init__(name=name)
        self._coef = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConcreteDecoder":
        self._validate_input(X, y)
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]
        # Simple linear regression
        self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return X @ self._coef


class ConcreteOnlineDecoder(OnlineDecoder):
    """Concrete implementation of OnlineDecoder for testing."""

    def __init__(self, name: str = "ConcreteOnline", learning_rate: float = 0.01):
        super().__init__(name=name, learning_rate=learning_rate)
        self._coef = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConcreteOnlineDecoder":
        self._validate_input(X, y)
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]
        self._coef = np.linalg.lstsq(X, y, rcond=None)[0]
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return X @ self._coef

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        coef_new = np.linalg.lstsq(X, y, rcond=None)[0]
        self._coef = (1 - self.learning_rate) * self._coef + self.learning_rate * coef_new
        self._update_count += 1


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples, n_features, n_outputs = 100, 10, 2
    X = np.random.randn(n_samples, n_features)
    coef = np.random.randn(n_features, n_outputs)
    y = X @ coef + 0.1 * np.random.randn(n_samples, n_outputs)
    return X, y


@pytest.mark.unit
class TestBaseDecoder:
    """Tests for BaseDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = ConcreteDecoder(name="TestDecoder")
        assert decoder.name == "TestDecoder"
        assert not decoder.is_fitted
        assert decoder.n_features is None
        assert decoder.n_outputs is None

    def test_fit(self, sample_data):
        """Test fitting decoder."""
        X, y = sample_data
        decoder = ConcreteDecoder()
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder.n_features == X.shape[1]
        assert decoder.n_outputs == y.shape[1]

    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        decoder = ConcreteDecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X)
        assert y_pred.shape == y.shape

    def test_predict_not_fitted(self, sample_data):
        """Test prediction raises error when not fitted."""
        X, _ = sample_data
        decoder = ConcreteDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_evaluate(self, sample_data):
        """Test evaluation."""
        X, y = sample_data
        decoder = ConcreteDecoder()
        decoder.fit(X, y)

        metrics = decoder.evaluate(X, y)
        assert "r2" in metrics
        assert "mse" in metrics
        assert metrics["r2"] > 0.9  # Should fit well

    def test_validate_input_wrong_dims(self):
        """Test validation rejects wrong dimensions."""
        decoder = ConcreteDecoder()

        with pytest.raises(ValueError):
            decoder._validate_input(np.array([1, 2, 3]), None)  # 1D instead of 2D

    def test_validate_input_mismatched_samples(self):
        """Test validation rejects mismatched sample counts."""
        decoder = ConcreteDecoder()
        X = np.random.randn(10, 5)
        y = np.random.randn(20, 2)  # Different number of samples

        with pytest.raises(ValueError):
            decoder._validate_input(X, y)

    def test_compute_r2(self, sample_data):
        """Test R² computation."""
        X, y = sample_data
        decoder = ConcreteDecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X)
        r2 = decoder._compute_r2(y, y_pred)

        assert 0 <= r2 <= 1

    def test_compute_mse(self, sample_data):
        """Test MSE computation."""
        X, y = sample_data
        decoder = ConcreteDecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X)
        mse = decoder._compute_mse(y, y_pred)

        assert mse >= 0

    def test_get_params(self):
        """Test get_params method."""
        decoder = ConcreteDecoder(name="MyDecoder")
        params = decoder.get_params()

        assert params["name"] == "MyDecoder"
        assert params["is_fitted"] is False

    def test_repr(self):
        """Test string representation."""
        decoder = ConcreteDecoder(name="TestDecoder")
        repr_str = repr(decoder)

        assert "ConcreteDecoder" in repr_str
        assert "TestDecoder" in repr_str


@pytest.mark.unit
class TestOnlineDecoder:
    """Tests for OnlineDecoder class."""

    def test_init(self):
        """Test online decoder initialization."""
        decoder = ConcreteOnlineDecoder(learning_rate=0.05)
        assert decoder.learning_rate == 0.05
        assert decoder._update_count == 0

    def test_update(self, sample_data):
        """Test online update."""
        X, y = sample_data
        decoder = ConcreteOnlineDecoder()
        decoder.fit(X[:50], y[:50])

        initial_coef = decoder._coef.copy()
        decoder.update(X[50:], y[50:])

        assert decoder._update_count == 1
        assert not np.allclose(decoder._coef, initial_coef)

    def test_partial_fit(self, sample_data):
        """Test partial_fit method."""
        X, y = sample_data
        decoder = ConcreteOnlineDecoder()

        # First call should fit
        decoder.partial_fit(X[:50], y[:50])
        assert decoder.is_fitted
        assert decoder._update_count == 0

        # Second call should update
        decoder.partial_fit(X[50:], y[50:])
        assert decoder._update_count == 1

    def test_get_params_online(self, sample_data):
        """Test get_params includes online-specific params."""
        X, y = sample_data
        decoder = ConcreteOnlineDecoder(learning_rate=0.1)
        decoder.fit(X, y)
        decoder.update(X, y)

        params = decoder.get_params()
        assert params["learning_rate"] == 0.1
        assert params["update_count"] == 1
