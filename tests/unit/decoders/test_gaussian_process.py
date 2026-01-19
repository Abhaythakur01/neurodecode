"""
Unit tests for Gaussian Process decoder.
"""

import numpy as np
import pytest

from src.decoders.ml.gaussian_process import (
    GaussianProcessDecoder,
    GPClassifier,
    SparseGPDecoder,
)


@pytest.fixture
def regression_data():
    """Generate regression data for GP."""
    np.random.seed(42)
    n_samples = 100  # Smaller for GP (computational cost)
    n_features = 10
    n_outputs = 2

    # Generate smooth function
    X = np.random.randn(n_samples, n_features)

    # Target is smooth function of features
    y = np.column_stack([
        np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples),
        np.cos(X[:, 2]) - 0.3 * X[:, 3] + 0.1 * np.random.randn(n_samples),
    ])

    return X, y


@pytest.fixture
def classification_data():
    """Generate classification data for GP."""
    np.random.seed(42)
    n_samples_per_class = 40  # Smaller for GP
    n_features = 8
    n_classes = 3

    X_list = []
    y_list = []

    for c in range(n_classes):
        mean = np.zeros(n_features)
        mean[c * 2 : (c + 1) * 2] = 2.5
        X_c = np.random.randn(n_samples_per_class, n_features) + mean
        y_c = np.full(n_samples_per_class, c)
        X_list.append(X_c)
        y_list.append(y_c)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


@pytest.mark.unit
@pytest.mark.decoder
class TestGaussianProcessDecoder:
    """Tests for GaussianProcessDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = GaussianProcessDecoder(kernel="rbf", length_scale=2.0)

        assert decoder.kernel_type == "rbf"
        assert decoder.length_scale == 2.0
        assert not decoder.is_fitted

    def test_fit(self, regression_data):
        """Test fitting the decoder."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert len(decoder._models) == 2  # One per output

    def test_predict(self, regression_data):
        """Test prediction."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape

    def test_predict_not_fitted(self, regression_data):
        """Test prediction raises error when not fitted."""
        X, _ = regression_data
        decoder = GaussianProcessDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_predict_with_uncertainty(self, regression_data):
        """Test prediction with uncertainty estimates."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        y_pred, y_std = decoder.predict_with_uncertainty(X)

        assert y_pred.shape == y.shape
        assert y_std.shape == y.shape
        assert np.all(y_std >= 0)  # Standard deviation is non-negative

    def test_get_uncertainty(self, regression_data):
        """Test getting uncertainty only."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        uncertainty = decoder.get_uncertainty(X)

        assert uncertainty.shape == y.shape
        assert np.all(uncertainty >= 0)

    def test_sample_predictions(self, regression_data):
        """Test sampling from posterior."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        samples = decoder.sample_predictions(X[:10], n_samples=5)

        assert samples.shape == (5, 10, 2)

    def test_kernels(self, regression_data):
        """Test different kernel types."""
        X, y = regression_data

        for kernel in ["rbf", "matern", "linear"]:
            decoder = GaussianProcessDecoder(
                kernel=kernel, n_restarts_optimizer=1, random_state=42
            )
            decoder.fit(X, y)
            assert decoder.is_fitted

    def test_normalization(self, regression_data):
        """Test with and without normalization."""
        X, y = regression_data

        decoder_norm = GaussianProcessDecoder(
            normalize_X=True, normalize_y=True, n_restarts_optimizer=1
        )
        decoder_no_norm = GaussianProcessDecoder(
            normalize_X=False, normalize_y=False, n_restarts_optimizer=1
        )

        decoder_norm.fit(X, y)
        decoder_no_norm.fit(X, y)

        assert decoder_norm.is_fitted
        assert decoder_no_norm.is_fitted

    def test_get_kernel_params(self, regression_data):
        """Test getting optimized kernel parameters."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="rbf", n_restarts_optimizer=1, random_state=42
        )
        decoder.fit(X, y)

        kernel_params = decoder.get_kernel_params()

        assert "output_0" in kernel_params
        assert "kernel" in kernel_params["output_0"]
        assert "log_marginal_likelihood" in kernel_params["output_0"]

    def test_get_params(self, regression_data):
        """Test get_params method."""
        X, y = regression_data
        decoder = GaussianProcessDecoder(
            kernel="matern", length_scale=1.5, n_restarts_optimizer=1
        )
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["kernel"] == "matern"
        assert params["length_scale"] == 1.5
        assert params["n_outputs"] == 2


@pytest.mark.unit
@pytest.mark.decoder
class TestSparseGPDecoder:
    """Tests for SparseGPDecoder class."""

    def test_init(self):
        """Test sparse GP initialization."""
        decoder = SparseGPDecoder(n_inducing=50, kernel="rbf")

        assert decoder.n_inducing == 50
        assert decoder.kernel_type == "rbf"
        assert not decoder.is_fitted

    def test_fit(self, regression_data):
        """Test fitting sparse GP."""
        X, y = regression_data
        decoder = SparseGPDecoder(n_inducing=30, random_state=42)
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder._inducing_points is not None
        assert len(decoder._inducing_points) <= decoder.n_inducing

    def test_predict(self, regression_data):
        """Test prediction."""
        X, y = regression_data
        decoder = SparseGPDecoder(n_inducing=30, random_state=42)
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape

    def test_predict_with_uncertainty(self, regression_data):
        """Test prediction with uncertainty."""
        X, y = regression_data
        decoder = SparseGPDecoder(n_inducing=30, random_state=42)
        decoder.fit(X, y)

        y_pred, y_std = decoder.predict_with_uncertainty(X)

        assert y_pred.shape == y.shape
        assert y_std.shape == y.shape

    def test_get_params(self, regression_data):
        """Test get_params method."""
        X, y = regression_data
        decoder = SparseGPDecoder(n_inducing=50, kernel="matern")
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["n_inducing"] == 50
        assert params["kernel"] == "matern"


@pytest.mark.unit
@pytest.mark.decoder
class TestGPClassifier:
    """Tests for GPClassifier class."""

    def test_init(self):
        """Test classifier initialization."""
        clf = GPClassifier(kernel="rbf", length_scale=1.0)

        assert clf.kernel_type == "rbf"
        assert clf.length_scale == 1.0
        assert not clf.is_fitted

    def test_fit(self, classification_data):
        """Test fitting the classifier."""
        X, y = classification_data
        clf = GPClassifier(kernel="rbf", n_restarts_optimizer=1, random_state=42)
        clf.fit(X, y)

        assert clf.is_fitted
        assert clf.classes_ is not None
        assert len(clf.classes_) == 3

    def test_predict(self, classification_data):
        """Test class prediction."""
        X, y = classification_data
        clf = GPClassifier(kernel="rbf", n_restarts_optimizer=1, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(clf.classes_))

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        clf = GPClassifier(kernel="rbf", n_restarts_optimizer=1, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)

    def test_evaluate(self, classification_data):
        """Test evaluation metrics."""
        X, y = classification_data

        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        clf = GPClassifier(kernel="rbf", n_restarts_optimizer=1, random_state=42)
        clf.fit(X_train, y_train)

        metrics = clf.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_get_params(self, classification_data):
        """Test get_params method."""
        X, y = classification_data
        clf = GPClassifier(kernel="matern", length_scale=2.0)
        clf.fit(X, y)

        params = clf.get_params()

        assert params["kernel"] == "matern"
        assert params["length_scale"] == 2.0
        assert params["n_classes"] == 3
