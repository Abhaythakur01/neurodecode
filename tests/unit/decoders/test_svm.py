"""
Unit tests for SVM decoder.
"""

import numpy as np
import pytest

from src.decoders.ml.svm import SVMClassifier, SVMDecoder


@pytest.fixture
def regression_data():
    """Generate regression data for SVM."""
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    n_outputs = 2

    # Generate smooth kinematics
    t = np.linspace(0, 5, n_samples)
    y = np.column_stack([np.sin(t), np.cos(t)])

    # Generate features with nonlinear relationship
    H = np.random.randn(n_features, n_outputs)
    X = np.tanh(y @ H.T) + 0.1 * np.random.randn(n_samples, n_features)

    return X, y


@pytest.fixture
def classification_data():
    """Generate classification data for SVM."""
    np.random.seed(42)
    n_samples_per_class = 100
    n_features = 15
    n_classes = 3

    X_list = []
    y_list = []

    for c in range(n_classes):
        mean = np.zeros(n_features)
        mean[c * 5 : (c + 1) * 5] = 2
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
class TestSVMDecoder:
    """Tests for SVMDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = SVMDecoder(kernel="rbf", C=10.0, epsilon=0.2)

        assert decoder.kernel == "rbf"
        assert decoder.C == 10.0
        assert decoder.epsilon == 0.2
        assert not decoder.is_fitted

    def test_fit(self, regression_data):
        """Test fitting the decoder."""
        X, y = regression_data
        decoder = SVMDecoder(kernel="rbf", C=1.0)
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder._model is not None

    def test_predict(self, regression_data):
        """Test prediction."""
        X, y = regression_data
        decoder = SVMDecoder(kernel="rbf", C=1.0)
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape

    def test_predict_not_fitted(self, regression_data):
        """Test prediction raises error when not fitted."""
        X, _ = regression_data
        decoder = SVMDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_kernels(self, regression_data):
        """Test different kernel types."""
        X, y = regression_data

        for kernel in ["linear", "rbf", "poly"]:
            decoder = SVMDecoder(kernel=kernel, C=1.0)
            decoder.fit(X, y)
            assert decoder.is_fitted

            y_pred = decoder.predict(X)
            assert y_pred.shape == y.shape

    def test_normalization(self, regression_data):
        """Test with and without normalization."""
        X, y = regression_data

        decoder_norm = SVMDecoder(normalize=True)
        decoder_no_norm = SVMDecoder(normalize=False)

        decoder_norm.fit(X, y)
        decoder_no_norm.fit(X, y)

        assert decoder_norm.is_fitted
        assert decoder_no_norm.is_fitted

    def test_single_output(self, regression_data):
        """Test with single output dimension."""
        X, y = regression_data
        y_single = y[:, 0]

        decoder = SVMDecoder()
        decoder.fit(X, y_single)

        y_pred = decoder.predict(X)
        assert y_pred.shape == (len(X), 1)

    def test_get_params(self, regression_data):
        """Test get_params method."""
        X, y = regression_data
        decoder = SVMDecoder(kernel="rbf", C=5.0, epsilon=0.1)
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["kernel"] == "rbf"
        assert params["C"] == 5.0
        assert params["epsilon"] == 0.1


@pytest.mark.unit
@pytest.mark.decoder
class TestSVMClassifier:
    """Tests for SVMClassifier class."""

    def test_init(self):
        """Test classifier initialization."""
        clf = SVMClassifier(kernel="rbf", C=1.0, probability=True)

        assert clf.kernel == "rbf"
        assert clf.C == 1.0
        assert clf.probability is True
        assert not clf.is_fitted

    def test_fit(self, classification_data):
        """Test fitting the classifier."""
        X, y = classification_data
        clf = SVMClassifier()
        clf.fit(X, y)

        assert clf.is_fitted
        assert clf.classes_ is not None
        assert len(clf.classes_) == 3

    def test_predict(self, classification_data):
        """Test class prediction."""
        X, y = classification_data
        clf = SVMClassifier()
        clf.fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(clf.classes_))

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        clf = SVMClassifier(probability=True)
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

        clf = SVMClassifier()
        clf.fit(X_train, y_train)

        metrics = clf.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_kernels(self, classification_data):
        """Test different kernel types."""
        X, y = classification_data

        for kernel in ["linear", "rbf", "poly"]:
            clf = SVMClassifier(kernel=kernel)
            clf.fit(X, y)
            assert clf.is_fitted

    def test_get_params(self, classification_data):
        """Test get_params method."""
        X, y = classification_data
        clf = SVMClassifier(kernel="linear", C=2.0)
        clf.fit(X, y)

        params = clf.get_params()

        assert params["kernel"] == "linear"
        assert params["C"] == 2.0
        assert params["n_classes"] == 3
