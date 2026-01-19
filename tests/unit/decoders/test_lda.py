"""
Unit tests for LDA decoder.
"""

import numpy as np
import pytest

from src.decoders.classic.lda import LDADecoder, ShrinkageLDA


@pytest.fixture
def classification_data():
    """Generate well-separated classification data."""
    np.random.seed(42)
    n_samples_per_class = 100
    n_features = 20
    n_classes = 4

    X_list = []
    y_list = []

    for c in range(n_classes):
        # Each class has a different mean
        mean = np.zeros(n_features)
        mean[c * 5 : (c + 1) * 5] = 3  # Different features active for each class

        X_c = np.random.randn(n_samples_per_class, n_features) + mean
        y_c = np.full(n_samples_per_class, c)

        X_list.append(X_c)
        y_list.append(y_c)

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


@pytest.fixture
def binary_classification_data():
    """Generate simple binary classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    # Class 0: centered at -1
    X0 = np.random.randn(n_samples // 2, n_features) - 1
    y0 = np.zeros(n_samples // 2)

    # Class 1: centered at +1
    X1 = np.random.randn(n_samples // 2, n_features) + 1
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


@pytest.mark.unit
@pytest.mark.decoder
class TestLDADecoder:
    """Tests for LDADecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = LDADecoder(n_components=3, regularization=1e-3)

        assert decoder.n_components == 3
        assert decoder.regularization == 1e-3
        assert not decoder.is_fitted

    def test_fit(self, classification_data):
        """Test fitting the decoder."""
        X, y = classification_data
        decoder = LDADecoder()
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder.n_classes_ == 4
        assert decoder.classes_ is not None
        assert len(decoder.classes_) == 4
        assert decoder.means_ is not None
        assert decoder.covariance_ is not None

    def test_predict(self, classification_data):
        """Test prediction."""
        X, y = classification_data
        decoder = LDADecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(decoder.classes_))

    def test_predict_not_fitted(self, classification_data):
        """Test prediction raises error when not fitted."""
        X, _ = classification_data
        decoder = LDADecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        decoder = LDADecoder()
        decoder.fit(X, y)

        proba = decoder.predict_proba(X)

        assert proba.shape == (len(X), 4)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)
        # Probabilities should be non-negative
        assert np.all(proba >= 0)

    def test_transform(self, classification_data):
        """Test LDA transform (dimensionality reduction)."""
        X, y = classification_data
        decoder = LDADecoder(n_components=2)
        decoder.fit(X, y)

        X_transformed = decoder.transform(X)

        # Should reduce to n_components dimensions
        assert X_transformed.shape == (len(X), 2)

    def test_classification_accuracy(self, classification_data):
        """Test classification accuracy on well-separated data."""
        X, y = classification_data

        # Train/test split
        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        decoder = LDADecoder()
        decoder.fit(X_train, y_train)

        metrics = decoder.evaluate(X_test, y_test)

        # Should have high accuracy on well-separated data
        assert metrics["accuracy"] > 0.8

    def test_binary_classification(self, binary_classification_data):
        """Test binary classification."""
        X, y = binary_classification_data

        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        decoder = LDADecoder()
        decoder.fit(X_train, y_train)

        assert decoder.n_classes_ == 2

        metrics = decoder.evaluate(X_test, y_test)
        assert metrics["accuracy"] > 0.9  # Should be very accurate

    def test_evaluate_metrics(self, classification_data):
        """Test evaluation returns expected metrics."""
        X, y = classification_data
        decoder = LDADecoder()
        decoder.fit(X, y)

        metrics = decoder.evaluate(X, y)

        assert "accuracy" in metrics
        assert "r2" in metrics  # For compatibility
        assert "mse" in metrics
        # Per-class accuracies
        assert any("accuracy_class" in k for k in metrics.keys())

    def test_priors(self, classification_data):
        """Test custom priors."""
        X, y = classification_data
        custom_priors = [0.1, 0.2, 0.3, 0.4]

        decoder = LDADecoder(prior=custom_priors)
        decoder.fit(X, y)

        assert np.allclose(decoder.priors_, custom_priors)

    def test_regularization_effect(self, classification_data):
        """Test that regularization affects the model."""
        X, y = classification_data

        decoder_low = LDADecoder(regularization=1e-6)
        decoder_high = LDADecoder(regularization=1.0)

        decoder_low.fit(X, y)
        decoder_high.fit(X, y)

        # Covariances should differ
        assert not np.allclose(decoder_low.covariance_, decoder_high.covariance_)

    def test_get_params(self, classification_data):
        """Test get_params method."""
        X, y = classification_data
        decoder = LDADecoder(n_components=2, regularization=0.01)
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["n_components"] == 2
        assert params["regularization"] == 0.01
        assert params["n_classes"] == 4

    def test_single_sample_predict(self, classification_data):
        """Test prediction on single sample."""
        X, y = classification_data
        decoder = LDADecoder()
        decoder.fit(X, y)

        y_pred = decoder.predict(X[0])
        assert y_pred.shape == (1,)


@pytest.mark.unit
@pytest.mark.decoder
class TestShrinkageLDA:
    """Tests for ShrinkageLDA class."""

    def test_init(self):
        """Test shrinkage LDA initialization."""
        decoder = ShrinkageLDA(shrinkage=0.5)
        assert decoder.shrinkage == 0.5

    def test_fit(self, classification_data):
        """Test fitting shrinkage LDA."""
        X, y = classification_data
        decoder = ShrinkageLDA()
        decoder.fit(X, y)

        assert decoder.is_fitted

    def test_auto_shrinkage(self, classification_data):
        """Test automatic shrinkage estimation."""
        X, y = classification_data
        decoder = ShrinkageLDA(shrinkage=None)  # Auto-estimate
        decoder.fit(X, y)

        assert decoder.is_fitted

    def test_shrinkage_improves_small_sample(self):
        """Test shrinkage helps with small sample sizes."""
        np.random.seed(42)
        # Small sample, many features (p > n scenario)
        n_samples = 20
        n_features = 50
        n_classes = 2

        X = np.random.randn(n_samples, n_features)
        y = np.array([0] * 10 + [1] * 10)

        # Add class separation
        X[:10, :10] += 2
        X[10:, 10:20] += 2

        # Regular LDA might fail or perform poorly
        shrinkage_decoder = ShrinkageLDA(shrinkage=0.5)
        shrinkage_decoder.fit(X, y)

        # Should be able to fit without error
        assert shrinkage_decoder.is_fitted
        y_pred = shrinkage_decoder.predict(X)
        assert len(y_pred) == n_samples
