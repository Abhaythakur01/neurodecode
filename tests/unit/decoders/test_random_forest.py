"""
Unit tests for Random Forest decoder.
"""

import numpy as np
import pytest

from src.decoders.ml.random_forest import RandomForestClassifierDecoder, RandomForestDecoder


@pytest.fixture
def regression_data():
    """Generate regression data for Random Forest."""
    np.random.seed(42)
    n_samples = 300
    n_features = 20
    n_outputs = 2

    # Generate features with some informative and some noise features
    X = np.random.randn(n_samples, n_features)

    # Target depends on first few features
    y = np.column_stack(
        [
            2 * X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples),
            X[:, 0] - 2 * X[:, 3] + X[:, 4] + 0.5 * np.random.randn(n_samples),
        ]
    )

    return X, y


@pytest.fixture
def classification_data():
    """Generate classification data for Random Forest."""
    np.random.seed(42)
    n_samples_per_class = 100
    n_features = 15
    n_classes = 4

    X_list = []
    y_list = []

    for c in range(n_classes):
        mean = np.zeros(n_features)
        mean[c * 3 : (c + 1) * 3] = 2.5
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
class TestRandomForestDecoder:
    """Tests for RandomForestDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = RandomForestDecoder(n_estimators=50, max_depth=10)

        assert decoder.n_estimators == 50
        assert decoder.max_depth == 10
        assert not decoder.is_fitted

    def test_fit(self, regression_data):
        """Test fitting the decoder."""
        X, y = regression_data
        decoder = RandomForestDecoder(n_estimators=20, random_state=42)
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder._model is not None
        assert decoder.feature_importances_ is not None

    def test_predict(self, regression_data):
        """Test prediction."""
        X, y = regression_data
        decoder = RandomForestDecoder(n_estimators=20, random_state=42)
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        assert y_pred.shape == y.shape

    def test_predict_not_fitted(self, regression_data):
        """Test prediction raises error when not fitted."""
        X, _ = regression_data
        decoder = RandomForestDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_feature_importance(self, regression_data):
        """Test feature importance extraction."""
        X, y = regression_data
        decoder = RandomForestDecoder(n_estimators=50, random_state=42)
        decoder.fit(X, y)

        importance = decoder.get_feature_importance()

        assert len(importance) == X.shape[1]
        # First few features should have higher importance
        top_features = decoder.get_top_features(5)
        assert len(top_features) == 5

    def test_oob_score(self, regression_data):
        """Test out-of-bag score."""
        X, y = regression_data
        decoder = RandomForestDecoder(n_estimators=50, oob_score=True, random_state=42)
        decoder.fit(X, y)

        assert decoder.oob_score_ is not None
        assert 0 <= decoder.oob_score_ <= 1

    def test_decoding_accuracy(self, regression_data):
        """Test decoding accuracy."""
        X, y = regression_data

        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        decoder = RandomForestDecoder(n_estimators=100, random_state=42)
        decoder.fit(X_train, y_train)

        metrics = decoder.evaluate(X_test, y_test)

        # Should have reasonable R² on this data
        assert metrics["r2"] > 0.3

    def test_get_params(self, regression_data):
        """Test get_params method."""
        X, y = regression_data
        decoder = RandomForestDecoder(
            n_estimators=100, max_depth=15, oob_score=True, random_state=42
        )
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["n_estimators"] == 100
        assert params["max_depth"] == 15
        assert "oob_score" in params


@pytest.mark.unit
@pytest.mark.decoder
class TestRandomForestClassifierDecoder:
    """Tests for RandomForestClassifierDecoder class."""

    def test_init(self):
        """Test classifier initialization."""
        clf = RandomForestClassifierDecoder(n_estimators=50, max_depth=10)

        assert clf.n_estimators == 50
        assert clf.max_depth == 10
        assert not clf.is_fitted

    def test_fit(self, classification_data):
        """Test fitting the classifier."""
        X, y = classification_data
        clf = RandomForestClassifierDecoder(n_estimators=20, random_state=42)
        clf.fit(X, y)

        assert clf.is_fitted
        assert clf.classes_ is not None
        assert len(clf.classes_) == 4
        assert clf.feature_importances_ is not None

    def test_predict(self, classification_data):
        """Test class prediction."""
        X, y = classification_data
        clf = RandomForestClassifierDecoder(n_estimators=20, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(clf.classes_))

    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        clf = RandomForestClassifierDecoder(n_estimators=20, random_state=42)
        clf.fit(X, y)

        proba = clf.predict_proba(X)

        assert proba.shape == (len(X), 4)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)

    def test_evaluate(self, classification_data):
        """Test evaluation metrics."""
        X, y = classification_data

        train_idx = int(0.8 * len(X))
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_test, y_test = X[train_idx:], y[train_idx:]

        clf = RandomForestClassifierDecoder(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        metrics = clf.evaluate(X_test, y_test)

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.7  # Should do well on separated data

    def test_feature_importance(self, classification_data):
        """Test feature importance extraction."""
        X, y = classification_data
        clf = RandomForestClassifierDecoder(n_estimators=50, random_state=42)
        clf.fit(X, y)

        importance = clf.get_feature_importance()

        assert len(importance) == X.shape[1]

    def test_get_params(self, classification_data):
        """Test get_params method."""
        X, y = classification_data
        clf = RandomForestClassifierDecoder(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(X, y)

        params = clf.get_params()

        assert params["n_estimators"] == 100
        assert params["max_depth"] == 10
        assert params["n_classes"] == 4
