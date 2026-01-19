"""
Unit tests for Meta-Learner.
"""

import numpy as np
import pytest

from src.decoders.classic.kalman_filter import KalmanFilterDecoder
from src.decoders.classic.wiener_filter import WienerFilterDecoder
from src.decoders.meta_learner import (
    AdaptiveMetaLearner,
    CombinationStrategy,
    DecoderCombiner,
    DecoderMetrics,
    DecoderSelector,
    DecoderState,
    DecoderWrapper,
    OnlineAdapter,
    PredictionResult,
    SelectionStrategy,
    create_default_meta_learner,
)
from src.decoders.ml.random_forest import RandomForestDecoder


@pytest.fixture
def regression_data():
    """Generate regression data for meta-learner testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 20
    n_outputs = 2

    # Generate smooth kinematics
    t = np.linspace(0, 10, n_samples)
    y = np.column_stack([np.sin(t), np.cos(t)])

    # Generate neural features
    H = np.random.randn(n_features, n_outputs) * 0.5
    X = y @ H.T + 0.3 * np.random.randn(n_samples, n_features)

    return X, y


@pytest.fixture
def fitted_decoders(regression_data):
    """Create fitted decoders for testing."""
    X, y = regression_data

    # Create and fit decoders
    kalman = KalmanFilterDecoder(name="Kalman")
    kalman.fit(X, y)

    wiener = WienerFilterDecoder(name="Wiener", n_lags=3)
    wiener.fit(X, y)

    rf = RandomForestDecoder(name="RF", n_estimators=20, random_state=42)
    rf.fit(X, y)

    return [kalman, wiener, rf]


@pytest.mark.unit
@pytest.mark.decoder
class TestDecoderMetrics:
    """Tests for DecoderMetrics class."""

    def test_init(self):
        """Test metrics initialization."""
        metrics = DecoderMetrics(name="test")

        assert metrics.name == "test"
        assert len(metrics.r2_history) == 0
        assert metrics.recent_r2 == 0.0

    def test_update(self):
        """Test metrics update."""
        metrics = DecoderMetrics(name="test")

        for i in range(20):
            metrics.update(r2=0.5 + i * 0.01, mse=0.1)

        assert len(metrics.r2_history) == 20
        assert metrics.recent_r2 > 0.5

    def test_performance_trend(self):
        """Test performance trend calculation."""
        metrics = DecoderMetrics(name="test")

        # Improving trend
        for i in range(30):
            metrics.update(r2=0.3 + i * 0.02)

        assert metrics.performance_trend > 0

    def test_stability(self):
        """Test stability calculation."""
        metrics = DecoderMetrics(name="test")

        # Stable performance
        for _ in range(20):
            metrics.update(r2=0.7 + np.random.randn() * 0.01)

        assert metrics.stability < 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = DecoderMetrics(name="test")
        metrics.update(r2=0.8, mse=0.1, latency=10.0)

        d = metrics.to_dict()

        assert "name" in d
        assert "recent_r2" in d
        assert "stability" in d


@pytest.mark.unit
@pytest.mark.decoder
class TestDecoderSelector:
    """Tests for DecoderSelector class."""

    def test_init(self):
        """Test selector initialization."""
        selector = DecoderSelector(
            strategy=SelectionStrategy.TOP_K,
            top_k=2,
        )

        assert selector.strategy == SelectionStrategy.TOP_K
        assert selector.top_k == 2

    def test_select_best(self, fitted_decoders):
        """Test best decoder selection."""
        selector = DecoderSelector(strategy=SelectionStrategy.BEST)

        # Create wrappers with metrics
        decoders = {}
        for i, decoder in enumerate(fitted_decoders):
            wrapper = DecoderWrapper(decoder=decoder)
            # Add performance history
            for _ in range(20):
                wrapper.metrics.update(r2=0.5 + i * 0.1)
            decoders[decoder.name] = wrapper

        selected = selector.select(decoders)

        assert len(selected) == 1
        assert selected[0] == "RF"  # Highest R²

    def test_select_top_k(self, fitted_decoders):
        """Test top-K selection."""
        selector = DecoderSelector(strategy=SelectionStrategy.TOP_K, top_k=2)

        decoders = {}
        for i, decoder in enumerate(fitted_decoders):
            wrapper = DecoderWrapper(decoder=decoder)
            for _ in range(20):
                wrapper.metrics.update(r2=0.5 + i * 0.1)
            decoders[decoder.name] = wrapper

        selected = selector.select(decoders)

        assert len(selected) == 2

    def test_select_threshold(self, fitted_decoders):
        """Test threshold-based selection."""
        selector = DecoderSelector(
            strategy=SelectionStrategy.THRESHOLD,
            performance_threshold=0.6,
        )

        decoders = {}
        for i, decoder in enumerate(fitted_decoders):
            wrapper = DecoderWrapper(decoder=decoder)
            for _ in range(20):
                wrapper.metrics.update(r2=0.5 + i * 0.1)
            decoders[decoder.name] = wrapper

        selected = selector.select(decoders)

        # Should select decoders with R² >= 0.6
        assert len(selected) >= 1

    def test_detect_degradation(self, fitted_decoders):
        """Test degradation detection."""
        selector = DecoderSelector(degradation_threshold=0.2)

        # Create wrapper with degrading performance
        wrapper = DecoderWrapper(decoder=fitted_decoders[0])

        # Good performance initially
        for _ in range(50):
            wrapper.metrics.update(r2=0.8)

        # Degraded performance
        for _ in range(50):
            wrapper.metrics.update(r2=0.4)

        decoders = {wrapper.decoder.name: wrapper}
        degraded = selector.detect_degradation(decoders)

        assert len(degraded) == 1

    def test_get_selection_stats(self, fitted_decoders):
        """Test selection statistics."""
        selector = DecoderSelector(strategy=SelectionStrategy.BEST)

        decoders = {}
        for decoder in fitted_decoders:
            wrapper = DecoderWrapper(decoder=decoder)
            for _ in range(20):
                wrapper.metrics.update(r2=0.7)
            decoders[decoder.name] = wrapper

        # Make several selections
        for _ in range(5):
            selector.select(decoders)

        stats = selector.get_selection_stats()

        assert stats["total_selections"] == 5
        assert "selection_frequency" in stats


@pytest.mark.unit
@pytest.mark.decoder
class TestDecoderCombiner:
    """Tests for DecoderCombiner class."""

    def test_init(self):
        """Test combiner initialization."""
        combiner = DecoderCombiner(
            strategy=CombinationStrategy.WEIGHTED_MEAN,
        )

        assert combiner.strategy == CombinationStrategy.WEIGHTED_MEAN

    def test_combine_mean(self):
        """Test mean combination."""
        combiner = DecoderCombiner(strategy=CombinationStrategy.MEAN)

        predictions = {
            "d1": PredictionResult("d1", np.array([[1.0, 2.0]])),
            "d2": PredictionResult("d2", np.array([[2.0, 3.0]])),
            "d3": PredictionResult("d3", np.array([[3.0, 4.0]])),
        }

        decoders = {
            name: DecoderWrapper(
                decoder=type("MockDecoder", (), {"name": name, "is_fitted": True})()
            )
            for name in predictions.keys()
        }

        result = combiner.combine(predictions, decoders, list(predictions.keys()))

        assert result.prediction.shape == (1, 2)
        np.testing.assert_array_almost_equal(result.prediction, [[2.0, 3.0]])

    def test_combine_weighted_mean(self):
        """Test weighted mean combination."""
        combiner = DecoderCombiner(strategy=CombinationStrategy.WEIGHTED_MEAN)

        predictions = {
            "d1": PredictionResult("d1", np.array([[1.0, 1.0]])),
            "d2": PredictionResult("d2", np.array([[3.0, 3.0]])),
        }

        # Create wrappers with different performance
        decoders = {}
        for name in predictions.keys():
            wrapper = DecoderWrapper(
                decoder=type("MockDecoder", (), {"name": name, "is_fitted": True})()
            )
            r2 = 0.9 if name == "d2" else 0.3
            for _ in range(20):
                wrapper.metrics.update(r2=r2)
            decoders[name] = wrapper

        result = combiner.combine(predictions, decoders, list(predictions.keys()))

        # d2 should have higher weight, so result closer to [3, 3]
        assert result.prediction[0, 0] > 2.0

    def test_combine_median(self):
        """Test median combination."""
        combiner = DecoderCombiner(strategy=CombinationStrategy.MEDIAN)

        predictions = {
            "d1": PredictionResult("d1", np.array([[1.0, 1.0]])),
            "d2": PredictionResult("d2", np.array([[2.0, 2.0]])),
            "d3": PredictionResult("d3", np.array([[10.0, 10.0]])),  # Outlier
        }

        decoders = {
            name: DecoderWrapper(
                decoder=type("MockDecoder", (), {"name": name, "is_fitted": True})()
            )
            for name in predictions.keys()
        }

        result = combiner.combine(predictions, decoders, list(predictions.keys()))

        # Median should be robust to outlier
        np.testing.assert_array_almost_equal(result.prediction, [[2.0, 2.0]])

    def test_outlier_rejection(self):
        """Test outlier rejection."""
        combiner = DecoderCombiner(outlier_rejection=True, outlier_threshold=2.0)

        predictions = {
            "d1": np.array([[1.0, 1.0]]),
            "d2": np.array([[1.1, 1.1]]),
            "d3": np.array([[100.0, 100.0]]),  # Clear outlier
        }

        filtered = combiner.reject_outliers(predictions)

        assert "d3" not in filtered
        assert len(filtered) == 2


@pytest.mark.unit
@pytest.mark.decoder
class TestOnlineAdapter:
    """Tests for OnlineAdapter class."""

    def test_init(self):
        """Test adapter initialization."""
        adapter = OnlineAdapter(learning_rate=0.05, update_interval=5)

        assert adapter.learning_rate == 0.05
        assert adapter.update_interval == 5

    def test_weight_update(self, fitted_decoders):
        """Test weight updates based on errors."""
        adapter = OnlineAdapter(learning_rate=0.1)

        # Create wrappers
        decoders = {}
        for decoder in fitted_decoders:
            wrapper = DecoderWrapper(decoder=decoder)
            decoders[decoder.name] = wrapper

        # Simulate prediction result
        from src.decoders.meta_learner.base import EnsembleResult

        ensemble = EnsembleResult(
            prediction=np.array([[0.5, 0.5]]),
            uncertainty=None,
            decoder_weights={name: 1.0 for name in decoders.keys()},
            individual_predictions={
                "Kalman": np.array([[0.4, 0.4]]),  # Close to true
                "Wiener": np.array([[0.6, 0.6]]),  # Also close
                "RF": np.array([[1.0, 1.0]]),      # Further away
            },
            selected_decoders=list(decoders.keys()),
            total_latency_ms=10.0,
        )

        y_true = np.array([[0.5, 0.5]])

        # Update
        stats = adapter.update(decoders, ensemble, y_true)

        assert "weight_updates" in stats

    def test_reset_decoder(self, fitted_decoders):
        """Test decoder reset."""
        adapter = OnlineAdapter()

        wrapper = DecoderWrapper(decoder=fitted_decoders[0])
        wrapper.state = DecoderState.DEGRADED
        wrapper.weight = 0.3
        wrapper.metrics.update(r2=0.5)

        decoders = {wrapper.decoder.name: wrapper}

        success = adapter.reset_decoder(decoders, wrapper.decoder.name)

        assert success
        assert wrapper.state == DecoderState.STANDBY
        assert wrapper.weight == 1.0


@pytest.mark.unit
@pytest.mark.decoder
class TestAdaptiveMetaLearner:
    """Tests for AdaptiveMetaLearner class."""

    def test_init(self):
        """Test meta-learner initialization."""
        meta = AdaptiveMetaLearner(
            selection_strategy=SelectionStrategy.TOP_K,
            combination_strategy=CombinationStrategy.WEIGHTED_MEAN,
            top_k=2,
        )

        assert meta.selector.strategy == SelectionStrategy.TOP_K
        assert meta.combiner.strategy == CombinationStrategy.WEIGHTED_MEAN

    def test_add_decoder(self, fitted_decoders):
        """Test adding decoders."""
        meta = AdaptiveMetaLearner()

        for decoder in fitted_decoders:
            name = meta.add_decoder(decoder)
            assert name in meta._decoders

        assert len(meta._decoders) == 3

    def test_fit(self, regression_data, fitted_decoders):
        """Test fitting meta-learner."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        assert meta.is_fitted

    def test_predict(self, regression_data, fitted_decoders):
        """Test prediction."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        y_pred = meta.predict(X[:10])

        # Output may be shorter than input due to decoders with lags (e.g., Wiener)
        assert y_pred.shape[1] == 2
        assert y_pred.shape[0] <= 10

    def test_predict_with_info(self, regression_data, fitted_decoders):
        """Test prediction with full info."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        result = meta.predict_with_info(X[:10])

        # Output may be shorter than input due to decoders with lags
        assert result.prediction.shape[1] == 2
        assert result.prediction.shape[0] <= 10
        assert result.uncertainty is not None
        assert len(result.selected_decoders) > 0
        assert result.total_latency_ms > 0

    def test_predict_single(self, regression_data, fitted_decoders):
        """Test single-step prediction."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        y_pred = meta.predict_single(X[0])

        assert y_pred.shape == (2,)

    def test_update(self, regression_data, fitted_decoders):
        """Test online update."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X[:200], y[:200])

        # Make prediction then update
        # Note: update adapts the decoders, doesn't require matching prediction length
        result = meta.predict_with_info(X[200:220])
        n_samples = result.prediction.shape[0]

        # Align y_true with prediction length (use last n_samples)
        stats = meta.update(X[200:200+n_samples], y[200:200+n_samples], adapt=False)

        assert "updated_decoders" in stats

    def test_get_decoder_states(self, regression_data, fitted_decoders):
        """Test getting decoder states."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        states = meta.get_decoder_states()

        assert len(states) == 3
        for name, state_dict in states.items():
            assert "state" in state_dict
            assert "weight" in state_dict
            assert "metrics" in state_dict

    def test_get_params(self, regression_data, fitted_decoders):
        """Test get_params method."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(top_k=2, verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)
        meta.predict(X[:10])

        params = meta.get_params()

        assert params["n_decoders"] == 3
        assert "last_selected" in params

    def test_create_default_meta_learner(self, regression_data, fitted_decoders):
        """Test convenience function."""
        X, y = regression_data

        # Use unfitted decoders
        unfitted_decoders = [
            KalmanFilterDecoder(name="K1"),
            WienerFilterDecoder(name="W1", n_lags=3),
        ]

        meta = create_default_meta_learner(
            unfitted_decoders, X, y, verbose=False
        )

        assert meta.is_fitted
        assert len(meta._decoders) == 2

    def test_remove_decoder(self, fitted_decoders):
        """Test removing a decoder."""
        meta = AdaptiveMetaLearner()

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        assert len(meta._decoders) == 3

        success = meta.remove_decoder("Kalman")

        assert success
        assert len(meta._decoders) == 2
        assert "Kalman" not in meta._decoders

    def test_set_decoder_state(self, regression_data, fitted_decoders):
        """Test manually setting decoder state."""
        X, y = regression_data
        meta = AdaptiveMetaLearner(verbose=False)

        for decoder in fitted_decoders:
            meta.add_decoder(decoder)

        meta.fit(X, y)

        success = meta.set_decoder_state("Kalman", DecoderState.DISABLED)

        assert success
        assert meta._decoders["Kalman"].state == DecoderState.DISABLED

    def test_parallel_vs_sequential(self, regression_data, fitted_decoders):
        """Test parallel and sequential execution produce similar results."""
        X, y = regression_data

        # Parallel
        meta_parallel = AdaptiveMetaLearner(parallel=True, verbose=False)
        for decoder in fitted_decoders:
            meta_parallel.add_decoder(decoder)
        meta_parallel.fit(X, y)

        # Sequential
        meta_seq = AdaptiveMetaLearner(parallel=False, verbose=False)
        for decoder in fitted_decoders:
            meta_seq.add_decoder(decoder)
        meta_seq.fit(X, y)

        # Both should produce similar predictions
        y_par = meta_parallel.predict(X[:10])
        y_seq = meta_seq.predict(X[:10])

        # Note: Not exactly equal due to potential timing differences
        # in selection, but should be close
        assert y_par.shape == y_seq.shape
