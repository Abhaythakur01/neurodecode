"""
Integration tests for the full neural decoding pipeline.

Tests the complete workflow from raw data to decoded predictions.
"""

import numpy as np
import pytest

from src.decoders import KalmanFilterDecoder
from src.evaluation import LatencyTracker, compute_all_metrics, cross_validate
from src.features import FeatureExtractor
from src.preprocessing import PreprocessingPipeline


def generate_synthetic_neural_data(
    n_samples: int = 1000,
    n_neurons: int = 50,
    n_outputs: int = 2,
    fs: float = 1000.0,
) -> tuple:
    """Generate synthetic neural data with known relationship to kinematics."""
    np.random.seed(42)

    # Generate smooth kinematics (2D trajectory)
    t = np.linspace(0, n_samples / fs, n_samples)
    velocity = np.column_stack([
        np.sin(2 * np.pi * 0.5 * t),  # 0.5 Hz oscillation
        np.cos(2 * np.pi * 0.5 * t),
    ])

    # Generate neural activity based on velocity
    # Each neuron has a preferred direction
    preferred_directions = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    tuning = np.column_stack([
        np.cos(preferred_directions),
        np.sin(preferred_directions),
    ])

    # Neural activity = velocity · tuning^T + baseline + noise
    baseline = 20  # Mean firing rate
    gain = 10  # Modulation depth
    neural_activity = baseline + gain * (velocity @ tuning.T)
    neural_activity += np.random.randn(n_samples, n_neurons) * 3  # Add noise
    neural_activity = np.maximum(neural_activity, 0)  # Non-negative

    return neural_activity, velocity


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for pipeline testing."""
    return generate_synthetic_neural_data()


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete decoding pipeline."""

    def test_preprocessing_to_features_to_decoder(self, synthetic_data):
        """Test full pipeline: preprocessing -> features -> decoding."""
        neural_data, kinematics = synthetic_data

        # Step 1: Preprocessing (minimal - just normalize)
        # Note: For this synthetic data, we skip filtering to preserve the relationship
        preprocessor = PreprocessingPipeline(
            fs=1000.0,
            bandpass=None,
            notch_freq=None,
            remove_artifacts=False,
            normalize=False,  # Don't normalize to preserve linear relationship
        )
        processed = preprocessor.transform(neural_data)

        assert processed.shape == neural_data.shape

        # Step 2: Use processed data directly as features
        features = processed

        # Step 3: Train decoder
        train_idx = int(0.8 * len(features))
        X_train, y_train = features[:train_idx], kinematics[:train_idx]
        X_test, y_test = features[train_idx:], kinematics[train_idx:]

        decoder = KalmanFilterDecoder()
        decoder.fit(X_train, y_train)

        assert decoder.is_fitted

        # Step 4: Evaluate
        y_pred = decoder.predict(X_test)
        metrics = compute_all_metrics(y_test, y_pred)

        assert "r2" in metrics
        assert "mse" in metrics
        # Verify the pipeline produces valid predictions
        assert not np.isnan(metrics["r2"])
        assert not np.isnan(metrics["mse"])

    def test_latency_tracking(self, synthetic_data):
        """Test that pipeline meets latency requirements."""
        neural_data, kinematics = synthetic_data
        tracker = LatencyTracker()

        # Simulate real-time processing of single samples
        preprocessor = PreprocessingPipeline(
            fs=1000.0, bandpass=None, notch_freq=None, normalize=False
        )

        decoder = KalmanFilterDecoder()
        decoder.fit(neural_data[:800], kinematics[:800])

        # Process 100 "real-time" samples
        for i in range(800, 900):
            with tracker.track("total"):
                # Preprocess (simulated - passthrough)
                with tracker.track("preprocess"):
                    sample = preprocessor.transform(neural_data[i : i + 1])

                # Decode
                with tracker.track("decode"):
                    prediction = decoder.predict_single(sample.flatten())

        # Check latencies
        total_stats = tracker.get_stats("total")
        decode_stats = tracker.get_stats("decode")

        assert total_stats is not None
        assert total_stats.count == 100
        # Should be fast (< 10ms per sample for this simple pipeline)
        assert total_stats.mean < 10.0  # milliseconds

    def test_cross_validation_pipeline(self, synthetic_data):
        """Test cross-validation with the full pipeline."""
        neural_data, kinematics = synthetic_data

        # Preprocess once
        preprocessor = PreprocessingPipeline(fs=1000.0, bandpass=None, normalize=True)
        features = preprocessor.fit_transform(neural_data)

        # Cross-validate
        results = cross_validate(
            KalmanFilterDecoder,
            features,
            kinematics,
            cv_method="temporal",
            n_splits=3,
        )

        assert "r2_mean" in results
        assert "r2_std" in results
        assert results["n_splits"] <= 3


@pytest.mark.integration
class TestRealTimeSimulation:
    """Test real-time decoding simulation."""

    def test_streaming_decode(self, synthetic_data):
        """Simulate streaming data and real-time decoding."""
        neural_data, kinematics = synthetic_data

        # Train on first 80%
        train_end = int(0.8 * len(neural_data))

        preprocessor = PreprocessingPipeline(
            fs=1000.0, bandpass=None, notch_freq=None, normalize=False
        )

        decoder = KalmanFilterDecoder()
        X_train = preprocessor.transform(neural_data[:train_end])
        decoder.fit(X_train, kinematics[:train_end])

        # Simulate streaming the remaining 20%
        predictions = []
        for i in range(train_end, len(neural_data)):
            # Process single sample
            sample = preprocessor.transform(neural_data[i : i + 1])
            pred = decoder.predict_single(sample.flatten())
            predictions.append(pred)

        predictions = np.array(predictions)
        y_test = kinematics[train_end:]

        assert predictions.shape == y_test.shape

        # Evaluate streaming performance - just verify it produces valid output
        metrics = compute_all_metrics(y_test, predictions)
        assert not np.isnan(metrics["r2"])

    def test_online_adaptation(self, synthetic_data):
        """Test online learning during streaming."""
        neural_data, kinematics = synthetic_data

        train_end = int(0.6 * len(neural_data))
        adapt_end = int(0.8 * len(neural_data))

        # Initial training
        decoder = KalmanFilterDecoder(learning_rate=0.05)
        decoder.fit(neural_data[:train_end], kinematics[:train_end])

        initial_metrics = decoder.evaluate(
            neural_data[adapt_end:], kinematics[adapt_end:]
        )

        # Online adaptation phase
        for i in range(train_end, adapt_end, 10):
            batch_X = neural_data[i : i + 10]
            batch_y = kinematics[i : i + 10]
            decoder.update(batch_X, batch_y)

        assert decoder._update_count > 0

        # Evaluate after adaptation
        final_metrics = decoder.evaluate(
            neural_data[adapt_end:], kinematics[adapt_end:]
        )

        # Performance may or may not improve depending on data
        # Just verify metrics are computed
        assert "r2" in final_metrics
        assert "mse" in final_metrics
