"""
Unit tests for LSTM decoder.
"""

import numpy as np
import pytest

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.decoders.deep_learning.lstm import (
        BidirectionalLSTMDecoder,
        LSTMDecoder,
    )


@pytest.fixture
def temporal_data():
    """Generate temporal data suitable for LSTM."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    n_outputs = 2

    # Generate smooth kinematics with temporal structure
    t = np.linspace(0, 10, n_samples)
    y = np.column_stack([np.sin(2 * np.pi * 0.5 * t), np.cos(2 * np.pi * 0.5 * t)])

    # Neural features depend on recent kinematics
    X = np.zeros((n_samples, n_features))
    H = np.random.randn(n_features, n_outputs)
    for i in range(n_samples):
        # Use recent kinematics
        start = max(0, i - 5)
        recent_y = y[start : i + 1].mean(axis=0) if i > 0 else y[0]
        X[i] = recent_y @ H.T + 0.5 * np.random.randn(n_features)

    return X, y


@pytest.mark.unit
@pytest.mark.decoder
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMNetwork:
    """Tests for LSTMNetwork module (via LSTMDecoder)."""

    def test_init(self, temporal_data):
        """Test network initialization through decoder."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=64, num_layers=2, sequence_length=10, n_epochs=1, verbose=False
        )
        decoder.fit(X, y)

        assert decoder._model.hidden_size == 64
        assert decoder._model.num_layers == 2

    def test_forward(self, temporal_data):
        """Test forward pass."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=64, sequence_length=10, n_epochs=1, verbose=False
        )
        decoder.fit(X, y)

        batch_size = 8
        seq_len = 10
        x = torch.randn(batch_size, seq_len, X.shape[1])

        output, (h_n, c_n) = decoder._model(x)

        assert output.shape == (batch_size, seq_len, y.shape[1])
        assert h_n.shape[1] == batch_size
        assert c_n.shape[1] == batch_size

    def test_init_hidden(self, temporal_data):
        """Test hidden state initialization."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=64, num_layers=2, sequence_length=10, n_epochs=1, verbose=False
        )
        decoder.fit(X, y)

        h, c = decoder._model.init_hidden(batch_size=4, device=torch.device("cpu"))

        assert h.shape == (2, 4, 64)  # (num_layers, batch, hidden)
        assert c.shape == (2, 4, 64)

    def test_bidirectional(self, temporal_data):
        """Test bidirectional LSTM."""
        X, y = temporal_data
        decoder = BidirectionalLSTMDecoder(
            hidden_size=64, sequence_length=10, n_epochs=1, verbose=False
        )
        decoder.fit(X, y)

        x = torch.randn(4, 10, X.shape[1])
        output, _ = decoder._model(x)

        assert output.shape == (4, 10, y.shape[1])


@pytest.mark.unit
@pytest.mark.decoder
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMDecoder:
    """Tests for LSTMDecoder class."""

    def test_init(self):
        """Test decoder initialization."""
        decoder = LSTMDecoder(
            hidden_size=64,
            num_layers=2,
            sequence_length=10,
        )

        assert decoder.hidden_size == 64
        assert decoder.num_layers == 2
        assert decoder.sequence_length == 10
        assert not decoder.is_fitted

    def test_fit(self, temporal_data):
        """Test fitting the decoder."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            num_layers=1,
            sequence_length=10,
            n_epochs=5,  # Short for testing
            verbose=False,
        )
        decoder.fit(X, y)

        assert decoder.is_fitted
        assert decoder._model is not None
        assert len(decoder._train_losses) > 0

    def test_predict(self, temporal_data):
        """Test prediction."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=5,
            verbose=False,
        )
        decoder.fit(X, y)

        y_pred = decoder.predict(X)

        # Output has fewer samples due to sequence length
        expected_samples = X.shape[0] - 10 + 1
        assert y_pred.shape == (expected_samples, y.shape[1])

    def test_predict_not_fitted(self, temporal_data):
        """Test prediction raises error when not fitted."""
        X, _ = temporal_data
        decoder = LSTMDecoder()

        with pytest.raises(RuntimeError):
            decoder.predict(X)

    def test_predict_single(self, temporal_data):
        """Test single-step prediction for real-time use."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=5,
            verbose=False,
        )
        decoder.fit(X, y)

        # Reset hidden state
        decoder.reset_hidden()

        # Predict single samples
        predictions = []
        for i in range(20):
            pred = decoder.predict_single(X[i])
            predictions.append(pred)

        predictions = np.array(predictions)
        assert predictions.shape == (20, y.shape[1])

    def test_update(self, temporal_data):
        """Test online update."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=5,
            verbose=False,
        )
        decoder.fit(X[:300], y[:300])

        initial_update_count = decoder._update_count

        # Online update
        decoder.update(X[300:350], y[300:350])

        assert decoder._update_count > initial_update_count

    def test_reset_hidden(self, temporal_data):
        """Test hidden state reset."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        # Make predictions to set hidden state
        decoder.predict_single(X[0])
        assert decoder._hidden is not None

        # Reset
        decoder.reset_hidden()
        assert decoder._hidden is None

    def test_early_stopping(self, temporal_data):
        """Test early stopping works."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=100,  # Many epochs
            early_stopping_patience=3,  # But stop early
            verbose=False,
        )
        decoder.fit(X, y)

        # Should stop before 100 epochs
        assert len(decoder._train_losses) < 100

    def test_normalization(self, temporal_data):
        """Test that data is normalized internally."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        # Check normalization parameters are set
        assert decoder._X_mean is not None
        assert decoder._X_std is not None
        assert decoder._y_mean is not None
        assert decoder._y_std is not None

    def test_get_params(self, temporal_data):
        """Test get_params method."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=64,
            num_layers=3,
            sequence_length=15,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        params = decoder.get_params()

        assert params["hidden_size"] == 64
        assert params["num_layers"] == 3
        assert params["sequence_length"] == 15
        assert "train_losses" in params

    def test_dropout(self, temporal_data):
        """Test dropout configuration."""
        X, y = temporal_data
        decoder = LSTMDecoder(
            hidden_size=32,
            num_layers=2,  # Need >1 layer for dropout
            dropout=0.5,
            sequence_length=10,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        assert decoder.is_fitted


@pytest.mark.unit
@pytest.mark.decoder
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestBidirectionalLSTMDecoder:
    """Tests for BidirectionalLSTMDecoder class."""

    def test_init(self):
        """Test bidirectional decoder initialization."""
        decoder = BidirectionalLSTMDecoder(hidden_size=64)

        assert decoder.bidirectional is True
        assert decoder.name == "BiLSTM"

    def test_fit_and_predict(self, temporal_data):
        """Test fitting and prediction."""
        X, y = temporal_data
        decoder = BidirectionalLSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        y_pred = decoder.predict(X)
        assert y_pred.shape[1] == y.shape[1]

    def test_predict_single_not_supported(self, temporal_data):
        """Test that predict_single raises error for bidirectional."""
        X, y = temporal_data
        decoder = BidirectionalLSTMDecoder(
            hidden_size=32,
            sequence_length=10,
            n_epochs=3,
            verbose=False,
        )
        decoder.fit(X, y)

        with pytest.raises(NotImplementedError):
            decoder.predict_single(X[0])
