"""
Unit tests for HMM decoder.
"""

import numpy as np
import pytest

from src.decoders.classic.hmm import DiscreteHMM, GaussianHMM


@pytest.fixture
def state_sequence_data():
    """Generate data with clear state structure."""
    np.random.seed(42)
    n_samples = 300
    n_features = 10

    # Generate states (alternating blocks)
    states = np.zeros(n_samples, dtype=int)
    states[100:200] = 1

    # Generate observations based on states
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if states[i] == 0:
            X[i] = np.random.randn(n_features) + np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            X[i] = np.random.randn(n_features) + np.array([0, 0, 0, 0, 0, 2, 0, 0, 0, 0])

    return X, states


@pytest.fixture
def multistate_data():
    """Generate data with multiple states."""
    np.random.seed(42)
    n_samples = 400
    n_features = 5
    n_states = 4

    # Generate states
    states = np.zeros(n_samples, dtype=int)
    for i in range(n_states):
        start = i * 100
        end = (i + 1) * 100
        states[start:end] = i

    # Generate observations
    X = np.zeros((n_samples, n_features))
    means = np.random.randn(n_states, n_features) * 3
    for i in range(n_samples):
        X[i] = means[states[i]] + np.random.randn(n_features) * 0.5

    return X, states


@pytest.mark.unit
@pytest.mark.decoder
class TestGaussianHMM:
    """Tests for GaussianHMM class."""

    def test_init(self):
        """Test HMM initialization."""
        hmm = GaussianHMM(n_states=3, covariance_type="diag")

        assert hmm.n_states == 3
        assert hmm.covariance_type == "diag"
        assert not hmm.is_fitted

    def test_fit(self, state_sequence_data):
        """Test fitting the HMM."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=20, random_state=42)
        hmm.fit(X)

        assert hmm.is_fitted
        assert hmm.startprob_ is not None
        assert hmm.transmat_ is not None
        assert hmm.means_ is not None
        assert hmm.covars_ is not None
        assert len(hmm._log_likelihood_history) > 0

    def test_fit_with_labels(self, state_sequence_data):
        """Test fitting with supervised initialization."""
        X, y = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=10, random_state=42)
        hmm.fit(X, y)

        assert hmm.is_fitted

    def test_predict(self, state_sequence_data):
        """Test state prediction."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=30, random_state=42)
        hmm.fit(X)

        states = hmm.predict(X)

        assert states.shape == (len(X),)
        assert set(states).issubset({0, 1})

    def test_predict_not_fitted(self, state_sequence_data):
        """Test prediction raises error when not fitted."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2)

        with pytest.raises(RuntimeError):
            hmm.predict(X)

    def test_predict_proba(self, state_sequence_data):
        """Test probability prediction."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=20, random_state=42)
        hmm.fit(X)

        proba = hmm.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)

    def test_score(self, state_sequence_data):
        """Test log-likelihood scoring."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=20, random_state=42)
        hmm.fit(X)

        ll = hmm.score(X)

        assert isinstance(ll, float)
        assert np.isfinite(ll)

    def test_sample(self, state_sequence_data):
        """Test sampling from fitted model."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=20, random_state=42)
        hmm.fit(X)

        observations, states = hmm.sample(50)

        assert observations.shape == (50, X.shape[1])
        assert states.shape == (50,)
        assert set(states).issubset({0, 1})

    def test_covariance_types(self, state_sequence_data):
        """Test different covariance types."""
        X, _ = state_sequence_data

        for cov_type in ["diag", "spherical", "full"]:
            hmm = GaussianHMM(n_states=2, covariance_type=cov_type, n_iter=10, random_state=42)
            hmm.fit(X)
            assert hmm.is_fitted

    def test_multiple_states(self, multistate_data):
        """Test with multiple states."""
        X, _ = multistate_data
        hmm = GaussianHMM(n_states=4, n_iter=30, random_state=42)
        hmm.fit(X)

        states = hmm.predict(X)

        assert len(np.unique(states)) <= 4

    def test_evaluate(self, state_sequence_data):
        """Test evaluation metrics."""
        X, y = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=30, random_state=42)
        hmm.fit(X, y)

        metrics = hmm.evaluate(X, y)

        assert "accuracy" in metrics
        assert "log_likelihood" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_get_params(self, state_sequence_data):
        """Test get_params method."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=3, covariance_type="full", n_iter=10)
        hmm.fit(X)

        params = hmm.get_params()

        assert params["n_states"] == 3
        assert params["covariance_type"] == "full"
        assert "log_likelihood_history" in params

    def test_convergence(self, state_sequence_data):
        """Test that log-likelihood improves during training."""
        X, _ = state_sequence_data
        hmm = GaussianHMM(n_states=2, n_iter=50, tol=0, random_state=42)
        hmm.fit(X)

        # Log-likelihood should generally improve
        ll_history = hmm._log_likelihood_history
        # Allow some fluctuation but overall trend should be up
        assert ll_history[-1] >= ll_history[0]


@pytest.mark.unit
@pytest.mark.decoder
class TestDiscreteHMM:
    """Tests for DiscreteHMM class."""

    @pytest.fixture
    def discrete_data(self):
        """Generate discrete observation data."""
        np.random.seed(42)
        n_samples = 200
        n_symbols = 5

        # Generate states
        states = np.zeros(n_samples, dtype=int)
        states[100:] = 1

        # Generate observations (different symbol distributions per state)
        X = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if states[i] == 0:
                X[i] = np.random.choice(n_symbols, p=[0.5, 0.3, 0.1, 0.05, 0.05])
            else:
                X[i] = np.random.choice(n_symbols, p=[0.05, 0.05, 0.1, 0.3, 0.5])

        return X, states

    def test_init(self):
        """Test discrete HMM initialization."""
        hmm = DiscreteHMM(n_states=3, n_symbols=10)

        assert hmm.n_states == 3
        assert hmm.n_symbols == 10
        assert not hmm.is_fitted

    def test_fit(self, discrete_data):
        """Test fitting discrete HMM."""
        X, _ = discrete_data
        hmm = DiscreteHMM(n_states=2, n_symbols=5, n_iter=20, random_state=42)
        hmm.fit(X)

        assert hmm.is_fitted
        assert hmm.startprob_ is not None
        assert hmm.transmat_ is not None
        assert hmm.emissionprob_ is not None

    def test_predict(self, discrete_data):
        """Test state prediction."""
        X, _ = discrete_data
        hmm = DiscreteHMM(n_states=2, n_symbols=5, n_iter=20, random_state=42)
        hmm.fit(X)

        states = hmm.predict(X)

        assert states.shape == (len(X),)
        assert set(states).issubset({0, 1})

    def test_get_params(self, discrete_data):
        """Test get_params method."""
        X, _ = discrete_data
        hmm = DiscreteHMM(n_states=2, n_symbols=5, n_iter=10)
        hmm.fit(X)

        params = hmm.get_params()

        assert params["n_states"] == 2
        assert params["n_symbols"] == 5
