"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
from typing import Tuple


@pytest.fixture
def sample_neural_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample neural data for testing.

    Returns:
        Tuple of (X, y) where:
        - X: Neural activity (n_samples, n_neurons, n_timebins)
        - y: Movement kinematics (n_samples, n_dimensions)
    """
    n_samples = 100
    n_neurons = 50
    n_timebins = 20
    n_dimensions = 2

    # Simulate neural data with some structure
    np.random.seed(42)
    X = np.random.poisson(lam=5, size=(n_samples, n_neurons, n_timebins)).astype(float)

    # Simulate movement data
    y = np.random.randn(n_samples, n_dimensions)

    return X, y


@pytest.fixture
def sample_spike_train() -> np.ndarray:
    """
    Generate sample spike train data.

    Returns:
        Spike times array (n_spikes,)
    """
    np.random.seed(42)
    # Generate spike times in 1 second window
    n_spikes = 50
    spike_times = np.sort(np.random.rand(n_spikes))
    return spike_times


@pytest.fixture
def sample_firing_rates() -> np.ndarray:
    """
    Generate sample firing rate data.

    Returns:
        Firing rates (n_samples, n_neurons)
    """
    np.random.seed(42)
    n_samples = 100
    n_neurons = 50
    firing_rates = np.random.poisson(lam=10, size=(n_samples, n_neurons)).astype(float)
    return firing_rates


@pytest.fixture
def decoder_config() -> dict:
    """
    Sample decoder configuration.

    Returns:
        Configuration dictionary
    """
    return {
        "bin_size": 0.02,  # 20ms bins
        "lag": 0,
        "n_neurons": 50,
        "n_dimensions": 2,
        "learning_rate": 0.001,
        "batch_size": 32,
    }


@pytest.fixture
def temp_model_path(tmp_path):
    """
    Temporary path for saving models during tests.

    Args:
        tmp_path: pytest fixture for temporary directory

    Returns:
        Path to temporary model directory
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# Markers
def pytest_configure(config):
    """
    Register custom markers.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Skip slow tests by default in CI
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection.
    """
    if config.getoption("-m") == "not slow":
        skip_slow = pytest.mark.skip(reason="Slow test skipped")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
