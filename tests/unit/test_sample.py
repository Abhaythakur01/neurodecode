"""
Sample test file to demonstrate testing structure
"""

import numpy as np
import pytest


@pytest.mark.unit
def test_sample_neural_data_fixture(sample_neural_data):
    """Test that the sample neural data fixture works correctly."""
    X, y = sample_neural_data

    # Check shapes
    assert X.shape[0] == 100  # n_samples
    assert X.shape[1] == 50  # n_neurons
    assert X.shape[2] == 20  # n_timebins

    assert y.shape[0] == 100  # n_samples
    assert y.shape[1] == 2  # n_dimensions

    # Check data types
    assert X.dtype == float
    assert y.dtype == float


@pytest.mark.unit
def test_sample_spike_train_fixture(sample_spike_train):
    """Test that the sample spike train fixture works correctly."""
    assert len(sample_spike_train) == 50
    assert np.all(sample_spike_train >= 0)
    assert np.all(sample_spike_train <= 1)
    # Check that spikes are sorted
    assert np.all(np.diff(sample_spike_train) >= 0)


@pytest.mark.unit
def test_decoder_config_fixture(decoder_config):
    """Test that the decoder config fixture works correctly."""
    assert "bin_size" in decoder_config
    assert "n_neurons" in decoder_config
    assert decoder_config["n_neurons"] == 50
    assert decoder_config["n_dimensions"] == 2


@pytest.mark.unit
def test_numpy_operations():
    """Simple test to verify numpy is working."""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


@pytest.mark.unit
def test_basic_math():
    """Basic sanity check test."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


# Example of a parametrized test
@pytest.mark.unit
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 1),  # 0! = 1
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
    ],
)
def test_factorial(n, expected):
    """Test factorial calculation (placeholder)."""
    import math

    assert math.factorial(n) == expected


# Example of testing exceptions
@pytest.mark.unit
def test_division_by_zero():
    """Test that division by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0


# Example of a slow test
@pytest.mark.slow
@pytest.mark.unit
def test_slow_operation():
    """Example of a test marked as slow."""
    # This test would be skipped with: pytest -m "not slow"
    import time

    time.sleep(0.1)  # Simulate slow operation
    assert True
