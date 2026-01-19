"""
Firing rate computation from spike trains.

Implements various methods for converting spike trains to
continuous firing rate estimates suitable for decoding.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage


def compute_firing_rates(
    spike_times: Union[np.ndarray, List[np.ndarray]],
    bin_size: float = 0.02,
    duration: Optional[float] = None,
    fs: float = 1000.0,
) -> np.ndarray:
    """
    Compute binned firing rates from spike times.

    Args:
        spike_times: Spike times in seconds (1D array for single neuron,
            or list of arrays for multiple neurons).
        bin_size: Bin width in seconds (default 20ms).
        duration: Total recording duration in seconds. If None, inferred from spike times.
        fs: Original sampling frequency (used if spike_times are in samples).

    Returns:
        Firing rates array of shape (n_bins,) or (n_bins, n_neurons).
    """
    # Handle single neuron case
    if isinstance(spike_times, np.ndarray) and spike_times.ndim == 1:
        spike_times = [spike_times]

    n_neurons = len(spike_times)

    # Determine duration
    if duration is None:
        max_spike = max(
            (st.max() if len(st) > 0 else 0 for st in spike_times),
            default=0,
        )
        duration = max_spike + bin_size

    # Create bins
    n_bins = int(np.ceil(duration / bin_size))
    bins = np.arange(n_bins + 1) * bin_size

    # Compute firing rates for each neuron
    firing_rates = np.zeros((n_bins, n_neurons))

    for i, st in enumerate(spike_times):
        if len(st) > 0:
            counts, _ = np.histogram(st, bins=bins)
            firing_rates[:, i] = counts / bin_size  # Convert to Hz

    return np.squeeze(firing_rates)


def bin_spikes(
    spike_train: np.ndarray,
    bin_size_samples: int,
    n_bins: Optional[int] = None,
) -> np.ndarray:
    """
    Bin a binary spike train into spike counts.

    Args:
        spike_train: Binary spike train (0s and 1s).
        bin_size_samples: Number of samples per bin.
        n_bins: Number of output bins. If None, computed from data.

    Returns:
        Array of spike counts per bin.
    """
    n_samples = len(spike_train)

    if n_bins is None:
        n_bins = n_samples // bin_size_samples

    # Truncate to fit bins exactly
    n_samples_use = n_bins * bin_size_samples
    spike_train = spike_train[:n_samples_use]

    # Reshape and sum
    binned = spike_train.reshape(n_bins, bin_size_samples).sum(axis=1)

    return binned


def gaussian_smooth_firing_rate(
    firing_rates: np.ndarray,
    sigma_bins: float = 2.0,
    axis: int = 0,
) -> np.ndarray:
    """
    Smooth firing rates with Gaussian kernel.

    Args:
        firing_rates: Firing rate array.
        sigma_bins: Standard deviation of Gaussian in bins.
        axis: Axis along which to smooth.

    Returns:
        Smoothed firing rates.
    """
    return ndimage.gaussian_filter1d(firing_rates, sigma=sigma_bins, axis=axis)


def causal_smooth_firing_rate(
    firing_rates: np.ndarray,
    tau_bins: float = 3.0,
    axis: int = 0,
) -> np.ndarray:
    """
    Smooth firing rates with causal exponential kernel.

    Uses only past data points (suitable for real-time applications).

    Args:
        firing_rates: Firing rate array.
        tau_bins: Time constant in bins.
        axis: Axis along which to smooth.

    Returns:
        Smoothed firing rates.
    """
    if tau_bins <= 0:
        return firing_rates

    # Create exponential kernel
    kernel_length = int(5 * tau_bins)
    t = np.arange(kernel_length)
    kernel = np.exp(-t / tau_bins)
    kernel = kernel / kernel.sum()

    # Move axis to first position for convolution
    moved = np.moveaxis(firing_rates, axis, 0)
    original_shape = moved.shape

    if moved.ndim == 1:
        smoothed = np.convolve(moved, kernel, mode="full")[:len(moved)]
    else:
        # Flatten all but first axis
        flat = moved.reshape(moved.shape[0], -1)
        smoothed = np.zeros_like(flat)
        for i in range(flat.shape[1]):
            smoothed[:, i] = np.convolve(flat[:, i], kernel, mode="full")[: flat.shape[0]]
        smoothed = smoothed.reshape(original_shape)

    return np.moveaxis(smoothed, 0, axis)


def spike_train_to_firing_rate_multichannel(
    spike_trains: np.ndarray,
    bin_size_samples: int,
    smooth: bool = True,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    Convert multi-channel binary spike trains to firing rates.

    Args:
        spike_trains: Binary spike trains of shape (n_samples, n_neurons).
        bin_size_samples: Number of samples per bin.
        smooth: Whether to apply Gaussian smoothing.
        smooth_sigma: Smoothing kernel width in bins.

    Returns:
        Firing rates of shape (n_bins, n_neurons).
    """
    if spike_trains.ndim == 1:
        spike_trains = spike_trains.reshape(-1, 1)

    n_samples, n_neurons = spike_trains.shape
    n_bins = n_samples // bin_size_samples

    # Bin each channel
    firing_rates = np.zeros((n_bins, n_neurons))
    for i in range(n_neurons):
        firing_rates[:, i] = bin_spikes(spike_trains[:, i], bin_size_samples, n_bins)

    # Smooth if requested
    if smooth:
        firing_rates = gaussian_smooth_firing_rate(firing_rates, smooth_sigma, axis=0)

    return firing_rates


def compute_instantaneous_firing_rate(
    spike_times: np.ndarray,
    eval_times: np.ndarray,
    bandwidth: float = 0.02,
) -> np.ndarray:
    """
    Compute instantaneous firing rate using kernel density estimation.

    Args:
        spike_times: Array of spike times in seconds.
        eval_times: Times at which to evaluate the firing rate.
        bandwidth: Kernel bandwidth in seconds.

    Returns:
        Instantaneous firing rate at each eval_time.
    """
    if len(spike_times) == 0:
        return np.zeros_like(eval_times)

    # Gaussian kernel evaluation
    rates = np.zeros_like(eval_times, dtype=float)

    for t_spike in spike_times:
        # Add contribution from each spike
        contribution = np.exp(-0.5 * ((eval_times - t_spike) / bandwidth) ** 2)
        rates += contribution

    # Normalize by bandwidth
    rates /= bandwidth * np.sqrt(2 * np.pi)

    return rates
