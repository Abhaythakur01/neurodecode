"""
Spectral feature extraction for neural data.

Implements power spectral density estimation and band power
computation for LFP and other continuous neural signals.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

# Standard frequency bands for neural analysis
FREQUENCY_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low_gamma": (30, 50),
    "high_gamma": (50, 100),
    "gamma": (30, 100),
}


def compute_psd(
    data: np.ndarray,
    fs: float,
    method: str = "welch",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density of neural signal.

    Args:
        data: Input signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        method: PSD method ('welch', 'periodogram').
        nperseg: Length of each segment for Welch method.
        noverlap: Number of overlapping samples.

    Returns:
        Tuple of (frequencies, psd) arrays.
    """
    if nperseg is None:
        nperseg = min(256, len(data))

    if noverlap is None:
        noverlap = nperseg // 2

    if method == "welch":
        freqs, psd = signal.welch(
            data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=0,
        )
    elif method == "periodogram":
        freqs, psd = signal.periodogram(data, fs=fs, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return freqs, psd


def compute_band_power(
    data: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Compute power in specified frequency bands.

    Args:
        data: Input signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        bands: Dictionary of band names to (low_freq, high_freq) tuples.
            If None, uses standard neural frequency bands.
        relative: If True, return relative (normalized) band power.

    Returns:
        Dictionary mapping band names to power values.
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    freqs, psd = compute_psd(data, fs)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    band_powers = {}
    total_power = np.sum(psd, axis=0) * freq_res if relative else 1.0

    for band_name, (low, high) in bands.items():
        # Find frequency indices in band
        idx = np.logical_and(freqs >= low, freqs <= high)

        if psd.ndim == 1:
            band_power = np.sum(psd[idx]) * freq_res
        else:
            band_power = np.sum(psd[idx, :], axis=0) * freq_res

        if relative:
            band_power = band_power / total_power

        band_powers[band_name] = band_power

    return band_powers


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram of neural signal.

    Args:
        data: Input signal of shape (n_samples,).
        fs: Sampling frequency in Hz.
        nperseg: Length of each segment.
        noverlap: Number of overlapping samples.
        freq_range: Tuple of (min_freq, max_freq) to return.

    Returns:
        Tuple of (frequencies, times, spectrogram).
    """
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = signal.spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    if freq_range is not None:
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[freq_mask]
        Sxx = Sxx[freq_mask, :]

    return freqs, times, Sxx


def compute_band_power_timeseries(
    data: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    window_size: float = 0.5,
    step_size: float = 0.02,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute band power as a function of time.

    Args:
        data: Input signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        bands: Dictionary of band names to frequency ranges.
        window_size: Window size in seconds.
        step_size: Step size in seconds.

    Returns:
        Tuple of (time_points, band_powers_dict).
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    # Compute time points
    n_windows = (n_samples - window_samples) // step_samples + 1
    times = np.arange(n_windows) * step_size + window_size / 2

    # Initialize output
    band_powers = {band: np.zeros((n_windows, n_channels)) for band in bands}

    # Compute band power for each window
    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        window_data = data[start:end, :]

        window_powers = compute_band_power(window_data, fs, bands)

        for band in bands:
            band_powers[band][i, :] = window_powers[band]

    return times, band_powers


def extract_spectral_features(
    data: np.ndarray,
    fs: float,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extract spectral features for decoding.

    Computes band powers and returns as feature vector.

    Args:
        data: Input signal of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        bands: Dictionary of band names to frequency ranges.

    Returns:
        Feature array of shape (n_bands,) or (n_bands * n_channels,).
    """
    band_powers = compute_band_power(data, fs, bands, relative=True)

    # Stack into feature vector
    features = np.concatenate(
        [np.atleast_1d(band_powers[band]) for band in sorted(band_powers.keys())]
    )

    return features
