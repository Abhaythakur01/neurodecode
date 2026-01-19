"""
Spike detection algorithms for neural data.

Implements threshold-based and template-matching spike detection
methods for extracting spike times from continuous neural recordings.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal as scipy_signal


def threshold_crossing(
    data: np.ndarray,
    threshold: float,
    fs: float,
    direction: str = "negative",
    refractory_ms: float = 1.0,
) -> np.ndarray:
    """
    Detect spikes using threshold crossing.

    Args:
        data: Continuous neural signal (1D array).
        threshold: Voltage threshold for spike detection.
        fs: Sampling frequency in Hz.
        direction: 'negative', 'positive', or 'both'.
        refractory_ms: Refractory period in milliseconds.

    Returns:
        Array of spike times (in samples).
    """
    refractory_samples = int(refractory_ms * fs / 1000)

    if direction == "negative":
        crossings = np.where(np.diff((data < threshold).astype(int)) == 1)[0]
    elif direction == "positive":
        crossings = np.where(np.diff((data > threshold).astype(int)) == 1)[0]
    elif direction == "both":
        crossings = np.where(np.diff((np.abs(data) > np.abs(threshold)).astype(int)) == 1)[0]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Apply refractory period
    if len(crossings) == 0:
        return crossings

    spike_times = [crossings[0]]
    for t in crossings[1:]:
        if t - spike_times[-1] >= refractory_samples:
            spike_times.append(t)

    return np.array(spike_times)


def detect_spikes_multichannel(
    data: np.ndarray,
    fs: float,
    threshold_std: float = 4.0,
    direction: str = "negative",
    refractory_ms: float = 1.0,
) -> List[np.ndarray]:
    """
    Detect spikes on multiple channels using adaptive thresholding.

    Uses median absolute deviation (MAD) for robust threshold estimation.

    Args:
        data: Multi-channel data of shape (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        threshold_std: Number of standard deviations for threshold.
        direction: 'negative', 'positive', or 'both'.
        refractory_ms: Refractory period in milliseconds.

    Returns:
        List of spike time arrays, one per channel.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_channels = data.shape[1]
    spike_times_list = []

    for ch in range(n_channels):
        channel_data = data[:, ch]

        # Robust threshold estimation using MAD
        median = np.median(channel_data)
        mad = np.median(np.abs(channel_data - median))
        sigma = mad / 0.6745  # Convert MAD to standard deviation estimate

        threshold = median - threshold_std * sigma if direction == "negative" else median + threshold_std * sigma

        if direction == "both":
            threshold = threshold_std * sigma

        spikes = threshold_crossing(channel_data, threshold, fs, direction, refractory_ms)
        spike_times_list.append(spikes)

    return spike_times_list


def extract_waveforms(
    data: np.ndarray,
    spike_times: np.ndarray,
    pre_samples: int = 10,
    post_samples: int = 20,
) -> np.ndarray:
    """
    Extract spike waveforms around detected spike times.

    Args:
        data: Continuous neural signal (1D array).
        spike_times: Array of spike times (in samples).
        pre_samples: Number of samples before spike peak.
        post_samples: Number of samples after spike peak.

    Returns:
        Array of waveforms of shape (n_spikes, pre_samples + post_samples).
    """
    waveform_length = pre_samples + post_samples
    n_spikes = len(spike_times)

    # Filter valid spike times (not too close to edges)
    valid_mask = (spike_times >= pre_samples) & (spike_times < len(data) - post_samples)
    valid_times = spike_times[valid_mask]

    waveforms = np.zeros((len(valid_times), waveform_length))

    for i, t in enumerate(valid_times):
        waveforms[i] = data[t - pre_samples : t + post_samples]

    return waveforms


def compute_spike_features(waveforms: np.ndarray) -> dict:
    """
    Compute features from spike waveforms for sorting/clustering.

    Args:
        waveforms: Array of waveforms of shape (n_spikes, waveform_length).

    Returns:
        Dictionary of spike features.
    """
    if len(waveforms) == 0:
        return {
            "peak_amplitude": np.array([]),
            "trough_amplitude": np.array([]),
            "peak_to_trough": np.array([]),
            "width": np.array([]),
        }

    features = {}

    # Peak and trough amplitudes
    features["peak_amplitude"] = np.max(waveforms, axis=1)
    features["trough_amplitude"] = np.min(waveforms, axis=1)
    features["peak_to_trough"] = features["peak_amplitude"] - features["trough_amplitude"]

    # Width at half maximum
    half_max = features["trough_amplitude"] / 2
    widths = []
    for i, wf in enumerate(waveforms):
        below_half = wf < half_max[i]
        if np.any(below_half):
            first_cross = np.argmax(below_half)
            last_cross = len(below_half) - np.argmax(below_half[::-1]) - 1
            widths.append(last_cross - first_cross)
        else:
            widths.append(0)
    features["width"] = np.array(widths)

    return features


def spike_times_to_binary(
    spike_times: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Convert spike times to binary spike train.

    Args:
        spike_times: Array of spike times (in samples).
        n_samples: Total number of samples in the recording.

    Returns:
        Binary array where 1 indicates a spike.
    """
    binary = np.zeros(n_samples, dtype=np.int8)
    valid_times = spike_times[(spike_times >= 0) & (spike_times < n_samples)]
    binary[valid_times.astype(int)] = 1
    return binary


def binary_to_spike_times(binary: np.ndarray) -> np.ndarray:
    """
    Convert binary spike train to spike times.

    Args:
        binary: Binary array where 1 indicates a spike.

    Returns:
        Array of spike times (in samples).
    """
    return np.where(binary > 0)[0]
