"""
Signal filtering functions for neural data preprocessing.

Implements bandpass, lowpass, highpass, and notch filters
using scipy's signal processing capabilities.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import signal


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to neural data.

    Args:
        data: Input signal array.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 4).
        axis: Axis along which to filter (default -1, last axis).

    Returns:
        Filtered signal array.
    """
    nyquist = fs / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist

    # Clip to valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        raise ValueError(f"low_freq ({low_freq}) must be less than high_freq ({high_freq})")

    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data, axis=axis)


def lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter to neural data.

    Args:
        data: Input signal array.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 4).
        axis: Axis along which to filter (default -1).

    Returns:
        Filtered signal array.
    """
    nyquist = fs / 2.0
    normalized_cutoff = min(cutoff / nyquist, 0.999)

    b, a = signal.butter(order, normalized_cutoff, btype="low")
    return signal.filtfilt(b, a, data, axis=axis)


def highpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply Butterworth highpass filter to neural data.

    Args:
        data: Input signal array.
        cutoff: Cutoff frequency in Hz.
        fs: Sampling frequency in Hz.
        order: Filter order (default 4).
        axis: Axis along which to filter (default -1).

    Returns:
        Filtered signal array.
    """
    nyquist = fs / 2.0
    normalized_cutoff = max(cutoff / nyquist, 0.001)

    b, a = signal.butter(order, normalized_cutoff, btype="high")
    return signal.filtfilt(b, a, data, axis=axis)


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply notch filter to remove line noise (e.g., 50/60 Hz).

    Args:
        data: Input signal array.
        notch_freq: Frequency to remove in Hz.
        fs: Sampling frequency in Hz.
        quality_factor: Quality factor (higher = narrower notch).
        axis: Axis along which to filter (default -1).

    Returns:
        Filtered signal array.
    """
    nyquist = fs / 2.0
    if notch_freq >= nyquist:
        # Can't notch at or above Nyquist, return unchanged
        return data

    b, a = signal.iirnotch(notch_freq, quality_factor, fs)
    return signal.filtfilt(b, a, data, axis=axis)


def remove_line_noise(
    data: np.ndarray,
    fs: float,
    line_freq: float = 60.0,
    harmonics: int = 3,
    axis: int = -1,
) -> np.ndarray:
    """
    Remove power line noise and its harmonics.

    Args:
        data: Input signal array.
        fs: Sampling frequency in Hz.
        line_freq: Power line frequency (50 or 60 Hz).
        harmonics: Number of harmonics to remove.
        axis: Axis along which to filter.

    Returns:
        Filtered signal array.
    """
    result = data.copy()
    nyquist = fs / 2.0

    for h in range(1, harmonics + 1):
        freq = line_freq * h
        if freq < nyquist:
            result = notch_filter(result, freq, fs, axis=axis)

    return result
