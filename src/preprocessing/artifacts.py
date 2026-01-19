"""
Artifact detection and removal for neural data.

Implements methods for detecting and handling artifacts in
neural recordings such as movement artifacts, saturation,
and outliers.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def detect_outliers(
    data: np.ndarray,
    threshold: float = 5.0,
    method: str = "zscore",
) -> np.ndarray:
    """
    Detect outlier samples in neural data.

    Args:
        data: Input data of shape (n_samples, n_channels).
        threshold: Threshold for outlier detection.
        method: Detection method ('zscore', 'mad', 'iqr').

    Returns:
        Boolean mask where True indicates an outlier sample.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if method == "zscore":
        z_scores = np.abs(stats.zscore(data, axis=0))
        outliers = np.any(z_scores > threshold, axis=1)

    elif method == "mad":
        # Median Absolute Deviation - more robust to outliers
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        mad[mad == 0] = 1.0
        modified_z = 0.6745 * (data - median) / mad
        outliers = np.any(np.abs(modified_z) > threshold, axis=1)

    elif method == "iqr":
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        outliers = np.any((data < lower) | (data > upper), axis=1)

    else:
        raise ValueError(f"Unknown method: {method}")

    return outliers


def remove_outliers(
    data: np.ndarray,
    threshold: float = 5.0,
    method: str = "zscore",
    replacement: str = "nan",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove or replace outlier samples.

    Args:
        data: Input data of shape (n_samples, n_channels).
        threshold: Threshold for outlier detection.
        method: Detection method ('zscore', 'mad', 'iqr').
        replacement: How to handle outliers ('nan', 'interpolate', 'clip', 'remove').

    Returns:
        Tuple of (cleaned_data, outlier_mask).
    """
    outlier_mask = detect_outliers(data, threshold, method)
    cleaned = data.copy()

    if replacement == "nan":
        cleaned[outlier_mask] = np.nan

    elif replacement == "interpolate":
        # Linear interpolation for outlier samples
        for ch in range(cleaned.shape[1] if cleaned.ndim > 1 else 1):
            col = cleaned[:, ch] if cleaned.ndim > 1 else cleaned
            valid_idx = np.where(~outlier_mask)[0]
            outlier_idx = np.where(outlier_mask)[0]

            if len(valid_idx) > 1 and len(outlier_idx) > 0:
                col[outlier_idx] = np.interp(outlier_idx, valid_idx, col[valid_idx])

    elif replacement == "clip":
        if method == "zscore":
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr

        cleaned = np.clip(cleaned, lower, upper)

    elif replacement == "remove":
        cleaned = cleaned[~outlier_mask]

    return cleaned, outlier_mask


def detect_saturation(
    data: np.ndarray,
    saturation_value: Optional[float] = None,
    min_consecutive: int = 3,
) -> np.ndarray:
    """
    Detect periods of signal saturation (clipping).

    Args:
        data: Input data of shape (n_samples, n_channels).
        saturation_value: Value indicating saturation. If None, uses max/min of data.
        min_consecutive: Minimum consecutive samples to count as saturation.

    Returns:
        Boolean mask where True indicates saturation.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if saturation_value is None:
        # Detect saturation at extremes
        data_min = np.min(data)
        data_max = np.max(data)
        at_min = data == data_min
        at_max = data == data_max
        at_extreme = at_min | at_max
    else:
        at_extreme = np.abs(data) >= np.abs(saturation_value)

    # Find consecutive runs
    saturation_mask = np.zeros(data.shape[0], dtype=bool)

    for ch in range(data.shape[1]):
        channel_extreme = at_extreme[:, ch]

        # Find runs of consecutive True values
        runs = np.diff(np.concatenate([[0], channel_extreme.astype(int), [0]]))
        run_starts = np.where(runs == 1)[0]
        run_ends = np.where(runs == -1)[0]
        run_lengths = run_ends - run_starts

        for start, length in zip(run_starts, run_lengths):
            if length >= min_consecutive:
                saturation_mask[start : start + length] = True

    return saturation_mask


def interpolate_bad_segments(
    data: np.ndarray,
    bad_mask: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate over bad data segments.

    Args:
        data: Input data of shape (n_samples,) or (n_samples, n_channels).
        bad_mask: Boolean mask where True indicates bad samples.
        method: Interpolation method ('linear', 'cubic', 'nearest').

    Returns:
        Data with bad segments interpolated.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    result = data.copy().astype(float)
    good_idx = np.where(~bad_mask)[0]
    bad_idx = np.where(bad_mask)[0]

    if len(good_idx) < 2 or len(bad_idx) == 0:
        return np.squeeze(result) if squeeze else result

    for ch in range(result.shape[1]):
        if method == "linear":
            result[bad_idx, ch] = np.interp(bad_idx, good_idx, result[good_idx, ch])
        elif method == "nearest":
            # Find nearest good index for each bad index
            for idx in bad_idx:
                nearest = good_idx[np.argmin(np.abs(good_idx - idx))]
                result[idx, ch] = result[nearest, ch]
        elif method == "cubic":
            from scipy.interpolate import interp1d

            if len(good_idx) >= 4:
                f = interp1d(good_idx, result[good_idx, ch], kind="cubic", fill_value="extrapolate")
                result[bad_idx, ch] = f(bad_idx)
            else:
                result[bad_idx, ch] = np.interp(bad_idx, good_idx, result[good_idx, ch])

    return np.squeeze(result) if squeeze else result
