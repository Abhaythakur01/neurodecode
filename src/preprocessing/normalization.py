"""
Normalization functions for neural data preprocessing.

Implements various normalization strategies including z-score,
min-max scaling, and robust scaling.
"""

from typing import Optional, Tuple

import numpy as np


class Normalizer:
    """
    Normalizer for neural data with fit/transform interface.

    Supports z-score, min-max, and robust normalization methods.
    Normalization parameters are computed per-neuron (column).
    """

    def __init__(self, method: str = "zscore"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('zscore', 'minmax', 'robust').
        """
        if method not in ("zscore", "minmax", "robust"):
            raise ValueError(f"Unknown method: {method}. Use 'zscore', 'minmax', or 'robust'.")

        self.method = method
        self.is_fitted = False
        self._params: dict = {}

    def fit(self, X: np.ndarray) -> "Normalizer":
        """
        Compute normalization parameters from training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self: Fitted normalizer.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "zscore":
            self._params["mean"] = np.mean(X, axis=0)
            self._params["std"] = np.std(X, axis=0)
            # Avoid division by zero
            self._params["std"][self._params["std"] == 0] = 1.0

        elif self.method == "minmax":
            self._params["min"] = np.min(X, axis=0)
            self._params["max"] = np.max(X, axis=0)
            self._params["range"] = self._params["max"] - self._params["min"]
            self._params["range"][self._params["range"] == 0] = 1.0

        elif self.method == "robust":
            self._params["median"] = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self._params["iqr"] = q75 - q25
            self._params["iqr"][self._params["iqr"] == 0] = 1.0

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to data.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Normalized data.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "zscore":
            return (X - self._params["mean"]) / self._params["std"]

        elif self.method == "minmax":
            return (X - self._params["min"]) / self._params["range"]

        elif self.method == "robust":
            return (X - self._params["median"]) / self._params["iqr"]

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit normalizer and transform data in one step.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            Normalized data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse the normalization.

        Args:
            X: Normalized data of shape (n_samples, n_features).

        Returns:
            Original scale data.
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.method == "zscore":
            return X * self._params["std"] + self._params["mean"]

        elif self.method == "minmax":
            return X * self._params["range"] + self._params["min"]

        elif self.method == "robust":
            return X * self._params["iqr"] + self._params["median"]

        return X

    def get_params(self) -> dict:
        """Get normalization parameters."""
        return {"method": self.method, "is_fitted": self.is_fitted, **self._params}


def zscore_normalize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize data (zero mean, unit variance).

    Args:
        X: Input data array.
        mean: Pre-computed mean (if None, computed from X).
        std: Pre-computed std (if None, computed from X).
        axis: Axis along which to normalize.

    Returns:
        Tuple of (normalized_data, mean, std).
    """
    if mean is None:
        mean = np.mean(X, axis=axis, keepdims=True)
    if std is None:
        std = np.std(X, axis=axis, keepdims=True)
        std[std == 0] = 1.0

    normalized = (X - mean) / std
    return normalized, np.squeeze(mean), np.squeeze(std)


def soft_normalize(
    X: np.ndarray,
    baseline_period: Optional[Tuple[int, int]] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Soft normalization using baseline period.

    Commonly used in neural data analysis to normalize relative to
    a baseline period (e.g., pre-stimulus).

    Args:
        X: Input data array.
        baseline_period: Tuple of (start_idx, end_idx) for baseline.
            If None, uses first 10% of data.
        axis: Axis along which to normalize.

    Returns:
        Normalized data.
    """
    if baseline_period is None:
        n = X.shape[axis]
        baseline_period = (0, max(1, int(n * 0.1)))

    start, end = baseline_period

    if axis == 0:
        baseline = X[start:end]
    elif axis == 1:
        baseline = X[:, start:end]
    else:
        # Generic slicing
        slices = [slice(None)] * X.ndim
        slices[axis] = slice(start, end)
        baseline = X[tuple(slices)]

    baseline_mean = np.mean(baseline, axis=axis, keepdims=True)
    baseline_std = np.std(baseline, axis=axis, keepdims=True)
    baseline_std[baseline_std == 0] = 1.0

    return (X - baseline_mean) / baseline_std
