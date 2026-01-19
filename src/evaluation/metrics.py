"""
Evaluation metrics for neural decoder performance.

Implements standard metrics for assessing decoder accuracy
including R², correlation, MSE, and per-dimension metrics.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np


def r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """
    Compute R² (coefficient of determination) score.

    R² = 1 - SS_res / SS_tot

    Args:
        y_true: Ground truth values of shape (n_samples,) or (n_samples, n_outputs).
        y_pred: Predicted values of same shape as y_true.
        multioutput: How to aggregate multiple outputs:
            - 'uniform_average': Average R² across outputs (default).
            - 'raw_values': Return R² for each output.
            - 'variance_weighted': Weight by output variance.

    Returns:
        R² score (float if averaged, array if raw_values).
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    if y_true.shape[0] == 1:
        y_true = y_true.T
        y_pred = y_pred.T

    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)

    # Handle zero variance case
    nonzero_mask = ss_tot != 0
    r2 = np.ones(ss_tot.shape)
    r2[nonzero_mask] = 1 - ss_res[nonzero_mask] / ss_tot[nonzero_mask]
    r2[~nonzero_mask] = 0.0

    if multioutput == "raw_values":
        return r2
    elif multioutput == "uniform_average":
        return float(np.mean(r2))
    elif multioutput == "variance_weighted":
        weights = ss_tot / np.sum(ss_tot) if np.sum(ss_tot) > 0 else np.ones_like(ss_tot)
        return float(np.sum(weights * r2))
    else:
        raise ValueError(f"Unknown multioutput: {multioutput}")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MSE value.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE value.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """
    Compute Pearson correlation coefficient.

    Args:
        y_true: Ground truth values of shape (n_samples,) or (n_samples, n_outputs).
        y_pred: Predicted values of same shape.
        multioutput: How to aggregate ('uniform_average' or 'raw_values').

    Returns:
        Correlation coefficient(s).
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    if y_true.shape[0] == 1:
        y_true = y_true.T
        y_pred = y_pred.T

    n_outputs = y_true.shape[1]
    corrs = np.zeros(n_outputs)

    for i in range(n_outputs):
        if np.std(y_true[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            corrs[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        else:
            corrs[i] = 0.0

    if multioutput == "raw_values":
        return corrs
    else:
        return float(np.mean(corrs))


def snr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    SNR = 10 * log10(var(y_true) / var(y_true - y_pred))

    Args:
        y_true: Ground truth signal.
        y_pred: Predicted signal.

    Returns:
        SNR in decibels.
    """
    signal_var = np.var(y_true)
    noise_var = np.var(y_true - y_pred)

    if noise_var == 0:
        return float("inf")
    if signal_var == 0:
        return float("-inf")

    return float(10 * np.log10(signal_var / noise_var))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all standard evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary of metric names to values.
    """
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "snr_db": snr(y_true, y_pred),
    }


def compute_metrics_per_dimension(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dim_names: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each output dimension separately.

    Args:
        y_true: Ground truth values of shape (n_samples, n_outputs).
        y_pred: Predicted values of same shape.
        dim_names: Optional names for dimensions (e.g., ['x', 'y', 'z']).

    Returns:
        Nested dictionary: {dimension: {metric: value}}.
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    if y_true.shape[0] == 1:
        y_true = y_true.T
        y_pred = y_pred.T

    n_outputs = y_true.shape[1]

    if dim_names is None:
        dim_names = [f"dim_{i}" for i in range(n_outputs)]

    results = {}
    for i, name in enumerate(dim_names):
        results[name] = compute_all_metrics(y_true[:, i], y_pred[:, i])

    return results
