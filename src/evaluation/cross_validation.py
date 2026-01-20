"""
Cross-validation for neural decoder evaluation.

Implements temporal cross-validation schemes appropriate for
time-series neural data (no shuffling, preserves temporal order).
"""

from typing import Dict, Generator, List, Optional, Tuple, Type

import numpy as np

from src.decoders.base import BaseDecoder
from src.evaluation.metrics import compute_all_metrics


def temporal_split(
    n_samples: int,
    n_splits: int = 5,
    test_size: Optional[float] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train/test indices for temporal cross-validation.

    Uses expanding window: train on all data before test fold.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of splits.
        test_size: Size of test set as fraction (default: 1/n_splits).

    Yields:
        Tuples of (train_indices, test_indices).
    """
    if test_size is None:
        test_size = 1.0 / n_splits

    test_samples = int(n_samples * test_size)
    min_train_samples = test_samples  # At least as many training samples as test

    for i in range(n_splits):
        test_start = min_train_samples + i * test_samples
        test_end = test_start + test_samples

        if test_end > n_samples:
            break

        train_idx = np.arange(test_start)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, test_idx


def sliding_window_split(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train/test indices using sliding window.

    Args:
        n_samples: Total number of samples.
        train_size: Number of samples in training window.
        test_size: Number of samples in test window.
        step: Step size between windows (default: test_size).

    Yields:
        Tuples of (train_indices, test_indices).
    """
    if step is None:
        step = test_size

    start = 0
    while start + train_size + test_size <= n_samples:
        train_idx = np.arange(start, start + train_size)
        test_idx = np.arange(start + train_size, start + train_size + test_size)

        yield train_idx, test_idx
        start += step


def blocked_split(
    n_samples: int,
    n_blocks: int = 5,
    test_blocks: int = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train/test indices using blocked cross-validation.

    Divides data into blocks and uses each block as test set.

    Args:
        n_samples: Total number of samples.
        n_blocks: Number of blocks.
        test_blocks: Number of consecutive blocks for test.

    Yields:
        Tuples of (train_indices, test_indices).
    """
    block_size = n_samples // n_blocks
    indices = np.arange(n_samples)

    for i in range(n_blocks - test_blocks + 1):
        test_start = i * block_size
        test_end = (i + test_blocks) * block_size

        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        yield train_idx, test_idx


def cross_validate(
    decoder_class: Type[BaseDecoder],
    X: np.ndarray,
    y: np.ndarray,
    cv_method: str = "temporal",
    n_splits: int = 5,
    decoder_params: Optional[Dict] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Perform cross-validation on a decoder.

    Args:
        decoder_class: Decoder class to instantiate and evaluate.
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples, n_outputs).
        cv_method: Cross-validation method ('temporal', 'sliding', 'blocked').
        n_splits: Number of CV splits.
        decoder_params: Parameters to pass to decoder constructor.
        return_predictions: Whether to return predictions.

    Returns:
        Dictionary containing CV results with metrics per fold.
    """
    if decoder_params is None:
        decoder_params = {}

    n_samples = X.shape[0]

    # Select CV method
    if cv_method == "temporal":
        splits = list(temporal_split(n_samples, n_splits))
    elif cv_method == "sliding":
        train_size = n_samples // (n_splits + 1)
        test_size = train_size // 2
        splits = list(sliding_window_split(n_samples, train_size, test_size))
    elif cv_method == "blocked":
        splits = list(blocked_split(n_samples, n_splits))
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")

    # Run CV
    fold_metrics = []
    predictions = [] if return_predictions else None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create and train decoder
        decoder = decoder_class(**decoder_params)
        decoder.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = decoder.predict(X_test)
        metrics = compute_all_metrics(y_test, y_pred)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

        if return_predictions:
            predictions.append(
                {
                    "test_idx": test_idx,
                    "y_true": y_test,
                    "y_pred": y_pred,
                }
            )

    # Aggregate results
    results = {
        "n_splits": len(splits),
        "cv_method": cv_method,
        "fold_metrics": fold_metrics,
    }

    # Compute mean and std of metrics
    metric_names = [k for k in fold_metrics[0].keys() if k != "fold"]
    for metric in metric_names:
        values = [f[metric] for f in fold_metrics]
        results[f"{metric}_mean"] = float(np.mean(values))
        results[f"{metric}_std"] = float(np.std(values))

    if return_predictions:
        results["predictions"] = predictions

    return results


def compare_decoders(
    decoder_classes: Dict[str, Type[BaseDecoder]],
    X: np.ndarray,
    y: np.ndarray,
    cv_method: str = "temporal",
    n_splits: int = 5,
    decoder_params: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    """
    Compare multiple decoders using cross-validation.

    Args:
        decoder_classes: Dictionary mapping decoder names to classes.
        X: Features of shape (n_samples, n_features).
        y: Targets of shape (n_samples, n_outputs).
        cv_method: Cross-validation method.
        n_splits: Number of CV splits.
        decoder_params: Optional dict mapping decoder names to params.

    Returns:
        Dictionary mapping decoder names to CV results.
    """
    if decoder_params is None:
        decoder_params = {}

    results = {}

    for name, decoder_class in decoder_classes.items():
        params = decoder_params.get(name, {})
        results[name] = cross_validate(
            decoder_class,
            X,
            y,
            cv_method=cv_method,
            n_splits=n_splits,
            decoder_params=params,
        )

    return results
