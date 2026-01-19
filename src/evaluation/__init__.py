"""
Evaluation module for neural decoder performance.

Provides metrics (R², MSE, correlation), latency measurement,
and cross-validation utilities.
"""

from src.evaluation.cross_validation import (
    blocked_split,
    compare_decoders,
    cross_validate,
    sliding_window_split,
    temporal_split,
)
from src.evaluation.latency import (
    LatencyStats,
    LatencyTracker,
    check_latency_requirement,
    measure_latency,
)
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_metrics_per_dimension,
    correlation,
    mae,
    mse,
    r2_score,
    rmse,
    snr,
)

__all__ = [
    # Metrics
    "r2_score",
    "mse",
    "rmse",
    "mae",
    "correlation",
    "snr",
    "compute_all_metrics",
    "compute_metrics_per_dimension",
    # Latency
    "LatencyStats",
    "LatencyTracker",
    "measure_latency",
    "check_latency_requirement",
    # Cross-validation
    "temporal_split",
    "sliding_window_split",
    "blocked_split",
    "cross_validate",
    "compare_decoders",
]
