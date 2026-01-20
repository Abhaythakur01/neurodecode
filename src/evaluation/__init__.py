"""
Evaluation module for neural decoder performance.

Provides metrics (R², MSE, correlation), latency measurement,
cross-validation utilities, and BCI-specific metrics (ITR, throughput).
"""

from src.evaluation.bci_metrics import (
    BCIPerformanceMetrics,
    bits_per_second_continuous,
    compute_bci_metrics,
    effective_throughput,
    fitts_throughput,
    information_transfer_rate,
    movement_variability,
    path_efficiency,
    success_rate,
    target_acquisition_metrics,
)
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
    # Standard Metrics
    "r2_score",
    "mse",
    "rmse",
    "mae",
    "correlation",
    "snr",
    "compute_all_metrics",
    "compute_metrics_per_dimension",
    # BCI Metrics
    "information_transfer_rate",
    "fitts_throughput",
    "effective_throughput",
    "path_efficiency",
    "movement_variability",
    "target_acquisition_metrics",
    "success_rate",
    "bits_per_second_continuous",
    "compute_bci_metrics",
    "BCIPerformanceMetrics",
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
