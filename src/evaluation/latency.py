"""
Latency measurement for real-time decoder performance.

Provides tools for measuring and analyzing processing latency
to ensure <50ms requirement is met.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""

    measurements: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.measurements)

    @property
    def mean(self) -> float:
        return float(np.mean(self.measurements)) if self.measurements else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.measurements)) if self.measurements else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.measurements)) if self.measurements else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.measurements)) if self.measurements else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.measurements)) if self.measurements else 0.0

    @property
    def p95(self) -> float:
        """95th percentile latency."""
        return float(np.percentile(self.measurements, 95)) if self.measurements else 0.0

    @property
    def p99(self) -> float:
        """99th percentile latency."""
        return float(np.percentile(self.measurements, 99)) if self.measurements else 0.0

    def add(self, latency_ms: float) -> None:
        """Add a latency measurement in milliseconds."""
        self.measurements.append(latency_ms)

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary."""
        return {
            "count": self.count,
            "mean_ms": self.mean,
            "std_ms": self.std,
            "min_ms": self.min,
            "max_ms": self.max,
            "median_ms": self.median,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
        }

    def reset(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()


class LatencyTracker:
    """
    Track latency across multiple components.

    Useful for profiling the full decoding pipeline.
    """

    def __init__(self):
        self.components: Dict[str, LatencyStats] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, component: str) -> None:
        """Start timing a component."""
        self._start_times[component] = time.perf_counter()

    def stop(self, component: str) -> float:
        """
        Stop timing a component and record latency.

        Args:
            component: Name of the component.

        Returns:
            Latency in milliseconds.
        """
        if component not in self._start_times:
            raise ValueError(f"Timer for '{component}' was not started.")

        elapsed = (time.perf_counter() - self._start_times[component]) * 1000
        del self._start_times[component]

        if component not in self.components:
            self.components[component] = LatencyStats()

        self.components[component].add(elapsed)
        return elapsed

    @contextmanager
    def track(self, component: str):
        """
        Context manager for tracking component latency.

        Usage:
            with tracker.track("preprocessing"):
                preprocess(data)
        """
        self.start(component)
        try:
            yield
        finally:
            self.stop(component)

    def get_stats(self, component: str) -> Optional[LatencyStats]:
        """Get stats for a specific component."""
        return self.components.get(component)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all components."""
        return {name: stats.to_dict() for name, stats in self.components.items()}

    def get_total_latency(self) -> LatencyStats:
        """
        Get combined latency stats.

        Note: This sums per-measurement, assuming components are sequential.
        """
        if not self.components:
            return LatencyStats()

        # Get component with most measurements as reference
        max_count = max(c.count for c in self.components.values())

        # Sum latencies for each measurement index
        total = LatencyStats()
        for i in range(max_count):
            total_ms = sum(
                c.measurements[i] if i < c.count else c.mean for c in self.components.values()
            )
            total.add(total_ms)

        return total

    def reset(self) -> None:
        """Reset all component stats."""
        self.components.clear()
        self._start_times.clear()

    def print_summary(self) -> None:
        """Print latency summary to console."""
        print("\n=== Latency Summary ===")
        for name, stats in self.components.items():
            print(f"\n{name}:")
            print(f"  Mean:   {stats.mean:.2f} ms")
            print(f"  Std:    {stats.std:.2f} ms")
            print(f"  Min:    {stats.min:.2f} ms")
            print(f"  Max:    {stats.max:.2f} ms")
            print(f"  P95:    {stats.p95:.2f} ms")
            print(f"  P99:    {stats.p99:.2f} ms")

        total = self.get_total_latency()
        print("\nTotal Pipeline:")
        print(f"  Mean:   {total.mean:.2f} ms")
        print(f"  P95:    {total.p95:.2f} ms")
        print(f"  P99:    {total.p99:.2f} ms")


def measure_latency(
    func: Callable,
    *args,
    n_iterations: int = 100,
    warmup: int = 10,
    **kwargs,
) -> LatencyStats:
    """
    Measure execution latency of a function.

    Args:
        func: Function to measure.
        *args: Positional arguments for func.
        n_iterations: Number of measurement iterations.
        warmup: Number of warmup iterations (not measured).
        **kwargs: Keyword arguments for func.

    Returns:
        LatencyStats with measurements.
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Measure
    stats = LatencyStats()
    for _ in range(n_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats.add(elapsed_ms)

    return stats


def check_latency_requirement(
    stats: LatencyStats,
    max_latency_ms: float = 50.0,
    percentile: float = 95,
) -> bool:
    """
    Check if latency meets real-time requirement.

    Args:
        stats: Latency statistics.
        max_latency_ms: Maximum allowed latency in ms.
        percentile: Percentile to check (e.g., 95 for p95).

    Returns:
        True if requirement is met.
    """
    if percentile == 95:
        actual = stats.p95
    elif percentile == 99:
        actual = stats.p99
    else:
        actual = float(np.percentile(stats.measurements, percentile))

    return actual <= max_latency_ms
