"""
BCI-specific performance metrics.

Implements standard Brain-Computer Interface metrics including:
- Information Transfer Rate (ITR)
- Fitts' Law Throughput
- Path Efficiency
- Target Acquisition Time
- Success Rate

Reference:
    Wolpaw et al. (2002) "Brain-computer interfaces for communication
    and control" Clinical Neurophysiology
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class BCIPerformanceMetrics:
    """Container for BCI performance metrics."""

    itr_bits_per_trial: float
    itr_bits_per_min: float
    throughput_bits_per_s: Optional[float]
    success_rate: float
    mean_acquisition_time: float
    path_efficiency: Optional[float]
    overshoot_rate: Optional[float]


def information_transfer_rate(
    n_targets: int,
    accuracy: float,
    trial_duration_s: float,
) -> Tuple[float, float]:
    """
    Calculate Information Transfer Rate (ITR).

    ITR measures the communication rate of a BCI system in bits per unit time.

    Formula:
        B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))

    where N is number of targets and P is accuracy.

    Args:
        n_targets: Number of possible targets/classes.
        accuracy: Classification accuracy (0 to 1).
        trial_duration_s: Duration of each trial in seconds.

    Returns:
        Tuple of (bits_per_trial, bits_per_minute).

    Reference:
        Wolpaw et al. (2002) Clinical Neurophysiology
    """
    if n_targets < 2:
        raise ValueError("Number of targets must be at least 2")

    if accuracy <= 0 or accuracy > 1:
        raise ValueError("Accuracy must be in (0, 1]")

    # Handle edge cases
    if accuracy == 1.0:
        bits_per_trial = np.log2(n_targets)
    elif accuracy <= 1 / n_targets:
        bits_per_trial = 0.0
    else:
        p = accuracy
        n = n_targets

        # ITR formula
        term1 = np.log2(n)
        term2 = p * np.log2(p) if p > 0 else 0
        term3 = (1 - p) * np.log2((1 - p) / (n - 1)) if p < 1 else 0

        bits_per_trial = term1 + term2 + term3

    # Convert to bits per minute
    trials_per_min = 60.0 / trial_duration_s
    bits_per_min = bits_per_trial * trials_per_min

    return float(bits_per_trial), float(bits_per_min)


def fitts_throughput(
    target_distances: np.ndarray,
    target_widths: np.ndarray,
    movement_times: np.ndarray,
    method: str = "shannon",
) -> float:
    """
    Calculate Fitts' Law throughput.

    Throughput measures the information capacity of the motor control system.

    Args:
        target_distances: Array of target distances (amplitude).
        target_widths: Array of target widths (tolerance).
        movement_times: Array of movement times in seconds.
        method: Index of difficulty formula:
            - 'shannon': ID = log2(D/W + 1) [recommended by ISO 9241-9]
            - 'welford': ID = log2(D/W + 0.5)
            - 'fitts': ID = log2(2D/W)

    Returns:
        Throughput in bits per second.

    Reference:
        MacKenzie (1992) "Fitts' law as a research and design tool in HCI"
    """
    target_distances = np.asarray(target_distances)
    target_widths = np.asarray(target_widths)
    movement_times = np.asarray(movement_times)

    # Calculate Index of Difficulty (ID)
    if method == "shannon":
        # ISO 9241-9 recommended
        id_values = np.log2(target_distances / target_widths + 1)
    elif method == "welford":
        id_values = np.log2(target_distances / target_widths + 0.5)
    elif method == "fitts":
        id_values = np.log2(2 * target_distances / target_widths)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'shannon', 'welford', or 'fitts'")

    # Throughput = mean(ID / MT)
    # Or equivalently, use effective ID and effective width
    throughput = float(np.mean(id_values / movement_times))

    return throughput


def effective_throughput(
    target_distances: np.ndarray,
    endpoint_scatter: np.ndarray,
    movement_times: np.ndarray,
) -> float:
    """
    Calculate effective throughput using actual endpoint variability.

    Uses effective target width based on endpoint distribution.

    Args:
        target_distances: Array of target distances.
        endpoint_scatter: Standard deviation of endpoints for each condition.
        movement_times: Array of movement times in seconds.

    Returns:
        Effective throughput in bits per second.

    Reference:
        Soukoreff & MacKenzie (2004) "Towards a standard for pointing
        device evaluation"
    """
    # Effective width = 4.133 * SD (for 96% of endpoints within target)
    effective_widths = 4.133 * np.asarray(endpoint_scatter)

    # Effective ID
    effective_id = np.log2(np.asarray(target_distances) / effective_widths + 1)

    # Throughput
    throughput = float(np.mean(effective_id / np.asarray(movement_times)))

    return throughput


def path_efficiency(
    actual_path: np.ndarray,
    straight_line_distance: float,
) -> float:
    """
    Calculate path efficiency (straightness).

    Ratio of optimal (straight-line) distance to actual path length.

    Args:
        actual_path: Array of positions along movement path, shape (n_points, n_dims).
        straight_line_distance: Direct distance from start to target.

    Returns:
        Path efficiency ratio (1.0 = perfectly straight).
    """
    # Calculate actual path length
    if actual_path.ndim == 1:
        actual_path = actual_path.reshape(-1, 1)

    path_segments = np.diff(actual_path, axis=0)
    path_length = float(np.sum(np.linalg.norm(path_segments, axis=1)))

    if path_length == 0:
        return 1.0

    efficiency = straight_line_distance / path_length

    return float(np.clip(efficiency, 0, 1))


def movement_variability(
    trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    Calculate movement variability metrics across repeated movements.

    Args:
        trajectories: List of trajectory arrays, each of shape (n_points, n_dims).

    Returns:
        Dictionary with variability metrics.
    """
    if len(trajectories) < 2:
        raise ValueError("Need at least 2 trajectories for variability analysis")

    # Normalize trajectory lengths
    max_len = max(len(t) for t in trajectories)
    normalized = []

    for traj in trajectories:
        if len(traj) < max_len:
            # Interpolate to common length
            old_indices = np.linspace(0, 1, len(traj))
            new_indices = np.linspace(0, 1, max_len)

            interp_traj = np.zeros((max_len, traj.shape[1]))
            for dim in range(traj.shape[1]):
                interp_traj[:, dim] = np.interp(new_indices, old_indices, traj[:, dim])
            normalized.append(interp_traj)
        else:
            normalized.append(traj)

    trajectories_array = np.array(normalized)  # (n_trials, n_points, n_dims)

    # Calculate variability at each point
    point_variability = np.std(trajectories_array, axis=0)  # (n_points, n_dims)
    mean_variability = float(np.mean(point_variability))
    max_variability = float(np.max(point_variability))

    # Endpoint variability
    endpoints = trajectories_array[:, -1, :]  # (n_trials, n_dims)
    endpoint_std = float(np.mean(np.std(endpoints, axis=0)))

    return {
        "mean_variability": mean_variability,
        "max_variability": max_variability,
        "endpoint_variability": endpoint_std,
    }


def target_acquisition_metrics(
    cursor_positions: np.ndarray,
    target_position: np.ndarray,
    target_radius: float,
    timestamps: np.ndarray,
    dwell_time_required: float = 0.0,
) -> Dict[str, Union[float, bool]]:
    """
    Calculate target acquisition metrics for a single trial.

    Args:
        cursor_positions: Array of cursor positions, shape (n_samples, n_dims).
        target_position: Target center position.
        target_radius: Target radius.
        timestamps: Time stamps for each position sample.
        dwell_time_required: Required dwell time inside target for success.

    Returns:
        Dictionary with acquisition metrics.
    """
    target_position = np.asarray(target_position)
    cursor_positions = np.asarray(cursor_positions)

    # Calculate distance to target at each time point
    distances = np.linalg.norm(cursor_positions - target_position, axis=1)

    # Find when cursor first enters target
    inside_target = distances <= target_radius

    if not np.any(inside_target):
        return {
            "success": False,
            "acquisition_time": float("inf"),
            "time_in_target": 0.0,
            "n_entries": 0,
            "overshoot_count": 0,
            "final_distance": float(distances[-1]),
        }

    # First entry time
    first_entry_idx = np.argmax(inside_target)
    acquisition_time = float(timestamps[first_entry_idx] - timestamps[0])

    # Time spent in target
    time_in_target = float(np.sum(np.diff(timestamps)[inside_target[:-1]]))

    # Count entries (transitions from outside to inside)
    entry_transitions = np.diff(inside_target.astype(int))
    n_entries = int(np.sum(entry_transitions == 1))

    # Count overshoots (passed through target without stopping)
    overshoot_count = max(0, n_entries - 1)

    # Check dwell time success
    if dwell_time_required > 0:
        # Find consecutive time inside target
        success = False
        consecutive_time = 0.0
        for i in range(len(inside_target)):
            if inside_target[i]:
                if i > 0:
                    consecutive_time += timestamps[i] - timestamps[i - 1]
                if consecutive_time >= dwell_time_required:
                    success = True
                    break
            else:
                consecutive_time = 0.0
    else:
        success = True

    return {
        "success": success,
        "acquisition_time": acquisition_time,
        "time_in_target": time_in_target,
        "n_entries": n_entries,
        "overshoot_count": overshoot_count,
        "final_distance": float(distances[-1]),
    }


def success_rate(
    outcomes: np.ndarray,
) -> float:
    """
    Calculate success rate from trial outcomes.

    Args:
        outcomes: Boolean array of trial outcomes (True = success).

    Returns:
        Success rate (0 to 1).
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    return float(np.mean(outcomes))


def bits_per_second_continuous(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sampling_rate: float,
) -> float:
    """
    Estimate information rate for continuous decoding.

    Uses mutual information approximation based on prediction accuracy.

    Args:
        y_true: True continuous signals, shape (n_samples, n_dims).
        y_pred: Predicted signals.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Estimated bits per second.

    Note:
        This is an approximation. True mutual information estimation
        requires more sophisticated methods.
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)

    if y_true.shape[0] == 1:
        y_true = y_true.T
        y_pred = y_pred.T

    n_dims = y_true.shape[1]

    # Calculate SNR for each dimension
    total_bits_per_sample = 0.0

    for i in range(n_dims):
        signal_var = np.var(y_true[:, i])
        noise_var = np.var(y_true[:, i] - y_pred[:, i])

        if noise_var > 0 and signal_var > noise_var:
            # Channel capacity approximation
            snr = signal_var / noise_var
            bits = 0.5 * np.log2(1 + snr)
            total_bits_per_sample += bits

    # Convert to bits per second
    bits_per_second = float(total_bits_per_sample * sampling_rate)

    return bits_per_second


def compute_bci_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_targets: Optional[int] = None,
    trial_duration_s: Optional[float] = None,
    target_distances: Optional[np.ndarray] = None,
    target_widths: Optional[np.ndarray] = None,
    movement_times: Optional[np.ndarray] = None,
    sampling_rate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive BCI performance metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        n_targets: Number of targets (for discrete ITR).
        trial_duration_s: Trial duration in seconds.
        target_distances: Target distances (for Fitts' throughput).
        target_widths: Target widths.
        movement_times: Movement times in seconds.
        sampling_rate: Sampling rate in Hz (for continuous metrics).

    Returns:
        Dictionary of metric names to values.
    """
    metrics = {}

    # Classification accuracy (if discrete)
    if n_targets is not None and y_true.ndim == 1:
        accuracy = float(np.mean(y_true == y_pred))
        metrics["accuracy"] = accuracy

        if trial_duration_s is not None:
            bits_trial, bits_min = information_transfer_rate(n_targets, accuracy, trial_duration_s)
            metrics["itr_bits_per_trial"] = bits_trial
            metrics["itr_bits_per_min"] = bits_min

    # Continuous decoding metrics
    if y_true.ndim >= 1 and y_pred.ndim >= 1:
        # R² score
        y_t = np.atleast_2d(y_true)
        y_p = np.atleast_2d(y_pred)
        if y_t.shape[0] == 1:
            y_t, y_p = y_t.T, y_p.T

        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t, axis=0)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics["r2"] = float(r2)

        # Correlation
        if y_t.shape[1] == 1:
            corr = np.corrcoef(y_t.flatten(), y_p.flatten())[0, 1]
        else:
            corrs = [np.corrcoef(y_t[:, i], y_p[:, i])[0, 1] for i in range(y_t.shape[1])]
            corr = np.mean(corrs)
        metrics["correlation"] = float(corr) if not np.isnan(corr) else 0.0

        # Bits per second (continuous)
        if sampling_rate is not None:
            metrics["bits_per_second"] = bits_per_second_continuous(y_true, y_pred, sampling_rate)

    # Fitts' throughput
    if target_distances is not None and target_widths is not None and movement_times is not None:
        metrics["fitts_throughput_bps"] = fitts_throughput(
            target_distances, target_widths, movement_times
        )

    return metrics
