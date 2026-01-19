"""
Online Adapter for the Meta-Learner.

Implements online adaptation mechanisms to adjust decoder weights
and update decoders based on recent performance.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.meta_learner.base import (
    DecoderMetrics,
    DecoderState,
    DecoderWrapper,
    EnsembleResult,
)


class OnlineAdapter:
    """
    Online adaptation component for the Meta-Learner.

    Responsibilities:
    - Update decoder weights based on recent errors
    - Detect performance degradation (electrode dropout)
    - Trigger decoder updates when needed
    - Manage decoder states (active, standby, degraded)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        update_interval: int = 10,
        degradation_window: int = 50,
        degradation_threshold: float = 0.3,
        recovery_threshold: float = 0.1,
        min_weight: float = 0.05,
        max_weight: float = 2.0,
    ):
        """
        Initialize online adapter.

        Args:
            learning_rate: Rate for weight updates.
            momentum: Momentum for weight updates.
            weight_decay: L2 regularization toward uniform weights.
            update_interval: Steps between decoder updates.
            degradation_window: Window for detecting degradation.
            degradation_threshold: Error increase to trigger degradation.
            recovery_threshold: Error decrease to recover from degradation.
            min_weight: Minimum decoder weight.
            max_weight: Maximum decoder weight.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.update_interval = update_interval
        self.degradation_window = degradation_window
        self.degradation_threshold = degradation_threshold
        self.recovery_threshold = recovery_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Tracking
        self._step_count = 0
        self._weight_velocities: Dict[str, float] = {}
        self._error_history: Dict[str, List[float]] = {}
        self._baseline_errors: Dict[str, float] = {}

        # Buffers for decoder updates
        self._update_buffer_X: List[np.ndarray] = []
        self._update_buffer_y: List[np.ndarray] = []
        self._buffer_size = 100

    def update(
        self,
        decoders: Dict[str, DecoderWrapper],
        ensemble_result: EnsembleResult,
        y_true: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform online adaptation step.

        Args:
            decoders: Dictionary of decoder wrappers.
            ensemble_result: Result from last prediction.
            y_true: True target values.
            X: Optional input features (for decoder updates).

        Returns:
            Dictionary with adaptation statistics.
        """
        self._step_count += 1

        stats = {
            "step": self._step_count,
            "weight_updates": {},
            "state_changes": [],
            "decoder_updates": [],
        }

        # Compute errors for each decoder
        errors = self._compute_errors(ensemble_result, y_true)

        # Update metrics for each decoder
        for name, error in errors.items():
            if name in decoders:
                wrapper = decoders[name]
                wrapper.metrics.update(
                    r2=1.0 - error,  # Approximate R² from normalized error
                    mse=error,
                )

                # Track error history
                if name not in self._error_history:
                    self._error_history[name] = []
                self._error_history[name].append(error)

        # Update weights based on errors
        weight_updates = self._update_weights(decoders, errors)
        stats["weight_updates"] = weight_updates

        # Check for degradation
        state_changes = self._check_degradation(decoders)
        stats["state_changes"] = state_changes

        # Check for recovery
        recovery_changes = self._check_recovery(decoders)
        stats["state_changes"].extend(recovery_changes)

        # Buffer data for decoder updates
        if X is not None:
            self._update_buffer_X.append(X)
            self._update_buffer_y.append(y_true)

            # Trim buffer
            if len(self._update_buffer_X) > self._buffer_size:
                self._update_buffer_X = self._update_buffer_X[-self._buffer_size:]
                self._update_buffer_y = self._update_buffer_y[-self._buffer_size:]

        # Periodically update decoders
        if self._step_count % self.update_interval == 0:
            updated = self._update_decoders(decoders)
            stats["decoder_updates"] = updated

        return stats

    def _compute_errors(
        self,
        ensemble_result: EnsembleResult,
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """Compute normalized errors for each decoder."""
        errors = {}

        for name, prediction in ensemble_result.individual_predictions.items():
            # Compute MSE
            mse = np.mean((prediction - y_true) ** 2)

            # Normalize by target variance
            target_var = np.var(y_true) + 1e-10
            normalized_error = mse / target_var

            errors[name] = normalized_error

        return errors

    def _update_weights(
        self,
        decoders: Dict[str, DecoderWrapper],
        errors: Dict[str, float],
    ) -> Dict[str, float]:
        """Update decoder weights based on errors."""
        updates = {}

        if not errors:
            return updates

        # Compute relative errors
        mean_error = np.mean(list(errors.values()))

        for name, error in errors.items():
            if name not in decoders:
                continue

            wrapper = decoders[name]

            # Skip disabled decoders
            if wrapper.state == DecoderState.DISABLED:
                continue

            # Compute gradient (negative = better than average)
            gradient = error - mean_error

            # Apply momentum
            if name not in self._weight_velocities:
                self._weight_velocities[name] = 0.0

            velocity = (
                self.momentum * self._weight_velocities[name]
                - self.learning_rate * gradient
            )
            self._weight_velocities[name] = velocity

            # Weight decay (pull toward 1.0)
            decay = self.weight_decay * (wrapper.weight - 1.0)

            # Update weight
            new_weight = wrapper.weight + velocity - decay

            # Clamp to valid range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))

            # Apply update
            old_weight = wrapper.weight
            wrapper.weight = new_weight
            updates[name] = new_weight - old_weight

        return updates

    def _check_degradation(
        self,
        decoders: Dict[str, DecoderWrapper],
    ) -> List[Dict[str, Any]]:
        """Check for and handle decoder degradation."""
        changes = []

        for name, wrapper in decoders.items():
            if wrapper.state == DecoderState.DISABLED:
                continue

            if wrapper.state == DecoderState.DEGRADED:
                continue

            # Check error history
            if name not in self._error_history:
                continue

            history = self._error_history[name]
            if len(history) < self.degradation_window * 2:
                continue

            # Compare recent to baseline
            if name not in self._baseline_errors:
                # Establish baseline
                self._baseline_errors[name] = np.mean(
                    history[:self.degradation_window]
                )

            baseline = self._baseline_errors[name]
            recent = np.mean(history[-self.degradation_window:])

            # Check for significant degradation
            relative_increase = (recent - baseline) / (baseline + 1e-10)

            if relative_increase > self.degradation_threshold:
                # Mark as degraded
                old_state = wrapper.state
                wrapper.state = DecoderState.DEGRADED
                wrapper.weight *= 0.5  # Reduce weight

                changes.append({
                    "decoder": name,
                    "old_state": old_state.value,
                    "new_state": DecoderState.DEGRADED.value,
                    "reason": f"Error increased by {relative_increase:.1%}",
                })

        return changes

    def _check_recovery(
        self,
        decoders: Dict[str, DecoderWrapper],
    ) -> List[Dict[str, Any]]:
        """Check for decoder recovery from degradation."""
        changes = []

        for name, wrapper in decoders.items():
            if wrapper.state != DecoderState.DEGRADED:
                continue

            # Check error history
            if name not in self._error_history:
                continue

            history = self._error_history[name]
            if len(history) < self.degradation_window:
                continue

            # Check if error has improved
            if name not in self._baseline_errors:
                continue

            baseline = self._baseline_errors[name]
            recent = np.mean(history[-self.degradation_window:])

            relative_diff = (recent - baseline) / (baseline + 1e-10)

            if relative_diff < self.recovery_threshold:
                # Recovered
                wrapper.state = DecoderState.STANDBY
                wrapper.weight = min(1.0, wrapper.weight * 1.5)

                changes.append({
                    "decoder": name,
                    "old_state": DecoderState.DEGRADED.value,
                    "new_state": DecoderState.STANDBY.value,
                    "reason": "Performance recovered",
                })

        return changes

    def _update_decoders(
        self,
        decoders: Dict[str, DecoderWrapper],
    ) -> List[str]:
        """Update decoders with buffered data."""
        updated = []

        if not self._update_buffer_X or not self._update_buffer_y:
            return updated

        # Concatenate buffers
        X = np.vstack(self._update_buffer_X)
        y = np.vstack(self._update_buffer_y)

        for name, wrapper in decoders.items():
            if wrapper.state == DecoderState.DISABLED:
                continue

            decoder = wrapper.decoder

            # Check if decoder supports online updates
            if hasattr(decoder, 'update'):
                try:
                    decoder.update(X, y)
                    updated.append(name)
                except Exception:
                    # Update failed, continue
                    pass

        return updated

    def set_baseline(
        self,
        decoders: Dict[str, DecoderWrapper],
    ) -> None:
        """
        Set baseline errors for all decoders.

        Call after initial fitting to establish reference performance.
        """
        for name, wrapper in decoders.items():
            if name in self._error_history and self._error_history[name]:
                self._baseline_errors[name] = np.mean(
                    self._error_history[name][-self.degradation_window:]
                )

    def force_update(
        self,
        decoders: Dict[str, DecoderWrapper],
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[str]:
        """
        Force immediate update of all decoders.

        Args:
            decoders: Dictionary of decoder wrappers.
            X: Input features.
            y: Target values.

        Returns:
            List of updated decoder names.
        """
        updated = []

        for name, wrapper in decoders.items():
            if wrapper.state == DecoderState.DISABLED:
                continue

            decoder = wrapper.decoder

            if hasattr(decoder, 'update'):
                try:
                    decoder.update(X, y)
                    updated.append(name)
                except Exception:
                    pass
            elif hasattr(decoder, 'partial_fit'):
                try:
                    decoder.partial_fit(X, y)
                    updated.append(name)
                except Exception:
                    pass

        return updated

    def reset_decoder(
        self,
        decoders: Dict[str, DecoderWrapper],
        name: str,
    ) -> bool:
        """
        Reset a specific decoder's state and metrics.

        Args:
            decoders: Dictionary of decoder wrappers.
            name: Name of decoder to reset.

        Returns:
            True if reset successful.
        """
        if name not in decoders:
            return False

        wrapper = decoders[name]

        # Reset state
        wrapper.state = DecoderState.STANDBY
        wrapper.weight = 1.0

        # Reset metrics
        wrapper.metrics = DecoderMetrics(name=name)

        # Clear history
        if name in self._error_history:
            self._error_history[name].clear()
        if name in self._baseline_errors:
            del self._baseline_errors[name]
        if name in self._weight_velocities:
            del self._weight_velocities[name]

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "step_count": self._step_count,
            "buffer_size": len(self._update_buffer_X),
            "tracked_decoders": list(self._error_history.keys()),
            "baseline_decoders": list(self._baseline_errors.keys()),
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
        }

    def reset(self) -> None:
        """Reset adapter state."""
        self._step_count = 0
        self._weight_velocities.clear()
        self._error_history.clear()
        self._baseline_errors.clear()
        self._update_buffer_X.clear()
        self._update_buffer_y.clear()
