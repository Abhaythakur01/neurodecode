"""
Decoder Combiner for the Meta-Learner.

Implements strategies for combining predictions from multiple decoders
into a single robust output.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.meta_learner.base import (
    CombinationStrategy,
    DecoderWrapper,
    EnsembleResult,
    PredictionResult,
)


class DecoderCombiner:
    """
    Combines predictions from multiple decoders.

    Implements multiple combination strategies:
    - Mean: Simple average of predictions
    - Weighted Mean: Performance-weighted average
    - Median: Robust median (outlier resistant)
    - Uncertainty Weighted: Weight by confidence
    - Stacking: Meta-model on decoder outputs
    """

    def __init__(
        self,
        strategy: CombinationStrategy = CombinationStrategy.UNCERTAINTY_WEIGHTED,
        min_weight: float = 0.1,
        normalize_weights: bool = True,
        outlier_rejection: bool = True,
        outlier_threshold: float = 2.5,
    ):
        """
        Initialize decoder combiner.

        Args:
            strategy: Combination strategy to use.
            min_weight: Minimum weight for any decoder.
            normalize_weights: Whether to normalize weights to sum to 1.
            outlier_rejection: Whether to reject outlier predictions.
            outlier_threshold: Z-score threshold for outlier rejection.
        """
        self.strategy = strategy
        self.min_weight = min_weight
        self.normalize_weights = normalize_weights
        self.outlier_rejection = outlier_rejection
        self.outlier_threshold = outlier_threshold

        # For stacking strategy
        self._stacking_weights: Optional[np.ndarray] = None
        self._stacking_bias: Optional[np.ndarray] = None

    def combine(
        self,
        predictions: Dict[str, PredictionResult],
        decoders: Dict[str, DecoderWrapper],
        selected: List[str],
    ) -> EnsembleResult:
        """
        Combine predictions from selected decoders.

        Args:
            predictions: Prediction results from each decoder.
            decoders: Decoder wrappers with metrics.
            selected: Names of selected decoders to combine.

        Returns:
            Combined ensemble result.
        """
        # Filter to selected decoders
        selected_predictions = {
            name: pred for name, pred in predictions.items()
            if name in selected
        }

        if not selected_predictions:
            raise ValueError("No predictions to combine")

        # Align predictions to common shape (use minimum length)
        min_samples = min(
            pred.prediction.shape[0] for pred in selected_predictions.values()
        )

        # Truncate all predictions to minimum length (take last samples)
        aligned_predictions = {}
        for name, pred in selected_predictions.items():
            n_samples = pred.prediction.shape[0]
            if n_samples > min_samples:
                # Take last min_samples (most recent)
                aligned_pred = PredictionResult(
                    decoder_name=pred.decoder_name,
                    prediction=pred.prediction[-min_samples:],
                    uncertainty=pred.uncertainty[-min_samples:] if pred.uncertainty is not None else None,
                    latency_ms=pred.latency_ms,
                )
                aligned_predictions[name] = aligned_pred
            else:
                aligned_predictions[name] = pred

        selected_predictions = aligned_predictions

        # Get prediction arrays
        pred_arrays = {
            name: pred.prediction for name, pred in selected_predictions.items()
        }

        # Compute weights
        weights = self._compute_weights(selected_predictions, decoders)

        # Apply combination strategy
        if self.strategy == CombinationStrategy.MEAN:
            combined, uncertainty = self._combine_mean(pred_arrays)
        elif self.strategy == CombinationStrategy.WEIGHTED_MEAN:
            combined, uncertainty = self._combine_weighted_mean(pred_arrays, weights)
        elif self.strategy == CombinationStrategy.MEDIAN:
            combined, uncertainty = self._combine_median(pred_arrays)
        elif self.strategy == CombinationStrategy.UNCERTAINTY_WEIGHTED:
            combined, uncertainty = self._combine_uncertainty_weighted(
                selected_predictions, decoders
            )
        elif self.strategy == CombinationStrategy.STACKING:
            combined, uncertainty = self._combine_stacking(pred_arrays, weights)
        else:
            combined, uncertainty = self._combine_weighted_mean(pred_arrays, weights)

        # Compute total latency
        total_latency = max(
            pred.latency_ms for pred in selected_predictions.values()
        )

        return EnsembleResult(
            prediction=combined,
            uncertainty=uncertainty,
            decoder_weights=weights,
            individual_predictions=pred_arrays,
            selected_decoders=selected,
            total_latency_ms=total_latency,
        )

    def _compute_weights(
        self,
        predictions: Dict[str, PredictionResult],
        decoders: Dict[str, DecoderWrapper],
    ) -> Dict[str, float]:
        """Compute combination weights for each decoder."""
        weights = {}

        for name in predictions.keys():
            wrapper = decoders.get(name)

            if wrapper is None:
                weights[name] = 1.0
                continue

            # Base weight from recent R² performance
            base_weight = max(self.min_weight, wrapper.metrics.recent_r2)

            # Adjust by explicit decoder weight
            weight = base_weight * wrapper.weight

            weights[name] = weight

        # Normalize if requested
        if self.normalize_weights and weights:
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}

        return weights

    def _combine_mean(
        self, predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple mean combination."""
        if not predictions:
            raise ValueError("No predictions to combine")

        pred_stack = np.stack(list(predictions.values()), axis=0)
        combined = np.mean(pred_stack, axis=0)

        # Uncertainty from disagreement
        if len(predictions) > 1:
            uncertainty = np.std(pred_stack, axis=0)
        else:
            uncertainty = np.zeros_like(combined)

        return combined, uncertainty

    def _combine_weighted_mean(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted mean combination."""
        if not predictions:
            raise ValueError("No predictions to combine")

        # Stack predictions and weights
        names = list(predictions.keys())
        pred_stack = np.stack([predictions[n] for n in names], axis=0)
        weight_array = np.array([weights.get(n, 1.0) for n in names])

        # Normalize weights
        weight_array = weight_array / (weight_array.sum() + 1e-10)

        # Weighted average
        combined = np.tensordot(weight_array, pred_stack, axes=([0], [0]))

        # Weighted uncertainty
        if len(predictions) > 1:
            mean_expanded = combined[np.newaxis, ...]
            variance = np.sum(
                weight_array[:, np.newaxis] * (pred_stack - mean_expanded) ** 2,
                axis=0
            )
            uncertainty = np.sqrt(variance)
        else:
            uncertainty = np.zeros_like(combined)

        return combined, uncertainty

    def _combine_median(
        self, predictions: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Median combination (robust to outliers)."""
        if not predictions:
            raise ValueError("No predictions to combine")

        pred_stack = np.stack(list(predictions.values()), axis=0)
        combined = np.median(pred_stack, axis=0)

        # MAD-based uncertainty
        if len(predictions) > 1:
            mad = np.median(np.abs(pred_stack - combined), axis=0)
            uncertainty = 1.4826 * mad  # Scale to match std
        else:
            uncertainty = np.zeros_like(combined)

        return combined, uncertainty

    def _combine_uncertainty_weighted(
        self,
        predictions: Dict[str, PredictionResult],
        decoders: Dict[str, DecoderWrapper],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine weighted by inverse uncertainty (confidence)."""
        if not predictions:
            raise ValueError("No predictions to combine")

        names = list(predictions.keys())
        pred_stack = np.stack([predictions[n].prediction for n in names], axis=0)

        # Compute confidence weights
        confidence_weights = []
        for name in names:
            pred = predictions[name]

            if pred.uncertainty is not None:
                # Use prediction-specific uncertainty
                mean_uncertainty = np.mean(pred.uncertainty)
                confidence = 1.0 / (1.0 + mean_uncertainty)
            elif decoders.get(name) is not None:
                # Use historical uncertainty
                wrapper = decoders[name]
                uncertainty = wrapper.metrics.recent_uncertainty
                confidence = 1.0 / (1.0 + uncertainty)
            else:
                confidence = 1.0

            # Also factor in R² performance
            if name in decoders:
                r2 = max(0.1, decoders[name].metrics.recent_r2)
                confidence *= r2

            confidence_weights.append(confidence)

        weight_array = np.array(confidence_weights)
        weight_array = weight_array / (weight_array.sum() + 1e-10)

        # Weighted average
        combined = np.tensordot(weight_array, pred_stack, axes=([0], [0]))

        # Combined uncertainty
        uncertainties = []
        for name in names:
            pred = predictions[name]
            if pred.uncertainty is not None:
                uncertainties.append(pred.uncertainty)
            else:
                # Estimate from metrics
                wrapper = decoders.get(name)
                if wrapper:
                    unc = wrapper.metrics.recent_uncertainty
                    uncertainties.append(np.full_like(pred.prediction, unc))
                else:
                    uncertainties.append(np.ones_like(pred.prediction) * 0.5)

        unc_stack = np.stack(uncertainties, axis=0)

        # Weighted uncertainty (propagated)
        # Reshape weight_array for proper broadcasting
        # unc_stack shape: (n_decoders, n_samples, n_outputs)
        # weight_array shape: (n_decoders,)
        weight_broadcast = weight_array.reshape(-1, 1, 1)
        combined_uncertainty = np.sqrt(np.sum(
            (weight_broadcast ** 2) * (unc_stack ** 2),
            axis=0
        ))

        return combined, combined_uncertainty

    def _combine_stacking(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stacking combination using learned weights.

        If stacking weights not learned, falls back to weighted mean.
        """
        if self._stacking_weights is None:
            return self._combine_weighted_mean(predictions, weights)

        names = list(predictions.keys())
        pred_stack = np.stack([predictions[n] for n in names], axis=0)

        # Apply stacking weights (simplified linear stacking)
        n_decoders, n_samples, n_outputs = pred_stack.shape

        # Reshape for matmul: (n_samples, n_decoders, n_outputs)
        pred_stack = pred_stack.transpose(1, 0, 2)

        # Apply weights: (n_samples, n_outputs)
        combined = np.einsum('ijk,jk->ik', pred_stack, self._stacking_weights[:n_decoders])

        if self._stacking_bias is not None:
            combined += self._stacking_bias

        # Uncertainty from ensemble disagreement
        uncertainty = np.std(pred_stack, axis=1)

        return combined, uncertainty

    def fit_stacking(
        self,
        predictions_history: List[Dict[str, np.ndarray]],
        targets: np.ndarray,
    ) -> None:
        """
        Fit stacking weights from historical predictions.

        Args:
            predictions_history: List of prediction dicts over time.
            targets: True target values.
        """
        if len(predictions_history) < 10:
            return

        # Get decoder names (consistent across history)
        decoder_names = list(predictions_history[0].keys())
        n_decoders = len(decoder_names)

        # Stack all predictions
        # Shape: (n_samples, n_decoders, n_outputs)
        all_preds = []
        for pred_dict in predictions_history:
            preds = [pred_dict[name] for name in decoder_names]
            all_preds.append(np.stack(preds, axis=0))

        X = np.concatenate(all_preds, axis=1).transpose(1, 0, 2)  # (n_samples, n_decoders, n_outputs)
        y = targets

        # Simple least squares for each output dimension
        n_samples, _, n_outputs = X.shape

        self._stacking_weights = np.zeros((n_decoders, n_outputs))
        self._stacking_bias = np.zeros(n_outputs)

        for o in range(n_outputs):
            # Solve: X @ w = y
            X_o = X[:, :, o]  # (n_samples, n_decoders)
            y_o = y[:, o]     # (n_samples,)

            # Ridge regression
            lambda_reg = 0.1
            XtX = X_o.T @ X_o + lambda_reg * np.eye(n_decoders)
            Xty = X_o.T @ y_o

            w = np.linalg.solve(XtX, Xty)

            self._stacking_weights[:, o] = w
            self._stacking_bias[o] = np.mean(y_o - X_o @ w)

    def reject_outliers(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Reject outlier predictions.

        Uses z-score based rejection compared to ensemble median.

        Args:
            predictions: Dictionary of predictions.

        Returns:
            Filtered dictionary with outliers removed.
        """
        if len(predictions) < 3:
            return predictions

        pred_stack = np.stack(list(predictions.values()), axis=0)
        median = np.median(pred_stack, axis=0)
        mad = np.median(np.abs(pred_stack - median), axis=0) + 1e-10

        # Z-score relative to median
        z_scores = np.abs(pred_stack - median) / (1.4826 * mad)
        max_z = np.max(z_scores, axis=(1, 2) if pred_stack.ndim == 3 else 1)

        # Filter predictions
        filtered = {}
        for i, (name, pred) in enumerate(predictions.items()):
            if max_z[i] < self.outlier_threshold:
                filtered[name] = pred

        # Keep at least one
        if not filtered:
            best_idx = np.argmin(max_z)
            name = list(predictions.keys())[best_idx]
            filtered[name] = predictions[name]

        return filtered

    def get_params(self) -> Dict[str, Any]:
        """Get combiner parameters."""
        return {
            "strategy": self.strategy.value,
            "min_weight": self.min_weight,
            "normalize_weights": self.normalize_weights,
            "outlier_rejection": self.outlier_rejection,
            "has_stacking_weights": self._stacking_weights is not None,
        }
