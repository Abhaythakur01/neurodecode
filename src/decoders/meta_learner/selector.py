"""
Decoder Selector for the Meta-Learner.

Implements strategies for selecting which decoders to use based on
performance, uncertainty, and contextual factors.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.meta_learner.base import (
    DecoderMetrics,
    DecoderState,
    DecoderWrapper,
    SelectionStrategy,
)


class DecoderSelector:
    """
    Selects decoders based on performance and uncertainty.

    Implements multiple selection strategies:
    - Best: Select the single best-performing decoder
    - Top-K: Select the K best decoders
    - Threshold: Select all decoders above a performance threshold
    - Uncertainty-aware: Consider prediction uncertainty in selection
    - Adaptive: Dynamically switch strategies based on conditions
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
        top_k: int = 3,
        performance_threshold: float = 0.5,
        uncertainty_weight: float = 0.3,
        latency_weight: float = 0.1,
        stability_weight: float = 0.2,
        min_history: int = 10,
        degradation_threshold: float = 0.2,
    ):
        """
        Initialize decoder selector.

        Args:
            strategy: Selection strategy to use.
            top_k: Number of decoders for top-K strategy.
            performance_threshold: Minimum R² for threshold strategy.
            uncertainty_weight: Weight for uncertainty in scoring.
            latency_weight: Weight for latency in scoring.
            stability_weight: Weight for stability in scoring.
            min_history: Minimum history before using metrics.
            degradation_threshold: R² drop to trigger degradation.
        """
        self.strategy = strategy
        self.top_k = top_k
        self.performance_threshold = performance_threshold
        self.uncertainty_weight = uncertainty_weight
        self.latency_weight = latency_weight
        self.stability_weight = stability_weight
        self.min_history = min_history
        self.degradation_threshold = degradation_threshold

        # Track selection history
        self._selection_history: List[List[str]] = []
        self._score_history: Dict[str, List[float]] = {}

    def select(
        self,
        decoders: Dict[str, DecoderWrapper],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Select decoders to use for prediction.

        Args:
            decoders: Dictionary of decoder wrappers.
            context: Optional context information.

        Returns:
            List of selected decoder names.
        """
        # Filter to active/standby decoders
        available = {
            name: wrapper for name, wrapper in decoders.items()
            if wrapper.state in (DecoderState.ACTIVE, DecoderState.STANDBY)
        }

        if not available:
            # Fallback: include degraded decoders if nothing else available
            available = {
                name: wrapper for name, wrapper in decoders.items()
                if wrapper.state != DecoderState.DISABLED
            }

        if not available:
            return []

        # Compute scores for each decoder
        scores = self._compute_scores(available, context)

        # Apply selection strategy
        if self.strategy == SelectionStrategy.BEST:
            selected = self._select_best(scores)
        elif self.strategy == SelectionStrategy.TOP_K:
            selected = self._select_top_k(scores)
        elif self.strategy == SelectionStrategy.THRESHOLD:
            selected = self._select_threshold(scores)
        elif self.strategy == SelectionStrategy.UNCERTAINTY_AWARE:
            selected = self._select_uncertainty_aware(scores, available)
        else:  # ADAPTIVE
            selected = self._select_adaptive(scores, available, context)

        # Ensure at least one decoder selected
        if not selected and scores:
            selected = [max(scores, key=scores.get)]

        # Update history
        self._selection_history.append(selected)
        for name, score in scores.items():
            if name not in self._score_history:
                self._score_history[name] = []
            self._score_history[name].append(score)

        return selected

    def _compute_scores(
        self,
        decoders: Dict[str, DecoderWrapper],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Compute selection scores for each decoder.

        Score combines:
        - Recent R² performance (primary)
        - Uncertainty (lower is better)
        - Latency (lower is better)
        - Stability (lower variance is better)
        """
        scores = {}

        for name, wrapper in decoders.items():
            metrics = wrapper.metrics

            # Check if we have enough history
            if len(metrics.r2_history) < self.min_history:
                # Use default score for new decoders
                scores[name] = 0.5
                continue

            # Base score from R² performance (0-1 range)
            r2_score = max(0, min(1, metrics.recent_r2))

            # Uncertainty penalty (if available)
            uncertainty_penalty = 0.0
            if metrics.recent_uncertainty < 1.0:
                uncertainty_penalty = metrics.recent_uncertainty * self.uncertainty_weight

            # Latency penalty (normalized, assuming max ~50ms target)
            latency_penalty = 0.0
            if metrics.recent_latency > 0:
                normalized_latency = min(1.0, metrics.recent_latency / 50.0)
                latency_penalty = normalized_latency * self.latency_weight

            # Stability bonus (lower std is better)
            stability_bonus = 0.0
            if metrics.stability < 0.5:
                stability_bonus = (0.5 - metrics.stability) * self.stability_weight

            # Trend bonus (improving decoders get boost)
            trend_bonus = max(-0.1, min(0.1, metrics.performance_trend))

            # Compute final score
            score = (
                r2_score
                - uncertainty_penalty
                - latency_penalty
                + stability_bonus
                + trend_bonus
            )

            # State adjustment
            if wrapper.state == DecoderState.DEGRADED:
                score *= 0.5  # Penalize degraded decoders

            scores[name] = max(0, score)

        return scores

    def _select_best(self, scores: Dict[str, float]) -> List[str]:
        """Select single best decoder."""
        if not scores:
            return []
        return [max(scores, key=scores.get)]

    def _select_top_k(self, scores: Dict[str, float]) -> List[str]:
        """Select top K decoders."""
        if not scores:
            return []

        sorted_decoders = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_decoders[:min(self.top_k, len(sorted_decoders))]

    def _select_threshold(self, scores: Dict[str, float]) -> List[str]:
        """Select all decoders above threshold."""
        selected = [
            name for name, score in scores.items()
            if score >= self.performance_threshold
        ]

        # Ensure at least one
        if not selected and scores:
            selected = [max(scores, key=scores.get)]

        return selected

    def _select_uncertainty_aware(
        self,
        scores: Dict[str, float],
        decoders: Dict[str, DecoderWrapper],
    ) -> List[str]:
        """Select decoders with uncertainty weighting."""
        # Adjust scores by uncertainty
        adjusted_scores = {}

        for name, score in scores.items():
            wrapper = decoders[name]
            if wrapper.supports_uncertainty:
                # Boost decoders that can provide uncertainty
                uncertainty = wrapper.metrics.recent_uncertainty
                confidence = 1.0 / (1.0 + uncertainty)
                adjusted_scores[name] = score * (0.7 + 0.3 * confidence)
            else:
                adjusted_scores[name] = score * 0.9  # Slight penalty

        return self._select_top_k(adjusted_scores)

    def _select_adaptive(
        self,
        scores: Dict[str, float],
        decoders: Dict[str, DecoderWrapper],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Adaptively select strategy based on conditions.

        Rules:
        - If one decoder clearly best (>0.2 gap), use BEST
        - If high variability in recent performance, use ensemble (TOP_K)
        - If uncertainty available and high, use more decoders
        - Default to TOP_K for robustness
        """
        if not scores:
            return []

        sorted_scores = sorted(scores.values(), reverse=True)

        # Check if one decoder is clearly better
        if len(sorted_scores) >= 2:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap > 0.2:
                return self._select_best(scores)

        # Check recent selection stability
        if len(self._selection_history) >= 5:
            recent_selections = self._selection_history[-5:]
            unique_selected = set()
            for sel in recent_selections:
                unique_selected.update(sel)

            # If selections are unstable, use more decoders
            if len(unique_selected) > self.top_k:
                return self._select_top_k(scores)

        # Check if high uncertainty situation
        avg_uncertainty = np.mean([
            decoders[name].metrics.recent_uncertainty
            for name in scores.keys()
            if decoders[name].metrics.recent_uncertainty < 1.0
        ] or [0.5])

        if avg_uncertainty > 0.5:
            # High uncertainty: use more decoders
            return self._select_top_k(
                {k: v for k, v in scores.items()},
            )

        # Default: top-K for robustness
        return self._select_top_k(scores)

    def detect_degradation(
        self,
        decoders: Dict[str, DecoderWrapper],
    ) -> List[str]:
        """
        Detect decoders with degraded performance.

        Returns list of decoder names that have degraded.
        """
        degraded = []

        for name, wrapper in decoders.items():
            metrics = wrapper.metrics

            if len(metrics.r2_history) < self.min_history * 2:
                continue

            # Compare recent to historical performance
            all_r2 = metrics.r2_history
            historical_avg = np.mean(all_r2[:-self.min_history])
            recent_avg = np.mean(all_r2[-self.min_history:])

            if historical_avg - recent_avg > self.degradation_threshold:
                degraded.append(name)

        return degraded

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about decoder selection."""
        if not self._selection_history:
            return {"total_selections": 0}

        # Count selection frequency
        frequency = {}
        for selection in self._selection_history:
            for name in selection:
                frequency[name] = frequency.get(name, 0) + 1

        total = len(self._selection_history)
        selection_rates = {
            name: count / total for name, count in frequency.items()
        }

        # Average scores
        avg_scores = {
            name: np.mean(scores) if scores else 0
            for name, scores in self._score_history.items()
        }

        return {
            "total_selections": total,
            "selection_frequency": frequency,
            "selection_rates": selection_rates,
            "average_scores": avg_scores,
            "strategy": self.strategy.value,
        }

    def reset(self) -> None:
        """Reset selection history."""
        self._selection_history.clear()
        self._score_history.clear()
