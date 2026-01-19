"""
Base classes for the Meta-Learner system.

Defines interfaces and data structures used by all meta-learner components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.base import BaseDecoder


class DecoderState(Enum):
    """State of a decoder in the meta-learner."""
    ACTIVE = "active"           # Currently being used
    STANDBY = "standby"         # Ready but not selected
    DEGRADED = "degraded"       # Performance has dropped
    DISABLED = "disabled"       # Temporarily disabled


@dataclass
class DecoderMetrics:
    """Performance metrics for a single decoder."""
    name: str
    r2_history: List[float] = field(default_factory=list)
    mse_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    uncertainty_history: List[float] = field(default_factory=list)

    # Rolling statistics
    window_size: int = 50

    @property
    def recent_r2(self) -> float:
        """Get recent average R²."""
        if not self.r2_history:
            return 0.0
        recent = self.r2_history[-self.window_size:]
        return np.mean(recent)

    @property
    def recent_mse(self) -> float:
        """Get recent average MSE."""
        if not self.mse_history:
            return float('inf')
        recent = self.mse_history[-self.window_size:]
        return np.mean(recent)

    @property
    def recent_latency(self) -> float:
        """Get recent average latency."""
        if not self.latency_history:
            return 0.0
        recent = self.latency_history[-self.window_size:]
        return np.mean(recent)

    @property
    def recent_uncertainty(self) -> float:
        """Get recent average uncertainty."""
        if not self.uncertainty_history:
            return 1.0
        recent = self.uncertainty_history[-self.window_size:]
        return np.mean(recent)

    @property
    def performance_trend(self) -> float:
        """Calculate performance trend (positive = improving)."""
        if len(self.r2_history) < 10:
            return 0.0

        recent = self.r2_history[-20:]
        if len(recent) < 10:
            return 0.0

        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        return second_half - first_half

    @property
    def stability(self) -> float:
        """Calculate performance stability (lower = more stable)."""
        if len(self.r2_history) < 5:
            return 1.0
        recent = self.r2_history[-self.window_size:]
        return np.std(recent) if len(recent) > 1 else 0.0

    def update(
        self,
        r2: Optional[float] = None,
        mse: Optional[float] = None,
        latency: Optional[float] = None,
        uncertainty: Optional[float] = None,
    ) -> None:
        """Update metrics with new values."""
        if r2 is not None:
            self.r2_history.append(r2)
        if mse is not None:
            self.mse_history.append(mse)
        if latency is not None:
            self.latency_history.append(latency)
        if uncertainty is not None:
            self.uncertainty_history.append(uncertainty)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "recent_r2": self.recent_r2,
            "recent_mse": self.recent_mse,
            "recent_latency": self.recent_latency,
            "recent_uncertainty": self.recent_uncertainty,
            "performance_trend": self.performance_trend,
            "stability": self.stability,
            "n_updates": len(self.r2_history),
        }


@dataclass
class DecoderWrapper:
    """Wrapper for a decoder with its state and metrics."""
    decoder: BaseDecoder
    state: DecoderState = DecoderState.STANDBY
    metrics: DecoderMetrics = None
    weight: float = 1.0
    supports_uncertainty: bool = False

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = DecoderMetrics(name=self.decoder.name)

        # Check if decoder supports uncertainty estimation
        self.supports_uncertainty = hasattr(self.decoder, 'predict_with_uncertainty')


@dataclass
class PredictionResult:
    """Result from a decoder prediction."""
    decoder_name: str
    prediction: np.ndarray
    uncertainty: Optional[np.ndarray] = None
    latency_ms: float = 0.0

    @property
    def confidence(self) -> float:
        """Convert uncertainty to confidence score."""
        if self.uncertainty is None:
            return 1.0
        # Higher uncertainty = lower confidence
        mean_uncertainty = np.mean(self.uncertainty)
        return 1.0 / (1.0 + mean_uncertainty)


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    prediction: np.ndarray
    uncertainty: Optional[np.ndarray]
    decoder_weights: Dict[str, float]
    individual_predictions: Dict[str, np.ndarray]
    selected_decoders: List[str]
    total_latency_ms: float


class SelectionStrategy(Enum):
    """Strategy for decoder selection."""
    BEST = "best"                     # Select single best decoder
    TOP_K = "top_k"                   # Select top K decoders
    THRESHOLD = "threshold"           # Select all above threshold
    UNCERTAINTY_AWARE = "uncertainty" # Weight by uncertainty
    ADAPTIVE = "adaptive"             # Dynamically adjust strategy


class CombinationStrategy(Enum):
    """Strategy for combining decoder outputs."""
    MEAN = "mean"                     # Simple average
    WEIGHTED_MEAN = "weighted_mean"   # Weighted by performance
    MEDIAN = "median"                 # Robust median
    STACKING = "stacking"             # Meta-model stacking
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"  # Weight by confidence
