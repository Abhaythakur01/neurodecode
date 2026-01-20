"""
Decoder service for real-time neural decoding.

Integrates the AdaptiveMetaLearner with the FastAPI backend,
handling calibration, prediction, and state management.
"""

import logging
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.backend.config import settings
from src.backend.models.schemas import DecoderInfo, DecoderStateEnum
from src.decoders import (
    AdaptiveMetaLearner,
    CombinationStrategy,
    KalmanFilterDecoder,
    RandomForestDecoder,
    SelectionStrategy,
    SteadyStateKalmanFilter,
    SVMDecoder,
    WienerFilterDecoder,
)
from src.decoders.meta_learner.base import EnsembleResult

logger = logging.getLogger(__name__)


class DecoderService:
    """
    Service for managing the AdaptiveMetaLearner.

    Provides thread-safe access to the meta-learner for predictions,
    tracks performance metrics, and handles calibration.
    """

    def __init__(self):
        """Initialize decoder service."""
        self._meta_learner: Optional[AdaptiveMetaLearner] = None
        self._is_ready = False
        self._lock = Lock()

        # Performance tracking
        self._latency_history: deque = deque(maxlen=100)
        self._prediction_count = 0
        self._last_prediction_time = 0.0

        # Calibration data
        self._calibration_X: Optional[np.ndarray] = None
        self._calibration_y: Optional[np.ndarray] = None

        logger.info("DecoderService initialized")

    @property
    def is_ready(self) -> bool:
        """Check if meta-learner is calibrated and ready."""
        return self._is_ready

    @property
    def average_latency(self) -> float:
        """Get average prediction latency in ms."""
        if not self._latency_history:
            return 0.0
        return np.mean(self._latency_history)

    @property
    def predictions_per_second(self) -> float:
        """Get approximate predictions per second."""
        if self.average_latency == 0:
            return 0.0
        return 1000.0 / self.average_latency

    def initialize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        include_decoders: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Initialize and calibrate the meta-learner.

        Args:
            X_train: Training neural features (n_samples, n_neurons)
            y_train: Training targets (n_samples, n_outputs)
            include_decoders: Optional list of decoder names to include

        Returns:
            Dictionary of decoder names to their initial R² scores
        """
        with self._lock:
            logger.info(
                f"Initializing meta-learner with data shape: "
                f"X={X_train.shape}, y={y_train.shape}"
            )

            # Store calibration data
            self._calibration_X = X_train
            self._calibration_y = y_train

            # Create decoders
            decoders = self._create_decoders(include_decoders)

            # Create meta-learner
            self._meta_learner = AdaptiveMetaLearner(
                name="BCIMetaLearner",
                selection_strategy=SelectionStrategy.ADAPTIVE,
                combination_strategy=CombinationStrategy.UNCERTAINTY_WEIGHTED,
                top_k=settings.meta_learner_top_k,
                parallel=settings.meta_learner_parallel,
                max_latency_ms=settings.max_latency_ms,
                verbose=settings.debug,
            )

            # Add decoders
            for decoder in decoders:
                self._meta_learner.add_decoder(decoder)

            # Fit meta-learner
            start_time = time.perf_counter()
            self._meta_learner.fit(X_train, y_train)
            fit_time = (time.perf_counter() - start_time) * 1000

            logger.info(f"Meta-learner fitted in {fit_time:.1f}ms")

            # Get initial scores
            scores = {}
            decoder_states = self._meta_learner.get_decoder_states()
            for name, state in decoder_states.items():
                scores[name] = state["metrics"]["recent_r2"]

            self._is_ready = True
            logger.info(f"Meta-learner ready. Decoder scores: {scores}")

            return scores

    def _create_decoders(self, include_decoders: Optional[List[str]] = None) -> List[Any]:
        """Create decoder instances based on configuration."""
        all_decoders = {
            "Kalman": lambda: KalmanFilterDecoder(name="Kalman"),
            "SteadyKalman": lambda: SteadyStateKalmanFilter(name="SteadyKalman"),
            "Wiener": lambda: WienerFilterDecoder(name="Wiener", n_lags=5),
            "SVM": lambda: SVMDecoder(name="SVM", kernel="rbf", C=1.0),
            "RandomForest": lambda: RandomForestDecoder(
                name="RandomForest", n_estimators=50, max_depth=10
            ),
        }

        if include_decoders is None:
            # Default: use all decoders
            include_decoders = list(all_decoders.keys())

        decoders = []
        for name in include_decoders:
            if name in all_decoders:
                try:
                    decoder = all_decoders[name]()
                    decoders.append(decoder)
                    logger.info(f"Created decoder: {name}")
                except Exception as e:
                    logger.warning(f"Failed to create decoder {name}: {e}")

        if not decoders:
            raise ValueError("No decoders could be created")

        return decoders

    def decode(self, features: np.ndarray) -> Tuple[EnsembleResult, float]:
        """
        Decode neural features to movement prediction.

        Args:
            features: Neural features of shape (n_samples, n_neurons)
                     or (n_neurons,) for single sample

        Returns:
            Tuple of (EnsembleResult, latency_ms)

        Raises:
            RuntimeError: If meta-learner is not ready
        """
        if not self._is_ready:
            raise RuntimeError("Meta-learner not initialized. Call initialize() first.")

        start_time = time.perf_counter()

        with self._lock:
            # Ensure 2D input
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Get prediction
            result = self._meta_learner.predict_with_info(features)

        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self._latency_history.append(latency_ms)
        self._prediction_count += 1
        self._last_prediction_time = time.time()

        return result, latency_ms

    def update_online(self, features: np.ndarray, true_target: np.ndarray) -> Dict[str, Any]:
        """
        Online update with true target feedback.

        Args:
            features: Neural features used for last prediction
            true_target: Actual target values

        Returns:
            Update statistics
        """
        if not self._is_ready:
            raise RuntimeError("Meta-learner not initialized")

        with self._lock:
            return self._meta_learner.update(features, true_target)

    def get_decoder_states(self) -> List[DecoderInfo]:
        """Get current state of all decoders."""
        if not self._is_ready:
            return []

        with self._lock:
            states = self._meta_learner.get_decoder_states()

        decoder_infos = []
        for name, state in states.items():
            decoder_infos.append(
                DecoderInfo(
                    name=name,
                    state=DecoderStateEnum(state["state"]),
                    weight=state["weight"],
                    r2_score=state["metrics"]["recent_r2"],
                    latency_ms=state["metrics"]["recent_latency"],
                    uncertainty=state["metrics"]["recent_uncertainty"],
                )
            )

        return decoder_infos

    def get_meta_learner_state(self) -> Dict[str, Any]:
        """Get full meta-learner state."""
        if not self._is_ready:
            return {"status": "not_initialized"}

        with self._lock:
            params = self._meta_learner.get_params()
            selection_stats = self._meta_learner.get_selection_stats()
            adaptation_stats = self._meta_learner.get_adaptation_stats()

        return {
            "status": "ready",
            "params": params,
            "selection_stats": selection_stats,
            "adaptation_stats": adaptation_stats,
            "service_stats": {
                "prediction_count": self._prediction_count,
                "average_latency_ms": self.average_latency,
                "predictions_per_second": self.predictions_per_second,
            },
        }

    def recalibrate(
        self,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        include_decoders: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Recalibrate the meta-learner.

        Args:
            X_train: New training features (uses stored if None)
            y_train: New training targets (uses stored if None)
            include_decoders: Decoders to include

        Returns:
            New decoder scores
        """
        if X_train is None:
            X_train = self._calibration_X
        if y_train is None:
            y_train = self._calibration_y

        if X_train is None or y_train is None:
            raise ValueError("No calibration data available")

        logger.info("Recalibrating meta-learner...")

        # Reset state
        self._is_ready = False
        self._latency_history.clear()
        self._prediction_count = 0

        # Reinitialize
        return self.initialize(X_train, y_train, include_decoders)

    def shutdown(self):
        """Clean up resources."""
        with self._lock:
            if self._meta_learner is not None:
                # Cleanup executor if exists
                if hasattr(self._meta_learner, "_executor"):
                    if self._meta_learner._executor:
                        self._meta_learner._executor.shutdown(wait=False)

            self._meta_learner = None
            self._is_ready = False

        logger.info("DecoderService shut down")


# Global singleton instance
decoder_service = DecoderService()
