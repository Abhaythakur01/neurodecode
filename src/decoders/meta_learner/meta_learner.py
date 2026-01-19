"""
Adaptive Meta-Learner for Neural Decoding.

The core innovation: an adaptive system that automatically selects and
combines multiple decoders based on brain state, with online adaptation
and uncertainty quantification.

Reference:
    Novel approach combining ensemble methods with online adaptation
    for robust BCI decoding.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.decoders.base import BaseDecoder, OnlineDecoder
from src.decoders.meta_learner.adapter import OnlineAdapter
from src.decoders.meta_learner.base import (
    CombinationStrategy,
    DecoderState,
    DecoderWrapper,
    EnsembleResult,
    PredictionResult,
    SelectionStrategy,
)
from src.decoders.meta_learner.combiner import DecoderCombiner
from src.decoders.meta_learner.selector import DecoderSelector


class AdaptiveMetaLearner(OnlineDecoder):
    """
    Adaptive Meta-Learner for neural decoding.

    Orchestrates multiple base decoders, selecting and combining them
    dynamically based on performance and uncertainty. Handles:

    - Automatic decoder selection based on recent performance
    - Confidence-weighted ensemble combination
    - Online adaptation to changing neural signals
    - Electrode dropout detection and recovery
    - Uncertainty quantification for safe BCI operation
    """

    def __init__(
        self,
        name: str = "AdaptiveMetaLearner",
        # Selection parameters
        selection_strategy: SelectionStrategy = SelectionStrategy.ADAPTIVE,
        top_k: int = 3,
        selection_threshold: float = 0.5,
        # Combination parameters
        combination_strategy: CombinationStrategy = CombinationStrategy.UNCERTAINTY_WEIGHTED,
        # Adaptation parameters
        learning_rate: float = 0.1,
        adaptation_interval: int = 10,
        # Parallel execution
        parallel: bool = True,
        max_workers: int = 4,
        # Latency constraint
        max_latency_ms: float = 50.0,
        # Verbose
        verbose: bool = False,
    ):
        """
        Initialize Adaptive Meta-Learner.

        Args:
            name: Meta-learner name.
            selection_strategy: Strategy for decoder selection.
            top_k: Number of decoders for top-K selection.
            selection_threshold: Threshold for threshold-based selection.
            combination_strategy: Strategy for combining predictions.
            learning_rate: Learning rate for online adaptation.
            adaptation_interval: Steps between decoder updates.
            parallel: Whether to run decoders in parallel.
            max_workers: Maximum parallel workers.
            max_latency_ms: Maximum allowed latency in milliseconds.
            verbose: Print detailed information.
        """
        super().__init__(name=name, learning_rate=learning_rate)

        # Components
        self.selector = DecoderSelector(
            strategy=selection_strategy,
            top_k=top_k,
            performance_threshold=selection_threshold,
        )
        self.combiner = DecoderCombiner(
            strategy=combination_strategy,
        )
        self.adapter = OnlineAdapter(
            learning_rate=learning_rate,
            update_interval=adaptation_interval,
        )

        # Configuration
        self.parallel = parallel
        self.max_workers = max_workers
        self.max_latency_ms = max_latency_ms
        self.verbose = verbose

        # Decoder registry
        self._decoders: Dict[str, DecoderWrapper] = {}

        # State
        self._last_prediction: Optional[EnsembleResult] = None
        self._prediction_count = 0

        # Thread pool for parallel execution
        self._executor: Optional[ThreadPoolExecutor] = None

    def add_decoder(
        self,
        decoder: BaseDecoder,
        name: Optional[str] = None,
        weight: float = 1.0,
        active: bool = True,
    ) -> str:
        """
        Add a decoder to the meta-learner.

        Args:
            decoder: The decoder instance (should be fitted).
            name: Optional name (defaults to decoder.name).
            weight: Initial weight for this decoder.
            active: Whether decoder starts as active.

        Returns:
            Name assigned to the decoder.
        """
        if name is None:
            name = decoder.name

        # Ensure unique name
        if name in self._decoders:
            i = 1
            while f"{name}_{i}" in self._decoders:
                i += 1
            name = f"{name}_{i}"

        state = DecoderState.ACTIVE if active else DecoderState.STANDBY
        wrapper = DecoderWrapper(
            decoder=decoder,
            state=state,
            weight=weight,
        )

        self._decoders[name] = wrapper

        if self.verbose:
            print(f"Added decoder: {name} (state={state.value}, weight={weight})")

        return name

    def remove_decoder(self, name: str) -> bool:
        """
        Remove a decoder from the meta-learner.

        Args:
            name: Name of decoder to remove.

        Returns:
            True if decoder was removed.
        """
        if name in self._decoders:
            del self._decoders[name]
            return True
        return False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        decoders: Optional[List[BaseDecoder]] = None,
    ) -> "AdaptiveMetaLearner":
        """
        Fit the meta-learner.

        If decoders are provided, adds and fits them.
        If decoders already added, fits unfitted ones.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Targets of shape (n_samples, n_outputs).
            decoders: Optional list of decoders to add and fit.

        Returns:
            self: Fitted meta-learner.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1] if y.ndim > 1 else 1

        # Add provided decoders
        if decoders:
            for decoder in decoders:
                self.add_decoder(decoder)

        if not self._decoders:
            raise ValueError("No decoders registered. Add decoders before fitting.")

        # Fit unfitted decoders
        for name, wrapper in self._decoders.items():
            if not wrapper.decoder.is_fitted:
                if self.verbose:
                    print(f"Fitting decoder: {name}")
                wrapper.decoder.fit(X, y)

        # Initialize parallel executor
        if self.parallel:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Evaluate initial performance on training data
        self._evaluate_decoders(X, y)

        # Set baselines for adaptation
        self.adapter.set_baseline(self._decoders)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before prediction.")

        result = self.predict_with_info(X)
        return result.prediction

    def predict_with_info(
        self,
        X: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> EnsembleResult:
        """
        Make prediction with full ensemble information.

        Args:
            X: Neural features of shape (n_samples, n_features).
            context: Optional context for selection.

        Returns:
            EnsembleResult with prediction, uncertainty, and metadata.
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        start_time = time.perf_counter()

        # Select decoders to use
        selected = self.selector.select(self._decoders, context)

        if not selected:
            raise RuntimeError("No decoders available for prediction.")

        # Get predictions from selected decoders
        predictions = self._get_predictions(X, selected)

        # Combine predictions
        result = self.combiner.combine(predictions, self._decoders, selected)

        # Update timing
        result.total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Check latency constraint
        if result.total_latency_ms > self.max_latency_ms and self.verbose:
            print(f"Warning: Latency {result.total_latency_ms:.1f}ms exceeds "
                  f"target {self.max_latency_ms}ms")

        self._last_prediction = result
        self._prediction_count += 1

        return result

    def predict_single(self, x: np.ndarray) -> np.ndarray:
        """
        Single-step prediction for real-time use.

        Args:
            x: Single feature vector of shape (n_features,).

        Returns:
            Predicted values of shape (n_outputs,).
        """
        result = self.predict_with_info(x.reshape(1, -1))
        return result.prediction.flatten()

    def update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        adapt: bool = True,
    ) -> Dict[str, Any]:
        """
        Online update with new data.

        Args:
            X: New neural features.
            y: New target values.
            adapt: Whether to run adaptation (weight updates, etc.).

        Returns:
            Dictionary with update statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before update.")

        stats = {"updated_decoders": [], "adaptation": None}

        # Update metrics with true values
        if self._last_prediction is not None and adapt:
            adapt_stats = self.adapter.update(
                self._decoders,
                self._last_prediction,
                y,
                X,
            )
            stats["adaptation"] = adapt_stats

        # Update individual decoders
        for name, wrapper in self._decoders.items():
            if wrapper.state == DecoderState.DISABLED:
                continue

            if hasattr(wrapper.decoder, 'update'):
                try:
                    wrapper.decoder.update(X, y)
                    stats["updated_decoders"].append(name)
                except Exception as e:
                    if self.verbose:
                        print(f"Update failed for {name}: {e}")

        self._update_count += 1
        return stats

    def _get_predictions(
        self,
        X: np.ndarray,
        selected: List[str],
    ) -> Dict[str, PredictionResult]:
        """Get predictions from selected decoders."""
        predictions = {}

        if self.parallel and self._executor and len(selected) > 1:
            # Parallel execution
            futures = {}
            for name in selected:
                wrapper = self._decoders[name]
                future = self._executor.submit(
                    self._predict_single_decoder, wrapper, X
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=self.max_latency_ms / 1000)
                    predictions[name] = result
                except Exception as e:
                    if self.verbose:
                        print(f"Prediction failed for {name}: {e}")
        else:
            # Sequential execution
            for name in selected:
                wrapper = self._decoders[name]
                try:
                    result = self._predict_single_decoder(wrapper, X)
                    predictions[name] = result
                except Exception as e:
                    if self.verbose:
                        print(f"Prediction failed for {name}: {e}")

        return predictions

    def _predict_single_decoder(
        self,
        wrapper: DecoderWrapper,
        X: np.ndarray,
    ) -> PredictionResult:
        """Make prediction with a single decoder."""
        start_time = time.perf_counter()

        decoder = wrapper.decoder

        # Get prediction
        if wrapper.supports_uncertainty and hasattr(decoder, 'predict_with_uncertainty'):
            prediction, uncertainty = decoder.predict_with_uncertainty(X)
        else:
            prediction = decoder.predict(X)
            uncertainty = None

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Update latency metric
        wrapper.metrics.update(latency=latency_ms)

        return PredictionResult(
            decoder_name=wrapper.decoder.name,
            prediction=prediction,
            uncertainty=uncertainty,
            latency_ms=latency_ms,
        )

    def _evaluate_decoders(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate all decoders and update metrics."""
        scores = {}

        for name, wrapper in self._decoders.items():
            try:
                y_pred = wrapper.decoder.predict(X)

                # Handle shape mismatches
                if y_pred.shape[0] != y.shape[0]:
                    # Some decoders output different lengths
                    min_len = min(y_pred.shape[0], y.shape[0])
                    y_pred = y_pred[:min_len]
                    y_eval = y[:min_len]
                else:
                    y_eval = y

                # Compute R²
                ss_res = np.sum((y_eval - y_pred) ** 2)
                ss_tot = np.sum((y_eval - np.mean(y_eval, axis=0)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)

                # Compute MSE
                mse = np.mean((y_eval - y_pred) ** 2)

                # Update metrics
                wrapper.metrics.update(r2=r2, mse=mse)
                scores[name] = r2

                if self.verbose:
                    print(f"  {name}: R²={r2:.4f}, MSE={mse:.6f}")

            except Exception as e:
                if self.verbose:
                    print(f"  {name}: Evaluation failed - {e}")
                scores[name] = 0.0

        return scores

    def get_decoder_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current state of all decoders."""
        states = {}
        for name, wrapper in self._decoders.items():
            states[name] = {
                "state": wrapper.state.value,
                "weight": wrapper.weight,
                "metrics": wrapper.metrics.to_dict(),
                "supports_uncertainty": wrapper.supports_uncertainty,
            }
        return states

    def set_decoder_state(
        self,
        name: str,
        state: DecoderState,
    ) -> bool:
        """
        Manually set decoder state.

        Args:
            name: Decoder name.
            state: New state.

        Returns:
            True if successful.
        """
        if name not in self._decoders:
            return False

        self._decoders[name].state = state
        return True

    def get_params(self) -> Dict[str, Any]:
        """Get meta-learner parameters."""
        params = super().get_params()
        params.update({
            "n_decoders": len(self._decoders),
            "decoder_names": list(self._decoders.keys()),
            "selection_strategy": self.selector.strategy.value,
            "combination_strategy": self.combiner.strategy.value,
            "parallel": self.parallel,
            "max_latency_ms": self.max_latency_ms,
            "prediction_count": self._prediction_count,
        })

        # Add last prediction info
        if self._last_prediction:
            params["last_selected"] = self._last_prediction.selected_decoders
            params["last_latency_ms"] = self._last_prediction.total_latency_ms

        return params

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get decoder selection statistics."""
        return self.selector.get_selection_stats()

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return self.adapter.get_stats()

    def reset(self) -> None:
        """Reset meta-learner state (not decoder weights)."""
        self.selector.reset()
        self.adapter.reset()
        self._last_prediction = None
        self._prediction_count = 0

        # Reset decoder states
        for wrapper in self._decoders.values():
            wrapper.state = DecoderState.STANDBY

    def __del__(self):
        """Cleanup thread pool."""
        if self._executor:
            self._executor.shutdown(wait=False)


def create_default_meta_learner(
    decoders: List[BaseDecoder],
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = False,
) -> AdaptiveMetaLearner:
    """
    Create a meta-learner with default configuration.

    Convenience function for quick setup.

    Args:
        decoders: List of decoder instances.
        X_train: Training features.
        y_train: Training targets.
        verbose: Print progress.

    Returns:
        Fitted AdaptiveMetaLearner.
    """
    meta = AdaptiveMetaLearner(
        selection_strategy=SelectionStrategy.ADAPTIVE,
        combination_strategy=CombinationStrategy.UNCERTAINTY_WEIGHTED,
        top_k=min(3, len(decoders)),
        parallel=len(decoders) > 2,
        verbose=verbose,
    )

    # Add decoders
    for decoder in decoders:
        meta.add_decoder(decoder)

    # Fit
    meta.fit(X_train, y_train)

    return meta
