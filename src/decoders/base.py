"""
Base decoder abstract class for all neural decoders.

All decoders in the NeuroDecode system must inherit from BaseDecoder
and implement the required interface methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseDecoder(ABC):
    """
    Abstract base class for neural decoders.

    All decoders must implement fit(), predict(), and evaluate() methods.
    The update() method is optional for online learning capable decoders.

    Attributes:
        is_fitted: Whether the decoder has been trained.
        n_features: Number of input features (set after fitting).
        n_outputs: Number of output dimensions (set after fitting).
        name: Human-readable name for the decoder.
    """

    def __init__(self, name: str = "BaseDecoder"):
        self.name = name
        self.is_fitted = False
        self.n_features: Optional[int] = None
        self.n_outputs: Optional[int] = None
        self._model: Any = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecoder":
        """
        Train the decoder on neural data and target kinematics.

        Args:
            X: Neural features array of shape (n_samples, n_features).
            y: Target kinematics array of shape (n_samples, n_outputs).

        Returns:
            self: The fitted decoder instance.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode movement from neural features.

        Args:
            X: Neural features array of shape (n_samples, n_features).

        Returns:
            Predicted kinematics array of shape (n_samples, n_outputs).
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate decoder performance using R² score.

        Args:
            X: Neural features array of shape (n_samples, n_features).
            y: True kinematics array of shape (n_samples, n_outputs).

        Returns:
            Dictionary containing evaluation metrics including 'r2' score.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before evaluation.")

        y_pred = self.predict(X)
        r2 = self._compute_r2(y, y_pred)
        mse = self._compute_mse(y, y_pred)

        return {"r2": r2, "mse": mse}

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online learning update (optional).

        Override this method for decoders that support online adaptation.

        Args:
            X: Neural features array of shape (n_samples, n_features).
            y: Target kinematics array of shape (n_samples, n_outputs).
        """
        raise NotImplementedError(
            f"{self.name} does not support online learning. "
            "Override update() to enable this feature."
        )

    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute R² (coefficient of determination) score.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            R² score (1.0 is perfect, 0.0 is baseline, negative is worse).
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1.0 - (ss_res / ss_tot)

    def _compute_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            MSE value.
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data shapes and types.

        Args:
            X: Neural features array.
            y: Optional target array.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if y is not None:
            if y.ndim != 2:
                raise ValueError(f"y must be 2D array, got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples. "
                    f"Got X: {X.shape[0]}, y: {y.shape[0]}"
                )

    def get_params(self) -> Dict[str, Any]:
        """
        Get decoder parameters.

        Returns:
            Dictionary of decoder parameters.
        """
        return {"name": self.name, "is_fitted": self.is_fitted}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"


class OnlineDecoder(BaseDecoder):
    """
    Base class for decoders that support online learning.

    Extends BaseDecoder with additional methods for streaming data
    and incremental updates.
    """

    def __init__(self, name: str = "OnlineDecoder", learning_rate: float = 0.01):
        super().__init__(name=name)
        self.learning_rate = learning_rate
        self._update_count = 0

    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Perform online update with new data.

        Args:
            X: Neural features array of shape (n_samples, n_features).
            y: Target kinematics array of shape (n_samples, n_outputs).
        """
        pass

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> "OnlineDecoder":
        """
        Incrementally fit the decoder (alias for update after initial fit).

        Args:
            X: Neural features array.
            y: Target kinematics array.

        Returns:
            self: The updated decoder instance.
        """
        if not self.is_fitted:
            return self.fit(X, y)
        self.update(X, y)
        return self

    def get_params(self) -> Dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "learning_rate": self.learning_rate,
                "update_count": self._update_count,
            }
        )
        return params
