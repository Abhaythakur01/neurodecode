"""
Wiener Filter decoder for neural signals.

Implements the optimal linear filter that minimizes mean squared error
between decoded and actual kinematics. Supports multiple time lags
to capture temporal dynamics in neural-kinematic relationships.

Reference:
    Carmena et al. (2003) "Learning to control a brain-machine interface
    for reaching and grasping by primates" PLoS Biology
"""

from typing import Any, Dict, Optional

import numpy as np

from src.decoders.base import BaseDecoder


class WienerFilterDecoder(BaseDecoder):
    """
    Wiener Filter decoder for neural-to-kinematic mapping.

    The Wiener filter finds the optimal linear weights W that minimize:
        ||y - X @ W||^2

    where X can include multiple time lags of neural activity.

    Attributes:
        n_lags: Number of time lags to include (0 = current only).
        regularization: L2 regularization strength (ridge regression).
        weights: Learned filter weights after fitting.
    """

    def __init__(
        self,
        name: str = "WienerFilter",
        n_lags: int = 10,
        regularization: float = 1e-4,
    ):
        """
        Initialize Wiener Filter decoder.

        Args:
            name: Decoder name.
            n_lags: Number of past time bins to include (default 10).
                Total features = n_features * (n_lags + 1).
            regularization: L2 regularization parameter (ridge).
                Higher values prevent overfitting but may reduce accuracy.
        """
        super().__init__(name=name)
        self.n_lags = n_lags
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self._original_n_features: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WienerFilterDecoder":
        """
        Fit Wiener Filter using regularized least squares.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self._original_n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Create lagged feature matrix
        X_lagged = self._create_lagged_features(X)
        y_aligned = y[self.n_lags :]  # Align targets with lagged features

        self.n_features = X_lagged.shape[1]

        # Ridge regression: W = (X^T X + λI)^{-1} X^T y
        XtX = X_lagged.T @ X_lagged
        XtX += self.regularization * np.eye(XtX.shape[0])
        Xty = X_lagged.T @ y_aligned

        self.weights = np.linalg.solve(XtX, Xty)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode kinematics from neural features.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Decoded kinematics of shape (n_samples - n_lags, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Create lagged features
        X_lagged = self._create_lagged_features(X)

        # Linear prediction
        return X_lagged @ self.weights

    def _create_lagged_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create feature matrix with time-lagged copies.

        Args:
            X: Original features of shape (n_samples, n_features).

        Returns:
            Lagged features of shape (n_samples - n_lags, n_features * (n_lags + 1)).
        """
        n_samples, n_features = X.shape
        n_output_samples = n_samples - self.n_lags

        if n_output_samples <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_lags} lags. "
                f"Need at least {self.n_lags + 1} samples."
            )

        # Create lagged feature matrix
        # Column order: [X(t), X(t-1), X(t-2), ..., X(t-n_lags)]
        X_lagged = np.zeros((n_output_samples, n_features * (self.n_lags + 1)))

        for lag in range(self.n_lags + 1):
            start_col = lag * n_features
            end_col = (lag + 1) * n_features
            # X(t-lag) for samples from n_lags to end
            X_lagged[:, start_col:end_col] = X[self.n_lags - lag : n_samples - lag]

        return X_lagged

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "n_lags": self.n_lags,
                "regularization": self.regularization,
                "original_n_features": self._original_n_features,
            }
        )
        return params


class CausalWienerFilter(WienerFilterDecoder):
    """
    Causal Wiener Filter that only uses past neural activity.

    This is the standard formulation for real-time BCI applications
    where future neural data is not available.
    """

    def __init__(
        self,
        name: str = "CausalWiener",
        n_lags: int = 10,
        regularization: float = 1e-4,
    ):
        super().__init__(name=name, n_lags=n_lags, regularization=regularization)


class NonCausalWienerFilter(WienerFilterDecoder):
    """
    Non-causal Wiener Filter that uses both past and future neural activity.

    Useful for offline analysis where future data is available.
    Generally achieves higher accuracy than causal filters.
    """

    def __init__(
        self,
        name: str = "NonCausalWiener",
        n_lags_past: int = 5,
        n_lags_future: int = 5,
        regularization: float = 1e-4,
    ):
        """
        Initialize non-causal Wiener Filter.

        Args:
            name: Decoder name.
            n_lags_past: Number of past time bins.
            n_lags_future: Number of future time bins.
            regularization: L2 regularization parameter.
        """
        # Total lags for parent class
        super().__init__(
            name=name,
            n_lags=n_lags_past + n_lags_future,
            regularization=regularization,
        )
        self.n_lags_past = n_lags_past
        self.n_lags_future = n_lags_future

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NonCausalWienerFilter":
        """Fit non-causal Wiener Filter."""
        self._validate_input(X, y)

        self._original_n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Create lagged feature matrix (past and future)
        X_lagged = self._create_noncausal_features(X)

        # Align targets: skip first n_lags_future and last n_lags_past
        y_aligned = y[self.n_lags_future : -self.n_lags_past if self.n_lags_past > 0 else None]

        self.n_features = X_lagged.shape[1]

        # Ridge regression
        XtX = X_lagged.T @ X_lagged
        XtX += self.regularization * np.eye(XtX.shape[0])
        Xty = X_lagged.T @ y_aligned

        self.weights = np.linalg.solve(XtX, Xty)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using non-causal filter."""
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        X_lagged = self._create_noncausal_features(X)
        return X_lagged @ self.weights

    def _create_noncausal_features(self, X: np.ndarray) -> np.ndarray:
        """Create feature matrix with past and future lags."""
        n_samples, n_features = X.shape
        total_lags = self.n_lags_past + self.n_lags_future + 1
        n_output = n_samples - self.n_lags_past - self.n_lags_future

        if n_output <= 0:
            raise ValueError(
                f"Not enough samples ({n_samples}) for lags. "
                f"Need at least {total_lags} samples."
            )

        X_lagged = np.zeros((n_output, n_features * total_lags))

        # Future lags: X(t+n_lags_future), ..., X(t+1)
        for i, lag in enumerate(range(self.n_lags_future, 0, -1)):
            start_col = i * n_features
            end_col = (i + 1) * n_features
            X_lagged[:, start_col:end_col] = X[
                self.n_lags_future + lag : n_samples - self.n_lags_past + lag
            ]

        # Current: X(t)
        idx = self.n_lags_future
        X_lagged[:, idx * n_features : (idx + 1) * n_features] = X[
            self.n_lags_future : -self.n_lags_past if self.n_lags_past > 0 else None
        ]

        # Past lags: X(t-1), ..., X(t-n_lags_past)
        for i, lag in enumerate(range(1, self.n_lags_past + 1)):
            idx = self.n_lags_future + 1 + i
            start_col = idx * n_features
            end_col = (idx + 1) * n_features
            X_lagged[:, start_col:end_col] = X[
                self.n_lags_future - lag : n_samples - self.n_lags_past - lag
            ]

        return X_lagged

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "n_lags_past": self.n_lags_past,
                "n_lags_future": self.n_lags_future,
            }
        )
        return params
