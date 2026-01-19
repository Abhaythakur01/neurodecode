"""
Gaussian Process decoder for neural signals.

Implements GP regression for neural decoding with built-in
uncertainty quantification - critical for BCI safety.

Reference:
    Rasmussen & Williams (2006) "Gaussian Processes for Machine Learning"
    Wu et al. (2006) "Bayesian Population Decoding of Motor Cortical Activity
    Using a Kalman Filter" Neural Comput
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

from src.decoders.base import BaseDecoder


class GaussianProcessDecoder(BaseDecoder):
    """
    Gaussian Process decoder for continuous kinematic decoding.

    Provides uncertainty estimates for each prediction, which is critical
    for safe BCI operation - high uncertainty can trigger fallback modes.
    """

    def __init__(
        self,
        name: str = "GaussianProcess",
        kernel: Optional[str] = "rbf",
        length_scale: float = 1.0,
        length_scale_bounds: Tuple[float, float] = (1e-3, 1e3),
        noise_level: float = 1e-2,
        noise_level_bounds: Tuple[float, float] = (1e-5, 1e1),
        normalize_y: bool = True,
        normalize_X: bool = True,
        n_restarts_optimizer: int = 3,
        alpha: float = 1e-10,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Gaussian Process decoder.

        Args:
            name: Decoder name.
            kernel: Kernel type ('rbf', 'matern', 'rational_quadratic', 'linear').
            length_scale: Initial length scale for kernel.
            length_scale_bounds: Bounds for length scale optimization.
            noise_level: Initial noise level.
            noise_level_bounds: Bounds for noise level optimization.
            normalize_y: Whether to normalize targets.
            normalize_X: Whether to normalize features.
            n_restarts_optimizer: Number of optimizer restarts.
            alpha: Value added to diagonal for numerical stability.
            random_state: Random seed.
        """
        super().__init__(name=name)

        self.kernel_type = kernel
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha
        self.random_state = random_state

        self._models = []
        self._scaler = StandardScaler() if normalize_X else None

    def _build_kernel(self):
        """Build the GP kernel."""
        # Base kernel based on type
        if self.kernel_type == "rbf":
            base_kernel = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
            )
        elif self.kernel_type == "matern":
            base_kernel = Matern(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
                nu=2.5,
            )
        elif self.kernel_type == "rational_quadratic":
            base_kernel = RationalQuadratic(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
            )
        elif self.kernel_type == "linear":
            base_kernel = DotProduct(sigma_0=1.0)
        else:
            base_kernel = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=self.length_scale_bounds,
            )

        # Combine with constant kernel and white noise
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + WhiteKernel(
            noise_level=self.noise_level,
            noise_level_bounds=self.noise_level_bounds,
        )

        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessDecoder":
        """
        Fit Gaussian Process decoder.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1] if y.ndim > 1 else 1

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Normalize features
        if self.normalize_X:
            X = self._scaler.fit_transform(X)

        # Train one GP per output dimension
        self._models = []

        for i in range(self.n_outputs):
            kernel = self._build_kernel()

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=self.n_restarts_optimizer,
                alpha=self.alpha,
                random_state=self.random_state,
            )

            gp.fit(X, y[:, i])
            self._models.append(gp)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode kinematics from neural features.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Decoded kinematics of shape (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        if self.normalize_X:
            X = self._scaler.transform(X)

        # Predict from each model
        predictions = np.zeros((X.shape[0], self.n_outputs))
        for i, model in enumerate(self._models):
            predictions[:, i] = model.predict(X)

        return predictions

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode kinematics with uncertainty estimates.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Tuple of (predictions, standard_deviations).
            Both have shape (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        if self.normalize_X:
            X = self._scaler.transform(X)

        # Predict with uncertainty
        predictions = np.zeros((X.shape[0], self.n_outputs))
        uncertainties = np.zeros((X.shape[0], self.n_outputs))

        for i, model in enumerate(self._models):
            mean, std = model.predict(X, return_std=True)
            predictions[:, i] = mean
            uncertainties[:, i] = std

        return predictions, uncertainties

    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction uncertainty only.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Standard deviations of shape (n_samples, n_outputs).
        """
        _, uncertainty = self.predict_with_uncertainty(X)
        return uncertainty

    def sample_predictions(
        self, X: np.ndarray, n_samples: int = 10
    ) -> np.ndarray:
        """
        Sample from the posterior predictive distribution.

        Args:
            X: Neural features of shape (n_samples, n_features).
            n_samples: Number of samples to draw.

        Returns:
            Samples of shape (n_samples, n_data_points, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Normalize
        if self.normalize_X:
            X = self._scaler.transform(X)

        # Sample from each model
        samples = np.zeros((n_samples, X.shape[0], self.n_outputs))

        for i, model in enumerate(self._models):
            samples[:, :, i] = model.sample_y(
                X, n_samples=n_samples, random_state=self.random_state
            ).T

        return samples

    def get_kernel_params(self) -> Dict[str, Any]:
        """
        Get optimized kernel parameters.

        Returns:
            Dictionary of kernel parameters for each output.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted first.")

        params = {}
        for i, model in enumerate(self._models):
            params[f"output_{i}"] = {
                "kernel": str(model.kernel_),
                "log_marginal_likelihood": model.log_marginal_likelihood_value_,
            }

        return params

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update({
            "kernel": self.kernel_type,
            "length_scale": self.length_scale,
            "noise_level": self.noise_level,
            "normalize_y": self.normalize_y,
            "normalize_X": self.normalize_X,
        })
        if self.is_fitted:
            params["n_outputs"] = self.n_outputs
        return params


class SparseGPDecoder(BaseDecoder):
    """
    Sparse Gaussian Process decoder for large datasets.

    Uses inducing points to reduce computational complexity
    from O(n³) to O(nm²) where m << n.
    """

    def __init__(
        self,
        name: str = "SparseGP",
        n_inducing: int = 100,
        kernel: str = "rbf",
        length_scale: float = 1.0,
        normalize_X: bool = True,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Sparse GP decoder.

        Args:
            name: Decoder name.
            n_inducing: Number of inducing points.
            kernel: Kernel type.
            length_scale: Kernel length scale.
            normalize_X: Normalize features.
            random_state: Random seed.
        """
        super().__init__(name=name)

        self.n_inducing = n_inducing
        self.kernel_type = kernel
        self.length_scale = length_scale
        self.normalize_X = normalize_X
        self.random_state = random_state

        self._scaler = StandardScaler() if normalize_X else None
        self._inducing_points = None
        self._models = []

    def _select_inducing_points(self, X: np.ndarray) -> np.ndarray:
        """Select inducing points using k-means."""
        if len(X) <= self.n_inducing:
            return X.copy()

        # Simple k-means for inducing point selection
        np.random.seed(self.random_state)

        # Initialize with random points
        idx = np.random.choice(len(X), self.n_inducing, replace=False)
        inducing = X[idx].copy()

        # K-means iterations
        for _ in range(10):
            # Assign points to nearest inducing point
            distances = np.array([
                np.sum((X - inducing[k]) ** 2, axis=1)
                for k in range(self.n_inducing)
            ]).T
            assignments = np.argmin(distances, axis=1)

            # Update inducing points
            new_inducing = np.array([
                X[assignments == k].mean(axis=0) if np.any(assignments == k)
                else inducing[k]
                for k in range(self.n_inducing)
            ])

            if np.allclose(inducing, new_inducing):
                break
            inducing = new_inducing

        return inducing

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparseGPDecoder":
        """
        Fit Sparse GP decoder.

        Uses Nyström approximation with inducing points.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1] if y.ndim > 1 else 1

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Normalize
        if self.normalize_X:
            X = self._scaler.fit_transform(X)

        # Select inducing points
        self._inducing_points = self._select_inducing_points(X)

        # Fit GP on inducing points
        # For simplicity, use standard GP on subset
        self._models = []

        for i in range(self.n_outputs):
            # Find nearest training point for each inducing point
            distances = np.array([
                np.sum((X - self._inducing_points[k]) ** 2, axis=1)
                for k in range(len(self._inducing_points))
            ])
            nearest_idx = np.argmin(distances, axis=1)
            y_inducing = y[nearest_idx, i]

            # Build kernel
            if self.kernel_type == "rbf":
                kernel = ConstantKernel() * RBF(length_scale=self.length_scale)
            else:
                kernel = ConstantKernel() * Matern(length_scale=self.length_scale)

            kernel = kernel + WhiteKernel(noise_level=0.1)

            gp = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                random_state=self.random_state,
            )

            gp.fit(self._inducing_points, y_inducing)
            self._models.append(gp)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sparse GP."""
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize_X:
            X = self._scaler.transform(X)

        predictions = np.zeros((X.shape[0], self.n_outputs))
        for i, model in enumerate(self._models):
            predictions[:, i] = model.predict(X)

        return predictions

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates."""
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize_X:
            X = self._scaler.transform(X)

        predictions = np.zeros((X.shape[0], self.n_outputs))
        uncertainties = np.zeros((X.shape[0], self.n_outputs))

        for i, model in enumerate(self._models):
            mean, std = model.predict(X, return_std=True)
            predictions[:, i] = mean
            uncertainties[:, i] = std

        return predictions, uncertainties

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update({
            "n_inducing": self.n_inducing,
            "kernel": self.kernel_type,
            "length_scale": self.length_scale,
        })
        return params


class GPClassifier(BaseDecoder):
    """
    Gaussian Process classifier for discrete state decoding.

    Uses Laplace approximation for classification with
    uncertainty estimates.
    """

    def __init__(
        self,
        name: str = "GPClassifier",
        kernel: str = "rbf",
        length_scale: float = 1.0,
        normalize_X: bool = True,
        n_restarts_optimizer: int = 3,
        max_iter_predict: int = 100,
        random_state: Optional[int] = None,
    ):
        """
        Initialize GP classifier.

        Args:
            name: Decoder name.
            kernel: Kernel type.
            length_scale: Kernel length scale.
            normalize_X: Normalize features.
            n_restarts_optimizer: Optimizer restarts.
            max_iter_predict: Max iterations for prediction.
            random_state: Random seed.
        """
        super().__init__(name=name)

        self.kernel_type = kernel
        self.length_scale = length_scale
        self.normalize_X = normalize_X
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.random_state = random_state

        self._model = None
        self._scaler = StandardScaler() if normalize_X else None
        self.classes_: Optional[np.ndarray] = None

    def _build_kernel(self):
        """Build the GP kernel for classification."""
        if self.kernel_type == "rbf":
            kernel = ConstantKernel() * RBF(length_scale=self.length_scale)
        elif self.kernel_type == "matern":
            kernel = ConstantKernel() * Matern(length_scale=self.length_scale)
        else:
            kernel = ConstantKernel() * RBF(length_scale=self.length_scale)

        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPClassifier":
        """
        Fit GP classifier.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).

        Returns:
            self: Fitted classifier.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)

        # Normalize
        if self.normalize_X:
            X = self._scaler.fit_transform(X)

        # Build model
        kernel = self._build_kernel()

        self._model = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            random_state=self.random_state,
        )

        self._model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize_X:
            X = self._scaler.transform(X)

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize_X:
            X = self._scaler.transform(X)

        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate classifier."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before evaluation.")

        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        metrics = {
            "accuracy": accuracy,
            "r2": accuracy,
            "mse": 1.0 - accuracy,
        }

        for c in self.classes_:
            mask = y == c
            if np.any(mask):
                metrics[f"accuracy_class_{c}"] = np.mean(y_pred[mask] == y[mask])

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        params = super().get_params()
        params.update({
            "kernel": self.kernel_type,
            "length_scale": self.length_scale,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
        })
        return params
