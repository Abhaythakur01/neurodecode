"""
Linear Discriminant Analysis decoder for discrete state classification.

Implements LDA for classifying discrete states (e.g., movement directions,
mental states) from neural activity patterns.

Reference:
    Santhanam et al. (2006) "A high-performance brain-computer interface"
    Nature
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.decoders.base import BaseDecoder


class LDADecoder(BaseDecoder):
    """
    Linear Discriminant Analysis decoder for discrete classification.

    LDA finds linear combinations of features that best separate classes
    by maximizing between-class variance while minimizing within-class variance.

    Supports:
    - Multi-class classification
    - Dimensionality reduction (projection to LDA space)
    - Probability estimation via softmax
    """

    def __init__(
        self,
        name: str = "LDA",
        n_components: Optional[int] = None,
        regularization: float = 1e-4,
        prior: Optional[np.ndarray] = None,
    ):
        """
        Initialize LDA decoder.

        Args:
            name: Decoder name.
            n_components: Number of discriminant components to use.
                If None, uses min(n_classes - 1, n_features).
            regularization: Regularization for covariance estimation.
            prior: Prior probabilities for each class.
                If None, estimated from training data.
        """
        super().__init__(name=name)
        self.n_components = n_components
        self.regularization = regularization
        self.prior = prior

        # Learned parameters
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None  # Class means
        self.covariance_: Optional[np.ndarray] = None  # Pooled covariance
        self.scalings_: Optional[np.ndarray] = None  # LDA projection matrix
        self.intercept_: Optional[np.ndarray] = None  # Decision intercepts

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDADecoder":
        """
        Fit LDA decoder on training data.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,) - must be discrete integers.

        Returns:
            self: Fitted decoder.
        """
        if y.ndim > 1:
            y = y.ravel()

        self._validate_classification_input(X, y)

        self.n_features = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_outputs = 1  # Classification output

        # Compute class priors
        if self.prior is not None:
            self.priors_ = np.array(self.prior)
        else:
            self.priors_ = np.array([np.mean(y == c) for c in self.classes_])

        # Compute class means
        self.means_ = np.zeros((self.n_classes_, self.n_features))
        for i, c in enumerate(self.classes_):
            self.means_[i] = X[y == c].mean(axis=0)

        # Compute pooled within-class covariance
        self.covariance_ = self._compute_pooled_covariance(X, y)

        # Regularize covariance
        self.covariance_ += self.regularization * np.eye(self.n_features)

        # Compute LDA scalings (projection matrix)
        self._compute_scalings()

        # Compute intercepts for decision function
        self._compute_intercept()

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Compute decision function values
        scores = self._decision_function(X)

        # Return class with highest score
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using softmax.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = self._decision_function(X)

        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto LDA discriminant space.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Projected data of shape (n_samples, n_components).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before transform.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X @ self.scalings_

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classification performance.

        Args:
            X: Neural features.
            y: True class labels.

        Returns:
            Dictionary with accuracy and other metrics.
        """
        if y.ndim > 1:
            y = y.ravel()

        y_pred = self.predict(X)

        accuracy = np.mean(y_pred == y)

        # Per-class accuracy
        class_acc = {}
        for c in self.classes_:
            mask = y == c
            if np.sum(mask) > 0:
                class_acc[f"accuracy_class_{c}"] = float(np.mean(y_pred[mask] == c))

        return {
            "accuracy": float(accuracy),
            "r2": float(accuracy),  # For compatibility with BaseDecoder
            "mse": float(1 - accuracy),
            **class_acc,
        }

    def _validate_classification_input(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate input for classification."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D for classification, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
            )

    def _compute_pooled_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute pooled within-class covariance matrix."""
        n_samples = X.shape[0]
        cov = np.zeros((self.n_features, self.n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            X_c_centered = X_c - self.means_[i]
            cov += X_c_centered.T @ X_c_centered

        # Normalize by total samples - n_classes (unbiased estimate)
        cov /= (n_samples - self.n_classes_)

        return cov

    def _compute_scalings(self) -> None:
        """Compute LDA projection matrix."""
        # Compute between-class scatter matrix
        overall_mean = np.mean(self.means_, axis=0)
        Sb = np.zeros((self.n_features, self.n_features))

        for i, c in enumerate(self.classes_):
            n_c = self.priors_[i] * 1  # Will scale by total later
            diff = (self.means_[i] - overall_mean).reshape(-1, 1)
            Sb += n_c * (diff @ diff.T)

        # Solve generalized eigenvalue problem: Sb @ v = λ @ Sw @ v
        # Equivalent to: inv(Sw) @ Sb @ v = λ @ v
        Sw_inv = np.linalg.inv(self.covariance_)
        M = Sw_inv @ Sb

        eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top components
        n_components = self.n_components
        if n_components is None:
            n_components = min(self.n_classes_ - 1, self.n_features)

        self.scalings_ = eigenvectors[:, :n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / np.sum(eigenvalues)

    def _compute_intercept(self) -> None:
        """Compute decision function intercepts."""
        # For LDA: decision = X @ Sigma^{-1} @ mu_k - 0.5 * mu_k @ Sigma^{-1} @ mu_k + log(prior_k)
        Sw_inv = np.linalg.inv(self.covariance_)

        self.coef_ = self.means_ @ Sw_inv  # (n_classes, n_features)
        self.intercept_ = (
            -0.5 * np.sum(self.means_ * (self.means_ @ Sw_inv), axis=1)
            + np.log(self.priors_)
        )

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values for all classes."""
        return X @ self.coef_.T + self.intercept_

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update({
            "n_components": self.n_components,
            "regularization": self.regularization,
            "n_classes": self.n_classes_,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        })
        return params


class ShrinkageLDA(LDADecoder):
    """
    LDA with Ledoit-Wolf shrinkage for covariance estimation.

    More robust when number of features is large relative to samples.
    """

    def __init__(
        self,
        name: str = "ShrinkageLDA",
        n_components: Optional[int] = None,
        shrinkage: Optional[float] = None,
    ):
        """
        Initialize Shrinkage LDA.

        Args:
            name: Decoder name.
            n_components: Number of discriminant components.
            shrinkage: Shrinkage parameter (0 to 1).
                If None, estimated using Ledoit-Wolf formula.
        """
        super().__init__(name=name, n_components=n_components, regularization=0)
        self.shrinkage = shrinkage

    def _compute_pooled_covariance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute pooled covariance with shrinkage."""
        # Standard pooled covariance
        cov = super()._compute_pooled_covariance(X, y)

        # Compute shrinkage parameter if not provided
        if self.shrinkage is None:
            shrinkage = self._ledoit_wolf_shrinkage(X, y, cov)
        else:
            shrinkage = self.shrinkage

        # Apply shrinkage: S_shrunk = (1 - shrinkage) * S + shrinkage * trace(S)/p * I
        trace = np.trace(cov)
        p = cov.shape[0]
        target = (trace / p) * np.eye(p)

        return (1 - shrinkage) * cov + shrinkage * target

    def _ledoit_wolf_shrinkage(
        self, X: np.ndarray, y: np.ndarray, cov: np.ndarray
    ) -> float:
        """Estimate optimal shrinkage using Ledoit-Wolf formula."""
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Simplified Ledoit-Wolf shrinkage estimation
        trace = np.trace(cov)
        trace_sq = np.trace(cov @ cov)

        # Estimate optimal shrinkage
        mu = trace / n_features
        delta = (trace_sq + trace**2) / ((n_samples + 1) * (trace_sq - trace**2 / n_features))

        shrinkage = max(0, min(1, delta))
        return shrinkage
