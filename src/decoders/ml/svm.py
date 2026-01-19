"""
Support Vector Machine decoder for neural signals.

Implements SVM for both regression (kinematic decoding) and classification
(discrete state decoding) from neural features.

Reference:
    Lal et al. (2005) "Support Vector Channel Selection in BCI"
    IEEE Trans Biomed Eng
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.decoders.base import BaseDecoder


class SVMDecoder(BaseDecoder):
    """
    Support Vector Machine decoder for continuous kinematic decoding.

    Uses SVR (Support Vector Regression) with optional multi-output support.
    """

    def __init__(
        self,
        name: str = "SVM",
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        normalize: bool = True,
        cache_size: int = 200,
        max_iter: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize SVM decoder.

        Args:
            name: Decoder name.
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
            C: Regularization parameter.
            epsilon: Epsilon in epsilon-SVR model.
            gamma: Kernel coefficient ('scale', 'auto', or float).
            degree: Degree for polynomial kernel.
            normalize: Whether to normalize features.
            cache_size: Kernel cache size in MB.
            max_iter: Maximum iterations (-1 for no limit).
            verbose: Print training progress.
        """
        super().__init__(name=name)

        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.normalize = normalize
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.verbose = verbose

        self._model = None
        self._scaler = StandardScaler() if normalize else None
        self._y_scaler = StandardScaler() if normalize else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMDecoder":
        """
        Fit SVM decoder.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs) or (n_samples,).

        Returns:
            self: Fitted decoder.
        """
        # Handle 1D y array
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Normalize features
        if self.normalize:
            X = self._scaler.fit_transform(X)
            y = self._y_scaler.fit_transform(y)

        # Create SVR model
        base_svr = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            degree=self.degree,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )

        # Multi-output wrapper
        if self.n_outputs > 1:
            self._model = MultiOutputRegressor(base_svr, n_jobs=-1)
        else:
            self._model = base_svr

        # Fit model
        if self.n_outputs == 1:
            self._model.fit(X, y.ravel())
        else:
            self._model.fit(X, y)

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
        if self.normalize:
            X = self._scaler.transform(X)

        # Predict
        y_pred = self._model.predict(X)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Denormalize
        if self.normalize:
            y_pred = self._y_scaler.inverse_transform(y_pred)

        return y_pred

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update({
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "normalize": self.normalize,
        })
        if self.is_fitted and hasattr(self._model, 'n_support_'):
            params["n_support_vectors"] = int(np.sum(self._model.n_support_))
        return params


class SVMClassifier(BaseDecoder):
    """
    Support Vector Machine classifier for discrete state decoding.

    Uses SVC (Support Vector Classification) for neural state classification.
    """

    def __init__(
        self,
        name: str = "SVMClassifier",
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        probability: bool = True,
        normalize: bool = True,
        class_weight: Optional[Union[str, Dict]] = "balanced",
        cache_size: int = 200,
        max_iter: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize SVM classifier.

        Args:
            name: Decoder name.
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid').
            C: Regularization parameter.
            gamma: Kernel coefficient.
            degree: Degree for polynomial kernel.
            probability: Enable probability estimates.
            normalize: Whether to normalize features.
            class_weight: Class weights ('balanced' or dict).
            cache_size: Kernel cache size in MB.
            max_iter: Maximum iterations.
            random_state: Random seed.
            verbose: Print training progress.
        """
        super().__init__(name=name)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.normalize = normalize
        self.class_weight = class_weight
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self._scaler = StandardScaler() if normalize else None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """
        Fit SVM classifier.

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

        # Normalize features
        if self.normalize:
            X = self._scaler.fit_transform(X)

        # Create SVC model
        self._model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            class_weight=self.class_weight,
            cache_size=self.cache_size,
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Fit model
        self._model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize:
            X = self._scaler.transform(X)

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction.")

        if not self.probability:
            raise RuntimeError("Probability estimation not enabled. "
                             "Set probability=True during initialization.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.normalize:
            X = self._scaler.transform(X)

        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier.

        Args:
            X: Neural features.
            y: True labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before evaluation.")

        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        metrics = {
            "accuracy": accuracy,
            "r2": accuracy,  # For compatibility
            "mse": 1.0 - accuracy,
        }

        # Per-class accuracy
        for c in self.classes_:
            mask = y == c
            if np.any(mask):
                metrics[f"accuracy_class_{c}"] = np.mean(y_pred[mask] == y[mask])

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        params = super().get_params()
        params.update({
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "normalize": self.normalize,
            "n_classes": len(self.classes_) if self.classes_ is not None else None,
        })
        if self.is_fitted:
            params["n_support_vectors"] = int(np.sum(self._model.n_support_))
        return params
