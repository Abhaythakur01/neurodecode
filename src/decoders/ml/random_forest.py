"""
Random Forest decoder for neural signals.

Implements ensemble tree-based methods for neural decoding with
built-in feature importance estimation.

Reference:
    Glaser et al. (2020) "Machine Learning for Neural Decoding"
    eNeuro
"""

from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.decoders.base import BaseDecoder


class RandomForestDecoder(BaseDecoder):
    """
    Random Forest decoder for continuous kinematic decoding.

    Ensemble of decision trees with bootstrap aggregation (bagging).
    Provides feature importance for neural feature selection.
    """

    def __init__(
        self,
        name: str = "RandomForest",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        Initialize Random Forest decoder.

        Args:
            name: Decoder name.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples to split internal node.
            min_samples_leaf: Minimum samples in leaf node.
            max_features: Features to consider for best split.
            bootstrap: Whether to use bootstrap samples.
            oob_score: Use out-of-bag samples for generalization estimate.
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            random_state: Random seed for reproducibility.
            verbose: Verbosity level.
        """
        super().__init__(name=name)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestDecoder":
        """
        Fit Random Forest decoder.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1] if y.ndim > 1 else 1

        # Create model
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Fit model
        self._model.fit(X, y)

        # Store feature importances
        self.feature_importances_ = self._model.feature_importances_

        # Store OOB score if available
        if self.oob_score and hasattr(self._model, "oob_score_"):
            self.oob_score_ = self._model.oob_score_

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

        y_pred = self._model.predict(X)

        if y_pred.ndim == 1 and self.n_outputs > 1:
            y_pred = y_pred.reshape(-1, self.n_outputs)
        elif y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance ranking.

        Args:
            feature_names: Optional list of feature names.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted first.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features)]

        return dict(zip(feature_names, self.feature_importances_))

    def get_top_features(self, n: int = 10) -> np.ndarray:
        """
        Get indices of top N most important features.

        Args:
            n: Number of top features to return.

        Returns:
            Indices of top features sorted by importance.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted first.")

        return np.argsort(self.feature_importances_)[::-1][:n]

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
            }
        )
        if self.is_fitted:
            if self.oob_score_ is not None:
                params["oob_score"] = self.oob_score_
            params["n_features"] = self.n_features
        return params


class RandomForestClassifierDecoder(BaseDecoder):
    """
    Random Forest classifier for discrete state decoding.

    Ensemble classification with built-in feature importance
    and probability estimates.
    """

    def __init__(
        self,
        name: str = "RFClassifier",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = True,
        class_weight: Optional[Union[str, Dict]] = "balanced",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        Initialize Random Forest classifier.

        Args:
            name: Decoder name.
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples for split.
            min_samples_leaf: Minimum samples in leaf.
            max_features: Features to consider for split.
            bootstrap: Use bootstrap samples.
            oob_score: Compute out-of-bag score.
            class_weight: Class weights.
            n_jobs: Parallel jobs.
            random_state: Random seed.
            verbose: Verbosity level.
        """
        super().__init__(name=name)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self._model = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifierDecoder":
        """
        Fit Random Forest classifier.

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

        # Create model
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Fit model
        self._model.fit(X, y)

        # Store feature importances and OOB score
        self.feature_importances_ = self._model.feature_importances_
        if self.oob_score and hasattr(self._model, "oob_score_"):
            self.oob_score_ = self._model.oob_score_

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

        if X.ndim == 1:
            X = X.reshape(1, -1)

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
            "r2": accuracy,
            "mse": 1.0 - accuracy,
        }

        # Per-class accuracy
        for c in self.classes_:
            mask = y == c
            if np.any(mask):
                metrics[f"accuracy_class_{c}"] = np.mean(y_pred[mask] == y[mask])

        if self.oob_score_ is not None:
            metrics["oob_score"] = self.oob_score_

        return metrics

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """Get feature importance ranking."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted first.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features)]

        return dict(zip(feature_names, self.feature_importances_))

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "n_classes": len(self.classes_) if self.classes_ is not None else None,
            }
        )
        if self.oob_score_ is not None:
            params["oob_score"] = self.oob_score_
        return params
