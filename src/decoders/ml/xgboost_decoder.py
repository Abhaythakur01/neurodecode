"""
XGBoost decoder for neural signals.

Implements gradient boosting for neural decoding with high accuracy
and efficient handling of large feature spaces.

Reference:
    Glaser et al. (2020) "Machine Learning for Neural Decoding"
    eNeuro
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from src.decoders.base import BaseDecoder

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


class XGBoostDecoder(BaseDecoder):
    """
    XGBoost decoder for continuous kinematic decoding.

    Gradient boosting with regularization for high-dimensional
    neural feature decoding.
    """

    def __init__(
        self,
        name: str = "XGBoost",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        objective: str = "reg:squarederror",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False,
    ):
        """
        Initialize XGBoost decoder.

        Args:
            name: Decoder name.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns for each tree.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            min_child_weight: Minimum sum of instance weight in child.
            objective: Learning objective.
            n_jobs: Number of parallel threads.
            random_state: Random seed.
            early_stopping_rounds: Early stopping rounds (None to disable).
            verbose: Print training progress.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for XGBoostDecoder. " "Install with: pip install xgboost"
            )

        super().__init__(name=name)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self._models: List = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.best_iteration_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostDecoder":
        """
        Fit XGBoost decoder.

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

        # Train one model per output dimension
        self._models = []
        importances = np.zeros(self.n_features)

        for i in range(self.n_outputs):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
                objective=self.objective,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=1 if self.verbose else 0,
            )

            # Split for early stopping if enabled
            if self.early_stopping_rounds:
                val_split = int(0.9 * len(X))
                X_train, X_val = X[:val_split], X[val_split:]
                y_train, y_val = y[:val_split, i], y[val_split:, i]

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=self.verbose,
                )
            else:
                model.fit(X, y[:, i], verbose=self.verbose)

            self._models.append(model)
            importances += model.feature_importances_

        # Average feature importances
        self.feature_importances_ = importances / self.n_outputs

        # Store best iteration from first model
        if hasattr(self._models[0], "best_iteration"):
            self.best_iteration_ = self._models[0].best_iteration

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

        # Predict from each model
        predictions = np.zeros((X.shape[0], self.n_outputs))
        for i, model in enumerate(self._models):
            predictions[:, i] = model.predict(X)

        return predictions

    def get_feature_importance(
        self, feature_names: Optional[list] = None, importance_type: str = "weight"
    ) -> Dict[str, float]:
        """
        Get feature importance ranking.

        Args:
            feature_names: Optional list of feature names.
            importance_type: Type of importance ('weight', 'gain', 'cover').

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted first.")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features)]

        # Get importance from first model (they should be similar)
        if importance_type != "weight":
            booster = self._models[0].get_booster()
            importance = booster.get_score(importance_type=importance_type)
            # Convert to array
            scores = np.zeros(self.n_features)
            for k, v in importance.items():
                idx = int(k.replace("f", ""))
                scores[idx] = v
            return dict(zip(feature_names, scores))

        return dict(zip(feature_names, self.feature_importances_))

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
            }
        )
        if self.best_iteration_ is not None:
            params["best_iteration"] = self.best_iteration_
        return params


class XGBoostClassifier(BaseDecoder):
    """
    XGBoost classifier for discrete state decoding.

    Gradient boosting classification with built-in regularization.
    """

    def __init__(
        self,
        name: str = "XGBClassifier",
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        objective: str = "multi:softprob",
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = False,
    ):
        """
        Initialize XGBoost classifier.

        Args:
            name: Decoder name.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            subsample: Subsample ratio.
            colsample_bytree: Column subsample ratio.
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.
            min_child_weight: Minimum child weight.
            objective: Learning objective.
            n_jobs: Parallel threads.
            random_state: Random seed.
            early_stopping_rounds: Early stopping rounds.
            verbose: Print progress.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is required for XGBoostClassifier. " "Install with: pip install xgboost"
            )

        super().__init__(name=name)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self._model = None
        self.classes_: Optional[np.ndarray] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """
        Fit XGBoost classifier.

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
        n_classes = len(self.classes_)

        # Determine objective based on number of classes
        if n_classes == 2:
            objective = "binary:logistic"
        else:
            objective = self.objective

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            objective=objective,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbosity=1 if self.verbose else 0,
            use_label_encoder=False,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
        )

        # Split for early stopping if enabled
        if self.early_stopping_rounds:
            val_split = int(0.9 * len(X))
            X_train, X_val = X[:val_split], X[val_split:]
            y_train, y_val = y[:val_split], y[val_split:]

            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=self.verbose,
            )
        else:
            self._model.fit(X, y, verbose=self.verbose)

        self.feature_importances_ = self._model.feature_importances_

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

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        params = super().get_params()
        params.update(
            {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "n_classes": len(self.classes_) if self.classes_ is not None else None,
            }
        )
        return params
