"""
Kalman Filter decoder for neural signals.

Implements the standard Kalman Filter approach widely used in
Brain-Computer Interfaces for continuous decoding of movement.

Reference:
    Wu et al. (2006) "Bayesian population decoding of motor cortical activity
    using a Kalman filter" Neural Computation
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.decoders.base import OnlineDecoder


class KalmanFilterDecoder(OnlineDecoder):
    """
    Kalman Filter decoder for neural-to-kinematic mapping.

    Uses a linear state-space model:
        State equation:     x_t = A * x_{t-1} + w_t    (w_t ~ N(0, W))
        Observation eq:     y_t = H * x_t + q_t       (q_t ~ N(0, Q))

    Where:
        x_t: Kinematic state (position, velocity, etc.)
        y_t: Neural observations (firing rates)
        A: State transition matrix
        H: Observation matrix
        W: Process noise covariance
        Q: Observation noise covariance
    """

    def __init__(
        self,
        name: str = "KalmanFilter",
        learning_rate: float = 0.01,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-2,
    ):
        """
        Initialize Kalman Filter decoder.

        Args:
            name: Decoder name.
            learning_rate: Learning rate for online updates.
            process_noise: Initial process noise variance.
            observation_noise: Initial observation noise variance.
        """
        super().__init__(name=name, learning_rate=learning_rate)
        self.process_noise = process_noise
        self.observation_noise = observation_noise

        # Model parameters (set during fit)
        self.A: Optional[np.ndarray] = None  # State transition
        self.H: Optional[np.ndarray] = None  # Observation matrix
        self.W: Optional[np.ndarray] = None  # Process noise covariance
        self.Q: Optional[np.ndarray] = None  # Observation noise covariance

        # State estimation
        self._x: Optional[np.ndarray] = None  # Current state estimate
        self._P: Optional[np.ndarray] = None  # State covariance

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KalmanFilterDecoder":
        """
        Fit Kalman Filter parameters from training data.

        Uses least squares to estimate A and H matrices, then
        computes noise covariances from residuals.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Kinematics of shape (n_samples, n_outputs).

        Returns:
            self: Fitted decoder.
        """
        self._validate_input(X, y)

        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]

        # Estimate state transition matrix A using consecutive kinematics
        # x_t = A * x_{t-1}
        y_prev = y[:-1]
        y_curr = y[1:]

        # Least squares: A = (y_curr^T @ y_prev) @ (y_prev^T @ y_prev)^{-1}
        self.A = np.linalg.lstsq(y_prev, y_curr, rcond=None)[0].T

        # Estimate observation matrix H
        # X = H @ y (neural = H @ kinematics)
        self.H = np.linalg.lstsq(y, X, rcond=None)[0].T

        # Compute process noise covariance W from state equation residuals
        y_pred_state = y_prev @ self.A.T
        state_residuals = y_curr - y_pred_state
        self.W = np.cov(state_residuals.T) + self.process_noise * np.eye(self.n_outputs)

        # Ensure W is symmetric positive definite
        self.W = (self.W + self.W.T) / 2
        self.W = self._ensure_positive_definite(self.W)

        # Compute observation noise covariance Q from observation equation residuals
        X_pred = y @ self.H.T
        obs_residuals = X - X_pred
        self.Q = np.cov(obs_residuals.T) + self.observation_noise * np.eye(self.n_features)

        # Ensure Q is symmetric positive definite
        self.Q = (self.Q + self.Q.T) / 2
        self.Q = self._ensure_positive_definite(self.Q)

        # Initialize state
        self._x = np.zeros(self.n_outputs)
        self._P = np.eye(self.n_outputs)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Decode kinematics from neural features using Kalman Filter.

        Args:
            X: Neural features of shape (n_samples, n_features).

        Returns:
            Decoded kinematics of shape (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_outputs))

        # Reset state for batch prediction
        x = self._x.copy()
        P = self._P.copy()

        for t in range(n_samples):
            # Kalman filter step
            x, P = self._kalman_step(X[t], x, P)
            predictions[t] = x

        return predictions

    def predict_single(self, x_neural: np.ndarray) -> np.ndarray:
        """
        Predict single time step (for real-time use).

        Updates internal state and returns prediction.

        Args:
            x_neural: Neural features for single time step.

        Returns:
            Predicted kinematics.
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        self._x, self._P = self._kalman_step(x_neural, self._x, self._P)
        return self._x.copy()

    def _kalman_step(
        self,
        observation: np.ndarray,
        x_prior: np.ndarray,
        P_prior: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single Kalman filter update step.

        Args:
            observation: Neural observation vector.
            x_prior: Prior state estimate.
            P_prior: Prior state covariance.

        Returns:
            Tuple of (posterior_state, posterior_covariance).
        """
        # Predict step
        x_pred = self.A @ x_prior
        P_pred = self.A @ P_prior @ self.A.T + self.W

        # Update step
        # Innovation
        y_pred = self.H @ x_pred
        innovation = observation - y_pred

        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + self.Q

        # Kalman gain
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Posterior
        x_post = x_pred + K @ innovation
        P_post = (np.eye(self.n_outputs) - K @ self.H) @ P_pred

        return x_post, P_post

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Online update of Kalman Filter parameters.

        Uses exponential moving average to update model parameters.

        Args:
            X: New neural features of shape (n_samples, n_features).
            y: New kinematics of shape (n_samples, n_outputs).
        """
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before update.")

        self._validate_input(X, y)

        alpha = self.learning_rate

        # Update observation matrix H with new data
        H_new = np.linalg.lstsq(y, X, rcond=None)[0].T
        self.H = (1 - alpha) * self.H + alpha * H_new

        # Update state transition A if we have consecutive samples
        if len(y) > 1:
            y_prev = y[:-1]
            y_curr = y[1:]
            A_new = np.linalg.lstsq(y_prev, y_curr, rcond=None)[0].T
            self.A = (1 - alpha) * self.A + alpha * A_new

        # Update noise covariances
        X_pred = y @ self.H.T
        obs_residuals = X - X_pred
        Q_new = np.cov(obs_residuals.T) + self.observation_noise * np.eye(self.n_features)
        self.Q = (1 - alpha) * self.Q + alpha * Q_new

        self._update_count += 1

    def reset_state(self) -> None:
        """Reset internal state estimate to zero."""
        if self.is_fitted:
            self._x = np.zeros(self.n_outputs)
            self._P = np.eye(self.n_outputs)

    def get_kalman_gain(self) -> Optional[np.ndarray]:
        """Get current Kalman gain matrix."""
        if not self.is_fitted:
            return None

        # Compute steady-state Kalman gain
        P_pred = self.A @ self._P @ self.A.T + self.W
        S = self.H @ P_pred @ self.H.T + self.Q
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        return K

    def _ensure_positive_definite(self, matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive definite by adding small diagonal."""
        eigvals = np.linalg.eigvalsh(matrix)
        if np.min(eigvals) < eps:
            matrix = matrix + (eps - np.min(eigvals) + eps) * np.eye(matrix.shape[0])
        return matrix

    def get_params(self) -> Dict[str, Any]:
        """Get decoder parameters."""
        params = super().get_params()
        params.update(
            {
                "process_noise": self.process_noise,
                "observation_noise": self.observation_noise,
                "n_features": self.n_features,
                "n_outputs": self.n_outputs,
            }
        )
        return params


class SteadyStateKalmanFilter(KalmanFilterDecoder):
    """
    Steady-state Kalman Filter for faster inference.

    Pre-computes the steady-state Kalman gain, avoiding
    the covariance update at each step.
    """

    def __init__(self, name: str = "SteadyStateKalman", **kwargs):
        super().__init__(name=name, **kwargs)
        self._K_ss: Optional[np.ndarray] = None  # Steady-state Kalman gain

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SteadyStateKalmanFilter":
        """Fit and compute steady-state Kalman gain."""
        super().fit(X, y)

        # Solve discrete algebraic Riccati equation for steady-state P
        P = self._solve_dare()
        self._P = P

        # Compute steady-state Kalman gain
        S = self.H @ P @ self.H.T + self.Q
        self._K_ss = P @ self.H.T @ np.linalg.inv(S)

        return self

    def _solve_dare(self, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
        """
        Solve Discrete Algebraic Riccati Equation iteratively.

        Returns steady-state covariance P.
        """
        P = np.eye(self.n_outputs)

        for _ in range(max_iter):
            P_pred = self.A @ P @ self.A.T + self.W
            S = self.H @ P_pred @ self.H.T + self.Q
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            P_new = (np.eye(self.n_outputs) - K @ self.H) @ P_pred

            if np.max(np.abs(P_new - P)) < tol:
                break

            P = P_new

        return P

    def predict_single(self, x_neural: np.ndarray) -> np.ndarray:
        """Fast single-step prediction using steady-state gain."""
        if not self.is_fitted:
            raise RuntimeError("Decoder must be fitted before prediction.")

        # Predict step
        x_pred = self.A @ self._x

        # Update with steady-state Kalman gain
        y_pred = self.H @ x_pred
        innovation = x_neural - y_pred
        self._x = x_pred + self._K_ss @ innovation

        return self._x.copy()
