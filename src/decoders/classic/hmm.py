"""
Hidden Markov Model decoder for neural signals.

Implements Gaussian HMM for discrete state inference from continuous
neural features. Useful for detecting movement intentions, brain states,
or action sequences.

Reference:
    Kemere et al. (2008) "Detecting neural-state transitions using hidden
    Markov models for motor cortical prostheses" J Neurophysiol
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from src.decoders.base import BaseDecoder


class GaussianHMM(BaseDecoder):
    """
    Gaussian Hidden Markov Model for neural state decoding.

    Uses continuous Gaussian emissions with diagonal or full covariance.
    Trained using Expectation-Maximization (Baum-Welch algorithm).
    """

    def __init__(
        self,
        name: str = "GaussianHMM",
        n_states: int = 2,
        covariance_type: str = "diag",
        n_iter: int = 100,
        tol: float = 1e-4,
        init_method: str = "kmeans",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize Gaussian HMM decoder.

        Args:
            name: Decoder name.
            n_states: Number of hidden states.
            covariance_type: Type of covariance ('diag', 'full', 'spherical').
            n_iter: Maximum EM iterations.
            tol: Convergence tolerance.
            init_method: Initialization method ('kmeans', 'random').
            random_state: Random seed for reproducibility.
            verbose: Print training progress.
        """
        super().__init__(name=name)

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters (set during fit)
        self.startprob_: Optional[np.ndarray] = None  # Initial state probabilities
        self.transmat_: Optional[np.ndarray] = None  # Transition matrix
        self.means_: Optional[np.ndarray] = None  # Emission means
        self.covars_: Optional[np.ndarray] = None  # Emission covariances

        self._log_likelihood_history: List[float] = []

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GaussianHMM":
        """
        Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            X: Observations of shape (n_samples, n_features).
            y: Optional state labels for supervised initialization.

        Returns:
            self: Fitted model.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.n_features = n_features

        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize parameters
        self._initialize_parameters(X, y)

        # EM iterations
        prev_log_likelihood = -np.inf

        for iteration in range(self.n_iter):
            # E-step: compute responsibilities
            log_likelihood, posteriors, xi = self._e_step(X)
            self._log_likelihood_history.append(log_likelihood)

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}/{self.n_iter}: "
                    f"Log-likelihood = {log_likelihood:.4f}"
                )

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            prev_log_likelihood = log_likelihood

            # M-step: update parameters
            self._m_step(X, posteriors, xi)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely state sequence using Viterbi algorithm.

        Args:
            X: Observations of shape (n_samples, n_features).

        Returns:
            Most likely state sequence of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._viterbi(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute state probabilities using forward-backward algorithm.

        Args:
            X: Observations of shape (n_samples, n_features).

        Returns:
            State probabilities of shape (n_samples, n_states).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        log_alpha = self._forward(X)
        log_beta = self._backward(X)

        # Compute posteriors
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        return np.exp(log_gamma)

    def score(self, X: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.

        Args:
            X: Observations of shape (n_samples, n_features).

        Returns:
            Log-likelihood of the sequence.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        log_alpha = self._forward(X)
        return logsumexp(log_alpha[-1])

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from the model.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Tuple of (observations, states).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before sampling.")

        states = np.zeros(n_samples, dtype=int)
        observations = np.zeros((n_samples, self.n_features))

        # Sample initial state
        states[0] = np.random.choice(self.n_states, p=self.startprob_)

        # Sample first observation
        observations[0] = self._sample_emission(states[0])

        # Sample remaining
        for t in range(1, n_samples):
            states[t] = np.random.choice(self.n_states, p=self.transmat_[states[t - 1]])
            observations[t] = self._sample_emission(states[t])

        return observations, states

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on labeled data.

        Args:
            X: Neural features.
            y: True state labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        y_pred = self.predict(X)

        # Compute accuracy (handling label permutation)
        accuracy = self._compute_accuracy_with_permutation(y, y_pred)

        return {
            "accuracy": accuracy,
            "log_likelihood": self.score(X),
            "r2": accuracy,  # For compatibility
            "mse": 1.0 - accuracy,
        }

    def _initialize_parameters(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Initialize HMM parameters."""
        n_samples, n_features = X.shape

        # Initialize start probabilities (uniform)
        self.startprob_ = np.ones(self.n_states) / self.n_states

        # Initialize transition matrix (slightly favor staying in state)
        self.transmat_ = np.ones((self.n_states, self.n_states)) / self.n_states
        self.transmat_ += 0.1 * np.eye(self.n_states)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Initialize emission parameters
        if y is not None and len(np.unique(y)) == self.n_states:
            # Supervised initialization from labels
            self.means_ = np.array([X[y == s].mean(axis=0) for s in range(self.n_states)])
            self._init_covars_from_labels(X, y)
        elif self.init_method == "kmeans":
            # K-means initialization
            self._kmeans_init(X)
        else:
            # Random initialization
            self._random_init(X)

    def _kmeans_init(self, X: np.ndarray, max_iter: int = 10) -> None:
        """Initialize using k-means clustering."""
        n_samples = X.shape[0]

        # Random initial centroids
        idx = np.random.choice(n_samples, self.n_states, replace=False)
        self.means_ = X[idx].copy()

        # K-means iterations
        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.array(
                [np.sum((X - self.means_[k]) ** 2, axis=1) for k in range(self.n_states)]
            ).T
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_means = np.array(
                [
                    X[labels == k].mean(axis=0) if np.any(labels == k) else self.means_[k]
                    for k in range(self.n_states)
                ]
            )

            if np.allclose(self.means_, new_means):
                break
            self.means_ = new_means

        # Initialize covariances from clusters
        self._init_covars_from_labels(X, labels)

    def _random_init(self, X: np.ndarray) -> None:
        """Random initialization."""
        n_features = X.shape[1]

        # Random means within data range
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        self.means_ = np.random.uniform(X_min, X_max, (self.n_states, n_features))

        # Initialize covariances as data variance
        var = np.var(X, axis=0)
        if self.covariance_type == "diag":
            self.covars_ = np.tile(var, (self.n_states, 1))
        elif self.covariance_type == "spherical":
            self.covars_ = np.full(self.n_states, var.mean())
        else:  # full
            self.covars_ = np.tile(np.diag(var), (self.n_states, 1, 1))

    def _init_covars_from_labels(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Initialize covariances from cluster labels."""
        n_features = X.shape[1]

        if self.covariance_type == "diag":
            self.covars_ = np.zeros((self.n_states, n_features))
            for k in range(self.n_states):
                if np.sum(labels == k) > 1:
                    self.covars_[k] = np.var(X[labels == k], axis=0) + 1e-6
                else:
                    self.covars_[k] = np.var(X, axis=0) + 1e-6

        elif self.covariance_type == "spherical":
            self.covars_ = np.zeros(self.n_states)
            for k in range(self.n_states):
                if np.sum(labels == k) > 1:
                    self.covars_[k] = np.var(X[labels == k]) + 1e-6
                else:
                    self.covars_[k] = np.var(X) + 1e-6

        else:  # full
            self.covars_ = np.zeros((self.n_states, n_features, n_features))
            for k in range(self.n_states):
                if np.sum(labels == k) > 1:
                    self.covars_[k] = np.cov(X[labels == k].T) + 1e-6 * np.eye(n_features)
                else:
                    self.covars_[k] = np.cov(X.T) + 1e-6 * np.eye(n_features)

    def _compute_log_emission(self, X: np.ndarray) -> np.ndarray:
        """Compute log emission probabilities for all states."""
        n_samples = X.shape[0]
        log_prob = np.zeros((n_samples, self.n_states))

        for k in range(self.n_states):
            if self.covariance_type == "diag":
                log_prob[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=np.diag(self.covars_[k])
                )
            elif self.covariance_type == "spherical":
                log_prob[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covars_[k] * np.eye(self.n_features)
                )
            else:  # full
                log_prob[:, k] = stats.multivariate_normal.logpdf(
                    X, mean=self.means_[k], cov=self.covars_[k]
                )

        return log_prob

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm (log-space)."""
        n_samples = X.shape[0]
        log_emission = self._compute_log_emission(X)

        log_alpha = np.zeros((n_samples, self.n_states))

        # Initialize
        log_alpha[0] = np.log(self.startprob_ + 1e-10) + log_emission[0]

        # Forward pass
        log_transmat = np.log(self.transmat_ + 1e-10)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t - 1] + log_transmat[:, j]) + log_emission[t, j]
                )

        return log_alpha

    def _backward(self, X: np.ndarray) -> np.ndarray:
        """Backward algorithm (log-space)."""
        n_samples = X.shape[0]
        log_emission = self._compute_log_emission(X)

        log_beta = np.zeros((n_samples, self.n_states))

        # Initialize (log(1) = 0)
        log_beta[-1] = 0

        # Backward pass
        log_transmat = np.log(self.transmat_ + 1e-10)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    log_transmat[i, :] + log_emission[t + 1] + log_beta[t + 1]
                )

        return log_beta

    def _e_step(self, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """E-step: compute posteriors and expected transitions."""
        n_samples = X.shape[0]
        log_emission = self._compute_log_emission(X)

        log_alpha = self._forward(X)
        log_beta = self._backward(X)

        # Log-likelihood
        log_likelihood = logsumexp(log_alpha[-1])

        # Posteriors (gamma)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        posteriors = np.exp(log_gamma)

        # Expected transitions (xi)
        xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
        log_transmat = np.log(self.transmat_ + 1e-10)

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (
                        log_alpha[t, i]
                        + log_transmat[i, j]
                        + log_emission[t + 1, j]
                        + log_beta[t + 1, j]
                    )
            xi[t] -= logsumexp(xi[t])

        xi = np.exp(xi)

        return log_likelihood, posteriors, xi

    def _m_step(self, X: np.ndarray, posteriors: np.ndarray, xi: np.ndarray) -> None:
        """M-step: update parameters."""
        n_samples = X.shape[0]

        # Update start probabilities
        self.startprob_ = posteriors[0] + 1e-10
        self.startprob_ /= self.startprob_.sum()

        # Update transition matrix
        self.transmat_ = xi.sum(axis=0) + 1e-10
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Update emission means
        weights = posteriors.sum(axis=0)
        self.means_ = (posteriors.T @ X) / (weights[:, np.newaxis] + 1e-10)

        # Update emission covariances
        self._update_covars(X, posteriors, weights)

    def _update_covars(self, X: np.ndarray, posteriors: np.ndarray, weights: np.ndarray) -> None:
        """Update covariance parameters."""
        n_features = X.shape[1]

        if self.covariance_type == "diag":
            self.covars_ = np.zeros((self.n_states, n_features))
            for k in range(self.n_states):
                diff = X - self.means_[k]
                self.covars_[k] = (posteriors[:, k : k + 1] * diff**2).sum(axis=0)
                self.covars_[k] /= weights[k] + 1e-10
                self.covars_[k] += 1e-6  # Regularization

        elif self.covariance_type == "spherical":
            self.covars_ = np.zeros(self.n_states)
            for k in range(self.n_states):
                diff = X - self.means_[k]
                self.covars_[k] = (posteriors[:, k] * (diff**2).sum(axis=1)).sum()
                self.covars_[k] /= weights[k] * n_features + 1e-10
                self.covars_[k] += 1e-6

        else:  # full
            self.covars_ = np.zeros((self.n_states, n_features, n_features))
            for k in range(self.n_states):
                diff = X - self.means_[k]
                self.covars_[k] = (posteriors[:, k : k + 1] * diff).T @ diff
                self.covars_[k] /= weights[k] + 1e-10
                self.covars_[k] += 1e-6 * np.eye(n_features)

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence."""
        n_samples = X.shape[0]
        log_emission = self._compute_log_emission(X)

        # Viterbi variables
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        # Initialize
        viterbi[0] = np.log(self.startprob_ + 1e-10) + log_emission[0]

        # Forward pass
        log_transmat = np.log(self.transmat_ + 1e-10)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                scores = viterbi[t - 1] + log_transmat[:, j]
                backpointer[t, j] = np.argmax(scores)
                viterbi[t, j] = scores[backpointer[t, j]] + log_emission[t, j]

        # Backtrack
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def _sample_emission(self, state: int) -> np.ndarray:
        """Sample from emission distribution of given state."""
        if self.covariance_type == "diag":
            cov = np.diag(self.covars_[state])
        elif self.covariance_type == "spherical":
            cov = self.covars_[state] * np.eye(self.n_features)
        else:
            cov = self.covars_[state]

        return np.random.multivariate_normal(self.means_[state], cov)

    def _compute_accuracy_with_permutation(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy handling label permutation."""
        from itertools import permutations

        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)

        if len(unique_pred) > len(unique_true):
            # More predicted states than true states
            return np.mean(y_true == y_pred)

        # Try all permutations and find best mapping
        best_accuracy = 0.0

        for perm in permutations(range(self.n_states)):
            # Create mapping
            y_mapped = np.array([perm[s] for s in y_pred])
            accuracy = np.mean(y_true == y_mapped)
            best_accuracy = max(best_accuracy, accuracy)

        return best_accuracy

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update(
            {
                "n_states": self.n_states,
                "covariance_type": self.covariance_type,
                "n_iter": self.n_iter,
                "log_likelihood_history": self._log_likelihood_history[-10:],
            }
        )
        if self.is_fitted:
            params["final_log_likelihood"] = self._log_likelihood_history[-1]
        return params


class DiscreteHMM(BaseDecoder):
    """
    Discrete (Multinomial) Hidden Markov Model.

    For use with discretized/binned neural features.
    """

    def __init__(
        self,
        name: str = "DiscreteHMM",
        n_states: int = 2,
        n_symbols: int = 10,
        n_iter: int = 100,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize Discrete HMM.

        Args:
            name: Decoder name.
            n_states: Number of hidden states.
            n_symbols: Number of discrete observation symbols.
            n_iter: Maximum EM iterations.
            tol: Convergence tolerance.
            random_state: Random seed.
            verbose: Print training progress.
        """
        super().__init__(name=name)

        self.n_states = n_states
        self.n_symbols = n_symbols
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

        # Model parameters
        self.startprob_: Optional[np.ndarray] = None
        self.transmat_: Optional[np.ndarray] = None
        self.emissionprob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DiscreteHMM":
        """
        Fit discrete HMM.

        Args:
            X: Discrete observations of shape (n_samples,) with values in [0, n_symbols).
            y: Optional state labels.

        Returns:
            self: Fitted model.
        """
        if X.ndim > 1:
            X = X.ravel()

        X = X.astype(int)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize
        self.startprob_ = np.ones(self.n_states) / self.n_states
        self.transmat_ = np.ones((self.n_states, self.n_states)) / self.n_states
        self.emissionprob_ = np.random.dirichlet(np.ones(self.n_symbols), size=self.n_states)

        # EM iterations
        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            # E-step
            log_alpha = self._forward(X)
            log_beta = self._backward(X)
            ll = logsumexp(log_alpha[-1])

            if self.verbose:
                print(f"Iteration {iteration + 1}: LL = {ll:.4f}")

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # Posteriors
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # M-step
            self.startprob_ = gamma[0] + 1e-10
            self.startprob_ /= self.startprob_.sum()

            # Transition matrix
            n_samples = len(X)
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(n_samples - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[i, j] += np.exp(
                            log_alpha[t, i]
                            + np.log(self.transmat_[i, j] + 1e-10)
                            + np.log(self.emissionprob_[j, X[t + 1]] + 1e-10)
                            + log_beta[t + 1, j]
                            - ll
                        )

            self.transmat_ = xi + 1e-10
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

            # Emission probabilities
            for k in range(self.n_states):
                for s in range(self.n_symbols):
                    self.emissionprob_[k, s] = gamma[X == s, k].sum() + 1e-10
                self.emissionprob_[k] /= self.emissionprob_[k].sum()

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict states using Viterbi."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if X.ndim > 1:
            X = X.ravel()
        X = X.astype(int)

        return self._viterbi(X)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward algorithm."""
        n_samples = len(X)
        log_alpha = np.zeros((n_samples, self.n_states))

        log_alpha[0] = np.log(self.startprob_ + 1e-10) + np.log(self.emissionprob_[:, X[0]] + 1e-10)

        for t in range(1, n_samples):
            for j in range(self.n_states):
                log_alpha[t, j] = logsumexp(
                    log_alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-10)
                ) + np.log(self.emissionprob_[j, X[t]] + 1e-10)

        return log_alpha

    def _backward(self, X: np.ndarray) -> np.ndarray:
        """Backward algorithm."""
        n_samples = len(X)
        log_beta = np.zeros((n_samples, self.n_states))

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = logsumexp(
                    np.log(self.transmat_[i, :] + 1e-10)
                    + np.log(self.emissionprob_[:, X[t + 1]] + 1e-10)
                    + log_beta[t + 1]
                )

        return log_beta

    def _viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding."""
        n_samples = len(X)
        viterbi = np.zeros((n_samples, self.n_states))
        backpointer = np.zeros((n_samples, self.n_states), dtype=int)

        viterbi[0] = np.log(self.startprob_ + 1e-10) + np.log(self.emissionprob_[:, X[0]] + 1e-10)

        for t in range(1, n_samples):
            for j in range(self.n_states):
                scores = viterbi[t - 1] + np.log(self.transmat_[:, j] + 1e-10)
                backpointer[t, j] = np.argmax(scores)
                viterbi[t, j] = scores[backpointer[t, j]] + np.log(
                    self.emissionprob_[j, X[t]] + 1e-10
                )

        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]

        return states

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update(
            {
                "n_states": self.n_states,
                "n_symbols": self.n_symbols,
            }
        )
        return params
