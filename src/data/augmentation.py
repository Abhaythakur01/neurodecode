"""
Data augmentation utilities for neural decoding.

Provides various augmentation strategies to improve decoder generalization
and robustness by artificially expanding training data.

Reference:
    Farshchian et al. (2019) "Adversarial Domain Adaptation for Stable
    Brain-Machine Interfaces"
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np


class NeuralDataAugmenter:
    """
    Data augmentation for neural signals.

    Applies various transformations to neural data while preserving
    the underlying structure for decoder training.
    """

    def __init__(
        self,
        augmentations: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize augmenter.

        Args:
            augmentations: List of augmentation names to apply.
                         If None, uses default set.
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        default_augmentations = [
            "jitter",
            "scaling",
            "dropout",
            "time_warp",
        ]
        self.augmentations = augmentations or default_augmentations

        self._augmentation_funcs = {
            "jitter": self.add_jitter,
            "scaling": self.apply_scaling,
            "dropout": self.apply_dropout,
            "time_warp": self.apply_time_warp,
            "channel_shuffle": self.shuffle_channels,
            "magnitude_warp": self.apply_magnitude_warp,
            "permutation": self.apply_permutation,
            "rotation": self.apply_rotation,
            "gaussian_noise": self.add_gaussian_noise,
            "poisson_noise": self.add_poisson_noise,
        }

    def augment(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_augmentations: int = 1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply augmentations to data.

        Args:
            X: Neural features of shape (n_samples, n_features).
            y: Optional kinematics of shape (n_samples, n_outputs).
            n_augmentations: Number of augmented copies to generate.

        Returns:
            Augmented (X, y) tuple. Original data is included.
        """
        X_aug_list = [X]
        y_aug_list = [y] if y is not None else []

        for _ in range(n_augmentations):
            X_new = X.copy()
            y_new = y.copy() if y is not None else None

            # Apply random subset of augmentations
            n_to_apply = self.rng.randint(1, len(self.augmentations) + 1)
            augmentations_to_apply = self.rng.choice(
                self.augmentations, size=n_to_apply, replace=False
            )

            for aug_name in augmentations_to_apply:
                if aug_name in self._augmentation_funcs:
                    X_new, y_new = self._augmentation_funcs[aug_name](X_new, y_new)

            X_aug_list.append(X_new)
            if y_new is not None:
                y_aug_list.append(y_new)

        X_augmented = np.concatenate(X_aug_list, axis=0)
        y_augmented = np.concatenate(y_aug_list, axis=0) if y_aug_list else None

        return X_augmented, y_augmented

    def add_jitter(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sigma: float = 0.03,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Add random jitter (Gaussian noise) to neural signals.

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            sigma: Standard deviation of noise relative to data std.

        Returns:
            Augmented (X, y) tuple.
        """
        noise = self.rng.randn(*X.shape) * sigma * np.std(X, axis=0, keepdims=True)
        X_aug = X + noise
        return X_aug.astype(np.float32), y

    def apply_scaling(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sigma: float = 0.1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random scaling to each channel.

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            sigma: Standard deviation of scaling factors.

        Returns:
            Augmented (X, y) tuple.
        """
        scaling_factors = self.rng.randn(X.shape[1]) * sigma + 1.0
        X_aug = X * scaling_factors
        return X_aug.astype(np.float32), y

    def apply_dropout(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        p: float = 0.1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Randomly drop neural channels (simulate channel loss).

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            p: Probability of dropping each channel.

        Returns:
            Augmented (X, y) tuple.
        """
        mask = self.rng.rand(X.shape[1]) > p
        X_aug = X.copy()
        X_aug[:, ~mask] = 0
        return X_aug.astype(np.float32), y

    def apply_time_warp(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sigma: float = 0.2,
        n_knots: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply time warping to stretch/compress time axis.

        Args:
            X: Neural features.
            y: Kinematics.
            sigma: Standard deviation of warp.
            n_knots: Number of spline knots.

        Returns:
            Augmented (X, y) tuple.
        """
        from scipy.interpolate import CubicSpline

        n_samples = X.shape[0]

        # Create random time warp
        knot_positions = np.linspace(0, n_samples - 1, n_knots + 2)
        knot_values = knot_positions + self.rng.randn(n_knots + 2) * sigma * n_samples
        knot_values = np.sort(knot_values)
        knot_values = np.clip(knot_values, 0, n_samples - 1)

        # Ensure monotonicity
        for i in range(1, len(knot_values)):
            if knot_values[i] <= knot_values[i - 1]:
                knot_values[i] = knot_values[i - 1] + 0.1

        # Create spline
        cs = CubicSpline(knot_positions, knot_values)
        warped_indices = cs(np.arange(n_samples))
        warped_indices = np.clip(warped_indices, 0, n_samples - 1).astype(int)

        X_aug = X[warped_indices]
        y_aug = y[warped_indices] if y is not None else None

        return X_aug.astype(np.float32), y_aug

    def shuffle_channels(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fraction: float = 0.1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Randomly shuffle a fraction of channels.

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            fraction: Fraction of channels to shuffle.

        Returns:
            Augmented (X, y) tuple.
        """
        n_channels = X.shape[1]
        n_shuffle = int(n_channels * fraction)

        if n_shuffle < 2:
            return X, y

        shuffle_indices = self.rng.choice(n_channels, size=n_shuffle, replace=False)
        shuffled = shuffle_indices.copy()
        self.rng.shuffle(shuffled)

        X_aug = X.copy()
        X_aug[:, shuffle_indices] = X[:, shuffled]

        return X_aug.astype(np.float32), y

    def apply_magnitude_warp(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sigma: float = 0.2,
        n_knots: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply smooth magnitude warping over time.

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            sigma: Standard deviation of magnitude changes.
            n_knots: Number of spline knots.

        Returns:
            Augmented (X, y) tuple.
        """
        from scipy.interpolate import CubicSpline

        n_samples = X.shape[0]

        # Create smooth magnitude curve
        knot_positions = np.linspace(0, n_samples - 1, n_knots + 2)
        knot_values = self.rng.randn(n_knots + 2) * sigma + 1.0

        cs = CubicSpline(knot_positions, knot_values)
        magnitude = cs(np.arange(n_samples)).reshape(-1, 1)

        X_aug = X * magnitude
        return X_aug.astype(np.float32), y

    def apply_permutation(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_segments: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Permute temporal segments.

        Args:
            X: Neural features.
            y: Kinematics.
            n_segments: Number of segments to create and permute.

        Returns:
            Augmented (X, y) tuple.
        """
        n_samples = X.shape[0]
        segment_size = n_samples // n_segments

        segments_X = [X[i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]
        remainder_X = X[n_segments * segment_size :]

        perm = self.rng.permutation(n_segments)
        X_aug = np.concatenate([segments_X[i] for i in perm] + [remainder_X], axis=0)

        if y is not None:
            segments_y = [y[i * segment_size : (i + 1) * segment_size] for i in range(n_segments)]
            remainder_y = y[n_segments * segment_size :]
            y_aug = np.concatenate([segments_y[i] for i in perm] + [remainder_y], axis=0)
        else:
            y_aug = None

        return X_aug.astype(np.float32), y_aug

    def apply_rotation(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        angle_std: float = 0.1,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random rotation in PCA space (for high-dimensional data).

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            angle_std: Standard deviation of rotation angles in radians.

        Returns:
            Augmented (X, y) tuple.
        """
        n_features = X.shape[1]

        if n_features < 2:
            return X, y

        # Create random rotation matrix
        # Use QR decomposition of random matrix
        random_matrix = self.rng.randn(n_features, n_features)
        Q, R = np.linalg.qr(random_matrix)

        # Make it a small rotation by interpolating with identity
        rotation_strength = self.rng.randn() * angle_std
        identity = np.eye(n_features)
        rotation = identity + rotation_strength * (Q - identity)

        X_aug = X @ rotation
        return X_aug.astype(np.float32), y

    def add_gaussian_noise(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        snr_db: float = 20.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Add Gaussian noise at specified SNR.

        Args:
            X: Neural features.
            y: Kinematics (unchanged).
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Augmented (X, y) tuple.
        """
        signal_power = np.mean(X**2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = self.rng.randn(*X.shape) * np.sqrt(noise_power)

        X_aug = X + noise
        return X_aug.astype(np.float32), y

    def add_poisson_noise(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Add Poisson-like noise (appropriate for spike counts).

        Args:
            X: Neural features (assumed to be spike counts or rates).
            y: Kinematics (unchanged).
            scale: Scaling factor for noise variance.

        Returns:
            Augmented (X, y) tuple.
        """
        # Ensure non-negative for Poisson-like noise
        X_pos = np.maximum(X, 0)

        # Add Poisson-like noise (variance proportional to mean)
        noise = self.rng.randn(*X.shape) * np.sqrt(X_pos * scale)
        X_aug = X + noise

        return X_aug.astype(np.float32), y


def augment_trials(
    trials_X: List[np.ndarray],
    trials_y: Optional[List[np.ndarray]] = None,
    n_augmentations: int = 2,
    augmentations: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Augment trial-based neural data.

    Args:
        trials_X: List of neural feature arrays, one per trial.
        trials_y: List of kinematics arrays, one per trial.
        n_augmentations: Number of augmented copies per trial.
        augmentations: List of augmentation names to use.
        random_state: Random seed.

    Returns:
        Augmented lists of (trials_X, trials_y).
    """
    augmenter = NeuralDataAugmenter(
        augmentations=augmentations,
        random_state=random_state,
    )

    aug_X = list(trials_X)
    aug_y = list(trials_y) if trials_y else []

    for i, X in enumerate(trials_X):
        y = trials_y[i] if trials_y else None

        for _ in range(n_augmentations):
            X_new, y_new = augmenter.augment(X, y, n_augmentations=0)

            # Apply single random augmentation
            aug_name = augmenter.rng.choice(augmenter.augmentations)
            X_new, y_new = augmenter._augmentation_funcs[aug_name](X_new, y_new)

            aug_X.append(X_new)
            if y_new is not None:
                aug_y.append(y_new)

    return aug_X, aug_y if aug_y else None


def mixup(
    X1: np.ndarray,
    y1: np.ndarray,
    X2: np.ndarray,
    y2: np.ndarray,
    alpha: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply mixup augmentation.

    Linearly interpolates between two samples.

    Args:
        X1, y1: First sample (features, labels).
        X2, y2: Second sample (features, labels).
        alpha: Beta distribution parameter for mixing coefficient.
        random_state: Random seed.

    Returns:
        Mixed (X, y) sample.
    """
    rng = np.random.RandomState(random_state)

    # Sample mixing coefficient from Beta distribution
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5

    X_mixed = lam * X1 + (1 - lam) * X2
    y_mixed = lam * y1 + (1 - lam) * y2

    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)


def cutmix_temporal(
    X1: np.ndarray,
    y1: np.ndarray,
    X2: np.ndarray,
    y2: np.ndarray,
    alpha: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CutMix augmentation in temporal domain.

    Replaces a temporal segment with data from another sample.

    Args:
        X1, y1: First sample (features, labels).
        X2, y2: Second sample (features, labels).
        alpha: Beta distribution parameter for cut ratio.
        random_state: Random seed.

    Returns:
        Mixed (X, y) sample.
    """
    rng = np.random.RandomState(random_state)

    n_samples = min(len(X1), len(X2))
    X1, X2 = X1[:n_samples], X2[:n_samples]
    y1, y2 = y1[:n_samples], y2[:n_samples]

    # Sample cut ratio
    lam = rng.beta(alpha, alpha) if alpha > 0 else 0.5
    cut_len = int(n_samples * (1 - lam))

    # Random cut position
    cut_start = rng.randint(0, n_samples - cut_len + 1)
    cut_end = cut_start + cut_len

    X_mixed = X1.copy()
    X_mixed[cut_start:cut_end] = X2[cut_start:cut_end]

    y_mixed = y1.copy()
    y_mixed[cut_start:cut_end] = y2[cut_start:cut_end]

    return X_mixed.astype(np.float32), y_mixed.astype(np.float32)
