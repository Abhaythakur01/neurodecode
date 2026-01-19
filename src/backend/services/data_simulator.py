"""
Synthetic neural data simulator for BCI testing.

Generates realistic neural firing patterns that correlate with
movement trajectories for testing the decoding pipeline.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np

from src.backend.config import settings

logger = logging.getLogger(__name__)


class MovementPattern(str, Enum):
    """Available movement patterns for simulation."""

    CIRCULAR = "circular"
    REACHING = "reaching"
    RANDOM = "random"
    FIGURE_EIGHT = "figure_eight"


class NeuralDataSimulator:
    """
    Simulates neural firing rates correlated with movement.

    The simulator generates synthetic neural data that follows
    cosine tuning curves, a common model for motor cortex neurons.
    """

    def __init__(
        self,
        n_neurons: int = 50,
        noise_level: float = 0.1,
        pattern: MovementPattern = MovementPattern.CIRCULAR,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize the simulator.

        Args:
            n_neurons: Number of neurons to simulate
            noise_level: Standard deviation of noise (0 = no noise)
            pattern: Movement pattern to simulate
            speed: Speed multiplier for movement
            seed: Random seed for reproducibility
        """
        self.n_neurons = n_neurons
        self.noise_level = noise_level
        self.pattern = pattern
        self.speed = speed

        # Random state
        self._rng = np.random.default_rng(seed)

        # Generate random preferred directions for each neuron
        self._preferred_directions = self._rng.uniform(
            0, 2 * np.pi, size=n_neurons
        )

        # Generate random baseline firing rates (Hz)
        self._baseline_rates = self._rng.uniform(5, 20, size=n_neurons)

        # Generate random modulation depths
        self._modulation_depths = self._rng.uniform(10, 50, size=n_neurons)

        # State
        self._time = 0.0
        self._position = np.array([0.0, 0.0])
        self._target_position: Optional[np.ndarray] = None
        self._is_running = False

        logger.info(
            f"Simulator initialized: {n_neurons} neurons, "
            f"pattern={pattern.value}, noise={noise_level}"
        )

    def reset(self):
        """Reset simulator state."""
        self._time = 0.0
        self._position = np.array([0.0, 0.0])
        self._target_position = None

    def _get_velocity(self, t: float) -> np.ndarray:
        """Get velocity at time t based on movement pattern."""
        if self.pattern == MovementPattern.CIRCULAR:
            # Circular movement
            omega = 0.5 * self.speed  # rad/s
            vx = -np.sin(omega * t) * omega
            vy = np.cos(omega * t) * omega
            return np.array([vx, vy])

        elif self.pattern == MovementPattern.FIGURE_EIGHT:
            # Figure-eight pattern
            omega = 0.3 * self.speed
            vx = np.cos(omega * t) * omega
            vy = np.cos(2 * omega * t) * 2 * omega
            return np.array([vx, vy])

        elif self.pattern == MovementPattern.REACHING:
            # Reaching movements to random targets
            if self._target_position is None or np.linalg.norm(
                self._position - self._target_position
            ) < 0.05:
                # Generate new target
                self._target_position = self._rng.uniform(-1, 1, size=2)

            # Move toward target with bell-shaped velocity profile
            direction = self._target_position - self._position
            distance = np.linalg.norm(direction)
            if distance > 0.01:
                direction = direction / distance
                # Bell-shaped speed profile
                speed = self.speed * min(distance * 2, 1.0) * (1 - np.exp(-distance * 5))
                return direction * speed
            return np.array([0.0, 0.0])

        elif self.pattern == MovementPattern.RANDOM:
            # Random walk
            return self._rng.normal(0, 0.3 * self.speed, size=2)

        return np.array([0.0, 0.0])

    def _compute_firing_rates(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute firing rates based on velocity using cosine tuning.

        Each neuron has a preferred direction, and fires more when
        movement is in that direction.

        Args:
            velocity: 2D velocity vector [vx, vy]

        Returns:
            Array of firing rates for each neuron
        """
        # Compute movement direction
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            # No movement: return baseline rates
            rates = self._baseline_rates.copy()
        else:
            direction = np.arctan2(velocity[1], velocity[0])

            # Cosine tuning: rate = baseline + depth * cos(direction - preferred) * speed
            cos_tuning = np.cos(direction - self._preferred_directions)
            rates = self._baseline_rates + self._modulation_depths * cos_tuning * speed

        # Ensure non-negative before adding noise
        rates = np.maximum(rates, 0.1)  # Minimum baseline rate

        # Add noise (use absolute rates for scale)
        if self.noise_level > 0:
            noise = self._rng.normal(0, self.noise_level * np.abs(rates) + 0.01)
            rates = rates + noise

        # Ensure non-negative after noise
        rates = np.maximum(rates, 0)

        # Normalize to [0, 1] range for easier processing
        rates = rates / (np.max(self._baseline_rates + self._modulation_depths) + 1e-6)

        return rates

    def generate_sample(self, dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single time sample of neural data.

        Args:
            dt: Time step in seconds (default 20ms)

        Returns:
            Tuple of (firing_rates, position)
        """
        # Get velocity for current time
        velocity = self._get_velocity(self._time)

        # Update position
        self._position = self._position + velocity * dt

        # Constrain position to [-1, 1] range
        self._position = np.clip(self._position, -1, 1)

        # Compute firing rates
        firing_rates = self._compute_firing_rates(velocity)

        # Advance time
        self._time += dt

        return firing_rates, self._position.copy()

    def generate_batch(
        self, n_samples: int, dt: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a batch of neural data samples.

        Args:
            n_samples: Number of samples to generate
            dt: Time step in seconds

        Returns:
            Tuple of (firing_rates, positions) with shapes
            (n_samples, n_neurons) and (n_samples, 2)
        """
        firing_rates = np.zeros((n_samples, self.n_neurons))
        positions = np.zeros((n_samples, 2))

        for i in range(n_samples):
            firing_rates[i], positions[i] = self.generate_sample(dt)

        return firing_rates, positions

    def generate_calibration_data(
        self, n_samples: int = 500, dt: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate calibration data for training decoders.

        Uses varied movement patterns to cover the workspace.

        Args:
            n_samples: Number of calibration samples
            dt: Time step in seconds

        Returns:
            Tuple of (neural_features, targets)
        """
        # Save state
        original_pattern = self.pattern

        # Generate data for each pattern
        samples_per_pattern = n_samples // 3
        all_features = []
        all_targets = []

        for pattern in [
            MovementPattern.CIRCULAR,
            MovementPattern.REACHING,
            MovementPattern.FIGURE_EIGHT,
        ]:
            self.pattern = pattern
            self.reset()
            features, targets = self.generate_batch(samples_per_pattern, dt)
            all_features.append(features)
            all_targets.append(targets)

        # Restore state
        self.pattern = original_pattern
        self.reset()

        # Combine
        features = np.vstack(all_features)
        targets = np.vstack(all_targets)

        # Shuffle
        indices = self._rng.permutation(len(features))
        return features[indices], targets[indices]


class SimulationRunner:
    """
    Manages running simulation in real-time.

    Handles start/stop, timing, and callback dispatch.
    """

    def __init__(self):
        """Initialize simulation runner."""
        self._simulator: Optional[NeuralDataSimulator] = None
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._callback: Optional[Callable[[np.ndarray, np.ndarray, float], None]] = None

        # Stats
        self._sample_count = 0
        self._start_time: Optional[float] = None

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running

    @property
    def sample_count(self) -> int:
        """Get total samples generated."""
        return self._sample_count

    def configure(
        self,
        n_neurons: int = 50,
        noise_level: float = 0.1,
        pattern: str = "circular",
        speed: float = 1.0,
    ):
        """Configure the simulator."""
        try:
            movement_pattern = MovementPattern(pattern)
        except ValueError:
            movement_pattern = MovementPattern.CIRCULAR

        self._simulator = NeuralDataSimulator(
            n_neurons=n_neurons,
            noise_level=noise_level,
            pattern=movement_pattern,
            speed=speed,
        )

    def start(self, callback: Callable[[np.ndarray, np.ndarray, float], None]):
        """
        Start real-time simulation.

        Args:
            callback: Function called with (firing_rates, position, timestamp)
                     for each sample
        """
        if self._is_running:
            logger.warning("Simulation already running")
            return

        if self._simulator is None:
            self.configure()

        self._callback = callback
        self._is_running = True
        self._sample_count = 0
        self._start_time = time.time()
        self._simulator.reset()

        logger.info("Simulation started")

    async def run_async(self):
        """Run simulation loop asynchronously."""
        if not self._is_running or self._simulator is None:
            return

        interval = 1.0 / settings.simulation_rate_hz
        next_time = time.perf_counter()

        while self._is_running:
            # Generate sample
            firing_rates, position = self._simulator.generate_sample(interval)
            timestamp = time.time() * 1000  # Unix timestamp in ms

            self._sample_count += 1

            # Call callback
            if self._callback:
                try:
                    self._callback(firing_rates, position, timestamp)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Sleep until next sample time
            next_time += interval
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # We're behind, skip to catch up
                next_time = time.perf_counter()

    def stop(self) -> dict:
        """
        Stop simulation.

        Returns:
            Statistics dictionary
        """
        self._is_running = False

        stats = {
            "total_samples": self._sample_count,
            "duration_seconds": 0.0,
            "samples_per_second": 0.0,
        }

        if self._start_time:
            duration = time.time() - self._start_time
            stats["duration_seconds"] = duration
            if duration > 0:
                stats["samples_per_second"] = self._sample_count / duration

        logger.info(f"Simulation stopped: {stats}")
        return stats

    def generate_calibration_data(
        self, n_samples: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate calibration data for decoder training."""
        if self._simulator is None:
            self.configure()

        return self._simulator.generate_calibration_data(n_samples)


# Global singleton
simulation_runner = SimulationRunner()
