"""
Neural Latents Benchmark (NLB) dataset loader.

Provides utilities for loading and processing NLB benchmark datasets
for neural decoding evaluation.

Reference:
    Pei et al. (2021) "Neural Latents Benchmark '21: Evaluating latent
    variable models of neural population activity"

Datasets available:
    - MC_Maze: Monkey reaching with maze obstacles
    - MC_RTT: Monkey random target task
    - Area2_Bump: Somatosensory cortex bump task
    - DMFC_RSG: Dorsomedial frontal cortex ready-set-go task
    - MC_Maze_Large: Extended MC_Maze dataset
    - MC_Maze_Medium: Medium-sized MC_Maze dataset
    - MC_Maze_Small: Small MC_Maze dataset
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    h5py = None

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class NLBDataLoader:
    """
    Loader for Neural Latents Benchmark datasets.

    Supports loading NLB datasets in HDF5 format and converting them
    to formats suitable for neural decoder training.
    """

    AVAILABLE_DATASETS = [
        "mc_maze",
        "mc_maze_large",
        "mc_maze_medium",
        "mc_maze_small",
        "mc_rtt",
        "area2_bump",
        "dmfc_rsg",
    ]

    NLB_BASE_URL = "https://dandiarchive.org/dandiset/"

    DATASET_INFO = {
        "mc_maze": {
            "description": "Monkey reaching with maze obstacles",
            "brain_area": "M1/PMd",
            "task": "Center-out reaching with barriers",
            "dandiset_id": "000128",
        },
        "mc_maze_large": {
            "description": "Extended MC_Maze dataset",
            "brain_area": "M1/PMd",
            "task": "Center-out reaching with barriers",
            "dandiset_id": "000128",
        },
        "mc_maze_medium": {
            "description": "Medium MC_Maze dataset",
            "brain_area": "M1/PMd",
            "task": "Center-out reaching with barriers",
            "dandiset_id": "000128",
        },
        "mc_maze_small": {
            "description": "Small MC_Maze dataset",
            "brain_area": "M1/PMd",
            "task": "Center-out reaching with barriers",
            "dandiset_id": "000128",
        },
        "mc_rtt": {
            "description": "Monkey random target task",
            "brain_area": "M1",
            "task": "Random target reaching",
            "dandiset_id": "000129",
        },
        "area2_bump": {
            "description": "Somatosensory cortex bump task",
            "brain_area": "Area 2",
            "task": "Active/passive arm movement",
            "dandiset_id": "000127",
        },
        "dmfc_rsg": {
            "description": "Ready-set-go timing task",
            "brain_area": "DMFC",
            "task": "Interval timing reproduction",
            "dandiset_id": "000130",
        },
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_name: str = "mc_maze",
        bin_size_ms: int = 5,
        verbose: bool = False,
    ):
        """
        Initialize NLB data loader.

        Args:
            data_dir: Directory containing NLB data files.
            dataset_name: Name of the NLB dataset to load.
            bin_size_ms: Bin size for spike counts in milliseconds.
            verbose: Print loading progress.
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name.lower()
        self.bin_size_ms = bin_size_ms
        self.verbose = verbose

        if self.dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. " f"Available: {self.AVAILABLE_DATASETS}"
            )

        self._data: Optional[Dict[str, Any]] = None
        self._train_data: Optional[Dict[str, np.ndarray]] = None
        self._test_data: Optional[Dict[str, np.ndarray]] = None

    def load(self) -> "NLBDataLoader":
        """
        Load dataset from disk.

        Returns:
            self: Loader with data loaded.
        """
        if not H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for loading NLB datasets. " "Install with: pip install h5py"
            )

        # Try different file patterns
        file_patterns = [
            f"{self.dataset_name}_train.h5",
            f"{self.dataset_name}.h5",
            f"{self.dataset_name}_data.h5",
        ]

        data_file = None
        for pattern in file_patterns:
            potential_file = self.data_dir / pattern
            if potential_file.exists():
                data_file = potential_file
                break

        if data_file is None:
            raise FileNotFoundError(
                f"No data file found for {self.dataset_name} in {self.data_dir}. "
                f"Expected one of: {file_patterns}"
            )

        if self.verbose:
            print(f"Loading {self.dataset_name} from {data_file}")

        self._data = self._load_h5_file(data_file)

        # Process into train/test splits if available
        self._process_data()

        return self

    def _load_h5_file(self, filepath: Path) -> Dict[str, Any]:
        """Load HDF5 file into dictionary."""
        data = {}

        with h5py.File(filepath, "r") as f:
            data = self._recursively_load_h5(f)

        return data

    def _recursively_load_h5(self, h5_obj: Union[h5py.File, h5py.Group]) -> Dict[str, Any]:
        """Recursively load HDF5 groups and datasets."""
        result = {}

        for key in h5_obj.keys():
            item = h5_obj[key]
            if isinstance(item, h5py.Dataset):
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                result[key] = self._recursively_load_h5(item)

        return result

    def _process_data(self) -> None:
        """Process loaded data into train/test arrays."""
        if self._data is None:
            return

        # NLB format typically has spikes and behavior data
        self._train_data = {}
        self._test_data = {}

        # Extract spike data
        if "train_spikes_heldin" in self._data:
            # Standard NLB format
            self._train_data["spikes"] = self._data["train_spikes_heldin"]
            if "train_spikes_heldout" in self._data:
                self._train_data["spikes_heldout"] = self._data["train_spikes_heldout"]

        elif "spikes" in self._data:
            # Simple format - split 80/20
            spikes = self._data["spikes"]
            split_idx = int(0.8 * len(spikes))
            self._train_data["spikes"] = spikes[:split_idx]
            self._test_data["spikes"] = spikes[split_idx:]

        # Extract behavior/kinematics data
        behavior_keys = ["hand_pos", "hand_vel", "cursor_pos", "cursor_vel", "behavior"]
        for key in behavior_keys:
            if key in self._data:
                behavior = self._data[key]
                if "train_spikes_heldin" in self._data:
                    self._train_data["behavior"] = behavior
                else:
                    split_idx = int(0.8 * len(behavior))
                    self._train_data["behavior"] = behavior[:split_idx]
                    self._test_data["behavior"] = behavior[split_idx:]
                break

        # Handle test data if separate
        if "test_spikes_heldin" in self._data:
            self._test_data["spikes"] = self._data["test_spikes_heldin"]
            if "test_spikes_heldout" in self._data:
                self._test_data["spikes_heldout"] = self._data["test_spikes_heldout"]

        if self.verbose:
            self._print_data_summary()

    def _print_data_summary(self) -> None:
        """Print summary of loaded data."""
        print(f"\n=== {self.dataset_name.upper()} Dataset Summary ===")

        if self._train_data:
            print("\nTraining Data:")
            for key, value in self._train_data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        if self._test_data:
            print("\nTest Data:")
            for key, value in self._test_data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

    def get_train_data(self, return_trials: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data as (neural_features, kinematics) arrays.

        Args:
            return_trials: If True, return data split by trials.

        Returns:
            Tuple of (X, y) arrays for training.
        """
        if self._train_data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        spikes = self._train_data.get("spikes")
        behavior = self._train_data.get("behavior")

        if spikes is None:
            raise ValueError("No spike data found in training set.")

        # Flatten trials if not returning by trial
        if not return_trials and spikes.ndim == 3:
            # Shape: (trials, time, neurons) -> (time*trials, neurons)
            n_trials, n_time, n_neurons = spikes.shape
            spikes = spikes.reshape(-1, n_neurons)

            if behavior is not None and behavior.ndim == 3:
                behavior = behavior.reshape(-1, behavior.shape[-1])

        X = spikes.astype(np.float32)
        y = behavior.astype(np.float32) if behavior is not None else None

        return X, y

    def get_test_data(self, return_trials: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get test data as (neural_features, kinematics) arrays.

        Args:
            return_trials: If True, return data split by trials.

        Returns:
            Tuple of (X, y) arrays for testing. y may be None for held-out test sets.
        """
        if self._test_data is None or not self._test_data:
            raise RuntimeError("No test data available.")

        spikes = self._test_data.get("spikes")
        behavior = self._test_data.get("behavior")

        if spikes is None:
            raise ValueError("No spike data found in test set.")

        if not return_trials and spikes.ndim == 3:
            n_trials, n_time, n_neurons = spikes.shape
            spikes = spikes.reshape(-1, n_neurons)

            if behavior is not None and behavior.ndim == 3:
                behavior = behavior.reshape(-1, behavior.shape[-1])

        X = spikes.astype(np.float32)
        y = behavior.astype(np.float32) if behavior is not None else None

        return X, y

    def get_trial_data(self, trial_idx: int) -> Dict[str, np.ndarray]:
        """
        Get data for a specific trial.

        Args:
            trial_idx: Index of the trial to retrieve.

        Returns:
            Dictionary with 'spikes' and optionally 'behavior' for the trial.
        """
        if self._train_data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        result = {}

        spikes = self._train_data.get("spikes")
        if spikes is not None and spikes.ndim == 3:
            if trial_idx >= len(spikes):
                raise IndexError(f"Trial index {trial_idx} out of range (max: {len(spikes)-1})")
            result["spikes"] = spikes[trial_idx]

        behavior = self._train_data.get("behavior")
        if behavior is not None and behavior.ndim == 3:
            result["behavior"] = behavior[trial_idx]

        return result

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the current dataset."""
        info = self.DATASET_INFO.get(self.dataset_name, {}).copy()
        info["name"] = self.dataset_name
        info["bin_size_ms"] = self.bin_size_ms

        if self._train_data:
            spikes = self._train_data.get("spikes")
            if spikes is not None:
                if spikes.ndim == 3:
                    info["n_trials"] = spikes.shape[0]
                    info["n_timepoints"] = spikes.shape[1]
                    info["n_neurons"] = spikes.shape[2]
                else:
                    info["n_timepoints"] = spikes.shape[0]
                    info["n_neurons"] = spikes.shape[1]

            behavior = self._train_data.get("behavior")
            if behavior is not None:
                info["n_behavior_dims"] = behavior.shape[-1]

        return info

    @classmethod
    def list_datasets(cls) -> List[Dict[str, str]]:
        """List all available NLB datasets with descriptions."""
        datasets = []
        for name in cls.AVAILABLE_DATASETS:
            info = cls.DATASET_INFO.get(name, {})
            datasets.append(
                {
                    "name": name,
                    "description": info.get("description", ""),
                    "brain_area": info.get("brain_area", ""),
                    "task": info.get("task", ""),
                }
            )
        return datasets

    @classmethod
    def download_dataset(
        cls,
        dataset_name: str,
        output_dir: Union[str, Path],
        verbose: bool = True,
    ) -> Path:
        """
        Download NLB dataset from DANDI archive.

        Args:
            dataset_name: Name of the dataset to download.
            output_dir: Directory to save downloaded files.
            verbose: Print download progress.

        Returns:
            Path to the downloaded data directory.
        """
        try:
            from dandi.download import download as dandi_download
        except ImportError:
            raise ImportError(
                "dandi package is required for downloading. " "Install with: pip install dandi"
            )

        dataset_name = dataset_name.lower()
        if dataset_name not in cls.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        info = cls.DATASET_INFO.get(dataset_name, {})
        dandiset_id = info.get("dandiset_id")

        if not dandiset_id:
            raise ValueError(f"No DANDI ID found for {dataset_name}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://dandiarchive.org/dandiset/{dandiset_id}"

        if verbose:
            print(f"Downloading {dataset_name} from {url}")
            print(f"Saving to: {output_dir}")

        dandi_download(url, output_dir=str(output_dir))

        return output_dir / dandiset_id

    def create_decoder_dataset(
        self,
        lag_bins: int = 0,
        smooth_spikes: bool = True,
        smooth_sigma: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test datasets ready for decoder training.

        Args:
            lag_bins: Number of time bins to lag behavior behind spikes.
            smooth_spikes: Whether to smooth spike counts.
            smooth_sigma: Gaussian smoothing sigma in ms.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test).
        """
        X_train, y_train = self.get_train_data()

        # Apply smoothing if requested
        if smooth_spikes:
            X_train = self._smooth_spikes(X_train, smooth_sigma)

        # Apply lag if requested
        if lag_bins > 0 and y_train is not None:
            X_train = X_train[:-lag_bins]
            y_train = y_train[lag_bins:]

        # Get test data if available
        try:
            X_test, y_test = self.get_test_data()
            if smooth_spikes:
                X_test = self._smooth_spikes(X_test, smooth_sigma)
            if lag_bins > 0 and y_test is not None:
                X_test = X_test[:-lag_bins]
                y_test = y_test[lag_bins:]
        except (RuntimeError, ValueError):
            # No test data - create from training
            split_idx = int(0.8 * len(X_train))
            X_test = X_train[split_idx:]
            y_test = y_train[split_idx:] if y_train is not None else None
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx] if y_train is not None else None

        return X_train, y_train, X_test, y_test

    def _smooth_spikes(self, spikes: np.ndarray, sigma_ms: float) -> np.ndarray:
        """Apply Gaussian smoothing to spike counts."""
        from scipy.ndimage import gaussian_filter1d

        sigma_bins = sigma_ms / self.bin_size_ms
        smoothed = gaussian_filter1d(spikes, sigma=sigma_bins, axis=0)
        return smoothed.astype(np.float32)
