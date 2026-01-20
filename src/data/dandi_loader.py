"""
DANDI Archive dataset loader.

Provides utilities for loading neural data from the DANDI Archive,
which hosts neurophysiology datasets in NWB (Neurodata Without Borders) format.

Reference:
    Rübel et al. (2022) "The Neurodata Without Borders ecosystem for
    neurophysiological data science"
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from pynwb import NWBHDF5IO

    PYNWB_AVAILABLE = True
except ImportError:
    PYNWB_AVAILABLE = False
    NWBHDF5IO = None

try:
    from dandi.dandiapi import DandiAPIClient

    DANDI_AVAILABLE = True
except ImportError:
    DANDI_AVAILABLE = False
    DandiAPIClient = None


class DANDIDataLoader:
    """
    Loader for DANDI Archive datasets.

    Supports browsing DANDI archives, downloading NWB files,
    and extracting neural data for decoder training.
    """

    RECOMMENDED_DANDISETS = {
        "000128": {
            "name": "MC_Maze (NLB)",
            "description": "Neural Latents Benchmark - Monkey reaching with maze",
            "species": "Macaque",
            "brain_area": "M1/PMd",
        },
        "000129": {
            "name": "MC_RTT (NLB)",
            "description": "Neural Latents Benchmark - Random target task",
            "species": "Macaque",
            "brain_area": "M1",
        },
        "000127": {
            "name": "Area2_Bump (NLB)",
            "description": "Neural Latents Benchmark - Somatosensory bump task",
            "species": "Macaque",
            "brain_area": "Area 2",
        },
        "000121": {
            "name": "Shenoy Lab Motor Cortex",
            "description": "Motor cortex recordings during reaching",
            "species": "Macaque",
            "brain_area": "M1/PMd",
        },
        "000053": {
            "name": "Hausser Lab Purkinje Cells",
            "description": "Cerebellar Purkinje cell recordings",
            "species": "Mouse",
            "brain_area": "Cerebellum",
        },
    }

    def __init__(
        self,
        data_dir: Union[str, Path],
        dandiset_id: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize DANDI data loader.

        Args:
            data_dir: Directory for storing downloaded data.
            dandiset_id: DANDI dataset ID (e.g., "000128").
            verbose: Print loading progress.
        """
        self.data_dir = Path(data_dir)
        self.dandiset_id = dandiset_id
        self.verbose = verbose

        self._nwb_files: List[Path] = []
        self._current_nwb = None
        self._spikes: Optional[np.ndarray] = None
        self._behavior: Optional[np.ndarray] = None

    def list_dandisets(self, search_term: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available dandisets, optionally filtered by search term.

        Args:
            search_term: Optional term to filter results.

        Returns:
            List of dandiset information dictionaries.
        """
        if not DANDI_AVAILABLE:
            raise ImportError("dandi package is required. Install with: pip install dandi")

        results = []

        with DandiAPIClient() as client:
            dandisets = client.get_dandisets()

            for ds in dandisets:
                info = {
                    "id": ds.identifier,
                    "name": ds.get_raw_metadata().get("name", ""),
                    "description": ds.get_raw_metadata().get("description", "")[:200],
                }

                if search_term:
                    search_lower = search_term.lower()
                    if (
                        search_lower in info["name"].lower()
                        or search_lower in info["description"].lower()
                    ):
                        results.append(info)
                else:
                    results.append(info)

                if len(results) >= 50:  # Limit results
                    break

        return results

    def download_dandiset(
        self,
        dandiset_id: Optional[str] = None,
        n_files: Optional[int] = None,
    ) -> List[Path]:
        """
        Download NWB files from a dandiset.

        Args:
            dandiset_id: DANDI dataset ID. Uses self.dandiset_id if not provided.
            n_files: Maximum number of files to download. None for all.

        Returns:
            List of downloaded file paths.
        """
        if not DANDI_AVAILABLE:
            raise ImportError("dandi package is required. Install with: pip install dandi")

        dandiset_id = dandiset_id or self.dandiset_id
        if not dandiset_id:
            raise ValueError("No dandiset_id provided")

        output_dir = self.data_dir / dandiset_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Downloading dandiset {dandiset_id} to {output_dir}")

        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(dandiset_id)
            assets = list(dandiset.get_assets())

            if n_files:
                assets = assets[:n_files]

            downloaded = []
            for asset in assets:
                if asset.path.endswith(".nwb"):
                    filepath = output_dir / asset.path
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    if not filepath.exists():
                        if self.verbose:
                            print(f"  Downloading {asset.path}")
                        asset.download(filepath)

                    downloaded.append(filepath)

        self._nwb_files = downloaded
        return downloaded

    def find_local_nwb_files(self, dandiset_id: Optional[str] = None) -> List[Path]:
        """
        Find locally stored NWB files.

        Args:
            dandiset_id: DANDI dataset ID to search in.

        Returns:
            List of NWB file paths.
        """
        search_dir = self.data_dir
        if dandiset_id or self.dandiset_id:
            search_dir = self.data_dir / (dandiset_id or self.dandiset_id)

        nwb_files = list(search_dir.rglob("*.nwb"))
        self._nwb_files = nwb_files

        if self.verbose:
            print(f"Found {len(nwb_files)} NWB files in {search_dir}")

        return nwb_files

    def load_nwb_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from a single NWB file.

        Args:
            filepath: Path to NWB file.

        Returns:
            Dictionary containing extracted neural and behavioral data.
        """
        if not PYNWB_AVAILABLE:
            raise ImportError("pynwb is required. Install with: pip install pynwb")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"NWB file not found: {filepath}")

        if self.verbose:
            print(f"Loading {filepath}")

        data = {
            "filepath": str(filepath),
            "units": None,
            "spike_times": [],
            "behavior": None,
            "metadata": {},
        }

        with NWBHDF5IO(str(filepath), "r") as io:
            nwbfile = io.read()

            # Extract metadata
            data["metadata"] = {
                "session_description": nwbfile.session_description,
                "identifier": nwbfile.identifier,
                "session_start_time": str(nwbfile.session_start_time),
            }

            # Extract units (spike data)
            if nwbfile.units is not None:
                units_df = nwbfile.units.to_dataframe()
                data["units"] = units_df

                # Extract spike times
                for idx in range(len(units_df)):
                    if "spike_times" in units_df.columns:
                        spike_times = units_df.iloc[idx]["spike_times"]
                        data["spike_times"].append(np.array(spike_times))

            # Extract behavioral data
            if nwbfile.processing:
                for module_name in nwbfile.processing:
                    module = nwbfile.processing[module_name]

                    # Look for behavior data
                    if "behavior" in module_name.lower():
                        for container_name in module.data_interfaces:
                            container = module.data_interfaces[container_name]

                            if hasattr(container, "spatial_series"):
                                for series_name in container.spatial_series:
                                    series = container.spatial_series[series_name]
                                    data["behavior"] = {
                                        "name": series_name,
                                        "data": np.array(series.data),
                                        "timestamps": (
                                            np.array(series.timestamps)
                                            if series.timestamps is not None
                                            else None
                                        ),
                                    }
                                    break

            # Also check acquisition for raw behavioral signals
            if data["behavior"] is None and nwbfile.acquisition:
                for acq_name in nwbfile.acquisition:
                    acq = nwbfile.acquisition[acq_name]
                    if "position" in acq_name.lower() or "cursor" in acq_name.lower():
                        data["behavior"] = {
                            "name": acq_name,
                            "data": np.array(acq.data),
                            "timestamps": (
                                np.array(acq.timestamps)
                                if hasattr(acq, "timestamps") and acq.timestamps is not None
                                else None
                            ),
                        }
                        break

        self._current_nwb = data
        return data

    def spike_times_to_binned(
        self,
        spike_times_list: List[np.ndarray],
        bin_size_s: float = 0.005,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert spike times to binned spike counts.

        Args:
            spike_times_list: List of spike time arrays for each unit.
            bin_size_s: Bin size in seconds.
            start_time: Start time for binning. None for auto.
            end_time: End time for binning. None for auto.

        Returns:
            Binned spike counts array of shape (n_bins, n_units).
        """
        if not spike_times_list:
            raise ValueError("No spike times provided")

        # Determine time range
        if start_time is None:
            start_time = min(st.min() for st in spike_times_list if len(st) > 0)
        if end_time is None:
            end_time = max(st.max() for st in spike_times_list if len(st) > 0)

        # Create bins
        n_bins = int(np.ceil((end_time - start_time) / bin_size_s))
        bin_edges = np.linspace(start_time, start_time + n_bins * bin_size_s, n_bins + 1)

        # Bin spike times
        n_units = len(spike_times_list)
        binned = np.zeros((n_bins, n_units), dtype=np.float32)

        for i, spike_times in enumerate(spike_times_list):
            if len(spike_times) > 0:
                counts, _ = np.histogram(spike_times, bins=bin_edges)
                binned[:, i] = counts

        return binned

    def get_decoder_data(
        self,
        bin_size_s: float = 0.005,
        smooth_sigma_ms: float = 50.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get data formatted for decoder training.

        Args:
            bin_size_s: Bin size for spike counts in seconds.
            smooth_sigma_ms: Gaussian smoothing sigma in ms.

        Returns:
            Tuple of (neural_features, kinematics). Kinematics may be None.
        """
        if self._current_nwb is None:
            raise RuntimeError("No NWB file loaded. Call load_nwb_file first.")

        # Get binned spikes
        if not self._current_nwb["spike_times"]:
            raise ValueError("No spike data in loaded NWB file")

        X = self.spike_times_to_binned(
            self._current_nwb["spike_times"],
            bin_size_s=bin_size_s,
        )

        # Apply smoothing
        if smooth_sigma_ms > 0:
            from scipy.ndimage import gaussian_filter1d

            sigma_bins = smooth_sigma_ms / (bin_size_s * 1000)
            X = gaussian_filter1d(X, sigma=sigma_bins, axis=0)

        # Get behavior if available
        y = None
        if self._current_nwb["behavior"] is not None:
            behavior = self._current_nwb["behavior"]["data"]
            timestamps = self._current_nwb["behavior"]["timestamps"]

            if timestamps is not None:
                # Resample behavior to match spike bins
                bin_times = np.arange(len(X)) * bin_size_s
                y = self._resample_behavior(behavior, timestamps, bin_times)
            else:
                # Assume same sampling
                min_len = min(len(X), len(behavior))
                X = X[:min_len]
                y = behavior[:min_len]

        return X.astype(np.float32), y.astype(np.float32) if y is not None else None

    def _resample_behavior(
        self,
        behavior: np.ndarray,
        behavior_times: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Resample behavioral data to target times using interpolation."""
        from scipy.interpolate import interp1d

        if behavior.ndim == 1:
            behavior = behavior.reshape(-1, 1)

        n_dims = behavior.shape[1]
        resampled = np.zeros((len(target_times), n_dims), dtype=np.float32)

        for i in range(n_dims):
            interp_func = interp1d(
                behavior_times,
                behavior[:, i],
                kind="linear",
                fill_value="extrapolate",
            )
            resampled[:, i] = interp_func(target_times)

        return resampled

    def load_multiple_files(
        self,
        max_files: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load and concatenate data from multiple NWB files.

        Args:
            max_files: Maximum number of files to load.

        Returns:
            Concatenated (neural_features, kinematics) arrays.
        """
        if not self._nwb_files:
            self.find_local_nwb_files()

        if not self._nwb_files:
            raise RuntimeError("No NWB files found")

        files_to_load = self._nwb_files[:max_files] if max_files else self._nwb_files

        all_X = []
        all_y = []

        for filepath in files_to_load:
            try:
                self.load_nwb_file(filepath)
                X, y = self.get_decoder_data()
                all_X.append(X)
                if y is not None:
                    all_y.append(y)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load {filepath}: {e}")

        X_concat = np.concatenate(all_X, axis=0)
        y_concat = np.concatenate(all_y, axis=0) if all_y else None

        return X_concat, y_concat

    @classmethod
    def get_recommended_datasets(cls) -> List[Dict[str, Any]]:
        """Get list of recommended dandisets for neural decoding."""
        return [{"id": id_, **info} for id_, info in cls.RECOMMENDED_DANDISETS.items()]
