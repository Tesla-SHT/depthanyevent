"""
MVSEC Dataset Sequence Handler

This module provides a PyTorch Dataset implementation for the MVSEC (Multi Vehicle 
Stereo Event Camera) dataset. It handles loading and processing of event data, 
depth maps, and RGB images from MVSEC sequences for event-based depth estimation.

Key MVSEC Features:
- DAVIS sensor event data stored in HDF5 format
- Dual data sources: precomputed voxel grids or raw events
- Automotive/outdoor scenarios with challenging lighting
- Stereo vision setup with synchronized RGB cameras
- Ground truth depth from LiDAR sensors

MVSEC Dataset Structure:
sequence_name (e.g. outdoor_day1)
├── depth/
│   └── data/
│       ├── 000000.npy
│       └── timestamps.txt
├── rgb/
│   ├── davis/  (or davis_left_sync for test)
│   │   ├── 000000.png  
│   │   └── timestamps.txt
│   └── davis_{postfix}/  (for self-supervised mode)
├── events/
│   └── voxels/  (optional precomputed)
│       ├── 000000.npy
│       └── timestamps.txt
└── ../hdf5/
    └── data.hdf5  (raw event data)

The dataset supports both voxel-based (precomputed) and raw event processing modes,
with automatic fallback between them based on data availability.
"""

import os.path
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import weakref

import cv2
import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset

from ...events import EventRepresentation
from .eventslicer import EventSlicer
from ..constants import MVSEC_HEIGHT, MVSEC_WIDTH
from ...utils import Augmentator

def find_closest(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Find closest values in array b for each value in array a.
    
    This function performs temporal alignment by finding the nearest timestamp
    matches between two sorted arrays, commonly used for synchronizing different
    sensor data streams in the MVSEC dataset.
    
    Args:
        a: Query array - timestamps to find matches for
        b: Target array - sorted timestamps to search within
        
    Returns:
        Array of indices in b corresponding to closest values for each element in a
        
    Example:
        >>> a = np.array([1.5, 2.7, 4.1])
        >>> b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> find_closest(a, b)
        array([1, 1, 3])  # Indices of closest values in b
    """
    idx = np.searchsorted(b, a)  # Find insertion indices for elements of a in sorted b
    idx = np.clip(idx, 1, len(b) - 1)  # Ensure indices stay within valid bounds
    left = b[idx - 1]  # Values to the left of insertion point
    right = b[idx]  # Values to the right of insertion point
    # Choose left or right based on which is closer
    closest = np.where(np.abs(a - left) <= np.abs(a - right), idx - 1, idx)
    return closest


class MVSECSequence(Dataset):
    """
    MVSEC dataset PyTorch Dataset implementation for event-based depth estimation.
    
    This class handles loading and processing of MVSEC sequences, supporting both
    precomputed voxel grids and raw event data. It provides synchronized access
    to event data, depth maps, and RGB images with proper temporal alignment.
    
    MVSEC Sequence Structure Expected:
    sequence_path/
    ├── depth/
    │   └── data/
    │       ├── 000000.npy (depth maps)
    │       └── timestamps.txt
    ├── rgb/
    │   ├── davis/ (or davis_left_sync for test)
    │   │   ├── 000000.png
    │   │   └── timestamps.txt  
    │   └── davis_{postfix}/ (for self-supervised)
    ├── events/
    │   └── voxels/ (optional precomputed)
    │       ├── 000000.npy
    │       └── timestamps.txt
    └── ../hdf5/
        └── data.hdf5 (raw event data)
    
    Args:
        sequence_path: Path to specific MVSEC sequence directory
        event_representation: Event representation method (VoxelGrid, etc.)
        time_window_ms: Time window in milliseconds for event accumulation
        augmentator: Optional data augmentation handler
        load_images: Whether to load RGB images (default: False)
        use_voxels: Use precomputed voxels vs raw events (default: False) 
        overfit: Enable overfitting mode for debugging (default: False)
        sequence_window: Number of frames per sequence window (default: 1)
        sequence_step: Step size between sequence windows (default: 1)
        split: Dataset split identifier ("train", "val", "test")
        self_supervised: Enable self-supervised mode (default: False)
        postfix: Postfix for RGB directory selection (default: "")
    """
    
    def __init__(
        self,
        sequence_path: str,
        event_representation: EventRepresentation,
        time_window_ms: int,
        augmentator: Optional[Augmentator] = None,
        load_images: bool = False,
        use_voxels: bool = False,
        overfit: bool = False,
        sequence_window: int = 1,
        sequence_step: int = 1,
        split: str = "train",
        self_supervised: bool = False,
        postfix: str = "",
    ):
        """Initialize MVSEC sequence dataset with validation and preprocessing."""
        
        # Store core configuration
        self.event_representation = event_representation
        self.load_images = load_images
        self.augmentator = augmentator
        self.self_supervised = self_supervised
        self.postfix = postfix

        # Validate and adjust sequence parameters
        if sequence_window < 1:
            print(f"Warning: sequence_window={sequence_window} < 1. Setting to 1.")
            sequence_window = 1

        self.sequence_window = sequence_window

        if sequence_window < sequence_step:
            print(f"Warning: sequence_window={sequence_window} < sequence_step={sequence_step}. Setting sequence_step to {sequence_window}")
            sequence_step = sequence_window

        self.sequence_step = sequence_step

        # Save output dimensions
        self.height = MVSEC_HEIGHT
        self.width = MVSEC_WIDTH

        # Check if hdf5 file exists for raw event data
        event_data_hdf5_path = os.path.join(sequence_path, "..", "hdf5", "data.hdf5")
        self.event_data_hdf5_exists = os.path.isfile(event_data_hdf5_path)

        # Force voxel mode if raw events are not available
        if not self.event_data_hdf5_exists and not use_voxels:
            print(f"Warning: Cannot find {event_data_hdf5_path}, setting use_voxels=True.")
            use_voxels = True

        self.use_voxels = use_voxels

        if use_voxels:
            time_window_ms = 50

        # Convert time window to microseconds
        delta_t_us = time_window_ms * 1000

        # Load depth timestamps (primary temporal reference)
        depth_timestamps_path = os.path.join(sequence_path, "depth", "data", "timestamps.txt")
        self.timestamps_depth = (
            np.loadtxt(depth_timestamps_path)[:, 1] * 1e6
        ).astype("int64")

        # Determine RGB folder name based on dataset split
        self.rgb_foldername = "davis_left_sync" if split == "test" else "davis"

        # Load RGB timestamps if images are required
        rgb_timestamps_path = os.path.join(
            sequence_path, "rgb", self.rgb_foldername, "timestamps.txt"
        )
        
        if os.path.exists(rgb_timestamps_path):
            self.timestamps_rgb = (
                np.loadtxt(rgb_timestamps_path)[:, 1] * 1e6
            ).astype("int64")
        else:
            self.timestamps_rgb = None
            if load_images:
                print(f"Warning: RGB timestamps not found at {rgb_timestamps_path}, disabling image loading")
                self.load_images = False

        # Handle self-supervised mode
        if self.self_supervised:
            self.timestamps_depth = self.timestamps_rgb if self.timestamps_rgb is not None else self.timestamps_depth

        # Setup RGB-depth temporal alignment if images are loaded
        if self.load_images and self.timestamps_rgb is not None:
            # For each depth timestamp, find the closest rgb timestamp
            self.rgb_indices = find_closest(self.timestamps_depth, self.timestamps_rgb)
        else:
            self.rgb_indices = None

        self.dataset_length = len(self.timestamps_depth)
        
        # Let's get the correct disparity for each frame
        if not self.self_supervised:
            self.base_depth_path = os.path.join(sequence_path, "depth", "data")
        else:
            self.base_depth_path = os.path.join(sequence_path, "rgb", f"{self.rgb_foldername}_{self.postfix}")

        depth_files = []
        for entry in os.listdir(self.base_depth_path):
            if entry.endswith(".npy" if not self.self_supervised else ".npz"):
                depth_files.append(entry)

        depth_files.sort()
        self.depths = depth_files
        assert len(self.depths) == len(self.timestamps_depth)

        if use_voxels:
            self.base_voxels_path = os.path.join(sequence_path, "events", "voxels")

            # Same as timestamps_depth - 1
            self.timestamps_voxels = (np.loadtxt(os.path.join(sequence_path, "events", "voxels", "timestamps.txt"))[:,1] * 1e6).astype("int64")

            voxels_files = []
            for entry in os.listdir(self.base_voxels_path):
                if entry.endswith(".npy"):
                    voxels_files.append(entry)

            if len(voxels_files) != len(self.timestamps_depth):
                print(
                    f"len(voxels_files)={len(voxels_files)} != len(self.timestamps_depth)={len(self.timestamps_depth)}"
                )
                
                # print("Skipping first depth frame")
                # # Skip the first frame because we don't have events before it
                # self.timestamps_depth = self.timestamps_depth[1:]
                # self.depths = self.depths[1:]
                # self.dataset_length = len(self.timestamps_depth)

                # Maybe better use searchsorted?
                self.depth_indices = find_closest(self.timestamps_voxels, self.timestamps_depth)
                self.timestamps_depth = self.timestamps_depth[self.depth_indices]
                self.depths = [self.depths[i] for i in self.depth_indices]
                self.dataset_length = len(self.timestamps_depth)
            else:
                self.depth_indices = np.arange(len(self.timestamps_depth))

            assert len(voxels_files) == len(
                self.timestamps_depth
            ), f"len(voxels_files)={len(voxels_files)} != len(self.timestamps_depth)={len(self.timestamps_depth)}"

            voxels_files.sort()
            self.voxels = voxels_files

        # Load and validate image files if image loading is enabled
        if self.load_images and self.timestamps_rgb is not None and self.rgb_indices is not None:
            self.base_left_images_path = os.path.join(sequence_path, "rgb", self.rgb_foldername)

            left_images_files = []
            for entry in os.listdir(self.base_left_images_path):
                if entry.endswith(".png"):
                    left_images_files.append(entry)
            
            # Validate RGB data availability
            if len(left_images_files) != len(self.timestamps_rgb):
                print(f"Warning: RGB files count ({len(left_images_files)}) != RGB timestamps count ({len(self.timestamps_rgb)})")
                print("Disabling image loading due to data mismatch")
                self.load_images = False
                self.timestamps_rgb = None
                self.rgb_indices = None
            else:
                left_images_files.sort()
                self.left_images = left_images_files
                
                # Apply temporal alignment indices
                self.left_images = [self.left_images[i] for i in self.rgb_indices]
                aligned_timestamps_rgb = self.timestamps_rgb[self.rgb_indices]
                
                # Apply depth window filtering if applicable
                if hasattr(self, 'depth_indices'):
                    self.left_images = [self.left_images[i] for i in self.depth_indices]
                    self.timestamps_rgb = aligned_timestamps_rgb[self.depth_indices]
                else:
                    self.timestamps_rgb = aligned_timestamps_rgb

                # Final validation
                expected_length = len(self.timestamps_depth)
                if len(self.left_images) != expected_length or len(self.timestamps_rgb) != expected_length:
                    print(f"Warning: Image alignment failed - expected {expected_length} samples, got {len(self.left_images)} images and {len(self.timestamps_rgb)} RGB timestamps")
                    self.load_images = False
                    self.timestamps_rgb = None
        else:
            # Ensure no image-related attributes if loading is disabled
            self.timestamps_rgb = None
            self.rgb_indices = None

        # Prepare event time windows for depth-aligned processing
        self.depth_aligned_event_windows = []
        self.rgb_aligned_event_windows = []

        for timestamp in self.timestamps_depth:
            self.depth_aligned_event_windows.append(
                (timestamp - delta_t_us, timestamp)
            )

        # Prepare RGB-aligned event windows only if images are loaded
        if self.load_images and self.timestamps_rgb is not None:
            for timestamp in self.timestamps_rgb:
                self.rgb_aligned_event_windows.append(
                    (timestamp - delta_t_us, timestamp)
                )

        # Load HDF5 files
        hdf5_files_dict = {}

        if self.event_data_hdf5_exists and not use_voxels:
            hdf5_files_dict["data"] = h5py.File(event_data_hdf5_path, "r")

            self.event_slicer = {
                "data": EventSlicer(hdf5_files_dict["data"]),
            }

            # if self.event_gt_hdf5_exists:
            #     hdf5_files_dict["gt"] = h5py.File(event_gt_hdf5_path, 'r')

        # Let's close the h5 file when finished
        self._finalizer = weakref.finalize(
            self, self.close_callback, hdf5_files_dict
        )

        if self.dataset_length < self.sequence_window:
            print(
                f"Warning: dataset_length={self.dataset_length} < sequence_window={self.sequence_window}. Setting sequence_window to {self.dataset_length}"
            )
            self.sequence_window = self.dataset_length
        else:
            self.dataset_length = (self.dataset_length - self.sequence_window) // self.sequence_step + 1

        # Handle overfitting mode (for debugging/testing)
        if overfit:
            self.depths = self.depths[0:self.sequence_window]
            self.depth_aligned_event_windows = self.depth_aligned_event_windows[0:self.sequence_window]
            self.rgb_aligned_event_windows = self.rgb_aligned_event_windows[0:self.sequence_window]
            self.timestamps_depth = self.timestamps_depth[0:self.sequence_window]
            
            # Only slice RGB timestamps if they exist
            if self.timestamps_rgb is not None:
                self.timestamps_rgb = self.timestamps_rgb[0:self.sequence_window]
                
            self.dataset_length = self.sequence_window

            if use_voxels:
                self.voxels = self.voxels[0:self.sequence_window]

            if self.load_images and hasattr(self, 'left_images'):
                self.left_images = self.left_images[0:self.sequence_window]

    def events_to_representation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, t: np.ndarray) -> Optional[torch.Tensor]:
        """
        Convert raw event data to specified event representation.
        
        This method normalizes event coordinates and timestamps, then applies
        the configured event representation (e.g., VoxelGrid, Histogram).
        
        Args:
            x: X coordinates of events
            y: Y coordinates of events  
            p: Polarities of events (+1 or -1)
            t: Timestamps of events in microseconds
            
        Returns:
            Tensor representation of events according to self.event_representation
            
        Note:
            Timestamps are normalized to [0, 1] range relative to the time window
        """
        # Normalize timestamps to [0, 1] range
        t = (t - t[0]).astype("float32")
        t = t / t[-1] if t[-1] > 0 else t  # Avoid division by zero
        
        # Convert coordinates and polarities to float32
        x = x.astype("float32")
        y = y.astype("float32")
        pol = p.astype("float32")
        return self.event_representation.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
            torch.from_numpy(t),
        )

    def get_depth_map(self, filepath: str) -> torch.Tensor:
        """
        Load and preprocess a depth map from file.
        
        This method loads depth data from either .npy (normal mode) or .npz 
        (self-supervised mode) files, converts to float32, and cleans invalid values.
        
        Args:
            filepath: Path to depth file (.npy for normal, .npz for self-supervised)
            
        Returns:
            Preprocessed depth map tensor with NaN/Inf values replaced by 0
            
        Raises:
            AssertionError: If file doesn't exist
            
        Note:
            In self-supervised mode, extracts 'depth' key from .npz file
        """
        assert os.path.isfile(filepath), f"Depth file not found: {filepath}"
        
        # Load depth data (handle both .npy and .npz formats)
        depth_data = np.load(filepath)
        depth_array = depth_data['depth'] if self.self_supervised else depth_data
        depth_array = depth_array.astype("float32")
        
        # Convert to tensor and clean invalid values
        depthmap = torch.tensor(depth_array)
        depthmap[torch.isnan(depthmap)] = 0
        depthmap[torch.isinf(depthmap)] = 0
        
        return depthmap

    @staticmethod
    def close_callback(h5f_dict: Dict[str, h5py.File]) -> None:
        """
        Safely close all HDF5 file handles in dictionary.
        
        Used as cleanup callback for HDF5 file management to prevent
        resource leaks when the dataset is garbage collected.
        
        Args:
            h5f_dict: Dictionary of HDF5 file handles to close
        """
        for key in h5f_dict:
            try:
                h5f_dict[key].close()
            except Exception as e:
                print(f"Warning: Failed to close HDF5 file {key}: {e}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.dataset_length

    @staticmethod
    def load_voxel(filepath: str) -> torch.Tensor:
        """
        Load precomputed voxel grid from file.
        
        Args:
            filepath: Path to .npy file containing voxel grid data
            
        Returns:
            Voxel grid tensor converted to float32
            
        Raises:
            AssertionError: If file doesn't exist
        """
        assert os.path.isfile(filepath), f"Voxel file not found: {filepath}"
        return torch.tensor(np.load(filepath).astype("float32"))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        This method loads a sequence of frames starting from the given index,
        including depth maps, event data (raw or voxel), and RGB images if enabled.
        The sequence length is determined by self.sequence_window.
        
        Args:
            index: Starting index for the sequence
            
        Returns:
            Dictionary containing:
            - 'depth': Depth map tensor(s) [TCHW] where T=sequence_window
            - 'depth_aligned_events': Event representation aligned with depth
            - 'rgb_aligned_events': Event representation aligned with RGB (if load_images=True)
            - 'rgb': RGB image tensor(s) (if load_images=True)
            
        Note:
            For sequence_window > 1, returns stacked tensors along time dimension.
            Applies data augmentation if augmentator is specified.
        """
        start_index = index * self.sequence_step
        end_index = start_index + self.sequence_window

        to_return_list = []

        for _index in range(start_index, end_index):
            to_return = {}

            depth = self.get_depth_map(
                os.path.join(self.base_depth_path, self.depths[_index])
            )
            to_return["depth"] = depth
            # to_return["timestamp_depth"] = self.timestamps_depth[_index]

            if self.use_voxels:
                voxel = self.load_voxel(
                    os.path.join(self.base_voxels_path, self.voxels[_index])
                )
                this_er = self.event_representation.convert_from_voxels(voxel)
                to_return["depth_aligned_events"] = this_er

                if self.load_images:
                    to_return["rgb_aligned_events"] = (
                        this_er  # TODO: load voxels aligned with images for validation and test sets
                    )
            else:
                if self.load_images:
                    _tmp = zip(
                        ["depth_aligned_events", "rgb_aligned_events"],
                        [
                            self.depth_aligned_event_windows,
                            self.rgb_aligned_event_windows,
                        ],
                    )
                else:
                    _tmp = zip(
                        ["depth_aligned_events"], [self.depth_aligned_event_windows]
                    )

                for key, timestamp_window in _tmp:
                    event_data = self.event_slicer["data"].get_events(
                        *timestamp_window[index]
                    )

                    p = event_data["p"]
                    t = event_data["t"]
                    x = event_data["x"]
                    y = event_data["y"]

                    event_representation = self.events_to_representation(x, y, p, t)
                    to_return[key] = event_representation

            if self.load_images:
                left_image = cv2.imread(
                    os.path.join(self.base_left_images_path, self.left_images[_index])
                )
                
                # Ensure image was loaded successfully
                if left_image is not None:
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                    if len(left_image.shape) == 2:
                        left_image = left_image[..., None]
                    left_image = left_image.transpose(2, 0, 1) / 255.0
                    left_image = left_image.astype("float32")
                    to_return["rgb"] = torch.tensor(left_image)
                else:
                    print(f"Warning: Failed to load image {self.left_images[_index]}")
                    # Create a zero tensor as fallback
                    to_return["rgb"] = torch.zeros((3, self.height, self.width), dtype=torch.float32)

            # Check if all tensors have three dimensions CHW
            for key in to_return:
                if len(to_return[key].shape) == 2:
                    to_return[key] = to_return[key].unsqueeze(0)
            
            to_return_list.append(to_return)

        # stack all tensors in new axis=0 -> new tensor shape TCHW
        to_return = {}
        for key in to_return_list[0]:
            to_return[key] = torch.stack([to_return[key] for to_return in to_return_list], dim=0)


        # Apply data augmentations if augmentator is available
        if self.augmentator is not None:
            augmentation_keys = [
                "rgb",
                "depth", 
                "depth_aligned_events",
                "rgb_aligned_events",
            ]
            
            # Prepare data for augmentation (only keys that exist)
            augmentation_data = {
                key: to_return[key] 
                for key in augmentation_keys 
                if key in to_return
            }
            
            # Apply augmentation and update results
            augmented_data = self.augmentator(augmentation_data)
            
            if augmented_data is not None:
                to_return.update(augmented_data)
            else:
                print("Warning: Augmentator returned None, skipping augmentation")

        return to_return
