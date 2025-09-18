"""
DSEC Dataset Sequence Handler

This module provides a PyTorch Dataset implementation for the DSEC (Dynamic Stereo 
Event Camera) dataset. It handles loading and processing of event data, depth/disparity 
maps, and RGB images from DSEC sequences.

The dataset supports:
- Event-based depth estimation
- Multi-modal input (events + RGB images)
- Temporal sequences for recurrent models
- Data augmentation
- Self-supervised learning modes

DSEC Dataset Structure:
seq_name (e.g. zurich_city_11_a)
├── disparity
│   ├── event
│   │   ├── 000000.png
│   │   └── ...
│   └── timestamps.txt
├── images
│   ├── left
│   │   ├── ev_inf
│   │   └── ...
│   └── timestamps.txt
├── events
│   ├── left
│   │   ├── events.h5
│   │   └── rectify_map.h5
│   └── right
│       ├── events.h5
│       └── rectify_map.h5
└── calibration
    └── cam_to_cam.yaml
"""

import os.path
from pathlib import Path
import weakref
import cv2
import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

from ...events import EventRepresentation
from .eventslicer import EventSlicer
from ..constants import DSEC_WIDTH, DSEC_HEIGHT
from ...utils import Augmentator

def find_closest(array_a: np.ndarray, array_b: np.ndarray) -> np.ndarray:
    """
    Find the closest elements in array_b for each element in array_a.
    
    This function uses binary search to efficiently find the closest matches,
    which is particularly useful for timestamp alignment between different
    data streams (e.g., events and RGB images).
    
    Args:
        array_a (np.ndarray): Array of query values (e.g., depth timestamps)
        array_b (np.ndarray): Array of reference values (e.g., RGB timestamps)
        
    Returns:
        np.ndarray: Indices of closest elements in array_b for each element in array_a
    """
    # Find insertion points using binary search
    indices = np.searchsorted(array_b, array_a)
    
    # Ensure indices are within valid range [1, len(array_b)-1]
    indices = np.clip(indices, 1, len(array_b) - 1)
    
    # Get left and right neighbors
    left_values = array_b[indices - 1]
    right_values = array_b[indices]
    
    # Choose the closest neighbor based on absolute distance
    closest_indices = np.where(
        np.abs(array_a - left_values) <= np.abs(array_a - right_values),
        indices - 1,  # Left neighbor is closer
        indices       # Right neighbor is closer
    )
    
    return closest_indices

class DsecSequence(Dataset):
    """
    PyTorch Dataset for DSEC (Dynamic Stereo Event Camera) sequences.
    
    This dataset handles loading and processing of:
    - Event data from stereo event cameras
    - Depth/disparity maps
    - RGB images (optional)
    - Camera calibration parameters
    
    The dataset supports temporal sequences for recurrent models and various
    data augmentation techniques.
    
    Args:
        sequence_path: Path to the DSEC sequence directory
        event_representation: Event representation method (histogram, voxel grid, etc.)
        time_window_ms: Time window for event accumulation in milliseconds
        augmentator: Data augmentation pipeline (optional)
        load_images: Whether to load RGB images alongside events
        overfit: If True, use only sequence_window frames for overfitting
        sequence_window: Number of consecutive frames per sample
        sequence_step: Step size between consecutive samples
        split: Dataset split ('train', 'val', 'test')
        self_supervised: Whether to use self-supervised learning mode
        postfix: Postfix for file naming in self-supervised mode
    """

    def __init__(
        self,
        sequence_path: str,
        event_representation: EventRepresentation,
        time_window_ms: int,
        augmentator: Augmentator | None = None,
        load_images: bool = False,
        overfit: bool = False,
        sequence_window: int = 1,
        sequence_step: int = 1,
        split: str = "train",
        self_supervised: bool = False,
        postfix: str = "",
    ):
        # Store core parameters
        self.event_representation = event_representation
        self.load_images = load_images
        self.augmentator = augmentator
        self.self_supervised = self_supervised
        self.postfix = postfix

        # Validate and set sequence parameters
        if sequence_window < 1:
            print(f"Warning: sequence_window={sequence_window} < 1. Setting to 1.")
            sequence_window = 1
        self.sequence_window = sequence_window

        if sequence_window < sequence_step:
            print(f"Warning: sequence_window={sequence_window} < sequence_step={sequence_step}. "
                  f"Setting sequence_step to {sequence_window}")
            sequence_step = sequence_window
        self.sequence_step = sequence_step

        # Set output dimensions from DSEC constants
        self.height = DSEC_HEIGHT
        self.width = DSEC_WIDTH

        # Convert time window from milliseconds to microseconds
        delta_t_us = time_window_ms * 1000

        # Load timestamps and setup data paths
        self._load_timestamps(sequence_path)
        self._setup_data_paths(sequence_path)
        self._load_disparity_files()
        
        if load_images:
            self._load_image_files()
            self._align_timestamps()

        # Setup event windows for each timestamp
        self._setup_event_windows(delta_t_us)

        # Load event data and calibration
        self._load_event_data(sequence_path)
        self._load_calibration(sequence_path)

        # Adjust dataset length for sequences
        self._adjust_dataset_length()

        # Handle overfitting mode
        if overfit:
            self._apply_overfit_mode()

    def _load_timestamps(self, sequence_path: str) -> None:
        """Load timestamp data from the sequence directory."""
        # Load disparity timestamps (skip first frame as no events precede it)
        disparity_timestamp_path = os.path.join(sequence_path, "disparity", "timestamps.txt")
        self.timestamps_disparity = np.loadtxt(disparity_timestamp_path, dtype="int64")[1:]

        # Load RGB timestamps
        rgb_timestamp_path = os.path.join(sequence_path, "images", "timestamps.txt")
        self.timestamps_rgb = np.loadtxt(rgb_timestamp_path, dtype="int64")

        # In self-supervised mode, use RGB timestamps for disparity
        if self.self_supervised:
            self.timestamps_disparity = self.timestamps_rgb[1:]

        self.disparity_indices = np.arange(len(self.timestamps_disparity))
        self.dataset_length = len(self.timestamps_disparity)

    def _setup_data_paths(self, sequence_path: str) -> None:
        """Setup paths for disparity/depth and image data."""
        if not self.self_supervised:
            # Standard mode: use disparity maps from event-based estimation
            self.base_disparity_path = os.path.join(sequence_path, "disparity", "event")
        else:
            # Self-supervised mode: use depth predictions
            self.base_disparity_path = os.path.join(
                sequence_path, "images", "left", f"ev_inf_{self.postfix}"
            )

        if self.load_images:
            self.base_left_images_path = os.path.join(sequence_path, "images", "left", "ev_inf")

    def _load_disparity_files(self) -> None:
        """Load and sort disparity/depth files."""
        file_extension = ".png" if not self.self_supervised else ".npz"
        
        disparity_files = [
            entry for entry in os.listdir(self.base_disparity_path)
            if entry.endswith(file_extension)
        ]
        
        disparity_files.sort()
        # Skip first file as we skip first timestamp
        self.disparities = disparity_files[1:]
        
        assert len(self.disparities) == len(self.timestamps_disparity), (
            f"Mismatch: {len(self.disparities)} disparity files != "
            f"{len(self.timestamps_disparity)} timestamps"
        )

    def _load_image_files(self) -> None:
        """Load and sort RGB image files."""
        if not self.load_images:
            return
            
        image_files = [
            entry for entry in os.listdir(self.base_left_images_path)
            if entry.endswith(".png")
        ]
        
        assert len(image_files) == len(self.timestamps_rgb), (
            f"Mismatch: {len(image_files)} image files != {len(self.timestamps_rgb)} RGB timestamps"
        )
        
        image_files.sort()
        self.left_images = image_files

    def _align_timestamps(self) -> None:
        """Align RGB timestamps with disparity timestamps."""
        if not self.load_images:
            return
            
        # Find closest RGB timestamp for each disparity timestamp
        self.rgb_indices = find_closest(self.timestamps_disparity, self.timestamps_rgb)
        
        # Align image files and timestamps
        self.left_images = [self.left_images[i] for i in self.rgb_indices]
        self.left_images = [self.left_images[i] for i in self.disparity_indices]
        
        aligned_rgb_timestamps = self.timestamps_rgb[self.rgb_indices]
        self.timestamps_rgb = aligned_rgb_timestamps[self.disparity_indices]

        # Verify alignment
        assert len(self.left_images) == len(self.timestamps_disparity) == len(self.timestamps_rgb), (
            f"Alignment failed: {len(self.left_images)} images, "
            f"{len(self.timestamps_disparity)} disparity timestamps, "
            f"{len(self.timestamps_rgb)} RGB timestamps"
        )

    def _setup_event_windows(self, delta_t_us: int) -> None:
        """Setup event time windows for each timestamp."""
        # Create event windows aligned with disparity timestamps
        self.disparity_aligned_event_windows = []
        for timestamp in self.timestamps_disparity:
            self.disparity_aligned_event_windows.append(
                (timestamp - delta_t_us, timestamp)
            )

        # Create event windows aligned with RGB timestamps if needed
        self.rgb_aligned_event_windows = []
        if self.load_images:
            for timestamp in self.timestamps_rgb:
                self.rgb_aligned_event_windows.append(
                    (timestamp - delta_t_us, timestamp)
                )

    def _load_event_data(self, sequence_path: str) -> None:
        """Load event data files and rectification maps."""
        # Load event files (currently only left camera supported)
        event_file = {
            "left": h5py.File(os.path.join(sequence_path, "events", "left", "events.h5"), "r"),
        }

        # Load rectification maps for event cameras
        self.rectify_event_maps = {}
        for stereo in ["left"]:
            rectify_path = os.path.join(sequence_path, "events", stereo, "rectify_map.h5")
            with h5py.File(rectify_path, "r") as h5_rect:
                # Load rectification map data
                rectify_data = h5_rect["rectify_map"]
                # Convert HDF5 dataset to numpy array
                self.rectify_event_maps[stereo] = np.array(rectify_data)

        # Create event slicers for efficient temporal access
        self.event_slicer = {
            "left": EventSlicer(event_file["left"]),
        }
        
        # Ensure proper cleanup of HDF5 files
        self._finalizer = weakref.finalize(self, self.close_callback, event_file)

    def _load_calibration(self, sequence_path: str) -> None:
        """Load camera calibration parameters from YAML file."""
        cam_to_cam_path = os.path.join(sequence_path, "calibration", "cam_to_cam.yaml")
        
        with open(cam_to_cam_path, "r") as file:
            cam_to_cam = yaml.safe_load(file)
            
            # Load intrinsic parameters
            self._load_intrinsics(cam_to_cam["intrinsics"])
            
            # Load extrinsic parameters
            self._load_extrinsics(cam_to_cam["extrinsics"])
            
            # Load disparity-to-depth conversion matrices
            self._load_disparity_conversion(cam_to_cam["disparity_to_depth"])

    def _load_intrinsics(self, intrinsics: dict) -> None:
        """
        Load intrinsic camera parameters.
        
        DSEC Camera Layout:
        - cam0: Event camera left
        - cam1: Frame camera left  
        - cam2: Frame camera right
        - cam3: Event camera right
        - camRectX: Rectified version of camX
        """
        # Load left event camera intrinsics
        cam0_matrix = np.array(intrinsics["cam0"]["camera_matrix"])
        self.intrinsic_event_l = np.array([
            [cam0_matrix[0], 0, cam0_matrix[2]],
            [0, cam0_matrix[1], cam0_matrix[3]],
            [0, 0, 1]
        ])
        
        # Load distortion parameters
        self.distortion_parameters = np.array(intrinsics["cam0"]["distortion_coeffs"])

    def _load_extrinsics(self, extrinsics: dict) -> None:
        """
        Load extrinsic camera parameters.
        
        Transformations:
        - T_XY: Transforms point from camY to camX coordinate frame
        - R_rectX: Rotates point from camX to camRectX coordinate frame
        """
        # Compose transformation: T_32 @ T_21 @ T_10 (cam0 -> cam3)
        self.T_event = (np.array(extrinsics["T_32"]) @ 
                       np.array(extrinsics["T_21"]) @ 
                       np.array(extrinsics["T_10"]))
        
        # Compose rotation: inv(R_rect3) @ R_rect0
        self.R_event = (np.linalg.inv(np.array(extrinsics["R_rect3"])) @ 
                       np.array(extrinsics["R_rect0"]))

    def _load_disparity_conversion(self, disparity_to_depth: dict) -> None:
        """
        Load disparity-to-depth conversion matrices.
        
        Available conversions:
        - cams_03: Event cameras (cam0 <-> cam3)
        - cams_12: Frame cameras (cam1 <-> cam2)
        """
        self.q_event = np.array(disparity_to_depth["cams_03"])
        self.q_frame = np.array(disparity_to_depth["cams_12"])

    def _adjust_dataset_length(self) -> None:
        """Adjust dataset length based on sequence parameters."""
        if self.dataset_length < self.sequence_window:
            print(f"Warning: dataset_length={self.dataset_length} < "
                  f"sequence_window={self.sequence_window}. "
                  f"Setting sequence_window to {self.dataset_length}")
            self.sequence_window = self.dataset_length
        else:
            # Calculate number of possible sequences
            self.dataset_length = ((self.dataset_length - self.sequence_window) // 
                                 self.sequence_step + 1)

    def _apply_overfit_mode(self) -> None:
        """Apply overfitting mode by limiting data to sequence_window size."""
        # Limit all data arrays to sequence_window size
        self.disparities = self.disparities[:self.sequence_window]
        self.disparity_aligned_event_windows = self.disparity_aligned_event_windows[:self.sequence_window]
        self.rgb_aligned_event_windows = self.rgb_aligned_event_windows[:self.sequence_window]
        self.timestamps_disparity = self.timestamps_disparity[:self.sequence_window]
        
        if self.load_images:
            self.timestamps_rgb = self.timestamps_rgb[:self.sequence_window]
            self.left_images = self.left_images[:self.sequence_window]
            
        self.dataset_length = self.sequence_window


    def events_to_representation(self, x: np.ndarray, y: np.ndarray, p: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """
        Convert raw event data to the specified event representation.
        
        This method normalizes the temporal dimension and converts numpy arrays
        to PyTorch tensors before applying the event representation conversion.
        
        Args:
            x: Event x-coordinates
            y: Event y-coordinates  
            p: Event polarities
            t: Event timestamps
            
        Returns:
            torch.Tensor: Converted event representation
        """
        # Normalize timestamps to [0, 1] range
        t_normalized = (t - t[0]).astype("float32")
        if len(t_normalized) > 1:
            t_normalized = t_normalized / t_normalized[-1]
        
        # Convert to float32 for consistency
        x = x.astype("float32")
        y = y.astype("float32")
        pol = p.astype("float32")
        
        # Convert to PyTorch tensors and apply representation
        result = self.event_representation.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
            torch.from_numpy(t_normalized)
        )
        
        # Ensure we return a valid tensor
        if result is None:
            # Create empty tensor with proper shape if conversion fails
            # Fallback to standard dimensions if nr_of_channels not available
            channels = getattr(self.event_representation, 'nr_of_channels', 3)
            return torch.zeros((channels, self.height, self.width), dtype=torch.float32)
        
        return result

    def get_disparity_map(self, filepath: str) -> torch.Tensor:
        """
        Load disparity or depth map from file.
        
        Args:
            filepath: Path to the disparity/depth file
            
        Returns:
            torch.Tensor: Loaded disparity or depth map
        """
        assert os.path.isfile(filepath), f"File not found: {filepath}"
        
        if not self.self_supervised:
            # Standard mode: load 16-bit disparity map and convert to float
            disp_16bit = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH)
            assert disp_16bit is not None, f"Failed to load disparity map: {filepath}"
            return torch.tensor(disp_16bit.astype("float32") / 256.0)
        else:
            # Self-supervised mode: load depth map from NPZ file
            depth_data = np.load(filepath)
            depthmap = torch.tensor(depth_data['depth'].astype("float32"))
            
            # Clean invalid values
            depthmap[torch.isnan(depthmap)] = 0
            depthmap[torch.isinf(depthmap)] = 0
            return depthmap

    @staticmethod
    def close_callback(h5f_dict: dict) -> None:
        """
        Callback function to properly close HDF5 files.
        
        This is used with weakref.finalize to ensure HDF5 files are closed
        when the dataset object is garbage collected.
        
        Args:
            h5f_dict: Dictionary of HDF5 file handles to close
        """
        for key in h5f_dict:
            h5f_dict[key].close()

    def rectify_events(self, x: np.ndarray, y: np.ndarray, stereo: str) -> np.ndarray:
        """
        Apply rectification to event coordinates.
        
        This corrects for camera distortion using precomputed rectification maps.
        
        Args:
            x: Event x-coordinates
            y: Event y-coordinates
            stereo: Camera identifier ("left" or "right")
            
        Returns:
            np.ndarray: Rectified coordinates as [N, 2] array
        """
        assert stereo in ["left", "right"], f"Invalid stereo camera: {stereo}"
        
        # Get rectification map for the specified camera
        rectify_map = self.rectify_event_maps[stereo]
        assert rectify_map.shape == (self.height, self.width, 2), (
            f"Invalid rectify map shape: {rectify_map.shape}, expected ({self.height}, {self.width}, 2)"
        )
        
        # Validate coordinate bounds
        assert x.max() < self.width, f"X coordinate {x.max()} exceeds width {self.width}"
        assert y.max() < self.height, f"Y coordinate {y.max()} exceeds height {self.height}"
        
        # Apply rectification using the precomputed map
        return rectify_map[y, x]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.dataset_length
    
    def disparity2depth(self, disparity: torch.Tensor) -> torch.Tensor:
        """
        Convert disparity map to depth map using camera parameters.
        
        Uses the standard stereo vision formula: depth = (focal * baseline) / disparity
        
        Args:
            disparity: Disparity map tensor
            
        Returns:
            torch.Tensor: Depth map tensor
        """
        # Extract focal length and baseline from Q matrix
        focal = abs(self.q_event[2, 3])
        baseline = abs(1 / self.q_event[3, 2])
        
        # Initialize depth map
        depth = torch.clone(disparity)
        
        # Apply depth conversion only to valid disparity values
        valid_mask = disparity > 0
        depth[valid_mask] = (focal * baseline) / disparity[valid_mask]
        
        return depth

    def __getitem__(self, index: int) -> dict:
        """
        Get a sequence of data samples.
        
        Returns a temporal sequence of synchronized depth maps, event representations,
        and optionally RGB images for training or evaluation.
        
        Args:
            index: Sequence index in the dataset
            
        Returns:
            dict: Dictionary containing:
                - depth: Depth maps tensor [T, C, H, W]
                - depth_aligned_events: Event representations aligned to depth [T, C, H, W]
                - rgb_aligned_events: Event representations aligned to RGB [T, C, H, W] (if load_images=True)
                - rgb: RGB images tensor [T, C, H, W] (if load_images=True)
        """
        # Calculate sequence bounds
        start_index = index * self.sequence_step
        end_index = start_index + self.sequence_window

        to_return_list = []

        # Process each frame in the sequence
        for _index in range(start_index, end_index):
            to_return = {}

            # Load and process depth/disparity map
            disparity_path = os.path.join(self.base_disparity_path, self.disparities[_index])
            disparity = self.get_disparity_map(disparity_path)
            
            # Convert disparity to depth if needed (for supervised training)
            depth = self.disparity2depth(disparity) if not self.self_supervised else disparity                
            to_return["depth"] = depth

            # Setup event data sources based on available data
            if self.load_images:
                event_sources = [
                    ("depth_aligned_events", self.disparity_aligned_event_windows),
                    ("rgb_aligned_events", self.rgb_aligned_event_windows),
                ]
            else:
                event_sources = [
                    ("depth_aligned_events", self.disparity_aligned_event_windows)
                ]

            # Process event data for each source
            for key, timestamp_windows in event_sources:
                # Extract events within the time window
                event_data = self.event_slicer["left"].get_events(
                    *timestamp_windows[_index]
                )

                # Get event coordinates and properties
                p = event_data["p"]
                t = event_data["t"]
                x = event_data["x"]
                y = event_data["y"]

                # Apply rectification to correct for distortion
                xy_rect = self.rectify_events(x, y, "left")
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                # Convert to event representation
                event_representation = self.events_to_representation(x_rect, y_rect, p, t)
                to_return[key] = event_representation            

            # Load RGB image if requested
            if self.load_images:
                image_path = os.path.join(self.base_left_images_path, self.left_images[_index])
                left_image = cv2.imread(image_path)
                
                # Validate image loading
                if left_image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                # Convert BGR to RGB
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                
                # Handle grayscale images
                if len(left_image.shape) == 2:
                    left_image = left_image[..., None]
                
                # Normalize and convert to tensor format (CHW)
                left_image = left_image.transpose(2, 0, 1) / 255.0
                left_image = left_image.astype("float32")
                to_return["rgb"] = torch.tensor(left_image)

            # Ensure all tensors have channel dimension (CHW format)
            for key in to_return:
                if len(to_return[key].shape) == 2:
                    to_return[key] = to_return[key].unsqueeze(0)

            to_return_list.append(to_return)

        # Stack all frames into temporal sequences (TCHW format)
        to_return = {}
        for key in to_return_list[0]:
            to_return[key] = torch.stack([frame_data[key] for frame_data in to_return_list], dim=0)

        # Apply augmentations if configured
        if self.augmentator is not None:
            # Only augment visual data (RGB, depth, events)
            augmentation_keys = ["rgb", "depth", "depth_aligned_events", "rgb_aligned_events"]
            augmentation_data = {
                key: to_return[key] 
                for key in augmentation_keys 
                if key in to_return
            }
            
            # Apply augmentations and update the return dictionary
            augmented_data = self.augmentator(augmentation_data)
            if augmented_data is not None:
                to_return.update(augmented_data)

        return to_return
