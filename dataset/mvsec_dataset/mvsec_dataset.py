"""
MVSEC Dataset Loader and Validator

This module provides comprehensive functionality for loading and validating the MVSEC
(Multi Vehicle Stereo Event Camera) dataset. It handles dataset structure validation,
sequence loading, and PyTorch Dataset creation for event-based depth estimation tasks.

Key Features:
- Automatic dataset structure validation
- Support for multiple data splits (train/validation/test)
- Flexible sequence loading with different slicing strategies
- Hybrid training mode combining supervised and self-supervised data
- Comprehensive data integrity checking
- Support for both voxel-based and raw event processing

MVSEC Dataset Organization:
mvsec_root/
├── sequence_name_1/
│   ├── depth/data/          # Ground truth depth maps
│   ├── events/voxels/       # Precomputed event voxel grids
│   ├── rgb/davis*/          # RGB camera frames
│   └── ../hdf5/data.hdf5    # Raw event data
├── sequence_name_2/
└── ...

Supported Slicing Types:
- SBT (Slice by Time):   Temporal sequence processing
- SBN (Slice by Number of events): Number of events sequence processing (future)

Usage Examples:
    ```python
    # Load training dataset
    datasets = load_datasets(
        mvsec_path="/data/mvsec",
        data_split="train", 
        slicing={"slicing_type": "sbt", "time_window_ms": 50},
        event_representation=VoxelGrid(5, 480, 640),
        augmentator=train_augmentator,
        use_voxels=True
    )
    
    # Combine into single dataset for training
    combined = ConcatDataset(list(datasets.values()))
    ```
"""

import os
from typing import Dict, List, Tuple, Type, Optional
from tqdm import tqdm
import h5py
import hdf5plugin
import cv2
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .sbt.mvsec_sequence import MVSECSequence as SBT_MVSECSequence
from ..events import EventRepresentation
from ..utils import Augmentator
from .constants import (
    MVSEC_HEIGHT,
    MVSEC_WIDTH, 
    MVSEC_TRAIN,
    MVSEC_TEST,
    MVSEC_VALIDATION,
    MVSEC_ALL_DATA_FOLDERS,
)


def load_datasets(
    mvsec_path: str,
    data_split: str,
    slicing: Dict,
    event_representation: EventRepresentation,
    augmentator: Optional[Augmentator],
    load_images: bool = False,
    use_voxels: bool = True,
    overfit: bool = False,
    sequence_window: int = 1,
    sequence_step: int = 1,
    self_supervised: bool = False,
    hybrid: bool = False,
    postfix: str = "",
) -> Dict[str, Dataset]:
    """
    Load MVSEC dataset with specified configuration and return PyTorch datasets.
    
    This function serves as the main entry point for loading MVSEC sequences.
    It handles dataset validation, sequence creation, and supports various
    training modes including supervised, self-supervised, and hybrid approaches.
    
    Args:
        mvsec_path: Root path to the MVSEC dataset directory
        data_split: Dataset split to load ("train", "validation", "test")
        slicing: Dictionary specifying slicing strategy and parameters
            Example: {"slicing_type": "sbt", "time_window_ms": 50}
        event_representation: Event processing method (VoxelGrid, Histogram, etc.)
        augmentator: Data augmentation handler (None to disable)
        load_images: Whether to load RGB images alongside event data (default: False)
        use_voxels: Use precomputed voxels vs raw events (default: True)
        overfit: Enable overfitting mode for debugging (default: False)
        sequence_window: Number of frames per temporal sequence (default: 1)
        sequence_step: Step size between sequence windows (default: 1)  
        self_supervised: Enable self-supervised learning mode (default: False)
        hybrid: Combine supervised and self-supervised data (default: False)
        postfix: RGB directory postfix for self-supervised mode (default: "")
        
    Returns:
        Dictionary mapping sequence names to PyTorch Dataset objects
        
    Raises:
        AssertionError: If data_split is not valid
        Exception: If slicing_type is unsupported or dataset validation fails
        
    Example:
        ```python
        # Load training sequences with voxel grids
        datasets = load_datasets(
            mvsec_path="/data/mvsec",
            data_split="train",
            slicing={"slicing_type": "sbt", "time_window_ms": 50},
            event_representation=VoxelGrid(5, 480, 640),
            augmentator=augmentation_handler,
            load_images=True,
            use_voxels=True
        )
        
        # Access individual sequences
        outdoor_day1 = datasets["outdoor_day1"]
        ```
        
    Note:
        - Validates entire dataset structure before loading any sequences
        - Supports automatic fallback from raw events to voxel grids
        - Hybrid mode creates combined datasets with both supervised and self-supervised samples
    """
    # Validate input parameters
    assert data_split in ["train", "validation", "test"], (
        f"Invalid data_split '{data_split}'. Must be one of: train, validation, test"
    )

    print(f"Loading MVSEC dataset from: {mvsec_path}")
    print(f"Data split: {data_split}, Load images: {load_images}, Use voxels: {use_voxels}")

    # Determine sequences to load based on data split
    if data_split == "train":
        data_folders = MVSEC_TRAIN
    elif data_split == "validation":
        data_folders = MVSEC_VALIDATION
    elif data_split == "test":
        data_folders = MVSEC_TEST
    else:
        raise Exception(f"Unrecognized data_split: {data_split}")

    # Validate and configure slicing strategy
    slicing_type = slicing["slicing_type"]
    assert slicing_type in ["sbt", "sbn"], (
        f"Unsupported slicing_type '{slicing_type}'. Must be 'sbt' or 'sbn'"
    )

    if slicing_type == "sbt":
        sequence_class = SBT_MVSECSequence
        sequence_parameters = {k: v for k, v in slicing.items() if k != "slicing_type"}
    elif slicing_type == "sbn":
        raise Exception("SBN (Slice by Number of events) not implemented yet!")
    else:
        raise Exception(f"Unrecognized slicing type: {slicing_type}")

    # Load and validate dataset sequences
    datasets = check_dataset(
        mvsec_path=mvsec_path,
        data_folders=data_folders,
        images=load_images,
        sequence_class=sequence_class,
        sequence_parameters=sequence_parameters,
        event_representation=event_representation,
        use_voxels=use_voxels,
        augmentator=augmentator,
        overfit=overfit,
        sequence_window=sequence_window,
        sequence_step=sequence_step,
        data_split=data_split,
        self_supervised=self_supervised,
        hybrid=hybrid,
        postfix=postfix
    )

    print(f"Successfully loaded {len(datasets)} sequences: {list(datasets.keys())}")
    return datasets


def check_dataset(
    mvsec_path: str,
    data_folders: Dict[str, int],
    images: bool,
    sequence_class: Type[Dataset],
    sequence_parameters: Dict,
    event_representation: EventRepresentation,
    use_voxels: bool,
    augmentator: Optional[Augmentator],
    overfit: bool,
    sequence_window: int,
    sequence_step: int,
    data_split: str,
    self_supervised: bool = False,
    hybrid: bool = False,
    postfix: str = "",
) -> Dict[str, Dataset]:
    """
    Validate MVSEC dataset structure and create PyTorch Dataset objects.
    
    This function performs comprehensive validation of the MVSEC dataset structure,
    ensuring all required files and directories are present, then creates
    individual sequence datasets for each specified data folder.
    
    Args:
        mvsec_path: Root path to MVSEC dataset
        data_folders: Dictionary mapping folder names to expected frame counts
        images: Whether RGB images should be loaded and validated
        sequence_class: PyTorch Dataset class to instantiate for each sequence
        sequence_parameters: Additional parameters for sequence class constructor
        event_representation: Event processing representation method
        use_voxels: Whether to use precomputed voxel grids
        augmentator: Data augmentation handler (None to disable)
        overfit: Enable overfitting mode for debugging
        sequence_window: Number of frames per temporal sequence
        sequence_step: Step size between sequence windows
        data_split: Dataset split identifier ("train", "validation", "test")
        self_supervised: Enable self-supervised learning mode
        hybrid: Create hybrid datasets combining supervised and self-supervised
        postfix: RGB directory postfix for self-supervised mode
        
    Returns:
        Dictionary mapping sequence names to PyTorch Dataset instances
        
    Raises:
        Exception: If dataset validation fails or required files are missing
        
    Validation Process:
    1. Verify dataset root directory exists
    2. Check all sequence folders are present  
    3. Validate each sequence's internal structure
    4. Verify file counts match expected values
    5. Create Dataset objects for valid sequences
    
    Note:
        In hybrid mode, combines supervised and self-supervised datasets
        using ConcatDataset for enhanced training diversity.
    """
    print(f"Validating MVSEC dataset structure in: {mvsec_path}")

    # Validate dataset root directory
    if not os.path.isdir(mvsec_path):
        raise Exception(f"Dataset path is not a directory: {mvsec_path}")

    # Check that all required sequence folders are present
    check_folder_content(
        folder_path=mvsec_path,
        subfolders=list(data_folders.keys()),
        files=[],
        optional_subfolders=[
            unused_folder
            for unused_folder in MVSEC_ALL_DATA_FOLDERS
            if unused_folder not in data_folders.keys()
        ],
    )

    # Validate and create dataset for each sequence
    datasets = {}
    for folder_name in tqdm(data_folders.keys(), desc="Processing sequences"):
        expected_frames = data_folders[folder_name]
        
        # Validate sequence folder structure
        check_dataset_subfolder(
            mvsec_path=mvsec_path,
            folder_name=folder_name,
            folder_size=expected_frames,
            images=images,
            voxels=use_voxels,
        )

        # Create primary sequence dataset
        sequence_path = os.path.join(mvsec_path, folder_name)
        
        # Prepare arguments for sequence constructor
        sequence_args = {
            'sequence_path': sequence_path,
            'event_representation': event_representation,
            'augmentator': augmentator,
            'load_images': images,
            'use_voxels': use_voxels,
            'overfit': overfit,
            'sequence_window': sequence_window,
            'sequence_step': sequence_step,
            'split': data_split,
            'self_supervised': self_supervised,
            'postfix': postfix,
            **sequence_parameters
        }
        
        datasets[folder_name] = sequence_class(**sequence_args)  # type: ignore

        # Create hybrid dataset if requested
        if hybrid and self_supervised:
            print(f"Creating hybrid dataset for {folder_name}")
            
            # Create supervised counterpart with modified parameters
            supervised_args = sequence_args.copy()
            supervised_args.update({
                'self_supervised': False,  # Supervised mode
                'postfix': "",  # No postfix for supervised
            })
            
            supervised_dataset = sequence_class(**supervised_args)  # type: ignore

            # Combine datasets for hybrid training
            datasets[folder_name] = ConcatDataset([
                datasets[folder_name],    # Self-supervised
                supervised_dataset        # Supervised
            ])

    print(f"Dataset validation completed. Loaded {len(datasets)} sequences.")
    return datasets

def split_path(path: str) -> List[str]:
    """
    Split a file path into its individual components.
    
    This utility function recursively splits a path into its directory
    and filename components, useful for path analysis and validation.
    
    Args:
        path: File or directory path to split
        
    Returns:
        List of path components from root to leaf
        
    Example:
        >>> split_path("/home/user/data/sequence1")
        ["home", "user", "data", "sequence1"]
        
        >>> split_path("relative/path/file.txt") 
        ["relative", "path", "file.txt"]
    """
    components = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            components.append(folder)
        else:
            break
    return components[::-1]


def check_folder_content(
    folder_path: str, 
    subfolders: List[str], 
    files: List[str], 
    optional_subfolders: Optional[List[str]] = None, 
    optional_files: Optional[List[str]] = None
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """
    Validate folder contents and check for required and optional items.
    
    This function performs comprehensive folder structure validation, ensuring
    all required subfolders and files are present while tracking optional items.
    It's essential for dataset integrity validation.
    
    Args:
        folder_path: Absolute path to folder to validate
        subfolders: List of required subdirectories that must exist
        files: List of required files that must exist
        optional_subfolders: List of subdirectories that may exist (default: None)
        optional_files: List of files that may exist (default: None)
        
    Returns:
        Tuple of (optional_subfolders_found, optional_files_found) dictionaries
        mapping item names to boolean presence indicators
        
    Raises:
        Exception: If any required subfolder or file is missing
        
    Example:
        ```python
        # Check dataset sequence structure
        check_folder_content(
            "/data/mvsec/outdoor_day1",
            subfolders=["depth", "events", "rgb"],
            files=[],
            optional_subfolders=["calibration"]
        )
        ```
        
    Note:
        Prints informational messages when optional items are found.
        This helps with debugging dataset structure variations.
    """
    # Initialize optional lists if not provided
    if optional_subfolders is None:
        optional_subfolders = []
    if optional_files is None:
        optional_files = []

    # Check all required subfolders exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            raise Exception(f"Missing required subfolder: {subfolder} in {folder_path}")

    # Check all required files exist
    for file in files:
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            raise Exception(f"Missing required file: {file} in {folder_path}")

    # Check optional items and track their presence
    optional_subfolders_found = {
        subfolder: os.path.isdir(os.path.join(folder_path, subfolder))
        for subfolder in optional_subfolders
    }
    
    optional_files_found = {
        file: os.path.isfile(os.path.join(folder_path, file)) 
        for file in optional_files
    }

    # Provide informational feedback about optional items found
    for subfolder_name, is_present in optional_subfolders_found.items():
        if is_present:
            print(f"Info: Found optional subfolder '{subfolder_name}' in {folder_path}")

    for file_name, is_present in optional_files_found.items():
        if is_present:
            print(f"Info: Found optional file '{file_name}' in {folder_path}")

    return optional_subfolders_found, optional_files_found


def check_dataset_subfolder(
    mvsec_path: str, 
    folder_name: str, 
    folder_size: int, 
    images: bool, 
    voxels: bool
) -> None:
    """
    Validate the internal structure of a single MVSEC sequence folder.
    
    This function performs detailed validation of a specific MVSEC sequence,
    ensuring all required data files are present with correct naming patterns
    and expected quantities. It validates depth maps, event data, and optionally
    RGB images based on the specified configuration.
    
    Expected MVSEC Sequence Structure:
    sequence_folder/
    ├── depth/
    │   └── data/
    │       ├── depth_0000000000.npy (x folder_size)
    │       └── timestamps.txt
    ├── events/
    │   └── voxels/
    │       ├── event_tensor_0000000000.npy (x folder_size)
    │       └── timestamps.txt
    └── rgb/ (if images=True)
        ├── davis/ or davis_left_sync/
        │   ├── frame_0000000000.png (x folder_size)
        │   └── timestamps.txt
        
    Args:
        mvsec_path: Root path to MVSEC dataset
        folder_name: Name of sequence folder to validate
        folder_size: Expected number of frames in sequence
        images: Whether RGB images should be present and validated
        voxels: Whether voxel data should be validated (always True for MVSEC)
        
    Raises:
        Exception: If any required files or folders are missing
        
    Note:
        - Depth files use pattern: depth_{i:010}.npy
        - Voxel files use pattern: event_tensor_{i:010}.npy  
        - RGB files use pattern: frame_{i:010}.png
        - All timestamps are stored in timestamps.txt files
        - RGB folder name varies: 'davis' for train/val, 'davis_left_sync' for test
    """
    sequence_path = os.path.join(mvsec_path, folder_name)
    
    print(f"Validating sequence structure: {folder_name}")
    
    # Validate top-level sequence structure
    check_folder_content(
        folder_path=sequence_path,
        subfolders=["depth", "events", "rgb"],
        files=[],
        optional_subfolders=[],
    )

    # Validate depth folder structure
    depth_path = os.path.join(sequence_path, "depth")
    check_folder_content(
        folder_path=depth_path,
        subfolders=["data"],
        files=[],
        optional_subfolders=["frames"],  # Sometimes present for visualization
    )
    
    # Validate depth data files
    depth_data_path = os.path.join(depth_path, "data")
    expected_depth_files = [f"depth_{i:010}.npy" for i in range(folder_size)]
    expected_depth_files.append("timestamps.txt")
    
    check_folder_content(
        folder_path=depth_data_path,
        subfolders=[],
        files=expected_depth_files,
    )

    # Validate events folder structure
    events_path = os.path.join(sequence_path, "events") 
    check_folder_content(
        folder_path=events_path,
        subfolders=["voxels"],
        files=[],
    )
    
    # Validate voxel data files
    voxels_path = os.path.join(events_path, "voxels")
    expected_voxel_files = [f"event_tensor_{i:010}.npy" for i in range(folder_size)]
    expected_voxel_files.append("timestamps.txt")
    
    check_folder_content(
        folder_path=voxels_path,
        subfolders=[],
        files=expected_voxel_files,
    )

    # Validate RGB images if required
    if images:
        rgb_path = os.path.join(sequence_path, "rgb")
        
        # Check for available RGB folders (varies by split)
        optional_rgb_folders, _ = check_folder_content(
            folder_path=rgb_path,
            subfolders=[],
            files=[],
            optional_subfolders=["davis", "davis_left_sync"],
        )

        # Validate RGB data for available folders
        expected_rgb_files = [f"frame_{i:010}.png" for i in range(folder_size)]
        expected_rgb_files.append("timestamps.txt")
        
        if optional_rgb_folders.get("davis", False):
            davis_path = os.path.join(rgb_path, "davis")
            check_folder_content(
                folder_path=davis_path,
                subfolders=[],
                files=expected_rgb_files,
            )

        if optional_rgb_folders.get("davis_left_sync", False):
            davis_sync_path = os.path.join(rgb_path, "davis_left_sync")
            check_folder_content(
                folder_path=davis_sync_path,
                subfolders=[],
                files=expected_rgb_files,
            )
    
    print(f"✓ Sequence {folder_name} validation completed successfully")
