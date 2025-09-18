"""
DSEC Dataset Loader and Validator

This module provides comprehensive functionality for loading and validating 
the DSEC (Dynamic Stereo Event Camera) dataset. It handles:

- Dataset integrity validation and folder structure verification
- Multi-sequence dataset loading with customizable parameters
- Support for different slicing strategies (SBT - Slice by Time)
- Hybrid training mode combining supervised and self-supervised data
- Flexible configuration for training, validation, and testing splits

Key Components:
- load_datasets(): Main interface for creating PyTorch datasets from DSEC sequences
- check_dataset(): Validates dataset integrity and creates sequence objects
- check_folder_content(): Comprehensive folder/file validation utility
- check_dataset_subfolder(): DSEC-specific structure validation

The module supports both standard supervised training (using disparity ground truth)
and self-supervised learning modes (using pseudo-labels or depth predictions).

Example Usage:
    datasets = load_datasets(
        dsec_path="/path/to/dsec",
        data_split="train", 
        slicing={"slicing_type": "sbt", "time_window_ms": 50},
        event_representation=VoxelGrid(),
        augmentator=MyAugmentator(),
        load_images=True
    )
"""

import os
from typing import Dict, List, Tuple, Type, Optional, Any
from pathlib import Path

from tqdm import tqdm
import h5py
import hdf5plugin
import cv2
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .sbt.dsec_sequence import DsecSequence as SBT_DsecSequence
from ..events import EventRepresentation
from ..utils import Augmentator
from .constants import DSEC_TRAIN, DSEC_VALIDATION, DSEC_ALL_DATA_FOLDERS, DSEC_HEIGHT, DSEC_WIDTH

def load_datasets(  
    dsec_path: str,
    data_split: str,
    slicing: Dict[str, Any],
    event_representation: EventRepresentation,
    augmentator: Optional[Augmentator],
    load_images: bool = False,
    overfit: bool = False,
    sequence_window: int = 1,
    sequence_step: int = 1,
    self_supervised: bool = False,
    hybrid: bool = False,    
    postfix: str = "",
) -> Dict[str, Dataset]:
    """
    Load DSEC datasets for training or validation.
    
    This is the main interface for creating PyTorch datasets from DSEC sequences.
    It handles dataset validation, sequence creation, and supports various training modes.
    
    Args:
        dsec_path: Root path to the DSEC dataset directory
        data_split: Dataset split ("train" or "validation")
        slicing: Dictionary containing slicing configuration:
            - "slicing_type": Type of slicing strategy ("sbt" or "sbn")
            - Additional parameters specific to the slicing strategy
        event_representation: Event representation method (e.g., VoxelGrid, Histogram)
        augmentator: Data augmentation pipeline (optional)
        load_images: Whether to load RGB images alongside events
        overfit: If True, limit dataset size for overfitting experiments
        sequence_window: Number of consecutive frames per sequence sample
        sequence_step: Step size between consecutive sequence samples
        self_supervised: Enable self-supervised learning mode
        hybrid: Enable hybrid training (combines supervised + self-supervised)
        postfix: Filename postfix for self-supervised mode file selection
        
    Returns:
        Dict[str, Dataset]: Dictionary mapping sequence names to PyTorch datasets
        
    Raises:
        Exception: If data_split is not "train" or "validation"
        Exception: If slicing_type is not supported
        Exception: If dataset validation fails
        
    Example:
        datasets = load_datasets(
            dsec_path="/path/to/dsec",
            data_split="train",
            slicing={"slicing_type": "sbt", "time_window_ms": 50},
            event_representation=VoxelGrid(channels=5),
            augmentator=RandomCrop(size=256),
            load_images=True,
            sequence_window=3
        )
    """
    # Validate input parameters
    assert data_split in ["train", "validation"], (
        f"data_split must be 'train' or 'validation', got '{data_split}'"
    )

    print(f"Loading DSEC datasets from: {dsec_path}")

    # Determine which sequence folders to use based on split
    if data_split == "train":
        data_folders = DSEC_TRAIN
    elif data_split == "validation":
        data_folders = DSEC_VALIDATION
    else:
        raise Exception(f"data_split must be either 'train' or 'validation' but got '{data_split}'")

    # Extract and validate slicing configuration
    slicing_type = slicing["slicing_type"]
    assert slicing_type in ["sbt", "sbn"], (
        f"Unsupported slicing_type: {slicing_type}. Must be 'sbt' or 'sbn'"
    )

    # Select appropriate sequence class and parameters
    if slicing_type == "sbt":
        sequence_class = SBT_DsecSequence
        # Extract parameters specific to the sequence class (excluding slicing_type)
        sequence_parameters = {k: v for k, v in slicing.items() if k != "slicing_type"}
    elif slicing_type == "sbn":
        raise Exception("SBN (Slice by Number) slicing not implemented yet!")
    else:
        raise Exception(f"Unrecognized slicing type: {slicing_type}")

    # Validate dataset structure and create sequence objects
    print("Validating dataset structure and creating sequences...")
    datasets = check_dataset(
        dsec_path=dsec_path,
        data_folders=data_folders,
        images=load_images,
        sequence_class=sequence_class,
        sequence_parameters=sequence_parameters,
        event_representation=event_representation,
        augmentator=augmentator,
        overfit=overfit,
        sequence_window=sequence_window,
        sequence_step=sequence_step,
        data_split=data_split,
        self_supervised=self_supervised,
        hybrid=hybrid,
        postfix=postfix
    )

    print(f"Successfully loaded {len(datasets)} sequence datasets")
    return datasets


def check_dataset(
    dsec_path: str,
    data_folders: Dict[str, int],
    images: bool,
    sequence_class: Type[Dataset],
    sequence_parameters: Dict[str, Any],
    event_representation: EventRepresentation,
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
    Validate dataset structure and create sequence objects.
    
    This function performs comprehensive validation of the DSEC dataset structure
    and creates PyTorch Dataset objects for each sequence folder.
    
    Args:
        dsec_path: Root path to the DSEC dataset
        data_folders: Dictionary mapping folder names to expected sizes
        images: Whether RGB images should be loaded
        sequence_class: PyTorch Dataset class to instantiate
        sequence_parameters: Additional parameters for sequence class initialization
        event_representation: Event representation method
        augmentator: Data augmentation pipeline (optional)
        overfit: Enable overfitting mode with limited data
        sequence_window: Number of frames per sequence
        sequence_step: Step size between sequences
        data_split: Dataset split identifier
        self_supervised: Enable self-supervised learning mode
        hybrid: Enable hybrid training mode
        postfix: File postfix for self-supervised mode
        
    Returns:
        Dict[str, Dataset]: Dictionary mapping sequence names to datasets
        
    Raises:
        Exception: If dataset path is invalid or structure validation fails
    """
    print(f"Validating dataset structure in: {dsec_path}")

    # Validate that the root path exists and is a directory
    if not os.path.isdir(dsec_path):
        raise Exception(f"Dataset path is not a directory: {dsec_path}")

    # Validate that all required data folders are present
    train_path = os.path.join(dsec_path, "train")
    check_folder_content(
        folder_path=train_path,
        subfolders=list(data_folders.keys()),
        files=[],
        optional_subfolders=[
            unused_folder for unused_folder in DSEC_ALL_DATA_FOLDERS
            if unused_folder not in data_folders.keys()
        ]
    )

    # Create dataset objects for each sequence folder
    datasets = {}
    print("Creating sequence datasets...")
    
    for folder in tqdm(data_folders.keys(), desc="Processing sequences"):
        # Validate individual sequence folder structure
        check_dataset_subfolder(
            dsec_path=dsec_path, 
            folder_name=folder, 
            folder_size=data_folders[folder],
            images=images
        )

        # Create primary dataset (self-supervised or supervised)
        print(f"Creating sequence dataset for: {folder}")
        # Note: Type checker cannot resolve dynamic class parameters
        datasets[folder] = sequence_class(
            sequence_path=os.path.join(dsec_path, "train", folder), # type: ignore
            event_representation=event_representation, # type: ignore
            augmentator=augmentator, # type: ignore
            load_images=images, # type: ignore
            overfit=overfit, # type: ignore
            sequence_window=sequence_window, # type: ignore
            sequence_step=sequence_step, # type: ignore
            split=data_split, # type: ignore
            self_supervised=self_supervised, # type: ignore
            postfix=postfix, # type: ignore
            **sequence_parameters,
        )

        # Create hybrid dataset if requested (combines supervised + self-supervised)
        if hybrid and self_supervised:
            print(f"Creating supervised counterpart for hybrid training: {folder}")
            # Note: Type checker cannot resolve dynamic class parameters
            supervised_dataset = sequence_class(
                sequence_path=os.path.join(dsec_path, "train", folder), # type: ignore
                event_representation=event_representation, # type: ignore
                augmentator=augmentator, # type: ignore
                load_images=images, # type: ignore
                overfit=overfit, # type: ignore
                sequence_window=sequence_window, # type: ignore
                sequence_step=sequence_step, # type: ignore
                split=data_split, # type: ignore
                self_supervised=self_supervised, # type: ignore
                postfix=postfix, # type: ignore
                **sequence_parameters,
            )

            # Combine both datasets using ConcatDataset
            datasets[folder] = ConcatDataset([datasets[folder], supervised_dataset])
            print(f"Created hybrid dataset for: {folder}")

    print(f"Successfully created {len(datasets)} sequence datasets")
    return datasets

def split_path(path: str) -> List[str]:
    """
    Split a file system path into its individual components.
    
    This utility function recursively splits a path using os.path.split() 
    to extract all directory and file components.
    
    Args:
        path: The file system path to split
        
    Returns:
        List[str]: List of path components from root to leaf
        
    Example:
        split_path("/home/user/documents/file.txt")
        # Returns: ["home", "user", "documents", "file.txt"]
        
        split_path("./data/sequences")
        # Returns: ["data", "sequences"]
    """
    components = []
    path = os.path.normpath(path)  # Normalize path separators
    
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            components.append(folder)
        elif path:
            # Handle absolute paths (root component)
            components.append(path)
            break
        else:
            break
    
    # Reverse to get components from root to leaf
    return components[::-1]

def check_folder_content(
    folder_path: str, 
    subfolders: List[str], 
    files: List[str], 
    optional_subfolders: Optional[List[str]] = None, 
    optional_files: Optional[List[str]] = None
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """
    Validate folder structure and report missing/optional components.
    
    This function performs comprehensive validation of a folder's contents,
    checking for required files and subfolders while reporting on optional
    components. It's the core validation utility for DSEC dataset structure.
    
    Args:
        folder_path: Absolute path to the folder to validate
        subfolders: List of required subfolder names that must exist
        files: List of required file names that must exist
        optional_subfolders: List of subfolder names that may exist (optional)
        optional_files: List of file names that may exist (optional)
        
    Returns:
        Tuple[Dict[str, bool], Dict[str, bool]]: Two dictionaries:
            - First: Mapping of optional subfolder names to existence status
            - Second: Mapping of optional file names to existence status
            
    Raises:
        Exception: If any required subfolder or file is missing
        
    Example:
        found_dirs, found_files = check_folder_content(
            "/path/to/sequence",
            subfolders=["events", "disparity"],
            files=["timestamps.txt"],
            optional_subfolders=["images"],
            optional_files=["metadata.yaml"]
        )
    """
    # Initialize optional lists if not provided
    if optional_subfolders is None:
        optional_subfolders = []
    if optional_files is None:
        optional_files = []

    # Validate that the folder exists
    if not os.path.isdir(folder_path):
        raise Exception(f"Folder does not exist: {folder_path}")

    # Check for required subfolders
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            raise Exception(f"Missing required subfolder: '{subfolder}' in {folder_path}")

    # Check for required files
    for file in files:
        file_path = os.path.join(folder_path, file)
        if not os.path.isfile(file_path):
            raise Exception(f"Missing required file: '{file}' in {folder_path}")

    # Check optional components and track their presence
    optional_subfolders_found = {
        subfolder: os.path.isdir(os.path.join(folder_path, subfolder))
        for subfolder in optional_subfolders
    }
    
    optional_files_found = {
        file: os.path.isfile(os.path.join(folder_path, file))
        for file in optional_files
    }

    # Report found optional components
    for subfolder, found in optional_subfolders_found.items():
        if found:
            print(f"Info: Found optional subfolder '{subfolder}' in {folder_path}")

    for file, found in optional_files_found.items():
        if found:
            print(f"Info: Found optional file '{file}' in {folder_path}")

    return optional_subfolders_found, optional_files_found

def check_dataset_subfolder(dsec_path: str, folder_name: str, folder_size: int, images: bool) -> None:
    """
    Validate the structure of a specific DSEC sequence folder.
    
    This function performs detailed validation of a single DSEC sequence,
    checking the presence and structure of all required components according
    to the DSEC dataset specification.
    
    DSEC Sequence Structure:
    - calibration/: Camera calibration files
    - disparity/: Ground truth depth/disparity maps
    - events/: Event data from stereo cameras
    - images/: RGB images (optional)
    
    Args:
        dsec_path: Root path to the DSEC dataset
        folder_name: Name of the specific sequence folder to validate
        folder_size: Expected number of frames in the sequence
        images: Whether RGB images should be present
        
    Raises:
        Exception: If any required component is missing or malformed
        Exception: If images are requested but not available
        
    Note:
        The folder_size parameter determines how many frame files should exist.
        For disparity maps: files are named with even indices (0, 2, 4, ...)
        For images: files are numbered sequentially (0, 1, 2, ...)
    """
    sequence_path = os.path.join(dsec_path, "train", folder_name)
    print(f"Validating sequence structure: {folder_name}")
    
    # (1) Validate main sequence folder structure
    optional_subfolders, _ = check_folder_content(
        folder_path=sequence_path,
        subfolders=["calibration", "disparity", "events"],
        files=[],
        optional_subfolders=["images"]
    )
    
    # Check if images are required but missing
    if not optional_subfolders["images"] and images:
        raise Exception(
            f"RGB images requested but not found in sequence: {folder_name} "
            f"(path: {sequence_path})"
        )

    # (2) Validate calibration folder
    calibration_path = os.path.join(sequence_path, "calibration")
    check_folder_content(
        folder_path=calibration_path,
        subfolders=[],
        files=["cam_to_cam.yaml", "cam_to_lidar.yaml"]
    )

    # (3) Validate disparity folder structure
    disparity_path = os.path.join(sequence_path, "disparity")
    check_folder_content(
        folder_path=disparity_path,
        subfolders=["event", "image"],
        files=["timestamps.txt"],
        optional_subfolders=["raw"]
    )
    
    # Validate disparity maps (event-based estimations)
    disparity_event_path = os.path.join(disparity_path, "event")
    expected_disparity_files = [f"{i*2:06d}.png" for i in range(folder_size)]
    check_folder_content(
        folder_path=disparity_event_path,
        subfolders=[],
        files=expected_disparity_files
    )
    
    # Validate disparity maps (image-based estimations)
    disparity_image_path = os.path.join(disparity_path, "image")
    check_folder_content(
        folder_path=disparity_image_path,
        subfolders=[],
        files=expected_disparity_files
    )
    
    # (4) Validate events folder structure
    events_path = os.path.join(sequence_path, "events")
    check_folder_content(
        folder_path=events_path,
        subfolders=["left", "right"],
        files=[]
    )
    
    # Validate left camera event data
    events_left_path = os.path.join(events_path, "left")
    check_folder_content(
        folder_path=events_left_path,
        subfolders=[],
        files=["events.h5", "rectify_map.h5"]
    )
    
    # Validate right camera event data
    events_right_path = os.path.join(events_path, "right")
    check_folder_content(
        folder_path=events_right_path,
        subfolders=[],
        files=["events.h5", "rectify_map.h5"]
    )

    # (5) Validate images folder if requested
    if images:
        images_path = os.path.join(sequence_path, "images")
        check_folder_content(
            folder_path=images_path,
            subfolders=["left", "right"],
            files=["timestamps.txt"]
        )
        
        # Validate left camera images
        images_left_path = os.path.join(images_path, "left")
        check_folder_content(
            folder_path=images_left_path,
            subfolders=["rectified"],
            files=["exposure_timestamps.txt"]
        )
        
        # Validate right camera images
        images_right_path = os.path.join(images_path, "right")
        check_folder_content(
            folder_path=images_right_path,
            subfolders=["rectified"],
            files=["exposure_timestamps.txt"]
        )

        # Validate rectified image files (left camera)
        # Note: Images use sequential numbering and there's one less than 2*folder_size
        expected_image_count = folder_size * 2 - 1
        expected_image_files = [f"{i:06d}.png" for i in range(expected_image_count)]
        
        images_left_rectified_path = os.path.join(images_left_path, "rectified")
        check_folder_content(
            folder_path=images_left_rectified_path,
            subfolders=[],
            files=expected_image_files
        )
        
        # Validate rectified image files (right camera)
        images_right_rectified_path = os.path.join(images_right_path, "rectified")
        check_folder_content(
            folder_path=images_right_rectified_path,
            subfolders=[],
            files=expected_image_files
        )
    
    print(f"✓ Sequence validation completed: {folder_name} ({folder_size} frames)")
    if images:
        print(f"  - RGB images: ✓ ({folder_size * 2 - 1} per camera)")
    print(f"  - Event data: ✓ (stereo)")
    print(f"  - Disparity maps: ✓ ({folder_size} maps)")
