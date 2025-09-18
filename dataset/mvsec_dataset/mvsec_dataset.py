import os
from tqdm import tqdm
import h5py
import hdf5plugin
import cv2
import numpy as np
from typing import Dict, List, Tuple, Type
from threading import Thread, Lock
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
    augmentator: Augmentator,
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
    This function load given a mvsec path a dataloader.
    This function can be used both for training, validation and test.

    :param mvsec_path: the path to the MVSEC dataset
    :param data_split: either "train" or "validation" or "test"
    :param slicing: the slicing to use
    :param event_representation: the event representation to use
    :param load_images: whether to load images or not
    :return:
    """
    # Assumptions
    assert data_split in ["train", "validation", "test"]

    print(f"Loading from: {mvsec_path}")

    # Let's update the folder list where to search
    if data_split == "train":
        data_folders = MVSEC_TRAIN
    elif data_split == "validation":
        data_folders = MVSEC_VALIDATION
    elif data_split == "test":
        data_folders = MVSEC_TEST
    else:
        raise Exception(f"Unrecognized train_validation: {data_split}")

    # assert (
    #     not load_images or data_split == "train"
    # ), "You can only load images from the training set!"

    slicing_type = slicing["slicing_type"]
    assert slicing_type in ["sbt", "sbn"]

    if slicing_type == "sbt":
        sequence_class = SBT_MVSECSequence
        sequence_parameters = {k: v for k, v in slicing.items() if k != "slicing_type"}
    elif slicing_type == "sbn":
        raise Exception("Not implemented yet!")
    else:
        raise Exception(f"Unrecognized slicing type! ({slicing_type})")

    # Let's Check The Dataset and while checking let's calculate the train and validation sequences
    datasets = check_dataset(
        mvsec_path,
        data_folders,
        load_images,
        sequence_class,
        sequence_parameters,
        event_representation,
        use_voxels,
        augmentator,
        overfit,
        sequence_window,
        sequence_step,
        data_split,
        self_supervised,
        hybrid,
        postfix
    )
    """
    # Let's create the Torch Dataset
    dataset = torch.utils.data.ConcatDataset(sequence)

    # Let's compute the Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    """
    return datasets


def check_dataset(
    mvsec_path: str,
    data_folders: Dict[str, int],
    images: bool,
    sequence_class: Type[Dataset],
    sequence_parameters: Dict,
    event_representation: EventRepresentation,
    use_voxels: bool,
    augmentator: Augmentator,
    overfit: bool,
    sequence_window: int,
    sequence_step: int,
    data_split: str,
    self_supervised: bool = False,
    hybrid: bool = False,
    postfix: str = "",
) -> Dict[str, Dataset]:
    print(f"Checking dataset in {mvsec_path}!")

    # (1) Let's check that mvsec_path is a directory
    if not os.path.isdir(mvsec_path):
        raise Exception(f"{mvsec_path} is not a directory!")

    # (3) Let's check that all the data folders are there
    check_folder_content(
        mvsec_path,
        subfolders=data_folders.keys(),
        files=[],
        optional_subfolders=[
            not_used_data_folder
            for not_used_data_folder in MVSEC_ALL_DATA_FOLDERS
            if not_used_data_folder not in data_folders.keys()
        ],
    )

    # (4) Let's check each data folder
    datasets = {}
    for folder in tqdm(data_folders.keys()):
        check_dataset_subfolder(
            mvsec_path=mvsec_path,
            folder_name=folder,
            folder_size=data_folders[folder],
            images=images,
            voxels=use_voxels,
        )

        # Crete a Sequence Object
        datasets[folder] = sequence_class(
            sequence_path=os.path.join(mvsec_path, folder),
            event_representation=event_representation,
            augmentator=augmentator,
            load_images=images,
            use_voxels=use_voxels,
            overfit=overfit,
            sequence_window=sequence_window,
            sequence_step=sequence_step,
            split=data_split,
            self_supervised=self_supervised,
            postfix=postfix,
            **sequence_parameters,
        )

        if hybrid and self_supervised:
            supervided_dataset = sequence_class(
                sequence_path=os.path.join(mvsec_path, folder),
                event_representation=event_representation,
                augmentator=augmentator,
                load_images=images,
                use_voxels=use_voxels,
                overfit=overfit,
                sequence_window=sequence_window,
                sequence_step=sequence_step,
                split=data_split,
                self_supervised=False,
                postfix="",
                **sequence_parameters,
            )

            datasets[folder] = ConcatDataset(
                [datasets[folder], supervided_dataset]
            )



    return datasets

def split_path(path):
    """
    A function that split a path into its components
    :param path: The path to split
    :return: A list of the components of the path
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
    folder_path, subfolders, files, optional_subfolders=None, optional_files=None
) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """
    A function that check the content of a folder and return exception if some folders/files are missing
    and print warnings if there are extras of them.
    :param folder_path: Simply an absolute path to a folder!
    :param subfolders: A list of subfolders to check.
        Example:
        ["subfolder_A", "subfolder_B", "subfolder_C", "subfolder_D", "subfolder_E" ]
    :param files: A list of files to check.
        Example:
        ["file_A", "file_B", "file_C", "file_D", "file_E" ]
    :param optional_subfolders: A list of subfolders that are not mandatory to be there but that are not printed as
        warning if it's the case.
        Example:
        ["optional_subfolder_A", "optional_subfolder_B", "optional_subfolder_C" ]
    :param optional_files: A list of files that are not mandatory to be there but that are not printed as
        warning if it's the case.
        Example:
        ["optional_file_A", "optional_file_B", "optional_file_C" ]
    :return: Nothing
    """

    if optional_subfolders is None:
        optional_subfolders = []
    if optional_files is None:
        optional_files = []

    for subfolder in subfolders:
        if not os.path.isdir(os.path.join(folder_path, subfolder)):
            raise Exception(f"Missing subfolder: {subfolder} in {folder_path}")

    for file in files:
        if not os.path.isfile(os.path.join(folder_path, file)):
            raise Exception(f"Missing file: {file} in {folder_path}")

    is_there_optional_subfolders = {
        subfolder: os.path.isdir(os.path.join(folder_path, subfolder))
        for subfolder in optional_subfolders
    }
    is_there_optional_files = {
        file: os.path.isfile(os.path.join(folder_path, file)) for file in optional_files
    }

    for key in is_there_optional_subfolders.keys():
        if is_there_optional_subfolders[key]:
            print(f"Info: {key} found optional subfolder!")

    for key in is_there_optional_files.keys():
        if is_there_optional_files[key]:
            print(f"Info: {key} found optional file!")

    return is_there_optional_subfolders, is_there_optional_files


def check_dataset_subfolder(
    mvsec_path: str, folder_name: str, folder_size: int, images: bool, voxels: bool
):
    # (1) Let's check that each data folder contains depth and events for sure and images
    check_folder_content(
        os.path.join(mvsec_path, folder_name),
        subfolders=["depth", "events", "rgb"],
        files=[],
        optional_subfolders=[],
    )

    # (3) Let's check the depth folder
    check_folder_content(
        os.path.join(mvsec_path, folder_name, "depth"),
        subfolders=["data"],
        files=[],
        optional_subfolders=["frames"],
    )
    check_folder_content(
        os.path.join(mvsec_path, folder_name, "depth", "data"),
        subfolders=[],
        files=[f"depth_{i:010}.npy" for i in range(folder_size)] + ["timestamps.txt"],
    )

    # (4) Let's check the events folder
    check_folder_content(
        os.path.join(mvsec_path, folder_name, "events"),
        subfolders=["voxels"],
        files=[],
    )
    check_folder_content(
        os.path.join(mvsec_path, folder_name, "events", "voxels"),
        subfolders=[],
        files=[f"event_tensor_{i:010}.npy" for i in range(folder_size)]
        + ["timestamps.txt"],
    )

    if images:
        # (5) Let's check the image folder
        opt_subfolders, _ = check_folder_content(
            os.path.join(mvsec_path, folder_name, "rgb"),
            subfolders=[],
            files=[],
            optional_subfolders=["davis", "davis_left_sync"],
        )

        if opt_subfolders["davis"]:
            check_folder_content(
                os.path.join(mvsec_path, folder_name, "rgb", "davis"),
                subfolders=[],
                files=[f"frame_{i:010}.png" for i in range(folder_size)]
                + ["timestamps.txt"],
            )

        if opt_subfolders["davis_left_sync"]:
            check_folder_content(
                os.path.join(mvsec_path, folder_name, "rgb", "davis_left_sync"),
                subfolders=[],
                files=[f"frame_{i:010}.png" for i in range(folder_size)]
                + ["timestamps.txt"],
            )
