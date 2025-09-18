import os.path
from pathlib import Path
import weakref
from xml.sax.saxutils import escape

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

def find_closest(a, b):
    idx = np.searchsorted(b, a)  # Find indices where elements of a should be inserted in b
    idx = np.clip(idx, 1, len(b) - 1)  # Ensure indices are within valid range
    left = b[idx - 1]  # Left neighbor
    right = b[idx]  # Right neighbor
    closest = np.where(np.abs(a - left) <= np.abs(a - right), idx - 1, idx)  # Pick closest
    return closest

class MVSECSequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    # ...

    def __init__(
        self,
        sequence_path: str,
        event_representation: EventRepresentation,
        time_window_ms: int,
        augmentator: Augmentator = None,
        load_images: bool = False,
        use_voxels: bool = False,
        overfit: bool = False,
        sequence_window: int = 1,
        sequence_step: int = 1,
        split: str = "train",
        self_supervised: bool = False,
        postfix: str = "",
    ):

        self.event_representation = event_representation
        self.load_images = load_images
        self.augmentator = augmentator
        self.self_supervised = self_supervised
        self.postfix = postfix

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

        # Check if hdf5 file exists
        event_data_hdf5_path = os.path.join(sequence_path, "..", "hdf5", "data.hdf5")
        self.event_data_hdf5_exists = os.path.isfile(event_data_hdf5_path)

        # event_gt_hdf5_path = os.path.join(sequence_path, "..", "hdf5", "gt.hdf5")
        # self.event_gt_hdf5_exists = os.path.isfile(event_gt_hdf5_path)

        # you can use voxels only if you have the hdf5 file
        if not self.event_data_hdf5_exists and not use_voxels:
            print(
                f"Ops! I can't find the file {event_data_hdf5_path}: setting use_voxels=True."
            )
            use_voxels = True

        self.use_voxels = use_voxels

        if use_voxels:
            time_window_ms = 50

        # Save delta timestamp in us
        delta_t_us = time_window_ms * 1000

        # Let's calculate the set of timestamps that this dataset is composed of
        # We will get timestamps from all the frames with a disparity
        self.timestamps_depth = (
            np.loadtxt(os.path.join(sequence_path, "depth", "data", "timestamps.txt"))[:, 1] * 1e6
        ).astype("int64")

        self.rgb_foldername = "davis_left_sync" if split == "test" else "davis"

        self.timestamps_rgb = (
            np.loadtxt(
                os.path.join(sequence_path, "rgb", self.rgb_foldername, "timestamps.txt")
            )[:, 1] * 1e6
        ).astype("int64")

        if self.self_supervised:
            self.timestamps_depth = self.timestamps_rgb

        if load_images:
            # For each depth timestamp, find the closest rgb timestamp
            self.rgb_indices = find_closest(self.timestamps_depth, self.timestamps_rgb)
        else:
            self.timestamps_rgb = None

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

        # Let's get the correct image for each frame
        if load_images:
            self.base_left_images_path = os.path.join(sequence_path, "rgb", self.rgb_foldername)

            left_images_files = []
            for entry in os.listdir(self.base_left_images_path):
                if entry.endswith(".png"):
                    left_images_files.append(entry)
            assert len(left_images_files) == len(self.timestamps_rgb)

            left_images_files.sort()
            self.left_images = left_images_files
            
            # REMEMBER: apply self.rgb_indices then self.depth_indices
            self.left_images = [self.left_images[i] for i in self.rgb_indices]
            self.left_images = [self.left_images[i] for i in self.depth_indices]
            self.timestamps_rgb = self.timestamps_rgb[self.rgb_indices]
            self.timestamps_rgb = self.timestamps_rgb[self.depth_indices]

            assert len(self.left_images) == len(self.timestamps_depth) == len(self.timestamps_rgb), f"len(self.left_images)={len(self.left_images)} != len(self.timestamps_depth)={len(self.timestamps_depth)} != len(self.timestamps_rgb)={len(self.timestamps_rgb)}"

        # Let's prepare the correct event window for each frame
        # ASSUMPTION: depth_aligned_event_windows == voxel_aligned_event_windows
        self.depth_aligned_event_widows = []
        self.rgb_aligned_event_windows = []

        for timestamp in self.timestamps_depth:
            self.depth_aligned_event_widows.append(
                (timestamp - delta_t_us, timestamp)
            )

        if load_images:
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

        if overfit:
            self.depths = self.depths[0:self.sequence_window]
            self.depth_aligned_event_widows = self.depth_aligned_event_widows[0:self.sequence_window]
            self.rgb_aligned_event_windows = self.rgb_aligned_event_windows[0:self.sequence_window]
            self.timestamps_depth = self.timestamps_depth[0:self.sequence_window]
            self.timestamps_rgb = self.timestamps_rgb[0:self.sequence_window]
            self.dataset_length = self.sequence_window

            if use_voxels:
                self.voxels = self.voxels[0:self.sequence_window]

            if load_images:
                self.left_images = self.left_images[0:self.sequence_window]

    def events_to_representation(self, x, y, p, t):
        t = (t - t[0]).astype("float32")
        t = t / t[-1]
        x = x.astype("float32")
        y = y.astype("float32")
        pol = p.astype("float32")
        return self.event_representation.convert(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pol),
            torch.from_numpy(t),
        )

    def get_depth_map(self, filepath: str):
        assert os.path.isfile(filepath)
        _tmp = np.load(filepath)
        _tmp = _tmp['depth'] if self.self_supervised else _tmp
        _tmp = _tmp.astype("float32")
        depthmap = torch.tensor(_tmp)
        depthmap[torch.isnan(depthmap)] = 0
        depthmap[torch.isinf(depthmap)] = 0
        return depthmap

    @staticmethod
    def close_callback(h5f_dict):
        for key in h5f_dict:
            h5f_dict[key].close()

    def __len__(self):
        return self.dataset_length

    @staticmethod
    def load_voxel(filepath: str):
        assert os.path.isfile(filepath)
        return torch.tensor(np.load(filepath).astype("float32"))

    def __getitem__(self, index):
        
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
                            self.depth_aligned_event_widows,
                            self.rgb_aligned_event_windows,
                        ],
                    )
                else:
                    _tmp = zip(
                        ["depth_aligned_events"], [self.depth_aligned_event_widows]
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
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                if len(left_image.shape) == 2:
                    left_image = left_image[..., None]
                left_image = left_image.transpose(2, 0, 1) / 255.0
                left_image = left_image.astype("float32")
                to_return["rgb"] = torch.tensor(left_image)
                # to_return["timestamp_rgb"] = self.timestamps_rgb[_index]

            # Check if all tensors have three dimensions CHW
            for key in to_return:
                if len(to_return[key].shape) == 2:
                    to_return[key] = to_return[key].unsqueeze(0)
            
            to_return_list.append(to_return)

        # stack all tensors in new axis=0 -> new tensor shape TCHW
        to_return = {}
        for key in to_return_list[0]:
            to_return[key] = torch.stack([to_return[key] for to_return in to_return_list], dim=0)


        # Apply same augmentations only to 'rgb, 'depth', 'depth_aligned_events' 'rgb_aligned_events'
        if self.augmentator is not None:
            to_return.update(
                self.augmentator(
                    {
                        key: to_return[key]
                        for key in [
                            "rgb",
                            "depth",
                            "depth_aligned_events",
                            "rgb_aligned_events",
                        ]
                        if key in to_return
                    }
                )
            )

        # keys: 'depth', 'timestamp_depth', 'depth_aligned_events', 'rgb_aligned_events', 'rgb', 'timestamp_rgb'
        return to_return
