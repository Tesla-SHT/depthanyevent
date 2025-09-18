"""
Event Slicer for DSEC Dataset

This module provides efficient temporal access to event data stored in HDF5 format.
The EventSlicer class enables fast extraction of events within specified time windows
using precomputed time-to-index mappings.

Key Features:
- Millisecond-resolution time indexing for fast lookups
- Conservative time window computation to ensure completeness  
- Microsecond-precision event extraction within windows
- Memory-efficient streaming access to large event files
- Numba-accelerated time index computation

The time indexing system works as follows:
- Events are stored with microsecond timestamps
- A mapping from milliseconds to event indices is precomputed
- For any time window, we first find a conservative millisecond window
- Then refine to exact microsecond boundaries within that window

Example Usage:
    slicer = EventSlicer(h5py_file)
    events = slicer.get_events(start_time_us, end_time_us)
    # Returns dict with keys: 'x', 'y', 'p', 't'
"""

import math
from typing import Dict, Tuple, Optional

import h5py
import hdf5plugin
from numba import jit
import numpy as np


class EventSlicer:
    """
    Efficient temporal access to event data stored in HDF5 format.
    
    This class provides fast extraction of events within specified time windows
    using precomputed millisecond-to-index mappings. It's designed for efficient
    streaming access to large event datasets without loading everything into memory.
    
    Time Indexing System:
    The slicer uses a two-level indexing approach:
    1. Coarse millisecond-level mapping (ms_to_idx) for fast initial lookup
    2. Fine microsecond-level binary search within the coarse window
    
    This enables O(1) coarse lookup + O(log n) fine refinement instead of 
    O(log N) search over the entire dataset.
    
    Args:
        h5f: HDF5 file handle containing event data and time indices
        
    Attributes:
        events: Dict mapping event property names to HDF5 datasets
        ms_to_idx: Millisecond-to-index mapping array
        t_offset: Time offset for GPS time conversion
        t_final: Final timestamp in the dataset
    """
    
    def __init__(self, h5f: h5py.File):
        """
        Initialize the EventSlicer with an HDF5 file handle.
        
        Args:
            h5f: Open HDF5 file containing event data with structure:
                - events/p: Polarity data
                - events/x: X coordinates  
                - events/y: Y coordinates
                - events/t: Timestamps
                - ms_to_idx: Millisecond-to-index mapping
                - t_offset: Optional time offset
        """
        self.h5f = h5f

        # Load event datasets for efficient access
        self.events = {}
        for dataset_str in ['p', 'x', 'y', 't']:
            self.events[dataset_str] = self.h5f[f"events/{dataset_str}"]

        # Load the millisecond-to-index mapping
        # This mapping is defined such that:
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # where 'ms' is time in milliseconds and 't' is timestamps in microseconds
        #
        # Example with events at times (microseconds): 0, 500, 2100, 5000, 5000, 7100, 7200, 7200, 8100, 9000
        # and millisecond indices:                     0,  1,    2,    3,    4,    5,    6,    7,    8,    9
        # The ms_to_idx mapping becomes:               0,  2,    2,    3,    3,    3,    5,    5,    8,    9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        # Load time offset if available (for GPS time conversion)
        if "t_offset" in list(h5f.keys()):
            # Handle HDF5 dataset access properly - read as numpy array
            t_offset_dataset = h5f['t_offset']
            t_offset_array = np.array(t_offset_dataset)
            self.t_offset = int(t_offset_array.item() if t_offset_array.size == 1 else t_offset_array[0])
        else:
            self.t_offset = 0
            
        # Compute final timestamp with offset
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self) -> int:
        """
        Get the start time of the event stream in microseconds.
        
        Returns:
            int: Start time in microseconds (including offset)
        """
        return self.t_offset

    def get_final_time_us(self) -> int:
        """
        Get the final time of the event stream in microseconds.
        
        Returns:
            int: Final time in microseconds (including offset)
        """
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """
        Extract events within the specified time window.
        
        This is the main interface for accessing event data. It uses a two-stage 
        approach for efficiency:
        1. Find a conservative millisecond-level window that contains the target range
        2. Refine to exact microsecond boundaries within that window
        
        Args:
            t_start_us: Start time in microseconds (GPS time with offset)
            t_end_us: End time in microseconds (GPS time with offset)
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing event data:
                - 'p': Polarity values (1 for positive, 0 for negative)
                - 'x': X coordinates of events
                - 'y': Y coordinates of events  
                - 't': Timestamps in microseconds (GPS time)
            Returns an empty dictionary if the time window cannot be satisfied.
            
        Note:
            The returned events satisfy: t_start_us <= event_times < t_end_us
        """
        assert t_start_us < t_end_us, f"Invalid time window: {t_start_us} >= {t_end_us}"

        # Convert GPS time to dataset-local time by subtracting offset
        t_start_us_local = t_start_us - self.t_offset
        t_end_us_local = t_end_us - self.t_offset

        # Find conservative millisecond window that contains our target
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us_local, t_end_us_local)
        
        # Convert millisecond times to dataset indices
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size - requested time is beyond dataset bounds
            events = {}
            events['t'] = np.array([], dtype='int64')
            events['x'] = np.array([], dtype='int16')
            events['y'] = np.array([], dtype='int16')
            events['p'] = np.array([], dtype='int8')
            return events

        # Load timestamps within the conservative window
        time_array_conservative = np.asarray(
            self.events['t'][t_start_ms_idx:t_end_ms_idx], 
            dtype='int64'
        )
        
        # Find exact microsecond boundaries within the conservative window
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us_local, t_end_us_local
        )
        
        # Calculate absolute indices in the full dataset
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        
        # Extract event data within the exact time bounds
        events = {}
        
        # Extract timestamps and convert back to GPS time by adding offset
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        
        # Extract other event properties
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            # Verify consistency - all properties should have same number of events
            assert events[dset_str].size == events['t'].size, (
                f"Size mismatch: {dset_str} has {events[dset_str].size} events, "
                f"but timestamps have {events['t'].size}"
            )
            
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us: int) -> Tuple[int, int]:
        """
        Compute a conservative time window with millisecond resolution.
        
        Since we have millisecond-to-index mapping, we need to compute the 
        smallest millisecond window that fully contains the requested microsecond window.
        This ensures we don't miss any events at the boundaries.
        
        Args:
            ts_start_us: Start time in microseconds (local dataset time)
            ts_end_us: End time in microseconds (local dataset time)
            
        Returns:
            Tuple[int, int]: (window_start_ms, window_end_ms)
                - window_start_ms: Conservative start time in milliseconds (rounded down)
                - window_end_ms: Conservative end time in milliseconds (rounded up)
                
        Example:
            For a window [1500us, 3200us]:
            - window_start_ms = floor(1500/1000) = 1ms  
            - window_end_ms = ceil(3200/1000) = 4ms
            This ensures all events in [1500us, 3200us) are within [1ms, 4ms).
        """
        assert ts_end_us > ts_start_us, f"Invalid time range: {ts_start_us} >= {ts_end_us}"
        
        # Round down start time to get conservative lower bound
        window_start_ms = math.floor(ts_start_us / 1000)
        
        # Round up end time to get conservative upper bound  
        window_end_ms = math.ceil(ts_end_us / 1000)
        
        return window_start_ms, window_end_ms

    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """
        Find exact microsecond boundaries within a timestamp array.
        
        This function is Numba-compiled for high performance. It performs binary
        search-like operations to find the precise start and end indices for
        the requested time window within the provided timestamp array.
        
        Args:
            time_array: 1D array of timestamps in microseconds (sorted ascending)
            time_start_us: Start time in microseconds  
            time_end_us: End time in microseconds
            
        Returns:
            Tuple[int, int]: (idx_start, idx_end) such that:
                - time_array[idx_start:idx_end] contains all events in [time_start_us, time_end_us)
                - time_array[idx_start] >= time_start_us (first event at or after start)
                - time_array[idx_end-1] < time_end_us (last event before end)
                
        Edge Cases:
            - If no events satisfy the condition, returns (array.size, array.size) for empty slice
            - Handles duplicate timestamps correctly
            - Optimized for typical case where events exist within the window
            
        Performance:
            - O(n) worst case, but typically much faster due to early termination
            - Numba compilation provides significant speedup for large arrays
        """
        assert time_array.ndim == 1, "Time array must be 1-dimensional"

        # Handle edge case: all events are before the requested start time
        if time_array[-1] < time_start_us:
            # Example scenario:
            # time_array = [1016, 1500, 1984]  
            # time_start_us = 1990, time_end_us = 2000
            # No events satisfy the condition, return empty slice
            return time_array.size, time_array.size

        # Find start index: first event at or after time_start_us
        idx_start = -1
        for idx_from_start in range(0, time_array.size, 1):
            if time_array[idx_from_start] >= time_start_us:
                idx_start = idx_from_start
                break
        
        assert idx_start >= 0, "Start index should be found if array[-1] >= time_start_us"

        # Find end index: first event at or after time_end_us
        # We search backwards for efficiency in typical use cases
        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                # Found last event before time_end_us, end search
                break

        # Verification: ensure our indices satisfy the contract
        assert time_array[idx_start] >= time_start_us, (
            f"Start condition violated: time_array[{idx_start}]={time_array[idx_start]} < {time_start_us}"
        )
        
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us, (
                f"End condition violated: time_array[{idx_end}]={time_array[idx_end]} < {time_end_us}"
            )
            
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us, (
                f"Start boundary violated: time_array[{idx_start-1}]={time_array[idx_start-1]} >= {time_start_us}"
            )
            
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us, (
                f"End boundary violated: time_array[{idx_end-1}]={time_array[idx_end-1]} >= {time_end_us}"
            )
            
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> Optional[int]:
        """
        Convert millisecond timestamp to dataset index.
        
        Uses the precomputed millisecond-to-index mapping for O(1) lookup.
        This is the core of the efficient temporal indexing system.
        
        Args:
            time_ms: Time in milliseconds (non-negative)
            
        Returns:
            Optional[int]: Dataset index corresponding to the millisecond timestamp.
                Returns None if the requested time is beyond the dataset bounds.
                
        Note:
            The returned index satisfies:
            - events['t'][index] >= time_ms * 1000 (first event at or after time_ms)
            - events['t'][index-1] < time_ms * 1000 (if index > 0)
        """
        assert time_ms >= 0, f"Time must be non-negative, got {time_ms}"
        
        # Check if requested time is beyond dataset bounds
        if time_ms >= self.ms_to_idx.size:
            return None
            
        return self.ms_to_idx[time_ms]