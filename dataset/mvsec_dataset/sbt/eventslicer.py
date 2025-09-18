"""
Event Slicer for MVSEC Dataset

This module provides efficient temporal access to event data from the MVSEC 
(Multi Vehicle Stereo Event Camera) dataset stored in HDF5 format.

Key Differences from DSEC EventSlicer:
- Data structure: MVSEC stores events as 4-column arrays [x, y, t, p]
- Time conversion: Converts floating-point timestamps to int64 microseconds
- Index mapping: Uses np.searchsorted instead of precomputed ms_to_idx
- No GPS offset: MVSEC timestamps are relative to sequence start

The MVSEC dataset uses DAVIS sensors which store event data in a different
format compared to DSEC. This slicer adapts the indexing system to work
efficiently with MVSEC's data structure.

Example Usage:
    with h5py.File('mvsec_sequence.h5', 'r') as h5f:
        slicer = EventSlicer(h5f)
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
    Efficient temporal access to MVSEC event data.
    
    This class provides fast extraction of events within specified time windows
    from MVSEC dataset files. Unlike DSEC, MVSEC uses DAVIS sensors which store
    events in a 4-column format [x, y, t, p] and require different processing.
    
    Key MVSEC-specific features:
    - Events stored as davis/left/events dataset with 4 columns
    - Timestamps in floating-point seconds, converted to int64 microseconds
    - Dynamic ms_to_idx mapping using np.searchsorted (not precomputed)
    - No GPS time offset (sequence-relative timestamps)
    
    Args:
        h5f: HDF5 file handle containing MVSEC event data
        
    Attributes:
        events: Dict mapping event property names to numpy arrays
        ms_to_idx: Dynamically computed millisecond-to-index mapping
        t_offset: Time offset (always 0 for MVSEC)
        t_start: Start timestamp of the sequence
        t_final: Final timestamp of the sequence
    """
    
    def __init__(self, h5f: h5py.File):
        """
        Initialize the MVSEC EventSlicer.
        
        Args:
            h5f: Open HDF5 file containing MVSEC event data with structure:
                - davis/left/events: 4-column array [x, y, t, p]
        """
        self.h5f = h5f

        # Load event data from MVSEC format
        # MVSEC stores events as 4-column arrays: [x, y, t, p]
        self.events = {}
        event_data_path = "davis/left/events"
        
        # Handle HDF5 dataset access safely
        event_dataset = self.h5f[event_data_path]
        event_array = np.array(event_dataset)  # Load full array to memory
        
        for i, dataset_str in enumerate(['x', 'y', 't', 'p']):
            self.events[dataset_str] = event_array[:, i]
            
            # Convert timestamps from float seconds to int64 microseconds
            if dataset_str == 't':
                self.events[dataset_str] = (self.events[dataset_str] * 1e6).astype('int64')

        # Create millisecond-to-index mapping using searchsorted
        # Unlike DSEC, MVSEC doesn't have precomputed ms_to_idx
        # We compute it dynamically for the available time range
        t_start_ms = self.events['t'][0] // 1000
        t_end_ms = self.events['t'][-1] // 1000
        ms_range = np.arange(t_start_ms, t_end_ms + 1, 1) * 1000
        self.ms_to_idx = np.searchsorted(self.events['t'], ms_range)

        # Set time boundaries (no GPS offset in MVSEC)
        self.t_offset = 0
        self.t_start = int(self.events['t'][0]) + self.t_offset
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self) -> int:
        """
        Get the start time of the event sequence in microseconds.
        
        Returns:
            int: Start time in microseconds (sequence-relative)
        """
        return self.t_start

    def get_final_time_us(self) -> int:
        """
        Get the final time of the event sequence in microseconds.
        
        Returns:
            int: Final time in microseconds (sequence-relative)
        """
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """
        Extract events within the specified time window.
        
        This method uses a two-stage approach similar to DSEC but adapted for MVSEC:
        1. Find a conservative millisecond-level window
        2. Refine to exact microsecond boundaries
        
        Args:
            t_start_us: Start time in microseconds (sequence-relative)
            t_end_us: End time in microseconds (sequence-relative)
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing event data:
                - 'x': X coordinates of events
                - 'y': Y coordinates of events
                - 'p': Polarity values (1 for positive, 0 for negative)
                - 't': Timestamps in microseconds (sequence-relative)
                
        Note:
            The returned events satisfy: t_start_us <= event_times < t_end_us
        """
        assert t_start_us < t_end_us, f"Invalid time window: {t_start_us} >= {t_end_us}"

        # Convert to dataset-local time (subtract offset, which is 0 for MVSEC)
        t_start_us_local = t_start_us - self.t_offset
        t_end_us_local = t_end_us - self.t_offset

        # Find conservative millisecond window
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us_local, t_end_us_local)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        # Extract events within conservative window
        events = {}
        time_array_conservative = np.asarray(
            self.events['t'][t_start_ms_idx:t_end_ms_idx], 
            dtype='int64'
        )
        
        # Refine to exact microsecond boundaries
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us_local, t_end_us_local
        )
        
        # Calculate absolute indices in the full dataset
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        
        # Extract timestamps and convert back to sequence time
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
        
        This function is identical to the DSEC version - it computes the 
        smallest millisecond window that fully contains the requested microsecond window.
        
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
        """
        assert ts_end_us > ts_start_us, f"Invalid time range: {ts_start_us} >= {ts_end_us}"
        
        window_start_ms = math.floor(ts_start_us / 1000)
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
        
        This function is identical to the DSEC version and uses Numba compilation
        for high performance. It performs linear search to find precise start 
        and end indices for the requested time window.
        
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
            
        Performance:
            - O(n) worst case with early termination optimizations
            - Numba compilation provides significant speedup for large arrays
        """
        assert time_array.ndim == 1, "Time array must be 1-dimensional"

        # Handle edge case: all events are before the requested start time
        if time_array[-1] < time_start_us:
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
        # Search backwards for efficiency in typical use cases
        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                # Found last event before time_end_us, stop search
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

    def ms2idx(self, time_ms: int) -> int:
        """
        Convert millisecond timestamp to dataset index with MVSEC-specific handling.
        
        This method differs from DSEC by using relative time indexing and
        boundary clamping to handle edge cases gracefully.
        
        Args:
            time_ms: Time in milliseconds (sequence-relative)
            
        Returns:
            int: Dataset index corresponding to the millisecond timestamp.
                Always returns a valid index within bounds.
                
        Note:
            MVSEC-specific behavior:
            - Converts to sequence-relative time by subtracting t_start
            - Clamps negative times to 0 (beginning of sequence)
            - Clamps times beyond sequence to last valid index
        """
        # Convert to sequence-relative time
        time_ms_relative = time_ms - self.t_start // 1000
        
        # Clamp to valid range
        if time_ms_relative < 0:
            time_ms_relative = 0
        elif time_ms_relative >= self.ms_to_idx.size:
            time_ms_relative = self.ms_to_idx.size - 1

        return self.ms_to_idx[time_ms_relative]