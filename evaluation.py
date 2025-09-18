"""
Event-based Depth Estimation Evaluation Module

This module provides comprehensive evaluation tools for depth estimation models,
specifically designed for event-based vision datasets. It includes metrics calculation,
data preprocessing, visualization tools, and depth conversion utilities.

The module supports various depth representations (linear, logarithmic, inverse)
and provides standard evaluation metrics used in depth estimation research.
"""

import numpy as np
import torch
import argparse
import glob
from os.path import join
import tqdm
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def setup_argument_parser():
    """
    Set up command line argument parser for evaluation script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Event-based Depth Data Evaluation")

    # Dataset paths
    parser.add_argument("--target_dataset", default="gt", 
                        help="Path to ground truth depth dataset")
    parser.add_argument("--predictions_dataset", default="pred_depth",
                        help="Path to predicted depth dataset")
    parser.add_argument("--event_masks", default="",
                        help="Path to event masks (optional)")
    
    # Processing parameters
    parser.add_argument("--crop_ymax", default=260, type=int,
                        help="Maximum Y coordinate for cropping depth maps")
    parser.add_argument("--clip_distance", type=float, default=80.0,
                        help="Maximum depth distance for clipping")
    
    # File processing options
    parser.add_argument("--prediction_offset", type=int, default=0,
                        help="Offset for prediction file indexing")
    parser.add_argument("--target_offset", type=int, default=0,
                        help="Offset for target file indexing")
    
    # Debugging and visualization
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with visualizations")
    parser.add_argument("--idx", type=int, default=-1,
                        help="Specific frame index for debugging")
    parser.add_argument("--start_idx", type=int, default=-1,
                        help="Starting frame index for processing")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Output folder for saving debug visualizations")
    
    # Depth processing
    parser.add_argument("--inv", action="store_true",
                        help="Convert inverse log depth to regular depth")

    return parser.parse_args()

# Standard depth evaluation metrics for different depth thresholds
# These metrics cover overall evaluation (_) and threshold-specific evaluation (10, 20, 30 meters)
METRICS_KEYWORDS = [
    # Overall metrics
    "_abs_rel_diff",              # Absolute relative difference
    "_squ_rel_diff",              # Squared relative difference  
    "_RMS_linear",                # Root mean square linear error
    "_RMS_log",                   # Root mean square logarithmic error
    "_SILog",                     # Scale-invariant logarithmic error
    "_mean_target_depth",         # Mean ground truth depth
    "_median_target_depth",       # Median ground truth depth
    "_mean_prediction_depth",     # Mean predicted depth
    "_median_prediction_depth",   # Median predicted depth
    "_mean_depth_error",          # Mean absolute depth error
    "_median_diff",               # Median difference between GT and prediction
    "_threshold_delta_1.25",      # Percentage of pixels with ratio < 1.25
    "_threshold_delta_1.25^2",    # Percentage of pixels with ratio < 1.25^2
    "_threshold_delta_1.25^3",    # Percentage of pixels with ratio < 1.25^3
    
    # Metrics for depth threshold 10m
    "_10_mean_target_depth", "_10_median_target_depth",
    "_10_mean_prediction_depth", "_10_median_prediction_depth",
    "_10_abs_rel_diff", "_10_squ_rel_diff", "_10_RMS_linear",
    "_10_RMS_log", "_10_SILog", "_10_mean_depth_error",
    "_10_median_diff", "_10_threshold_delta_1.25",
    "_10_threshold_delta_1.25^2", "_10_threshold_delta_1.25^3",
    
    # Metrics for depth threshold 20m
    "_20_abs_rel_diff", "_20_squ_rel_diff", "_20_RMS_linear",
    "_20_RMS_log", "_20_SILog", "_20_mean_target_depth",
    "_20_median_target_depth", "_20_mean_prediction_depth",
    "_20_median_prediction_depth", "_20_mean_depth_error",
    "_20_median_diff", "_20_threshold_delta_1.25",
    "_20_threshold_delta_1.25^2", "_20_threshold_delta_1.25^3",
    
    # Metrics for depth threshold 30m
    "_30_abs_rel_diff", "_30_squ_rel_diff", "_30_RMS_linear",
    "_30_RMS_log", "_30_SILog", "_30_mean_target_depth",
    "_30_median_target_depth", "_30_mean_prediction_depth",
    "_30_median_prediction_depth", "_30_mean_depth_error",
    "_30_median_diff", "_30_threshold_delta_1.25",
    "_30_threshold_delta_1.25^2", "_30_threshold_delta_1.25^3",
]


def inv_depth_to_depth(prediction, reg_factor=3.70378):
    """
    Convert inverse log depth to regular depth representation.
    
    Args:
        prediction (np.ndarray): Input prediction in inverse log depth format
        reg_factor (float): Regularization factor for log depth conversion
        
    Returns:
        np.ndarray: Converted depth in regular format
    """
    # Convert from log inverse depth to normalized depth
    prediction = np.exp(reg_factor * (prediction - np.ones_like(prediction)))
    
    # Perform inverse operation (now is normalized depth)
    prediction = 1 / prediction
    prediction = prediction / np.amax(prediction)
    
    # Convert back to log depth format
    prediction = np.ones_like(prediction) + np.log(prediction) / reg_factor
    return prediction


def prepare_depth_data(target, prediction, clip_distance, reg_factor=3.70378):
    """
    Prepare both target and prediction depth data for evaluation.
    
    Args:
        target (np.ndarray): Ground truth depth data
        prediction (np.ndarray): Predicted depth data
        clip_distance (float): Maximum depth value for clipping
        reg_factor (float): Regularization factor for log depth conversion
        
    Returns:
        tuple: (processed_target, processed_prediction)
    """
    # Normalize prediction from log depth to linear depth (0-1)
    prediction = np.exp(reg_factor * (prediction - np.ones_like(prediction)))
    
    # Clip and normalize target depth
    target = np.clip(target, 0, clip_distance)
    target = target / np.amax(target[~np.isnan(target)])
    
    # Scale both to absolute values
    target *= clip_distance
    prediction *= clip_distance
    
    return target, prediction


def prepare_prediction_data(prediction, clip_distance, reg_factor=3.70378):
    """
    Prepare prediction data for evaluation by converting from log depth to linear depth.
    
    Args:
        prediction (np.ndarray): Predicted depth in log format
        clip_distance (float): Maximum depth value for scaling
        reg_factor (float): Regularization factor for log depth conversion
        
    Returns:
        np.ndarray: Processed prediction in linear depth format
    """
    # Convert from log depth to normalized linear depth (0-1)
    prediction = np.exp(reg_factor * (prediction - np.ones_like(prediction)))
    
    # Normalize to maximum value
    prediction = prediction / np.amax(prediction[~np.isnan(prediction)])
    
    # Scale to absolute depth values
    prediction *= clip_distance
    
    return prediction


def prepare_target_data(target, clip_distance):
    """
    Prepare target (ground truth) depth data for evaluation.
    
    Args:
        target (np.ndarray): Ground truth depth data
        clip_distance (float): Maximum depth value for clipping
        
    Returns:
        np.ndarray: Processed target depth data
    """
    # Clip and normalize target depth
    target = np.clip(target, 0, clip_distance)
    target = target / np.amax(target[~np.isnan(target)])
    
    # Scale to absolute depth values
    target *= clip_distance
    
    return target

def prepare_prediction_data_torch(prediction, clip_distance, reg_factor=3.70378):
    """
    Prepare prediction data using PyTorch tensors (GPU-friendly version).
    
    Args:
        prediction (torch.Tensor): Predicted depth in log format
        clip_distance (float): Maximum depth value for scaling
        reg_factor (float): Regularization factor for log depth conversion
        
    Returns:
        torch.Tensor: Processed prediction in linear depth format
    """
    # Convert from log depth to normalized linear depth (0-1)
    prediction = torch.exp(reg_factor * (prediction - torch.ones(
        (prediction.shape[-2], prediction.shape[-1]), 
        dtype=torch.float32, 
        device=prediction.device
    )))
    
    # Normalize to maximum value
    prediction = prediction / torch.amax(prediction[~torch.isnan(prediction)])
    
    # Scale to absolute depth values
    prediction *= clip_distance
    
    return prediction


def depth2log_depth_torch(depth, clip_distance, reg_factor=3.70378):
    """
    Convert linear depth to log depth representation using PyTorch.
    
    Args:
        depth (torch.Tensor): Linear depth values
        clip_distance (float): Maximum depth value for clipping
        reg_factor (float): Regularization factor for log conversion
        
    Returns:
        torch.Tensor: Log depth representation (0-1 range)
    """
    # Clamp and normalize depth
    depth = torch.clamp(depth, 0, clip_distance)
    depth = depth / clip_distance
    
    # Convert to log depth
    depth = torch.ones_like(depth) + torch.log(depth) / reg_factor
    depth = torch.clamp(depth, 0, 1)
    
    return depth


def log_depth2depth_torch(depth, clip_distance, reg_factor=3.70378):
    """
    Convert log depth to linear depth using PyTorch.
    
    Args:
        depth (torch.Tensor): Log depth values
        clip_distance (float): Maximum depth value for scaling
        reg_factor (float): Regularization factor for conversion
        
    Returns:
        torch.Tensor: Linear depth values
    """
    # Convert from log depth to linear depth
    depth = torch.exp(reg_factor * (depth - torch.ones_like(depth)))
    depth = depth * clip_distance
    
    return depth


def prepare_target_data_torch(target, clip_distance):
    """
    Prepare target depth data using PyTorch tensors (GPU-friendly version).
    
    Args:
        target (torch.Tensor): Ground truth depth data
        clip_distance (float): Maximum depth value for clipping
        
    Returns:
        torch.Tensor: Processed target depth data
    """
    # Clip and normalize target depth
    target = torch.clamp(target, 0, clip_distance)
    target = target / torch.amax(target[~torch.isnan(target)])
    
    # Scale to absolute depth values
    target *= clip_distance
    
    return target

def display_high_contrast_colormap(idx, target, prediction, prefix="", colormap='terrain', 
                                 debug=False, folder_name=None):
    """
    Display high contrast depth colormap for target and prediction comparison.
    
    Args:
        idx (int): Frame index for naming saved files
        target (np.ndarray): Ground truth depth data
        prediction (np.ndarray): Predicted depth data
        prefix (str): Prefix for window title
        colormap (str): Matplotlib colormap name
        debug (bool): Whether to show interactive plot
        folder_name (str): Directory to save plots (optional)
    """
    if folder_name is not None or debug:
        percent = 1.0
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10, 8))
        
        # Plot ground truth depth
        target_plot = np.flip(np.fliplr(np.clip(target, 0, percent * np.max(target))))
        pcm = ax[0].pcolormesh(target_plot, cmap=colormap, 
                              vmin=np.min(target), vmax=percent * np.max(target))
        ax[0].set_xticklabels([])  # Remove tick labels
        ax[0].set_title("Ground Truth Depth")
        fig.colorbar(pcm, ax=ax[0], extend='both', orientation='vertical')
        
        # Plot predicted depth
        prediction_plot = np.flip(np.fliplr(np.clip(prediction, 0, percent * np.max(prediction))))
        pcm = ax[1].pcolormesh(prediction_plot, cmap=colormap, 
                              vmin=np.min(target), vmax=percent * np.max(target))
        ax[1].set_title("Predicted Depth")
        fig.colorbar(pcm, ax=ax[1], extend='both', orientation='vertical')
        
        plt.suptitle(f"{prefix}High Contrast Depth Evaluation")
    
    if folder_name is not None:
        plt.savefig(f'{folder_name}/frame_{idx:010d}.png')
        plt.close(fig)
    if debug:
        plt.show()


def display_high_contrast_color_logmap(idx, data, prefix="", name="data", colormap='tab20c', 
                                     debug=False, folder_name=None):
    """
    Display data using logarithmic colormap scaling.
    
    Args:
        idx (int): Frame index for naming saved files
        data (np.ndarray): Data to visualize
        prefix (str): Prefix for window title
        name (str): Name for saved file
        colormap (str): Matplotlib colormap name
        debug (bool): Whether to show interactive plot and save file
        folder_name (str): Directory to save plots (optional)
    """
    if debug and folder_name is not None:
        percent = 1.0
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
        
        # Apply log scaling and flip for display
        data_plot = np.flip(np.fliplr(np.clip(data, 0, percent * np.max(data))))
        pcm = ax.pcolormesh(data_plot, 
                           norm=colors.LogNorm(vmin=np.min(data), vmax=np.max(data)), 
                           cmap=colormap)
        
        # Remove axis ticks
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        
        plt.suptitle(f"{prefix}High Contrast Depth Evaluation")
        plt.savefig(f'{folder_name}/{name}_frame_{idx:010d}.png')
        plt.close(fig)

def add_to_metrics(idx, metrics, target_, prediction_, mask, event_frame=None, 
                  prefix="", debug=False, output_folder=None):
    """
    Calculate and accumulate depth evaluation metrics.
    
    Args:
        idx (int): Current frame index
        metrics (dict): Dictionary to accumulate metrics
        target_ (np.ndarray): Ground truth depth data (full image)
        prediction_ (np.ndarray): Predicted depth data (full image)
        mask (np.ndarray): Valid pixel mask
        event_frame (np.ndarray, optional): Event frame for visualization
        prefix (str): Prefix for metric names
        debug (bool): Enable debug visualizations
        output_folder (str, optional): Folder to save debug plots
        
    Returns:
        dict: Updated metrics dictionary
    """
    # Initialize metrics dictionary if empty
    if len(metrics) == 0:
        metrics = {k: 0 for k in METRICS_KEYWORDS}

    # Create prediction and depth validity masks
    max_target = np.amax(target_[~np.isnan(target_)])
    prediction_mask = (prediction_ > 0) & (prediction_ < max_target)
    depth_mask = (target_ > 0) & (target_ < max_target)
    
    # Combine all masks
    valid_mask = mask & depth_mask & prediction_mask
    eps = 1e-5  # Small epsilon to avoid division by zero

    # Extract valid pixels only
    target = target_[valid_mask]
    prediction = prediction_[valid_mask]

    # Calculate threshold metrics (accuracy within certain ratios)
    ratio = np.max(np.stack([target/(prediction+eps), prediction/(target+eps)]), axis=0)

    # Initialize new metrics for this frame
    new_metrics = {}
    
    # Threshold accuracy metrics
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)

    # Calculate absolute and logarithmic differences
    log_diff = np.log(target + eps) - np.log(prediction + eps)
    abs_diff = np.abs(target - prediction)

    # Error metrics
    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff / (target + eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2 / (target**2 + eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff**2).mean() - (log_diff.mean())**2
    
    # Statistical metrics
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))

    # Accumulate metrics
    for k, v in new_metrics.items():
        metrics[k] += v

    # Debug visualization
    if debug:
        print("Current frame metrics:")
        pprint(new_metrics)
        
        # Create comprehensive debug visualization
        fig, ax = plt.subplots(ncols=3, nrows=4, figsize=(15, 20))
        
        # Row 1: Basic depth visualization
        ax[0, 0].imshow(target_, vmin=0, vmax=200, cmap='viridis')
        ax[0, 0].set_title("Ground Truth Depth")
        
        ax[0, 1].imshow(prediction_, vmin=0, vmax=200, cmap='viridis')
        ax[0, 1].set_title("Predicted Depth")
        
        target_debug = target_.copy()
        target_debug[~valid_mask] = 0
        ax[0, 2].imshow(target_debug, vmin=0, vmax=200, cmap='viridis')
        ax[0, 2].set_title("Ground Truth (Masked)")

        # Row 2: Logarithmic visualization
        ax[1, 0].imshow(np.log(target_ + eps), vmin=0, vmax=np.log(200), cmap='plasma')
        ax[1, 0].set_title("Log Ground Truth")
        
        ax[1, 1].imshow(np.log(prediction_ + eps), vmin=0, vmax=np.log(200), cmap='plasma')
        ax[1, 1].set_title("Log Prediction")
        
        ax[1, 2].imshow(np.max(np.stack([target_ / (prediction_ + eps), 
                                       prediction_ / (target_ + eps)]), axis=0), cmap='jet')
        ax[1, 2].set_title("Max Ratio")

        # Row 3: Error visualization
        ax[2, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)), cmap='hot')
        ax[2, 0].set_title("Absolute Log Difference")
        
        ax[2, 1].imshow(np.abs(target_ - prediction_), cmap='hot')
        ax[2, 1].set_title("Absolute Difference")
        
        # Event frame visualization (if available)
        if event_frame is not None:
            event_viz = np.zeros((*event_frame.shape[:2], 3))
            event_viz[:, :, 0] = (np.sum(event_frame.astype("float32"), axis=-1) > 0)
            event_viz[:, :, 1] = np.clip(target_.copy(), 0, 1)
            ax[2, 2].imshow(event_viz)
            ax[2, 2].set_title("Event Frame Overlay")

        # Row 4: Masked error visualization
        log_diff_masked = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
        log_diff_masked[~valid_mask] = 0
        ax[3, 0].imshow(log_diff_masked, cmap='hot')
        ax[3, 0].set_title("Abs Log Diff (Masked)")
        
        abs_diff_masked = np.abs(target_ - prediction_)
        abs_diff_masked[~valid_mask] = 0
        ax[3, 1].imshow(abs_diff_masked, cmap='hot')
        ax[3, 1].set_title("Abs Diff (Masked)")
        
        ax[3, 2].imshow(valid_mask, cmap='gray')
        ax[3, 2].set_title("Valid Pixel Mask")

        plt.tight_layout()
        plt.suptitle(f"{prefix}Depth Evaluation - Frame {idx}")
        
        if output_folder:
            plt.savefig(f"{output_folder}/debug_frame_{idx:06d}.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    return metrics


def load_and_process_files(flags):
    """
    Load and validate prediction and target files.
    
    Args:
        flags: Parsed command line arguments
        
    Returns:
        tuple: (prediction_files, target_files, use_event_masks)
    """
    # Load prediction files
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, '*.npy')))
    prediction_files = prediction_files[flags.prediction_offset:]

    # Load target files
    target_files = sorted(glob.glob(join(flags.target_dataset, '*.npy')))
    target_files = target_files[flags.target_offset:]

    # Load event mask files if specified
    use_event_masks = False
    if flags.event_masks != "":
        event_frame_files = sorted(glob.glob(join(flags.event_masks, '*.png')))
        event_frame_files = event_frame_files[flags.prediction_offset:]
        use_event_masks = len(event_frame_files) > 0

    # Print dataset information
    print(f"Number of prediction files: {len(prediction_files)}")
    print(f"Number of target files: {len(target_files)}")
    if flags.event_masks != "":
        print(f"Number of event files: {len(event_frame_files) if 'event_frame_files' in locals() else 0}")

    # Validate file counts
    assert len(prediction_files) > 0, "No prediction files found"
    assert len(target_files) > 0, "No target files found"
    assert len(prediction_files) == len(target_files), "Mismatch between prediction and target file counts"

    return prediction_files, target_files, use_event_masks


def process_single_frame(prediction_file, target_file, flags, frame_idx):
    """
    Process a single frame pair (prediction and target).
    
    Args:
        prediction_file (str): Path to prediction file
        target_file (str): Path to target file
        flags: Command line arguments
        frame_idx (int): Current frame index
        
    Returns:
        tuple: (target_depth, predicted_depth, valid_mask)
    """
    # Load ground truth depth
    target_depth = np.load(target_file)
    target_depth = np.squeeze(target_depth)
    target_depth = target_depth[:flags.crop_ymax]  # Crop height

    # Load predicted depth
    predicted_depth = np.load(prediction_file)
    predicted_depth = np.squeeze(predicted_depth)
    predicted_depth = predicted_depth[:flags.crop_ymax]  # Crop height

    # Convert from inverse log depth if needed
    if flags.inv:
        predicted_depth = inv_depth_to_depth(predicted_depth)

    # Process depth data
    target_depth, predicted_depth = prepare_depth_data(
        target_depth, predicted_depth, flags.clip_distance
    )

    # Validate shapes match
    assert predicted_depth.shape == target_depth.shape, \
        f"Shape mismatch: prediction {predicted_depth.shape} vs target {target_depth.shape}"

    # Create valid mask (currently accepts all pixels)
    valid_mask = np.ones_like(target_depth, dtype=bool)

    return target_depth, predicted_depth, valid_mask


def main():
    """Main evaluation function."""
    # Parse command line arguments
    flags = setup_argument_parser()

    # Load and validate files
    prediction_files, target_files, use_event_masks = load_and_process_files(flags)

    # Initialize metrics accumulator
    metrics = {}
    num_frames = len(target_files)

    print(f"Starting evaluation of {num_frames} frames...")

    # Process each frame
    for idx in tqdm.tqdm(range(num_frames), desc="Evaluating frames"):
        pred_file, target_file = prediction_files[idx], target_files[idx]

        # Process current frame
        target_depth, predicted_depth, valid_mask = process_single_frame(
            pred_file, target_file, flags, idx
        )

        # Determine if this frame should show debug info
        debug_this_frame = flags.debug and (flags.idx == -1 or idx == flags.idx)

        # Calculate overall metrics
        metrics = add_to_metrics(
            idx, metrics, target_depth, predicted_depth, valid_mask, 
            event_frame=None, prefix="_", debug=debug_this_frame, 
            output_folder=flags.output_folder
        )

        # Calculate metrics for different depth thresholds
        for depth_threshold in [10, 20, 30]:
            threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
            combined_mask = valid_mask & threshold_mask
            
            add_to_metrics(
                -1, metrics, target_depth, predicted_depth, combined_mask,
                prefix=f"_{depth_threshold}_", debug=debug_this_frame
            )

    # Print final results
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    
    # Calculate averages
    final_metrics = {k: v/num_frames for k, v in metrics.items()}
    
    # Pretty print results
    pprint(final_metrics)
    
    # Also print in simple format
    print("\nDetailed metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()