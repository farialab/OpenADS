import numpy as np
from typing import Tuple, Union
from scipy.spatial.distance import directed_hausdorff

def calculate_dice_coefficient(y_true: np.ndarray, 
                              y_pred: np.ndarray, 
                              smooth: float = 1e-6) -> float:
    """
    Calculate Dice coefficient between two binary masks
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (0-1 range)
    """
    # Convert to binary masks if needed
    y_true_binary = y_true > 0.5
    y_pred_binary = y_pred > 0.5
    
    # Calculate intersection and union
    intersection = np.sum(y_true_binary & y_pred_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice

def calculate_hausdorff_distance(y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                percentile: int = 95) -> float:
    """
    Calculate the Hausdorff distance between two binary masks
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        percentile: Percentile for modified Hausdorff distance (default: 95)
    
    Returns:
        Hausdorff distance
    """
    # Convert to binary masks if needed
    y_true_binary = y_true > 0.5
    y_pred_binary = y_pred > 0.5
    
    # Get coordinates of mask points
    y_true_coords = np.array(np.where(y_true_binary)).T
    y_pred_coords = np.array(np.where(y_pred_binary)).T
    
    # Handle empty masks
    if len(y_true_coords) == 0 or len(y_pred_coords) == 0:
        if len(y_true_coords) == 0 and len(y_pred_coords) == 0:
            return 0.0  # Both masks are empty
        else:
            return float('inf')  # One mask is empty
    
    # Calculate Hausdorff distance
    d1, _ = directed_hausdorff(y_true_coords, y_pred_coords)
    d2, _ = directed_hausdorff(y_pred_coords, y_true_coords)
    
    return max(d1, d2)

def calculate_volume_metrics(y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           voxel_volume: float) -> dict:
    """
    Calculate volume-based metrics between two binary masks
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        voxel_volume: Volume of a single voxel in mm³ or ml
    
    Returns:
        Dictionary with volume metrics
    """
    # Convert to binary masks
    y_true_binary = y_true > 0.5
    y_pred_binary = y_pred > 0.5
    
    # Calculate volumes
    true_volume = np.sum(y_true_binary) * voxel_volume
    pred_volume = np.sum(y_pred_binary) * voxel_volume
    
    # Calculate absolute and relative volume difference
    abs_diff = abs(true_volume - pred_volume)
    rel_diff = abs_diff / true_volume if true_volume > 0 else float('inf')
    
    return {
        'true_volume': true_volume,
        'pred_volume': pred_volume,
        'abs_volume_diff': abs_diff,
        'rel_volume_diff': rel_diff
    }

def evaluate_segmentation(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         voxel_volume: float = 1.0) -> dict:
    """
    Comprehensive evaluation of segmentation performance
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        voxel_volume: Volume of a single voxel in mm³ or ml
    
    Returns:
        Dictionary with evaluation metrics
    """
    dice = calculate_dice_coefficient(y_true, y_pred)
    hausdorff = calculate_hausdorff_distance(y_true, y_pred)
    volume_metrics = calculate_volume_metrics(y_true, y_pred, voxel_volume)
    
    return {
        'dice': dice,
        'hausdorff': hausdorff,
        **volume_metrics
    }