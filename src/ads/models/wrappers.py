import os
import torch
import torch.nn as nn
import numpy as np
import ants
import sys
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
#from skimage import morphology
from scipy.ndimage import morphology, gaussian_filter

# Add project root to Python path
#sys.path.append(str(Path(__file__).parent.parent))

from .dagmnet_dwi import DAGMNet
from ads.core.logging import logger
from ads.core.preprocessing import (
    stroke_closing,
    remove_small_objects_in_slice,
    remove_small_objects
)

def pt_load_model(Lesion_model_name, device=None):
    """Load PyTorch model for inference
    Args:
        device: Optional[str] = None, will use GPU if available/specified
    """
    model = DAGMNet()
    # Determine device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(torch.load(
        Lesion_model_name, 
        map_location=device,
        weights_only=True
    ))
    model.eval()
    return model.to(device)

# def get_stroke_seg_MNI(model, #model: DAGMNet,
#                       dwi_img: np.ndarray, 
#                       adc_img: np.ndarray, 
#                       Prob_IS: Optional[np.ndarray] = None, 
#                       N_channel: int = 3,
#                       DS: int = 2) -> np.ndarray:
#         """
#         Performs stroke segmentation using DAGMNet model with multi-scale inference.
        
#         Args:
#             model: DAGMNet PyTorch model
#             dwi_img: Diffusion weighted image array
#             adc_img: ADC image array  
#             Prob_IS: Initial stroke probability map (optional)
#             N_channel: Number of input channels (2 or 3)
#             DS: Downsampling factor for multi-scale inference
        
#         Returns:
#             Binary stroke prediction mask after post-processing
#         """
#         stroke_pred_resampled = np.zeros_like(dwi_img)
        
#         # Multi-scale inference by offsetting starting points
#         for x_idx, y_idx, slice_idx in [(x,y,z) for x in range(DS) for y in range(DS) for z in range(2*DS)]:
#             # Extract downsampled views
#             dwi_DS_img = dwi_img[x_idx::DS, y_idx::DS, slice_idx::2*DS]
#             adc_DS_img = adc_img[x_idx::DS, y_idx::DS, slice_idx::2*DS]
            
#             if N_channel == 3:
#                 if Prob_IS is None:
#                     raise ValueError("Prob_IS is required for 3-channel input")
#                 prob_IS_DS = Prob_IS[x_idx::DS, y_idx::DS, slice_idx::2*DS]
#                 input_tensor = torch.stack([
#                     torch.FloatTensor(dwi_DS_img),
#                     torch.FloatTensor(adc_DS_img),
#                     torch.FloatTensor(prob_IS_DS)
#                 ]).unsqueeze(0)
#             else:
#                 input_tensor = torch.stack([
#                     torch.FloatTensor(dwi_DS_img),
#                     torch.FloatTensor(adc_DS_img)
#                 ]).unsqueeze(0)
                
#             # Inference
#             with torch.no_grad():
#                 output_fused, _, _, _, _ = model(input_tensor)
#                 output_fused = output_fused.cpu() 
#                 stroke_pred = output_fused.squeeze().numpy()
               
#             # Accumulate predictions
#             stroke_pred_resampled[x_idx::DS, y_idx::DS, slice_idx::2*DS] = stroke_pred

#         # Post-processing
#         stroke_pred_tmp = (stroke_pred_resampled > 0.5)
#         #stroke_pred_tmp = stroke_pred_resampled
#         stroke_pred_tmp = stroke_closing(stroke_pred_tmp)
#         stroke_pred_tmp = morphology.binary_fill_holes(stroke_pred_tmp)

#         return stroke_pred_tmp, stroke_pred

def get_stroke_seg_MNI(model, #model: DAGMNet,
                      dwi_img: np.ndarray, 
                      adc_img: np.ndarray, 
                      Prob_IS: Optional[np.ndarray] = None, 
                      N_channel: int = 3,
                      DS: int = 2,
                      device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs stroke segmentation using DAGMNet model with multi-scale inference.
    
    Args:
        model: DAGMNet PyTorch model
        dwi_img: Diffusion weighted image array
        adc_img: ADC image array  
        Prob_IS: Initial stroke probability map (optional)
        N_channel: Number of input channels (2 or 3)
        DS: Downsampling factor for multi-scale inference
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (binary stroke prediction mask, raw probability map)
    """
    # Ensure model is on correct device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    stroke_pred_resampled = np.zeros_like(dwi_img)
    
    # Multi-scale inference by offsetting starting points
    for x_idx, y_idx, slice_idx in [(x,y,z) for x in range(DS) for y in range(DS) for z in range(2*DS)]:
        # Extract downsampled views
        dwi_DS_img = dwi_img[x_idx::DS, y_idx::DS, slice_idx::2*DS]
        adc_DS_img = adc_img[x_idx::DS, y_idx::DS, slice_idx::2*DS]
        
        if N_channel == 3:
            if Prob_IS is None:
                raise ValueError("Prob_IS is required for 3-channel input")
            prob_IS_DS = Prob_IS[x_idx::DS, y_idx::DS, slice_idx::2*DS]
            input_tensor = torch.stack([
                torch.FloatTensor(dwi_DS_img),
                torch.FloatTensor(adc_DS_img),
                torch.FloatTensor(prob_IS_DS)
            ]).unsqueeze(0)
        else:
            input_tensor = torch.stack([
                torch.FloatTensor(dwi_DS_img),
                torch.FloatTensor(adc_DS_img)
            ]).unsqueeze(0)
        
        # Move input to same device as model
        input_tensor = input_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output_fused, _, _, _, _ = model(input_tensor)
            # Move output back to CPU for numpy operations
            stroke_pred = output_fused.squeeze().cpu().numpy()
           
        # Accumulate predictions
        stroke_pred_resampled[x_idx::DS, y_idx::DS, slice_idx::2*DS] = stroke_pred

    # Post-processing
    stroke_pred_tmp = (stroke_pred_resampled > 0.5)
    stroke_pred_tmp = stroke_closing(stroke_pred_tmp)
    stroke_pred_tmp = morphology.binary_fill_holes(stroke_pred_tmp)

    return stroke_pred_tmp, stroke_pred_resampled


def seg_postprocess(prediction: np.ndarray, 
                mask_img: Optional[np.ndarray] = None,
                min_size: int = 5) -> np.ndarray:
    """
    Apply post-processing to prediction
    
    Args:
        prediction: Raw prediction array
        mask_img: Brain mask for constraining predictions
        min_size: Minimum size of objects to keep
    
    Returns:
        Post-processed binary prediction
    """
    from skimage.morphology import remove_small_objects
    # Convert to binary
    binary_pred = (prediction > 0.35).astype(np.uint8) 
    
    # Apply morphological operations
    processed = stroke_closing(binary_pred)
    #processed = morphology.binary_fill_holes(processed)
    processed = remove_small_objects(processed.astype(bool), min_size=min_size).astype(prediction.dtype)
    
    # Apply mask if provided
    if mask_img is not None:
        processed = processed & (mask_img > 0.5)
        
    return processed
    
