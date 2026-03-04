import os
import warnings  # For deprecation warnings
from pathlib import Path
import numpy as np
import ants
import torch
from typing import Tuple, Optional, Union, Dict, Any, List
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import scipy.special
from scipy import ndimage
import scipy.stats

import nibabel as nib

from .logging import logger

__all__ = [
    'pad_to_size',
    'depad_to_size',
    'calculate_adc',
    'get_dwi_normalized',
    'get_pwi_normalized',
    'get_stroke_probability_map',
    'stroke_closing',
    'stroke_connected',
    'remove_small_objects',
    'remove_small_objects_in_slice',
    'prepare_model_input'
]

# Make your registered images (182,218,182) match the template (192,224,192)
def pad_to_size(img, target_shape=(192, 224, 192)):
    """DEPRECATED: Use ads.adapters.nifti.SpatialTransformer.pad_to_size instead."""
    warnings.warn(
        "ads.core.preprocessing.pad_to_size is deprecated. "
        "Use ads.adapters.nifti.SpatialTransformer.pad_to_size instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ads.adapters.nifti import SpatialTransformer
    return SpatialTransformer.pad_to_size(img, target_shape)

def depad_to_size(img, target_shape=(182, 218, 182)):
    """DEPRECATED: Use ads.adapters.nifti.SpatialTransformer.depad_to_size instead."""
    warnings.warn(
        "ads.core.preprocessing.depad_to_size is deprecated. "
        "Use ads.adapters.nifti.SpatialTransformer.depad_to_size instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ads.adapters.nifti import SpatialTransformer
    return SpatialTransformer.depad_to_size(img, target_shape)

def calculate_adc(
    dwi_path: os.PathLike | str,
    b0_path: os.PathLike | str,
    output_path: os.PathLike | str,
    bvalue: int = 1000,
) -> Path:
    """Calculate ADC map using nibabel (write to disk for downstream stages).

    DEPRECATED: This function is deprecated. Use ads.services.preprocessing.ADCCalculator instead.
    """
    warnings.warn(
        "ads.core.preprocessing.calculate_adc is deprecated. "
        "Use ads.services.preprocessing.ADCCalculator.compute_from_paths instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ads.services.preprocessing import ADCCalculator
    calculator = ADCCalculator(bvalue=bvalue)
    return calculator.compute_from_paths(dwi_path, b0_path, output_path)

def get_dwi_normalized(dwi_img, mask_img, percentile_low=1, percentile_high=99):
    """
    Normalize DWI image using robust scaling based on percentiles of intensities within the mask.

    DEPRECATED: Use ads.services.preprocessing.Normalizer instead.
    """
    warnings.warn(
        "ads.core.preprocessing.get_dwi_normalized is deprecated. "
        "Use ads.services.preprocessing.Normalizer.normalize_dwi instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ads.services.preprocessing import Normalizer
    normalizer = Normalizer(percentile_low=percentile_low, percentile_high=percentile_high)
    return normalizer.normalize_dwi(dwi_img, mask_img)


def get_pwi_normalized(
    dwi_mni: np.ndarray,
    adc_mni: np.ndarray,
    ttp_mni: np.ndarray,
    mask_mni: np.ndarray,
    stroke_mni: Optional[np.ndarray] = None,
    target_shape: Tuple[int, int, int] = (192, 224, 192),
    eps: float = 1e-6
) -> np.ndarray:
    """
    Normalize and prepare PWI inputs for model inference.

    This function implements the exact normalization logic from
    legacy DAGMNet PWI inference implementation to maintain numerical equivalence.

    Steps:
    1. Create brain mask from non-zero voxels
    2. Apply z-score normalization within mask to each channel
    3. Pad to target shape
    4. Stack channels together

    Args:
        dwi_mni: DWI image in MNI space
        adc_mni: ADC image in MNI space
        ttp_mni: TTP (Time to Peak) image in MNI space
        mask_mni: Brain mask in MNI space (not currently used, but kept for consistency)
        stroke_mni: Optional stroke mask for 4-channel mode
        target_shape: Target shape for padding (default: 192, 224, 192)
        eps: Epsilon for numerical stability

    Returns:
        Normalized and stacked array of shape [C, D, H, W] where C=3 or 4
    """
    # Collect channels
    channels = [dwi_mni, adc_mni, ttp_mni]
    if stroke_mni is not None:
        channels.append(stroke_mni)

    # Create brain mask from non-zero voxels (union of all channels)
    brain_mask = np.zeros_like(channels[0], dtype=bool)
    for ch in channels:
        brain_mask |= (ch != 0)

    # Z-score normalize each channel within the mask
    normalized = []
    for ch in channels:
        vals = ch[brain_mask > 0]
        if vals.size < 10:
            # Fall back to global statistics if mask too small
            m, s = ch.mean(), ch.std()
        else:
            m, s = float(vals.mean()), float(vals.std())

        s = s if s > eps else eps
        ch_norm = (ch - m) / s
        normalized.append(ch_norm)

    # Pad to target shape (center padding)
    padded = []
    for ch_norm in normalized:
        D, H, W = ch_norm.shape
        td, th, tw = target_shape
        pad = np.zeros(target_shape, dtype=ch_norm.dtype)
        sd = (td - D) // 2
        sh = (th - H) // 2
        sw = (tw - W) // 2
        pad[sd:sd + D, sh:sh + H, sw:sw + W] = ch_norm
        padded.append(pad)

    # Stack channels [C, D, H, W]
    return np.stack(padded, axis=0)


def get_stroke_probability_map(dwi_norm,
                     adc_img,
                     mask_img,
                     template_dir,
                     model_vars=[2, 1.5, 4, 0.5, 2, 2]):
    """
    Calculate probability map for ischemic stroke using ANTs images.
    
    Parameters:
    -----------
    dwi_norm : ants image
        Normalized DWI image in MNI space
    adc_img : ants image
        ADC image in MNI space
    mask_img : ants image
        Brain mask in MNI space
    template_dir : str
        Directory containing template images
    model_vars : list
        Model parameters [fwhm, alpha_dwi, lambda_dwi, alpha_adc, lambda_adc, id_isch_zth]
        
    Returns:
    --------
    ants image
        Probability map for ischemic stroke
    """
    if isinstance(dwi_norm, ants.ANTsImage):
        dwi_norm = dwi_norm.numpy()
    if isinstance(adc_img, ants.ANTsImage):
        adc_img = adc_img.numpy()
    if isinstance(mask_img, ants.ANTsImage):
        mask_img = mask_img.numpy() > 0.49

    # Helper functions
    # Q-function (tail probability of normal distribution)
    def qfunc(x):
        return 0.5 - 0.5 * scipy.special.erf(x/np.sqrt(2))
    
    def apply_gaussian_filter(img_data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply 3D Gaussian filter slice by slice"""
        filtered = np.zeros_like(img_data)
        for i in range(img_data.shape[-1]):
            filtered[:,:,i] = gaussian_filter(img_data[:,:,i], sigma)
        return filtered
    
    def process_dissimilarity(img_data: np.ndarray, 
                            template_mu: np.ndarray,
                            template_std: np.ndarray,
                            alpha: float,
                            lambda_val: float,
                            is_dwi: bool) -> np.ndarray:
        """Calculate dissimilarity map"""
        # Normalize input
        img_norm = (img_data - np.mean(img_data))/np.std(img_data)
        
        # Apply Gaussian filter
        g_sigma = model_vars[0]/2/np.sqrt(2*np.log(2))
        img_smooth = apply_gaussian_filter(img_norm, g_sigma)
        
        # Calculate dissimilarity
        dissim = np.tanh((img_smooth - template_mu)/template_std/alpha)
        
        if is_dwi:
            dissim[dissim < 0] = 0
            return dissim ** lambda_val
        else:
            dissim[dissim > 0] = 0
            return (-dissim) ** lambda_val
    
    # Load templates
    template_paths = {
        'dwi_mu': os.path.join(template_dir, 'normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz'),
        'dwi_std': os.path.join(template_dir, 'normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz'),
        'adc_mu': os.path.join(template_dir, 'normal_mu_ADC_Res_ss_MNI_normalized.nii.gz'),
        'adc_std': os.path.join(template_dir, 'normal_std_ADC_Res_ss_MNI_normalized.nii.gz')
    }
    
    templates = {k: ants.image_read(v).numpy() for k, v in template_paths.items()}
    
    # # Get numpy arrays from ANTs images
    # dwi_data = dwi_norm.numpy()
    # adc_data = adc_img.numpy()
    # mask_data = mask_img.numpy() > 0.49
    
    # Unpack model variables
    fwhm, alpha_dwi, lambda_dwi, alpha_adc, lambda_adc, id_isch_zth = model_vars
    
    # Calculate DWI dissimilarity
    dwi_h2 = process_dissimilarity(
        dwi_norm, 
        templates['dwi_mu'],
        templates['dwi_std'],
        alpha_dwi,
        lambda_dwi,
        is_dwi=True
    )
    dwi_h2[dwi_norm < id_isch_zth] = 0
    dwi_h2 = dwi_h2 * dwi_norm
    
    # Calculate ADC dissimilarity
    adc_h1 = process_dissimilarity(
        adc_img,
        templates['adc_mu'],
        templates['adc_std'],
        alpha_adc,
        lambda_adc,
        is_dwi=False
    )
    adc_h1 = adc_h1 * mask_img
    
    # Calculate ischemic indicator
    id_isch = (1 - qfunc(dwi_norm/id_isch_zth)) * (dwi_norm > id_isch_zth)
    
    # Calculate final probability map
    prob_is = dwi_h2 * adc_h1 * id_isch * mask_img
    
    return prob_is

def stroke_closing(img: np.ndarray) -> np.ndarray:
    """
    Perform morphological closing on stroke prediction image
    
    Args:
        img: Input binary image
    
    Returns:
        Closed binary image
    """
    return ndimage.binary_closing(img, structure=np.ones((2, 2, 2)))

def stroke_connected(img: np.ndarray, connect_radius: int = 1) -> np.ndarray:
    """
    Dilate stroke image to connect nearby regions
    
    Args:
        img: Input binary image
        connect_radius: Radius for dilation
    
    Returns:
        Dilated binary image
    """
    from skimage import morphology as skmorph
    return skmorph.binary_dilation(img, skmorph.ball(radius=connect_radius))

def remove_small_objects(img: np.ndarray, min_size: int = 5, structure: np.ndarray = np.ones((3, 3))) -> np.ndarray:
    """
    Remove small objects from binary image
    
    Args:
        img: Input binary image
        min_size: Minimum size of objects to keep
        structure: Structure for connected component analysis
    
    Returns:
        Filtered binary image
    """
    # Convert to binary
    binary = img.copy()
    binary[binary > 0] = 1
    
    # Label connected components
    label_result = ndimage.label(binary, structure=structure)
    labels = label_result[0]
    unique_labels = np.unique(labels)
    
    # Count voxels for each label
    labels_num = [np.sum(labels == label) for label in unique_labels]
    
    # Filter out small objects
    new_img = img.copy()
    for index, label in enumerate(unique_labels):
        if labels_num[index] < min_size:
            new_img[labels == label] = 0
            
    return new_img

def remove_small_objects_in_slice(img: np.ndarray, min_size: int = 5, structure: np.ndarray = np.ones((3, 3))) -> np.ndarray:
    """
    Remove small objects in each slice of a 3D volume
    
    Args:
        img: Input 3D binary image
        min_size: Minimum size of objects to keep
        structure: Structure for connected component analysis
    
    Returns:
        Filtered 3D binary image
    """
    img = np.squeeze(img)
    new_img = np.zeros_like(img)
    
    for idx in range(img.shape[-1]):
        new_img[:, :, idx] = remove_small_objects(
            img[:, :, idx], 
            min_size=min_size, 
            structure=structure
        )
        
    return new_img

def prepare_model_input(dwi_norm: np.ndarray,
                       adc_data: np.ndarray,
                       prob_is: Optional[np.ndarray] = None,
                       n_channel: int = 3) -> torch.Tensor:
    """
    Prepare input tensor for neural network model
    
    Args:
        dwi_norm: Normalized DWI data
        adc_data: ADC data
        prob_is: Ischemic stroke probability map (required for 3-channel input)
        n_channel: Number of input channels (2 or 3)
    
    Returns:
        PyTorch tensor ready for model input
    """
    if n_channel == 2:
        model_input = np.stack([dwi_norm, adc_data], axis=0)
    else:
        if prob_is is None:
            raise ValueError("prob_is required for 3-channel input")
        model_input = np.stack([dwi_norm, adc_data, prob_is], axis=0)
    
    return torch.FloatTensor(model_input).unsqueeze(0)
