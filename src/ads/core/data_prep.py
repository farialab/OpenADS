
import logging
import os
import shutil
import time
import warnings  # For deprecation warnings
from pathlib import Path
from typing import Tuple

import ants
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


logger = logging.getLogger(__name__)

def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a 3D numpy array."""
    return np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())

def fix_stroke_orientation(stroke_path, dwi_path_4D, output_dir, subject_id):
    """Fix stroke label orientation to match DWI."""
    stroke_fixed_path = output_dir / f"{subject_id}_stroke_fixed.nii.gz"
    
    ref = nib.as_closest_canonical(nib.load(str(dwi_path_4D)))
    stroke_data = nib.as_closest_canonical(nib.load(str(stroke_path))).get_fdata().squeeze()
    
    stroke = nib.Nifti1Image(
        stroke_data.astype(np.uint8),
        affine=ref.affine,
        header=ref.header
    )
    stroke.header.set_data_dtype(np.uint8)
    stroke.header['glmax'] = 1
    
    nib.save(nib.as_closest_canonical(stroke), stroke_fixed_path)
    time.sleep(3)
    return stroke_fixed_path

def extract_volumes_from_4d(img_4d: nib.Nifti1Image, 
                           b0_index: int = 0, 
                           dwi_index: int = 1) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Extract B0 and DWI volumes from 4D image.
    
    Args:
        img_4d: 4D NIfTI image
        b0_index: Volume index for B0 image
        dwi_index: Volume index for DWI image
        
    Returns:
        Tuple of (b0_img, dwi_img)
    """
    if len(img_4d.shape) != 4:
        raise ValueError(f"Expected 4D image, got shape {img_4d.shape}")
    
    data_4d = img_4d.get_fdata()
    
    b0_data = data_4d[:, :, :, b0_index]
    dwi_data = data_4d[:, :, :, dwi_index]
    
    b0_img = nib.as_closest_canonical(
        nib.Nifti1Image(b0_data, affine=img_4d.affine)
    )
    dwi_img = nib.as_closest_canonical(
        nib.Nifti1Image(dwi_data, affine=img_4d.affine)
    )
    
    return b0_img, dwi_img

def load_nib_ras(source_img: nib.Nifti1Image, 
                         reference_img: nib.Nifti1Image,
                         target_orientation: str = "RAS") -> nib.Nifti1Image:
    """
    Fix image orientation to match reference or target orientation.
    
    Args:
        source_img: Image to reorient
        reference_img: Reference image for affine
        target_orientation: Target orientation (default: "RAS")
        
    Returns:
        Reoriented image
    """
    # Use reference affine and header
    fixed_img = nib.Nifti1Image(
        source_img.get_fdata().astype(source_img.get_data_dtype()),
        affine=reference_img.affine,
        header=reference_img.header.copy()
    )
    
    # Ensure target orientation
    return nib.as_closest_canonical(fixed_img)

def save_transformation_files(transformation, output_dir, subject_id, suffix=""):
    """Save transformation files with optional suffix."""
    transform_dir = output_dir
    for transform_file in transformation['fwdtransforms']:
        if 'GenericAffine.mat' in transform_file:
            target_path = transform_dir / f"{subject_id}_GenericAffine{suffix}.mat"
            shutil.copy(transform_file, str(target_path))
        elif 'Warp.nii.gz' in transform_file:
            target_path = transform_dir / f"{subject_id}_Warp{suffix}.nii.gz"
            shutil.copy(transform_file, str(target_path))

    for transform_file in transformation['invtransforms']:
        if 'GenericAffine.mat' in transform_file:
            target_path = transform_dir / f"{subject_id}_InverseGenericAffine{suffix}.mat"
            shutil.copy(transform_file, str(target_path))
        elif 'InverseWarp.nii.gz' in transform_file:
            target_path = transform_dir / f"{subject_id}_InverseWarp{suffix}.nii.gz"
            shutil.copy(transform_file, str(target_path))


def segment(path: Path | str, subject_id: str, description: str) -> bool:
    """Return True if path exists; log an error otherwise."""
    resolved = Path(path)
    if resolved.exists():
        return True

    logger.error("[%s] Missing %s: %s", subject_id, description, resolved)
    return False


def log_image_info(image: nib.Nifti1Image, label: str) -> None:
    """Log basic information about a NIfTI image for QC."""
    shape = image.shape
    spacing = image.header.get_zooms()[:3]
    orientation = "".join(aff2axcodes(image.affine))
    dtype = image.get_data_dtype()

    logger.info(
        "%s -> shape=%s | spacing=%s | orientation=%s | dtype=%s",
        label,
        shape,
        spacing,
        orientation,
        dtype,
    )


def create_binary_image(
    image: nib.Nifti1Image | Path | str,
    output_path: Path | str,
    threshold: float = 0.5,
) -> Path:
    """Save a binary mask derived from ``image`` to ``output_path``."""
    img = image if isinstance(image, nib.Nifti1Image) else nib.as_closest_canonical(nib.load(str(image)))
    data = img.get_fdata()
    binary = (data > threshold).astype(np.uint8)

    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    binary_img = nib.Nifti1Image(binary, affine=img.affine, header=header)
    nib.save(binary_img, str(output_path))

    logger.debug("Saved binary mask to %s", output_path)
    return Path(output_path)

def calculate_adc(dwi_img, b0_img, adc_path: str, bvalue: int = 1000):
    """Calculate ADC from DWI and B0 images.

    DEPRECATED: This function is deprecated. Use ads.services.preprocessing.ADCCalculator instead.

    This function is kept for backward compatibility and forwards to the new implementation.
    """
    warnings.warn(
        "ads.core.data_prep.calculate_adc is deprecated. "
        "Use ads.services.preprocessing.ADCCalculator instead.",
        DeprecationWarning,
        stacklevel=2
    )

    from ads.services.preprocessing import ADCCalculator

    calculator = ADCCalculator(bvalue=bvalue)

    # If adc_path is empty, return in-memory result
    if not adc_path or adc_path == "":
        return calculator.compute(dwi_img, b0_img, save_path=None)

    # Otherwise, save to disk and return the image
    calculator.compute(dwi_img, b0_img, save_path=adc_path)
    time.sleep(0.5)  # Preserve original behavior
    return nib.as_closest_canonical(nib.load(adc_path))

def check_file_exists(file_path: Path | str, subject_id: str, description: str) -> bool:
    """Check if a file exists and log an error if not."""
    path = Path(file_path)
    if path.exists():
        return True
    logger.error("[%s] Missing %s: %s", subject_id, description, path)
    return False