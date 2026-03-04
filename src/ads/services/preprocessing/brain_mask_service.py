"""Brain mask processing service.

Encapsulates brain mask generation, binarization, and application logic.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict, List

import nibabel as nib
import numpy as np

from ads.core.brainmask import generate_brain_mask_with_synthstrip

logger = logging.getLogger(__name__)


class BrainMaskService:
    """Service for brain mask generation and application.

    This service encapsulates:
    - Brain mask generation via SynthStrip
    - Mask binarization
    - Skull stripping (applying mask to images)
    """

    def __init__(
        self,
        use_gpu: bool = True,
        no_csf: bool = False,
        model_path: Optional[Path] = None
    ):
        """Initialize brain mask service.

        Args:
            use_gpu: Whether to use GPU for SynthStrip
            no_csf: Whether to exclude CSF from brain boundary
            model_path: Optional path to custom SynthStrip model
        """
        self.use_gpu = use_gpu
        self.no_csf = no_csf
        self.model_path = model_path

    def generate_mask(
        self,
        input_image: nib.Nifti1Image,
        output_raw_path: Path
    ) -> nib.Nifti1Image:
        """Generate brain mask using SynthStrip.

        Args:
            input_image: Input DWI/PWI image
            output_raw_path: Path to save raw mask output

        Returns:
            Raw mask image from SynthStrip

        Note:
            This delegates to the SynthStrip model in ads.core.brainmask
        """
        # Save temp input for SynthStrip (it expects file paths)
        temp_input = output_raw_path.parent / f"_temp_input_{output_raw_path.stem}.nii.gz"
        nib.save(input_image, str(temp_input))

        try:
            # Call SynthStrip
            mask_volume = generate_brain_mask_with_synthstrip(
                str(temp_input),
                str(output_raw_path),
                use_gpu=self.use_gpu,
                no_csf=self.no_csf,
                model_path=str(self.model_path) if self.model_path else None,
            )

            # Load and return as NIfTI
            mask_img = nib.as_closest_canonical(nib.load(str(output_raw_path)))
            return mask_img

        finally:
            # Cleanup temp file
            if temp_input.exists():
                temp_input.unlink()

    def binarize_mask(
        self,
        mask_img: nib.Nifti1Image,
        threshold: float = 0.5
    ) -> nib.Nifti1Image:
        """Convert mask to binary (0/1).

        Args:
            mask_img: Input mask image
            threshold: Threshold for binarization (default: 0.5)

        Returns:
            Binary mask image
        """
        mask_data = (mask_img.get_fdata() > threshold).astype(np.uint8)
        binary_mask = nib.Nifti1Image(
            mask_data,
            affine=mask_img.affine,
            header=mask_img.header
        )
        return binary_mask

    def apply_mask(
        self,
        image: nib.Nifti1Image,
        mask: nib.Nifti1Image,
        check_dimensions: bool = True
    ) -> Optional[nib.Nifti1Image]:
        """Apply brain mask to an image (skull stripping).

        Args:
            image: Input image to strip
            mask: Binary brain mask
            check_dimensions: Whether to check dimension compatibility

        Returns:
            Skull-stripped image, or None if dimensions incompatible

        Note:
            Handles both 3D and 4D images. For 4D, mask is broadcast
            across time dimension.
        """
        img_data = image.get_fdata()
        mask_data = (mask.get_fdata() > 0.5)  # Ensure binary

        # Check shape compatibility
        if check_dimensions and img_data.shape[:3] != mask_data.shape[:3]:
            logger.warning(
                f"Shape mismatch: Image {img_data.shape} vs Mask {mask_data.shape}. "
                "Skipping mask application."
            )
            return None

        # Apply mask with proper broadcasting
        if img_data.ndim == 4:
            # Expand mask to (X, Y, Z, 1) for broadcasting against (X, Y, Z, T)
            mask_expanded = mask_data[..., np.newaxis]
            stripped_data = np.where(mask_expanded, img_data, 0)
        else:
            # Standard 3D case
            stripped_data = np.where(mask_data, img_data, 0)

        # Create output image
        stripped_img = nib.Nifti1Image(
            stripped_data.astype(image.get_data_dtype()),
            affine=image.affine,
            header=image.header
        )

        return stripped_img

    def apply_mask_to_multiple(
        self,
        images: Dict[str, nib.Nifti1Image],
        mask: nib.Nifti1Image
    ) -> Dict[str, nib.Nifti1Image]:
        """Apply mask to multiple images.

        Args:
            images: Dictionary of {name: image} to process
            mask: Binary brain mask

        Returns:
            Dictionary of {name: stripped_image} for successfully processed images

        Note:
            Skips images with dimension mismatches.
        """
        results = {}

        for name, img in images.items():
            stripped = self.apply_mask(img, mask, check_dimensions=True)
            if stripped is not None:
                results[name] = stripped
            else:
                logger.warning(f"Skipped {name} due to dimension mismatch")

        return results
