"""ADC (Apparent Diffusion Coefficient) calculation service.

This module consolidates ADC calculation logic from:
- ads.core.data_prep.calculate_adc (nibabel-based, saves to disk)
- ads.core.preprocessing.calculate_adc (path-based, returns Path)

The unified implementation supports both use cases.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union, Optional
import time

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class ADCCalculator:
    """Calculates ADC maps from DWI and B0 images.

    This class provides a unified interface for ADC calculation, supporting
    both in-memory computation and direct file I/O.
    """

    def __init__(self, bvalue: float = 1000.0, epsilon: float = 1e-10):
        """Initialize ADC calculator.

        Args:
            bvalue: B-value for DWI acquisition (default: 1000)
            epsilon: Small value to avoid division by zero
        """
        self.bvalue = float(bvalue)
        self.epsilon = epsilon

    def compute_from_images(
        self,
        dwi_img: nib.Nifti1Image,
        b0_img: nib.Nifti1Image
    ) -> nib.Nifti1Image:
        """Calculate ADC from DWI and B0 images (in-memory).

        Args:
            dwi_img: DWI (diffusion-weighted) image
            b0_img: B0 (baseline) image

        Returns:
            ADC image as NIfTI object

        Note:
            This implements the formula:
            ADC = -ln((DWI + ε) / (B0 + ε)) / b-value
            Only pixels where both DWI > 0 and B0 > 0 are calculated.

            Equivalent to the old data_prep.calculate_adc when used in-memory.
        """
        dwi_data = dwi_img.get_fdata().astype(np.float32)
        b0_data = b0_img.get_fdata().astype(np.float32)

        # Calculate valid pixels mask
        valid = (b0_data > 0) & (dwi_data > 0)

        # Initialize ADC array
        adc_data = np.zeros_like(dwi_data, dtype=np.float32)

        # Calculate ADC only for valid pixels
        adc_data[valid] = (
            -np.log((dwi_data[valid] + self.epsilon) / (b0_data[valid] + self.epsilon))
            / self.bvalue
        )

        # Create NIfTI image with same affine/header as DWI
        adc_img = nib.Nifti1Image(
            adc_data,
            affine=dwi_img.affine,
            header=dwi_img.header
        )

        return nib.as_closest_canonical(adc_img)

    def compute_from_paths(
        self,
        dwi_path: Union[Path, str],
        b0_path: Union[Path, str],
        output_path: Union[Path, str]
    ) -> Path:
        """Calculate ADC from file paths and save to disk.

        Args:
            dwi_path: Path to DWI image
            b0_path: Path to B0 image
            output_path: Path to save ADC image

        Returns:
            Path to saved ADC image

        Note:
            Equivalent to preprocessing.calculate_adc (path-based version)
        """
        # Load images in canonical orientation
        dwi_img = nib.as_closest_canonical(nib.load(str(dwi_path)))
        b0_img = nib.as_closest_canonical(nib.load(str(b0_path)))

        # Calculate ADC
        adc_img = self.compute_from_images(dwi_img, b0_img)

        # Save to disk
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(adc_img, str(output_path))

        logger.info(f"ADC image saved to {output_path}")
        return output_path

    def compute(
        self,
        dwi: Union[nib.Nifti1Image, Path, str],
        b0: Union[nib.Nifti1Image, Path, str],
        save_path: Optional[Union[Path, str]] = None
    ) -> Union[nib.Nifti1Image, Path]:
        """Unified ADC computation interface.

        Args:
            dwi: DWI image (NIfTI object or path)
            b0: B0 image (NIfTI object or path)
            save_path: Optional path to save result. If None, returns image.

        Returns:
            NIfTI image if save_path is None, otherwise Path to saved file

        Examples:
            # In-memory computation
            adc_img = calculator.compute(dwi_img, b0_img)

            # Direct file I/O
            adc_path = calculator.compute(
                "dwi.nii.gz",
                "b0.nii.gz",
                save_path="adc.nii.gz"
            )
        """
        # Load images if paths provided
        if isinstance(dwi, (Path, str)):
            dwi_img = nib.as_closest_canonical(nib.load(str(dwi)))
        else:
            dwi_img = dwi

        if isinstance(b0, (Path, str)):
            b0_img = nib.as_closest_canonical(nib.load(str(b0)))
        else:
            b0_img = b0

        # Calculate ADC
        adc_img = self.compute_from_images(dwi_img, b0_img)

        # Save if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(adc_img, str(save_path))
            logger.info(f"ADC image saved to {save_path}")
            time.sleep(0.5)  # Preserve original behavior from data_prep
            return save_path

        return adc_img
