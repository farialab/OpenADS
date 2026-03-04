"""Spatial transformation utilities for NIfTI images.

Handles orientation fixing, padding, and other spatial operations.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union, Optional
import time

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class OrientationFixer:
    """Handles orientation corrections for medical images."""

    @staticmethod
    def align_stroke_to_dwi(
        stroke_path: Union[Path, str],
        dwi_path_4d: Union[Path, str],
        output_dir: Path,
        subject_id: str
    ) -> Path:
        """Fix stroke label orientation to match DWI.

        Args:
            stroke_path: Path to stroke mask
            dwi_path_4d: Path to 4D DWI image (reference)
            output_dir: Output directory
            subject_id: Subject identifier

        Returns:
            Path to fixed stroke mask

        Note:
            Equivalent to `fix_stroke_orientation()` from data_prep.py
        """
        stroke_fixed_path = output_dir / f"{subject_id}_stroke_fixed.nii.gz"

        ref = nib.as_closest_canonical(nib.load(str(dwi_path_4d)))
        stroke_data = nib.as_closest_canonical(nib.load(str(stroke_path))).get_fdata().squeeze()

        stroke = nib.Nifti1Image(
            stroke_data.astype(np.uint8),
            affine=ref.affine,
            header=ref.header
        )
        stroke.header.set_data_dtype(np.uint8)
        stroke.header['glmax'] = 1

        nib.save(nib.as_closest_canonical(stroke), stroke_fixed_path)
        time.sleep(3)  # Preserve original behavior (file system sync?)
        return stroke_fixed_path


class SpatialTransformer:
    """Handles spatial transformations like padding and cropping."""

    @staticmethod
    def pad_to_size(
        img: np.ndarray,
        target_shape: tuple[int, int, int] = (192, 224, 192)
    ) -> np.ndarray:
        """Pad image to target shape by centering.

        Args:
            img: Input image array
            target_shape: Target shape (default: (192, 224, 192))

        Returns:
            Padded image array

        Note:
            Used to match registered images (182,218,182) to template (192,224,192)

        """
        current_shape = img.shape
        padded = np.zeros(target_shape, dtype=img.dtype)

        # Calculate padding offsets to center the image
        x_offset = (target_shape[0] - current_shape[0]) // 2
        y_offset = (target_shape[1] - current_shape[1]) // 2
        z_offset = (target_shape[2] - current_shape[2]) // 2

        # Place the image into the padded array
        padded[
            x_offset:x_offset+current_shape[0],
            y_offset:y_offset+current_shape[1],
            z_offset:z_offset+current_shape[2]
        ] = img

        return padded

    @staticmethod
    def depad_to_size(
        img: np.ndarray,
        target_shape: tuple[int, int, int] = (182, 218, 182)
    ) -> np.ndarray:
        """Remove padding to restore original size.

        Args:
            img: Padded image array
            target_shape: Target shape (default: (182, 218, 182))

        Returns:
            Depadded image array

        Note:
            Inverse operation of pad_to_size()

        """
        current_shape = img.shape
        depadded = img[
            (current_shape[0] - target_shape[0]) // 2:(current_shape[0] + target_shape[0]) // 2,
            (current_shape[1] - target_shape[1]) // 2:(current_shape[1] + target_shape[1]) // 2,
            (current_shape[2] - target_shape[2]) // 2:(current_shape[2] + target_shape[2]) // 2
        ]
        return depadded
