"""NIfTI file loading utilities.

This module provides utilities for loading NIfTI files with consistent
orientation handling. 
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class NiftiLoader:
    """Handles loading of NIfTI files with consistent orientation."""

    @staticmethod
    def load(path: Union[Path, str]) -> nib.Nifti1Image:
        """Load a NIfTI file and return canonical orientation.

        Args:
            path: Path to NIfTI file

        Returns:
            NIfTI image in canonical (RAS+) orientation

        Note:
            This is equivalent to the old `nib.as_closest_canonical(nib.load(...))`
            pattern used throughout the codebase.
        """
        img = nib.load(str(path))
        return nib.as_closest_canonical(img)

    @staticmethod
    def load_data(path: Union[Path, str]) -> np.ndarray:
        """Load a NIfTI file and return data as numpy array.

        Args:
            path: Path to NIfTI file

        Returns:
            3D numpy array in canonical orientation

        Note:
            Equivalent to old `load_nifti()` from data_prep.py
        """
        img = NiftiLoader.load(path)
        return np.squeeze(img.get_fdata())

    @staticmethod
    def load_with_reference(
        source_path: Union[Path, str],
        reference_img: nib.Nifti1Image,
        target_orientation: str = "RAS"
    ) -> nib.Nifti1Image:
        """Load image and align to reference affine/header.

        Args:
            source_path: Path to source image
            reference_img: Reference image for affine/header
            target_orientation: Target orientation (default: "RAS")

        Returns:
            Image with reference affine/header in canonical orientation

        Note:
            Equivalent to old `load_nib_ras()` from data_prep.py
        """
        source_img = nib.load(str(source_path))

        # Use reference affine and header
        fixed_img = nib.Nifti1Image(
            source_img.get_fdata().astype(source_img.get_data_dtype()),
            affine=reference_img.affine,
            header=reference_img.header.copy()
        )

        # Ensure target orientation
        return nib.as_closest_canonical(fixed_img)

    @staticmethod
    def extract_volumes_from_4d(
        img_4d: nib.Nifti1Image,
        b0_index: int = 0,
        dwi_index: int = 1
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """Extract B0 and DWI volumes from 4D image.

        Args:
            img_4d: 4D NIfTI image
            b0_index: Volume index for B0 image
            dwi_index: Volume index for DWI image

        Returns:
            Tuple of (b0_img, dwi_img) in canonical orientation

        Raises:
            ValueError: If image is not 4D

        Note:
            Equivalent to `extract_volumes_from_4d()` from data_prep.py
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
