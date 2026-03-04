"""NIfTI file saving utilities.

Provides consistent interfaces for saving NIfTI files.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class NiftiSaver:
    """Handles saving of NIfTI files with consistent conventions."""

    @staticmethod
    def save(
        img: nib.Nifti1Image,
        output_path: Union[Path, str],
        ensure_canonical: bool = True
    ) -> Path:
        """Save a NIfTI image to disk.

        Args:
            img: NIfTI image to save
            output_path: Output file path
            ensure_canonical: Convert to canonical orientation before saving

        Returns:
            Path to saved file

        Note:
            This standardizes the saving pattern used throughout the codebase.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if ensure_canonical:
            img = nib.as_closest_canonical(img)

        nib.save(img, str(output_path))
        logger.debug(f"Saved NIfTI image to {output_path}")

        return output_path

    @staticmethod
    def save_binary_mask(
        img: Union[nib.Nifti1Image, Path, str],
        output_path: Union[Path, str],
        threshold: float = 0.5
    ) -> Path:
        """Save a binary mask derived from an image.

        Args:
            img: Input image (NIfTI object or path)
            output_path: Output file path
            threshold: Threshold for binarization

        Returns:
            Path to saved file

        Note:
            Equivalent to `create_binary_image()` from data_prep.py
        """
        if isinstance(img, (Path, str)):
            img = nib.as_closest_canonical(nib.load(str(img)))

        data = img.get_fdata()
        binary = (data > threshold).astype(np.uint8)

        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        binary_img = nib.Nifti1Image(binary, affine=img.affine, header=header)

        return NiftiSaver.save(binary_img, output_path)

    @staticmethod
    def save_array_like(
        data: np.ndarray,
        reference_img: nib.Nifti1Image,
        output_path: Union[Path, str],
        dtype: np.dtype = np.float32,
    ) -> Path:
        """Save numpy array as NIfTI using affine/header from reference image."""
        header = reference_img.header.copy()
        header.set_data_dtype(dtype)
        out_img = nib.Nifti1Image(
            np.asarray(data, dtype=dtype),
            affine=reference_img.affine,
            header=header,
        )
        return NiftiSaver.save(out_img, output_path)
