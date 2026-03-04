"""Normalization service for affine-registered DWI/ADC images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import logging

import numpy as np

from ads.adapters.nifti import NiftiLoader, NiftiSaver
from ads.services.preprocessing.normalization import Normalizer

logger = logging.getLogger(__name__)


class RegistrationNormalizationService:
    """Create normalized DWI/ADC files in registration output folder."""

    def __init__(self) -> None:
        self._normalizer = Normalizer()

    def normalize_affine_modalities(
        self,
        dwi_affine_path: Path,
        adc_affine_path: Path,
        mask_affine_path: Path,
        dwi_output_path: Path,
        adc_output_path: Path,
    ) -> Dict[str, Path]:
        """Normalize affine-space DWI/ADC with mask-based robust normalization + z-score."""
        dwi_img = NiftiLoader.load(dwi_affine_path)
        adc_img = NiftiLoader.load(adc_affine_path)
        mask_img = NiftiLoader.load(mask_affine_path)

        dwi_arr = dwi_img.get_fdata()
        adc_arr = adc_img.get_fdata()
        mask_arr = mask_img.get_fdata()

        dwi_norm = self._normalizer.normalize_dwi(dwi_arr, mask_arr)
        adc_norm = self._normalizer.normalize_dwi(adc_arr, mask_arr)

        dwi_z = self._zscore_within_mask(dwi_norm, mask_arr)
        adc_z = self._zscore_within_mask(adc_norm, mask_arr)

        saved_dwi = NiftiSaver.save_array_like(dwi_z, dwi_img, dwi_output_path, dtype=np.float32)
        saved_adc = NiftiSaver.save_array_like(adc_z, adc_img, adc_output_path, dtype=np.float32)

        logger.debug("Saved normalized affine DWI: %s", saved_dwi)
        logger.debug("Saved normalized affine ADC: %s", saved_adc)
        return {"dwi_normalized": saved_dwi, "adc_normalized": saved_adc}

    def normalize_single_modality(
        self,
        image_path: Path,
        mask_path: Path,
        output_path: Path,
    ) -> Path:
        """Normalize one modality with mask-based robust normalization + z-score."""
        img = NiftiLoader.load(image_path)
        mask_img = NiftiLoader.load(mask_path)

        data = img.get_fdata()
        mask = mask_img.get_fdata()

        norm = self._normalizer.normalize_dwi(data, mask)
        norm_z = self._zscore_within_mask(norm, mask)
        saved = NiftiSaver.save_array_like(norm_z, img, output_path, dtype=np.float32)
        logger.debug("Saved normalized modality: %s", saved)
        return saved

    @staticmethod
    def _zscore_within_mask(img: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Z-score within mask with safe fallback for empty or zero-std masks."""
        mask_bin = mask > threshold
        vals = img[mask_bin]
        if vals.size == 0:
            return np.zeros_like(img, dtype=np.float32)

        std = float(vals.std())
        if std <= 1e-8:
            return np.zeros_like(img, dtype=np.float32)

        mean = float(vals.mean())
        return ((img - mean) / std).astype(np.float32)
