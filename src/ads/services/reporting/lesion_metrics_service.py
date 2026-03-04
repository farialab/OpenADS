"""Lesion metrics calculation service.

Pure business logic for calculating lesion volumes and spatial metrics.
No I/O - all operations on numpy arrays.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ads.domain.reporting import LesionMetrics


class LesionMetricsService:
    """
    Calculate lesion volumes and spatial metrics.
    Takes numpy arrays as input, returns domain objects.
    """

    def voxel_volume_ml(self, voxel_size_mm: Tuple[float, float, float]) -> float:
        """Calculate voxel volume in milliliters.

        Args:
            voxel_size_mm: Voxel dimensions in mm (x, y, z)

        Returns:
            Voxel volume in milliliters
        """
        return float(voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0

    def compute_volumes_ml(
        self,
        lesion_arr: np.ndarray,
        voxel_size_mm: Tuple[float, float, float],
        threshold: float = 0.5,
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute total, left, and right hemisphere lesion volumes.

        Args:
            lesion_arr: 3D lesion mask array
            voxel_size_mm: Voxel dimensions in mm (x, y, z)
            threshold: Threshold for binarization (default 0.5)

        Returns:
            Tuple of (total_ml, left_ml, right_ml, lesion_bin)
            - total_ml: Total lesion volume in ml
            - left_ml: Left hemisphere volume in ml
            - right_ml: Right hemisphere volume in ml
            - lesion_bin: Binarized lesion mask (uint8)
        """
        vv_ml = self.voxel_volume_ml(voxel_size_mm)

        # Binarize lesion mask
        lesion_bin = (lesion_arr > threshold).astype(np.uint8)

        # Calculate total volume
        total_ml = float(lesion_bin.sum() * vv_ml)

        # Split at midline (RAS orientation assumed)
        x_mid = lesion_bin.shape[0] // 2
        left_ml = float(lesion_bin[:x_mid, :, :].sum() * vv_ml)
        right_ml = float(lesion_bin[x_mid:, :, :].sum() * vv_ml)

        return total_ml, left_ml, right_ml, lesion_bin

    def estimate_icv_ml(
        self,
        brain_mask_arr: np.ndarray,
        voxel_size_mm: Tuple[float, float, float],
        threshold: float = 0.5,
        fallback_ml: float = 1200.0,
    ) -> float:
        """
        Estimate intracranial volume from brain mask.

        Args:
            brain_mask_arr: 3D brain mask array (or None)
            voxel_size_mm: Voxel dimensions in mm (x, y, z)
            threshold: Threshold for binarization (default 0.5)
            fallback_ml: Fallback volume if mask is empty (default 1200.0 ml)

        Returns:
            Estimated ICV in milliliters
        """
        if brain_mask_arr is None:
            return fallback_ml

        vv_ml = self.voxel_volume_ml(voxel_size_mm)

        # Binarize mask
        mask = (brain_mask_arr > threshold).astype(np.uint8)

        # Calculate volume
        icv_ml = float(mask.sum() * vv_ml)

        # Use fallback if mask is empty
        if icv_ml <= 0:
            return fallback_ml

        return icv_ml

    def calculate_lesion_metrics(
        self,
        lesion_arr: np.ndarray,
        voxel_size_mm: Tuple[float, float, float],
        brain_mask_arr: np.ndarray = None,
        lesion_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> LesionMetrics:
        """Calculate complete lesion metrics and return domain object.

        High-level method that combines volume calculations and returns
        a LesionMetrics domain object.

        Args:
            lesion_arr: 3D lesion mask array
            voxel_size_mm: Voxel dimensions in mm (x, y, z)
            brain_mask_arr: 3D brain mask array (optional)
            lesion_threshold: Threshold for lesion binarization
            mask_threshold: Threshold for brain mask binarization

        Returns:
            LesionMetrics domain object with all volumes and mask
        """
        # Compute lesion volumes
        total_ml, left_ml, right_ml, lesion_bin = self.compute_volumes_ml(
            lesion_arr, voxel_size_mm, threshold=lesion_threshold
        )

        # Estimate ICV
        icv_ml = self.estimate_icv_ml(
            brain_mask_arr, voxel_size_mm, threshold=mask_threshold
        )

        # Return domain object
        return LesionMetrics(
            total_ml=total_ml,
            left_ml=left_ml,
            right_ml=right_ml,
            icv_ml=icv_ml,
            lesion_mask=lesion_bin,
        )
