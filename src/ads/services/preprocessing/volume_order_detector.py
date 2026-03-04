"""Volume order detection service.

Detects and corrects the order of DWI and B0 volumes in 4D images
based on brightness and gradient variance heuristics.

This is algorithm logic that was previously embedded in the pipeline.
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VolumeOrderDetector:
    """Detects whether DWI and B0 volumes are in the expected order.

    This class implements heuristics to determine if a 4D image has its
    B0 and DWI volumes swapped. The detection is based on:
    1. Brightness: B0 should be brighter than DWI
    2. Gradient variance: DWI should have higher gradient variance (sharper edges)
    """

    def __init__(
        self,
        brightness_weight: float = 1.0,
        gradient_weight: float = 0.9,
        votes_threshold: int = 2
    ):
        """Initialize volume order detector.

        Args:
            brightness_weight: Weight for brightness comparison
            gradient_weight: Weight for gradient variance comparison
            votes_threshold: Number of votes needed to declare swap
        """
        self.brightness_weight = brightness_weight
        self.gradient_weight = gradient_weight
        self.votes_threshold = votes_threshold

    def is_order_swapped(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray
    ) -> bool:
        """Determine if two volumes are in swapped order.

        Args:
            vol0: First volume (assumed to be B0 if order is correct)
            vol1: Second volume (assumed to be DWI if order is correct)

        Returns:
            True if volumes appear to be swapped, False otherwise

        Note:
            Uses voting system based on:
            - Median brightness (B0 should be brighter)
            - Gradient variance (DWI should have sharper edges)
        """
        try:
            # Create a combined average to establish intensity range
            v_avg = (vol0.astype(np.float32) + vol1.astype(np.float32)) / 2.0
            nz = v_avg[v_avg > 0]

            if nz.size == 0:
                logger.warning("No non-zero voxels found, assuming correct order")
                return False

            # Robust intensity range
            lo, hi = np.percentile(nz, [20.0, 99.9])
            mask = (v_avg > lo) & (v_avg < hi)

            # Brightness comparison
            med0 = float(np.median(vol0[mask] if mask.any() else vol0))
            med1 = float(np.median(vol1[mask] if mask.any() else vol1))
            brightness_vote = med0 < med1  # If vol0 is dimmer, they're likely swapped

            # Gradient variance comparison
            gv0 = self._compute_gradient_variance(vol0, mask)
            gv1 = self._compute_gradient_variance(vol1, mask)
            gradient_vote = gv0 > (self.gradient_weight * gv1)  # If vol0 is sharper, they're likely swapped

            # Voting
            votes = int(brightness_vote * self.brightness_weight) + int(gradient_vote)
            is_swapped = votes >= self.votes_threshold

            if is_swapped:
                logger.info(
                    f"Volume order appears swapped: "
                    f"brightness_vote={brightness_vote}, gradient_vote={gradient_vote}, "
                    f"med0={med0:.1f}, med1={med1:.1f}, gv0={gv0:.2f}, gv1={gv1:.2f}"
                )

            return is_swapped

        except Exception as e:
            logger.error(f"Error during order check, assuming correct order: {e}")
            return False

    def _compute_gradient_variance(
        self,
        vol: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """Compute gradient magnitude variance within mask.

        Args:
            vol: 3D volume
            mask: Binary mask indicating valid voxels

        Returns:
            Variance of gradient magnitude
        """
        gx, gy, gz = np.gradient(vol.astype(np.float32))
        g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        return float(g_mag[mask].var() if mask.any() else g_mag.var())

    def detect_and_split(
        self,
        vol0: np.ndarray,
        vol1: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect order and return (b0, dwi) in correct order.

        Args:
            vol0: First volume
            vol1: Second volume

        Returns:
            Tuple of (b0_data, dwi_data) in correct order
        """
        if self.is_order_swapped(vol0, vol1):
            logger.info("Correcting swapped volume order")
            return vol1, vol0  # Swap
        else:
            return vol0, vol1  # Keep original order
