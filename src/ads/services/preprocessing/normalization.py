"""
DWI normalization service.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class Normalizer:
    """Normalizes DWI images using robust scaling."""

    def __init__(
        self,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        use_bimodal_fitting: bool = True
    ):
        """Initialize normalizer.

        Args:
            percentile_low: Lower percentile for fallback normalization
            percentile_high: Upper percentile for fallback normalization
            use_bimodal_fitting: Attempt bimodal fitting before percentiles
        """
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.use_bimodal_fitting = use_bimodal_fitting

    def normalize_dwi(
        self,
        dwi_img: np.ndarray,
        mask_img: np.ndarray
    ) -> np.ndarray:
        """Normalize DWI image using robust scaling within mask.

        Args:
            dwi_img: Input DWI image array
            mask_img: Brain mask array (binary)

        Returns:
            Normalized DWI image with values between 0 and 1

        Note:
            Equivalent to get_dwi_normalized() from preprocessing.py
        """
        # Ensure mask is binary
        mask_bin = mask_img > 0.5

        # Check if mask is empty
        if np.sum(mask_bin) == 0:
            logger.warning("Empty mask provided to normalize_dwi")
            return np.zeros_like(dwi_img)

        # Extract values inside the mask
        values = dwi_img[mask_bin]

        # Check if we have valid values
        if len(values) == 0 or np.all(np.isnan(values)) or np.max(values) == np.min(values):
            logger.warning("No valid values found in masked region")
            return np.zeros_like(dwi_img)

        # Remove negative values
        values = values[values >= 0]

        # Determine bounds for normalization
        lower_bound, upper_bound = self._compute_bounds(values)

        # Clip and normalize the image
        normalized = np.zeros_like(dwi_img)
        normalized[mask_bin] = np.clip(dwi_img[mask_bin], lower_bound, upper_bound)

        # Scale to [0, 1]
        if upper_bound > lower_bound:
            normalized[mask_bin] = (
                (normalized[mask_bin] - lower_bound) / (upper_bound - lower_bound)
            )

        return normalized

    def _compute_bounds(self, values: np.ndarray) -> tuple[float, float]:
        """Compute normalization bounds using bimodal fitting or percentiles.

        Args:
            values: Array of intensity values

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.use_bimodal_fitting or len(values) <= 100:
            return self._compute_bounds_percentile(values)

        try:
            # Compute histogram
            hist, bin_edges = np.histogram(values, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Try to find two peaks
            density = gaussian_kde(values)
            density_values = density(bin_centers)
            peaks, _ = find_peaks(density_values)

            if len(peaks) >= 2:
                # Get the two highest peaks
                sorted_peaks = peaks[np.argsort(-density_values[peaks])]
                first_peak = bin_centers[sorted_peaks[0]]
                second_peak = bin_centers[sorted_peaks[1]]

                # Use these peaks for normalization
                lower_bound = min(first_peak, second_peak) / 2
                upper_bound = max(first_peak, second_peak) * 1.5
                return lower_bound, upper_bound

        except Exception as e:
            logger.debug(f"Bimodal fitting failed: {e}. Using percentiles.")

        # Fallback to percentiles
        return self._compute_bounds_percentile(values)

    def _compute_bounds_percentile(self, values: np.ndarray) -> tuple[float, float]:
        """Compute bounds using percentiles.

        Args:
            values: Array of intensity values

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        lower_bound = np.percentile(values, self.percentile_low)
        upper_bound = np.percentile(values, self.percentile_high)
        return lower_bound, upper_bound
