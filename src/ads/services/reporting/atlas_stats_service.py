"""Atlas-based statistics service.

Pure business logic for calculating lesion statistics per atlas ROI.
No I/O - all operations on numpy arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class AtlasStatsService:
    """
    Calculate lesion statistics per atlas ROI.
    Takes numpy arrays as input, returns statistics.
    """

    def calculate_roi_counts(
        self,
        lesion_mask: np.ndarray,
        atlas_template: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Count lesion voxels per ROI label.

        Args:
            lesion_mask: 3D binary lesion mask (or probabilistic 0-1)
            atlas_template: 3D atlas with integer labels (1..max_label)
            threshold: Threshold for considering a voxel as lesion (default 0.5)

        Returns:
            1D array of counts, where index i contains count for ROI label (i+1)
            Length = max(atlas_template)
        """
        # Ensure arrays are numpy
        if not isinstance(lesion_mask, np.ndarray):
            lesion_mask = np.asarray(lesion_mask)
        if not isinstance(atlas_template, np.ndarray):
            atlas_template = np.asarray(atlas_template)

        # Get max label (number of ROIs)
        max_label = int(np.round(atlas_template.max()))

        # Initialize feature array
        counts = np.zeros(max_label, dtype=int)

        # Count lesion voxels per ROI (labels are 1..max_label)
        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (atlas_template == roi_label)
            # Count voxels where lesion is close to 1 (for binary or probabilistic masks)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0, atol=threshold)))

        return counts

    def calculate_roi_volumes(
        self,
        lesion_mask: np.ndarray,
        atlas_template: np.ndarray,
        voxel_size_mm: tuple[float, float, float],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Calculate lesion volume (in ml) per ROI.

        Args:
            lesion_mask: 3D binary lesion mask
            atlas_template: 3D atlas with integer labels (1..max_label)
            voxel_size_mm: Voxel dimensions in mm (x, y, z)
            threshold: Threshold for considering a voxel as lesion

        Returns:
            1D array of volumes in ml, where index i contains volume for ROI label (i+1)
        """
        counts = self.calculate_roi_counts(lesion_mask, atlas_template, threshold)

        # Convert counts to volumes
        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        volumes_ml = counts * voxel_volume_ml

        return volumes_ml

    def generate_lesion_stats(
        self,
        lesion_mask: np.ndarray,
        voxel_size_mm: tuple[float, float, float],
        icv_ml: float,
        lesion_ml: float,
        atlas_templates: Dict[str, np.ndarray],
        roi_labels: Dict[str, List[str]],
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """Generate comprehensive lesion statistics across multiple atlases.

        Args:
            lesion_mask: 3D binary lesion mask
            voxel_size_mm: Voxel dimensions in mm
            icv_ml: Intracranial volume in ml
            lesion_ml: Total lesion volume in ml
            atlas_templates: Dict mapping atlas names to template arrays
            roi_labels: Dict mapping atlas names to lists of ROI label names
            threshold: Threshold for lesion detection

        Returns:
            Dictionary with structure:
            {
                "summary": {"icv_ml": ..., "lesion_ml": ..., "lesion_ratio": ...},
                "vascular": {"ACA": vol_ml, "MCA": vol_ml, ...},
                "lobe": {"frontal": vol_ml, ...},
                ...
            }
        """
        stats = {
            "summary": {
                "icv_ml": icv_ml,
                "lesion_ml": lesion_ml,
                "lesion_to_icv_ratio": (lesion_ml / icv_ml * 100) if icv_ml > 0 else 0.0,
            }
        }

        # Calculate per-atlas statistics
        for atlas_name, template in atlas_templates.items():
            if template is None:
                continue

            # Get ROI volumes
            volumes_ml = self.calculate_roi_volumes(
                lesion_mask, template, voxel_size_mm, threshold
            )

            # Get ROI labels for this atlas
            labels = roi_labels.get(atlas_name, [])

            # Create dict of ROI name -> volume
            atlas_stats = {}
            for i, label in enumerate(labels):
                if i < len(volumes_ml):
                    atlas_stats[label] = float(volumes_ml[i])

            stats[atlas_name] = atlas_stats

        return stats

    def calculate_roi_probabilities(
        self,
        roi_counts: np.ndarray,
        roi_volumes: np.ndarray,
    ) -> np.ndarray:
        """Calculate ROI probabilities (QFV) from counts and expected volumes.

        This is used in QFV calculation to normalize counts by expected ROI volumes.

        Args:
            roi_counts: Array of lesion voxel counts per ROI
            roi_volumes: Array of expected volumes per ROI (from lookup table)

        Returns:
            Array of probabilities (normalized by volume)
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = roi_counts / roi_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities
