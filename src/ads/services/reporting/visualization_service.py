"""Visualization service for slice selection and contour finding.

Pure business logic for determining which slices to visualize.
No plotting or I/O - just slice selection algorithms.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ads.domain.reporting import SliceSelection


class VisualizationService:
    """
    Visualization logic without plotting.
    Determines which slices to visualize, finds contours, etc.
    """

    def select_slices(
        self,
        mask: np.ndarray,
        max_slices: int = 21,
        interval: int = 20,
        strategy: str = "combined",
    ) -> SliceSelection:
        """
        Select slices for visualization.

        Two strategies:
        1. "stroke_based": Select slices containing stroke (up to max_slices, centered)
        2. "regular_interval": Select every Nth slice
        3. "combined": Combine both strategies

        Args:
            mask: 3D binary mask (stroke or other ROI)
            max_slices: Maximum number of stroke-containing slices
            interval: Spacing for regular interval slices
            strategy: "stroke_based", "regular_interval", or "combined"

        Returns:
            SliceSelection domain object with sorted unique indices
        """
        depth = mask.shape[2]
        center = depth // 2

        # Strategy 1: Slices containing stroke
        stroke_slices = []
        if strategy in ["stroke_based", "combined"]:
            # Find slices with any stroke voxels
            has_stroke = np.any(mask > 0.5, axis=(0, 1))
            stroke_indices = np.where(has_stroke)[0].tolist()

            if stroke_indices:
                # Center the selection around middle of stroke extent
                stroke_center = len(stroke_indices) // 2
                half = max_slices // 2

                start_idx = max(0, stroke_center - half)
                end_idx = min(len(stroke_indices), stroke_center + half + 1)

                stroke_slices = stroke_indices[start_idx:end_idx]

        # Strategy 2: Regular interval slices
        interval_slices = []
        if strategy in ["regular_interval", "combined"]:
            interval_slices = list(range(0, depth, interval))

        # Combine strategies
        if strategy == "combined":
            all_slices = sorted(set(stroke_slices + interval_slices))
        elif strategy == "stroke_based":
            all_slices = sorted(stroke_slices) if stroke_slices else [center]
        else:  # regular_interval
            all_slices = sorted(interval_slices)

        # Ensure at least one slice
        if not all_slices:
            all_slices = [center]

        return SliceSelection(
            indices=all_slices,
            selection_strategy=strategy,
        )

    def find_stroke_extent(
        self,
        mask: np.ndarray,
        axis: int = 2,
    ) -> tuple[int, int]:
        """Find the extent of stroke along a given axis.

        Args:
            mask: 3D binary mask
            axis: Axis along which to find extent (0=x, 1=y, 2=z)

        Returns:
            Tuple of (min_idx, max_idx) along the axis
        """
        # Collapse other axes
        projection = np.any(mask > 0.5, axis=tuple(i for i in range(3) if i != axis))

        # Find where projection is non-zero
        indices = np.where(projection)[0]

        if len(indices) == 0:
            # No stroke found, return middle
            mid = mask.shape[axis] // 2
            return (mid, mid)

        return (int(indices[0]), int(indices[-1]))

    def calculate_slice_metrics(
        self,
        mask: np.ndarray,
        slice_idx: int,
    ) -> dict:
        """Calculate metrics for a single slice.

        Args:
            mask: 3D binary mask
            slice_idx: Slice index to analyze

        Returns:
            Dictionary with metrics:
            - num_voxels: Number of positive voxels
            - centroid: (x, y) centroid of mask
            - has_content: Whether slice has any positive voxels
        """
        slice_2d = mask[:, :, slice_idx]

        num_voxels = int(np.sum(slice_2d > 0.5))
        has_content = num_voxels > 0

        # Calculate centroid
        if has_content:
            y_coords, x_coords = np.where(slice_2d > 0.5)
            centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))
        else:
            centroid = (0.0, 0.0)

        return {
            "num_voxels": num_voxels,
            "centroid": centroid,
            "has_content": has_content,
        }

    def auto_select_best_slices(
        self,
        mask: np.ndarray,
        num_slices: int = 9,
    ) -> SliceSelection:
        """Automatically select best slices to show based on stroke content.

        Selects slices with most stroke content, evenly distributed.

        Args:
            mask: 3D binary mask
            num_slices: Number of slices to select

        Returns:
            SliceSelection with best slices
        """
        depth = mask.shape[2]

        # Calculate stroke content per slice
        slice_scores = []
        for idx in range(depth):
            metrics = self.calculate_slice_metrics(mask, idx)
            slice_scores.append((idx, metrics["num_voxels"]))

        # Sort by content (descending)
        slice_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top N slices
        selected_indices = [idx for idx, _ in slice_scores[:num_slices]]

        # Sort indices for display
        selected_indices.sort()

        return SliceSelection(
            indices=selected_indices,
            selection_strategy="auto_best_content",
        )
