"""
Morphological operations for stroke segmentation.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import morphology as skmorph

logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """Performs morphological operations on stroke predictions."""

    @staticmethod
    def closing(img: np.ndarray, structure_size: tuple = (2, 2, 2)) -> np.ndarray:
        """Perform morphological closing on binary image.

        Args:
            img: Input binary image
            structure_size: Size of structuring element

        Returns:
            Closed binary image

        Note:
            Equivalent to stroke_closing() from preprocessing.py
        """
        return ndimage.binary_closing(img, structure=np.ones(structure_size))

    @staticmethod
    def connected(img: np.ndarray, connect_radius: int = 1) -> np.ndarray:
        """Dilate image to connect nearby regions.

        Args:
            img: Input binary image
            connect_radius: Radius for dilation

        Returns:
            Dilated binary image

        Note:
            Equivalent to stroke_connected() from preprocessing.py
        """
        return skmorph.binary_dilation(img, skmorph.ball(radius=connect_radius))

    @staticmethod
    def remove_small_objects(
        img: np.ndarray,
        min_size: int = 5,
        structure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Remove small objects from binary image.

        Args:
            img: Input binary image
            min_size: Minimum size of objects to keep
            structure: Structure for connected component analysis

        Returns:
            Filtered binary image

        Note:
            Equivalent to remove_small_objects() from preprocessing.py
        """
        if structure is None:
            structure = np.ones((3, 3))

        # Convert to binary
        binary = img.copy()
        binary[binary > 0] = 1

        # Label connected components
        label_result = ndimage.label(binary, structure=structure)
        labels = label_result[0]
        unique_labels = np.unique(labels)

        # Count voxels for each label
        labels_num = [np.sum(labels == label) for label in unique_labels]

        # Filter out small objects
        new_img = img.copy()
        for index, label in enumerate(unique_labels):
            if labels_num[index] < min_size:
                new_img[labels == label] = 0

        return new_img

    @staticmethod
    def remove_small_objects_in_slice(
        img: np.ndarray,
        min_size: int = 5,
        structure: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Remove small objects in each slice of a 3D volume.

        Args:
            img: Input 3D binary image
            min_size: Minimum size of objects to keep
            structure: Structure for connected component analysis

        Returns:
            Filtered 3D binary image

        Note:
            Equivalent to remove_small_objects_in_slice() from preprocessing.py
        """
        if structure is None:
            structure = np.ones((3, 3))

        img = np.squeeze(img)
        new_img = np.zeros_like(img)

        for idx in range(img.shape[-1]):
            new_img[:, :, idx] = MorphologyProcessor.remove_small_objects(
                img[:, :, idx],
                min_size=min_size,
                structure=structure
            )

        return new_img
