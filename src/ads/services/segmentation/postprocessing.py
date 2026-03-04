"""Post-processing service for segmentation outputs.

Consolidates post-processing logic from:
- pipe_segment.py:318-347
- legacy DAGMNet PWI inference implementation (85-102 in original code)
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes
from ads.domain.segmentation_spec import PostProcessingSpec
from ads.core.preprocessing import (
    remove_small_objects_in_slice,
    stroke_closing
)


class PostProcessingService:
    """
    Unified post-processing service.

    Wraps existing post-processing functions to maintain compatibility.
    """

    def __init__(self, spec: PostProcessingSpec):
        self.spec = spec

    def process(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply full post-processing pipeline.

        This implements the exact logic from pipe_segment.py:342-347
        to maintain numerical equivalence.

        Steps:
        1. Remove small objects (slice-wise)
        2. Binary closing
        3. Fill holes

        Args:
            binary_mask: Binary prediction mask

        Returns:
            Post-processed binary mask
        """
        if not self.spec.apply_postprocessing:
            return binary_mask

        result = binary_mask.copy()

        # Step 1: Remove small objects (exact logic from pipe_segment.py:344)
        if self.spec.remove_small_objects:
            # Note: remove_small_objects_in_slice uses 'min_size' parameter
            from ads.core.preprocessing import remove_small_objects_in_slice as remove_objects
            result = remove_objects(result, min_size=self.spec.min_object_size)

        # Step 2: Binary closing (exact logic from pipe_segment.py:345)
        result = stroke_closing(result)

        # Step 3: Fill holes (exact logic from pipe_segment.py:346)
        result = binary_fill_holes(result).astype(np.uint8)

        return result
