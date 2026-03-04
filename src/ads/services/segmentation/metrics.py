"""Metrics computation service.

Consolidates all metrics computation logic from:
- pipe_segment.py:292-315 (compute_metrics)
- core/metrics.py
"""
from typing import Optional
import numpy as np
from ads.domain.segmentation_data import MetricsResult


class MetricsService:
    """
    Unified metrics computation service.

    Wraps existing metrics implementations without modification.
    """

    @staticmethod
    def compute(
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray],
        threshold: float = 0.5
    ) -> MetricsResult:
        """
        Compute segmentation metrics.

        This wraps the original compute_metrics logic from pipe_segment.py:292-315
        to maintain numerical equivalence.

        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth mask (optional)
            threshold: Binarization threshold

        Returns:
            MetricsResult with computed metrics
        """
        # Handle tuple/list predictions (from some models)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]

        pred_bin = (prediction > threshold).astype(np.float32)

        if ground_truth is None:
            # No ground truth - return minimal metrics
            return MetricsResult(
                dice=0.0,
                precision=0.0,
                sensitivity=0.0,
                sdr=1.0 if pred_bin.sum() > 0 else 0.0,
                pred_volume=float(pred_bin.sum()),
                true_volume=None
            )

        tgt_bin = (ground_truth > threshold).astype(np.float32)

        # Compute standard metrics (exact logic from pipe_segment.py:299-306)
        tp = (pred_bin * tgt_bin).sum()
        fp = (pred_bin * (1 - tgt_bin)).sum()
        fn = ((1 - pred_bin) * tgt_bin).sum()

        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        prec = (tp + 1e-6) / (tp + fp + 1e-6) if tp + fp > 0 else 1.0
        sens = (tp + 1e-6) / (tp + fn + 1e-6) if tp + fn > 0 else 1.0
        sdr = 1.0 if pred_bin.sum() > 0 else 0.0

        return MetricsResult(
            dice=float(dice),
            precision=float(prec),
            sensitivity=float(sens),
            sdr=float(sdr),
            pred_volume=float(pred_bin.sum()),
            true_volume=float(tgt_bin.sum())
        )

    @staticmethod
    def compute_with_postproc(
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray],
        postproc_fn,
        threshold: float = 0.5
    ) -> MetricsResult:
        """
        Compute metrics after applying post-processing.

        Args:
            prediction: Raw prediction
            ground_truth: Ground truth mask
            postproc_fn: Post-processing function to apply
            threshold: Binarization threshold

        Returns:
            MetricsResult after post-processing
        """
        pred_postproc = postproc_fn((prediction > threshold).astype(np.uint8))
        return MetricsService.compute(pred_postproc, ground_truth, threshold)
