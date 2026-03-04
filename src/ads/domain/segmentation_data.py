"""Domain models for segmentation module.

Pure data objects with no business logic or I/O operations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


@dataclass(frozen=True)
class SegmentationInputs:
    """Input files for segmentation pipeline."""

    # MNI space inputs (required)
    dwi_mni: Path
    adc_mni: Path
    mask_mni: Path

    # Native space inputs (required for transform back)
    dwi_native: Path
    mask_native: Path

    # Optional ground truth
    stroke_mni: Optional[Path] = None
    stroke_native: Optional[Path] = None

    # Registration transforms (for transform back to native)
    fwd_affine: Optional[Path] = None  # individual2MNI152

    # SyN registration (for affsyn space transform)
    adc_affsyn: Optional[Path] = None
    syn_affine: Optional[Path] = None
    syn_warp: Optional[Path] = None


@dataclass
class ModelInputData:
    """Prepared data ready for model inference.

    This is mutable to allow incremental preparation.
    """
    dwi_normalized: np.ndarray
    adc_normalized: np.ndarray
    mask: np.ndarray
    prob_is: Optional[np.ndarray] = None
    original_shape: Optional[tuple] = None


@dataclass(frozen=True)
class SegmentationOutputs:
    """Segmentation pipeline outputs."""
    pred_mni: Path
    pred_native: Path
    metrics_json: Path
    pred_mni_affsyn: Optional[Path] = None
    dwi_normalized_path: Optional[Path] = None


@dataclass(frozen=True)
class PWISegmentationInputs:
    """Input files for PWI segmentation pipeline."""

    # MNI space inputs (required)
    dwi_mni: Path
    adc_mni: Path
    ttp_mni: Path  # TTP (Time to Peak) for PWI
    mask_mni: Path

    # Native space inputs (required for transform back)
    dwi_native: Path
    mask_native: Path

    # Optional ground truth (HP = Hypoperfusion)
    hp_mni: Optional[Path] = None
    hp_native: Optional[Path] = None

    # Optional stroke channel (for 4-channel model)
    stroke_mni: Optional[Path] = None

    # Registration transforms (for transform back to native)
    fwd_affine: Optional[Path] = None


@dataclass(frozen=True)
class MetricsResult:
    """Segmentation evaluation metrics.

    Compatible with existing metrics.json format.
    """
    dice: float
    precision: float
    sensitivity: float
    sdr: float  # Segmentation detection rate
    pred_volume: float
    true_volume: Optional[float] = None

    # Optional additional metrics
    specificity: Optional[float] = None
    hausdorff: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'dice': float(self.dice),
            'precision': float(self.precision),
            'sensitivity': float(self.sensitivity),
            'sdr': float(self.sdr),
            'pred_volume': float(self.pred_volume),
        }

        if self.true_volume is not None:
            result['true_volume'] = float(self.true_volume)
        if self.specificity is not None:
            result['specificity'] = float(self.specificity)
        if self.hausdorff is not None:
            result['hausdorff'] = float(self.hausdorff)

        return result
