"""Domain models for report data.

Pure data classes representing radiology reports, model predictions, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class LesionMetrics:
    """Lesion volume and spatial metrics.

    Attributes:
        total_ml: Total lesion volume in milliliters
        left_ml: Left hemisphere lesion volume in milliliters
        right_ml: Right hemisphere lesion volume in milliliters
        icv_ml: Intracranial volume in milliliters
        lesion_mask: Binary lesion mask (3D numpy array)
    """

    total_ml: float
    left_ml: float
    right_ml: float
    icv_ml: float
    lesion_mask: np.ndarray

    def __post_init__(self):
        """Validate metrics."""
        if self.total_ml < 0:
            raise ValueError(f"Total volume must be non-negative, got {self.total_ml}")
        if self.left_ml < 0:
            raise ValueError(f"Left volume must be non-negative, got {self.left_ml}")
        if self.right_ml < 0:
            raise ValueError(f"Right volume must be non-negative, got {self.right_ml}")
        if self.icv_ml <= 0:
            raise ValueError(f"ICV volume must be positive, got {self.icv_ml}")

        # Validate mask is 3D
        if not isinstance(self.lesion_mask, np.ndarray):
            raise TypeError(f"Lesion mask must be numpy array, got {type(self.lesion_mask)}")
        if self.lesion_mask.ndim != 3:
            raise ValueError(f"Lesion mask must be 3D, got shape {self.lesion_mask.shape}")

    @property
    def lesion_to_icv_ratio(self) -> float:
        """Calculate lesion volume as percentage of ICV."""
        return (self.total_ml / self.icv_ml) * 100 if self.icv_ml > 0 else 0.0


@dataclass
class AAModelPrediction:
    """AA (Anterior/Posterior) model prediction for one ROI.

    Represents the output of a random forest or other classifier predicting
    whether a specific anatomical region is affected.

    Attributes:
        roi: Region of interest name (e.g., "ACA", "M1", "frontal")
        prediction: Binary prediction (0 = not affected, 1 = affected)
        probability: Probability of positive class (optional, model-dependent)
        qfv: QFV value for this ROI (input feature to model)
    """

    roi: str
    prediction: int
    qfv: float
    probability: Optional[float] = None

    def __post_init__(self):
        """Validate prediction."""
        if self.prediction not in [0, 1]:
            raise ValueError(f"Prediction must be 0 or 1, got {self.prediction}")
        if self.probability is not None:
            if not (0 <= self.probability <= 1):
                raise ValueError(f"Probability must be in [0, 1], got {self.probability}")


@dataclass
class RadiologyReport:
    """Radiology report content and metadata.

    Contains all information needed to generate a clinical radiology report,
    including lesion metrics and AA model predictions.

    Attributes:
        subject_id: Subject identifier
        lesion_metrics: Lesion volume and spatial metrics
        vascular_predictions: Vascular territory predictions (e.g., ACA, MCA, PCA)
        lobe_predictions: Brain lobe predictions (e.g., frontal, parietal)
        hydro_prediction: Hydrocephalus prediction
        aspects_predictions: ASPECTS region predictions (e.g., M1-M6)
        aspectpc_predictions: PCA-ASPECTS predictions (optional, posterior circulation)
    """

    subject_id: str
    lesion_metrics: LesionMetrics
    vascular_predictions: List[AAModelPrediction]
    lobe_predictions: List[AAModelPrediction]
    hydro_prediction: AAModelPrediction
    aspects_predictions: List[AAModelPrediction]
    aspectpc_predictions: Optional[List[AAModelPrediction]] = None

    def __post_init__(self):
        """Validate report structure."""
        if not self.vascular_predictions:
            raise ValueError("Vascular predictions cannot be empty")
        if not self.lobe_predictions:
            raise ValueError("Lobe predictions cannot be empty")
        if not self.aspects_predictions:
            raise ValueError("ASPECTS predictions cannot be empty")

    def get_affected_regions(self, category: str) -> List[str]:
        """Get list of affected regions for a given category.

        Args:
            category: One of 'vascular', 'lobe', 'aspects', 'aspectpc'

        Returns:
            List of ROI names where prediction == 1
        """
        predictions_map = {
            "vascular": self.vascular_predictions,
            "lobe": self.lobe_predictions,
            "aspects": self.aspects_predictions,
            "aspectpc": self.aspectpc_predictions or [],
        }

        if category not in predictions_map:
            raise ValueError(f"Unknown category: {category}")

        predictions = predictions_map[category]
        return [p.roi for p in predictions if p.prediction == 1]


@dataclass
class InterpretationReport:
    """SHAP interpretation report metadata.

    Attributes:
        subject_id: Subject identifier
        aa_type: AA model type ("vascular", "lobe", "aspects", "aspectpc")
        pdf_path: Path to generated interpretation PDF
        roi_list: List of ROI names in this interpretation
    """

    subject_id: str
    aa_type: str
    pdf_path: Path
    roi_list: List[str]

    def __post_init__(self):
        """Validate interpretation report."""
        valid_types = ["vascular", "lobe", "aspects", "aspectpc", "hydro"]
        if self.aa_type not in valid_types:
            raise ValueError(f"AA type must be one of {valid_types}, got {self.aa_type}")

        if not self.roi_list:
            raise ValueError("ROI list cannot be empty")

        # Convert pdf_path to Path if needed
        if not isinstance(self.pdf_path, Path):
            self.pdf_path = Path(self.pdf_path)
