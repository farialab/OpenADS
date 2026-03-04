"""Domain models for QFV (Quantified Feature Vector) data.

Pure data classes representing QFV calculation results and atlas-based features.
No business logic - just data structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AtlasFeatures:
    """Raw feature counts per ROI from atlas-based segmentation.

    This represents the intermediate output before QFV probability calculation.
    Each array contains counts/features for different ROI categories.

    Attributes:
        vascular: Vascular territory features (e.g., ACA, MCA, PCA)
        lobe: Brain lobe features (e.g., frontal, parietal, temporal)
        aspects: ASPECTS region features (e.g., M1-M6, caudate, lentiform)
        aspectpc: PCA-ASPECTS features (optional, posterior circulation)
        ventricles: Ventricular region features
        bpm: Baseline perfusion map features (optional)
        bmos: Baseline mean oxygen saturation (optional)
        bmis: Baseline mean ischemic severity (optional)
    """

    vascular: np.ndarray
    lobe: np.ndarray
    aspects: np.ndarray
    aspectpc: Optional[np.ndarray] = None
    ventricles: np.ndarray = None
    bpm: Optional[np.ndarray] = None
    bmos: Optional[np.ndarray] = None
    bmis: Optional[np.ndarray] = None


@dataclass
class QFVResult:
    """Quantified Feature Vectors - probability-based representation of stroke location.

    This is the final output of QFV calculation. Each QFV array contains probabilities
    (or normalized values) per ROI, indicating how much of the stroke affects that region.

    The QFV representation is used for:
    - AA (Anterior/Posterior) model predictions
    - Radiology report generation
    - Statistical analysis

    Attributes:
        subject_id: Subject identifier
        vascular_qfv: Vascular territory QFV (length = # vascular ROIs, typically 9)
        lobe_qfv: Brain lobe QFV (length = # lobe ROIs, typically 12)
        aspects_qfv: ASPECTS QFV (length = # ASPECTS ROIs, typically 10)
        aspectpc_qfv: PCA-ASPECTS QFV (optional, length = # posterior ROIs, typically 5)
        ventricles_qfv: Ventricular QFV (length = # ventricular ROIs, typically 20)
        bpm_qfv: Baseline perfusion map QFV (optional)
        lesion_volume_ml: Total lesion volume in milliliters
        icv_volume_ml: Intracranial volume in milliliters (for normalization)
    """

    subject_id: str
    vascular_qfv: np.ndarray
    lobe_qfv: np.ndarray
    aspects_qfv: np.ndarray
    ventricles_qfv: np.ndarray
    lesion_volume_ml: float
    icv_volume_ml: float
    aspectpc_qfv: Optional[np.ndarray] = None
    bpm_qfv: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate QFV arrays."""
        # Ensure QFV arrays are numpy arrays
        if not isinstance(self.vascular_qfv, np.ndarray):
            self.vascular_qfv = np.array(self.vascular_qfv, dtype=float)
        if not isinstance(self.lobe_qfv, np.ndarray):
            self.lobe_qfv = np.array(self.lobe_qfv, dtype=float)
        if not isinstance(self.aspects_qfv, np.ndarray):
            self.aspects_qfv = np.array(self.aspects_qfv, dtype=float)
        if not isinstance(self.ventricles_qfv, np.ndarray):
            self.ventricles_qfv = np.array(self.ventricles_qfv, dtype=float)

        # Validate volumes are non-negative
        if self.lesion_volume_ml < 0:
            raise ValueError(f"Lesion volume must be non-negative, got {self.lesion_volume_ml}")
        if self.icv_volume_ml <= 0:
            raise ValueError(f"ICV volume must be positive, got {self.icv_volume_ml}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "subject_id": self.subject_id,
            "vascular_qfv": self.vascular_qfv.tolist(),
            "lobe_qfv": self.lobe_qfv.tolist(),
            "aspects_qfv": self.aspects_qfv.tolist(),
            "ventricles_qfv": self.ventricles_qfv.tolist(),
            "aspectpc_qfv": self.aspectpc_qfv.tolist() if self.aspectpc_qfv is not None else None,
            "bpm_qfv": self.bpm_qfv.tolist() if self.bpm_qfv is not None else None,
            "lesion_volume_ml": self.lesion_volume_ml,
            "icv_volume_ml": self.icv_volume_ml,
        }
