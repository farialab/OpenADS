"""Individual QFV domain objects for each atlas type.

Each atlas calculation returns its own specific result type.
This allows independent calculation and usage of different atlas types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class VascularQFVResult:
    """Vascular territory QFV result.

    Attributes:
        subject_id: Subject identifier
        qfv: QFV probabilities per vascular territory (9 ROIs typically)
        roi_labels: Names of vascular territories (e.g., ACA, MCA, PCA, etc.)
        lesion_volume_ml: Total lesion volume
    """
    subject_id: str
    qfv: np.ndarray
    roi_labels: List[str]
    lesion_volume_ml: float

    def __post_init__(self):
        if not isinstance(self.qfv, np.ndarray):
            self.qfv = np.array(self.qfv, dtype=float)


@dataclass
class LobeQFVResult:
    """Brain lobe QFV result.

    Attributes:
        subject_id: Subject identifier
        qfv: QFV probabilities per lobe (12 ROIs typically)
        roi_labels: Names of lobes (e.g., frontal, parietal, temporal, etc.)
        lesion_volume_ml: Total lesion volume
    """
    subject_id: str
    qfv: np.ndarray
    roi_labels: List[str]
    lesion_volume_ml: float

    def __post_init__(self):
        if not isinstance(self.qfv, np.ndarray):
            self.qfv = np.array(self.qfv, dtype=float)


@dataclass
class AspectsQFVResult:
    """ASPECTS region QFV result.

    Attributes:
        subject_id: Subject identifier
        qfv: QFV probabilities per ASPECTS region (10 ROIs typically)
        roi_labels: Names of ASPECTS regions (e.g., M1-M6, caudate, etc.)
        lesion_volume_ml: Total lesion volume
    """
    subject_id: str
    qfv: np.ndarray
    roi_labels: List[str]
    lesion_volume_ml: float

    def __post_init__(self):
        if not isinstance(self.qfv, np.ndarray):
            self.qfv = np.array(self.qfv, dtype=float)


@dataclass
class AspectsPCQFVResult:
    """PCA-ASPECTS (posterior circulation) QFV result.

    Attributes:
        subject_id: Subject identifier
        qfv: QFV probabilities per PCA-ASPECTS region (5 ROIs typically)
        roi_labels: Names of posterior regions (e.g., PCA, thalamus, etc.)
        lesion_volume_ml: Total lesion volume
    """
    subject_id: str
    qfv: np.ndarray
    roi_labels: List[str]
    lesion_volume_ml: float

    def __post_init__(self):
        if not isinstance(self.qfv, np.ndarray):
            self.qfv = np.array(self.qfv, dtype=float)


@dataclass
class VentriclesQFVResult:
    """Ventricular region QFV result.

    Attributes:
        subject_id: Subject identifier
        qfv: QFV probabilities per ventricular region (20 ROIs typically)
        roi_labels: Names of ventricular regions
        lesion_volume_ml: Total lesion volume
    """
    subject_id: str
    qfv: np.ndarray
    roi_labels: List[str]
    lesion_volume_ml: float

    def __post_init__(self):
        if not isinstance(self.qfv, np.ndarray):
            self.qfv = np.array(self.qfv, dtype=float)


@dataclass
class AllQFVResults:
    """Container for all atlas QFV results.

    Allows running all atlases at once or independently.
    Optional fields for atlases that may not always be calculated.
    """
    subject_id: str
    vascular: VascularQFVResult
    lobe: LobeQFVResult
    aspects: AspectsQFVResult
    ventricles: VentriclesQFVResult
    aspectpc: Optional[AspectsPCQFVResult] = None
    icv_volume_ml: float = 0.0

    def get_atlas_result(self, atlas_name: str):
        """Get result for specific atlas by name."""
        atlas_map = {
            "vascular": self.vascular,
            "lobe": self.lobe,
            "aspects": self.aspects,
            "aspectpc": self.aspectpc,
            "ventricles": self.ventricles,
        }
        return atlas_map.get(atlas_name)
