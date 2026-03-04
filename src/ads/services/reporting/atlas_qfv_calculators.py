"""Individual QFV calculators for each atlas type.

Each atlas has its own calculator service that can be used independently.
This allows flexible usage - calculate only what you need.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ads.domain.reporting import (
    VascularQFVResult,
    LobeQFVResult,
    AspectsQFVResult,
    AspectsPCQFVResult,
    VentriclesQFVResult,
    AllQFVResults,
)


class VascularQFVCalculator:
    """Calculate QFV for vascular territories independently.

    Can be used standalone without calculating other atlases.
    """

    # Standard vascular territory ROI labels
    ROI_LABELS = [
        "ACA",
        "MCA",
        "PCA",
        "cerebellar",
        "basilar",
        "Lenticulostriate",
        "Choroidal & Thalamoperfurating",
        "watershed",
    ]

    def calculate(
        self,
        lesion_mask: np.ndarray,
        vascular_template: np.ndarray,
        vascular_volumes: pd.DataFrame,
        voxel_size_mm: Tuple[float, float, float],
        subject_id: str,
    ) -> VascularQFVResult:
        """Calculate vascular territory QFV.

        Args:
            lesion_mask: Binary lesion mask
            vascular_template: Vascular atlas template
            vascular_volumes: Vascular ROI volume lookup table
            voxel_size_mm: Voxel dimensions
            subject_id: Subject identifier

        Returns:
            VascularQFVResult with QFV values
        """
        # Count lesion voxels per ROI
        max_label = int(vascular_template.max())
        counts = np.zeros(max_label, dtype=int)

        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (vascular_template == roi_label)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0)))

        # Calculate QFV (probability per ROI)
        qfv = self._calculate_probabilities(counts, vascular_volumes)

        # Calculate lesion volume
        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        lesion_volume_ml = float(np.sum(lesion_mask > 0.5) * voxel_volume_ml)

        return VascularQFVResult(
            subject_id=subject_id,
            qfv=qfv,
            roi_labels=self.ROI_LABELS[:len(qfv)],
            lesion_volume_ml=lesion_volume_ml,
        )

    def _calculate_probabilities(
        self,
        counts: np.ndarray,
        volume_table: pd.DataFrame,
    ) -> np.ndarray:
        """Calculate QFV probabilities from counts."""
        # Extract expected volumes from lookup table
        if 'volume' in volume_table.columns:
            expected_volumes = volume_table['volume'].values
        else:
            # Assume first numeric column is volume
            expected_volumes = volume_table.iloc[:, 0].values

        # Normalize by expected volume
        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = counts / expected_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities


class LobeQFVCalculator:
    """Calculate QFV for brain lobes independently."""

    ROI_LABELS = [
        "basal ganglia",
        "deep white matter",
        "cerebellum",
        "frontal",
        "insula",
        "internal capsule",
        "brainstem",
        "occipital",
        "parietal",
        "temporal",
        "thalamus",
    ]

    def calculate(
        self,
        lesion_mask: np.ndarray,
        lobe_template: np.ndarray,
        lobe_volumes: pd.DataFrame,
        voxel_size_mm: Tuple[float, float, float],
        subject_id: str,
    ) -> LobeQFVResult:
        """Calculate lobe QFV."""
        max_label = int(lobe_template.max())
        counts = np.zeros(max_label, dtype=int)

        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (lobe_template == roi_label)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0)))

        qfv = self._calculate_probabilities(counts, lobe_volumes)

        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        lesion_volume_ml = float(np.sum(lesion_mask > 0.5) * voxel_volume_ml)

        return LobeQFVResult(
            subject_id=subject_id,
            qfv=qfv,
            roi_labels=self.ROI_LABELS[:len(qfv)],
            lesion_volume_ml=lesion_volume_ml,
        )

    def _calculate_probabilities(self, counts: np.ndarray, volume_table: pd.DataFrame) -> np.ndarray:
        if 'volume' in volume_table.columns:
            expected_volumes = volume_table['volume'].values
        else:
            expected_volumes = volume_table.iloc[:, 0].values

        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = counts / expected_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities


class AspectsQFVCalculator:
    """Calculate QFV for ASPECTS regions independently."""

    ROI_LABELS = [
        "Caudate",
        "lentiform",
        "IC",
        "insula",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
    ]

    def calculate(
        self,
        lesion_mask: np.ndarray,
        aspects_template: np.ndarray,
        aspects_volumes: pd.DataFrame,
        voxel_size_mm: Tuple[float, float, float],
        subject_id: str,
    ) -> AspectsQFVResult:
        """Calculate ASPECTS QFV."""
        max_label = int(aspects_template.max())
        counts = np.zeros(max_label, dtype=int)

        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (aspects_template == roi_label)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0)))

        qfv = self._calculate_probabilities(counts, aspects_volumes)

        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        lesion_volume_ml = float(np.sum(lesion_mask > 0.5) * voxel_volume_ml)

        return AspectsQFVResult(
            subject_id=subject_id,
            qfv=qfv,
            roi_labels=self.ROI_LABELS[:len(qfv)],
            lesion_volume_ml=lesion_volume_ml,
        )

    def _calculate_probabilities(self, counts: np.ndarray, volume_table: pd.DataFrame) -> np.ndarray:
        if 'volume' in volume_table.columns:
            expected_volumes = volume_table['volume'].values
        else:
            expected_volumes = volume_table.iloc[:, 0].values

        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = counts / expected_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities


class AspectsPCQFVCalculator:
    """Calculate QFV for PCA-ASPECTS (posterior circulation) independently."""

    ROI_LABELS = [
        "PCA",
        "Thalamus",
        "cerebellum",
        "pons",
        "midbrain",
    ]

    def calculate(
        self,
        lesion_mask: np.ndarray,
        aspectpc_template: np.ndarray,
        aspectpc_volumes: pd.DataFrame,
        voxel_size_mm: Tuple[float, float, float],
        subject_id: str,
    ) -> AspectsPCQFVResult:
        """Calculate PCA-ASPECTS QFV."""
        max_label = int(aspectpc_template.max())
        counts = np.zeros(max_label, dtype=int)

        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (aspectpc_template == roi_label)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0)))

        qfv = self._calculate_probabilities(counts, aspectpc_volumes)

        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        lesion_volume_ml = float(np.sum(lesion_mask > 0.5) * voxel_volume_ml)

        return AspectsPCQFVResult(
            subject_id=subject_id,
            qfv=qfv,
            roi_labels=self.ROI_LABELS[:len(qfv)],
            lesion_volume_ml=lesion_volume_ml,
        )

    def _calculate_probabilities(self, counts: np.ndarray, volume_table: pd.DataFrame) -> np.ndarray:
        if 'volume' in volume_table.columns:
            expected_volumes = volume_table['volume'].values
        else:
            expected_volumes = volume_table.iloc[:, 0].values

        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = counts / expected_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities


class VentriclesQFVCalculator:
    """Calculate QFV for ventricular regions independently."""

    def calculate(
        self,
        lesion_mask: np.ndarray,
        ventricles_template: np.ndarray,
        ventricles_volumes: pd.DataFrame,
        voxel_size_mm: Tuple[float, float, float],
        subject_id: str,
        roi_labels: Optional[list] = None,
    ) -> VentriclesQFVResult:
        """Calculate ventricles QFV."""
        max_label = int(ventricles_template.max())
        counts = np.zeros(max_label, dtype=int)

        for i in range(max_label):
            roi_label = i + 1
            roi_mask = (ventricles_template == roi_label)
            counts[i] = int(np.sum(np.isclose(lesion_mask[roi_mask], 1.0)))

        qfv = self._calculate_probabilities(counts, ventricles_volumes)

        voxel_volume_ml = (voxel_size_mm[0] * voxel_size_mm[1] * voxel_size_mm[2]) / 1000.0
        lesion_volume_ml = float(np.sum(lesion_mask > 0.5) * voxel_volume_ml)

        if roi_labels is None:
            roi_labels = [f"Ventricle_ROI_{i+1}" for i in range(len(qfv))]

        return VentriclesQFVResult(
            subject_id=subject_id,
            qfv=qfv,
            roi_labels=roi_labels,
            lesion_volume_ml=lesion_volume_ml,
        )

    def _calculate_probabilities(self, counts: np.ndarray, volume_table: pd.DataFrame) -> np.ndarray:
        if 'volume' in volume_table.columns:
            expected_volumes = volume_table['volume'].values
        else:
            expected_volumes = volume_table.iloc[:, 0].values

        with np.errstate(divide='ignore', invalid='ignore'):
            probabilities = counts / expected_volumes
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)

        return probabilities
