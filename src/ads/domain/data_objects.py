"""Domain data objects for preprocessing pipeline.

These classes represent the core data structures used throughout the preprocessing
pipeline. They are deliberately kept simple and free of I/O or computation logic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import nibabel as nib


@dataclass
class LoadedData:
    """Represents loaded imaging data for a subject.

    This is the output of data loading and the input to preprocessing services.

    Attributes:
        dwi: DWI (Diffusion Weighted Imaging) volume
        b0: B0 (baseline) volume, if available
        adc: ADC (Apparent Diffusion Coefficient) map, if available
        stroke: Stroke mask, if available
        subject_id: Subject identifier
        metadata: Additional metadata (e.g., b-values, acquisition params)
        provenance: Record of source file paths for traceability
    """

    dwi: nib.Nifti1Image
    subject_id: str
    b0: Optional[nib.Nifti1Image] = None
    adc: Optional[nib.Nifti1Image] = None
    stroke: Optional[nib.Nifti1Image] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Path] = field(default_factory=dict)

    def __post_init__(self):
        """Validate loaded data."""
        if self.dwi is None:
            raise ValueError("DWI image is required")
        if not self.subject_id:
            raise ValueError("subject_id is required")


@dataclass
class PreprocessResult:
    """Result of preprocessing operations.

    This encapsulates all outputs from the preprocessing stage, including
    processed images and derivative outputs.

    Attributes:
        dwi: Processed DWI image
        adc: Processed ADC map
        b0: Processed B0 image, if available
        stroke: Processed stroke mask, if available
        dwi_normalized: Normalized DWI image for model input
        subject_id: Subject identifier
        output_paths: Dictionary mapping output types to file paths
        metadata: Processing metadata (parameters used, timestamps, etc.)
    """

    dwi: nib.Nifti1Image
    adc: nib.Nifti1Image
    subject_id: str
    b0: Optional[nib.Nifti1Image] = None
    stroke: Optional[nib.Nifti1Image] = None
    dwi_normalized: Optional[nib.Nifti1Image] = None
    output_paths: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
