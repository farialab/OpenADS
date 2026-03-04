"""Subject data container for the refactored pipeline (NIfTI-first)."""

from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np


class SubjectData:
    """Lightweight holder for subject file paths and loaded NIfTI images."""

    _STAGES = ("preprocess", "register", "segment", "qfv", "report")

    def __init__(self, input_dir: Path, subject_id: str, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.subject_id = subject_id
        self.output_dir = Path(output_dir)

        self.stage_paths = {stage: self.output_dir / stage for stage in self._STAGES}

        # Primary inputs
        self.dwi_path: Optional[Path] = None
        self.adc_path: Optional[Path] = None
        self.b0_path: Optional[Path] = None

        # Derived artifact paths (populated during preprocessing/registration)
        self.dwi_preprocessed_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_DWI.nii.gz"
        self.adc_preprocessed_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_ADC.nii.gz"
        self.b0_preprocessed_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_B0.nii.gz"
        self.mask_preprocessed_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_Mask.nii.gz"
        self.dwi_brain_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_DWI_brain.nii.gz"
        self.adc_brain_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_ADC_brain.nii.gz"
        self.b0_brain_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_B0_brain.nii.gz"
        self.mask_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_Synthstrip_brain_mask_raw.nii.gz"
        self.mask_binary_path: Path = self.stage_paths["preprocess"] / f"{self.subject_id}_DWIbrain-mask.nii.gz"

        self.dwi_registered_path: Path = self.stage_paths["register"] / f"{self.subject_id}_DWI_space-MNI152_affsyn.nii.gz"
        self.adc_registered_path: Path = self.stage_paths["register"] / f"{self.subject_id}_ADC_space-MNI152_affsyn.nii.gz"
        self.mask_registered_path: Path = self.stage_paths["register"] / f"{self.subject_id}_DWIbrain-mask_space-MNI152_affsyn.nii.gz"
        self.b0_registered_path: Optional[Path] = None

        # In-memory NIfTI images (loaded lazily on demand)
        self.dwi_img: Optional[nib.Nifti1Image] = None
        self.adc_img: Optional[nib.Nifti1Image] = None
        self.b0_img: Optional[nib.Nifti1Image] = None
        self.mask_img: Optional[nib.Nifti1Image] = None

        # Registration transforms & outputs (populated later)
        self.transform: Optional[list[str]] = None
        self.inv_transform: Optional[list[str]] = None
        self.stroke_pred_mni = None
        self.stroke_pred_orig = None
        self.lesion_volume: Optional[float] = None
        self.stroke_pred_mni_path: Optional[Path] = None
        self.model_tag: Optional[str] = None
        self.model_mni_path: Optional[Path] = None
        self.lesion_volume_ml: Optional[float] = None
        self.qfv_features = None
        self.qfv_probabilities = None
        self.qfv_summary_path: Optional[Path] = None
        self.volume_brain_regions_path: Optional[Path] = None
        self.side_voxels: Optional[np.ndarray] = None
        self.icv_volume_mm3: Optional[float] = None
        self.stroke_logml: Optional[float] = None
        self.lesion_orig_path: Optional[Path] = None
        self.model_orig_path: Optional[Path] = None
        self.lesion_mni_binary_path: Optional[Path] = None

    def stage_dir(self, stage: str) -> Path:
        path = self.stage_paths.setdefault(stage, self.output_dir / stage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_files(self, files_dict: Dict[str, Optional[str]]) -> None:
        """Assign input file paths from a lookup dictionary."""
        dwi_path = files_dict.get("dwi")
        if dwi_path is None:
            raise FileNotFoundError(f"DWI image not found for {self.subject_id}")

        self.dwi_path = Path(dwi_path)
        adc_path = files_dict.get("adc")
        self.adc_path = Path(adc_path) if adc_path else None
        b0_path = files_dict.get("b0")
        self.b0_path = Path(b0_path) if b0_path else None

    def load_images(self) -> None:
        """Load available inputs using nibabel; called lazily by pipeline stages."""
        if self.dwi_path and self.dwi_img is None:
            self.dwi_img = nib.as_closest_canonical(nib.load(str(self.dwi_path)))

        if self.adc_path and self.adc_img is None:
            self.adc_img = nib.as_closest_canonical(nib.load(str(self.adc_path)))

        if self.b0_path and self.b0_img is None:
            self.b0_img = nib.as_closest_canonical(nib.load(str(self.b0_path)))

        if self.mask_path.exists() and self.mask_img is None:
            self.mask_img = nib.as_closest_canonical(nib.load(str(self.mask_path)))

    def refresh_mask(self) -> None:
        """Reload mask images from disk after external tools generate them."""
        if self.mask_path.exists():
            self.mask_img = nib.as_closest_canonical(nib.load(str(self.mask_path)))

    def get_voxel_volume(self) -> Optional[float]:
        """Return voxel volume in mm^3 based on the DWI image header."""
        if self.dwi_img is None:
            return None
        spacing = self.dwi_img.header.get_zooms()[:3]
        return float(np.prod(spacing))
