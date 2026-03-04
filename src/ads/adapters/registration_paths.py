"""Build output paths for registration results."""
from pathlib import Path
from typing import Dict, Optional


class RegistrationPathBuilder:
    """Construct registration output file paths."""

    def __init__(self, output_dir: Path, subject_id: str):
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id

    def affine_image_path(self, modality: str) -> Path:
        """Get path for affine-registered image."""
        return self.output_dir / f"{self.subject_id}_{modality}_space-MNI152_aff.nii.gz"

    def affine_normalized_image_path(self, modality: str) -> Path:
        """Get path for affine-registered normalized image."""
        return self.output_dir / f"{self.subject_id}_{modality}_space-MNI152_aff_desc-norm.nii.gz"

    def syn_image_path(self, modality: str) -> Path:
        """Get path for SyN-registered image."""
        return self.output_dir / f"{self.subject_id}_{modality}_space-MNI152_affsyn.nii.gz"

    def affine_paths(self, has_b0: bool = False, has_stroke: bool = False) -> Dict[str, Optional[Path]]:
        """Get all affine output paths."""
        return {
            'dwi': self.affine_image_path('DWI'),
            'adc': self.affine_image_path('ADC'),
            'mask': self.affine_image_path('DWIbrain-mask'),
            'b0': self.affine_image_path('B0') if has_b0 else None,
            'stroke': self.affine_image_path('stroke') if has_stroke else None,
        }

    def syn_paths(self, has_b0: bool = False, has_stroke: bool = False) -> Dict[str, Optional[Path]]:
        """Get all SyN output paths."""
        return {
            'dwi': self.syn_image_path('DWI'),
            'adc': self.syn_image_path('ADC'),
            'mask': self.syn_image_path('DWIbrain-mask'),
            'b0': self.syn_image_path('B0') if has_b0 else None,
            'stroke': self.syn_image_path('stroke') if has_stroke else None,
        }

    def affine_normalized_paths(self) -> Dict[str, Path]:
        """Get normalized affine output paths for DWI and ADC."""
        return {
            'dwi': self.affine_normalized_image_path('DWI'),
            'adc': self.affine_normalized_image_path('ADC'),
        }

    def manifest_paths(self) -> Dict[str, Path]:
        """Get manifest file paths."""
        return {
            'yaml': self.output_dir / f"{self.subject_id}_registration_manifest.yaml",
            'json': self.output_dir / f"{self.subject_id}_registration_manifest.json",
        }
