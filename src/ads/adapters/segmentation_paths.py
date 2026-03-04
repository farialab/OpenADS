"""Path builder for segmentation outputs.

Constructs output file paths following naming conventions.
"""
from pathlib import Path
from typing import Dict


class SegmentationPathBuilder:
    """
    Build output paths for segmentation results.

    Maintains exact naming conventions from original implementation.
    """

    def __init__(self, output_dir: Path, subject_id: str):
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id

    def get_output_paths(self) -> Dict[str, Path]:
        """
        Get all segmentation output paths.

        Returns paths matching original pipe_segment.py naming:
        - pred_mni: {subject_id}_stroke-mask_space-MNI152.nii.gz
        - pred_mni_affsyn: {subject_id}_stroke-mask_space-MNI152_affsyn.nii.gz
        - pred_native: {subject_id}_stroke-mask.nii.gz
        - metrics: {subject_id}_metrics.json
        """
        return {
            'pred_mni': self.output_dir / f"{self.subject_id}_stroke-mask_space-MNI152.nii.gz",
            'pred_mni_affsyn': self.output_dir / f"{self.subject_id}_stroke-mask_space-MNI152_affsyn.nii.gz",
            'pred_native': self.output_dir / f"{self.subject_id}_stroke-mask.nii.gz",
            'metrics': self.output_dir / f"{self.subject_id}_metrics.json",
        }

    def pred_mni_path(self) -> Path:
        """Get MNI prediction path."""
        return self.output_dir / f"{self.subject_id}_stroke-mask_space-MNI152.nii.gz"

    def pred_native_path(self) -> Path:
        """Get native space prediction path."""
        return self.output_dir / f"{self.subject_id}_stroke-mask.nii.gz"

    def pred_mni_affsyn_path(self) -> Path:
        """Get MNI affsyn prediction path."""
        return self.output_dir / f"{self.subject_id}_stroke-mask_space-MNI152_affsyn.nii.gz"

    def metrics_path(self) -> Path:
        """Get metrics JSON path."""
        return self.output_dir / f"{self.subject_id}_metrics.json"


class PWIPathBuilder:
    """
    Build output paths for PWI segmentation results.

    Maintains exact naming conventions for hypoperfusion (HP) predictions.
    """

    def __init__(self, output_dir: Path, subject_id: str):
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id

    def get_output_paths(self) -> Dict[str, Path]:
        """
        Get all PWI segmentation output paths.

        Returns paths for hypoperfusion predictions:
        - pred_mni: {subject_id}_HP_mask_space-MNI152.nii.gz
        - pred_native: {subject_id}_HP_mask.nii.gz
        - metrics: {subject_id}_metrics.json
        """
        return {
            'pred_mni': self.output_dir / f"{self.subject_id}_HP_mask_space-MNI152.nii.gz",
            'pred_native': self.output_dir / f"{self.subject_id}_HP_mask.nii.gz",
            'metrics': self.output_dir / f"{self.subject_id}_metrics.json",
        }

    def pred_mni_path(self) -> Path:
        """Get MNI prediction path."""
        return self.output_dir / f"{self.subject_id}_HP_mask_space-MNI152.nii.gz"

    def pred_native_path(self) -> Path:
        """Get native space prediction path."""
        return self.output_dir / f"{self.subject_id}_HP_mask.nii.gz"

    def metrics_path(self) -> Path:
        """Get metrics JSON path."""
        return self.output_dir / f"{self.subject_id}_metrics.json"
