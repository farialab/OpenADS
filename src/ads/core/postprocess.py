from typing import Dict, Any
import ants
import nibabel as nib
import numpy as np
import re
from pathlib import Path

from ads.models.wrappers import seg_postprocess
from ads.domain.subject_data import SubjectData
from typing import List, Tuple

def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a 3D numpy array."""
    return np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())

class Postprocessor:
    """Handle post-processing of model predictions"""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        
    def run(self, subject: SubjectData) -> SubjectData:
        """Run post-processing pipeline to convert MNI predictions back to original space"""
        self.logger.info(f"[{subject.subject_id}] Starting post-processing...")
        
        seg_dir = subject.stage_dir("segment")
        reg_dir = subject.stage_dir("registration")
        pp_dir = subject.stage_dir("preprocess")
        
        # Step 1: Load and threshold MNI prediction
        stroke_pred_mni_path = seg_dir / f"{subject.subject_id}_stroke-mask_space-MNI152.nii.gz"
        stroke_mni_img = nib.as_closest_canonical(nib.load(str(stroke_pred_mni_path)))
        stroke_mni_data = stroke_mni_img.get_fdata()
        stroke_thresholded = (stroke_mni_data > 0.49).astype(np.uint8)
        
        self.logger.info(f"[{subject.subject_id}] MNI non-zero voxels = {int(stroke_thresholded.sum())}")
        
        # Save binary MNI prediction with new naming
        binary_mni_path = seg_dir / f"{subject.subject_id}_stroke-mask_space-MNI152.nii.gz"
        nib.save(
            nib.Nifti1Image(stroke_thresholded, affine=stroke_mni_img.affine, header=stroke_mni_img.header),
            str(binary_mni_path)
        )
        
        # Step 2: Get inverse transforms
        use_two_steps = True
        
        if use_two_steps:
            # Two-step: ADC template → original
            inverse_transforms = [
                str(reg_dir / f"{subject.subject_id}_aff_space-individual2MNI152.mat"),
                str(reg_dir / f"{subject.subject_id}_syn_space-MNI1522MNI152.mat"),
                str(reg_dir / f"{subject.subject_id}_warp_space-MNI1522MNI152.nii.gz")
            ]
            whichtoinvert = [True, True, False]  # Invert affines, not warp
        else:
            # Single-step: MNI → original
            inverse_transforms = [
                str(reg_dir / f"{subject.subject_id}_aff_space-individual2MNI152.mat")
            ]
            whichtoinvert = [True]
        
        # Validate transforms exist
        for tf in inverse_transforms:
            if not Path(tf).exists():
                raise FileNotFoundError(f"Transform not found: {tf}")
        
        self.logger.info(f"[{subject.subject_id}] Applying inverse transforms:")
        for tf, inv in zip(inverse_transforms, whichtoinvert):
            inv_str = "inverted" if inv else "as-is"
            self.logger.info(f"  - {Path(tf).name} ({inv_str})")
        
        # Step 3: Apply inverse transformation
        stroke_pred_mni_ants = ants.image_read(str(binary_mni_path))
        fixed_orig = ants.image_read(str(subject.dwi_path))
        
        stroke_pred_orig_ants = ants.apply_transforms(
            fixed=fixed_orig,
            moving=stroke_pred_mni_ants,
            transformlist=inverse_transforms,
            whichtoinvert=whichtoinvert,
            interpolator='nearestNeighbor',
            default_value=0
        )
        #ants.image_write(stroke_pred_orig_ants, str(seg_dir / f"{subject.subject_id}_stroke-mask_orig_ants.nii.gz"))
        
        orig_array = stroke_pred_orig_ants.numpy()
        self.logger.info(f"[{subject.subject_id}] Original space non-zero voxels = {int((orig_array > 0.4).sum())}")
        
        # Step 4: Post-process and save
        min_size = self.config.get('postprocessing', {}).get('min_lesion_size', 5)
        stroke_cleaned = seg_postprocess(orig_array, min_size=min_size)
        
        final_orig_path = seg_dir / f"{subject.subject_id}_stroke-mask.nii.gz"
        # Extract affine and header from the original NIfTI file
        dwi_nifti = nib.load(str(subject.dwi_path))
        dwi_mask_path = pp_dir / f"{subject.subject_id}_DWIbrain-mask.nii.gz"
        if dwi_mask_path.exists():
            dwimask_nifti = nib.load(str(dwi_mask_path))
        
        stroke_cleaned *= (dwimask_nifti.get_fdata() > 0).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(stroke_cleaned.astype(np.uint8), affine=dwi_nifti.affine, header=dwi_nifti.header),
            str(final_orig_path)
        )
        
        # Step 5: Calculate volume
        voxel_volume = subject.get_voxel_volume()
        if voxel_volume:
            lesion_voxels = int((stroke_cleaned > 0.5).sum())
            subject.lesion_volume = voxel_volume * lesion_voxels
            self.logger.info(f"[{subject.subject_id}] Lesion volume: {subject.lesion_volume:.2f} mm³")
        
        # Update subject
        subject.lesion_orig_path = final_orig_path
        subject.lesion_mni_path = binary_mni_path
        subject.stroke_pred_orig = stroke_pred_orig_ants
        
        self.logger.info(f"[{subject.subject_id}] Post-processing completed")
        return subject