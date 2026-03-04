"""Service for discovering DWI registration transforms."""
from pathlib import Path
from typing import Optional, List
from ads.domain.pwi_registration_data import DWITransformArtifacts


class TransformDiscoveryService:
    """Find DWI registration artifacts to reuse for PWI."""

    @staticmethod
    def find_dwi_transforms(
        subject_dir: Path,
        subject_id: str,
        dwi_mask_template: Path,
        adc_template: Path,
    ) -> DWITransformArtifacts:
        """
        Find DWI→MNI registration transforms.

        Searches in:
        - {subject_dir}/DWI/registration/
        - {subject_dir}/PWI/registration/ (fallback)
        """
        # Determine search directories
        subject_root = subject_dir.parent if subject_dir.name in {"PWI", "DWI"} else subject_dir

        # Search PWI/registration first (priority), then DWI/registration (fallback)
        search_dirs = [
            subject_root / "PWI" / "registration",
            subject_root / "DWI" / "registration",
        ]
        search_dirs = [d for d in search_dirs if d.exists() and d.is_dir()]

        if not search_dirs:
            raise FileNotFoundError(
                f"No registration directories found for {subject_id}. "
                f"Searched: {subject_root}/{{DWI,PWI}}/registration/"
            )

        def find_file(filename: str) -> Optional[Path]:
            for d in search_dirs:
                candidate = d / filename
                if candidate.exists():
                    return candidate
            return None

        # Find transforms
        aff_mat = find_file(f"{subject_id}_aff_space-individual2MNI152.mat")
        warp = find_file(f"{subject_id}_warp_space-MNI1522MNI152.nii.gz")
        syn_mat = find_file(f"{subject_id}_syn_space-MNI1522MNI152.mat")

        # Validate
        if aff_mat is None:
            raise FileNotFoundError(
                f"Missing affine transform: {subject_id}_aff_space-individual2MNI152.mat. "
                f"Searched: {[str(d) for d in search_dirs]}"
            )
        if warp is None:
            raise FileNotFoundError(
                f"Missing SyN warp: {subject_id}_warp_space-MNI1522MNI152.nii.gz. "
                f"Searched: {[str(d) for d in search_dirs]}"
            )
        if syn_mat is None:
            raise FileNotFoundError(
                f"Missing SyN affine: {subject_id}_syn_space-MNI1522MNI152.mat. "
                f"Searched: {[str(d) for d in search_dirs]}"
            )

        # Validate templates
        if not dwi_mask_template.exists():
            raise FileNotFoundError(f"DWI mask template not found: {dwi_mask_template}")
        if not adc_template.exists():
            raise FileNotFoundError(f"ADC template not found: {adc_template}")

        return DWITransformArtifacts(
            aff_mat=aff_mat,
            warp=warp,
            syn_mat=syn_mat,
            dwi_mask_template=dwi_mask_template,
            adc_template=adc_template,
        )
