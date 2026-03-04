"""Service for reusing DWI registration results (optimization)."""
from pathlib import Path
from typing import Optional
import shutil


class DWIRegistrationReuseService:
    """Check and copy DWI registration results to PWI."""

    @staticmethod
    def check_dwi_registration_exists(subject_dir: Path) -> bool:
        """
        Check if DWI registration folder exists and contains required files.

        Args:
            subject_dir: Subject directory (e.g., output/sub-xxx/PWI/)

        Returns:
            True if DWI registration exists and is complete
        """
        # Determine subject root
        subject_root = subject_dir.parent if subject_dir.name in {"PWI", "DWI"} else subject_dir
        dwi_reg_dir = subject_root / "DWI" / "registration"

        if not dwi_reg_dir.exists():
            return False

        # Check for essential registration files
        required_patterns = [
            "*_aff_space-individual2MNI152.mat",
            "*_warp_space-MNI1522MNI152.nii.gz",
            "*_syn_space-MNI1522MNI152.mat",
        ]

        for pattern in required_patterns:
            matches = list(dwi_reg_dir.glob(pattern))
            if not matches:
                return False

        return True

    @staticmethod
    def copy_dwi_registration_to_pwi(
        subject_dir: Path,
        subject_id: str,
        logger=None
    ) -> bool:
        """
        Copy DWI registration transform files to PWI registration folder.

        Only copies transform files (.mat, warp.nii.gz), not images.
        PWI images will be transformed separately.

        Args:
            subject_dir: Subject directory (e.g., output/sub-xxx/PWI/)
            subject_id: Subject identifier
            logger: Optional logger

        Returns:
            True if successful
        """
        # Determine paths
        subject_root = subject_dir.parent if subject_dir.name in {"PWI", "DWI"} else subject_dir
        dwi_reg_dir = subject_root / "DWI" / "registration"
        pwi_reg_dir = subject_root / "PWI" / "registration"
        pwi_reg_dir.mkdir(parents=True, exist_ok=True)

        # Files to copy (transforms only)
        transform_files = [
            f"{subject_id}_aff_space-individual2MNI152.mat",
            f"{subject_id}_invaff_space-MNI1522individual.mat",
            f"{subject_id}_syn_space-MNI1522MNI152.mat",
            f"{subject_id}_invsyn_space-MNI1522MNI152.mat",
            f"{subject_id}_warp_space-MNI1522MNI152.nii.gz",
            f"{subject_id}_invwarp_space-MNI1522MNI152.nii.gz",
        ]

        copied_count = 0
        for filename in transform_files:
            src = dwi_reg_dir / filename
            dst = pwi_reg_dir / filename

            if src.exists():
                shutil.copy2(src, dst)
                copied_count += 1
                if logger:
                    logger.info(f"  Copied: {filename}")
            else:
                if logger:
                    logger.warning(f"  Missing (optional): {filename}")

        if logger:
            logger.info(f"Copied {copied_count} transform files from DWI to PWI")

        return copied_count > 0
