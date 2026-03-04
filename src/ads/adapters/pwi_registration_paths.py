"""Build output paths for PWI registration results."""
from pathlib import Path
from typing import Dict, Optional, List


class PWIRegistrationPathBuilder:
    """Construct PWI registration output file paths."""

    def __init__(self, output_dir: Path, subject_id: str):
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id

    def pwi_on_dwi_paths(self, has_hp: bool = False) -> Dict[str, Optional[Path]]:
        """Get paths for PWI images in DWI space."""
        return {
            'ttp': self.output_dir / f"{self.subject_id}_TTP_space-DWI.nii.gz",
            'hp_manual': self.output_dir / f"{self.subject_id}_HP_manual_space-DWI.nii.gz" if has_hp else None,
            # NOTE: pwi path removed - only TTP and HP_manual processed (matches old version)
        }

    def coreg_transform_paths(self) -> Dict[str, Path]:
        """Get paths for TTP→ADC(DWI-space) affine transform artifacts."""
        return {
            'fwd_affine': self.output_dir / f"{self.subject_id}_aff_space-individualTTP2ADC.mat",
            'inv_affine': self.output_dir / f"{self.subject_id}_invaff_space-ADC2individualTTP.mat",
        }

    def ttp_aff_normalized_path(self) -> Path:
        """Get path for normalized affine-space TTP in MNI space."""
        return self.output_dir / f"{self.subject_id}_TTP_space-MNI152_aff_desc-norm.nii.gz"

    def masked_paths(self, has_hp: bool = False) -> Dict[str, Optional[Path]]:
        """Get paths for brain-masked images."""
        return {
            'ttp': self.output_dir / f"{self.subject_id}_TTP_space-DWI_desc-brain.nii.gz",
            'hp_manual': self.output_dir / f"{self.subject_id}_HP_manual_space-DWI_desc-brain.nii.gz" if has_hp else None,
            # NOTE: pwi path removed - only TTP and HP_manual processed (matches old version)
        }

    def mni_paths(self, has_hp: bool = False) -> Dict[str, Dict[str, Optional[Path]]]:
        """Get paths for images in MNI space (aff and affsyn)."""
        return {
            'aff': {
                'ttp': self.output_dir / f"{self.subject_id}_TTP_space-MNI152_aff.nii.gz",
                'hp_manual': self.output_dir / f"{self.subject_id}_HP_manual_space-MNI152_aff.nii.gz" if has_hp else None,
                # TODO: Fix PWI convert to MNI problem - commented out for now
                # 'pwi': self.output_dir / f"{self.subject_id}_PWI_space-MNI152_aff.nii.gz",  # 4D
            },
            'affsyn': {
                'ttp': self.output_dir / f"{self.subject_id}_TTP_space-MNI152_affsyn.nii.gz",
                'hp_manual': self.output_dir / f"{self.subject_id}_HP_manual_space-MNI152_affsyn.nii.gz" if has_hp else None,
                # TODO: Fix PWI convert to MNI problem - commented out for now
                # 'pwi': self.output_dir / f"{self.subject_id}_PWI_space-MNI152_affsyn.nii.gz",  # 4D
            }
        }


class PWIInputDiscovery:
    """Discover PWI input files from preprocessing."""

    @staticmethod
    def find_pwi_inputs(preprocess_dir: Path, subject_id: str) -> Dict[str, Optional[Path]]:
        """Find PWI input files with fallback naming."""
        sid_clean = subject_id.replace("sub-", "")

        def find_first(filenames: list, search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
            """Search for file in multiple directories."""
            if search_dirs is None:
                search_dirs = [preprocess_dir]

            for directory in search_dirs:
                for name in filenames:
                    p = directory / name
                    if p.exists():
                        return p
            return None

        # For DWI mask, also search in DWI/preprocess/ (sibling directory)
        subject_root = preprocess_dir.parent.parent if preprocess_dir.parent.name == "PWI" else preprocess_dir.parent
        dwi_preprocess_dir = subject_root / "DWI" / "preprocess"
        dwi_search_dirs = [preprocess_dir, dwi_preprocess_dir] if dwi_preprocess_dir.exists() else [preprocess_dir]

        return {
            'ttp': find_first([
                f"{subject_id}_TTP.nii.gz",
                f"{sid_clean}_TTP.nii.gz",
                "TTP.nii.gz",
            ]),
            'hp_manual': find_first([
                f"{subject_id}_HP_manual2.nii.gz",
                f"{subject_id}_HP_manual.nii.gz",
                "HP_manual.nii.gz",
            ]),
            'pwi': find_first([
                f"{subject_id}_PWI.nii.gz",
                f"{sid_clean}_PWI.nii.gz",
                "PWI.nii.gz",
            ]),
            'pwi_mask': find_first([
                f"{subject_id}_PWIbrain-mask.nii.gz",
                f"{subject_id}_PWI_ss.nii.gz",
            ]),
            'dwi_mask': find_first([
                f"{subject_id}_DWIbrain-mask.nii.gz",
                "DWIbrain-mask.nii.gz",
            ], search_dirs=dwi_search_dirs),  # Search in both PWI and DWI preprocess
        }
