"""Transform file I/O operations."""
from pathlib import Path


class TransformIO:
    """Handle transform file operations."""

    @staticmethod
    def copy_transform(src: str, dst: Path) -> None:
        """Copy transform file to destination."""
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Transform not found: {src_path}")
        dst.write_bytes(src_path.read_bytes())

    @staticmethod
    def save_transforms(
        ants_result: dict,
        output_dir: Path,
        subject_id: str,
        stage: str,  # 'aff' or 'affsyn'
    ) -> dict:
        """
        Save ANTs transforms with proper naming.

        Returns dict with saved transform paths.
        """
        if stage == 'aff':
            fwd_affine_name = f"{subject_id}_aff_space-individual2MNI152.mat"
            inv_affine_name = f"{subject_id}_invaff_space-MNI1522individual.mat"

            paths = {}
            for t in ants_result['fwdtransforms']:
                if 'GenericAffine.mat' in t:
                    dst = output_dir / fwd_affine_name
                    TransformIO.copy_transform(t, dst)
                    paths['fwd_affine'] = dst

            for t in ants_result['invtransforms']:
                if 'GenericAffine.mat' in t:
                    dst = output_dir / inv_affine_name
                    TransformIO.copy_transform(t, dst)
                    paths['inv_affine'] = dst

            return paths

        elif stage == 'affsyn':
            fwd_affine_name = f"{subject_id}_syn_space-MNI1522MNI152.mat"
            fwd_warp_name = f"{subject_id}_warp_space-MNI1522MNI152.nii.gz"
            inv_affine_name = f"{subject_id}_invsyn_space-MNI1522MNI152.mat"
            inv_warp_name = f"{subject_id}_invwarp_space-MNI1522MNI152.nii.gz"

            paths = {}
            for t in ants_result['fwdtransforms']:
                if 'GenericAffine.mat' in t:
                    dst = output_dir / fwd_affine_name
                    TransformIO.copy_transform(t, dst)
                    paths['fwd_affine'] = dst
                elif 'Warp.nii.gz' in t:
                    dst = output_dir / fwd_warp_name
                    TransformIO.copy_transform(t, dst)
                    paths['fwd_warp'] = dst

            for t in ants_result['invtransforms']:
                if 'GenericAffine.mat' in t:
                    dst = output_dir / inv_affine_name
                    TransformIO.copy_transform(t, dst)
                    paths['inv_affine'] = dst
                elif 'InverseWarp.nii.gz' in t:
                    dst = output_dir / inv_warp_name
                    TransformIO.copy_transform(t, dst)
                    paths['inv_warp'] = dst

            return paths

        else:
            raise ValueError(f"Unknown stage: {stage}")
