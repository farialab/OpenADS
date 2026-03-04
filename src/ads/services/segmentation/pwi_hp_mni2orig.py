"""Restore PWI HP mask from MNI affine space to original/native space."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ants
import numpy as np


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_paths(subject_root: Path, subject_id: str) -> dict[str, Optional[Path]]:
    pwi = subject_root / "PWI"
    pwi_pp = pwi / "preprocess"
    pwi_reg = pwi / "registration"
    pwi_seg = pwi / "segmentation"
    dwi_reg = subject_root / "DWI" / "registration"

    hp_mni = pwi_seg / f"{subject_id}_HP-mask_space-MNI152.nii.gz"
    hp_native = pwi_seg / f"{subject_id}_HP-mask.nii.gz"

    adc_native = _first_existing(
        [
            pwi_pp / f"{subject_id}_ADC_brain.nii.gz",
            pwi_pp / f"{subject_id}_ADC.nii.gz",
            pwi_pp / f"{subject_id}_DWI_brain.nii.gz",
            pwi_pp / f"{subject_id}_DWI.nii.gz",
        ]
    )
    ttp_native = _first_existing(
        [
            pwi_pp / f"{subject_id}_TTP.nii.gz",
            pwi_pp / f"{subject_id}_TTP_brain.nii.gz",
        ]
    )
    native_mask = _first_existing(
        [
            pwi_pp / f"{subject_id}_PWIbrain-mask.nii.gz",
            pwi_pp / f"{subject_id}_DWIbrain-mask.nii.gz",
        ]
    )

    aff_ind2mni = _first_existing(
        [
            pwi_reg / f"{subject_id}_aff_space-individual2MNI152.mat",
            dwi_reg / f"{subject_id}_aff_space-individual2MNI152.mat",
        ]
    )
    adc2ttp = _first_existing(
        [
            pwi_reg / f"{subject_id}_invaff_space-ADC2individualTTP.mat",
            dwi_reg / f"{subject_id}_invaff_space-ADC2individualTTP.mat",
        ]
    )

    return {
        "hp_mni": hp_mni,
        "hp_native": hp_native,
        "adc_native": adc_native,
        "ttp_native": ttp_native,
        "native_mask": native_mask,
        "aff_ind2mni": aff_ind2mni,
        "adc2ttp": adc2ttp,
    }


def restore_hp_mni2orig(subject_root: str | Path, subject_id: str) -> Path:
    """Restore HP mask from MNI affine space to native space.

    Args:
        subject_root: output subject directory, e.g. `output/sub-xxxx`
        subject_id: subject ID, e.g. `sub-xxxx`

    Returns:
        Path to saved native mask in `PWI/segmentation/{subject_id}_HP-mask.nii.gz`.
    """
    root = Path(subject_root).expanduser().resolve()
    paths = _resolve_paths(root, subject_id)

    hp_mni = paths["hp_mni"]
    hp_native = paths["hp_native"]
    adc_native = paths["adc_native"]
    ttp_native = paths["ttp_native"]
    native_mask = paths["native_mask"]
    aff_ind2mni = paths["aff_ind2mni"]
    adc2ttp = paths["adc2ttp"]

    if hp_mni is None or not hp_mni.exists():
        raise FileNotFoundError(f"Missing MNI HP mask: {hp_mni}")
    if hp_native is None:
        raise FileNotFoundError("Cannot resolve native HP output path.")
    if adc_native is None:
        raise FileNotFoundError("Missing native ADC (or DWI fallback) in PWI/preprocess.")
    if aff_ind2mni is None:
        raise FileNotFoundError("Missing aff_space-individual2MNI152.mat in PWI/DWI registration.")

    moving_hp_mni = ants.image_read(str(hp_mni))
    fixed_adc = ants.image_read(str(adc_native))

    # Step 1: MNI affine -> ADC native.
    hp_adc = ants.apply_transforms(
        fixed=fixed_adc,
        moving=moving_hp_mni,
        transformlist=[str(aff_ind2mni)],
        whichtoinvert=[True],
        interpolator="nearestNeighbor",
    )

    # Step 2 (preferred): ADC native -> TTP native.
    target = hp_adc
    if ttp_native is not None and adc2ttp is not None:
        fixed_ttp = ants.image_read(str(ttp_native))
        target = ants.apply_transforms(
            fixed=fixed_ttp,
            moving=hp_adc,
            transformlist=[str(adc2ttp)],
            interpolator="nearestNeighbor",
        )

    data = (target.numpy() > 0.5).astype(np.float32)
    if native_mask is not None:
        mask_data = ants.image_read(str(native_mask)).numpy()
        if mask_data.shape == data.shape:
            data *= (mask_data > 0.5).astype(np.float32)

    out_img = ants.from_numpy(
        data,
        origin=target.origin,
        spacing=target.spacing,
        direction=target.direction,
    )

    hp_native.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(out_img, str(hp_native))
    return hp_native
