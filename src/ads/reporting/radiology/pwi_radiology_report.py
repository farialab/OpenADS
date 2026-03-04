# src/ads/utils/pwi_report.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib


HEADER = "-----------------------------------Automatic Radiological Report-----------------------------------\n"
FOOTER = "----------------------------------------------------------------------------------------------------\n"


def _voxel_volume_ml(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()[:3]  # mm
    return float(zooms[0] * zooms[1] * zooms[2]) / 1000.0  # mm^3 -> ml


def calc_mask_volume_ml(mask_path: Path, thr: float = 0.5) -> float:
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    vox = int((data > thr).sum())
    return vox * _voxel_volume_ml(img)


def write_pwi_radiological_report(
    *,
    pwi_root: Path,
    subject_id: str,
    hp_mask_path: Path,
    dwi_report_path: Path | None,
    dwi_stroke_mask_path: Path | None = None,
) -> Path:
    hp_ml = calc_mask_volume_ml(hp_mask_path)
    hp_line = f"The volume of hypoperfusion on TTP is {hp_ml:.2f} ml."
    mismatch_line = None

    if dwi_stroke_mask_path is not None and dwi_stroke_mask_path.exists():
        dwi_ml = calc_mask_volume_ml(dwi_stroke_mask_path)
        if dwi_ml > 0:
            mismatch_ml = hp_ml - dwi_ml
            mismatch_ratio = hp_ml / dwi_ml
            mismatch_line = (
                f"The mismatch volume is {mismatch_ml:.2f} ml, "
                f"the mismatch ratio is {mismatch_ratio:.2f}."
            )

    SEPARATOR = "-" * 100
    AUTO_HEADER = "Automatic Radiological Report"

    rep_dir = pwi_root / "reporting"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_txt = rep_dir / f"{subject_id}_automatic_radiological_report.txt"

    if dwi_report_path is not None and dwi_report_path.exists():
        base = dwi_report_path.read_text(encoding="utf-8", errors="ignore")

        header_idx = base.find(AUTO_HEADER)
        if header_idx != -1:
            sep_idx = base.find(SEPARATOR, header_idx)
            if sep_idx != -1:
                before = base[:sep_idx].rstrip()
                after = base[sep_idx:]
                insert_block = hp_line if mismatch_line is None else f"{hp_line}\n{mismatch_line}"
                merged = before + "\n\n" + insert_block + "\n\n" + after
            else:
                insert_block = hp_line if mismatch_line is None else f"{hp_line}\n{mismatch_line}"
                merged = base.rstrip() + "\n\n" + insert_block + "\n"
        else:
            insert_block = hp_line if mismatch_line is None else f"{hp_line}\n{mismatch_line}"
            merged = base.rstrip() + "\n\n" + insert_block + "\n"

        out_txt.write_text(merged, encoding="utf-8")

    else:
        insert_block = hp_line if mismatch_line is None else f"{hp_line}\n{mismatch_line}"
        minimal = HEADER + "\n\n" + insert_block + "\n\n" + FOOTER
        out_txt.write_text(minimal, encoding="utf-8")

    return out_txt
