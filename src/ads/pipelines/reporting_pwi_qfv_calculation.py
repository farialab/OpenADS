"""PWI HP load/QFV CSV generation using DWI-proven QFV pipeline logic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from ads.pipelines.reporting_qfv_calculation import StrokeQFVCalculator

logger = logging.getLogger("ADS.PWI.QFV")


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def process_pwi_qfv_single(
    template_dir: str | Path,
    pwi_dir: str | Path,
    output_dir: str | Path,
    subject_id: str,
    hp_mask_path: str | Path,
) -> Dict[str, Path]:
    """Generate PWI HPload/HPQFV CSVs.

    Reuses the DWI calculator end-to-end:
    - stroke mask -> HP mask
    - same atlas overlap
    - same QFV conversion
    """
    template_dir = Path(template_dir)
    pwi_dir = Path(pwi_dir)
    output_dir = Path(output_dir)
    hp_mask_path = Path(hp_mask_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hp_mask_path.exists():
        raise FileNotFoundError(f"HP mask not found: {hp_mask_path}")
    if "affsyn" not in hp_mask_path.name:
        raise ValueError(
            f"PWI reporting requires affsyn HP source, got: {hp_mask_path.name}"
        )

    pwi_reg = pwi_dir / "registration"
    dwi_reg = pwi_dir.parent / "DWI" / "registration"
    reg_priority = [pwi_reg, dwi_reg]

    mask_path = _first_existing(
        [d / f"{subject_id}_DWIbrain-mask_space-MNI152_affsyn.nii.gz" for d in reg_priority]
    )
    adc_path = _first_existing(
        [d / f"{subject_id}_ADC_space-MNI152_affsyn.nii.gz" for d in reg_priority]
    )
    if mask_path is None:
        raise FileNotFoundError(
            f"affsyn mask not found in PWI/DWI registration for {subject_id} "
            f"(expected *_DWIbrain-mask_space-MNI152_affsyn.nii.gz)"
        )
    if adc_path is None:
        raise FileNotFoundError(
            f"affsyn ADC not found in PWI/DWI registration for {subject_id} "
            f"(expected *_ADC_space-MNI152_affsyn.nii.gz)"
        )

    builder = StrokeQFVCalculator(template_dir=str(template_dir))
    results = builder.calculate(
        stroke_img_path=str(hp_mask_path),
        mask_raw_mni_path=str(mask_path),
        adc_mni_path=str(adc_path),
        adc_threshold=0.5490,
        precision=7,
    )
    builder.save_QFV_to_csv(
        results,
        subject_id=subject_id,
        output_dir=str(output_dir),
        qfv_suffix="HPQFV",
        lesionload_suffix="HPload",
        qfv_stem_map={
            "Vascular": "Vascular",
            "Lobe": "Lobe",
            "Aspects": "ASPECTS",
            "AspectsPC": "ASPECTSPC",
            "BPM": "BPM_TYPE1",
        },
        lesionload_stem_map={
            "vascular": "Vascular",
            "lobe": "Lobe",
            "aspects": "ASPECTS",
            "aspectpc": "ASPECTSPC",
            "bpm_type1": "BPM_TYPE1",
        },
    )

    expected = [
        f"{subject_id}_Vascular_HPQFV.csv",
        f"{subject_id}_Lobe_HPQFV.csv",
        f"{subject_id}_ASPECTS_HPQFV.csv",
        f"{subject_id}_ASPECTSPC_HPQFV.csv",
        f"{subject_id}_BPM_TYPE1_HPQFV.csv",
        f"{subject_id}_Vascular_HPload.csv",
        f"{subject_id}_Lobe_HPload.csv",
        f"{subject_id}_ASPECTS_HPload.csv",
        f"{subject_id}_ASPECTSPC_HPload.csv",
        f"{subject_id}_BPM_TYPE1_HPload.csv",
        f"{subject_id}_VENTRICLES_HPQFV.csv",
        f"{subject_id}_VENTRICLES_HPload.csv",
    ]
    return {name: (output_dir / name) for name in expected if (output_dir / name).exists()}
