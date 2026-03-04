"""Modular QFV calculation pipelines - each atlas independently.

This module provides individual functions for each atlas type,
allowing flexible usage - calculate only what you need.

Examples:
    # Calculate only vascular QFV:
    result = calculate_vascular_qfv(subject_dir, subject_id, atlas_dir)

    # Calculate only lobe QFV:
    result = calculate_lobe_qfv(subject_dir, subject_id, atlas_dir)

    # Calculate all atlases:
    all_results = calculate_all_qfv(subject_dir, subject_id, atlas_dir)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from ads.adapters.reporting import AtlasLoader, QFVCSVWriter
from ads.domain.reporting import (
    VascularQFVResult,
    LobeQFVResult,
    AspectsQFVResult,
    AspectsPCQFVResult,
    VentriclesQFVResult,
    AllQFVResults,
)
from ads.services.reporting import (
    VascularQFVCalculator,
    LobeQFVCalculator,
    AspectsQFVCalculator,
    AspectsPCQFVCalculator,
    VentriclesQFVCalculator,
)

logger = logging.getLogger("ADS.QFV.Modular")


def _load_lesion_and_voxel_size(
    subject_dir: Path,
    subject_id: str,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Helper to load lesion mask and get voxel size."""
    lesion_path = subject_dir / "segment" / f"{subject_id}_stroke-mask_space-MNI152_affsyn.nii.gz"

    if not lesion_path.exists():
        raise FileNotFoundError(f"Lesion mask not found: {lesion_path}")

    img = nib.load(str(lesion_path))
    lesion_mask = np.squeeze(nib.as_closest_canonical(img).get_fdata())

    # Get voxel size
    zooms = img.header.get_zooms()[:3]
    voxel_size_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    return lesion_mask, voxel_size_mm


def calculate_vascular_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
) -> VascularQFVResult:
    """Calculate ONLY vascular territory QFV.

    Independent function - doesn't require calculating other atlases.

    Args:
        subject_dir: Subject directory containing segment/ folder
        subject_id: Subject identifier
        atlas_dir: Directory containing atlas templates
        save_csv: Whether to save CSV output

    Returns:
        VascularQFVResult with QFV values

    Example:
        >>> result = calculate_vascular_qfv(
        ...     subject_dir=Path("/data/sub-001"),
        ...     subject_id="sub-001",
        ...     atlas_dir=Path("/atlases")
        ... )
        >>> print(result.qfv)  # Only vascular QFV
    """
    logger.info(f"Calculating vascular QFV for {subject_id}")

    # 1. Load lesion mask
    lesion_mask, voxel_size_mm = _load_lesion_and_voxel_size(subject_dir, subject_id)

    # 2. Load vascular atlas only
    atlas_loader = AtlasLoader(atlas_dir)
    vascular_template = atlas_loader.load_template("vascular")
    vascular_volumes = atlas_loader.load_volume_table("vascular")

    if vascular_template is None or vascular_volumes is None:
        raise ValueError("Vascular atlas not found")

    # 3. Calculate vascular QFV
    calculator = VascularQFVCalculator()
    result = calculator.calculate(
        lesion_mask=lesion_mask,
        vascular_template=vascular_template,
        vascular_volumes=vascular_volumes,
        voxel_size_mm=voxel_size_mm,
        subject_id=subject_id,
    )

    # 4. Save CSV if requested
    if save_csv:
        output_dir = subject_dir / "reporting"
        output_dir.mkdir(exist_ok=True)

        csv_path = output_dir / f"{subject_id}_VASCULAR_QFV.csv"
        import pandas as pd
        df = pd.DataFrame([result.qfv], columns=result.roi_labels)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved vascular QFV to {csv_path}")

    return result


def calculate_lobe_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
) -> LobeQFVResult:
    """Calculate ONLY brain lobe QFV.

    Independent function - doesn't require calculating other atlases.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        save_csv: Whether to save CSV

    Returns:
        LobeQFVResult
    """
    logger.info(f"Calculating lobe QFV for {subject_id}")

    lesion_mask, voxel_size_mm = _load_lesion_and_voxel_size(subject_dir, subject_id)

    atlas_loader = AtlasLoader(atlas_dir)
    lobe_template = atlas_loader.load_template("lobe")
    lobe_volumes = atlas_loader.load_volume_table("lobe")

    if lobe_template is None or lobe_volumes is None:
        raise ValueError("Lobe atlas not found")

    calculator = LobeQFVCalculator()
    result = calculator.calculate(
        lesion_mask=lesion_mask,
        lobe_template=lobe_template,
        lobe_volumes=lobe_volumes,
        voxel_size_mm=voxel_size_mm,
        subject_id=subject_id,
    )

    if save_csv:
        output_dir = subject_dir / "reporting"
        output_dir.mkdir(exist_ok=True)

        csv_path = output_dir / f"{subject_id}_LOBE_QFV.csv"
        import pandas as pd
        df = pd.DataFrame([result.qfv], columns=result.roi_labels)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved lobe QFV to {csv_path}")

    return result


def calculate_aspects_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
) -> AspectsQFVResult:
    """Calculate ONLY ASPECTS region QFV.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        save_csv: Whether to save CSV

    Returns:
        AspectsQFVResult
    """
    logger.info(f"Calculating ASPECTS QFV for {subject_id}")

    lesion_mask, voxel_size_mm = _load_lesion_and_voxel_size(subject_dir, subject_id)

    atlas_loader = AtlasLoader(atlas_dir)
    aspects_template = atlas_loader.load_template("aspects")
    aspects_volumes = atlas_loader.load_volume_table("aspects")

    if aspects_template is None or aspects_volumes is None:
        raise ValueError("ASPECTS atlas not found")

    calculator = AspectsQFVCalculator()
    result = calculator.calculate(
        lesion_mask=lesion_mask,
        aspects_template=aspects_template,
        aspects_volumes=aspects_volumes,
        voxel_size_mm=voxel_size_mm,
        subject_id=subject_id,
    )

    if save_csv:
        output_dir = subject_dir / "reporting"
        output_dir.mkdir(exist_ok=True)

        csv_path = output_dir / f"{subject_id}_ASPECTS_QFV.csv"
        import pandas as pd
        df = pd.DataFrame([result.qfv], columns=result.roi_labels)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved ASPECTS QFV to {csv_path}")

    return result


def calculate_aspectpc_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
) -> Optional[AspectsPCQFVResult]:
    """Calculate ONLY PCA-ASPECTS QFV (optional).

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        save_csv: Whether to save CSV

    Returns:
        AspectsPCQFVResult or None if atlas not available
    """
    logger.info(f"Calculating PCA-ASPECTS QFV for {subject_id}")

    try:
        lesion_mask, voxel_size_mm = _load_lesion_and_voxel_size(subject_dir, subject_id)

        atlas_loader = AtlasLoader(atlas_dir)
        aspectpc_template = atlas_loader.load_template("aspectpc")
        aspectpc_volumes = atlas_loader.load_volume_table("aspectpc")

        if aspectpc_template is None or aspectpc_volumes is None:
            logger.warning("PCA-ASPECTS atlas not found, skipping")
            return None

        calculator = AspectsPCQFVCalculator()
        result = calculator.calculate(
            lesion_mask=lesion_mask,
            aspectpc_template=aspectpc_template,
            aspectpc_volumes=aspectpc_volumes,
            voxel_size_mm=voxel_size_mm,
            subject_id=subject_id,
        )

        if save_csv:
            output_dir = subject_dir / "reporting"
            output_dir.mkdir(exist_ok=True)

            csv_path = output_dir / f"{subject_id}_ASPECTSPC_QFV.csv"
            import pandas as pd
            df = pd.DataFrame([result.qfv], columns=result.roi_labels)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved PCA-ASPECTS QFV to {csv_path}")

        return result

    except Exception as e:
        logger.warning(f"Failed to calculate PCA-ASPECTS QFV: {e}")
        return None


def calculate_ventricles_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
) -> VentriclesQFVResult:
    """Calculate ONLY ventricular region QFV.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        save_csv: Whether to save CSV

    Returns:
        VentriclesQFVResult
    """
    logger.info(f"Calculating ventricles QFV for {subject_id}")

    lesion_mask, voxel_size_mm = _load_lesion_and_voxel_size(subject_dir, subject_id)

    atlas_loader = AtlasLoader(atlas_dir)
    ventricles_template = atlas_loader.load_template("Ventricles")
    ventricles_volumes = atlas_loader.load_volume_table("Ventricles")

    if ventricles_template is None or ventricles_volumes is None:
        raise ValueError("Ventricles atlas not found")

    calculator = VentriclesQFVCalculator()
    result = calculator.calculate(
        lesion_mask=lesion_mask,
        ventricles_template=ventricles_template,
        ventricles_volumes=ventricles_volumes,
        voxel_size_mm=voxel_size_mm,
        subject_id=subject_id,
    )

    if save_csv:
        output_dir = subject_dir / "reporting"
        output_dir.mkdir(exist_ok=True)

        csv_path = output_dir / f"{subject_id}_VENTRICLES_QFV.csv"
        import pandas as pd
        df = pd.DataFrame([result.qfv], columns=result.roi_labels)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved ventricles QFV to {csv_path}")

    return result


def calculate_all_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    save_csv: bool = True,
    include_aspectpc: bool = True,
) -> AllQFVResults:
    """Calculate ALL atlas QFVs at once.

    Orchestrates individual calculators for convenience.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        save_csv: Whether to save CSV outputs
        include_aspectpc: Whether to include PCA-ASPECTS (optional)

    Returns:
        AllQFVResults containing all atlas results

    Example:
        >>> results = calculate_all_qfv(
        ...     subject_dir=Path("/data/sub-001"),
        ...     subject_id="sub-001",
        ...     atlas_dir=Path("/atlases")
        ... )
        >>> print(results.vascular.qfv)  # Access vascular
        >>> print(results.lobe.qfv)      # Access lobe
    """
    logger.info(f"Calculating ALL atlas QFVs for {subject_id}")

    # Calculate each atlas independently
    vascular = calculate_vascular_qfv(subject_dir, subject_id, atlas_dir, save_csv)
    lobe = calculate_lobe_qfv(subject_dir, subject_id, atlas_dir, save_csv)
    aspects = calculate_aspects_qfv(subject_dir, subject_id, atlas_dir, save_csv)
    ventricles = calculate_ventricles_qfv(subject_dir, subject_id, atlas_dir, save_csv)

    aspectpc = None
    if include_aspectpc:
        aspectpc = calculate_aspectpc_qfv(subject_dir, subject_id, atlas_dir, save_csv)

    # Combine into single result
    all_results = AllQFVResults(
        subject_id=subject_id,
        vascular=vascular,
        lobe=lobe,
        aspects=aspects,
        ventricles=ventricles,
        aspectpc=aspectpc,
        icv_volume_ml=0.0,  # TODO: Calculate ICV if needed
    )

    logger.info(f"Completed all QFV calculations for {subject_id}")

    return all_results


def calculate_selected_qfv(
    subject_dir: Path,
    subject_id: str,
    atlas_dir: Path,
    atlas_list: List[str],
    save_csv: bool = True,
) -> Dict[str, object]:
    """Calculate QFV for selected atlases only.

    Flexible function to calculate any combination of atlases.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        atlas_dir: Atlas directory
        atlas_list: List of atlas names to calculate (e.g., ["vascular", "lobe"])
        save_csv: Whether to save CSV outputs

    Returns:
        Dictionary mapping atlas name -> result object

    Example:
        >>> results = calculate_selected_qfv(
        ...     subject_dir=Path("/data/sub-001"),
        ...     subject_id="sub-001",
        ...     atlas_dir=Path("/atlases"),
        ...     atlas_list=["vascular", "lobe"]  # Only these two
        ... )
        >>> print(results["vascular"].qfv)
    """
    results = {}

    calculator_map = {
        "vascular": calculate_vascular_qfv,
        "lobe": calculate_lobe_qfv,
        "aspects": calculate_aspects_qfv,
        "aspectpc": calculate_aspectpc_qfv,
        "ventricles": calculate_ventricles_qfv,
    }

    for atlas_name in atlas_list:
        if atlas_name not in calculator_map:
            logger.warning(f"Unknown atlas: {atlas_name}, skipping")
            continue

        calculator_func = calculator_map[atlas_name]
        result = calculator_func(subject_dir, subject_id, atlas_dir, save_csv)
        results[atlas_name] = result

    return results
