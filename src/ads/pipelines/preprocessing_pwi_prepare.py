#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PWI preprocessing pipeline - Clean architecture implementation.

This module orchestrates the PWI preprocessing workflow, delegating:
- I/O operations to adapters (NiftiLoader, NiftiSaver, PWILayoutResolver)
- Algorithm logic to services (ADCCalculator, VolumeOrderDetector)
- Configuration to ConfigLoader

The pipeline itself contains ONLY orchestration logic.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import nibabel as nib
import numpy as np

# Core imports
from ads.core.config_loader import ConfigLoader

# Adapter imports
from ads.adapters.subject_discovery import InputOverrides
from ads.adapters.pwi_discovery import PWILayoutResolver
from ads.adapters.metadata_io import JsonSidecarHandler
from ads.adapters.nifti.loader import NiftiLoader
from ads.adapters.nifti.saver import NiftiSaver

# Service imports
from ads.services.preprocessing.adc_calculator import ADCCalculator
from ads.services.preprocessing.volume_order_detector import VolumeOrderDetector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PWIPrepConfig:
    """Configuration for PWI preprocessing pipeline.

    This is a data class - no business logic here.
    """
    subject_dir: Path
    subject_id: str
    base_id: str
    output_root: Path
    b_value: float
    force_adc_calc: bool
    check_order: bool
    overrides: InputOverrides
    yaml_paths: Dict[str, Optional[Path]]


class PWIRawDataProcessor:
    """PWI preprocessing pipeline orchestrator.

    This class ONLY orchestrates the workflow. All actual work is delegated to:
    - Adapters for I/O (NiftiLoader, NiftiSaver, PWILayoutResolver, JsonSidecarHandler)
    - Services for algorithms (ADCCalculator, VolumeOrderDetector)

    NO direct nibabel calls. NO algorithm implementation. Pure orchestration.
    """

    def __init__(self, config: PWIPrepConfig, logger: logging.Logger):
        """Initialize pipeline with configuration.

        Args:
            config: PWI preprocessing configuration
            logger: Logger instance
        """
        self.cfg = config
        self.logger = logger
        self.output_dir = self.cfg.output_root / self.cfg.subject_id / "PWI" / "preprocess"

        # Initialize adapters (I/O and external libs)
        self.layout_resolver = PWILayoutResolver(
            subject_dir=self.cfg.subject_dir,
            subject_id=self.cfg.subject_id,
            base_id=self.cfg.base_id,
            overrides=self.cfg.overrides
        )
        self.json_handler = JsonSidecarHandler()
        self.nifti_loader = NiftiLoader()
        self.nifti_saver = NiftiSaver()

        # Initialize services (business logic)
        self.adc_calculator = ADCCalculator(bvalue=self.cfg.b_value)
        self.order_detector = VolumeOrderDetector()

    def run(self) -> Dict[str, Path]:
        """Execute PWI preprocessing pipeline.

        Returns:
            Dictionary mapping modality names to output paths
        """
        self.logger.info(f"--- Starting processing for subject: {self.cfg.subject_id} ---")
        self._setup_directories()

        # Process DWI/ADC/B0 (same as DWI pipeline)
        dwi_path = self.layout_resolver.find_image("DWI")
        has_true_dwi = dwi_path is not None
        if not dwi_path:
            adc_fallback_path = self.layout_resolver.find_image("ADC")
            if not adc_fallback_path:
                raise FileNotFoundError(
                    f"Both DWI and ADC images not found for {self.cfg.subject_id}"
                )
            self.logger.warning(
                f"DWI image not found for {self.cfg.subject_id}. "
                f"Using ADC as fallback input for preprocessing."
            )
            dwi_path = adc_fallback_path

        dwi_raw_nii = self.nifti_loader.load(dwi_path)
        dwi_nii, b0_nii = self._process_dwi_volumes(dwi_raw_nii)
        adc_nii = self._get_or_calculate_adc(dwi_nii, b0_nii)

        # Save main files
        saved_paths = self._save_main_files(dwi_nii, adc_nii, b0_nii, save_dwi=has_true_dwi)

        # Process PWI-specific files
        self._process_pwi_files(saved_paths)

        self.logger.info(f"--- Successfully completed for subject: {self.cfg.subject_id} ---")
        return saved_paths

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _process_dwi_volumes(
        self,
        dwi_raw_nii: nib.Nifti1Image
    ) -> Tuple[nib.Nifti1Image, Optional[nib.Nifti1Image]]:
        """Process DWI volumes (4D split or 3D with separate B0).

        Delegates algorithm to VolumeOrderDetector service.

        Args:
            dwi_raw_nii: Raw loaded DWI image

        Returns:
            Tuple of (dwi_3d, b0_3d) where b0_3d may be None

        Raises:
            ValueError: If DWI has unsupported dimensions
        """
        data = dwi_raw_nii.get_fdata()

        if data.ndim == 3:
            # 3D DWI - look for separate B0 file
            self.logger.info("Input DWI is 3D. Looking for a separate b0 file.")
            b0_path = self.layout_resolver.find_image("b0")
            b0_nii = self.nifti_loader.load(b0_path) if b0_path else None
            return dwi_raw_nii, b0_nii

        elif data.ndim == 4:
            # 4D DWI - split and possibly reorder
            self.logger.info("Input DWI is 4D. Splitting volumes...")
            vol0_data = np.asarray(data[..., 0])
            vol1_data = np.asarray(data[..., 1])

            # Use service to detect if order is swapped
            if self.cfg.check_order and self.order_detector.is_order_swapped(vol0_data, vol1_data):
                self.logger.info("Detected volume order swap. Correcting: [B0, DWI] -> [DWI, B0]")
                dwi_data, b0_data = vol1_data, vol0_data
            else:
                b0_data, dwi_data = vol0_data, vol1_data

            # Create new 3D images
            affine, header = dwi_raw_nii.affine, dwi_raw_nii.header
            dwi_nii = nib.Nifti1Image(dwi_data, affine, header)
            b0_nii = nib.Nifti1Image(b0_data, affine, header)

            return dwi_nii, b0_nii

        else:
            raise ValueError(f"Unsupported DWI dimensions: {data.ndim}D (expected 3D or 4D)")

    def _get_or_calculate_adc(
        self,
        dwi_nii: nib.Nifti1Image,
        b0_nii: Optional[nib.Nifti1Image]
    ) -> nib.Nifti1Image:
        """Get existing ADC or calculate from DWI and B0.

        Delegates calculation to ADCCalculator service.

        Args:
            dwi_nii: DWI image
            b0_nii: B0 image (may be None)

        Returns:
            ADC image

        Raises:
            FileNotFoundError: If ADC not found and cannot be calculated
        """
        # Try to load existing ADC first
        adc_path = self.layout_resolver.find_image("ADC")
        if adc_path and not self.cfg.force_adc_calc:
            self.logger.info(f"Found existing ADC: {adc_path.name}")
            return self.nifti_loader.load(adc_path)

        # Calculate ADC using service
        if not b0_nii:
            raise FileNotFoundError("Cannot compute ADC: b0 volume unavailable.")

        self.logger.info("Calculating ADC from DWI and B0...")
        adc_nii = self.adc_calculator.compute(dwi_nii, b0_nii)
        self.logger.info("ADC calculation complete.")

        return adc_nii

    def _save_main_files(
        self,
        dwi_nii: nib.Nifti1Image,
        adc_nii: nib.Nifti1Image,
        b0_nii: Optional[nib.Nifti1Image],
        save_dwi: bool = True,
    ) -> Dict[str, Path]:
        """Save main processed files (DWI, ADC, B0).

        Args:
            dwi_nii: DWI image
            adc_nii: ADC image
            b0_nii: B0 image (may be None)

        Returns:
            Dictionary of saved file paths
        """
        self.logger.info("Saving processed NIfTI files...")

        saved_paths = {}

        # Save DWI only when real DWI input exists; avoid creating fake DWI from ADC fallback.
        if save_dwi:
            dwi_out = self.output_dir / f"{self.cfg.subject_id}_DWI.nii.gz"
            self.nifti_saver.save(dwi_nii, dwi_out)
            saved_paths["dwi"] = dwi_out
        else:
            self.logger.info("Skipping DWI save: real DWI input not found (ADC fallback mode).")

        # Save ADC
        adc_out = self.output_dir / f"{self.cfg.subject_id}_ADC.nii.gz"
        self.nifti_saver.save(adc_nii, adc_out)
        saved_paths["adc"] = adc_out

        # Save B0 if available
        if b0_nii:
            b0_out = self.output_dir / f"{self.cfg.subject_id}_B0.nii.gz"
            self.nifti_saver.save(b0_nii, b0_out)
            saved_paths["b0"] = b0_out

        return saved_paths

    def _process_pwi_files(self, saved_paths: Dict[str, Path]) -> None:
        """Process PWI-specific files (PWI image, HP_manual, JSON sidecar).

        This is orchestration logic - finding, loading, and saving files.
        No algorithm logic here.

        Args:
            saved_paths: Dictionary to update with PWI file paths
        """
        # Process PWI 4D image
        pwi_path = self.layout_resolver.find_image("PWI")
        if pwi_path:
            self.logger.info(f"Found PWI image: {pwi_path.name}")
            pwi_nii = self.nifti_loader.load(pwi_path)
            pwi_out = self.output_dir / f"{self.cfg.subject_id}_PWI.nii.gz"
            self.nifti_saver.save(pwi_nii, pwi_out)
            saved_paths["pwi"] = pwi_out

        # Process HP_manual (with alias handling for HP_manual2)
        hp_path = self.layout_resolver.find_image_with_alias("HP_manual")
        if hp_path:
            self.logger.info(f"Found HP_manual: {hp_path.name}")
            hp_nii = self.nifti_loader.load(hp_path)
            # Always save as HP_manual (even if input was HP_manual2)
            hp_out = self.output_dir / f"{self.cfg.subject_id}_HP_manual.nii.gz"
            self.nifti_saver.save(hp_nii, hp_out)
            saved_paths["hp_manual"] = hp_out
            self.logger.info(f"Saved HP_manual image: {hp_out}")

        # Process stroke mask (shared with DWI)
        stroke_path = self.layout_resolver.find_image("STROKE")
        if stroke_path:
            self.logger.info(f"Found stroke mask: {stroke_path.name}")
            stroke_nii = self.nifti_loader.load(stroke_path)
            stroke_out = self.output_dir / f"{self.cfg.subject_id}_stroke.nii.gz"
            self.nifti_saver.save(stroke_nii, stroke_out)
            saved_paths["stroke"] = stroke_out
            self.logger.info(f"Saved stroke image: {stroke_out}")

        # Process PWI JSON sidecar
        json_path = self.json_handler.find_json(
            self.cfg.subject_dir,
            self.cfg.subject_id,
            self.cfg.base_id,
            "PWI"
        )
        if json_path:
            json_out = self.output_dir / f"{self.cfg.subject_id}_PWI.json"
            self.json_handler.copy_json(json_path, json_out)
            saved_paths["pwi_json"] = json_out


def process_raw_single_subject(
    subject_dir: Path,
    output_root: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
    overrides: Optional[InputOverrides] = None
) -> Dict[str, Path]:
    """Process a single subject's raw PWI data.

    Public API function for PWI preprocessing.

    Args:
        subject_dir: Path to subject directory
        output_root: Root output directory
        config: Configuration dictionary
        logger: Logger instance
        overrides: Optional CLI path overrides

    Returns:
        Dictionary of output file paths

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data format is invalid
    """
    try:
        # Extract configuration
        data_cfg = config.get('data_raw', {})

        # Normalize subject ID
        subject_name = subject_dir.name
        if subject_name.startswith('sub-'):
            subject_id = subject_name
            base_id = subject_name[4:]
        else:
            subject_id = f"sub-{subject_name}"
            base_id = subject_name

        # Create preprocessing config
        prep_config = PWIPrepConfig(
            subject_dir=subject_dir,
            subject_id=subject_id,
            base_id=base_id,
            output_root=output_root,
            b_value=float(data_cfg.get('bvalue', 1000.0)),
            force_adc_calc=bool(data_cfg.get('force_adc_calc', False)),
            check_order=bool(data_cfg.get('detect_dwi_b0_chann', True)),
            overrides=overrides or InputOverrides(),
            yaml_paths={
                "DWI": Path(data_cfg["dwi_path"]) if data_cfg.get("dwi_path") else None,
                "ADC": Path(data_cfg["adc_path"]) if data_cfg.get("adc_path") else None,
                "B0": Path(data_cfg["b0_path"]) if data_cfg.get("b0_path") else None,
                "STROKE": Path(data_cfg["stroke_path"]) if data_cfg.get("stroke_path") else None,
                "PWI": Path(data_cfg["pwi_path"]) if data_cfg.get("pwi_path") else None,
            }
        )

        # Run pipeline
        processor = PWIRawDataProcessor(prep_config, logger)
        return processor.run()

    except Exception as e:
        logger.error(f"FAILED processing for {subject_dir.name}: {e}")
        raise
