from __future__ import annotations
"""Prepares raw DWI and ADC data for downstream processing.

This pipeline orchestrates the preprocessing workflow:
1. Discovers input files based on subject directory layout
2. Loads and validates DWI images (4D or 3D)
3. Detects and corrects DWI/B0 volume order if needed
4. Computes ADC maps when required
5. Saves processed outputs to standardized locations

The pipeline delegates all business logic to services and all I/O to adapters.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import nibabel as nib
import numpy as np

# Import from architecture layers
from ads.core.config_loader import ConfigLoader
from ads.adapters import SubjectLayoutResolver, InputOverrides
from ads.adapters.nifti import NiftiLoader, NiftiSaver
from ads.services.preprocessing import ADCCalculator, VolumeOrderDetector


class PrepConfig:
    """Preprocessing configuration for a single subject."""

    def __init__(
        self,
        subject_dir: Path,
        output_root: Path,
        config_dict: Dict[str, Any],
        overrides: Optional[InputOverrides] = None
    ):
        """Initialize preprocessing configuration.

        Args:
            subject_dir: Path to subject input directory
            output_root: Path to output root directory
            config_dict: Configuration dictionary from YAML
            overrides: Optional explicit path overrides
        """
        data_cfg = config_dict.get("data_raw", {})
        self.subject_dir = Path(subject_dir)
        self.output_root = Path(output_root)
        self.b_value = float(data_cfg.get("bvalue", 1000.0))
        self.force_adc_calc = bool(data_cfg.get("force_adc_calc", False))
        self.check_order = bool(data_cfg.get("detect_dwi_b0_chann", True))

        # Parse subject ID
        folder_name = subject_dir.name
        if folder_name.startswith("sub-"):
            self.subject_id = folder_name
            self.base_id = folder_name[4:]
        else:
            self.subject_id = folder_name
            self.base_id = folder_name

        # Merge CLI overrides with YAML overrides
        yaml_overrides = InputOverrides(
            dwi=Path(data_cfg["dwi_path"]) if data_cfg.get("dwi_path") else None,
            adc=Path(data_cfg["adc_path"]) if data_cfg.get("adc_path") else None,
            b0=Path(data_cfg["b0_path"]) if data_cfg.get("b0_path") else None,
            stroke=Path(data_cfg["stroke_path"]) if data_cfg.get("stroke_path") else None,
        )

        # CLI overrides take precedence
        self.overrides = InputOverrides(
            dwi=overrides.dwi if overrides and overrides.dwi else yaml_overrides.dwi,
            adc=overrides.adc if overrides and overrides.adc else yaml_overrides.adc,
            b0=overrides.b0 if overrides and overrides.b0 else yaml_overrides.b0,
            stroke=overrides.stroke if overrides and overrides.stroke else yaml_overrides.stroke,
        ) if overrides or any([yaml_overrides.dwi, yaml_overrides.adc, yaml_overrides.b0, yaml_overrides.stroke]) else InputOverrides()


class RawDataProcessor:
    """Orchestrates the preprocessing pipeline for a single subject.

    This class coordinates between:
    - SubjectLayoutResolver (finding inputs)
    - NiftiLoader/NiftiSaver (I/O)
    - VolumeOrderDetector (DWI/B0 order detection)
    - ADCCalculator (ADC computation)

    The processor itself contains NO business logic or I/O code.
    """

    def __init__(self, config: PrepConfig, logger: logging.Logger):
        """Initialize processor.

        Args:
            config: Preprocessing configuration
            logger: Logger instance
        """
        self.cfg = config
        self.logger = logger
        self.output_dir = self.cfg.output_root / self.cfg.subject_id / "DWI" / "preprocess"

        # Initialize services
        self.layout_resolver = SubjectLayoutResolver(
            subject_dir=self.cfg.subject_dir,
            subject_id=self.cfg.subject_id,
            base_id=self.cfg.base_id,
            overrides=self.cfg.overrides
        )
        self.adc_calculator = ADCCalculator(bvalue=self.cfg.b_value)
        self.order_detector = VolumeOrderDetector()
        self.nifti_loader = NiftiLoader()
        self.nifti_saver = NiftiSaver()

    def run(self) -> Dict[str, Path]:
        """Execute preprocessing pipeline.

        Returns:
            Dictionary of output file paths

        Raises:
            FileNotFoundError: If required inputs are missing
        """
        self.logger.info(f"--- Starting processing for subject: {self.cfg.subject_id} ---")
        self._setup_directories()

        # 1. Discover and load DWI
        dwi_path = self.layout_resolver.find_image("DWI")
        if not dwi_path:
            raise FileNotFoundError(f"DWI image not found for {self.cfg.subject_id}")

        self.logger.info(f"Found DWI image: {dwi_path}")
        dwi_raw_nii = self.nifti_loader.load(dwi_path)

        # 2. Process DWI volumes (4D split or 3D passthrough)
        dwi_nii, b0_nii = self._process_dwi_volumes(dwi_raw_nii)

        # 3. Get or calculate ADC
        adc_nii = self._get_or_calculate_adc(dwi_nii, b0_nii)

        # 4. Save processed volumes
        self.logger.info("Saving processed NIfTI files...")
        saved_paths = {}
        saved_paths["dwi"] = self.nifti_saver.save(
            dwi_nii,
            self.output_dir / f"{self.cfg.subject_id}_DWI.nii.gz"
        )
        saved_paths["adc"] = self.nifti_saver.save(
            adc_nii,
            self.output_dir / f"{self.cfg.subject_id}_ADC.nii.gz"
        )

        if b0_nii is not None:
            saved_paths["b0"] = self.nifti_saver.save(
                b0_nii,
                self.output_dir / f"{self.cfg.subject_id}_B0.nii.gz"
            )

        # 5. Handle stroke mask if available
        stroke_path = self._find_and_save_stroke()
        if stroke_path:
            saved_paths["stroke"] = stroke_path

        self.logger.info(f"--- Successfully completed for subject: {self.cfg.subject_id} ---")
        return saved_paths

    def _setup_directories(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Output directory set to: {self.output_dir}")

    def _process_dwi_volumes(
        self,
        dwi_raw_nii: nib.Nifti1Image
    ) -> Tuple[nib.Nifti1Image, Optional[nib.Nifti1Image]]:
        """Process DWI volumes (4D split or 3D passthrough).

        Args:
            dwi_raw_nii: Raw DWI image (3D or 4D)

        Returns:
            Tuple of (dwi_img, b0_img)

        Raises:
            ValueError: If image has unsupported dimensions
        """
        data = dwi_raw_nii.get_fdata()

        if data.ndim == 3:
            # 3D image - try to find separate B0
            self.logger.info("Input DWI is 3D. Looking for a separate b0 file.")
            b0_path = self.layout_resolver.find_image("b0")
            b0_nii = self.nifti_loader.load(b0_path) if b0_path else None
            return dwi_raw_nii, b0_nii

        elif data.ndim == 4:
            # 4D image - split volumes
            if data.shape[3] < 2:
                raise ValueError("4D DWI file must contain at least 2 volumes.")

            self.logger.info("Input DWI is 4D. Extracting volumes for DWI/b0.")

            vol0 = np.asarray(data[..., 0])
            vol1 = np.asarray(data[..., 1])

            # Detect order using service
            if self.cfg.check_order:
                b0_data, dwi_data = self.order_detector.detect_and_split(vol0, vol1)
            else:
                b0_data, dwi_data = vol0, vol1

            # Create NIfTI images
            affine, header = dwi_raw_nii.affine, dwi_raw_nii.header
            dwi_nii = nib.Nifti1Image(dwi_data, affine, header)
            b0_nii = nib.Nifti1Image(b0_data, affine, header)

            return dwi_nii, b0_nii

        else:
            raise ValueError(f"Unsupported DWI dimensions: {data.ndim}. Must be 3D or 4D.")

    def _get_or_calculate_adc(
        self,
        dwi_nii: nib.Nifti1Image,
        b0_nii: Optional[nib.Nifti1Image]
    ) -> nib.Nifti1Image:
        """Get existing ADC or calculate from DWI/B0.

        Args:
            dwi_nii: DWI image
            b0_nii: B0 image (may be None)

        Returns:
            ADC image

        Raises:
            FileNotFoundError: If ADC is required but cannot be computed
        """
        # Check for existing ADC
        adc_path = self.layout_resolver.find_image("ADC")

        if adc_path and not self.cfg.force_adc_calc:
            self.logger.info(f"Using existing ADC file: {adc_path}")
            return self.nifti_loader.load(adc_path)

        if adc_path and self.cfg.force_adc_calc:
            self.logger.warning("ADC file found but 'force_adc_calc' is True. Recalculating ADC.")

        # Calculate ADC
        if not b0_nii:
            raise FileNotFoundError(
                "Cannot compute ADC: b0 volume is not available and no valid ADC file found."
            )

        self.logger.info("Calculating ADC from DWI and B0...")
        adc_nii = self.adc_calculator.compute(dwi_nii, b0_nii, save_path=None)
        self.logger.info("ADC calculation complete.")

        return adc_nii

    def _find_and_save_stroke(self) -> Optional[Path]:
        """Find and save stroke mask if available.

        Returns:
            Path to saved stroke mask, or None if not found
        """
        stroke_path = self.layout_resolver.find_image("STROKE")

        if stroke_path:
            stroke_nii = self.nifti_loader.load(stroke_path)
            output_path = self.nifti_saver.save(
                stroke_nii,
                self.output_dir / f"{self.cfg.subject_id}_stroke.nii.gz"
            )
            self.logger.info(f"Saved stroke image: {output_path}")
            return output_path

        # Try fixed stroke variant
        stroke_path = self.layout_resolver.find_image("STROKE_FIXED")
        if stroke_path:
            stroke_nii = self.nifti_loader.load(stroke_path)
            output_path = self.nifti_saver.save(
                stroke_nii,
                self.output_dir / f"{self.cfg.subject_id}_stroke.nii.gz"
            )
            self.logger.info(f"Saved fixed stroke image: {output_path}")
            return output_path

        self.logger.info("No stroke image found.")
        return None


def process_raw_single_subject(
    subject_dir: Path,
    output_root: Path,
    config: Dict[str, Any],
    logger: logging.Logger,
    overrides: Optional[InputOverrides] = None,
) -> None:
    """Process a single subject directory.

    This is the main entry point for the preprocessing pipeline.

    Args:
        subject_dir: Path to the subject's input directory
        output_root: Path to the root output directory
        config: Configuration dictionary loaded from YAML
        logger: Logger instance for logging messages
        overrides: Optional explicit path overrides

    Raises:
        Exception: If processing fails for any reason
    """
    try:
        prep_config = PrepConfig(subject_dir, output_root, config, overrides=overrides)
        processor = RawDataProcessor(prep_config, logger)
        processor.run()
    except Exception as e:
        logger.error(f"FAILED processing for {subject_dir.name}: {e}")
        raise


def main():
    """Parses config and runs data preparation for a single subject.

    This is the CLI entry point when running as a script.
    """
    parser = argparse.ArgumentParser(
        description="Prepare raw DWI data using a YAML config file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--input_subdir",
        type=Path,
        required=True,
        help="Directory containing the subject's files."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save processed outputs."
    )
    parser.add_argument(
        "--dwi",
        type=Path,
        default=None,
        help="Explicit path to DWI NIfTI. Overrides auto discovery."
    )
    parser.add_argument(
        "--adc",
        type=Path,
        default=None,
        help="Explicit path to ADC NIfTI. Overrides auto discovery."
    )
    parser.add_argument(
        "--b0",
        type=Path,
        default=None,
        help="Explicit path to B0 NIfTI. Overrides auto discovery."
    )
    parser.add_argument(
        "--stroke",
        type=Path,
        default=None,
        help="Explicit path to stroke mask NIfTI. Overrides auto discovery."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # Load configuration using improved loader
        config_from_yaml = ConfigLoader.load(args.config)

        subject_dir = args.input_subdir
        if not subject_dir.is_dir():
            logger.error(f"Subject directory not found: {subject_dir}")
            return

        # Create overrides
        overrides = InputOverrides(
            dwi=args.dwi,
            adc=args.adc,
            b0=args.b0,
            stroke=args.stroke
        )

        # Run processing
        process_raw_single_subject(
            subject_dir,
            args.output_dir,
            config_from_yaml,
            logger,
            overrides=overrides
        )

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Processing failed for {args.input_subdir.name}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for {args.input_subdir.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
