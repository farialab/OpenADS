"""Brain masking and skull stripping pipeline.

This pipeline orchestrates brain mask generation and skull stripping:
1. Generates brain mask using SynthStrip
2. Binarizes the mask
3. Applies mask to images (skull stripping)

All I/O is delegated to adapters, all processing to services.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import from architecture layers
from ads.adapters.nifti import NiftiLoader, NiftiSaver
from ads.services.preprocessing import BrainMaskService

# Module-level logger
logger = logging.getLogger(__name__)


def generate_brain_mask(
    dwi_path: Path,
    output_dir: Path,
    subject_id: Optional[str] = None,
    *,
    use_gpu: bool = True,
    no_csf: bool = False,
    model_path: Optional[Path] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Generate SynthStrip brain mask for a subject.

    This function orchestrates:
    1. Loading input image
    2. Generating raw mask via SynthStrip
    3. Binarizing the mask
    4. Saving outputs

    Args:
        dwi_path: Path to input DWI/PWI image
        output_dir: Directory to save mask files
        subject_id: Subject identifier
        use_gpu: Whether to use GPU for SynthStrip
        no_csf: Whether to exclude CSF from brain boundary
        model_path: Optional path to custom SynthStrip model
        log: Optional logger instance

    Returns:
        Dictionary with keys 'raw' and 'binary' pointing to saved mask paths
    """
    log = log or logger
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse subject ID if not provided
    if subject_id is None:
        subject_id = dwi_path.name.split("_")[0]

    # Define output paths
    mask_raw = output_dir / f"{subject_id}_synthstrip_brain_mask_raw_surfa.nii.gz"
    mask_binary = output_dir / f"{subject_id}_DWIbrain-mask.nii.gz"

    # Check if masks already exist
    if mask_binary.exists() and mask_raw.exists():
        log.info("[%s] Using existing brain mask artifacts.", subject_id)
        return {"raw": mask_raw, "binary": mask_binary}

    log.info("[%s] Generating brain mask via SynthStrip...", subject_id)

    # Initialize services
    loader = NiftiLoader()
    saver = NiftiSaver()
    mask_service = BrainMaskService(
        use_gpu=use_gpu,
        no_csf=no_csf,
        model_path=model_path
    )

    try:
        # 1. Load input image
        input_img = loader.load(dwi_path)

        # 2. Generate raw mask
        raw_mask_img = mask_service.generate_mask(input_img, mask_raw)

        # 3. Binarize mask
        binary_mask_img = mask_service.binarize_mask(raw_mask_img)

        # 4. Save binary mask
        saver.save(binary_mask_img, mask_binary)
        mask_raw.unlink(missing_ok=True)

        log.info("[%s] Brain mask saved to %s", subject_id, mask_binary)
        return {"raw": mask_raw, "binary": mask_binary}

    except Exception as exc:
        log.error("[%s] Brain mask generation failed: %s", subject_id, exc)
        raise


def perform_skull_stripping(
    *,
    mask_path: Path,
    output_dir: Path,
    image_paths: List[Path],
    subject_id: Optional[str] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Apply brain mask to images (skull stripping).

    This function orchestrates:
    1. Loading binary mask
    2. Loading each input image
    3. Applying mask to each image
    4. Saving stripped outputs

    Args:
        mask_path: Path to binary brain mask
        output_dir: Directory to save skull-stripped images
        image_paths: List of image paths to strip
        subject_id: Subject identifier
        log: Optional logger instance

    Returns:
        Dictionary mapping image stems to output paths
    """
    log = log or logger
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading binary mask from: %s", mask_path)

    # Initialize services
    loader = NiftiLoader()
    saver = NiftiSaver()
    mask_service = BrainMaskService()

    # Load mask once
    mask_img = loader.load(mask_path)

    outputs: Dict[str, Path] = {}

    for src_path in image_paths:
        if not src_path.exists():
            log.warning("Input image path missing, skipping: %s", src_path)
            continue

        # Parse subject ID and output filename
        local_subject_id = subject_id or src_path.name.split("_")[0]
        name_parts = src_path.name.split(".nii", 1)
        base_name = name_parts[0]

        # Determine output filename
        if base_name.endswith('_brain'):
            output_filename = f"{base_name}.nii.gz"
        else:
            output_filename = f"{base_name}_brain.nii.gz"

        # Special case for stroke masks
        if 'stroke' in base_name.lower():
            output_filename = f"{local_subject_id}_stroke.nii.gz"

        out_path = output_dir / output_filename

        log.info("[%s] Skull stripping %s -> %s", local_subject_id, src_path.name, out_path.name)

        try:
            # 1. Load input image
            img = loader.load(src_path)

            # 2. Apply mask
            stripped_img = mask_service.apply_mask(img, mask_img, check_dimensions=True)

            if stripped_img is None:
                # Dimension mismatch - skip
                log.warning(
                    f"Skipping {src_path.name}: Shape mismatch with mask. "
                    "Likely different resolutions."
                )
                continue

            # 3. Save stripped image
            saver.save(stripped_img, out_path)
            outputs[src_path.stem] = out_path

        except Exception as e:
            log.error(f"Failed to strip {src_path.name}: {e}")

    return outputs


# --- Legacy Integration Wrapper ---

class MaskingPreprocessor:
    """Legacy wrapper for integration with existing pipelines.

    This class maintains the original interface while delegating to
    refactored functions.
    """

    def __init__(self, config: Dict, log: logging.Logger):
        """Initialize masking preprocessor.

        Args:
            config: Configuration dictionary
            log: Logger instance
        """
        self.config = config
        self.logger = log

    def run(self, subject):
        """Run masking and skull stripping for a subject.

        Args:
            subject: SubjectData object

        Returns:
            Updated SubjectData object
        """
        from ads.domain.subject_data import SubjectData

        preprocess_dir = subject.stage_dir("preprocess")

        # Generate mask
        mask_paths = generate_brain_mask(
            dwi_path=Path(subject.dwi_path),
            output_dir=preprocess_dir,
            subject_id=subject.subject_id,
            use_gpu=bool(self.config.get("brain_extraction", {}).get("use_gpu", True)),
            log=self.logger,
        )
        subject.mask_binary_path = mask_paths["binary"]
        subject.refresh_mask()

        # Collect images to strip
        images_to_strip = [
            Path(p) for p in [subject.dwi_path, subject.adc_path, subject.b0_path] if p
        ]

        # Perform skull stripping
        perform_skull_stripping(
            mask_path=mask_paths["binary"],
            output_dir=preprocess_dir,
            image_paths=images_to_strip,
            subject_id=subject.subject_id,
            log=self.logger,
        )

        # Update subject paths
        subject.dwi_brain_path = preprocess_dir / f"{subject.subject_id}_DWI_brain.nii.gz"
        subject.adc_brain_path = preprocess_dir / f"{subject.subject_id}_ADC_brain.nii.gz"
        subject.b0_brain_path = preprocess_dir / f"{subject.subject_id}_B0_brain.nii.gz"

        return subject


# --- Command-Line Interface ---

def main():
    """CLI entry point for masking and skull stripping tools."""
    parser = argparse.ArgumentParser(
        description="Brain masking and skull stripping tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Mask generation subcommand
    parser_mask = subparsers.add_parser(
        "mask",
        help="Generate a brain mask from a DWI image."
    )
    parser_mask.add_argument(
        "dwi_path",
        type=Path,
        help="Path to the input DWI NIfTI file."
    )
    parser_mask.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the generated mask files."
    )
    parser_mask.add_argument(
        "--subject_id",
        type=str,
        help="Optional subject ID."
    )
    parser_mask.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU."
    )

    # Skull stripping subcommand
    parser_ss = subparsers.add_parser(
        "skullstrip",
        help="Apply a mask to one or more images."
    )
    parser_ss.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="Path to the binary brain mask."
    )
    parser_ss.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save the skull-stripped images."
    )
    parser_ss.add_argument(
        "--images",
        type=Path,
        nargs='+',
        required=True,
        help="Images to be skull-stripped."
    )
    parser_ss.add_argument(
        "--subject_id",
        type=str,
        help="Optional subject ID."
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Execute command
    if args.command == "mask":
        generate_brain_mask(
            dwi_path=args.dwi_path,
            output_dir=args.output_dir,
            subject_id=args.subject_id,
            use_gpu=not args.cpu
        )
    elif args.command == "skullstrip":
        perform_skull_stripping(
            mask_path=args.mask,
            output_dir=args.output,
            image_paths=args.images,
            subject_id=args.subject_id
        )


if __name__ == "__main__":
    main()
