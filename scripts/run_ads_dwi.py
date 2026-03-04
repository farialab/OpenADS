#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADS DWI Pipeline Entry Point."""

from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# Project setup
project_root = Path(__file__).resolve().parents[1]
sys.path.extend([str(project_root), str(project_root / "src")])

import torch

from ads.core.config import load_config
from ads.pipelines.preprocessing_dwi_prepare import process_raw_single_subject
from ads.pipelines.preprocessing_brain_masking import generate_brain_mask, perform_skull_stripping
from ads.utils.output_cleanup import cleanup_subject_outputs

def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

def parse_stages(stages_str: str) -> set:
    """Parse comma-separated stage names into a set.
    
    Args:
        stages_str: Comma-separated stage names (e.g., "prepdata,inference,report")
        
    Returns:
        Set of stage names
    """
    return {s.strip() for s in stages_str.split(",") if s.strip()}


def _is_likely_dataset_root(path: Path) -> bool:
    """Heuristic: dataset root has subject subfolders but no image/json files directly."""
    if not path.is_dir():
        return False
    has_data_files = any(
        p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz") or p.name.endswith(".json"))
        for p in path.iterdir()
    )
    has_subject_dirs = any(p.is_dir() and p.name.startswith("sub-") for p in path.iterdir())
    return has_subject_dirs and not has_data_files

def main():
    stage_order = ["prepdata", "gen_mask", "skull_strip", "registration", "inference", "report"]
    parser = argparse.ArgumentParser(
        description="Run ADS DWI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available stages:
  prepdata       - Preprocess raw DWI data
  gen_mask       - Generate brain mask
  skull_strip    - Perform skull stripping
  registration   - Register to MNI space
  inference      - Run stroke segmentation model
  report         - Generate QFV metrics and reports

Examples:
  # Run all stages
  %(prog)s --subject-path /path/to/subject --all
  
  # Run specific stages
  %(prog)s --subject-path /path/to/subject --stages prepdata,inference,report
  
  # Use stages from config file
  %(prog)s --subject-path /path/to/subject
        """
    )
    parser.add_argument("--config", type=Path, default=project_root/"configs/dwi_pipeline.yaml",
                       help="Path to config YAML file (default: configs/dwi_pipeline.yaml)")
    parser.add_argument("--subject-path", type=Path, required=True,
                       help="Path to subject directory")
    parser.add_argument("--gpu", type=str, default="0",
                       help="GPU device ID (default: 0)")
    parser.add_argument("--all", action="store_true",
                       help="Run all pipeline stages (overrides config and --stages)")
    parser.add_argument("--stages", type=str,
                       help="Comma-separated list of stages to run (overrides config). "
                            "Example: prepdata,inference,report")
    parser.add_argument("--output-root", type=Path,
                       help="Override output root directory (overrides config paths.output_root)")
    
    args = parser.parse_args()
    _setup_logging()
    log = logging.getLogger("ADS.DWI")
    
    if args.gpu: 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Load Config (inherits defaults.yaml)
    cfg = load_config(args.config)
    if args.output_root:
        cfg.setdefault("paths", {})
        cfg["paths"]["output_root"] = str(args.output_root)
    cfg_stages = cfg.get("stages", {})
    
    subject_path = args.subject_path
    need_raw_input = bool(
        args.all
        or ("prepdata" in parse_stages(args.stages) if args.stages else cfg_stages.get("prepdata", False))
    )
    if need_raw_input and not subject_path.exists():
        log.error(f"Subject path does not exist (required for prepdata): {subject_path}")
        sys.exit(1)
    if subject_path.exists() and _is_likely_dataset_root(subject_path):
        log.error(
            "The provided --subject-path looks like a dataset root, not a subject folder: %s",
            subject_path,
        )
        log.error(
            "Please pass one subject directory, e.g. .../dwi/sub-XXXX, or use batch_run_dwi.sh for multiple subjects."
        )
        sys.exit(1)

    subject_id = subject_path.name
    out_root = Path(cfg["paths"]["output_root"])
    
    subj_out = out_root / subject_id
    dwi_root = subj_out / "DWI"
    pp_dir = dwi_root / "preprocess"
    reg_dir = dwi_root / "registration"

    dagmnet_path = Path(cfg["model"]["path"])
    tpl_dir = Path(cfg["paths"]["templates"]["root"])
    
    # Determine which stages to run (priority: --all > --stages > config)
    if args.all:
        # Run all stages
        stages = {k: True for k in stage_order}
        log.info("Running mode: ALL STAGES")
    elif args.stages:
        # Run only specified stages
        requested_stages = parse_stages(args.stages)
        
        # Validate stage names
        valid_stages = set(stage_order)
        invalid_stages = requested_stages - valid_stages
        if invalid_stages:
            log.error(f"Invalid stage names: {invalid_stages}")
            log.error(f"Valid stages are: {', '.join(sorted(valid_stages))}")
            sys.exit(1)
        
        # Set only requested stages to True
        stages = {k: (k in requested_stages) for k in stage_order}
        log.info(f"Running mode: SPECIFIC STAGES - {', '.join(sorted(requested_stages))}")
    else:
        # Use stages from config file
        stages = {k: bool(cfg_stages.get(k, False)) for k in stage_order}
        enabled_stages = [k for k, v in stages.items() if v]
        log.info(f"Running mode: CONFIG FILE - {', '.join(enabled_stages) if enabled_stages else 'No stages enabled'}")
    
    # Display stage status
    log.info("Stage configuration:")
    for stage_name in stage_order:
        enabled = bool(stages.get(stage_name, False))
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        log.info(f"  {stage_name:15s}: {status}")
    log.info("")

    try:
        # 1. Preprocess
        if stages.get("prepdata"):
            log.info("=" * 60)
            log.info("STAGE 1/7: Preprocessing raw data")
            log.info("=" * 60)
            process_raw_single_subject(subject_path, out_root, cfg, log)
            log.info("✓ Preprocessing completed\n")
            
        # 2. Mask
        if stages.get("gen_mask"):
            log.info("=" * 60)
            log.info("STAGE 2/7: Generating brain mask")
            log.info("=" * 60)
            generate_brain_mask(pp_dir / f"{subject_id}_DWI.nii.gz", pp_dir, subject_id, use_gpu=True)
            log.info("✓ Brain mask generation completed\n")
            
        # 3. Skull Strip
        if stages.get("skull_strip"):
            log.info("=" * 60)
            log.info("STAGE 3/7: Performing skull stripping")
            log.info("=" * 60)
            mask = pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz"
            imgs = [pp_dir / f"{subject_id}_{m}.nii.gz" for m in ["DWI", "ADC", "B0", "stroke"]]
            perform_skull_stripping(mask_path=mask, output_dir=pp_dir, image_paths=[i for i in imgs if i.exists()], subject_id=subject_id)
            log.info("✓ Skull stripping completed\n")

        # 4. Registration
        if stages.get("registration"):
            log.info("=" * 60)
            log.info("STAGE 4/7: Registration to MNI space")
            log.info("=" * 60)
            from ads.pipelines.registration_align import process_subject as run_registration
            from ads.domain.registration_data import TemplatePaths, RegistrationInputs
            from ads.domain.registration_spec import RegistrationSpec, AffineSpec, SyNSpec

            templates = TemplatePaths(
                dwi_mask_template=project_root / "assets/atlases/JHU_ICBM/JHU_MNI_SS_DWI_mask_to_MNI.nii",
                adc_template=project_root / "assets/atlases/JHU_ICBM/JHU_MNI_SS_ADC_ss_to_MNI.nii",
            )
            inputs = RegistrationInputs(
                dwi_brain=pp_dir / f"{subject_id}_DWI_brain.nii.gz",
                adc_brain=pp_dir / f"{subject_id}_ADC_brain.nii.gz",
                mask=pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz",
                b0_brain=(pp_dir / f"{subject_id}_B0_brain.nii.gz") if (pp_dir / f"{subject_id}_B0_brain.nii.gz").exists() else None,
                stroke=(pp_dir / f"{subject_id}_stroke.nii.gz") if (pp_dir / f"{subject_id}_stroke.nii.gz").exists() else None,
            )
            spec = RegistrationSpec(
                affine=AffineSpec(verbose=True),
                syn=SyNSpec(verbose=True),
                write_manifest=True,
            )
            outputs = run_registration(
                subject_dir=dwi_root,
                subject_id=subject_id,
                templates=templates,
                inputs=inputs,
                spec=spec,
                logger=log,
            )
            log.info(f"✓ Registration completed: {outputs.manifest_yaml.name}")

        # 5. Segment
        if stages.get("inference"):
            log.info("=" * 60)
            log.info("STAGE 5/6: Running stroke segmentation (includes postprocessing)")
            log.info("=" * 60)

            from ads.pipelines.segmentation_dwi_dagmnet import process_subject
            from ads.domain.model_config import ModelConfig
            from ads.domain.segmentation_spec import SegmentationSpec
            from ads.domain.segmentation_data import SegmentationInputs

            # Create model configuration
            model_config = ModelConfig(
                name="DAGMNet_DWI",
                weights_path=dagmnet_path,
                n_channels=3,  # or 2, depending on your model
                modality='dwi',
                architecture='dagmnet',
                template_dir=tpl_dir,
            )

            # Create segmentation spec
            spec = SegmentationSpec()
            spec.inference.device = "cuda" if torch.cuda.is_available() else "cpu"
            spec.inference.n_channel = 3  # Match model_config.n_channels
            spec.compute_metrics = True  # Enable metrics if ground truth available
            spec.save_normalized_inputs = False  # Normalized DWI/ADC are produced in registration stage

            # Check for optional ground truth files
            stroke_mni_path = reg_dir / f"{subject_id}_stroke_space-MNI152_aff.nii.gz"
            stroke_native_path = pp_dir / f"{subject_id}_stroke.nii.gz"
            adc_affsyn_path = reg_dir / f"{subject_id}_ADC_space-MNI152_affsyn.nii.gz"
            syn_affine_path = reg_dir / f"{subject_id}_syn_space-MNI1522MNI152.mat"
            syn_warp_path = reg_dir / f"{subject_id}_warp_space-MNI1522MNI152.nii.gz"

            # Build inputs with optional ground truth
            inputs_dict = {
                "dwi_mni": reg_dir / f"{subject_id}_DWI_space-MNI152_aff.nii.gz",
                "adc_mni": reg_dir / f"{subject_id}_ADC_space-MNI152_aff.nii.gz",
                "mask_mni": reg_dir / f"{subject_id}_DWIbrain-mask_space-MNI152_aff.nii.gz",
                "dwi_native": pp_dir / f"{subject_id}_DWI_brain.nii.gz",
                "mask_native": pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz",
                "fwd_affine": reg_dir / f"{subject_id}_aff_space-individual2MNI152.mat",
            }

            # Add optional ground truth if exists
            if stroke_mni_path.exists():
                inputs_dict["stroke_mni"] = stroke_mni_path
                log.info(f"✓ Found ground truth (MNI): {stroke_mni_path.name}")
            else:
                log.info("⚠ No ground truth found in MNI space - metrics will be null")

            if stroke_native_path.exists():
                inputs_dict["stroke_native"] = stroke_native_path
                log.info(f"✓ Found ground truth (native): {stroke_native_path.name}")
            else:
                log.info("⚠ No ground truth found in native space - metrics will be null")

            # Add optional SyN registration files if they exist
            if adc_affsyn_path.exists():
                inputs_dict["adc_affsyn"] = adc_affsyn_path
            if syn_affine_path.exists():
                inputs_dict["syn_affine"] = syn_affine_path
            if syn_warp_path.exists():
                inputs_dict["syn_warp"] = syn_warp_path

            inputs = SegmentationInputs(**inputs_dict)

            try:
                outputs = process_subject(
                    subject_dir=dwi_root,
                    subject_id=subject_id,
                    inputs=inputs,
                    model_config=model_config,
                    spec=spec,
                    logger=log,
                )
                log.info(f"✓ Segmentation outputs:")
                log.info(f"  - MNI prediction: {outputs.pred_mni}")
                log.info(f"  - Native prediction: {outputs.pred_native}")
                log.info(f"  - Metrics JSON: {outputs.metrics_json}")
                if outputs.pred_mni_affsyn:
                    log.info(f"  - MNI affsyn: {outputs.pred_mni_affsyn}")
                log.info("✓ Segmentation completed\n")
            except Exception as e:
                log.error(f"✗ Segmentation failed: {e}")
                raise

        # 6. QFV & Report
        if stages.get("report"):
            log.info("=" * 60)
            log.info("STAGE 6/6: Generating reports and visualizations")
            log.info("=" * 60)
            from ads.pipelines.reporting_qfv_calculation import process_qfv_single
            from ads.pipelines.reporting_report_generation import ReportPipeline
            
            rep_dir = dwi_root / "reporting"
            rep_dir.mkdir(parents=True, exist_ok=True)
            
            #tpl_dir = cfg["paths"]["templates"]["root"]
            
            _ = process_qfv_single(tpl_dir, str(dwi_root), str(rep_dir), subject_id=subject_id)
            ReportPipeline(dwi_root, atlas_dir=tpl_dir, aa_models_dir=cfg["paths"]["models"]["aa_models"], subject_id=subject_id).run(hydro_flag=False)
            
            (rep_dir / f"{subject_id}_volume_brain_regions.txt").unlink(missing_ok=True)

            # Generate visualizations
            log.info("Generating visualizations...")
            from ads.reporting.visualization.dwi_visualization import create_visualization_compare, create_visualization_compare_orig
            try:
                create_visualization_compare(dwi_root, subject_id)
                create_visualization_compare_orig(dwi_root, subject_id)
                log.info("✓ Visualizations created")
            except Exception as e:
                log.warning(f"Visualization generation failed: {e}")
            
            log.info("✓ Report generation completed\n")

        log.info("=" * 60)
        log.info("✅ DWI Pipeline Finished Successfully")
        log.info("=" * 60)

        cleanup_cfg = cfg.get("cleanup", {})
        if cleanup_cfg.get("enabled", True):
            keep_manifest = Path(cleanup_cfg.get("keep_manifest", project_root / "configs/keep_files.yaml"))
            cleanup_subject_outputs(
                output_root=out_root,
                subject_id=subject_id,
                modality="DWI",
                keep_manifest=keep_manifest,
                logger=log,
            )

    except Exception as e:
        log.error("=" * 60)
        log.error(f"❌ Pipeline Failed: {e}")
        log.error("=" * 60)
        log.error("", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
