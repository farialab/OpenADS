#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ADS PWI Pipeline Entry Point."""

from __future__ import annotations
import argparse
import logging
import os
import sys
import json
from pathlib import Path
import nibabel as nib
import ants
from typing import Optional
import numpy as np

# Project setup
project_root = Path(__file__).resolve().parents[1]
sys.path.extend([str(project_root), str(project_root / "src")])

from ads.core.config import load_config
from ads.pipelines.preprocessing_pwi_prepare import process_raw_single_subject
from ads.pipelines.preprocessing_brain_masking import generate_brain_mask, perform_skull_stripping
from ads.services.segmentation.metrics import MetricsService
from ads.services.segmentation.postprocessing import PostProcessingService
from ads.services.segmentation.pwi_hp_mni2orig import restore_hp_mni2orig
from ads.services.segmentation.postprocess_hp import postprocess_hp_mask
from ads.domain.segmentation_spec import PostProcessingSpec
from ads.utils.output_cleanup import cleanup_subject_outputs

def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def _find_first_existing(candidates: list[Path]) -> Optional[Path]:
    """Return the first existing path from candidates."""
    for p in candidates:
        if p.exists():
            return p
    return None


def _subject_id_variants(subject_id: str) -> list[str]:
    """Return common subject-id variants used in file naming."""
    variants: list[str] = []
    sid = subject_id.strip()
    sid_clean = sid[4:] if sid.startswith("sub-") else sid
    for cand in [sid, sid_clean, f"sub-{sid_clean}"]:
        if cand and cand not in variants:
            variants.append(cand)
    return variants


def _find_preprocess_modality(pp_dir: Path, subject_id: str, modality: str) -> Optional[Path]:
    """Find modality file in preprocess with flexible subject-id variants and extensions."""
    candidates: list[Path] = []
    for sid in _subject_id_variants(subject_id):
        candidates.extend(
            [
                pp_dir / f"{sid}_{modality}.nii.gz",
                pp_dir / f"{sid}_{modality}.nii",
            ]
        )
    found = _find_first_existing(candidates)
    if found is not None:
        return found
    glob_hits = sorted(pp_dir.glob(f"*_{modality}.nii*"))
    return glob_hits[0] if glob_hits else None


def _resolve_affsyn_transform_inputs(subject_root: Path, subject_id: str) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Resolve affsyn transform inputs with fallback:
    1) PWI/registration
    2) DWI/registration
    """
    pwi_reg = subject_root / "PWI" / "registration"
    dwi_reg = subject_root / "DWI" / "registration"
    reg_dirs = [pwi_reg, dwi_reg]

    adc_affsyn = _find_first_existing([d / f"{subject_id}_ADC_space-MNI152_affsyn.nii.gz" for d in reg_dirs])
    syn_affine = _find_first_existing([d / f"{subject_id}_syn_space-MNI1522MNI152.mat" for d in reg_dirs])
    syn_warp = _find_first_existing([d / f"{subject_id}_warp_space-MNI1522MNI152.nii.gz" for d in reg_dirs])
    return adc_affsyn, syn_affine, syn_warp


def _save_hp_affsyn(
    pred_aff_path: Path,
    pred_affsyn_path: Path,
    adc_affsyn_path: Path,
    syn_affine_path: Path,
    syn_warp_path: Optional[Path] = None,
) -> None:
    """Transform HP mask from MNI affine space to MNI affsyn space and save it."""
    fixed_affsyn = ants.image_read(str(adc_affsyn_path))
    moving_pred_aff = ants.image_read(str(pred_aff_path))

    transformlist = []
    if syn_warp_path is not None and syn_warp_path.exists():
        transformlist.append(str(syn_warp_path))
    transformlist.append(str(syn_affine_path))

    pred_affsyn = ants.apply_transforms(
        fixed=fixed_affsyn,
        moving=moving_pred_aff,
        transformlist=transformlist,
        interpolator="nearestNeighbor"
    )
    ants.reorient_image2(pred_affsyn, orientation="RAS").image_write(str(pred_affsyn_path))


def _save_pwi_metrics_json(
    metrics_path: Path,
    pred_mni_path: Path,
    pred_mni_postproc_path: Optional[Path],
    pred_native_path: Optional[Path],
    hp_mni_gt_path: Optional[Path],
    hp_native_gt_path: Optional[Path],
) -> None:
    """Save PWI segmentation metrics JSON similar to DWI pipeline behavior."""
    def _null_metrics(pred_volume=None):
        return {
            "dice": None,
            "precision": None,
            "sensitivity": None,
            "sdr": None,
            "pred_volume": pred_volume,
            "true_volume": None,
        }

    pred_mni = np.squeeze(nib.as_closest_canonical(nib.load(str(pred_mni_path))).get_fdata())
    pred_mni_bin = (pred_mni > 0.5).astype(np.float32)
    metrics = {
        "mni": _null_metrics(float(pred_mni_bin.sum())),
        "orig": _null_metrics(None),
        "mni_postproc": _null_metrics(None),
        "orig_postproc": _null_metrics(None),
    }

    postproc_service = PostProcessingService(PostProcessingSpec())
    pred_mni_post_bin = None
    if pred_mni_postproc_path is not None and pred_mni_postproc_path.exists():
        pred_mni_post = np.squeeze(nib.as_closest_canonical(nib.load(str(pred_mni_postproc_path))).get_fdata())
        pred_mni_post_bin = (pred_mni_post > 0.5).astype(np.float32)
    else:
        pred_mni_post_bin = postproc_service.process(pred_mni_bin).astype(np.float32)
    metrics["mni_postproc"] = _null_metrics(float(pred_mni_post_bin.sum()))

    pred_native_bin = None
    pred_native_post_bin = None
    if pred_native_path is not None and pred_native_path.exists():
        pred_native = np.squeeze(nib.as_closest_canonical(nib.load(str(pred_native_path))).get_fdata())
        pred_native_bin = (pred_native > 0.5).astype(np.float32)
        metrics["orig"] = _null_metrics(float(pred_native_bin.sum()))
        pred_native_post_bin = postproc_service.process(pred_native_bin).astype(np.float32)
        metrics["orig_postproc"] = _null_metrics(float(pred_native_post_bin.sum()))

    if hp_mni_gt_path is not None and hp_mni_gt_path.exists():
        gt_mni = np.squeeze(nib.as_closest_canonical(nib.load(str(hp_mni_gt_path))).get_fdata())
        metrics["mni"] = MetricsService.compute(
            prediction=pred_mni_bin,
            ground_truth=gt_mni,
            threshold=0.5,
        ).to_dict()
        metrics["mni_postproc"] = MetricsService.compute(
            prediction=pred_mni_post_bin,
            ground_truth=gt_mni,
            threshold=0.5,
        ).to_dict()

    if pred_native_bin is not None and hp_native_gt_path is not None and hp_native_gt_path.exists():
        gt_native = np.squeeze(nib.as_closest_canonical(nib.load(str(hp_native_gt_path))).get_fdata())
        metrics["orig"] = MetricsService.compute(
            prediction=pred_native_bin,
            ground_truth=gt_native,
            threshold=0.5,
        ).to_dict()
        metrics["orig_postproc"] = MetricsService.compute(
            prediction=pred_native_post_bin,
            ground_truth=gt_native,
            threshold=0.5,
        ).to_dict()

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _ensure_ttp_dimension_match(pp_dir: Path, subject_id: str, log: logging.Logger) -> None:
    """Ensure TTP spatial grid matches PWI mask grid (resample if needed)."""
    ttp_path = pp_dir / f"{subject_id}_TTP.nii.gz"
    pwi_mask_path = _find_first_existing(
        [
            pp_dir / f"{subject_id}_PWIbrain-mask.nii.gz",
            pp_dir / f"{subject_id}_PWI_ss.nii.gz",
            pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz",
        ]
    )
    if not ttp_path.exists():
        raise FileNotFoundError(f"Missing TTP output: {ttp_path}")
    if pwi_mask_path is None:
        raise FileNotFoundError(f"Missing target mask for TTP dimension check in {pp_dir}")

    ttp = ants.image_read(str(ttp_path))
    target = ants.image_read(str(pwi_mask_path))

    same_shape = tuple(ttp.shape) == tuple(target.shape)
    same_spacing = np.allclose(np.asarray(ttp.spacing), np.asarray(target.spacing), atol=1e-6)
    same_origin = np.allclose(np.asarray(ttp.origin), np.asarray(target.origin), atol=1e-5)
    same_direction = np.allclose(np.asarray(ttp.direction), np.asarray(target.direction), atol=1e-5)

    if same_shape and same_spacing and same_origin and same_direction:
        return

    log.warning(
        "TTP grid mismatch detected. Resampling TTP to target mask grid: ttp=%s target=%s",
        ttp_path.name,
        pwi_mask_path.name,
    )
    ttp_fixed = ants.resample_image_to_target(ttp, target, interp_type=0)
    ants.image_write(ttp_fixed, str(ttp_path))
    log.info("Saved resampled TTP: %s", ttp_path)


def _sanitize_nifti_scalar_intent(path: Path, log: logging.Logger) -> None:
    """Force a NIfTI to scalar intent so ANTs does not treat it as multi-component."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file to sanitize: {path}")

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    if data.ndim > 3:
        squeezed = np.squeeze(data)
        if squeezed.ndim != 3:
            raise ValueError(f"Expected scalar 3D image after squeeze, got {squeezed.shape} for {path.name}")
        data = squeezed

    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_intent("none", (), name="")
    out = nib.Nifti1Image(data.astype(np.float32), img.affine, hdr)
    out.update_header()
    out.header.set_intent("none", (), name="")
    nib.save(out, str(path))
    log.info("Sanitized scalar NIfTI intent: %s", path.name)


def _load_json_optional(pp_dir: Path, subject_id: str) -> dict:
    json_candidates = [pp_dir / f"{sid}_PWI.json" for sid in _subject_id_variants(subject_id)] + [pp_dir / "PWI.json"]
    p = _find_first_existing(json_candidates)
    if p is None:
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_ttp_fallback(pp_dir: Path, subject_id: str, log: logging.Logger) -> Path:
    """Generate fallback TTP map when legacy TTP generation fails."""
    stc_path = pp_dir / f"{subject_id}_PWI_STC.nii.gz"
    pwi_path = pp_dir / f"{subject_id}_PWI.nii.gz"
    src_path = stc_path if stc_path.exists() else pwi_path
    if not src_path.exists():
        raise FileNotFoundError(f"Cannot build fallback TTP. Missing: {stc_path} and {pwi_path}")

    src_img = nib.as_closest_canonical(nib.load(str(src_path)))
    src = np.asarray(src_img.get_fdata(), dtype=np.float32)
    src = np.squeeze(src)
    if src.ndim != 4:
        raise ValueError(f"Fallback TTP requires 4D PWI/STC after squeeze; got shape {src.shape} from {src_path.name}")

    t = src.shape[-1]
    if t < 2:
        raise ValueError(f"Fallback TTP requires at least 2 timepoints; got {t}")

    pwi_json = _load_json_optional(pp_dir, subject_id)
    tr = float(pwi_json.get("RepetitionTime", 1.7) or 1.7)

    mask_path = _find_first_existing(
        [
            pp_dir / f"{subject_id}_PWIbrain-mask.nii.gz",
            pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz",
        ]
    )
    if mask_path is not None:
        mask = np.asarray(nib.as_closest_canonical(nib.load(str(mask_path))).get_fdata()) > 0.5
        mask = np.squeeze(mask)
    else:
        mean_vol = src.mean(axis=-1)
        mask = mean_vol > np.percentile(mean_vol, 30)

    ttp_idx = np.argmin(src, axis=-1).astype(np.float32)
    ttp_sec = ttp_idx * tr
    ttp_sec[~mask] = 0.0

    ttp_path = pp_dir / f"{subject_id}_TTP.nii.gz"
    ttp_img = nib.Nifti1Image(ttp_sec.astype(np.float32), src_img.affine)
    ttp_img.set_data_dtype(np.float32)
    ttp_img.header.set_intent("none", (), name="")
    nib.save(ttp_img, str(ttp_path))
    log.warning("Generated fallback TTP: %s (TR=%.4f, shape=%s)", ttp_path.name, tr, tuple(ttp_sec.shape))
    return ttp_path


def _run_gen_ttp_with_json_fallback(
    subject_id: str,
    pwi_root: Path,
    pp_dir: Path,
    cfg: dict,
    log: logging.Logger,
) -> Path:
    """
    Run Stage-4 TTP generation:
    - If PWI.json exists: keep legacy path.
    - If PWI.json is missing: disable motion correction and use fallback TTP repair steps.
    """
    from ads.pipelines.preprocessing_pwi_genttp import process_subject_sequential, PWIPreprocessingConfig

    pwi_cfg = PWIPreprocessingConfig()
    pwi_pre_cfg = cfg.get("pwi_preprocessing", {})
    if "motion_correction" in pwi_pre_cfg:
        pwi_cfg.motion_correction.update(pwi_pre_cfg["motion_correction"])

    json_candidates = [pp_dir / f"{sid}_PWI.json" for sid in _subject_id_variants(subject_id)] + [pp_dir / "PWI.json"]
    has_pwi_json = any(p.exists() for p in json_candidates)

    ttp_path = pp_dir / f"{subject_id}_TTP.nii.gz"
    if has_pwi_json:
        ret = process_subject_sequential(subject_id, str(pwi_root), ['PWI'], pwi_cfg, output_dir=pp_dir)
        if (isinstance(ret, str) and ret.startswith("Error:")) or (not ttp_path.exists()):
            raise RuntimeError(f"TTP generation failed for {subject_id}")
        return ttp_path

    pwi_cfg.motion_correction["motion_correction"] = False
    pwi_cfg.motion_correction["save_corrected_PWI"] = False
    log.warning("No PWI.json in preprocess; disabling motion correction for %s.", subject_id)
    ret = process_subject_sequential(subject_id, str(pwi_root), ['PWI'], pwi_cfg, output_dir=pp_dir)
    if isinstance(ret, str) and ret.startswith("Error:"):
        log.warning("Legacy TTP generator returned error for %s, switching to fallback TTP builder.", subject_id)
    if not ttp_path.exists():
        _ = _generate_ttp_fallback(pp_dir, subject_id, log)
    _ensure_ttp_dimension_match(pp_dir, subject_id, log)
    _sanitize_nifti_scalar_intent(ttp_path, log)
    if not ttp_path.exists():
        raise RuntimeError(f"TTP generation failed for {subject_id}")
    return ttp_path


def parse_stages(stages_str: str) -> set:
    """Parse comma-separated stage names into a set.
    
    Args:
        stages_str: Comma-separated stage names (e.g., "prepdata,inference")
        
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

def run_pwi_single(cfg: dict, subject_path: Path) -> None:
    log = logging.getLogger("ADS.PWI")
    subject_name = subject_path.name
    subject_id = subject_name if subject_name.startswith("sub-") else f"sub-{subject_name}"
    
    # Expand output root
    out_root = Path(cfg["paths"]["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    
    subj_out = out_root / subject_id
    pwi_root = subj_out / "PWI"
    pp_dir = pwi_root / "preprocess"
    reg_dir = pwi_root / "registration"
    
    stages = cfg.get("stages", {})

    try:
        # 1. Prep Data
        if stages.get("prepdata"):
            log.info("=" * 60)
            log.info("STAGE 1/8: Preprocessing raw data")
            log.info("=" * 60)
            process_raw_single_subject(subject_path, out_root, cfg, log)
            log.info("✓ Preprocessing completed\n")

        # 2. Masking
        if stages.get("gen_mask"):
            log.info("=" * 60)
            log.info("STAGE 2/8: Generating brain mask")
            log.info("=" * 60)
            dwi_path = _find_preprocess_modality(pp_dir, subject_id, "DWI")
            adc_path = _find_preprocess_modality(pp_dir, subject_id, "ADC")
            
            if dwi_path is not None:
                generate_brain_mask(dwi_path, pp_dir, subject_id, use_gpu=True)
            elif adc_path is not None:
                generate_brain_mask(adc_path, pp_dir, subject_id, use_gpu=True)
            else:
                raise FileNotFoundError(
                    f"Missing both DWI and ADC for brain-mask generation: "
                    f"{subject_id}_DWI.nii.gz/.nii, {subject_id}_ADC.nii.gz/.nii"
                )
            log.info("✓ Brain mask generation completed\n")

        # 3. Skull Strip
        if stages.get("skull_strip"):
            log.info("=" * 60)
            log.info("STAGE 3/8: Performing skull stripping")
            log.info("=" * 60)
            mask = pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz"
            if not mask.exists(): 
                mask = pp_dir / f"{subject_id}_ADCbrain-mask.nii.gz"
                
            if mask.exists():
                imgs = [pp_dir / f"{subject_id}_{m}.nii.gz" for m in ["DWI", "ADC", "stroke"]]
                perform_skull_stripping(
                    mask_path=mask, 
                    output_dir=pp_dir, 
                    image_paths=[i for i in imgs if i.exists()], 
                    subject_id=subject_id
                )
            log.info("✓ Skull stripping completed\n")

        # 4. Generate TTP
        if stages.get("gen_ttp"):
            log.info("=" * 60)
            log.info("STAGE 4/8: Generating TTP maps")
            log.info("=" * 60)
            ttp_path = _run_gen_ttp_with_json_fallback(subject_id, pwi_root, pp_dir, cfg, log)
            log.info("✓ TTP generation completed: %s\n", ttp_path.name)

        # 5. Registration (DWI transforms will be reused by Stage 6 automatically)
        if stages.get("registration"):
            log.info("=" * 60)
            log.info("STAGE 5/8: Registration to MNI space")
            log.info("=" * 60)

            dwi_reg_dir = subj_out / "DWI" / "registration"
            always_reg = cfg.get("registration", {}).get("always_register", False)

            # Check if DWI registration exists
            if dwi_reg_dir.exists() and any(dwi_reg_dir.iterdir()) and not always_reg:
                log.info(f"✓ Found DWI registration: {dwi_reg_dir}")
                log.info(f"  Stage 6 will reuse transforms directly from DWI/registration/")
                log.info(f"  → Zero storage overhead, no copying needed")

            else:
                # Run new DWI→MNI registration
                if always_reg:
                    log.info("Running new registration (always_register=True)")
                else:
                    log.info("DWI registration not found, running new registration")

                reg_dir.mkdir(parents=True, exist_ok=True)

                from ads.pipelines.registration_align import process_subject as run_registration
                from ads.domain.registration_data import TemplatePaths, RegistrationInputs
                from ads.domain.registration_spec import RegistrationSpec, AffineSpec, SyNSpec

                dwi_brain = pp_dir / f"{subject_id}_DWI_brain.nii.gz"
                adc_brain = pp_dir / f"{subject_id}_ADC_brain.nii.gz"
                reg_mask = _find_first_existing(
                    [
                        pp_dir / f"{subject_id}_DWIbrain-mask.nii.gz",
                        pp_dir / f"{subject_id}_ADCbrain-mask.nii.gz",
                    ]
                )

                if not adc_brain.exists():
                    raise FileNotFoundError(f"Missing ADC brain image: {adc_brain}")
                if reg_mask is None:
                    raise FileNotFoundError(f"Missing brain mask in {pp_dir}")

                # DWI optional fallback: generate pseudoDWI from ADC brain if DWI brain is absent.
                has_true_dwi = dwi_brain.exists()
                dwi_input = dwi_brain if has_true_dwi else adc_brain
                if not has_true_dwi:
                    from ads.services.preprocessing.pseudo_dwi import PseudoDWIGenerator

                    pseudo_dwi_path = pp_dir / f"{subject_id}_pseudoDWI.nii.gz"
                    gen = PseudoDWIGenerator(
                        b_value=float(cfg.get("data_raw", {}).get("bvalue", 1000.0)),
                        adc_scale="auto",
                        brain_threshold=1e-6,
                    )
                    if not pseudo_dwi_path.exists():
                        _, scale = gen.generate_from_files(
                            adc_path=adc_brain,
                            output_path=pseudo_dwi_path,
                            mask_path=reg_mask,
                        )
                        log.warning(
                            "DWI brain missing; generated pseudoDWI from ADC brain (scale=%s): %s",
                            scale,
                            pseudo_dwi_path.name,
                        )
                    else:
                        log.warning(
                            "DWI brain missing; using pseudoDWI: %s",
                            pseudo_dwi_path.name,
                        )
                    dwi_input = pseudo_dwi_path

                templates = TemplatePaths(
                    dwi_mask_template=project_root / "assets/atlases/JHU_ICBM/JHU_MNI_SS_DWI_mask_to_MNI.nii",
                    adc_template=project_root / "assets/atlases/JHU_ICBM/JHU_MNI_SS_ADC_ss_to_MNI.nii",
                )
                inputs = RegistrationInputs(
                    dwi_brain=dwi_input,
                    adc_brain=adc_brain,
                    mask=reg_mask,
                    b0_brain=(pp_dir / f"{subject_id}_B0_brain.nii.gz") if (pp_dir / f"{subject_id}_B0_brain.nii.gz").exists() else None,
                    stroke=(pp_dir / f"{subject_id}_stroke.nii.gz") if (pp_dir / f"{subject_id}_stroke.nii.gz").exists() else None,
                )
                spec = RegistrationSpec(
                    affine=AffineSpec(
                        verbose=True,
                        type_of_transform="Affine",
                        reg_iterations=[100, 50, 25],
                        shrink_factors=[8, 4, 2],
                        smoothing_sigmas=[3.0, 2.0, 1.0],
                    ),
                    syn=SyNSpec(
                        verbose=True,
                        type_of_transform="SyN",
                        reg_iterations=[70, 50, 20],
                        shrink_factors=[8, 4, 2],
                        smoothing_sigmas=[3.0, 2.0, 1.0],
                    ),
                    write_manifest=False,
                )
                outputs = run_registration(
                    subject_dir=pwi_root,
                    subject_id=subject_id,
                    templates=templates,
                    inputs=inputs,
                    spec=spec,
                    logger=log,
                )
                log.info(f"✓ Registration completed: {outputs.manifest_yaml.name}")
                #if not has_true_dwi:
                #    (reg_dir / f"{subject_id}_DWI_space-MNI152_aff.nii.gz").unlink(missing_ok=True)
                #    (reg_dir / f"{subject_id}_DWI_space-MNI152_aff_desc-norm.nii.gz").unlink(missing_ok=True)
                #    (reg_dir / f"{subject_id}_DWI_space-MNI152_affsyn.nii.gz").unlink(missing_ok=True)
                #    log.info("Removed surrogate DWI registration outputs for %s", subject_id)


        # 6. TTP-ADC Coregistration
        if stages.get("ttpadc_coreg"):
            log.info("=" * 60)
            log.info("STAGE 6/8: TTP-ADC coregistration")
            log.info("=" * 60)

            # ============================================================
            # VALIDATION: Check required transforms exist
            # ============================================================
            always_reg = cfg.get("registration", {}).get("always_register", False)

            if always_reg:
                # When always_register=True, transforms MUST exist in PWI/registration/
                log.info("Validating transforms in PWI/registration/ (always_register=True)...")

                required_files = [
                    reg_dir / f"{subject_id}_aff_space-individual2MNI152.mat",
                    reg_dir / f"{subject_id}_warp_space-MNI1522MNI152.nii.gz",
                    reg_dir / f"{subject_id}_syn_space-MNI1522MNI152.mat",
                ]

                missing_files = [f for f in required_files if not f.exists()]

                if missing_files:
                    log.error("=" * 60)
                    log.error("❌ VALIDATION FAILED: Required transforms not found")
                    log.error("=" * 60)
                    log.error(f"always_register=True, but transforms missing in: {reg_dir}")
                    log.error("")
                    log.error("Missing files:")
                    for f in missing_files:
                        log.error(f"  ✗ {f.name}")
                    log.error("")
                    log.error("Possible causes:")
                    log.error("  1. Stage 5 (registration) was not run")
                    log.error("  2. Stage 5 failed to complete")
                    log.error("  3. Registration output directory is incorrect")
                    log.error("")
                    log.error("Solutions:")
                    log.error("  1. Run Stage 5 first: --stages registration,ttpadc_coreg")
                    log.error("  2. Or set always_register=false in config to use DWI transforms")
                    log.error("=" * 60)
                    raise FileNotFoundError(
                        f"Required transform files missing in {reg_dir}. "
                        f"Run registration stage first or set always_register=false in config."
                    )

                log.info("✓ All required transforms found in PWI/registration/")
                log.info("")
            else:
                # When always_register=False, use fallback search (PWI → DWI)
                log.info("Transform search strategy: PWI/registration → DWI/registration (fallback)")
                log.info("")

            # ============================================================
            # Proceed with coregistration
            # ============================================================
            from ads.pipelines.registration_pwi_align import process_subject as run_pwi_registration
            from ads.domain.pwi_registration_data import PWIRegistrationInputs
            from ads.domain.pwi_registration_spec import PWIRegistrationSpec, PWICoregSpec
            from ads.adapters.pwi_registration_paths import PWIInputDiscovery

            discovered = PWIInputDiscovery.find_pwi_inputs(pp_dir, subject_id)
            if not discovered["ttp"] or not discovered["pwi_mask"] or not discovered["dwi_mask"] or not discovered["pwi"]:
                raise FileNotFoundError(
                    f"Missing required PWI coreg inputs for {subject_id}. "
                    f"Found keys: { {k: str(v) if v else None for k, v in discovered.items()} }"
                )

            inputs = PWIRegistrationInputs(
                ttp=discovered["ttp"],
                hp_manual=discovered["hp_manual"],
                pwi=discovered["pwi"],
                pwi_mask=discovered["pwi_mask"],
                dwi_mask=discovered["dwi_mask"],
            )
            spec = PWIRegistrationSpec(
                coreg=PWICoregSpec(type_of_transform="Affine", verbose=True),
                output_orientation="RAS",
                interpolator_images="linear",
                interpolator_masks="nearestNeighbor",
                skip_reregistration=not always_reg,
                force_recompute=always_reg,
                raw_ttp_masked=bool(cfg.get("registration", {}).get("raw_ttp_masked", True)),
            )
            templates = {
                "dwi_mask_template": project_root / "assets/atlases/JHU_ICBM/JHU_MNI_SS_DWI_mask_to_MNI.nii",
                "adc_template": Path(cfg["paths"]["templates"]["root"]) / "JHU_MNI_SS_ADC_ss_to_MNI.nii",
            }

            _ = run_pwi_registration(
                subject_dir=pwi_root,
                subject_id=subject_id,
                inputs=inputs,
                templates=templates,
                spec=spec,
                logger=log,
            )
            log.info("✓ TTP-ADC coregistration completed\n")

        # 7. Inference
        if stages.get("inference"):
            log.info("=" * 60)
            log.info("STAGE 7/8: Running stroke segmentation")
            log.info("=" * 60)
            from ads.utils.normalize import load_nifti_as_ras, get_dwi_normalized, zscore_within_mask, new_nifti_like
            
            dwi_mni = reg_dir / f"{subject_id}_DWI_space-MNI152_aff.nii.gz"
            adc_mni = reg_dir / f"{subject_id}_ADC_space-MNI152_aff.nii.gz"
            mask_mni = reg_dir / f"{subject_id}_DWIbrain-mask_space-MNI152_aff.nii.gz"
            
            dwi_norm_path = reg_dir / f"{subject_id}_DWI_space-MNI152_aff_desc-norm.nii.gz"
            adc_norm_path = reg_dir / f"{subject_id}_ADC_space-MNI152_aff_desc-norm.nii.gz"

            # Normalize DWI if not already done
            if dwi_mni.exists() and mask_mni.exists() and not dwi_norm_path.exists():
                log.info("Normalizing DWI...")
                dwi_img = load_nifti_as_ras(dwi_mni)
                mask_img = load_nifti_as_ras(mask_mni)
                dwi_norm = get_dwi_normalized(dwi_img.get_fdata(), mask_img.get_fdata())
                dwi_z = zscore_within_mask(dwi_norm, mask_img.get_fdata())
                nib.save(new_nifti_like(dwi_z, dwi_img), dwi_norm_path)
                
            # Normalize ADC if not already done
            if adc_mni.exists() and mask_mni.exists() and not adc_norm_path.exists():
                log.info("Normalizing ADC...")
                adc_img = load_nifti_as_ras(adc_mni)
                mask_img = load_nifti_as_ras(mask_mni)
                adc_norm = get_dwi_normalized(adc_img.get_fdata(), mask_img.get_fdata()) 
                adc_z = zscore_within_mask(adc_norm, mask_img.get_fdata())
                nib.save(new_nifti_like(adc_z, adc_img), adc_norm_path)

            # Prepare paths dictionary
            stroke_mni_path = reg_dir / f"{subject_id}_stroke_space-MNI152_aff.nii.gz"
            
            paths = {
                'dwi_mni': dwi_norm_path,
                'adc_mni': adc_norm_path,
                'ttp_mni': reg_dir / f"{subject_id}_TTP_space-MNI152_aff.nii.gz",
                'stroke_mni': stroke_mni_path if stroke_mni_path.exists() else None,
                'mask_mni': mask_mni
            }
            
            # # Determine model path based on the presence of stroke segmentation
            # use_stroke = stroke_mni_path.exists()
            # model_key = "path_ch4" if use_stroke else "path_ch3"
            # model_path = Path(cfg["model"].get(model_key, ""))
            
            # if not model_path.exists():
            #     raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # # Create output directory
            # out_seg = pwi_root / "segmentation"
            # out_seg.mkdir(parents=True, exist_ok=True)
            # out_path = out_seg / f"{subject_id}_{model_path.stem}_HP-mask.nii.gz"
            
            # # Run inference with auto-detection
            # log.info(f"Model: {model_path}")
            # log.info(f"Stroke file: {'Found' if paths['stroke_mni'] else 'Not found (will use 3-channel mode)'}")
            
            # PWIInferencePipeline(str(model_path), paths, str(out_path), use_stroke)
            # log.info("✓ Segmentation completed\n")

            from ads.models.dagmnet_pwi import DAGMNetPredictor
        
            checkpoint_path = project_root / "assets" / "models" / "dagmnet_nostroke.pth"
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"DAGMNet model file not found: {checkpoint_path}")
            
            device = "cuda:1"
            log.info(f"Using device: {device}")
            
            predictor = DAGMNetPredictor(
                checkpoint_path=str(checkpoint_path),
                n_channels=4,
                threshold=0.49,
                device=device,
                auto_generate_symclassic=True,
                allow_missing_channels=True,
            )
            
            # Create output directory
            seg_dir = pwi_root / "segmentation"
            print(pwi_root)
            print(f"Creating segmentation directory at: {seg_dir}")
            seg_dir.mkdir(parents=True, exist_ok=True)
            
            # ✅ Use registration directory for input files
            reg_dir = pwi_root / "registration"
            if not reg_dir.exists():
                log.warning(f"Registration directory not found: {reg_dir}")
                reg_dir = pwi_root  # Fallback to main directory
            
            log.info(f"Looking for input files in: {reg_dir}")
            
            # Run prediction
            output_path = seg_dir / f"{subject_id}_HP-mask_space-MNI152.nii.gz"
            postproc_output_path = seg_dir / f"{subject_id}_HP-mask_space-MNI152_desc-postproc.nii.gz"
            deprecated_aff_output = seg_dir / f"{subject_id}_HP-mask_space-MNI152_aff.nii.gz"
            pred_mask = predictor.predict_subject(
                subject_dir=pwi_root, 
                subject_id=subject_id,
                save_path=str(output_path)
            )
            
            if pred_mask is not None:
                log.info(f"✓ DAGMNet inference completed")
                log.info(f"✓ Saved to: {output_path}")
                try:
                    postprocess_hp_mask(
                        output_path,
                        threshold=0.5,
                        open_radius=1,
                        close_radius=1,
                        fill_holes=True,
                        min_vox=100,
                        topk=0,
                        brain_mask_path=reg_dir / f"{subject_id}_DWIbrain-mask_space-MNI152_aff.nii.gz",
                        output_path=postproc_output_path,
                    )
                    log.info(f"✓ Saved postprocessed HP mask: {postproc_output_path.name}")
                except Exception as e:
                    log.warning(f"Could not postprocess HP mask: {e}")
                if deprecated_aff_output.exists():
                    deprecated_aff_output.unlink()
                    log.info(f"✓ Removed deprecated duplicate output: {deprecated_aff_output.name}")

                # Reuse existing DWI SyN transforms to generate affsyn-space HP mask.
                hp_affsyn_path = seg_dir / f"{subject_id}_HP-mask_space-MNI152_affsyn.nii.gz"
                adc_affsyn_path, syn_affine_path, syn_warp_path = _resolve_affsyn_transform_inputs(subj_out, subject_id)
                if adc_affsyn_path and syn_affine_path:
                    _save_hp_affsyn(
                        pred_aff_path=output_path,
                        pred_affsyn_path=hp_affsyn_path,
                        adc_affsyn_path=adc_affsyn_path,
                        syn_affine_path=syn_affine_path,
                        syn_warp_path=syn_warp_path,
                    )
                    log.info(f"✓ Saved affsyn prediction: {hp_affsyn_path}")
                else:
                    log.warning(
                        "Could not save affsyn HP mask (missing ADC affsyn or syn affine transform)."
                    )

                # Save native-space HP prediction (MNI -> ADC native -> TTP native when available).
                native_saved: Optional[Path] = None
                try:
                    native_saved = restore_hp_mni2orig(subj_out, subject_id)
                    log.info(f"✓ Saved native prediction: {native_saved}")
                except Exception as e:
                    log.warning(f"Could not save native HP prediction: {e}")

                # Save metrics JSON in segmentation folder
                metrics_path = seg_dir / f"{subject_id}_metrics.json"
                hp_gt_mni = reg_dir / f"{subject_id}_HP_manual_space-MNI152_aff.nii.gz"
                hp_gt_native = pp_dir / f"{subject_id}_HP_manual.nii.gz"
                _save_pwi_metrics_json(
                    metrics_path=metrics_path,
                    pred_mni_path=output_path,
                    pred_mni_postproc_path=postproc_output_path if postproc_output_path.exists() else None,
                    pred_native_path=native_saved,
                    hp_mni_gt_path=hp_gt_mni if hp_gt_mni.exists() else None,
                    hp_native_gt_path=hp_gt_native if hp_gt_native.exists() else None,
                )
                log.info(f"✓ Saved metrics: {metrics_path}")
            else:
                log.warning("❌ DAGMNet inference failed")

            log.info("✓ Segmentation completed\n")

        # 8. Report
        if stages.get("report"):
            log.info("=" * 60)
            log.info("STAGE 8/8: Generating reports and visualizations")
            log.info("=" * 60)

            seg_dir = pwi_root / "segmentation"
            output_path = seg_dir / f"{subject_id}_HP-mask_space-MNI152.nii.gz"

            if not output_path.exists():
                log.warning(f"Segmentation output not found: {output_path}")
                log.warning("Run inference stage first: --stages inference,report")
            else:
                # Generate radiological report
                from ads.reporting.radiology.pwi_radiology_report import write_pwi_radiological_report
                dwi_report = (subj_out / "DWI" / "reporting" / f"{subject_id}_automatic_radiological_report.txt")
                # Supervisor requirement: DWI lesion volume for mismatch must use affine-space prediction.
                dwi_stroke_mask = subj_out / "DWI" / "segment" / f"{subject_id}_stroke-mask_space-MNI152.nii.gz"
                reg_dirs = [reg_dir, subj_out / "DWI" / "registration"]
                ttp_affsyn_gt = _find_first_existing(
                    [d / f"{subject_id}_TTP_space-MNI152_affsyn.nii.gz" for d in reg_dirs]
                )
                if ttp_affsyn_gt is None:
                    raise FileNotFoundError(
                        f"Missing affsyn TTP for reporting: {subject_id}_TTP_space-MNI152_affsyn.nii.gz "
                        f"(searched in {[str(d) for d in reg_dirs]})"
                    )
                # Supervisor requirement:
                # - report volume text must use prediction mask in affine space (not affsyn)
                # - other reporting inputs remain in affsyn space
                hp_pred_aff = seg_dir / f"{subject_id}_HP-mask_space-MNI152.nii.gz"
                report_source = hp_pred_aff if hp_pred_aff.exists() else None
                if report_source is None:
                    raise FileNotFoundError(
                        f"Missing affine prediction HP mask for PWI report volume: "
                        f"{subject_id}_HP-mask_space-MNI152.nii.gz in PWI/segmentation."
                    )
                pwi_report_path = write_pwi_radiological_report(
                    pwi_root=pwi_root,
                    subject_id=subject_id,
                    hp_mask_path=report_source,
                    dwi_report_path=dwi_report,
                    dwi_stroke_mask_path=dwi_stroke_mask if dwi_stroke_mask.exists() else None,
                )
                log.info(f"✓ PWI radiological report saved: {pwi_report_path}")

                # Generate HP atlas-overlap CSV outputs (HPload + HPQFV)
                from ads.pipelines.reporting_pwi_qfv_calculation import process_pwi_qfv_single
                hp_pred_affsyn = seg_dir / f"{subject_id}_HP-mask_space-MNI152_affsyn.nii.gz"
                hp_mask_for_qfv = hp_pred_affsyn if hp_pred_affsyn.exists() else None
                tpl_dir = Path(cfg["paths"]["templates"]["root"])

                try:
                    if hp_mask_for_qfv is None:
                        raise FileNotFoundError(
                            f"Missing affsyn prediction HP mask for QFV: "
                            f"{subject_id}_HP-mask_space-MNI152_affsyn.nii.gz in PWI/segmentation."
                        )
                    csv_outputs = process_pwi_qfv_single(
                        template_dir=tpl_dir,
                        pwi_dir=pwi_root,
                        output_dir=pwi_root / "reporting",
                        subject_id=subject_id,
                        hp_mask_path=hp_mask_for_qfv,
                    )
                    log.info(f"✓ PWI HP CSVs saved: {len(csv_outputs)} files")
                except Exception as e:
                    log.warning(f"PWI HP CSV generation failed: {e}")

                # Generate visualization
                log.info("Generating visualizations...")
                from ads.reporting.visualization.pwi_visualization import (
                    create_hp_visualization_compare,
                    create_hp_visualization_compare_orig,
                )
                try:
                    out_mni = create_hp_visualization_compare(out_root, subject_id)
                    out_orig = create_hp_visualization_compare_orig(out_root, subject_id)
                    log.info(f"✓ Visualization saved: {out_mni}")
                    log.info(f"✓ Visualization saved: {out_orig}")
                except Exception as e:
                    log.warning(f"Visualization generation failed: {e}")

                log.info("✓ Report generation completed\n")


        log.info("=" * 60)
        log.info("✅ PWI Pipeline Finished Successfully")
        log.info("=" * 60)

        cleanup_cfg = cfg.get("cleanup", {})
        if cleanup_cfg.get("enabled", True):
            keep_manifest = Path(cleanup_cfg.get("keep_manifest", project_root / "configs/keep_files.yaml"))
            cleanup_subject_outputs(
                output_root=out_root,
                subject_id=subject_id,
                modality="PWI",
                keep_manifest=keep_manifest,
                logger=log,
            )

    except Exception as e:
        log.error("=" * 60)
        log.error(f"❌ Pipeline Failed: {e}")
        log.error("=" * 60)
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Run ADS PWI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available stages:
  prepdata      - Preprocess raw PWI data
  gen_mask      - Generate brain mask
  skull_strip   - Perform skull stripping
  gen_ttp       - Generate TTP maps
  registration  - Register to MNI space
  ttpadc_coreg  - TTP-ADC coregistration
  inference     - Run stroke segmentation model
  report        - Generate reports and visualizations

Examples:
  # Run all stages
  %(prog)s --subject-path assets/examples/pwi/sub-12345 --all
  
  # Run specific stages
  %(prog)s --subject-path assets/examples/pwi/sub-12345 --stages prepdata,inference
  
  # Use stages from config file
  %(prog)s --subject-path assets/examples/pwi/sub-12345
        """
    )
    parser.add_argument("--config", type=Path, default=project_root/"configs/pwi_pipeline.yaml",
                       help="Path to config YAML file (default: configs/pwi_pipeline.yaml)")
    parser.add_argument("--subject-path", type=Path, required=True,
                       help="Path to subject directory")
    parser.add_argument("--gpu", type=str, default="0",
                       help="GPU device ID (default: 0)")
    parser.add_argument("--all", action="store_true",
                       help="Run all pipeline stages (overrides config and --stages)")
    parser.add_argument("--stages", type=str,
                       help="Comma-separated list of stages to run (overrides config). "
                            "Example: prepdata,inference")
    parser.add_argument("--output-root", type=Path,
                       help="Override output root directory (overrides config paths.output_root)")
    
    args = parser.parse_args()
    _setup_logging()
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.gpu.lower() != "cpu" else ""
    
    cfg = load_config(args.config)
    if args.output_root:
        cfg.setdefault("paths", {})
        cfg["paths"]["output_root"] = str(args.output_root)
    
    # Determine which stages to run (priority: --all > --stages > config)
    stages = cfg.get("stages", {})
    log = logging.getLogger("ADS.PWI")
    
    if args.all:
        # Run all stages
        stages = {k: True for k in stages}
        log.info("Running mode: ALL STAGES")
    elif args.stages:
        # Run only specified stages
        requested_stages = parse_stages(args.stages)
        
        # Validate stage names
        valid_stages = {"prepdata", "gen_mask", "skull_strip", "gen_ttp",
                       "registration", "ttpadc_coreg", "inference", "report"}
        invalid_stages = requested_stages - valid_stages
        if invalid_stages:
            log.error(f"Invalid stage names: {invalid_stages}")
            log.error(f"Valid stages are: {', '.join(sorted(valid_stages))}")
            sys.exit(1)
        
        # Set only requested stages to True
        stages = {k: (k in requested_stages) for k in stages}
        log.info(f"Running mode: SPECIFIC STAGES - {', '.join(sorted(requested_stages))}")
    else:
        # Use stages from config file
        enabled_stages = [k for k, v in stages.items() if v]
        log.info(f"Running mode: CONFIG FILE - {', '.join(enabled_stages) if enabled_stages else 'No stages enabled'}")
    
    # Update config with determined stages
    cfg["stages"] = stages
    
    # Display stage status
    log.info("Stage configuration:")
    for stage_name, enabled in stages.items():
        status = "✓ ENABLED" if enabled else "✗ DISABLED"
        log.info(f"  {stage_name:15s}: {status}")
    log.info("")
    
    subject_path = args.subject_path
    
    need_raw_input = bool(stages.get("prepdata", False))
    if need_raw_input and not subject_path.exists():
        log.error(f"Subject path does not exist (required for prepdata): {subject_path}")
        sys.exit(1)
    if subject_path.exists() and _is_likely_dataset_root(subject_path):
        log.error(
            "The provided --subject-path looks like a dataset root, not a subject folder: %s",
            subject_path,
        )
        log.error(
            "Please pass one subject directory, e.g. .../pwi/sub-XXXX, or use batch_run_pwi.sh for multiple subjects."
        )
        sys.exit(1)
    
    try:
        run_pwi_single(cfg, subject_path)
    except Exception as e:
        log.error(f"Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
