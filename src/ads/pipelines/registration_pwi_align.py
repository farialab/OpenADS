"""
PWI registration pipeline - pure orchestration.

Two-stage registration process:
1. PWI → DWI coregistration (affine)
2. DWI → MNI registration (reuse DWI transforms)

This module contains ZERO I/O operations and ZERO algorithm logic.
All I/O delegated to adapters, all algorithms to services.
"""
from pathlib import Path
from typing import Optional
from ads.domain.pwi_registration_data import (
    PWIRegistrationInputs, PWIRegistrationResult,
    PWICoregResult, PWIMNIResult
)
from ads.domain.pwi_registration_spec import PWIRegistrationSpec
from ads.adapters.ants_io import ANTsImage, ANTsRegistration
from ads.adapters.pwi_registration_paths import PWIRegistrationPathBuilder
from ads.adapters.transform_io import TransformIO
from ads.services.pwi_coregistration import PWICoregistrationService
from ads.services.transform_discovery import TransformDiscoveryService
from ads.services.dwi_registration_reuse import DWIRegistrationReuseService
from ads.services.preprocessing import RegistrationNormalizationService


def process_subject(
    subject_dir: Path,
    subject_id: str,
    inputs: PWIRegistrationInputs,
    templates: dict,  # {dwi_mask_template: Path, adc_template: Path}
    spec: PWIRegistrationSpec,
    logger=None,
) -> PWIRegistrationResult:
    """
    Register PWI to DWI and then to MNI.

    Pure orchestration - delegates all work to services and adapters.

    Args:
        subject_dir: Subject directory (e.g., output/sub-xxx/PWI/)
        subject_id: Subject identifier
        inputs: PWI input file paths
        templates: Template paths (dwi_mask_template, adc_template)
        spec: PWI registration configuration
        logger: Optional logger instance

    Returns:
        PWIRegistrationResult with all output paths
    """
    _log(logger, f"[{subject_id}] Starting PWI registration pipeline")

    output_dir = Path(subject_dir) / "registration"
    output_dir.mkdir(parents=True, exist_ok=True)

    path_builder = PWIRegistrationPathBuilder(output_dir, subject_id)

    # Check if we can reuse existing transforms (no recomputation needed)
    skip_dwi_mni_computation = False
    if spec.skip_reregistration and not spec.force_recompute:
        _log(logger, f"[{subject_id}] Checking for existing registration transforms...")
        if DWIRegistrationReuseService.check_dwi_registration_exists(subject_dir):
            _log(logger, f"[{subject_id}] Found existing registration transforms")
            _log(logger, f"[{subject_id}] Will search: PWI/registration → DWI/registration (fallback)")
            skip_dwi_mni_computation = True
            # NO COPY - TransformDiscoveryService will find and use files from actual location

    # Step 1: PWI → DWI coregistration
    _log(logger, f"[{subject_id}] Step 1: PWI → DWI coregistration")
    coreg_result = _run_pwi_to_dwi_stage(
        inputs, spec, path_builder, subject_id, logger
    )

    # Step 2: DWI → MNI registration (reuse DWI transforms or apply)
    _log(logger, f"[{subject_id}] Step 2: DWI → MNI registration")
    mni_result = _run_dwi_to_mni_stage(
        coreg_result, subject_dir, subject_id, templates, spec,
        path_builder, skip_dwi_mni_computation, logger
    )

    # Build result
    result = PWIRegistrationResult(
        coreg=coreg_result,
        mni=mni_result,
    )

    _log(logger, f"[{subject_id}] PWI registration complete")
    return result


def _run_pwi_to_dwi_stage(
    inputs: PWIRegistrationInputs,
    spec: PWIRegistrationSpec,
    path_builder: PWIRegistrationPathBuilder,
    subject_id: str,
    logger,
) -> PWICoregResult:
    """Run PWI→DWI coregistration and masking."""
    _log(logger, f"[{subject_id}]   Loading images...")

    # Load masks for registration (delegated to adapter)
    pwi_mask = ANTsImage.load(str(inputs.pwi_mask))
    dwi_mask = ANTsImage.load(str(inputs.dwi_mask))

    # Load images to transform (delegated to adapter)
    ttp = ANTsImage.load(str(inputs.ttp))
    hp_manual = ANTsImage.load(str(inputs.hp_manual)) if inputs.hp_manual else None
    # NOTE: 4D PWI is NOT processed (matches old version behavior)

    # Perform registration (delegated to service)
    _log(logger, f"[{subject_id}]   Computing PWI→DWI affine registration...")
    ants_result = PWICoregistrationService.register_pwi_to_dwi(
        pwi_mask, dwi_mask, spec.coreg
    )
    transform_paths = _save_pwi_coreg_transforms(
        ants_result=ants_result,
        path_builder=path_builder,
        subject_id=subject_id,
    )

    # Apply transforms to 3D images (delegated to service)
    _log(logger, f"[{subject_id}]   Applying transforms to TTP and HP_manual...")
    images_3d = {'ttp': ttp, 'hp_manual': hp_manual}
    registered_3d = PWICoregistrationService.apply_transforms_to_pwi(
        dwi_mask, images_3d, ants_result['fwdtransforms'],
        spec.interpolator_images, spec.interpolator_masks
    )

    # NOTE: 4D PWI transformation removed to match old version (only TTP and HP_manual processed)

    # Get output paths (delegated to adapter)
    has_hp = inputs.hp_manual is not None
    on_dwi_paths = path_builder.pwi_on_dwi_paths(has_hp=has_hp)

    # Save registered images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving registered images...")
    ANTsImage.save(registered_3d['ttp'], str(on_dwi_paths['ttp']))
    if registered_3d['hp_manual'] is not None:
        ANTsImage.save(registered_3d['hp_manual'], str(on_dwi_paths['hp_manual']))
    # NOTE: 4D PWI save removed (matches old version)

    # Apply brain masking (delegated to service)
    _log(logger, f"[{subject_id}]   Applying brain masks...")

    # Get mask data
    dwi_mask_data = dwi_mask.numpy()

    # Mask TTP
    ttp_data = registered_3d['ttp'].numpy()
    ttp_masked_data = PWICoregistrationService.apply_brain_mask_numpy(
        ttp_data, dwi_mask_data
    )
    ttp_masked = ANTsImage.from_numpy(
        ttp_masked_data, registered_3d['ttp'].origin,
        registered_3d['ttp'].spacing, registered_3d['ttp'].direction
    )

    # Mask HP_manual
    hp_manual_masked = None
    if registered_3d['hp_manual'] is not None:
        hp_data = registered_3d['hp_manual'].numpy()
        hp_masked_data = PWICoregistrationService.apply_brain_mask_numpy(
            hp_data, dwi_mask_data
        )
        hp_manual_masked = ANTsImage.from_numpy(
            hp_masked_data, registered_3d['hp_manual'].origin,
            registered_3d['hp_manual'].spacing, registered_3d['hp_manual'].direction
        )

    # NOTE: 4D PWI masking removed (matches old version)

    # Get masked output paths (delegated to adapter)
    masked_paths = path_builder.masked_paths(has_hp=has_hp)

    # Save masked images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving masked images...")
    ANTsImage.save(ttp_masked, str(masked_paths['ttp']))
    if hp_manual_masked is not None:
        ANTsImage.save(hp_manual_masked, str(masked_paths['hp_manual']))
    # NOTE: 4D PWI masked save removed (matches old version)

    # Build result object
    result = PWICoregResult(
        ttp_on_dwi=on_dwi_paths['ttp'],
        hp_manual_on_dwi=on_dwi_paths['hp_manual'],
        transform_fwd=transform_paths['fwd_affine'],
        transform_inv=transform_paths['inv_affine'],
        ttp_masked=masked_paths['ttp'],
        hp_manual_masked=masked_paths['hp_manual'],
        # NOTE: pwi fields removed (matches old version - only TTP and HP_manual)
    )

    return result


def _run_dwi_to_mni_stage(
    coreg_result: PWICoregResult,
    subject_dir: Path,
    subject_id: str,
    templates: dict,
    spec: PWIRegistrationSpec,
    path_builder: PWIRegistrationPathBuilder,
    skip_computation: bool,
    logger,
) -> PWIMNIResult:
    """Run DWI→MNI registration using existing transforms."""
    has_hp = coreg_result.hp_manual_masked is not None

    if skip_computation:
        _log(logger, f"[{subject_id}]   Using copied DWI transforms")
    else:
        _log(logger, f"[{subject_id}]   Finding DWI registration transforms...")

    # Discover DWI transforms (delegated to service)
    artifacts = TransformDiscoveryService.find_dwi_transforms(
        subject_dir, subject_id,
        templates['dwi_mask_template'], templates['adc_template']
    )

    # Load images for downstream mapping.
    # raw_ttp_masked=True  -> use unmasked TTP on DWI space (avoid additional masking loss)
    # raw_ttp_masked=False -> keep legacy behavior using desc-brain TTP
    ttp_source_path = coreg_result.ttp_on_dwi if spec.raw_ttp_masked else coreg_result.ttp_masked
    _log(logger, f"[{subject_id}]   Loading TTP for downstream: {Path(ttp_source_path).name}")
    ttp_for_mni = ANTsImage.load(str(ttp_source_path))
    hp_manual_masked = ANTsImage.load(str(coreg_result.hp_manual_masked)) if has_hp else None
    # NOTE: pwi_masked loading removed (matches old version - only TTP and HP_manual)

    # Load template (delegated to adapter)
    dwi_mask_template = ANTsImage.load(str(artifacts.dwi_mask_template))

    # Apply affine transforms (delegated to service)
    _log(logger, f"[{subject_id}]   Applying affine transforms to MNI space...")
    aff_transforms = [str(artifacts.aff_mat)]

    # Apply to 3D images
    ttp_aff = ANTsRegistration.apply_transforms(
        fixed=dwi_mask_template,
        moving=ttp_for_mni,
        transforms=aff_transforms,
        interpolator=spec.interpolator_images
    )

    hp_manual_aff = None
    if hp_manual_masked is not None:
        hp_manual_aff = ANTsRegistration.apply_transforms(
            fixed=dwi_mask_template,
            moving=hp_manual_masked,
            transforms=aff_transforms,
            interpolator=spec.interpolator_masks
        )
        hp_manual_aff = ANTsImage.binarize(hp_manual_aff)

    # NOTE: 4D PWI affine transform removed (matches old version)

    # Apply SyN transforms (delegated to service)
    _log(logger, f"[{subject_id}]   Applying SyN transforms to MNI space...")
    syn_transforms = [str(artifacts.warp), str(artifacts.syn_mat)]

    # Apply to 3D images
    ttp_affsyn = ANTsRegistration.apply_transforms(
        fixed=dwi_mask_template,
        moving=ttp_aff,
        transforms=syn_transforms,
        interpolator=spec.interpolator_images
    )

    hp_manual_affsyn = None
    if hp_manual_aff is not None:
        hp_manual_affsyn = ANTsRegistration.apply_transforms(
            fixed=dwi_mask_template,
            moving=hp_manual_aff,
            transforms=syn_transforms,
            interpolator=spec.interpolator_masks
        )
        hp_manual_affsyn = ANTsImage.binarize(hp_manual_affsyn)

    # NOTE: 4D PWI affsyn transform removed (matches old version)

    # Get output paths (delegated to adapter)
    mni_paths = path_builder.mni_paths(has_hp=has_hp)

    # Save images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving MNI-space images...")

    # Save affine results
    ANTsImage.save(ttp_aff, str(mni_paths['aff']['ttp']))
    if hp_manual_aff is not None:
        ANTsImage.save(hp_manual_aff, str(mni_paths['aff']['hp_manual']))
    # NOTE: 4D PWI save removed (matches old version)

    # Save normalized TTP in MNI affine space
    ttp_aff_norm_path = path_builder.ttp_aff_normalized_path()
    norm_service = RegistrationNormalizationService()
    norm_service.normalize_single_modality(
        image_path=mni_paths['aff']['ttp'],
        mask_path=artifacts.dwi_mask_template,
        output_path=ttp_aff_norm_path,
    )

    # Save affsyn results
    ANTsImage.save(ttp_affsyn, str(mni_paths['affsyn']['ttp']))
    if hp_manual_affsyn is not None:
        ANTsImage.save(hp_manual_affsyn, str(mni_paths['affsyn']['hp_manual']))
    # NOTE: 4D PWI save removed (matches old version)

    # Build result object
    result = PWIMNIResult(
        ttp_aff=mni_paths['aff']['ttp'],
        ttp_aff_norm=ttp_aff_norm_path,
        ttp_affsyn=mni_paths['affsyn']['ttp'],
        hp_manual_aff=mni_paths['aff']['hp_manual'],
        hp_manual_affsyn=mni_paths['affsyn']['hp_manual'],
        # NOTE: pwi fields removed (matches old version - only TTP and HP_manual)
        from_copy=skip_computation,
    )

    return result


def _save_pwi_coreg_transforms(
    ants_result: dict,
    path_builder: PWIRegistrationPathBuilder,
    subject_id: str,
) -> dict:
    """Persist forward/inverse affine matrices for TTP alignment."""
    transform_paths = path_builder.coreg_transform_paths()

    fwd_mat = next((str(t) for t in ants_result.get('fwdtransforms', []) if str(t).endswith('.mat')), None)
    inv_mat = next((str(t) for t in ants_result.get('invtransforms', []) if str(t).endswith('.mat')), None)

    if fwd_mat is None:
        raise FileNotFoundError(
            f"[{subject_id}] Missing forward affine matrix in PWI coregistration result."
        )
    if inv_mat is None:
        raise FileNotFoundError(
            f"[{subject_id}] Missing inverse affine matrix in PWI coregistration result."
        )

    TransformIO.copy_transform(fwd_mat, transform_paths['fwd_affine'])
    TransformIO.copy_transform(inv_mat, transform_paths['inv_affine'])
    return transform_paths


def _log(logger, msg: str) -> None:
    """Unified logging helper."""
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg)
