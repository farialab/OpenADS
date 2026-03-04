"""
Registration pipeline - pure orchestration.

Two-step registration process:
1. Affine: Subject brain mask → MNI152 DWI mask template
2. SyN: Affine-registered ADC → MNI152 ADC template

This module contains ZERO I/O operations and ZERO algorithm logic.
All I/O delegated to adapters, all algorithms to services.
"""
from pathlib import Path
from typing import Optional
from ads.domain.registration_data import (
    TemplatePaths, RegistrationInputs, RegistrationResult,
    AffineResult, SyNResult
)
from ads.domain.registration_spec import RegistrationSpec, AffineSpec, SyNSpec
from ads.adapters.ants_io import ANTsImage
from ads.adapters.transform_io import TransformIO
from ads.adapters.registration_paths import RegistrationPathBuilder
from ads.services.registration import RegistrationService
from ads.services.manifest import ManifestService
from ads.services.preprocessing import RegistrationNormalizationService


def process_subject(
    subject_dir: Path,
    subject_id: str,
    templates: TemplatePaths,
    inputs: RegistrationInputs,
    spec: RegistrationSpec,
    logger=None,
) -> RegistrationResult:
    """
    Register subject to template space.

    Pure orchestration - delegates all work to services and adapters.

    Args:
        subject_dir: Subject directory (e.g., output/sub-xxx/DWI/)
        subject_id: Subject identifier
        templates: Template file paths
        inputs: Subject input file paths
        spec: Registration configuration
        logger: Optional logger instance

    Returns:
        RegistrationResult with all output paths
    """
    _log(logger, f"[{subject_id}] Starting registration pipeline")

    output_dir = Path(subject_dir) / "registration"
    output_dir.mkdir(parents=True, exist_ok=True)

    path_builder = RegistrationPathBuilder(output_dir, subject_id)

    # Step 1: Affine registration
    _log(logger, f"[{subject_id}] Step 1: Affine registration")
    affine_result, affine_ants_result = _run_affine_stage(
        inputs, templates, spec.affine, path_builder, subject_id, logger
    )

    # Step 1b: Normalize affine DWI/ADC and save into registration output folder
    _log(logger, f"[{subject_id}] Step 1b: Saving normalized affine DWI/ADC")
    _run_affine_normalization_stage(affine_result, path_builder, subject_id, logger)

    # Step 2: SyN registration
    _log(logger, f"[{subject_id}] Step 2: SyN registration")
    syn_result, syn_ants_result = _run_syn_stage(
        affine_result, templates, spec.syn, path_builder, subject_id, logger
    )

    # Step 3: Build result
    result = RegistrationResult(
        affine=affine_result,
        syn=syn_result,
        manifest_yaml=path_builder.manifest_paths()['yaml'],
        manifest_json=path_builder.manifest_paths()['json'],
    )

    # Step 4: Generate manifest
    if spec.write_manifest:
        _log(logger, f"[{subject_id}] Writing manifest")
        _write_manifest(
            result, subject_id, inputs, templates, spec,
            affine_ants_result, syn_ants_result, path_builder
        )

    _log(logger, f"[{subject_id}] Registration complete")
    return result


def _run_affine_stage(
    inputs: RegistrationInputs,
    templates: TemplatePaths,
    spec: AffineSpec,
    path_builder: RegistrationPathBuilder,
    subject_id: str,
    logger,
) -> tuple:
    """Run affine registration stage."""
    # Load images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Loading images...")
    template = ANTsImage.load(str(templates.dwi_mask_template))
    mask = ANTsImage.load(str(inputs.mask))
    dwi = ANTsImage.load(str(inputs.dwi_brain))
    adc = ANTsImage.load(str(inputs.adc_brain))
    b0 = ANTsImage.load(str(inputs.b0_brain)) if inputs.b0_brain else None
    stroke = ANTsImage.load(str(inputs.stroke)) if inputs.stroke else None

    # Perform registration (delegated to service)
    _log(logger, f"[{subject_id}]   Computing affine registration...")
    ants_result = RegistrationService.perform_affine(template, mask, spec)

    # Apply transforms (delegated to service)
    _log(logger, f"[{subject_id}]   Applying transforms to all images...")
    images = {'dwi': dwi, 'adc': adc, 'mask': mask, 'b0': b0, 'stroke': stroke}
    registered = RegistrationService.apply_affine_to_images(
        template, images, ants_result['fwdtransforms']
    )

    # Get output paths (delegated to adapter)
    paths = path_builder.affine_paths(
        has_b0=inputs.b0_brain is not None,
        has_stroke=inputs.stroke is not None,
    )

    # Save images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving registered images...")
    for key, img in registered.items():
        if img is not None and paths[key] is not None:
            ANTsImage.save(img, str(paths[key]))

    # Save transforms (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving transforms...")
    transform_paths = TransformIO.save_transforms(
        ants_result, path_builder.output_dir, subject_id, 'aff'
    )

    # Build result object
    result = AffineResult(
        dwi=paths['dwi'],
        adc=paths['adc'],
        mask=paths['mask'],
        b0=paths['b0'],
        stroke=paths['stroke'],
        transform_fwd=transform_paths['fwd_affine'],
        transform_inv=transform_paths['inv_affine'],
    )

    return result, ants_result


def _run_syn_stage(
    affine_result: AffineResult,
    templates: TemplatePaths,
    spec: SyNSpec,
    path_builder: RegistrationPathBuilder,
    subject_id: str,
    logger,
) -> tuple:
    """Run SyN registration stage."""
    # Load affine-registered images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Loading affine-registered images...")
    adc_template = ANTsImage.load(str(templates.adc_template))
    adc_aff = ANTsImage.load(str(affine_result.adc))
    dwi_aff = ANTsImage.load(str(affine_result.dwi))
    mask_aff = ANTsImage.load(str(affine_result.mask))
    b0_aff = ANTsImage.load(str(affine_result.b0)) if affine_result.b0 else None
    stroke_aff = ANTsImage.load(str(affine_result.stroke)) if affine_result.stroke else None

    # Perform SyN registration (delegated to service)
    _log(logger, f"[{subject_id}]   Computing SyN registration...")
    ants_result = RegistrationService.perform_syn(adc_template, adc_aff, spec)

    # Apply transforms (delegated to service)
    _log(logger, f"[{subject_id}]   Applying SyN transforms...")
    images = {'dwi': dwi_aff, 'adc': adc_aff, 'mask': mask_aff, 'b0': b0_aff, 'stroke': stroke_aff}
    registered = RegistrationService.apply_syn_to_images(
        adc_template, images, ants_result['fwdtransforms']
    )

    # Get output paths (delegated to adapter)
    paths = path_builder.syn_paths(
        has_b0=affine_result.b0 is not None,
        has_stroke=affine_result.stroke is not None,
    )

    # Save images (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving SyN-registered images...")
    for key, img in registered.items():
        if img is not None and paths[key] is not None:
            ANTsImage.save(img, str(paths[key]))

    # Save transforms (delegated to adapter)
    _log(logger, f"[{subject_id}]   Saving SyN transforms...")
    transform_paths = TransformIO.save_transforms(
        ants_result, path_builder.output_dir, subject_id, 'affsyn'
    )

    # Build result object
    result = SyNResult(
        dwi=paths['dwi'],
        adc=paths['adc'],
        mask=paths['mask'],
        b0=paths['b0'],
        stroke=paths['stroke'],
        affine_fwd=transform_paths['fwd_affine'],
        warp_fwd=transform_paths['fwd_warp'],
        affine_inv=transform_paths['inv_affine'],
        warp_inv=transform_paths['inv_warp'],
    )

    return result, ants_result


def _run_affine_normalization_stage(
    affine_result: AffineResult,
    path_builder: RegistrationPathBuilder,
    subject_id: str,
    logger,
) -> dict:
    """Normalize affine DWI/ADC and save to registration folder."""
    norm_paths = path_builder.affine_normalized_paths()
    service = RegistrationNormalizationService()
    saved = service.normalize_affine_modalities(
        dwi_affine_path=affine_result.dwi,
        adc_affine_path=affine_result.adc,
        mask_affine_path=affine_result.mask,
        dwi_output_path=norm_paths['dwi'],
        adc_output_path=norm_paths['adc'],
    )
    _log(logger, f"[{subject_id}]   Saved normalized DWI: {saved['dwi_normalized'].name}")
    _log(logger, f"[{subject_id}]   Saved normalized ADC: {saved['adc_normalized'].name}")
    return saved


def _write_manifest(
    result: RegistrationResult,
    subject_id: str,
    inputs: RegistrationInputs,
    templates: TemplatePaths,
    spec: RegistrationSpec,
    affine_ants_result: dict,
    syn_ants_result: dict,
    path_builder: RegistrationPathBuilder,
) -> None:
    """Generate and write manifest files (delegated to service)."""
    # Prepare inputs dict
    inputs_dict = {
        'dwi_brain': str(inputs.dwi_brain),
        'adc_brain': str(inputs.adc_brain),
        'mask': str(inputs.mask),
        'b0_brain': str(inputs.b0_brain) if inputs.b0_brain else None,
        'stroke': str(inputs.stroke) if inputs.stroke else None,
    }

    # Prepare templates dict
    templates_dict = {
        'dwi_mask_template': str(templates.dwi_mask_template),
        'adc_template': str(templates.adc_template),
    }

    # Prepare config dict
    config_dict = {
        'affine': {
            'type_of_transform': spec.affine.type_of_transform,
            'reg_iterations': spec.affine.reg_iterations,
            'shrink_factors': spec.affine.shrink_factors,
            'smoothing_sigmas': spec.affine.smoothing_sigmas,
            'grad_step': spec.affine.grad_step,
            'metric': spec.affine.metric,
            'sampling': spec.affine.sampling,
            'verbose': spec.affine.verbose,
        },
        'syn': {
            'type_of_transform': spec.syn.type_of_transform,
            'reg_iterations': spec.syn.reg_iterations,
            'verbose': spec.syn.verbose,
        },
    }

    # Prepare ANTs results dict
    ants_dict = {
        'affine_fwdtransforms': affine_ants_result.get('fwdtransforms', []),
        'affine_invtransforms': affine_ants_result.get('invtransforms', []),
        'syn_fwdtransforms': syn_ants_result.get('fwdtransforms', []),
        'syn_invtransforms': syn_ants_result.get('invtransforms', []),
    }

    # Create manifest (delegated to service)
    manifest = ManifestService.create_manifest(
        subject_id=subject_id,
        result=result,
        inputs=inputs_dict,
        templates=templates_dict,
        config=config_dict,
        ants_results=ants_dict,
    )

    # Write manifest (delegated to service)
    manifest_paths = path_builder.manifest_paths()
    ManifestService.write_manifest(
        manifest, manifest_paths['yaml'], manifest_paths['json']
    )


def _log(logger, msg: str) -> None:
    """Unified logging helper."""
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg)
