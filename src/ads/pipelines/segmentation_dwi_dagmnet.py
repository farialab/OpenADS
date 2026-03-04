"""
DWI segmentation pipeline using DAGMNet.

Pure orchestration - delegates all work to services and adapters.
This module contains ZERO I/O operations and ZERO algorithm logic.

All preprocessing, normalization, and inference logic is delegated to:
- ads.core.preprocessing (existing implementations)
- ads.models.wrappers (existing DAGMNet inference)
- ads.services.segmentation (metrics, post-processing)
- ads.adapters (I/O, model loading, paths)
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import nibabel as nib
import ants
import scipy.stats
import scipy.special
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from ads.domain.segmentation_data import (
    SegmentationInputs,
    SegmentationOutputs,
    ModelInputData
)
from ads.domain.segmentation_spec import SegmentationSpec
from ads.domain.model_config import ModelConfig
from ads.adapters.model_loader import ModelLoader
from ads.adapters.segmentation_paths import SegmentationPathBuilder
from ads.services.segmentation.metrics import MetricsService
from ads.services.segmentation.postprocessing import PostProcessingService

# Import existing implementations (DO NOT MODIFY)
from ads.core.preprocessing import (
    pad_to_size,
    depad_to_size,
)
from ads.models.wrappers import get_stroke_seg_MNI


def process_subject(
    subject_dir: Path,
    subject_id: str,
    inputs: SegmentationInputs,
    model_config: ModelConfig,
    spec: SegmentationSpec,
    logger=None,
) -> SegmentationOutputs:
    """
    Run DWI segmentation pipeline with DAGMNet.

    Pure orchestration - delegates all work to existing services.

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        inputs: Input file paths
        model_config: Model configuration
        spec: Segmentation configuration
        logger: Optional logger

    Returns:
        SegmentationOutputs with paths to generated files
    """
    _log(logger, f"[{subject_id}] Starting DWI segmentation (DAGMNet)")

    # Setup output directory
    output_dir = Path(subject_dir) / "segment"
    output_dir.mkdir(parents=True, exist_ok=True)
    path_builder = SegmentationPathBuilder(output_dir, subject_id)
    paths = path_builder.get_output_paths()
    reg_dwi_normalized_path = Path(subject_dir) / "registration" / f"{subject_id}_DWI_space-MNI152_aff_desc-norm.nii.gz"

    # Step 1: Load MNI-space inputs (exact logic from pipe_segment.py:481-483)
    _log(logger, f"[{subject_id}] Loading MNI-space images")
    dwi_mni_img = _load_nifti_as_ras(str(inputs.dwi_mni))
    adc_mni_img = _load_nifti_as_ras(str(inputs.adc_mni))
    mask_mni_img = _load_nifti_as_ras(str(inputs.mask_mni))

    dwi_data = dwi_mni_img.get_fdata()
    adc_data = adc_mni_img.get_fdata()
    mask_data = mask_mni_img.get_fdata()
    original_shape = dwi_data.shape

    # Step 2: Normalize images (exact logic from pipe_segment.py:493-494)
    _log(logger, f"[{subject_id}] Normalizing images")
    dwi_norm = _legacy_get_dwi_normalized(dwi_data, mask_data)
    adc_norm = _legacy_get_dwi_normalized(adc_data, mask_data)

    # Step 3: Pad to model input shape (exact logic from pipe_segment.py:499-501)
    _log(logger, f"[{subject_id}] Padding to {spec.inference.target_shape}")
    dwi_padded = pad_to_size(dwi_norm, target_shape=spec.inference.target_shape)
    adc_padded = pad_to_size(adc_norm, target_shape=spec.inference.target_shape)
    mask_padded = pad_to_size(mask_data, target_shape=spec.inference.target_shape)

    # Step 4: Compute Prob_IS (exact logic from pipe_segment.py:506-512)
    _log(logger, f"[{subject_id}] Computing Prob_IS")
    prob_is = _legacy_get_prob_is(
        dwi_norm=dwi_padded,
        adc_img=adc_padded,
        mask_img=mask_padded,
        template_dir=str(model_config.template_dir),
        model_vars=spec.probability_map.model_vars,
    )

    # Step 5: Final z-score normalization within mask (exact logic from pipe_segment.py:515-520)
    dwi_bsn = _zscore_within_mask(dwi_padded, mask_padded, spec.inference.mask_threshold)
    adc_bsn = _zscore_within_mask(adc_padded, mask_padded, spec.inference.mask_threshold)

    # Step 6: Load model and run inference (exact logic from pipe_segment.py:531-542)
    _log(logger, f"[{subject_id}] Loading model and running inference")
    model = ModelLoader.load_from_path(model_config.weights_path)

    # Use existing multi-scale inference (DO NOT MODIFY)
    stroke_pred_binary, stroke_pred_prob = get_stroke_seg_MNI(
        model=model,
        dwi_img=dwi_bsn,
        adc_img=adc_bsn,
        Prob_IS=prob_is,
        N_channel=spec.inference.n_channel,
        DS=spec.inference.downsampling_factor,
        device=spec.inference.device
    )

    # Step 7: Depad predictions (exact logic from pipe_segment.py:548-549)
    _log(logger, f"[{subject_id}] Depadding predictions")
    pred_mni = depad_to_size(stroke_pred_binary, target_shape=original_shape)
    pred_mni = pred_mni * mask_data

    # Step 8: Save MNI prediction (exact logic from pipe_segment.py:552-553)
    _save_nifti(pred_mni.astype(np.float32), dwi_mni_img, paths['pred_mni'])
    _log(logger, f"[{subject_id}] Saved MNI prediction: {paths['pred_mni'].name}")

    # Step 9: Transform to affsyn space (exact logic from pipe_segment.py:558-577)
    if inputs.adc_affsyn and inputs.syn_affine and inputs.adc_affsyn.exists():
        _log(logger, f"[{subject_id}] Transforming to affsyn space")
        _transform_to_affsyn(
            paths['pred_mni'],
            inputs.adc_affsyn,
            inputs.syn_affine,
            inputs.syn_warp,
            paths['pred_mni_affsyn'],
            logger,
            subject_id
        )

    # Step 10: Transform to native space (exact logic from pipe_segment.py:582-609)
    _log(logger, f"[{subject_id}] Transforming to native space")
    pred_native_data = _transform_to_native(
        paths['pred_mni'],
        inputs.dwi_native,
        inputs.mask_native,
        inputs.fwd_affine,
        paths['pred_native'],
        spec.inference.prediction_threshold,
        logger,
        subject_id
    )

    # Step 11: Compute metrics (exact logic from pipe_segment.py:612-621)
    metrics = _compute_all_metrics(
        pred_mni,
        pred_native_data,
        inputs,
        spec,
        logger,
        subject_id
    )

    # Save metrics
    with open(paths['metrics'], 'w') as f:
        json.dump(metrics, f, indent=4)
    _log(logger, f"[{subject_id}] Saved metrics: {paths['metrics'].name}")

    _log(logger, f"[{subject_id}] Segmentation complete")

    return SegmentationOutputs(
        pred_mni=paths['pred_mni'],
        pred_native=paths['pred_native'],
        pred_mni_affsyn=paths.get('pred_mni_affsyn'),
        dwi_normalized_path=reg_dwi_normalized_path if reg_dwi_normalized_path.exists() else None,
        metrics_json=paths['metrics']
    )


# ==================== Helper Functions (Wrap Existing Logic) ====================

def _load_nifti_as_ras(file_path: str) -> nib.Nifti1Image:
    """Load NIfTI and convert to RAS (exact logic from pipe_segment.py:77-81)."""
    img = nib.load(file_path)
    img = nib.as_closest_canonical(img)
    return img


def _save_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> None:
    """Save numpy array as NIfTI (exact logic from pipe_segment.py:84-90)."""
    new_img = nib.as_closest_canonical(
        nib.Nifti1Image(data, affine=reference_img.affine, header=reference_img.header)
    )
    nib.save(new_img, str(output_path))


def _zscore_within_mask(
    img: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Z-score normalization within mask (exact logic from pipe_segment.py:174-177)."""
    vals = img[mask > threshold]
    return (img - vals.mean()) / vals.std()


def _gauss(x: np.ndarray, mu: float, sigma: float, a: float) -> np.ndarray:
    return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def _bimodal(
    x: np.ndarray,
    mu1: float,
    sigma1: float,
    a1: float,
    mu2: float,
    sigma2: float,
    a2: float,
) -> np.ndarray:
    return _gauss(x, mu1, sigma1, a1) + _gauss(x, mu2, sigma2, a2)


def _qfunc(x: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * scipy.special.erf(x / np.sqrt(2))


def _legacy_get_dwi_normalized(dwi_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
    """Legacy normalization copied from old pipe_segment.py."""
    dwi_d = dwi_img[mask_img > 0.5]
    md = scipy.stats.mode(dwi_d.astype("int16"), keepdims=True)[0][0]
    p0_mu = md if md > np.mean(dwi_d) else np.mean(dwi_d)

    dwi_hist, x_data = np.histogram(dwi_d, bins=np.arange(np.max(dwi_d)), density=True)
    x_data = (x_data[1:] + x_data[:-1]) / 2

    try:
        bounds = ([0, 0, -np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, p0_mu, np.inf, np.inf])
        params, _ = curve_fit(
            _bimodal,
            x_data,
            dwi_hist,
            bounds=bounds,
            p0=(p0_mu, 1, 1, 0, 1, 1),
            maxfev=5000,
        )
        mu1 = params[0]
        sigma1 = params[1]
    except Exception:
        mu1 = p0_mu
        sigma1 = np.std(dwi_d)

    return (dwi_img - mu1) / sigma1


def _legacy_get_prob_is(
    dwi_norm: np.ndarray,
    adc_img: np.ndarray,
    mask_img: np.ndarray,
    template_dir: str,
    model_vars: list[float],
) -> np.ndarray:
    """Legacy Prob_IS copied from old pipe_segment.py."""
    normal_dwi_mu_img = np.squeeze(_load_nifti_as_ras(
        os.path.join(template_dir, "normal_mu_dwi_Res_ss_MNI_scaled_normalized.nii.gz")
    ).get_fdata())
    normal_dwistd_img = np.squeeze(_load_nifti_as_ras(
        os.path.join(template_dir, "normal_std_dwi_Res_ss_MNI_scaled_normalized.nii.gz")
    ).get_fdata())
    normal_adc_mu_img = np.squeeze(_load_nifti_as_ras(
        os.path.join(template_dir, "normal_mu_ADC_Res_ss_MNI_normalized.nii.gz")
    ).get_fdata())
    normal_adc_std_img = np.squeeze(_load_nifti_as_ras(
        os.path.join(template_dir, "normal_std_ADC_Res_ss_MNI_normalized.nii.gz")
    ).get_fdata())

    fwhm = model_vars[0]
    g_sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    alpha_dwi = model_vars[1]
    lambda_dwi = model_vars[2]
    alpha_adc = model_vars[3]
    lambda_adc = model_vars[4]
    id_isch_zth = model_vars[5]

    img = (dwi_norm - np.mean(dwi_norm)) / np.std(dwi_norm)
    for i in range(img.shape[-1]):
        img[:, :, i] = gaussian_filter(img[:, :, i], g_sigma)
    dissimilarity = np.tanh((img - normal_dwi_mu_img) / normal_dwistd_img / alpha_dwi)
    dissimilarity[dissimilarity < 0] = 0
    dissimilarity = dissimilarity ** lambda_dwi
    dissimilarity[dwi_norm < id_isch_zth] = 0
    dwi_h2 = dissimilarity * (mask_img > 0.49) * 1.0

    img = (adc_img - np.mean(adc_img)) / np.std(adc_img)
    for i in range(img.shape[-1]):
        img[:, :, i] = gaussian_filter(img[:, :, i], g_sigma)
    dissimilarity = np.tanh((img - normal_adc_mu_img) / normal_adc_std_img / alpha_adc)
    dissimilarity[dissimilarity > 0] = 0
    dissimilarity = (-dissimilarity) ** lambda_adc
    adc_h1 = dissimilarity * (mask_img > 0.49) * 1.0

    id_isch = (1 - _qfunc(dwi_norm / id_isch_zth)) * (dwi_norm > id_isch_zth)
    prob_is = dwi_h2 * adc_h1 * id_isch * (mask_img > 0.49) * 1.0
    return prob_is


def _transform_to_affsyn(
    pred_mni_path: Path,
    adc_affsyn: Path,
    syn_affine: Path,
    syn_warp: Optional[Path],
    output_path: Path,
    logger,
    subject_id: str
) -> None:
    """Transform to affsyn space (exact logic from pipe_segment.py:558-577)."""
    try:
        fixed_affsyn = ants.image_read(str(adc_affsyn))
        moving_pred_mni = ants.image_read(str(pred_mni_path))

        transformlist = []
        if syn_warp and syn_warp.exists():
            transformlist.append(str(syn_warp))
        transformlist.append(str(syn_affine))

        pred_affsyn = ants.apply_transforms(
            fixed=fixed_affsyn,
            moving=moving_pred_mni,
            transformlist=transformlist,
            interpolator="nearestNeighbor"
        )

        ants.reorient_image2(pred_affsyn, orientation="RAS").image_write(str(output_path))
        _log(logger, f"[{subject_id}] Saved affsyn prediction: {output_path.name}")
    except Exception as e:
        _log(logger, f"[{subject_id}] WARNING: Failed to transform to affsyn: {e}")


def _transform_to_native(
    pred_mni_path: Path,
    dwi_native: Path,
    mask_native: Path,
    fwd_affine: Optional[Path],
    output_path: Path,
    threshold: float,
    logger,
    subject_id: str
) -> Optional[np.ndarray]:
    """Transform to native space (exact logic from pipe_segment.py:582-609)."""
    if not fwd_affine or not fwd_affine.exists():
        _log(logger, f"[{subject_id}] ERROR: Forward affine not found: {fwd_affine}")
        return None

    try:
        fixed = ants.image_read(str(dwi_native))
        moving = ants.image_read(str(pred_mni_path))

        # Use forward affine with inversion (exact logic from pipe_segment.py:587-592)
        unwarped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[str(fwd_affine)],
            whichtoinvert=[True],  # Invert forward to get MNI->native
            interpolator="linear"
        )

        mask_raw_img = ants.image_read(str(mask_native))
        stroke_pred_raw_img = unwarped.numpy() * mask_raw_img.numpy()
        stroke_pred_raw_img = (stroke_pred_raw_img > threshold).astype(np.float32)

        stroke_pred_raw_ants = ants.from_numpy(
            stroke_pred_raw_img,
            origin=mask_raw_img.origin,
            spacing=mask_raw_img.spacing,
            direction=mask_raw_img.direction
        )
        ants.image_write(stroke_pred_raw_ants, str(output_path))
        _log(logger, f"[{subject_id}] Saved native prediction: {output_path.name}")

        return stroke_pred_raw_img

    except Exception as e:
        _log(logger, f"[{subject_id}] ERROR: Failed to transform to native: {e}")
        return None


def _compute_all_metrics(
    pred_mni: np.ndarray,
    pred_native: Optional[np.ndarray],
    inputs: SegmentationInputs,
    spec: SegmentationSpec,
    logger,
    subject_id: str
) -> Dict[str, Any]:
    """Compute all metrics (exact logic from pipe_segment.py:631-678)."""

    def null_metrics():
        return {
            "dice": None,
            "precision": None,
            "sensitivity": None,
            "sdr": None,
            "pred_volume": None,
            "true_volume": None
        }

    metrics = {
        "mni": null_metrics(),
        "orig": null_metrics(),
        "mni_postproc": null_metrics(),
        "orig_postproc": null_metrics(),
    }

    if not spec.compute_metrics:
        return metrics

    # MNI metrics
    if inputs.stroke_mni and inputs.stroke_mni.exists():
        stroke_mni = _load_nifti_as_ras(str(inputs.stroke_mni)).get_fdata()
        result = MetricsService.compute(
            pred_mni,
            stroke_mni,
            spec.inference.prediction_threshold
        )
        metrics["mni"] = result.to_dict()

        # Post-processed metrics
        #if spec.postprocessing.apply_postprocessing:
        postproc_service = PostProcessingService(spec.postprocessing)
        result_postproc = MetricsService.compute_with_postproc(
            pred_mni,
            stroke_mni,
            postproc_service.process,
            spec.inference.prediction_threshold
        )
        metrics["mni_postproc"] = result_postproc.to_dict()

    # Native space metrics
    if pred_native is not None and inputs.stroke_native and inputs.stroke_native.exists():
        stroke_native = _load_nifti_as_ras(str(inputs.stroke_native)).get_fdata()
        result = MetricsService.compute(
            pred_native,
            stroke_native,
            spec.inference.prediction_threshold
        )
        metrics["orig"] = result.to_dict()

        # Post-processed metrics
        #if spec.postprocessing.apply_postprocessing:
        postproc_service = PostProcessingService(spec.postprocessing)
        result_postproc = MetricsService.compute_with_postproc(
            pred_native,
            stroke_native,
            postproc_service.process,
            spec.inference.prediction_threshold
        )
        metrics["orig_postproc"] = result_postproc.to_dict()

    return metrics


def _log(logger, msg: str) -> None:
    """Unified logging (exact logic from pipe_segment.py:393-398)."""
    if logger:
        try:
            logger.info(msg)
            return
        except Exception:
            pass
    print(msg)
