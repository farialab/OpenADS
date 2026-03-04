"""
PWI segmentation pipeline using UNet3D.

Refactored from pipe_inference_pwi_v2.py.
Pure orchestration - delegates all work to services and adapters.

All preprocessing, normalization, and inference logic is delegated to:
- ads.models.unet3d_pwi (UNet3D model)
- ads.services.segmentation (metrics, post-processing)
- ads.adapters (I/O, model loading, paths)
"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
import nibabel as nib
import ants
import torch
import torch.nn.functional as F

from ads.domain.segmentation_data import (
    PWISegmentationInputs,
    SegmentationOutputs
)
from ads.domain.segmentation_spec import SegmentationSpec
from ads.domain.model_config import ModelConfig
from ads.adapters.model_loader import ModelLoader
from ads.adapters.segmentation_paths import PWIPathBuilder
from ads.services.segmentation.metrics import MetricsService
from ads.services.segmentation.postprocessing import PostProcessingService


# Constants from original implementation
INPUT_SIZE = (176, 224, 176)  # UNet3D uses different size than DAGMNet
THRESH = 0.5


def _log(logger, msg: str):
    """Helper to log if logger available."""
    if logger:
        logger.info(msg)
    else:
        print(msg)


def _load_nib_ras(path: str) -> nib.Nifti1Image:
    """Load NIfTI in RAS+ orientation."""
    return nib.as_closest_canonical(nib.load(path))


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score normalization."""
    return (x - x.mean()) / (x.std() + eps)


def _resize3d(arr: np.ndarray, size: tuple, mode: str) -> torch.Tensor:
    """Resize 3D array to target size."""
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if mode == "trilinear":
        y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    elif mode == "nearest":
        y = F.interpolate(x, size=size, mode="nearest")
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return y.squeeze(0).squeeze(0)


def process_subject(
    subject_dir: Path,
    subject_id: str,
    inputs: PWISegmentationInputs,
    model_config: ModelConfig,
    spec: SegmentationSpec,
    logger=None,
) -> SegmentationOutputs:
    """
    Run PWI hypoperfusion segmentation pipeline with UNet3D.

    Pure orchestration - delegates all work to existing services.
    Uses simpler inference strategy than DAGMNet (no Z-stride).

    Args:
        subject_dir: Subject directory
        subject_id: Subject identifier
        inputs: PWI input file paths
        model_config: Model configuration
        spec: Segmentation configuration
        logger: Optional logger

    Returns:
        SegmentationOutputs with paths to generated files
    """
    _log(logger, f"[{subject_id}] Starting PWI segmentation (UNet3D)")

    # Setup output directory
    output_dir = Path(subject_dir) / "segment"
    output_dir.mkdir(parents=True, exist_ok=True)
    path_builder = PWIPathBuilder(output_dir, subject_id)
    paths = path_builder.get_output_paths()

    # Step 1: Load MNI-space inputs
    _log(logger, f"[{subject_id}] Loading MNI-space images")
    dwi_mni_img = _load_nib_ras(str(inputs.dwi_mni))
    adc_mni_img = _load_nib_ras(str(inputs.adc_mni))
    ttp_mni_img = _load_nib_ras(str(inputs.ttp_mni))
    mask_mni_img = _load_nib_ras(str(inputs.mask_mni))

    dwi_data = dwi_mni_img.get_fdata()
    adc_data = adc_mni_img.get_fdata()
    ttp_data = ttp_mni_img.get_fdata()
    mask_data = mask_mni_img.get_fdata()
    original_shape = dwi_data.shape

    # Step 2: Prepare inputs (resize and normalize)
    _log(logger, f"[{subject_id}] Resizing and normalizing images")
    vols = []

    # Channel 1: DWI (resize with trilinear)
    dwi_resized = _resize3d(dwi_data, INPUT_SIZE, 'trilinear')
    vols.append(dwi_resized)

    # Channel 2: ADC (resize with trilinear)
    adc_resized = _resize3d(adc_data, INPUT_SIZE, 'trilinear')
    vols.append(adc_resized)

    # Channel 3: TTP (resize and z-score normalize)
    ttp_resized = _resize3d(ttp_data, INPUT_SIZE, 'trilinear')
    ttp_normalized = _zscore(ttp_resized)
    vols.append(ttp_normalized)

    # Channel 4: Stroke (optional, for 4-channel mode)
    if inputs.stroke_mni and inputs.stroke_mni.exists():
        stroke_mni_img = _load_nib_ras(str(inputs.stroke_mni))
        stroke_data = stroke_mni_img.get_fdata()
        stroke_resized = (_resize3d(stroke_data, INPUT_SIZE, 'nearest') > 0.5).float()
        vols.append(stroke_resized)
        _log(logger, f"[{subject_id}] Loaded stroke channel (4-channel mode)")

    # Stack inputs [C, D, H, W]
    inputs_tensor = torch.stack(vols, dim=0).unsqueeze(0)
    _log(logger, f"[{subject_id}] Input shape: {inputs_tensor.shape}")

    # Step 3: Load model and run inference
    _log(logger, f"[{subject_id}] Loading model and running inference")
    model = ModelLoader.load_model(model_config)
    device = torch.device(spec.inference.device)
    model = model.to(device)
    inputs_tensor = inputs_tensor.to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(inputs_tensor)
        pred_prob = torch.sigmoid(logits)[0, 0]

    # Step 4: Resize back to original shape
    _log(logger, f"[{subject_id}] Resizing prediction back to original shape")
    pred_resized = F.interpolate(
        pred_prob.unsqueeze(0).unsqueeze(0),
        size=original_shape,
        mode='nearest'
    )[0, 0].cpu().numpy()

    # Apply mask
    pred_masked = pred_resized * (mask_data > 0.5)

    # Threshold
    pred_bin = (pred_masked > THRESH).astype(np.uint8)

    # Step 5: Post-process if enabled
    if spec.postprocessing.apply_postprocessing:
        _log(logger, f"[{subject_id}] Post-processing predictions")
        post_processor = PostProcessingService(spec.postprocessing)
        pred_bin = post_processor.process(pred_bin)

    # Step 6: Save MNI prediction
    _log(logger, f"[{subject_id}] Saved MNI prediction: {paths['pred_mni'].name}")
    pred_mni_img = nib.Nifti1Image(pred_bin.astype(np.float32), dwi_mni_img.affine, dwi_mni_img.header)
    pred_mni_img.set_data_dtype(np.float32)
    nib.save(pred_mni_img, str(paths['pred_mni']))

    # Step 7: Transform to native space
    _log(logger, f"[{subject_id}] Transforming to native space")
    pred_mni_ants = ants.from_numpy(
        pred_bin.astype(np.float32),
        origin=dwi_mni_img.affine[:3, 3],
        spacing=np.abs(np.diag(dwi_mni_img.affine[:3, :3])),
        direction=np.sign(np.diag(dwi_mni_img.affine[:3, :3]))
    )

    if inputs.fwd_affine and inputs.fwd_affine.exists():
        # Load native space reference
        dwi_native_img = _load_nib_ras(str(inputs.dwi_native))
        dwi_native_ants = ants.image_read(str(inputs.dwi_native))

        # Transform back to native space (inverse of affine)
        pred_native_ants = ants.apply_transforms(
            fixed=dwi_native_ants,
            moving=pred_mni_ants,
            transformlist=[str(inputs.fwd_affine)],
            interpolator='nearestNeighbor',
            whichtoinvert=[True]
        )

        # Save native prediction
        pred_native_np = pred_native_ants.numpy()
        pred_native_img = nib.Nifti1Image(
            pred_native_np.astype(np.float32),
            dwi_native_img.affine,
            dwi_native_img.header
        )
        pred_native_img.set_data_dtype(np.float32)
        nib.save(pred_native_img, str(paths['pred_native']))
        _log(logger, f"[{subject_id}] Saved native prediction: {paths['pred_native'].name}")
    else:
        _log(logger, f"[{subject_id}] Warning: No forward affine, skipping native transform")
        paths['pred_native'] = None

    # Step 8: Compute metrics if ground truth available
    metrics_dict = {}
    if spec.compute_metrics:
        _log(logger, f"[{subject_id}] Computing metrics")

        # MNI space metrics
        if inputs.hp_mni and inputs.hp_mni.exists():
            hp_mni_img = _load_nib_ras(str(inputs.hp_mni))
            hp_mni_data = hp_mni_img.get_fdata()

            mni_metrics = MetricsService.compute(
                prediction=pred_bin,
                ground_truth=hp_mni_data,
                threshold=THRESH
            )
            metrics_dict['mni'] = mni_metrics.to_dict()
        else:
            metrics_dict['mni'] = {
                'dice': None,
                'precision': None,
                'sensitivity': None,
                'sdr': None,
                'pred_volume': float(pred_bin.sum())
            }

        # Native space metrics
        if inputs.hp_native and inputs.hp_native.exists() and paths['pred_native']:
            hp_native_img = _load_nib_ras(str(inputs.hp_native))
            hp_native_data = hp_native_img.get_fdata()

            pred_native_img_load = _load_nib_ras(str(paths['pred_native']))
            pred_native_data = pred_native_img_load.get_fdata()

            native_metrics = MetricsService.compute(
                prediction=pred_native_data,
                ground_truth=hp_native_data,
                threshold=0.5
            )
            metrics_dict['orig'] = native_metrics.to_dict()
        else:
            metrics_dict['orig'] = {
                'dice': None,
                'precision': None,
                'sensitivity': None,
                'sdr': None,
                'pred_volume': None
            }

    # Save metrics
    with open(paths['metrics'], 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    _log(logger, f"[{subject_id}] Saved metrics: {paths['metrics'].name}")

    _log(logger, f"[{subject_id}] Segmentation complete")

    return SegmentationOutputs(
        pred_mni=paths['pred_mni'],
        pred_native=paths['pred_native'],
        metrics_json=paths['metrics']
    )
