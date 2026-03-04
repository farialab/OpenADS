"""
PWI segmentation pipeline using DAGMNet (ReplicatedDAGMNet).

Refactored from pipe_inference_pwi_dagmnet.py.
Pure orchestration - delegates all work to services and adapters.

All preprocessing, normalization, and inference logic is delegated to:
- ads.core.preprocessing (PWI normalization)
- ads.models.dagmnet_pwi (ReplicatedDAGMNet model)
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
from contextlib import nullcontext
from torch.amp.autocast_mode import autocast

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

# Import existing implementations (DO NOT MODIFY)
from ads.core.preprocessing import get_pwi_normalized
from ads.adapters.nifti import SpatialTransformer


# Constants from original implementation
ORIG_SHAPE = (182, 218, 182)
PAD_SHAPE = (192, 224, 192)
POST_MIN_SIZE = 50
THRESH = 0.49


def _log(logger, msg: str):
    """Helper to log if logger available."""
    if logger:
        logger.info(msg)
    else:
        print(msg)


def _load_nifti_as_ras(path: str) -> nib.Nifti1Image:
    """Load NIfTI in RAS+ orientation."""
    return nib.as_closest_canonical(nib.load(path))


def _autocast_ctx_for(device: torch.device):
    """Get autocast context for device."""
    return autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def _predict_volume(model, x_full: torch.Tensor, device, ds: int = 2, z_stride_factor: int = 2):
    """
    Run multi-scale inference with Z-stride.

    This is the exact logic from pipe_inference_pwi_dagmnet.py:429-446
    to maintain numerical equivalence.

    Args:
        model: ReplicatedDAGMNet model
        x_full: Input tensor [C, D, H, W]
        device: Torch device
        ds: Downsampling factor
        z_stride_factor: Z-stride factor

    Returns:
        Prediction probability map [D, H, W]
    """
    C, D, H, W = x_full.shape
    model.eval()
    pred_canvas = torch.zeros((1, D, H, W), device=device)
    count_canvas = torch.zeros((1, D, H, W), device=device)

    for x_off in range(ds):
        for y_off in range(ds):
            for z_off in range(z_stride_factor * ds):
                sub_x = x_full[:, z_off::(z_stride_factor * ds), y_off::ds, x_off::ds]
                with _autocast_ctx_for(device):
                    sub_logits, *_ = model(sub_x.unsqueeze(0).to(device))
                    sub_prob = torch.sigmoid(sub_logits)
                pred_canvas[:, z_off::(z_stride_factor * ds), y_off::ds, x_off::ds] += sub_prob[0]
                count_canvas[:, z_off::(z_stride_factor * ds), y_off::ds, x_off::ds] += 1.0

    pred = torch.where(count_canvas > 0, pred_canvas / count_canvas, 0.0)
    return pred.squeeze(0).squeeze(0).detach().cpu().numpy()


def process_subject(
    subject_dir: Path,
    subject_id: str,
    inputs: PWISegmentationInputs,
    model_config: ModelConfig,
    spec: SegmentationSpec,
    logger=None,
) -> SegmentationOutputs:
    """
    Run PWI hypoperfusion segmentation pipeline with DAGMNet.

    Pure orchestration - delegates all work to existing services.

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
    _log(logger, f"[{subject_id}] Starting PWI segmentation (DAGMNet)")

    # Setup output directory
    output_dir = Path(subject_dir) / "segment"
    output_dir.mkdir(parents=True, exist_ok=True)
    path_builder = PWIPathBuilder(output_dir, subject_id)
    paths = path_builder.get_output_paths()

    # Step 1: Load MNI-space inputs
    _log(logger, f"[{subject_id}] Loading MNI-space images")
    dwi_mni_img = _load_nifti_as_ras(str(inputs.dwi_mni))
    adc_mni_img = _load_nifti_as_ras(str(inputs.adc_mni))
    ttp_mni_img = _load_nifti_as_ras(str(inputs.ttp_mni))
    mask_mni_img = _load_nifti_as_ras(str(inputs.mask_mni))

    dwi_data = dwi_mni_img.get_fdata()
    adc_data = adc_mni_img.get_fdata()
    ttp_data = ttp_mni_img.get_fdata()
    mask_data = mask_mni_img.get_fdata()
    original_shape = dwi_data.shape

    # Load optional stroke channel for 4-channel mode
    stroke_data = None
    if inputs.stroke_mni and inputs.stroke_mni.exists():
        stroke_mni_img = _load_nifti_as_ras(str(inputs.stroke_mni))
        stroke_data = stroke_mni_img.get_fdata()
        _log(logger, f"[{subject_id}] Loaded stroke channel (4-channel mode)")

    # Step 2: Normalize and prepare inputs
    _log(logger, f"[{subject_id}] Normalizing and padding images")
    x_full_np = get_pwi_normalized(
        dwi_mni=dwi_data,
        adc_mni=adc_data,
        ttp_mni=ttp_data,
        mask_mni=mask_data,
        stroke_mni=stroke_data,
        target_shape=PAD_SHAPE
    )
    x_full_t = torch.from_numpy(x_full_np.astype(np.float32))

    # Step 3: Load model and run inference
    _log(logger, f"[{subject_id}] Loading model and running inference")
    model = ModelLoader.load_model(model_config)
    device = torch.device(spec.inference.device)
    model = model.to(device)

    # Run inference with Z-stride
    prob_pad = _predict_volume(
        model,
        x_full_t,
        device,
        ds=spec.inference.downsampling_factor,
        z_stride_factor=getattr(spec.inference, 'z_stride_factor', 2)
    )

    # Step 4: Depad predictions
    _log(logger, f"[{subject_id}] Depadding predictions")
    prob_mni = SpatialTransformer.depad_to_size(prob_pad, ORIG_SHAPE)

    # Step 5: Post-process and threshold
    _log(logger, f"[{subject_id}] Post-processing predictions")
    pred_bin = (prob_mni > THRESH).astype(np.uint8)

    # Apply post-processing if enabled
    if spec.postprocessing.apply_postprocessing:
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
        dwi_native_img = _load_nifti_as_ras(str(inputs.dwi_native))
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
            hp_mni_img = _load_nifti_as_ras(str(inputs.hp_mni))
            hp_mni_data = hp_mni_img.get_fdata()
            hp_mni_depad = SpatialTransformer.depad_to_size(hp_mni_data, ORIG_SHAPE)

            mni_metrics = MetricsService.compute(
                prediction=pred_bin,
                ground_truth=hp_mni_depad,
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
            hp_native_img = _load_nifti_as_ras(str(inputs.hp_native))
            hp_native_data = hp_native_img.get_fdata()

            pred_native_img_load = _load_nifti_as_ras(str(paths['pred_native']))
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
