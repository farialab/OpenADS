"""Utility helpers for the ADS PWI inference workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch

from ads.models.unet3d_pwi import UNet3D, get_device, resize3d, zscore_


@dataclass
class SubjectModalities:
    """Container for resolved modality paths used during inference."""

    subject_id: str
    base_dir: Path
    dwi: Path
    adc: Path
    ttp: Path
    stroke: Path
    mask: Path
    target: Optional[Path] = None


def _canonical_subject_id(subject_id: str) -> str:
    sid = subject_id.strip()
    if not sid:
        raise ValueError("Subject identifier must be non-empty")
    if not sid.startswith("sub-"):
        sid = f"sub-{sid}"
    return sid


def _candidate_filenames(subject_id: str, base_name: str) -> Sequence[str]:
    sid_core = subject_id.replace("sub-", "")
    variants = [subject_id, sid_core, f"sub-{sid_core}"]
    out = []
    for sid in variants:
        if sid:
            out.append(f"{sid}_{base_name}")
    out.append(base_name)
    # Preserve order while dropping duplicates
    seen = set()
    result = []
    for name in out:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _find_existing(base_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    return None


def resolve_modalities(
    subject_dir: Path,
    subject_id: str,
    modality_map: Dict[str, str],
    target_label: Optional[str] = None,
) -> SubjectModalities:
    subject_dir = Path(subject_dir)
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    sid = _canonical_subject_id(subject_id if subject_id else subject_dir.name)

    resolved: Dict[str, Path] = {}
    for key, base_name in modality_map.items():
        candidates = _candidate_filenames(sid, base_name)
        path = _find_existing(subject_dir, candidates)
        if path is None:
            raise FileNotFoundError(
                f"Missing required modality '{key}' for subject {sid}. Looked for: {candidates}"
            )
        resolved[key] = path

    target_path = None
    if target_label:
        candidates = _candidate_filenames(sid, target_label)
        target_path = _find_existing(subject_dir, candidates)

    return SubjectModalities(
        subject_id=sid,
        base_dir=subject_dir,
        dwi=resolved["dwi"],
        adc=resolved["adc"],
        ttp=resolved["ttp"],
        stroke=resolved["stroke"],
        mask=resolved["mask"],
        target=target_path,
    )


def _clip_percentiles(arr: np.ndarray, percentiles: Tuple[float, float]) -> np.ndarray:
    if percentiles is None:
        return arr
    low, high = np.percentile(arr, percentiles)
    if np.isfinite(low) and np.isfinite(high) and high > low:
        return np.clip(arr, low, high)
    return arr


def load_input_tensor(
    modalities: SubjectModalities,
    input_size: Tuple[int, int, int],
    normalize_ttp: bool = True,
    clip_percentile: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, nib.Nifti1Image, nib.Nifti1Image]:
    """Load subject modalities into a model-ready tensor.

    Returns a tuple of (inputs, brain_mask, reference_image).
    """

    vols = []

    dwi_img = nib.as_closest_canonical(nib.load(str(modalities.dwi)))
    adc_img = nib.as_closest_canonical(nib.load(str(modalities.adc)))
    ttp_img = nib.as_closest_canonical(nib.load(str(modalities.ttp)))
    stroke_img = nib.as_closest_canonical(nib.load(str(modalities.stroke)))
    mask_img = nib.as_closest_canonical(nib.load(str(modalities.mask)))
    dwi_np = _clip_percentiles(dwi_img.get_fdata(), clip_percentile)
    adc_np = _clip_percentiles(adc_img.get_fdata(), clip_percentile)
    ttp_np = ttp_img.get_fdata()
    stroke_np = stroke_img.get_fdata()
    mask_np = (mask_img.get_fdata() > 0.5).astype(np.float32)

    dwi_tensor = resize3d(dwi_np, input_size, mode="trilinear")
    adc_tensor = resize3d(adc_np, input_size, mode="trilinear")
    ttp_tensor = resize3d(ttp_np, input_size, mode="trilinear")
    if normalize_ttp:
        ttp_tensor = zscore_(ttp_tensor)
    stroke_tensor = resize3d(stroke_np, input_size, mode="nearest")
    stroke_tensor = (stroke_tensor > 0.5).float()

    vols.extend([dwi_tensor, adc_tensor, ttp_tensor, stroke_tensor])

    inputs = torch.stack(vols, dim=0).float()

    mask_tensor = resize3d(mask_np, input_size, mode="nearest")
    mask_tensor = (mask_tensor > 0.5).float()

    return inputs, mask_tensor, ttp_img, mask_img


def load_trained_model(
    checkpoint_path: Path,
    in_channels: int,
    out_channels: int,
    init_features: int,
    device: Optional[torch.device] = None,
) -> UNet3D:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = device or get_device()
    model = UNet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_inference(
    model: UNet3D,
    inputs: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    with torch.no_grad():
        logits = model(inputs.unsqueeze(0).to(device))  # [1, C, D, H, W]
        probs = torch.sigmoid(logits)[0, 0].cpu()
    return probs


def upsample_to_reference(
    probs: torch.Tensor,
    reference_img: nib.Nifti1Image,
    mask_img: Optional[nib.Nifti1Image] = None,
) -> np.ndarray:
    ref_shape = reference_img.shape
    up = torch.nn.functional.interpolate(
        probs[None, None],
        size=ref_shape,
        mode="nearest",
    )[0, 0]
    pred = up.numpy().astype(np.float32)
    if mask_img is not None:
        mask_np = (mask_img.get_fdata() > 0.5).astype(np.float32)
        pred *= mask_np
    return pred


def save_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data.astype(np.float32), reference_img.affine, reference_img.header)
    out.set_data_dtype(np.float32)
    nib.save(out, str(output_path))
    return output_path
