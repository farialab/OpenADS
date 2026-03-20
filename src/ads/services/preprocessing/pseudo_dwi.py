"""Pseudo-DWI synthesis from ADC using S(b)=S0*exp(-b*ADC)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np

AdcScale = Literal["auto", "1", "1e-3", "1e-6"]


class PseudoDWIGenerator:
    """Generate pseudo-DWI from ADC using a diffusion signal model."""

    def __init__(
        self,
        *,
        b_value: float = 1000.0,
        adc_scale: AdcScale = "auto",
        brain_threshold: float = 1e-6,
    ) -> None:
        self.b_value = float(b_value)
        self.adc_scale = adc_scale
        self.brain_threshold = float(brain_threshold)

    @staticmethod
    def default_output_path(adc_path: Path) -> Path:
        """Return default pseudo-DWI output path based on ADC filename."""
        name = adc_path.name
        if "ADC" in name:
            return adc_path.with_name(name.replace("ADC", "DWI", 1))
        if name.endswith(".nii.gz"):
            return adc_path.with_name(name[:-7] + "_pseudoDWI.nii.gz")
        if name.endswith(".nii"):
            return adc_path.with_name(name[:-4] + "_pseudoDWI.nii")
        return adc_path.with_name(name + "_pseudoDWI.nii.gz")

    @staticmethod
    def infer_scale(adc_data: np.ndarray, brain_mask: np.ndarray) -> float:
        """Infer ADC unit scale to mm^2/s from robust percentiles."""
        vals = adc_data[np.isfinite(adc_data) & brain_mask & (adc_data > 0)]
        if vals.size == 0:
            return 1.0
        p99 = float(np.percentile(vals, 99))
        if p99 > 20.0:
            return 1e-6
        if p99 > 0.02:
            return 1e-3
        return 1.0

    @staticmethod
    def _resolve_mask(adc: np.ndarray, brain_mask: np.ndarray | None, threshold: float) -> np.ndarray:
        valid_adc = np.isfinite(adc) & (adc > float(threshold))
        if brain_mask is None:
            mask = valid_adc
        else:
            mask = np.asarray(brain_mask) > 0.5
            if mask.shape != adc.shape:
                if mask.shape == adc.shape[:-1] and adc.shape[-1] == 1:
                    mask = mask[..., None]
                else:
                    raise ValueError(f"Mask shape mismatch: mask={mask.shape}, adc={adc.shape}")
            mask = mask & valid_adc
        if int(mask.sum()) == 0:
            raise RuntimeError("Brain mask is empty; cannot synthesize pseudo DWI.")
        return mask

    def synthesize(
        self,
        adc_data: np.ndarray,
        *,
        brain_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Synthesize pseudo-DWI from ADC data.

        Returns:
            tuple: (pseudo_dwi, adc_scale_used, brain_mask_used)
        """
        adc = np.asarray(adc_data, dtype=np.float32)
        mask = self._resolve_mask(adc, brain_mask, self.brain_threshold)
        scale = self.infer_scale(adc, mask) if self.adc_scale == "auto" else float(self.adc_scale)

        adc_mm2s = np.nan_to_num(adc, nan=0.0, posinf=0.0, neginf=0.0) * scale
        adc_mm2s = np.clip(adc_mm2s, 0.0, None)

        dwi = np.zeros_like(adc_mm2s, dtype=np.float32)
        dwi[mask] = np.exp(-self.b_value * adc_mm2s[mask]).astype(np.float32)
        return dwi, float(scale), mask

    @staticmethod
    def save_like(
        data: np.ndarray,
        ref_img: nib.Nifti1Image,
        out_path: Path,
        *,
        dtype=np.float32,
    ) -> None:
        """Save data with reference image affine/header."""
        hdr = ref_img.header.copy()
        hdr.set_data_dtype(dtype)
        hdr.set_intent("none", (), name="")
        out = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref_img.affine, hdr)
        out.update_header()
        out.header.set_intent("none", (), name="")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out, str(out_path))

    def generate_from_files(
        self,
        *,
        adc_path: Path,
        output_path: Path | None = None,
        mask_path: Path | None = None,
    ) -> tuple[Path, float]:
        """Generate pseudo-DWI from ADC file and save it."""
        adc_img = nib.load(str(adc_path))
        adc = np.asarray(adc_img.get_fdata(), dtype=np.float32)
        mask_arr = (
            np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32)
            if mask_path is not None
            else None
        )

        pseudo_dwi, scale, _ = self.synthesize(adc, brain_mask=mask_arr)
        out = output_path or self.default_output_path(adc_path)
        self.save_like(pseudo_dwi, adc_img, out, dtype=np.float32)
        return out, scale


def default_pseudo_dwi_path(adc_path: Path) -> Path:
    return PseudoDWIGenerator.default_output_path(adc_path)


def infer_adc_scale_factor(adc_data: np.ndarray, brain_mask: np.ndarray) -> float:
    return PseudoDWIGenerator.infer_scale(adc_data, brain_mask)


def synthesize_pseudo_dwi(
    adc_data: np.ndarray,
    *,
    brain_mask: np.ndarray | None = None,
    b_value: float = 1000.0,
    adc_scale: AdcScale = "auto",
    brain_threshold: float = 1e-6,
) -> tuple[np.ndarray, float, np.ndarray]:
    gen = PseudoDWIGenerator(
        b_value=b_value,
        adc_scale=adc_scale,
        brain_threshold=brain_threshold,
    )
    return gen.synthesize(adc_data, brain_mask=brain_mask)


def save_like(
    data: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: Path,
    *,
    dtype=np.float32,
) -> None:
    PseudoDWIGenerator.save_like(data, ref_img, out_path, dtype=dtype)


def generate_and_save_pseudo_dwi(
    adc_path: Path,
    *,
    output_path: Path | None = None,
    mask_path: Path | None = None,
    b_value: float = 1000.0,
    adc_scale: AdcScale = "auto",
    brain_threshold: float = 1e-6,
) -> tuple[Path, float]:
    gen = PseudoDWIGenerator(
        b_value=b_value,
        adc_scale=adc_scale,
        brain_threshold=brain_threshold,
    )
    return gen.generate_from_files(
        adc_path=adc_path,
        output_path=output_path,
        mask_path=mask_path,
    )
