# project/utils/imaging.py
"""
Lightweight imaging utilities for DWI/ADC normalization and NIfTI I/O.

This module factors out the reusable pieces from a one-off script into a
testable, importable utilities layer. Nothing here prints to stdout; use
the returned values or exceptions in your pipeline/CLI layer.

Dependencies:
    numpy
    nibabel
    scipy
    scikit-image (only if you later add measure-based utilities)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.optimize import curve_fit
import scipy.stats

__all__ = [
    "gauss",
    "bimodal",
    "get_dwi_normalized",
    "zscore_within_mask",
    "load_nifti",
    "load_nifti_as_ras",
    "save_nifti_like",
    "find_subject_files",
    "new_nifti_like",
    "SubjectPaths",
]


# ---------- math helpers ----------

def gauss(x: np.ndarray, mu: float, sigma: float, A: float) -> np.ndarray:
    """Gaussian function A * exp(-(x-mu)^2 / (2*sigma^2))."""
    return A * np.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))


def bimodal(
    x: np.ndarray,
    mu1: float, sigma1: float, A1: float,
    mu2: float, sigma2: float, A2: float
) -> np.ndarray:
    """Sum of two Gaussian components."""
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)


# ---------- core normalization ----------

def _mode_int(values: np.ndarray) -> int:
    """
    Robust integer mode that tolerates SciPy API changes.
    """
    if values.size == 0:
        raise ValueError("Cannot compute mode of an empty array.")
    # int16 cast to control bin explosion and match original intent
    v = values.astype(np.int16, copy=False)
    # SciPy 1.9+ requires keepdims=True to preserve shape; older works too.
    m = scipy.stats.mode(v, keepdims=True)
    # m.mode can be 0-d or 1-d depending on SciPy; use np.asarray then take first.
    return int(np.asarray(m.mode).ravel()[0])


def get_dwi_normalized(
    dwi_img: np.ndarray,
    mask_img: np.ndarray
) -> np.ndarray:
    """
    Normalize DWI by estimating tissue peak via bimodal Gaussian fit
    and then standardizing: (DWI - mu1) / sigma1, where (mu1, sigma1)
    come from the primary (higher-amplitude) mode fit.

    Parameters
    ----------
    dwi_img : np.ndarray
        Raw (or skull-stripped+MNI) DWI volume.
    mask_img : np.ndarray
        Brain mask (values > 0.5 considered in-mask).

    Returns
    -------
    np.ndarray
        Normalized DWI volume (float32).

    Notes
    -----
    - Falls back to mean/std inside mask if fitting fails.
    - Guards against zero-std by returning zeros (safe default).
    """
    if dwi_img.shape != mask_img.shape:
        raise ValueError("dwi_img and mask_img must have the same shape.")

    in_mask = mask_img > 0.5
    dwi_vals = dwi_img[in_mask]
    if dwi_vals.size == 0:
        raise ValueError("Mask selects no voxels; cannot normalize.")

    p0_mu = _mode_int(dwi_vals)
    if p0_mu <= float(dwi_vals.mean()):
        p0_mu = float(dwi_vals.mean())

    vmax = max(int(dwi_vals.max()), 1)
    # Create integer-centered histogram bins: [0, 1, 2, ..., vmax]
    bins = np.arange(vmax + 1)
    hist, edges = np.histogram(dwi_vals, bins=bins, density=True)
    # Midpoints so x and y lengths match
    x = (edges[1:] + edges[:-1]) / 2.0

    mu1: float
    sigma1: float
    # Reasonable bounds: positive sigmas; allow negative amplitudes in practice,
    # but keep bounded to avoid absurd fits. We bias second mean <= p0_mu.
    try:
        bounds = (
            [0,          1e-6,  -np.inf,   0,        1e-6,  -np.inf],
            [np.inf,     np.inf, np.inf,   p0_mu,    np.inf, np.inf],
        )
        # Initial guesses: peak near p0_mu, a small secondary near ~0
        p0 = (p0_mu, 1.0, 1.0, 0.0, 1.0, 1.0)
        params, _ = curve_fit(bimodal, x, hist, bounds=bounds, p0=p0, maxfev=5000)
        mu1 = float(params[0])
        sigma1 = float(max(params[1], 1e-6))
    except Exception:
        mu1 = float(p0_mu)
        sigma1 = float(dwi_vals.std(ddof=0))
        if sigma1 <= 0:
            # Degenerate case: flat values in mask
            return np.zeros_like(dwi_img, dtype=np.float32)

    norm = (dwi_img - mu1) / sigma1
    return norm.astype(np.float32, copy=False)


def zscore_within_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Z-score an image using mean/std computed only within the mask.

    Returns zeros if std is 0 to avoid NaNs.
    """
    if img.shape != mask.shape:
        raise ValueError("img and mask must have the same shape.")
    sel = mask > 0.49
    vals = img[sel]
    if vals.size == 0:
        raise ValueError("Mask selects no voxels; cannot compute z-score.")
    m = float(vals.mean())
    s = float(vals.std(ddof=0))
    if s <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - m) / s).astype(np.float32, copy=False)


# ---------- NIfTI I/O ----------

def load_nifti(path: str | os.PathLike) -> Optional[np.ndarray]:
    """
    Load a NIfTI file and return its data array in canonical (RAS) orientation.

    Returns None if the path doesn't exist or load fails.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = nib.as_closest_canonical(nib.load(str(p)))
        data = np.squeeze(img.get_fdata())
        return data
    except Exception:
        return None


def load_nifti_as_ras(file_path: str | os.PathLike) -> nib.Nifti1Image:
    """
    Load and return a NIfTI as a nibabel image in RAS orientation.
    """
    img = nib.load(str(file_path))
    return nib.as_closest_canonical(img)

def save_nifti_like(
    data: np.ndarray,
    reference_img: nib.Nifti1Image,
    output_path: str | os.PathLike
) -> Path:
    """
    Save a numpy array as NIfTI, inheriting affine/header from a reference image.
    """
    out = new_nifti_like(data, reference_img)
    out_path = Path(output_path)
    nib.save(out, str(out_path))
    return out_path


def new_nifti_like(
    new_img: np.ndarray,
    reference_img: nib.Nifti1Image,
    dtype: np.dtype = np.float32,
) -> nib.Nifti1Image:
    """
    Build a NIfTI1 image using the reference's affine/header, with safe tweaks.

    - Forces canonical orientation on the resulting image.
    - Sets slope/intercept to identity to avoid legacy scaling surprises.
    - Updates min/max, units, and description.
    """
    if new_img.dtype != np.dtype(dtype):
        new_img = new_img.astype(dtype, copy=False)

    header = reference_img.header.copy()
    header.set_data_dtype(dtype)
    # Update metadata (not strictly required but helpful)
    with np.errstate(all="ignore"):
        header["glmax"] = float(np.nanmax(new_img))
        header["glmin"] = float(np.nanmin(new_img))
    header["xyzt_units"] = 0  # unknown
    header["descrip"] = "ADS Sept 2025"

    out = nib.Nifti1Image(new_img, reference_img.affine, header)
    out.header.set_slope_inter(1.0, 0.0)
    return nib.as_closest_canonical(out)


# ---------- subject file discovery ----------

@dataclass(frozen=True)
class SubjectPaths:
    dwi: Path
    mask: Path
    adc: Path


def find_subject_files(subject_dir: str | os.PathLike, subject_id: str) -> Optional[SubjectPaths]:
    """
    Find DWI/Mask/ADC files for a subject, supporting two naming patterns:
        - "{id}_DWI_registered.nii.gz", etc.
        - "sub-{id}_DWI_registered.nii.gz", etc.

    Returns
    -------
    SubjectPaths or None
        None if a complete triplet isn't found.
    """
    base = Path(subject_dir)
    patterns = [
        {
            "dwi": f"{subject_id}_DWI_registered.nii.gz",
            "mask": f"{subject_id}_mask_registered_binary.nii.gz",
            "adc": f"{subject_id}_ADC_registered.nii.gz",
        },
        {
            "dwi": f"sub-{subject_id}_DWI_registered.nii.gz",
            "mask": f"sub-{subject_id}_mask_registered_binary.nii.gz",
            "adc": f"sub-{subject_id}_ADC_registered.nii.gz",
        },
    ]

    for pat in patterns:
        dwi = base / pat["dwi"]
        mask = base / pat["mask"]
        adc = base / pat["adc"]
        if dwi.exists() and mask.exists() and adc.exists():
            return SubjectPaths(dwi=dwi, mask=mask, adc=adc)

    return None

# ---------- end of utils/normalize.py ----------
"""
Usage:
from pathlib import Path
import nibabel as nib
import numpy as np

from project.utils.imaging import (
    load_nifti_as_ras, get_dwi_normalized, zscore_within_mask,
    new_nifti_like, find_subject_files
)

base_dir = Path("/home/joshua/projects/ads_using/new/pwi_test/dataset/PWI_aug2024_resaved_reverified_temp")
for subject_id in [p.name for p in base_dir.iterdir() if p.is_dir()]:
    paths = find_subject_files(base_dir / subject_id, subject_id)
    if not paths:
        continue

    dwi_img  = load_nifti_as_ras(paths.dwi)
    mask_img = load_nifti_as_ras(paths.mask)
    adc_img  = load_nifti_as_ras(paths.adc)

    dwi_norm = get_dwi_normalized(dwi_img.get_fdata(), mask_img.get_fdata())
    dwi_z    = zscore_within_mask(dwi_norm,             mask_img.get_fdata())
    adc_z    = zscore_within_mask(adc_img.get_fdata(),  mask_img.get_fdata())

    nib.save(new_nifti_like(dwi_z, dwi_img, np.float32), paths.dwi.parent / f"{subject_id}_DWI_MNI_Norm.nii.gz")
    nib.save(new_nifti_like(adc_z, adc_img, np.float32), paths.adc.parent / f"{subject_id}_ADC_MNI_Norm.nii.gz")

"""