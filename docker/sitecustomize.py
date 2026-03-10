"""Runtime patches for Docker images.

Ensures calculating ADC without a target path no longer triggers nibabel save on
an empty string, which raised permission errors inside the container.

These patches apply to API, CLI, and optional GUI container runs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from ads.core import data_prep as _data_prep
except Exception:  # pragma: no cover - ADS might not be importable yet
    _data_prep = None  # type: ignore[assignment]

try:
    from ads.core import config as _core_config
except Exception:  # pragma: no cover - ADS might not be importable yet
    _core_config = None  # type: ignore[assignment]


def _patch_calculate_adc() -> None:
    if _data_prep is None:
        return

    original = getattr(_data_prep, "calculate_adc", None)
    if original is None:
        return

    def calculate_adc_patched(dwi_img, b0_img, adc_path: str, bvalue: int = 1000):
        """Drop-in replacement that skips saving when no path is provided."""
        dwi_data = dwi_img.get_fdata()
        b0_data = b0_img.get_fdata()
        epsilon = 1e-10

        adc_data = (-np.log((dwi_data + epsilon) / (b0_data + epsilon)) / bvalue) * (
            (b0_data > 0) & (dwi_data > 0)
        )

        adc_nib = nib.as_closest_canonical(
            nib.Nifti1Image(adc_data.astype(np.float32), affine=dwi_img.affine)
        )

        target = Path(adc_path)
        if adc_path:
            nib.save(adc_nib, target)
            time.sleep(0.5)
        return adc_nib

    setattr(_data_prep, "calculate_adc", calculate_adc_patched)


def _replace_project_root(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, dict):
        return {k: _replace_project_root(v, old_root, new_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_project_root(v, old_root, new_root) for v in value]
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    return value


def _patch_pwi_training_module() -> None:
    """Relocate hard-coded training artifacts under /home/joshua to a writable cache."""
    import os
    import importlib

    original_makedirs = os.makedirs

    def guarded_makedirs(path, mode=0o777, exist_ok=False):
        try:
            return original_makedirs(path, mode, exist_ok)
        except PermissionError:
            if str(path).startswith("/home/joshua"):
                return None
            raise

    os.makedirs = guarded_makedirs  # type: ignore[assignment]
    try:
        try:
            module = importlib.import_module("ads.modalities.pwi.train_pwi_segmentation")
        except ModuleNotFoundError:
            # Legacy module tree may not exist in current builds.
            return
    finally:
        os.makedirs = original_makedirs  # type: ignore[assignment]

    cache_root = Path(os.environ.get("ADS_CACHE_DIR", "/tmp/ads_cache")).resolve()
    results_dir = cache_root / "pwi_results"
    save_dir = results_dir / "checkpoints_pwi_adaptive"
    log_dir = results_dir / "runs" / "pwi_unet_adaptive"
    for d in (save_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    module.RESULTS_DIR = results_dir
    module.SAVE_DIR = str(save_dir)
    module.LOG_DIR = str(log_dir)


def _patch_load_config() -> None:
    if _core_config is None:
        return

    original = getattr(_core_config, "load_config", None)
    if original is None:
        return

    new_root = str(PROJECT_ROOT)

    def load_config_patched(config_path):  # type: ignore[override]
        cfg = original(config_path)
        if isinstance(cfg, dict):
            cfg = _replace_project_root(cfg, "${PROJECT_ROOT}", new_root)
            cfg.setdefault("project_dir", new_root)
        return cfg

    setattr(_core_config, "load_config", load_config_patched)


_patch_calculate_adc()
_patch_load_config()
_patch_pwi_training_module()
