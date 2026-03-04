"""HP mask postprocessing service (v2, aggressive cleanup).

Applies stronger morphology + connected-component filtering to PWI HP
predictions in MNI affine space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ants
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    generate_binary_structure,
    iterate_structure,
    label,
)

def _ball_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    base = generate_binary_structure(3, 1)
    return iterate_structure(base, int(radius)).astype(bool)


def _keep_components(mask: np.ndarray, min_vox: int = 0, topk: int = 0) -> np.ndarray:
    lab, n = label(mask > 0)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    counts = np.bincount(lab.ravel())
    pairs = [(idx, int(cnt)) for idx, cnt in enumerate(counts) if idx != 0]
    pairs.sort(key=lambda x: x[1], reverse=True)

    kept = pairs
    if min_vox > 0:
        kept = [p for p in kept if p[1] >= int(min_vox)]
    if topk and topk > 0:
        kept = kept[: int(topk)]

    out = np.zeros_like(mask, dtype=np.uint8)
    for idx, _ in kept:
        out[lab == idx] = 1
    return out


def postprocess_hp_mask(
    mask_path: str | Path,
    threshold: float = 0.5,
    open_radius: int = 1,
    close_radius: int = 1,
    fill_holes: bool = True,
    min_vox: int = 100,
    topk: int = 0,
    brain_mask_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Postprocess a predicted HP mask (aggressive v2).

    Args:
        mask_path: Path to predicted mask NIfTI.
        threshold: Binarization threshold before morphology cleanup.
        open_radius: Binary opening radius (voxels).
        close_radius: Binary closing radius (voxels).
        fill_holes: Fill enclosed holes in 3D.
        min_vox: Remove components smaller than this voxel count.
        topk: Keep top-K largest components (0 means keep all).
        brain_mask_path: Optional brain mask path for final masking.
        output_path: Optional output path. If None, writes in-place.

    Returns:
        Path to the saved postprocessed mask.
    """
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"HP mask not found: {path}")

    img = ants.image_read(str(path))
    raw = img.numpy()
    binary = (raw > float(threshold)).astype(np.uint8)

    # Step 1: opening/closing to suppress speckle and smooth boundaries.
    opened = binary_opening(binary.astype(bool), structure=_ball_structure(open_radius))
    closed = binary_closing(opened, structure=_ball_structure(close_radius))

    # Step 2: fill holes.
    if fill_holes:
        closed = binary_fill_holes(closed)

    cleaned = closed.astype(np.uint8)

    # Step 3: connected-component filtering.
    cleaned = _keep_components(cleaned, min_vox=min_vox, topk=topk).astype(np.uint8)

    # Step 4: optional brain masking.
    if brain_mask_path is not None:
        bm_path = Path(brain_mask_path)
        if bm_path.exists():
            bm_img = ants.image_read(str(bm_path))
            bm = (bm_img.numpy() > 0.5).astype(np.uint8)
            if bm.shape == cleaned.shape:
                cleaned = (cleaned * bm).astype(np.uint8)

    out = ants.from_numpy(
        cleaned.astype(np.float32),
        origin=img.origin,
        spacing=img.spacing,
        direction=img.direction,
    )
    out_path = Path(output_path) if output_path is not None else path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(out, str(out_path))
    return out_path
