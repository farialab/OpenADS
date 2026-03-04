#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import nibabel as nib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skimage import measure


def _load_ras(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = nib.as_closest_canonical(nib.load(str(p)))
        return np.squeeze(img.get_fdata())
    except Exception:
        return None


def _display_range(img: np.ndarray, p1: float = 1, p2: float = 99) -> Tuple[float, float]:
    if img is None or img.size == 0:
        return 0.0, 1.0
    v = img[np.isfinite(img)]
    if v.size == 0:
        return 0.0, 1.0
    try:
        lo, hi = np.percentile(v, [p1, p2])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return float(lo), float(hi)
    except Exception:
        pass
    return float(np.min(v)), float(np.max(v))


# def _select_slices(mask: np.ndarray, max_slices: int = 20) -> List[int]:
#     if mask is None or mask.size == 0:
#         return []
#     nz = np.where(np.any(mask > 0.5, axis=(0, 1)))[0]
#     if nz.size > 0:
#         sums = [int(np.sum(mask[:, :, k] > 0.5)) for k in nz]
#         order = np.argsort(sums)[::-1]
#         sel = sorted([int(nz[i]) for i in order[:max_slices]])
#     else:
#         sel = []
#     if len(sel) < 10:
#         mid = mask.shape[2] // 2
#         a = max(0, mid - 5)
#         b = min(mask.shape[2], mid + 5)
#         for k in range(a, b):
#             if k not in sel:
#                 sel.append(int(k))
#         sel = sorted(sel)
#     return sel

def _select_slices(mask: np.ndarray, max_slices: int = 21, interval: int = 20) -> List[int]:
    """
    Selects slices to visualize using two strategies:
    1. Slices containing stroke (prioritizing the center of the mask depth).
    2. Regular interval slices covering the whole brain.
    
    Args:
        mask: 3D numpy array of the lesion mask.
        max_slices: Maximum number of stroke-containing slices to select.
        interval: Spacing between regular interval slices.
        
    Returns:
        Sorted list of unique slice indices.
    """
    if mask is None or mask.ndim != 3:
        return []

    depth = mask.shape[2]
    center = depth // 2
    
    # Set 2: Regular interval slices (e.g., 0, 20, 40...) to ensure context
    set2 = list(range(0, depth, interval))
    
    # Set 1: Find all slices containing stroke (pixel value > 0.5)
    # np.any(mask > 0.5, axis=(0, 1)) creates a boolean array of length 'depth'
    has_stroke = np.any(mask > 0.5, axis=(0, 1))
    nonzero = np.where(has_stroke)[0].tolist()
    
    # Logic Case 1: No stroke slices found → only return interval slices
    if not nonzero:
        return sorted(list(set(set2)))
    
    # Logic Case 2: Stroke slices <= max_slices → return all stroke slices + intervals
    if len(nonzero) <= max_slices:
        combined = set(nonzero) | set(set2)
        return sorted(list(combined))
    
    # Logic Case 3: Stroke slices > max_slices → select 'max_slices' closest to center
    # Sort the nonzero indices by their distance from the center of the Z-axis
    sorted_by_dist = sorted(nonzero, key=lambda i: abs(i - center))
    
    # Take the top N closest to center
    set1 = sorted_by_dist[:max_slices]
    
    # Combine with interval slices and sort
    combined = set(set1) | set(set2)
    return sorted(list(combined))

def _find_first(base: Path, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        hits = sorted(base.glob(pat))
        if hits:
            return str(hits[0])
    return None


def _resolve_mni_paths(pwi_root: str, subject_id: str) -> Dict[str, Optional[str]]:
    """
    All modalities must be in MNI space.

    Reality in your tree:
      output/sub-xxx/PWI/registration contains DWI/ADC/mask/stroke/TTP/HP/warps too.

    Rule:
      1) Search PWI/registration first
      2) Fallback to DWI/registration
      3) Prediction is searched under PWI/(segmentation|segment|registration|PWI)
    """
    root = Path(pwi_root).expanduser().resolve() / subject_id
    pwi_dir = root / "PWI"
    dwi_dir = root / "DWI"

    pwi_reg = pwi_dir / "registration"
    dwi_reg = dwi_dir / "registration"

    def find_in(reg_dirs: List[Path], patterns: List[str]) -> Optional[str]:
        for rd in reg_dirs:
            if not rd.exists():
                continue
            hit = _find_first(rd, patterns)
            if hit:
                return hit
        return None

    reg_priority = [pwi_reg, dwi_reg]

    dwi_mni = find_in(reg_priority, [
        f"{subject_id}_DWI_space-MNI152_aff_desc-norm.nii*",
        f"{subject_id}_DWI_space-MNI152_aff.nii*",
    ])

    stroke_manual_mni = find_in([dwi_reg], [
        f"{subject_id}_stroke_space-MNI152_aff.nii*",
        f"{subject_id}_stroke_space-MNI152_affsyn.nii*",
    ])

    stroke_pred_mni = find_in([dwi_dir / "segment", dwi_dir / "segmentation"], [
        f"{subject_id}_stroke-mask_space-MNI152_affsyn.nii*",
        f"{subject_id}_stroke-mask_space-MNI152_aff.nii*",
    ])

    ttp_mni = find_in([pwi_reg], [
        f"{subject_id}_TTP_space-MNI152_aff_desc-norm.nii*",
        f"{subject_id}_TTP_space-MNI152_aff.nii*",
    ])

    hp_manual_mni = find_in([pwi_reg], [
        f"{subject_id}_HP_manual_space-MNI152_aff.nii*",
    ])

    pred = None
    for sd in [pwi_dir / "segmentation", pwi_dir / "segment", pwi_reg, pwi_dir]:
        if sd.exists():
            pred = _find_first(sd, [
                f"{subject_id}_HP-mask_space-MNI152.nii*",
                f"{subject_id}_HP-mask_space-MNI152_aff.nii*",
                f"{subject_id}*_HP-mask_space-MNI152*.nii*",
            ])
            if pred:
                break

    return {
        "root": str(root),
        "pwi_dir": str(pwi_dir),
        "pwi_reg": str(pwi_reg),
        "dwi_reg": str(dwi_reg),
        "dwi_mni": dwi_mni,
        "stroke_manual_mni": stroke_manual_mni,
        "stroke_pred_mni": stroke_pred_mni,
        "ttp_mni": ttp_mni,
        "hp_manual_mni": hp_manual_mni,
        "hp_pred_mni": pred,
    }

def create_hp_visualization_compare(pwi_root: str, subject_id: str, out_dir: Optional[str] = None) -> str:
    """
    Columns (all MNI space):
      1) DWI
      2) DWI + Stroke Mask (manual; optional)
      3) DWI + Stroke(Pred) (optional)
      4) TTP
      5) TTP + HP(manual) (optional)
      6) TTP + HP(pred) (required)

    Slice selection uses HP prediction mask in MNI.
    """
    paths = _resolve_mni_paths(pwi_root, subject_id)

    if not paths["hp_pred_mni"]:
        raise FileNotFoundError(f"Missing prediction: {subject_id}_HP-mask_space-MNI152.nii.gz under {paths['pwi_dir']}")
    if not paths["dwi_mni"]:
        raise FileNotFoundError(f"Missing DWI MNI for {subject_id} under {paths['dwi_reg']}")
    if not paths["ttp_mni"]:
        raise FileNotFoundError(f"Missing TTP MNI for {subject_id} under {paths['pwi_reg']}")

    dwi = _load_ras(paths["dwi_mni"])
    stroke_manual = _load_ras(paths["stroke_manual_mni"])
    stroke_pred = _load_ras(paths["stroke_pred_mni"])
    ttp = _load_ras(paths["ttp_mni"])
    hp_manual = _load_ras(paths["hp_manual_mni"])
    hp_pred = _load_ras(paths["hp_pred_mni"])

    if dwi is None or ttp is None or hp_pred is None:
        raise RuntimeError(f"Failed to load required MNI files for {subject_id}")

    if ttp.shape != hp_pred.shape:
        raise ValueError(f"TTP and HP pred shape mismatch: TTP={ttp.shape} pred={hp_pred.shape}")
    if dwi.shape != ttp.shape:
        raise ValueError(f"DWI and TTP shape mismatch: DWI={dwi.shape} TTP={ttp.shape}")
    if stroke_manual is not None and stroke_manual.shape != dwi.shape:
        stroke_manual = None
    if stroke_pred is not None and stroke_pred.shape != dwi.shape:
        stroke_pred = None
    if hp_manual is not None and hp_manual.shape != ttp.shape:
        hp_manual = None

    slices = _select_slices(hp_pred, max_slices=20)
    if not slices:
        raise RuntimeError(f"No slices selected from HP pred for {subject_id}")

    cols = [("DWI", "dwi")]
    if stroke_manual is not None:
        cols.append(("DWI + Stroke Mask", "dwi_stroke_manual"))
    if stroke_pred is not None:
        cols.append(("DWI + Stroke(Pred)", "dwi_stroke_pred"))
    cols.append(("TTP", "ttp"))
    if hp_manual is not None:
        cols.append(("TTP + HP(Manual)", "ttp_hp_manual"))
    cols.append(("TTP + HP(Pred)", "ttp_hp_pred"))

    n_rows = len(slices)
    n_cols = len(cols)

    out_base = Path(out_dir).expanduser().resolve() if out_dir else (Path(paths["pwi_dir"]) / "reporting")
    out_base.mkdir(parents=True, exist_ok=True)
    out_png = out_base / f"{subject_id}_HP_space-MNI152.png"

    # fig = plt.figure(figsize=(3.1 * n_cols, 2.6 * n_rows))
    # gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.03, hspace=-0.1)

    fig = plt.figure(figsize=(3.1 * n_cols, 2.6 * n_rows))

    top_axes = 0.94          # where the axes grid ends (higher = closer to titles)
    header_y = top_axes + 0.01  # titles just above the axes grid

    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.02, right=0.98,
        bottom=0.02, top=top_axes,
        wspace=0.01, hspace=0.02
    )

    for c, (title, _) in enumerate(cols):
        fig.text((c + 0.6) / n_cols, header_y, title, ha="center", va="top", fontsize=12, fontweight="bold")

    for r, k in enumerate(slices):
        for c, (_, key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

            if key == "dwi":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)

            elif key == "dwi_stroke_manual":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
                ms = np.rot90(stroke_manual[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "r", linewidth=1.0)

            elif key == "dwi_stroke_pred":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
                ms = np.rot90(stroke_pred[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "lime", linewidth=1.2)

            elif key == "ttp":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)

            elif key == "ttp_hp_manual":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)
                ms = np.rot90(hp_manual[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "w", linewidth=1.5)
                        ax.plot(contour[:, 1], contour[:, 0], "k", linewidth=0.5, alpha=0.7)

            elif key == "ttp_hp_pred":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)
                ms = np.rot90(hp_pred[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "lime", linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return str(out_png)


def _resolve_native_paths(pwi_root: str, subject_id: str) -> Dict[str, Optional[str]]:
    """Resolve native-space TTP and HP prediction paths for original-space visualization."""
    root = Path(pwi_root).expanduser().resolve() / subject_id
    pwi_dir = root / "PWI"
    pwi_pp_dir = pwi_dir / "preprocess"
    pwi_seg_dir = pwi_dir / "segmentation"
    dwi_pp_dir = root / "DWI" / "preprocess"
    dwi_seg_dir = root / "DWI" / "segment"

    def pick(base: Path, patterns: List[str]) -> Optional[str]:
        if not base.exists():
            return None
        return _find_first(base, patterns)

    ttp_native = pick(pwi_pp_dir, [f"{subject_id}_TTP.nii*", f"{subject_id}_TTP_brain.nii*"])
    dwi_native = pick(dwi_pp_dir, [f"{subject_id}_DWI_brain.nii*", f"{subject_id}_DWI.nii*"])
    stroke_manual_native = pick(dwi_pp_dir, [f"{subject_id}_stroke.nii*", f"{subject_id}_stroke_brain.nii*"])
    stroke_pred_native = pick(dwi_seg_dir, [f"{subject_id}_stroke-mask.nii*"])
    hp_manual_native = pick(pwi_pp_dir, [f"{subject_id}_HP_manual.nii*", f"{subject_id}_HP_manual_brain.nii*"])
    hp_native = pick(pwi_seg_dir, [f"{subject_id}_HP-mask.nii*", f"{subject_id}_HP_mask.nii*"])

    return {
        "pwi_dir": str(pwi_dir),
        "ttp_native": ttp_native,
        "dwi_native": dwi_native,
        "stroke_manual_native": stroke_manual_native,
        "stroke_pred_native": stroke_pred_native,
        "hp_manual_native": hp_manual_native,
        "hp_native": hp_native,
    }


def create_hp_visualization_compare_orig(pwi_root: str, subject_id: str, out_dir: Optional[str] = None) -> str:
    """Create native-space HP visualization and save as `{subject_id}_HP.png`."""
    paths = _resolve_native_paths(pwi_root, subject_id)
    if not paths["ttp_native"] or not paths["hp_native"]:
        raise FileNotFoundError(f"Missing native TTP or HP prediction for {subject_id}")

    ttp = _load_ras(paths["ttp_native"])
    hp = _load_ras(paths["hp_native"])
    dwi = _load_ras(paths["dwi_native"]) if paths["dwi_native"] else None
    stroke_manual = _load_ras(paths["stroke_manual_native"]) if paths["stroke_manual_native"] else None
    stroke_pred = _load_ras(paths["stroke_pred_native"]) if paths["stroke_pred_native"] else None
    hp_manual = _load_ras(paths["hp_manual_native"]) if paths["hp_manual_native"] else None

    if ttp is None or hp is None:
        raise RuntimeError(f"Failed to load required native files for {subject_id}")
    if ttp.shape != hp.shape:
        raise ValueError(f"Native TTP and HP shape mismatch: TTP={ttp.shape} HP={hp.shape}")

    # Fallback rule: if native DWI and native TTP are not shape-compatible,
    # build a TTP-only panel (manual HP must come from native preprocess).
    use_dwi_columns = dwi is not None and dwi.shape == ttp.shape
    if not use_dwi_columns:
        stroke_manual = None
        stroke_pred = None

    if hp_manual is not None and hp_manual.shape != ttp.shape:
        hp_manual = None
    if stroke_manual is not None and stroke_manual.shape != dwi.shape:
        stroke_manual = None
    if stroke_pred is not None and stroke_pred.shape != dwi.shape:
        stroke_pred = None

    slices = _select_slices(hp, max_slices=20)
    if not slices:
        raise RuntimeError(f"No slices selected from native HP mask for {subject_id}")

    cols = []
    if use_dwi_columns:
        cols.append(("DWI", "dwi"))
        if stroke_manual is not None:
            cols.append(("DWI + Stroke Mask", "dwi_stroke_manual"))
        if stroke_pred is not None:
            cols.append(("DWI + Stroke(Pred)", "dwi_stroke_pred"))
    cols.append(("TTP", "ttp"))
    if hp_manual is not None:
        cols.append(("TTP + HP(Manual)", "ttp_hp_manual"))
    cols.append(("TTP + HP(Pred)", "ttp_hp_pred"))

    n_rows = len(slices)
    n_cols = len(cols)

    out_base = Path(out_dir).expanduser().resolve() if out_dir else (Path(paths["pwi_dir"]) / "reporting")
    out_base.mkdir(parents=True, exist_ok=True)
    out_png = out_base / f"{subject_id}_HP.png"

    fig = plt.figure(figsize=(3.1 * n_cols, 2.6 * n_rows))
    top_axes = 0.94
    header_y = top_axes + 0.01
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        left=0.02, right=0.98, bottom=0.02, top=top_axes,
        wspace=0.01, hspace=0.02
    )

    for c, (title, _) in enumerate(cols):
        fig.text((c + 0.6) / n_cols, header_y, title, ha="center", va="top", fontsize=12, fontweight="bold")

    for r, k in enumerate(slices):
        for c, (_, key) in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")

            if key == "dwi":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
            elif key == "dwi_stroke_manual":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
                ms = np.rot90(stroke_manual[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "r", linewidth=1.0)
            elif key == "dwi_stroke_pred":
                sl = np.rot90(dwi[:, :, k])
                vmin, vmax = _display_range(sl, 1, 99)
                ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax)
                ms = np.rot90(stroke_pred[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "lime", linewidth=1.2)
            elif key == "ttp":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)
            elif key == "ttp_hp_manual":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)
                ms = np.rot90(hp_manual[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "w", linewidth=1.5)
                        ax.plot(contour[:, 1], contour[:, 0], "k", linewidth=0.5, alpha=0.7)
            elif key == "ttp_hp_pred":
                sl = np.rot90(ttp[:, :, k])
                vmin, vmax = _display_range(sl, 5, 95)
                ax.imshow(sl, cmap="jet", vmin=vmin, vmax=vmax)
                ms = np.rot90(hp[:, :, k])
                for contour in measure.find_contours(ms, 0.5) or []:
                    if len(contour) > 5:
                        ax.plot(contour[:, 1], contour[:, 0], "lime", linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(str(out_png), dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return str(out_png)


if __name__ == "__main__":
    pwi_root = "OpenADS/output"
    subject_id = "sub-05a971ae"
    out = create_hp_visualization_compare(pwi_root, subject_id)
    print(f"Saved: {out}")
