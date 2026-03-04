'''
Visualization module for stroke segmentation comparison.
Generates side-by-side visualizations of DWI, ADC, and stroke masks
in both MNI and original space.
'''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure
import nibabel as nib
import multiprocessing
import os
from matplotlib import gridspec
from typing import Optional, List


def load_nifti(path: Path) -> np.ndarray:
    """
    Load a NIfTI file and return a 3D numpy array.
    
    Args:
        path: Path to the NIfTI file
        
    Returns:
        3D numpy array of the image data
    """
    return np.squeeze(nib.as_closest_canonical(nib.load(str(path))).get_fdata())


def find_center_slices(stroke_img: np.ndarray, n_slices: int = 21, interval: int = 20) -> List[int]:
    """
    Find slices to visualize with two sets:
    - Set 1: Slices containing stroke (up to n_slices, centered)
    - Set 2: Regular interval slices covering whole brain (every `interval` slices)
    
    Args:
        stroke_img: 3D numpy array of stroke mask
        n_slices: Maximum number of stroke-containing slices to select
        interval: Spacing between regular interval slices
        
    Returns:
        Sorted list of unique slice indices
    """
    depth = stroke_img.shape[2]
    center = depth // 2
    
    # Set 2: Regular interval slices (e.g., 0, 20, 40, 60, ...)
    set2 = list(range(0, depth, interval))
    
    # Set 1: Find all slices containing stroke (pixel value > 0.5)
    nonzero = [i for i in range(depth) if np.any(stroke_img[:, :, i] > 0.5)]
    
    # Case 1: No stroke slices → only show Set 2
    if not nonzero:
        return sorted(set2)
    
    # Case 2: Stroke slices ≤ n_slices → show all stroke + Set 2
    if len(nonzero) <= n_slices:
        combined = set(nonzero) | set(set2)
        return sorted(combined)
    
    # Case 3: Stroke slices > n_slices → select n_slices closest to center + Set 2
    sorted_by_dist = sorted(nonzero, key=lambda i: abs(i - center))
    set1 = sorted_by_dist[:n_slices]
    combined = set(set1) | set(set2)
    return sorted(combined)


def create_visualization_compare(
    subj_dir: Path, 
    subject_id: str,
    use_prediction_for_slices: bool = True
):
    """
    Create MNI-space visualization comparing stroke label and prediction.
    
    Args:
        subj_dir: Subject directory containing registration/, segment/, reporting/
        subject_id: Subject identifier
        use_prediction_for_slices: If True, use prediction mask to determine slices;
                                   if False, use stroke label (if available)
    """
    # 1. Define paths
    reg_path = subj_dir / "registration"
    report_path = subj_dir / "reporting" 
    seg_path = subj_dir / "segment"

    if not reg_path.exists() or not report_path.exists() or not seg_path.exists():
        print(f"[Warning] Missing registration, reporting, or segmentation folder for subject {subject_id}, skipping.")
        return
    
    dwi_aff_path = reg_path / f"{subject_id}_DWI_space-MNI152_aff.nii.gz"
    adc_aff_path = reg_path / f"{subject_id}_ADC_space-MNI152_aff.nii.gz"
    stroke_aff_path = reg_path / f"{subject_id}_stroke_space-MNI152_aff.nii.gz"
    stroke_pred_path = seg_path / f"{subject_id}_stroke-mask_space-MNI152.nii.gz"
    
    # 2. Check required files
    required_paths = [dwi_aff_path, adc_aff_path, stroke_pred_path]
    for p in required_paths:
        if not p.exists():
            print(f"[Warning] Missing {p.name} for subject {subject_id}, skipping.")
            return

    # 3. Load volumes
    dwi_aff = load_nifti(dwi_aff_path)
    adc_aff = load_nifti(adc_aff_path)
    stroke_aff = load_nifti(stroke_aff_path) if stroke_aff_path.exists() else None
    stroke_pred = load_nifti(stroke_pred_path)

    has_label = stroke_aff is not None
    n_cols = 4 if has_label else 3

    # 4. Determine which mask to use for slice selection
    if use_prediction_for_slices:
        slice_reference = stroke_pred
        slice_source = "prediction"
    else:
        slice_reference = stroke_aff if has_label else stroke_pred
        slice_source = "label" if has_label else "prediction (fallback)"
    
    slices = find_center_slices(slice_reference, n_slices=21, interval=20)

    if not slices:
        print(f"[Info] No suitable slices for {subject_id}, skipping.")
        return

    print(f"[Info] Selected {len(slices)} slices for {subject_id} using {slice_source}: {slices}")

    # 5. Create figure
    n_rows = len(slices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))

    if has_label:
        col_titles = ["DWI (Aff/MNI)", "ADC (Aff/MNI)", "Stroke Label (Aff/MNI)", "Stroke Pred (CH3/MNI)"]
    else:
        col_titles = ["DWI (Aff/MNI)", "ADC (Aff/MNI)", "Stroke Pred (CH3/MNI)"]

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight='bold')

    # 6. Plot each slice
    for i, idx in enumerate(slices):
        # Column 0: DWI (Affine)
        axes[i, 0].imshow(np.rot90(dwi_aff[:, :, idx]), cmap='gray')
        axes[i, 0].axis('off')
        
        # Column 1: ADC (Affine)
        axes[i, 1].imshow(np.rot90(adc_aff[:, :, idx]), cmap='gray')
        axes[i, 1].axis('off')
        
        if has_label:
            # Column 2: Stroke Label overlay
            axes[i, 2].imshow(np.rot90(dwi_aff[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(stroke_aff[:, :, idx]), 0.5):
                axes[i, 2].plot(c[:, 1], c[:, 0], "blue", linewidth=1)
            axes[i, 2].axis("off")

            # Column 3: Prediction overlay
            axes[i, 3].imshow(np.rot90(dwi_aff[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(stroke_pred[:, :, idx]), 0.5):
                axes[i, 3].plot(c[:, 1], c[:, 0], "red", linewidth=1)
            axes[i, 3].axis("off")
        else:
            # Column 2: Prediction overlay
            axes[i, 2].imshow(np.rot90(dwi_aff[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(stroke_pred[:, :, idx]), 0.5):
                axes[i, 2].plot(c[:, 1], c[:, 0], "red", linewidth=1)
            axes[i, 2].axis("off")

    # 7. Save figure
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    report_path.mkdir(parents=True, exist_ok=True)
    output_path = report_path / f"{subject_id}_DWIstroke_space-MNI152.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ Saved MNI-space visualization: {output_path}")


def create_visualization_compare_orig(
    subj_dir: Path, 
    subject_id: str,
    use_prediction_for_slices: bool = True
):
    """
    Create original-space visualization (no MNI registration).
    
    Args:
        subj_dir: Subject directory containing preprocess/, segment/, reporting/
        subject_id: Subject identifier
        use_prediction_for_slices: If True, use prediction mask to determine slices;
                                   if False, use stroke label (if available)
    """
    pp_path = subj_dir / "preprocess"
    seg_path = subj_dir / "segment"
    report_path = subj_dir / "reporting"

    if not pp_path.exists() or not seg_path.exists():
        print(f"[Warning] Missing preprocess or segmentation folder for {subject_id}, skipping.")
        return

    # 1. Define paths
    dwi_brain_path = pp_path / f"{subject_id}_DWI_brain.nii.gz"
    adc_brain_path = pp_path / f"{subject_id}_ADC_brain.nii.gz"
    dwi_raw_path = pp_path / f"{subject_id}_DWI.nii.gz"
    adc_raw_path = pp_path / f"{subject_id}_ADC.nii.gz"

    dwi_path = dwi_brain_path if dwi_brain_path.exists() else dwi_raw_path
    adc_path = adc_brain_path if adc_brain_path.exists() else adc_raw_path
    stroke_path = pp_path / f"{subject_id}_stroke.nii.gz"
    pred_path = seg_path / f"{subject_id}_stroke-mask.nii.gz"

    required = [pred_path, dwi_path, adc_path]
    for p in required:
        if not p.exists():
            print(f"[Warning] Missing {p} for subject {subject_id}, skipping.")
            return

    print(f"[Info] DWI source: {dwi_path.name}")
    print(f"[Info] ADC source: {adc_path.name}")

    # 2. Load images
    dwi = load_nifti(dwi_path)
    adc = load_nifti(adc_path)
    stroke = load_nifti(stroke_path) if stroke_path.exists() else None
    pred = load_nifti(pred_path)

    has_label = stroke is not None
    n_cols = 4 if has_label else 3

    # 3. Shape validation
    if dwi.shape != pred.shape:
        print(f"[ERROR] Shape mismatch! DWI: {dwi.shape}, Pred: {pred.shape}")
        print(f"[INFO] Prediction may still be in MNI space. Need inverse transformation.")
        return

    # 4. Determine which mask to use for slice selection
    if use_prediction_for_slices:
        slice_reference = pred
        slice_source = "prediction"
    else:
        slice_reference = stroke if has_label else pred
        slice_source = "label" if has_label else "prediction (fallback)"
    
    slices = find_center_slices(slice_reference, n_slices=21, interval=20)

    if not slices:
        print(f"[Info] No suitable slices for {subject_id}, skipping.")
        return

    print(f"[Info] Selected {len(slices)} slices for {subject_id} (orig space) using {slice_source}: {slices}")

    # 5. Create figure
    n_rows = len(slices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    if has_label:
        col_titles = ["DWI (Orig)", "ADC (Orig)", "Stroke Label (Orig)", "Stroke Pred (CH3/Orig)"]
    else:
        col_titles = ["DWI (Orig)", "ADC (Orig)", "Stroke Pred (CH3/Orig)"]

    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    # 6. Plot each slice
    for i, idx in enumerate(slices):
        axes[i, 0].imshow(np.rot90(dwi[:, :, idx]), cmap="gray")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.rot90(adc[:, :, idx]), cmap="gray")
        axes[i, 1].axis("off")

        if has_label:
            # Column 2: Stroke Label overlay
            axes[i, 2].imshow(np.rot90(dwi[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(stroke[:, :, idx]), 0.5):
                axes[i, 2].plot(c[:, 1], c[:, 0], "blue", linewidth=1)
            axes[i, 2].axis("off")

            # Column 3: Prediction overlay
            axes[i, 3].imshow(np.rot90(dwi[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(pred[:, :, idx]), 0.5):
                axes[i, 3].plot(c[:, 1], c[:, 0], "red", linewidth=1)
            axes[i, 3].axis("off")
        else:
            # Column 2: Prediction overlay
            axes[i, 2].imshow(np.rot90(dwi[:, :, idx]), cmap="gray")
            for c in measure.find_contours(np.rot90(pred[:, :, idx]), 0.5):
                axes[i, 2].plot(c[:, 1], c[:, 0], "red", linewidth=1)
            axes[i, 2].axis("off")

    # 7. Save figure
    plt.subplots_adjust(wspace=-0.1, hspace=-0.1)
    report_path.mkdir(parents=True, exist_ok=True)
    out_png = report_path / f"{subject_id}_DWIstroke.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✓ Saved original-space visualization: {out_png}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of visualization functions.
    """
    
    # Define paths
    base_dir = Path("/path/to/your/data")
    subject_id = "sub-001"
    subj_dir = base_dir / subject_id
    
    # -------------------------------------------------------------------------
    # Example 1: MNI-space visualization with DEFAULT (use prediction for slices)
    # -------------------------------------------------------------------------
    print("\n=== Example 1: MNI-space (default: prediction-based slices) ===")
    create_visualization_compare(
        subj_dir=subj_dir,
        subject_id=subject_id
        # use_prediction_for_slices=True is the default
    )
# def worker_helper(subj_dir: Path, subject_id: str, fig_dir: Path, space: str = "MNI"):
#     """Unpack arguments for starmap so multiprocessing can pickle it."""
#     if space.lower() == "mni":
#         create_visualization_compare(subj_dir, subject_id, fig_dir)
#     elif space.lower() == "orig":
#         gen_result_png(str(subj_dir), lesion_name='stroke_pred_MNI_CH3', wspace=-0.1, hspace=-0.5)
#     elif space.lower() == "both":
#         create_visualization_compare(subj_dir, subject_id, fig_dir)
#         gen_result_png(str(subj_dir), lesion_name='stroke_pred_MNI_CH3', wspace=-0.1, hspace=-0.5)
