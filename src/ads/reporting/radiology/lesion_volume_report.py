
import os
import numpy as np
import pandas as pd
from typing import List
import nibabel as nib
from dataclasses import dataclass
from pathlib import Path
from typing import Union, List, Dict, Any, Callable, Optional

def _resolve_template_path(template_dir: str, candidates):
    for name in candidates:
        if not name:
            continue
        path = os.path.join(template_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"None of the candidate templates {candidates} were found under {template_dir}"
    )

@dataclass
class AtlasConfigEntry:
    """Configuration for a brain atlas with its image and volume lookup table."""
    atlas_name: Optional[str] = None
    volume_table: Optional[str] = None

AtlasConfig = {
    # Vascular territories
    "vascular": AtlasConfigEntry("ArterialAtlas_MNI182.nii", "Vas_Lookup_Volume_df_MNI.pkl"),
    "watershed": AtlasConfigEntry(),
    # Anatomical regions
    "lobe": AtlasConfigEntry("LobeAtlas_MNI182.nii", "Lobe_Lookup_Volume_df_MNI.pkl"),
    "aspects": AtlasConfigEntry("AspectsAtlas_MNI182.nii", "Aspects_Lookup_Volume_df_MNI.pkl"),
    # Ventricle-related
    "Ventricles": AtlasConfigEntry("VentriclesEnlargedAtlas_MNI182.nii"),
    "lvor": AtlasConfigEntry(volume_table="LVOR_Lookup_Volume_df_MNI.pkl"),
    "lvir": AtlasConfigEntry(volume_table="LVIR_Lookup_Volume_df_MNI.pkl"),
    # Brain matter segmentation
    "bmos": AtlasConfigEntry("BMOSAtlas_MNI182.nii", "BMOS_Lookup_Volume_df_MNI.pkl"),
    "bmis": AtlasConfigEntry("BMISAtlas_MNI182.nii", "BMIS_Lookup_Volume_df_MNI.pkl"),
    "bpm_type1": AtlasConfigEntry("BPMTypeIV2Atlas_MNI182.nii", "JHU_MNI_SS_BPM_TypeI_V2_Lookup_Volume_df_MNI.pkl")
}

def get_VasLobeTemp(TemplateDir):
    vas_pth = _resolve_template_path(
        TemplateDir,
        (
            'ArterialAtlas_MNI182.nii',
        ),
    )
    lobe_pth = _resolve_template_path(
        TemplateDir,
        (
            'LobeAtlas_MNI182.nii',
        ),
    )

    vas_img = nib.as_closest_canonical(nib.load(vas_pth)).get_fdata().squeeze()
    lobe_img = nib.as_closest_canonical(nib.load(lobe_pth)).get_fdata().squeeze()
    
    vas_table_pth = os.path.join(TemplateDir,'ArterialAtlasLables.txt')
    with open(vas_table_pth) as f:
        vas_contents = f.readlines()
    vas_contents = vas_contents[7:] #drop file header
    
    lobe_table_pth = os.path.join(TemplateDir,'LobesLabelLookupTable.txt') 
    with open(lobe_table_pth) as f:
        lobe_contents = f.readlines()
    
    vas_L1 = [ _.split('\t')[2] for _ in vas_contents]
    vas_L2 = [ _.split('\t')[5] for _ in vas_contents]
    vas_L1_name = [ _.split('\t')[0] for _ in vas_contents]
    
    lobe_L1 = [ _.split('\t')[1].replace('\n','') for _ in lobe_contents]
#     lobe_L1
    
    def get_L2_label_idx(L):
        if L == 'ACA':
            return 1
        elif L == 'MCA':
            return 2
        elif L == 'PCA':
            return 3
        elif L == 'VB':
            return 4
        else:
            return 0

    vas_combine = np.zeros_like(vas_img)
    for idx in range(len(vas_L1)):
        vas_combine[np.isclose(vas_img,idx+1)] = get_L2_label_idx(vas_L2[idx])
        
    vas_L2 = ['ACA', 'MCA', 'PCA', 'VB']
#     print(np.max(vas_combine))
    
    return vas_img, vas_combine, lobe_img, vas_L1, vas_L1_name, vas_L2, lobe_L1


def get_LookupTables(TemplateDir, LOOKUPS=["vascular", "lobe", "aspects", "lvor", "lvir", "bmos", "bmis"]) -> Dict[str, pd.DataFrame]:
    """
    Load volume tables for the specified atlases.

    Args:
        TemplateDir: Directory containing volume table files
        LOOKUPS: List of atlas names to load volume tables for

    Returns:
        Dictionary mapping atlas names to volume tables
    """
    lookup_tables = {}
    for atlas in LOOKUPS:
        lookup_file = AtlasConfig[atlas].volume_table
        if lookup_file:
            try:
                lookup_tables[atlas] = pd.read_pickle(os.path.join(TemplateDir, lookup_file))
            except FileNotFoundError:
                print(f"Warning: Volume table {lookup_file} not found for {atlas}")
    return lookup_tables


def get_TemplateImages(atlas_path, atlases=["vascular", "watershed", "lobe", "aspects", "Ventricles", "bmos", "bmis"]) -> Dict[str, np.ndarray]:
    """
    Load template images for the specified atlases.
    
    Args:
        atlas_path: Directory containing atlas image files
        atlases: List of atlas names to load images for
        
    Returns:
        Dictionary mapping atlas names to image data arrays
    """
    template_images = {}
    for atlas in atlases:
        atlas_name = AtlasConfig[atlas].atlas_name

        if atlas_name:
            try:
                atlas_full_path = os.path.join(atlas_path, atlas_name)
                if os.path.exists(atlas_full_path):
                    template_images[atlas] = np.squeeze(nib.as_closest_canonical(nib.load(atlas_full_path)).get_fdata())
                else:
                    print(f"Atlas file not found: {atlas_full_path}")
                    template_images[atlas] = None
            except FileNotFoundError as e:
                print(f"Error loading {atlas}: {e}")
                template_images[atlas] = None

    return template_images

def get_category_features(stroke_img, template) -> np.ndarray:
    """
    Count (binary) lesion voxels per ROI label (1..max(template)).
    stroke_img: ndarray or nib.Nifti1Image (values expected 0/1; isclose to 1 is counted)
    template:   ndarray or nib.Nifti1Image (integer labels; may be float with rounding elsewhere)
    """
    # To ndarray
    if hasattr(stroke_img, 'get_fdata'):
        stroke_img = stroke_img.get_fdata()
    elif hasattr(stroke_img, 'numpy'):
        stroke_img = stroke_img.numpy()
    stroke_img = np.asarray(stroke_img)

    if hasattr(template, 'get_fdata'):
        template = template.get_fdata()
    elif hasattr(template, 'numpy'):
        template = template.numpy()
    template = np.asarray(template)

    max_val = int(np.round(template.max()))
    feats = np.zeros(max_val, dtype=int)
    # ROI labels 1..max_val
    for i in range(max_val):
        mask = (template == (i + 1))
        feats[i] = int(np.sum(np.isclose(stroke_img[mask], 1)))
    return feats

def gen_lesion_report(subj_dir: Path, subj_id: str, lesion_img: np.ndarray,
                      ICV_vol: float, Lesion_vol: float, template_dir: Path) -> Path:
    vas_img, vas_combine, lobe_img, vas_L1, vas_L1_name, vas_L2, lobe_L1 = get_VasLobeTemp(str(template_dir))

    vas_v_L1 = get_category_features(lesion_img, vas_img)
    vas_v_L2 = get_category_features(lesion_img, vas_combine)
    lobe_v_L1 = get_category_features(lesion_img, lobe_img)

    out_txt = subj_dir / f"{subj_id}_volume_brain_regions.txt"
    with open(out_txt, 'w') as f:
        f.write(f"intracranial volume \t{int(ICV_vol)}\n\n")
        f.write(f"stroke volume \t{int(Lesion_vol)}\n\n")

        f.write("vascular territory\tnumber of voxel\n")
        for idx in range(len(vas_L1)):
            f.write(f"{vas_L1[idx]}\t{int(vas_v_L1[idx])}\t{vas_L1_name[idx]}\n")
        f.write("\n")

        f.write("vascular territory 2\tnumber of voxel\n")
        for idx in range(len(vas_L2)):
            f.write(f"{vas_L2[idx]}\t{int(vas_v_L2[idx])}\n")
        f.write("\n")

        f.write("area\tnumber of voxel\n")
        for idx in range(len(lobe_L1)):
            f.write(f"{lobe_L1[idx]}\t{int(lobe_v_L1[idx])}\n")

    return out_txt


# # Example usage:
# from pathlib import Path

# PROJECT_ROOT = Path("/home/joshua/Documents/GitHub/ADS_final_all/OpenADS")
# SUBJECT_ID   = "sub-9656c2a3"

# SUBJECT_ROOT = PROJECT_ROOT / "output" / SUBJECT_ID
# REPORT_DIR   = SUBJECT_ROOT / "reporting"
# REG_DIR      = SUBJECT_ROOT / "registration"
# TEMPLATE_DIR = PROJECT_ROOT / "assets" / "atlases" / "JHU_ICBM"
# AA_MODELS_DIR= PROJECT_ROOT / "assets" / "models" / "AA_models"

# LESION_NII   = REG_DIR / f"{SUBJECT_ID}_stroke_MNIreg_affsyn.nii.gz"
# ICV_MASK_NII = REG_DIR / f"{SUBJECT_ID}_mask_MNIreg_affsyn.nii.gz"

# print("PROJECT_ROOT:", PROJECT_ROOT)
# print("SUBJECT_ROOT:", SUBJECT_ROOT)
# print("REPORT_DIR  :", REPORT_DIR)
# print("REG_DIR     :", REG_DIR)
# print("TEMPLATE_DIR:", TEMPLATE_DIR)
# print("AA_MODELS_DIR:", AA_MODELS_DIR)

# def load_nifti(path: Path) -> nib.Nifti1Image:
#     if not path.exists():
#         raise FileNotFoundError(f"Missing NIfTI: {path}")
#     return nib.as_closest_canonical(nib.load(str(path)))
    
# def voxel_volume_ml(img: nib.Nifti1Image) -> float:
#     z = img.header.get_zooms()[:3]
#     return float(z[0]*z[1]*z[2]) / 1000.0  # mm^3 -> ml

# def compute_volumes_ml(lesion_arr: np.ndarray, img: nib.Nifti1Image):
#     vv_ml = voxel_volume_ml(img)
#     lesion_bin = (lesion_arr > 0.5).astype(np.uint8)
#     total_ml = lesion_bin.sum() * vv_ml
#     x_mid = lesion_bin.shape[0] // 2  # split along X assuming RAS
#     left_ml  = lesion_bin[:x_mid, :, :].sum() * vv_ml
#     right_ml = lesion_bin[x_mid:, :, :].sum() * vv_ml
#     return total_ml, left_ml, right_ml, lesion_bin

# def estimate_icv_ml(mask_path: Path, img: nib.Nifti1Image) -> float:
#     if mask_path.exists():
#         m = nib.load(str(mask_path))
#         vv_ml = voxel_volume_ml(m)
#         mask = (np.asanyarray(m.dataobj) > 0.5).astype(np.uint8)
#         return float(mask.sum() * vv_ml)
#     return 1200.0  # fallback if no brain mask

# lesion_img = load_nifti(LESION_NII)
# lesion_arr = np.asanyarray(lesion_img.dataobj)
# lesion_ml, left_ml, right_ml, lesion_bin = compute_volumes_ml(lesion_arr, lesion_img)
# icv_ml = estimate_icv_ml(ICV_MASK_NII, lesion_img)

# out_txt = gen_lesion_report(REPORT_DIR, SUBJECT_ID, lesion_bin, icv_ml, lesion_ml, TEMPLATE_DIR)
# print("volume_brain_regions:", out_txt)