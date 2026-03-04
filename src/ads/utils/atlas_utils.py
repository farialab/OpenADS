__all__ = ['get_VasLobeTemp', 'get_TemplateImages', 'get_LookupTables', 
           'get_category_features', 'vec_VascAtlas2visual', 'vec_LobeAtlas2visual', 
           'get_Vas_visual_prob_comb', 'get_Lobe_visual_prob_comb', 'get_Aspect_visual_prob_comb', 
           'get_LVS_visual_prob_comb', 'get_FV_prob', 'get_QFV']

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

        
def vec_VascAtlas2visual(vec):
    # Vascular ROIs combined and re-arranged to QFV ROIs
    visual_vec = np.zeros(16)

    visual_vec[0] = vec[0] + vec[2] # ACAL + MLSL
    visual_vec[1] = vec[1] + vec[3] # ACAR + MLSR

    visual_vec[2] = vec[6] + vec[8] + vec[10] + vec[12] + vec[14] # ['MCAFL', 'MCAPL', 'MCATL', 'MCAOL', 'MCAIL']
    visual_vec[3] = vec[7] + vec[9] + vec[11] + vec[13] + vec[15] # ['MCAFR', 'MCAPR', 'MCATR', 'MCAOR', 'MCAIR']

    visual_vec[4] = vec[16] + vec[18] # ['PCATL', 'PCAOL']
    visual_vec[5] = vec[17] + vec[19] # ['PCATR', 'PCAOR']

    visual_vec[6] = vec[26] + vec[28] # ['SCL', 'ICL']
    visual_vec[7] = vec[27] + vec[29] # ['SCR', 'ICR']

    visual_vec[8] = vec[24] # ['BL']
    visual_vec[9] = vec[25] # ['BR']

    visual_vec[10] = vec[4] # ['LLSL']
    visual_vec[11] = vec[5] # ['LLSR']

    visual_vec[12] = vec[20] + vec[22] # ['PCTPL', 'ACTPL']
    visual_vec[13] = vec[21] + vec[23] # ['PCTPR', 'ACTPR']

    visual_vec[14] = vec[30] # ['LVL']
    visual_vec[15] = vec[31] # ['LVR']
    return visual_vec

# Attention: lower and upper cases
def vec_LobeAtlas2visual(vec, Lobe_dict):
    # Lobe ROIs combined and re-arranged to QFV ROIs
    visual_vec = np.zeros(21)
    visual_vec[0] = vec[Lobe_dict['BasalGanglia_L']] # BasalGanglia_L
    visual_vec[1] = vec[Lobe_dict['BasalGanglia_R']] # BasalGanglia_R
    visual_vec[2] = vec[Lobe_dict['CSO_L']] + vec[Lobe_dict['CorRad_L']] # CSO_L + CorRad_L
    visual_vec[3] = vec[Lobe_dict['CSO_R']] + vec[Lobe_dict['CorRad_R']] # CSO_R + CorRad_R
    visual_vec[4] = vec[Lobe_dict['cerebellum_L']] # cerebellum_L
    visual_vec[5] = vec[Lobe_dict['cerebellum_R']] # cerebellum_R
    visual_vec[6] = vec[Lobe_dict['frontal_L']] # frontal_L
    visual_vec[7] = vec[Lobe_dict['frontal_R']] # frontal_R
    visual_vec[8] = vec[Lobe_dict['insula_L']] # insula_L
    visual_vec[9] = vec[Lobe_dict['insula_R']] # insula_R
    visual_vec[10] = vec[Lobe_dict['IntCapsule_L']] # IntCapsule_L
    visual_vec[11] = vec[Lobe_dict['IntCapsule_R']] # IntCapsule_R
    visual_vec[12] = vec[Lobe_dict['midbrain']] + vec[Lobe_dict['pons']] + vec[Lobe_dict['medulla']] 
    # midbrain + pons + medulla
    visual_vec[13] = vec[Lobe_dict['occipital_L']] # occipital_L
    visual_vec[14] = vec[Lobe_dict['occipital_R']] # occipital_R
    
    visual_vec[15] = vec[Lobe_dict['parietal_L']] # parietal_L
    visual_vec[16] = vec[Lobe_dict['parietal_R']] # parietal_R
    
    visual_vec[17] = vec[Lobe_dict['temporal_L']] # temporal_L
    visual_vec[18] = vec[Lobe_dict['temporal_R']] # temporal_R
    
    visual_vec[19] = vec[Lobe_dict['Thalamus_L']] # occipital_L
    visual_vec[20] = vec[Lobe_dict['Thalamus_R']] # occipital_R
    return visual_vec

def get_Vas_visual_prob_comb(vas_prob: np.ndarray) -> np.ndarray:
    return vas_prob[0::2] + vas_prob[1::2]  # L/R combine

def get_Lobe_visual_prob_comb(lobe_prob: np.ndarray) -> np.ndarray:
    out = np.zeros(11, dtype=lobe_prob.dtype)
    out[0:6] = lobe_prob[0:12:2] + lobe_prob[1:12:2]  # BG, DWM, CB, Frontal, Insula, IC
    out[6]   = lobe_prob[12] * 2                      # brainstem (already combined)
    out[7:]  = lobe_prob[13::2] + lobe_prob[14::2]    # Occ, Par, Temp, Thal
    return out

def get_Aspect_visual_prob_comb(aspect_prob: np.ndarray) -> np.ndarray:
    return aspect_prob[0:20:2] + aspect_prob[1:20:2]

def get_LVS_visual_prob_comb(LVS_prob: np.ndarray) -> np.ndarray:
    return LVS_prob[0::2] + LVS_prob[1::2]


def cal_vascular_prob(
    features: Dict[str, np.ndarray], 
    lookups: Dict[str, pd.DataFrame], 
    templates: Dict[str, np.ndarray],
    precision: int = 3
) -> np.ndarray:
    """Calculate vascular territory probability."""
    if "vascular" not in features:
        return None
        
    vas_volume = np.array(lookups["vascular"].T.reset_index()[0])
    vas_volume_visual = vec_VascAtlas2visual(vas_volume)
    vas_prob = vec_VascAtlas2visual(features["vascular"]) / (vas_volume_visual + 1e-6)
    
    # Add watershed data if available
    if "watershed" in features and templates.get("watershed") is not None:
        ws_volume = np.array([
            np.sum(np.isclose(templates["watershed"], 1)),
            np.sum(np.isclose(templates["watershed"], 2))
        ])
        ws_prob = features["watershed"] / (ws_volume + 1e-6)
        vas_prob = np.append(vas_prob[:-2], ws_prob)  # Remove LV from vascular
        
    return vas_prob.round(precision)

def cal_lobe_prob(
    features: Dict[str, np.ndarray], 
    lookups: Dict[str, pd.DataFrame],
    precision: int = 3
) -> np.ndarray:
    """Calculate lobe probability distribution."""
    if "lobe" not in features:
        return None
        
    lobe_df = lookups["lobe"].T.reset_index()
    lobe_dict = dict(zip(lobe_df['index'], range(len(lobe_df))))
    
    lobe_volume = np.array(lobe_df[0])
    lobe_volume_visual = vec_LobeAtlas2visual(lobe_volume, lobe_dict)
    lobe_prob = vec_LobeAtlas2visual(features["lobe"], lobe_dict) / (lobe_volume_visual + 1e-6)

    return lobe_prob.round(precision)
    
def cal_aspects_prob(features: Dict[str, np.ndarray], 
                          lookups: Dict[str, pd.DataFrame],
                          deci_prec: int = 3) -> np.ndarray:
    """Calculate ASPECTS probability vector"""
    if "aspects" not in features:
        return None
        
    aspects_volume = np.array(lookups["aspects"].T.reset_index()[0])
    aspects_prob = features["aspects"] / (aspects_volume + 1e-6)

    return aspects_prob.round(deci_prec)

def cal_ventricles_prob(
    features: Dict[str, np.ndarray], 
    lookups: Dict[str, pd.DataFrame],
    precision: int = 3
) -> np.ndarray:
    """Calculate ventricles probability distribution."""
    if "Ventricles" not in features:
        return None
        
    lvor_volume = np.array(lookups["lvor"].T.reset_index()[0])
    lvir_volume = np.array(lookups["lvir"].T.reset_index()[0])
    ventricles_prob = features["Ventricles"] / (np.append(lvor_volume, lvir_volume) + 1e-6)
    
    # Invert first 10 values (representing absence of ventricle enlargement)
    ventricles_prob[:10] = 1 - ventricles_prob[:10]
    
    return ventricles_prob.round(precision)

def cal_bms_prob(features: Dict[str, np.ndarray], 
                  lookups: Dict[str, pd.DataFrame],
                  deci_prec: int = 3) -> np.ndarray:
    """Calculate BMS probability vector"""
    if "bmos" not in features or "bmis" not in features:
        print("Warning: Missing BMS data")
        return None

    bmos_volume = np.array(lookups["bmos"].T.reset_index()[0])
    bmis_volume = np.array(lookups["bmis"].T.reset_index()[0])

    bmos_prob = 1 - (features["bmos"] / (bmos_volume + 1e-6))
    bmis_prob = features["bmis"] / (bmis_volume + 1e-6)
    bms_prob = np.append(bmos_prob, bmis_prob)
    
    return bms_prob.round(deci_prec)


def get_FV_prob(stroke: np.ndarray, 
               mask_raw_MNI_img: np.ndarray, 
               ADC_ss_MNI_img: np.ndarray,
               lookup_tables: Dict[str, pd.DataFrame], 
               template_images: Dict[str, np.ndarray], 
               adc_threshold: float = 1600, 
               deci_prec: int = 3) -> Dict[str, np.ndarray]:
    """
    Calculate probability vectors for different brain atlases.
    
    Args:
        stroke: Stroke image
        mask_raw_MNI_img: Raw mask in MNI space
        ADC_ss_MNI_img: ADC image in MNI space
        lookup_tables: Dictionary of lookup tables for different atlases
        template_images: Dictionary of template images for different atlases
        adc_threshold: Threshold for ADC values
        deci_prec: Decimal precision for rounding
        
    Returns:
        Dictionary of probability vectors for different atlases
    """
    # Prepare input images
    images = {
        "stroke": stroke,
        "adc_mask": (ADC_ss_MNI_img > adc_threshold).astype(float),
        "mask_raw": mask_raw_MNI_img
    }
    
    # Calculate feature vectors for all atlases
    feature_vectors = {}
    feature_vectors["vascular"] = get_category_features(images["stroke"], template_images["vascular"])
    if "watershed" in template_images and template_images["watershed"] is not None:
        feature_vectors["watershed"] = get_category_features(images["stroke"], template_images["watershed"])

    feature_vectors["lobe"] = get_category_features(images["stroke"], template_images["lobe"])
    feature_vectors["aspects"] = get_category_features(images["stroke"], template_images["aspects"])
    feature_vectors["Ventricles"] = get_category_features(images["adc_mask"], template_images["Ventricles"])
    feature_vectors["bmos"] = get_category_features(images["mask_raw"], template_images["bmos"])
    feature_vectors["bmis"] = get_category_features(images["mask_raw"], template_images["bmis"])
    
    # Calculate probability vectors
    results = {}

    results["Vas_WS_Prob"] = cal_vascular_prob(feature_vectors, lookup_tables, template_images, deci_prec)
    results["Lobe_prob"]  = cal_lobe_prob(feature_vectors, lookup_tables, deci_prec)
    results["Aspects_prob"] = cal_aspects_prob(feature_vectors, lookup_tables, deci_prec)
    results["Ventricles_prob"] = cal_ventricles_prob(feature_vectors, lookup_tables, deci_prec)
    results["BMS_prob"] = cal_bms_prob(feature_vectors, lookup_tables, deci_prec)
    
    return results

def get_QFV(FV_prob_list, predict_vol_logml, deci_prec=3):
    """Combine volume prediction with feature vectors for each atlas."""
    processors = [
        get_Vas_visual_prob_comb,
        get_Lobe_visual_prob_comb,
        get_Aspect_visual_prob_comb,
        lambda x: x,  # Identity function for Ventricles
        lambda x: x   # Identity function for BMS
    ]
    
    return [
        np.append(predict_vol_logml, processor(prob)).round(deci_prec)
        for processor, prob in zip(processors, FV_prob_list)
    ]



    
