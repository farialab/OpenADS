import json
import pandas as pd 
import os
from typing import Tuple, Union, List, Dict, Any, Callable, Optional
from pathlib import Path
import nibabel as nib
import numpy as np
from dataclasses import dataclass
import sys
import logging

# Setup logger
logger = logging.getLogger("ADS.QFV")

from ads.reporting.features.qfv_builder import (
    get_category_features,
    load_nifti_as_ras,
    vec_VascAtlas2visual,
    vec_LobeAtlas2visual,
    vec_BPMAtlas2visual,
    get_Vas_visual_prob_comb,
    get_Lobe_visual_prob_comb,
    get_Aspect_visual_prob_comb,
    get_Aspectpc_visual_prob_comb,
    get_BPM_visual_prob_comb
)

@dataclass
class AtlasConfigEntry:
    """Configuration for a brain atlas with its image and volume lookup table."""
    atlas_name: Optional[str] = None
    volume_table: Optional[str] = None

# Atlas definitions
AtlasConfig = {
    "vascular": AtlasConfigEntry("ArterialAtlas_MNI182.nii.gz", "Vas_Lookup_Volume_df_MNI.pkl"),
    "watershed": AtlasConfigEntry(),
    "lobe": AtlasConfigEntry("LobeAtlas_MNI182.nii.gz", "Lobe_Lookup_Volume_df_MNI.pkl"),
    "aspects": AtlasConfigEntry("AspectsAtlas_MNI182.nii.gz", "Aspects_Lookup_Volume_df_MNI.pkl"),
    "aspectpc": AtlasConfigEntry("AspectsPcaAtlas_MNI182.nii.gz", "Aspects_PCA_Lookup_Volume_df_MNI.pkl"),
    "Ventricles": AtlasConfigEntry("VentriclesEnlargedAtlas_MNI182.nii.gz"),
    "lvor": AtlasConfigEntry(volume_table="LVOR_Lookup_Volume_df_MNI.pkl"),
    "lvir": AtlasConfigEntry(volume_table="LVIR_Lookup_Volume_df_MNI.pkl"),
    "bmos": AtlasConfigEntry("BMOSAtlas_MNI182.nii.gz", "BMOS_Lookup_Volume_df_MNI.pkl"),
    "bmis": AtlasConfigEntry("BMISAtlas_MNI182.nii.gz", "BMIS_Lookup_Volume_df_MNI.pkl"),
    "bpm_type1": AtlasConfigEntry("BPMTypeIV2Atlas_MNI182.nii.gz", "JHU_MNI_SS_BPM_TypeI_V2_Lookup_Volume_df_MNI.pkl")
}

def get_LookupTables(TemplateDir, LOOKUPS=["vascular", "lobe", "aspects", "lvor", "lvir", "bmos", "bmis", "aspectpc","bpm_type1"]) -> Dict[str, pd.DataFrame]:
    lookup_tables = {}
    for atlas in LOOKUPS:
        lookup_file = AtlasConfig[atlas].volume_table
        if lookup_file:
            try:
                lookup_tables[atlas] = pd.read_pickle(os.path.join(TemplateDir, lookup_file))
            except FileNotFoundError:
                logger.warning(f"Volume table {lookup_file} not found for {atlas}")
    return lookup_tables

def get_TemplateImages(atlas_path, atlases=["vascular", "watershed", "lobe", "aspects", "Ventricles", "bmos", "bmis", "aspectpc","bpm_type1"]) -> Dict[str, np.ndarray]:
    template_images = {}
    for atlas in atlases:
        atlas_name = AtlasConfig[atlas].atlas_name
        if atlas_name:
            try:
                atlas_full_path = os.path.join(atlas_path, atlas_name)
                if os.path.exists(atlas_full_path):
                    template_images[atlas] = np.squeeze(load_nifti_as_ras(atlas_full_path).get_fdata())
                else:
                    template_images[atlas] = None
            except FileNotFoundError:
                template_images[atlas] = None
    return template_images

# --- Probability Calculations ---
def cal_vascular_prob(features, lookups, templates, precision=3):
    if "vascular" not in features: return None
    vas_volume = np.array(lookups["vascular"].T.reset_index()[0])
    vas_volume_visual = vec_VascAtlas2visual(vas_volume)
    vas_prob = vec_VascAtlas2visual(features["vascular"]) / (vas_volume_visual + 1e-6)
    if "watershed" in features and templates.get("watershed") is not None:
        ws_volume = np.array([
            np.sum(np.isclose(templates["watershed"], 1)),
            np.sum(np.isclose(templates["watershed"], 2))
        ])
        ws_prob = features["watershed"] / (ws_volume + 1e-6)
        vas_prob = np.append(vas_prob[:-2], ws_prob)
    return vas_prob.round(precision)

def cal_lobe_prob(features, lookups, precision=3):
    if "lobe" not in features: return None
    lobe_df = lookups["lobe"].T.reset_index()
    lobe_dict = dict(zip(lobe_df['index'], range(len(lobe_df))))
    lobe_volume = np.array(lobe_df[0])
    lobe_volume_visual = vec_LobeAtlas2visual(lobe_volume, lobe_dict)
    lobe_prob = vec_LobeAtlas2visual(features["lobe"], lobe_dict) / (lobe_volume_visual + 1e-6)
    return lobe_prob.round(precision)

def cal_aspects_prob(features, lookups, deci_prec=3):
    if "aspects" not in features: return None
    aspects_volume = np.array(lookups["aspects"].T.reset_index()[0])
    aspects_prob = features["aspects"] / (aspects_volume + 1e-6)
    return aspects_prob.round(deci_prec)

def cal_aspectpc_prob(features, lookups, deci_prec=3):
    if "aspectpc" not in features or features["aspectpc"] is None:
        return None
    aspectpc_volume = np.array(lookups["aspectpc"].T.reset_index()[0])
    aspectpc_prob = features["aspectpc"] / (aspectpc_volume + 1e-6)
    return aspectpc_prob.round(deci_prec)

def cal_ventricles_prob(features, lookups, precision=3):
    if "Ventricles_LVO" not in features or "Ventricles_LVI" not in features: return None
    lvor_volume = np.array(lookups["lvor"].T.reset_index()[0])
    lvir_volume = np.array(lookups["lvir"].T.reset_index()[0])
    ventricles_prob_LVO = features["Ventricles_LVO"][:10] / (lvor_volume + 1e-6)
    ventricles_prob_LVI = features["Ventricles_LVI"][10:20] / (lvir_volume + 1e-6)
    ventricles_prob = np.append(ventricles_prob_LVO, ventricles_prob_LVI)
    ventricles_prob[:10] = 1 - ventricles_prob[:10]
    return ventricles_prob.round(precision)

def cal_bpm_prob(features, lookups, precision=3):
    """Calculate BPM probabilities with robust error handling for missing keys."""
    if "bpm_type1" not in features or features["bpm_type1"] is None: 
        return None
    
    try:
        bpm_df = lookups["bpm_type1"].T.reset_index()
        # Clean keys (strip whitespace, ensure string)
        keys = bpm_df['index'].astype(str).str.strip().values
        BPMtype1_dict = dict(zip(keys, range(len(bpm_df))))
        
        # Check if critical keys needed for vec_BPMAtlas2visual exist
        if 'SFG_L' not in BPMtype1_dict:
            logger.warning(f"BPM Atlas lookup table missing standard keys (e.g., 'SFG_L'). Available sample: {list(BPMtype1_dict.keys())[:5]}. Skipping BPM.")
            return None

        bpm_volume = np.array(bpm_df[0])
        bpm_prob_raw = features["bpm_type1"] / (bpm_volume + 1e-6)
        bpm_visual_prob = vec_BPMAtlas2visual(bpm_prob_raw, BPMtype1_dict)
        return bpm_visual_prob.round(precision)
    
    except KeyError as e:
        logger.error(f"KeyError in BPM calculation: {e}. Skipping BPM QFV.")
        return None
    except Exception as e:
        logger.error(f"Error in BPM calculation: {e}. Skipping BPM QFV.")
        return None

def cal_bms_prob(features, lookups, deci_prec=3):
    if "bmos" not in features or "bmis" not in features: return None
    bmos_volume = np.array(lookups["bmos"].T.reset_index()[0])
    bmis_volume = np.array(lookups["bmis"].T.reset_index()[0])
    bmos_prob = 1 - (features["bmos"] / (bmos_volume + 1e-6))
    bmis_prob = features["bmis"] / (bmis_volume + 1e-6)
    bms_prob = np.append(bmos_prob, bmis_prob)
    return bms_prob.round(deci_prec)

def get_FV_prob(stroke, mask_raw_MNI_img, ADC_ss_MNI_img, lookup_tables, template_images, adc_threshold=0.5490, deci_prec=3):
    if ADC_ss_MNI_img is not None:
        mask = mask_raw_MNI_img if mask_raw_MNI_img is not None else (ADC_ss_MNI_img > 0)
        mask_bool = (mask > 0.5)
        masked_adc = ADC_ss_MNI_img[mask_bool]
        if masked_adc.size > 0:
            adc_norm = (ADC_ss_MNI_img - np.mean(masked_adc)) / (np.std(masked_adc)+ 1e-8)
        else:
            adc_norm = None
    else:
        adc_norm = None

    images = {
        "stroke": stroke,
        "adc_mask": ((adc_norm > adc_threshold) & mask_bool).astype(float) if adc_norm is not None else None,
        "adc_mask_LVO": ((adc_norm < adc_threshold) & mask_bool).astype(float) if adc_norm is not None else None,
        "adc_mask_LVI": ((adc_norm > adc_threshold) & mask_bool).astype(float) if adc_norm is not None else None,
        "mask_raw": mask_raw_MNI_img
    }
    
    feature_vectors = {}
    feature_vectors["vascular"] = get_category_features(images["stroke"], template_images["vascular"])
    if "watershed" in template_images and template_images["watershed"] is not None:
        feature_vectors["watershed"] = get_category_features(images["stroke"], template_images["watershed"])

    feature_vectors["lobe"] = get_category_features(images["stroke"], template_images["lobe"])
    feature_vectors["aspects"] = get_category_features(images["stroke"], template_images["aspects"])
    
    if "Ventricles" in template_images:
        feature_vectors["Ventricles_LVO"] = get_category_features(images["adc_mask_LVO"], template_images["Ventricles"])
        feature_vectors["Ventricles_LVI"] = get_category_features(images["adc_mask_LVI"], template_images["Ventricles"])
    feature_vectors["bmos"] = get_category_features(images["stroke"], template_images["bmos"])
    feature_vectors["bmis"] = get_category_features(images["stroke"], template_images["bmis"])
    feature_vectors["bpm_type1"] = get_category_features(images["stroke"], template_images["bpm_type1"])

    if "aspectpc" in template_images and template_images["aspectpc"] is not None:
        feature_vectors["aspectpc"] = get_category_features(images["stroke"], template_images["aspectpc"])
    else:
        feature_vectors["aspectpc"] = None

    results = {}
    results["Vas_WS_Prob"] = cal_vascular_prob(feature_vectors, lookup_tables, template_images, deci_prec)
    results["Lobe_prob"]  = cal_lobe_prob(feature_vectors, lookup_tables, deci_prec)
    results["Aspects_prob"] = cal_aspects_prob(feature_vectors, lookup_tables, deci_prec)
    results["AspectsPC_prob"] = cal_aspectpc_prob(feature_vectors, lookup_tables, deci_prec)
    results["Ventricles_prob"] = cal_ventricles_prob(feature_vectors, lookup_tables, deci_prec)
    results["BMS_prob"] = cal_bms_prob(feature_vectors, lookup_tables, deci_prec)
    results["BPM_prob"] = cal_bpm_prob(feature_vectors, lookup_tables, deci_prec)
    
    return feature_vectors, results

def get_QFV(FV_prob_list, predict_vol_logml, deci_prec=3):
    """Combine volume prediction with feature vectors. Handles None values in input list."""
    
    # FIX: Reordered to match input list: [Vascular, Lobe, Aspects, AspectsPC, Ventricles, BMS, BPM]
    processors = [
        get_Vas_visual_prob_comb,      # 0: Vas
        get_Lobe_visual_prob_comb,     # 1: Lobe
        get_Aspect_visual_prob_comb,   # 2: Aspect
        get_Aspectpc_visual_prob_comb, # new 3: AspectPC
        lambda x: x,                   # 4: Ventricles (Identity, matches index 4 in input)
        lambda x: x,                   # 5: BMS (Identity, matches index 5 in input)
        get_BPM_visual_prob_comb       # 6: BPM (matches index 6 in input)
    ]
    
    qfv_results = []
    # Ensure prob list and processors have matching length
    for i, processor in enumerate(processors):
        if i < len(FV_prob_list) and FV_prob_list[i] is not None:
            try:
                combined = processor(FV_prob_list[i])
                qfv_results.append(np.append(predict_vol_logml, combined).round(deci_prec))
            except Exception as e:
                logger.error(f"Error processing QFV index {i}: {e}. Skipping.")
                qfv_results.append(None)
        else:
            qfv_results.append(None)
            
    return qfv_results

class StrokeQFVCalculator:
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        self.lookup_tables = get_LookupTables(template_dir)
        self.template_images = get_TemplateImages(template_dir)
        
    def calculate(self, stroke_img_path, mask_raw_mni_path, adc_mni_path, adc_threshold=1600, precision=7):
        stroke_img = load_nifti_as_ras(stroke_img_path).get_fdata()
        mask_raw_mni = load_nifti_as_ras(mask_raw_mni_path).get_fdata()
        adc_mni = load_nifti_as_ras(adc_mni_path).get_fdata()
        
        stroke_logml = np.log10((np.sum(stroke_img)+1)/1000)

        FV, prob_vectors = get_FV_prob(
            stroke=stroke_img,
            mask_raw_MNI_img=mask_raw_mni,
            ADC_ss_MNI_img=adc_mni,
            lookup_tables=self.lookup_tables,
            template_images=self.template_images,
            adc_threshold=adc_threshold,
            deci_prec=precision
        )
        
        # Order MUST match the processors list in get_QFV
        prob_list = [
            prob_vectors.get("Vas_WS_Prob"),    # 0
            prob_vectors.get("Lobe_prob"),      # 1
            prob_vectors.get("Aspects_prob"),   # 2
            prob_vectors.get("AspectsPC_prob"),# new 3
            prob_vectors.get("Ventricles_prob"),# 4
            prob_vectors.get("BMS_prob"),       # 5
            prob_vectors.get("BPM_prob")        # 6
        ]
        
        qfv_features = get_QFV(prob_list, stroke_logml, precision)
        
        return {
            "FV": FV,
            "qfv_features": qfv_features,
            "probability_vectors": prob_vectors,
            "stroke_logml": stroke_logml
        }

    def save_QFV_to_csv(
        self,
        results: Dict[str, Any],
        subject_id: str,
        output_dir: str,
        qfv_suffix: str = "QFV",
        lesionload_suffix: str = "lesionload",
        qfv_stem_map: Dict[str, str] | None = None,
        lesionload_stem_map: Dict[str, str] | None = None,
    ):
        """
        Save QFV results and lesionload to CSV files.
        
        Args:
            results: Dictionary containing QFV features, FV, and probability vectors
            subject_id: Subject identifier for the output files
            output_dir: Directory to save the CSV files
        """
        qfv_types = ["Vascular", "Lobe", "Aspects", "AspectsPC", "Ventricles", "BMS", "BPM"]
        
        # Load column names from names.json
        try:
            with open(os.path.join(self.template_dir, "names.json"), "r") as f:
                name_dict = json.load(f)
        except FileNotFoundError:
            logger.warning(f"names.json not found in {self.template_dir}. Using default column names.")
            name_dict = {}
        
        # Mapping for QFV types to names.json keys
        qfv_to_dict_map = {
            "Vascular": "Arterial",
            "Lobe": "Lobes",
            "Aspects": "Aspects",
            "AspectsPC": "AspectsPC",
            "Ventricles": "Ventricles",
            "BMS": "BMS",
            "BPM": "BPM"
        }
        
        # ========== Save QFV files ==========
        for i, qfv_type in enumerate(qfv_types):
            if i < len(results["qfv_features"]) and results["qfv_features"][i] is not None:
                dict_key = qfv_to_dict_map.get(qfv_type)
                qfv_names = name_dict.get(dict_key, {}).get("QFV_names", []) if dict_key else []
                
                num_features = len(results["qfv_features"][i]) - 1
                
                # Use QFV_names if available, otherwise use default naming
                if qfv_names and len(qfv_names) > 0:
                    roi_columns = {
                        qfv_names[j] if j < len(qfv_names) else f'{qfv_type}_QFV_{j}': 
                        results["qfv_features"][i][j + 1] 
                        for j in range(num_features)
                    }
                else:
                    roi_columns = {
                        f'{qfv_type}_QFV_{j}': results["qfv_features"][i][j + 1] 
                        for j in range(num_features)
                    }
                
                df = pd.DataFrame({
                    'subject_id': [subject_id],
                    'volml': [results["qfv_features"][i][0]],
                    **roi_columns
                })
                
                qfv_base = qfv_stem_map.get(qfv_type, "BPM_TYPE1" if qfv_type == "BPM" else qfv_type.upper()) if qfv_stem_map else ("BPM_TYPE1" if qfv_type == "BPM" else qfv_type.upper())
                file_stem = f"{qfv_base}_{qfv_suffix}"
                output_path = os.path.join(output_dir, f"{subject_id}_{file_stem}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved {qfv_type} {qfv_suffix} to {output_path}")

        # ========== Save lesionload files ==========
        # Mapping for FV atlas keys to names.json keys
        fv_to_atlas_map = {
            "vascular": "Arterial",
            "watershed": "Watershed",
            "lobe": "Lobes",
            "aspects": "Aspects",
            "aspectpc": "AspectsPC",
            "bmos": "BMOS",
            "bmis": "BMIS",
            "bpm_type1": "BPM"
        }
        
        for atlas, vec in results["FV"].items():
            if vec is not None:
                # Skip Ventricles_LVO and Ventricles_LVI (handle separately)
                if atlas in ["Ventricles_LVO", "Ventricles_LVI"]:
                    continue
                
                dict_key = fv_to_atlas_map.get(atlas)
                label_names = name_dict.get(dict_key, {}).get("LabelLookupTable", []) if dict_key else []
                
                # Use LabelLookupTable if available, otherwise use default naming
                if label_names and len(label_names) > 0:
                    columns = {
                        label_names[i] if i < len(label_names) else f"{atlas}_ROI_{i+1}": vec[i]
                        for i in range(len(vec))
                    }
                else:
                    columns = {f"{atlas}_ROI_{i+1}": vec[i] for i in range(len(vec))}
                
                df = pd.DataFrame([columns])
                
                atlas_upper = atlas.upper()
                lesionload_base = lesionload_stem_map.get(atlas, "ASPECTSPC" if atlas_upper == "ASPECTPC" else atlas_upper) if lesionload_stem_map else ("ASPECTSPC" if atlas_upper == "ASPECTPC" else atlas_upper)
                output_path = os.path.join(output_dir, f"{subject_id}_{lesionload_base}_{lesionload_suffix}.csv")
                # if not BMOS and not BMIS, then save it.
                if atlas_upper not in ["BMOS", "BMIS"]:
                    df.to_csv(output_path, index=False)
                    logger.info(f"Saved {atlas} {lesionload_suffix} to {output_path}")
        
        # ========== Special handling for Ventricles ==========
        if "Ventricles_LVO" in results["FV"] and "Ventricles_LVI" in results["FV"]:
            vec_lvo = results["FV"]["Ventricles_LVO"]
            vec_lvi = results["FV"]["Ventricles_LVI"]
            
            if vec_lvo is not None and vec_lvi is not None:
                # Combine LVO (first 10) and LVI (last 10)
                vec_combined = np.append(vec_lvo[:10], vec_lvi[10:20])
                
                dict_key = "Ventricles"
                label_names = name_dict.get(dict_key, {}).get("LabelLookupTable", [])
                
                if label_names and len(label_names) > 0:
                    columns = {
                        label_names[i] if i < len(label_names) else f"Ventricles_ROI_{i+1}": vec_combined[i]
                        for i in range(len(vec_combined))
                    }
                else:
                    columns = {f"Ventricles_ROI_{i+1}": vec_combined[i] for i in range(len(vec_combined))}
                
                df = pd.DataFrame([columns])
                ventricles_base = lesionload_stem_map.get("Ventricles", "VENTRICLES") if lesionload_stem_map else "VENTRICLES"
                output_path = os.path.join(output_dir, f"{subject_id}_{ventricles_base}_{lesionload_suffix}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved Ventricles {lesionload_suffix} to {output_path}")


def process_qfv_single(template_dir: str, subj_dir: str, output_dir: str, subject_id: str = None):
    builder = StrokeQFVCalculator(template_dir=template_dir)
    
    if subject_id is None:
        subject_id = os.path.basename(subj_dir)

    reg_dir = os.path.join(subj_dir, "registration")

    stroke_img_path = os.path.join(subj_dir, "segment", f"{subject_id}_stroke-mask_space-MNI152_affsyn.nii.gz")
    if not os.path.exists(stroke_img_path):
        logger.warning(f"Stroke image not found at {stroke_img_path}")
        return
    mask_raw_mni_path = os.path.join(reg_dir, f"{subject_id}_DWIbrain-mask_space-MNI152_affsyn.nii.gz")
    adc_mni_path = os.path.join(reg_dir, f"{subject_id}_ADC_space-MNI152_affsyn.nii.gz")

    if not (os.path.exists(stroke_img_path) and 
            os.path.exists(mask_raw_mni_path) and 
            os.path.exists(adc_mni_path)):
        logger.warning(f"Skipping QFV for {subject_id}: Required files not found in {reg_dir} or segment directory")
        return

    results = builder.calculate(
        stroke_img_path=stroke_img_path,
        mask_raw_mni_path=mask_raw_mni_path,
        adc_mni_path=adc_mni_path,
        adc_threshold=0.5490,
    )

    builder.save_QFV_to_csv(results, subject_id=subject_id, output_dir=output_dir)
    return results
