#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_dataset_builder.py
Build QFV/FV training features from subject-level NIfTI in MNI grid.

Usage (import):
    from utils.feature_dataset_builder import FeatureDatasetBuilder
    builder = FeatureDatasetBuilder(
        template_dir="/path/to/templates/JHU_MNI_182",
        adc_threshold=0.0016,  # mm^2/s
        round_labels=True,
        deci_prec=4
    )
    # single subject (returns FV list + QFV list)
    fv_list, qfv_list = builder.process_single_subject("sub-05a971ae", "/path/to/data", pred=False)

    # parallel dataset build
    vascular_df, lobe_df, aspect_df, hydro_df = builder.create_training_dataset_parallel(
        sub_ids=["sub-05a971ae", "sub-1580a0a4"],
        data_dir="/path/to/data",
        n_workers=8,
        pred=False
    )

Usage (CLI):
    python utils/feature_dataset_builder.py \
        --data-dir /path/to/Reg_AffSyN_trainset \
        --template-dir /path/to/templates/JHU_MNI_182 \
        --save-dir /path/to/save \
        --n-workers 16 \
        --pred false
"""
from __future__ import annotations

import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import traceback


# ----------------------------- Helpers (pure functions) -----------------------------
def load_nifti_as_ras(file_path: str) -> nib.Nifti1Image:
    img = nib.load(file_path)
    return nib.as_closest_canonical(img)

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

def vec_BPMAtlas2visual(vec, BPMtype1_dict):
    # BPM Type 1 ROIs combined and re-arranged to 89 Visual ROIs
    visual_vec = np.zeros(89)
    visual_vec[0] = vec[BPMtype1_dict['SFG_L']] + vec[BPMtype1_dict['SFWM_L']] # SFG_L + SFWM_L
    visual_vec[1] = vec[BPMtype1_dict['SFG_R']] + vec[BPMtype1_dict['SFWM_R']] # SFG_R + SFWM_R
    visual_vec[2] = vec[BPMtype1_dict['SFG_PFC_L']] + vec[BPMtype1_dict['SFWM_PFC_L']] # SFG_PFC_L + SFWM_PFC_L
    visual_vec[3] = vec[BPMtype1_dict['SFG_PFC_R']] + vec[BPMtype1_dict['SFWM_PFC_R']] # SFG_PFC_R + SFWM_PFC_R
    visual_vec[4] = vec[BPMtype1_dict['SFG_pole_L']] + vec[BPMtype1_dict['SFWM_pole_L']] # SFG_pole_L + SFWM_pole_L
    visual_vec[5] = vec[BPMtype1_dict['SFG_pole_R']] + vec[BPMtype1_dict['SFWM_pole_R']] # SFG_pole_R + SFWM_pole_R
    visual_vec[6] = vec[BPMtype1_dict['MFG_L']] + vec[BPMtype1_dict['MFWM_L']] # MFG_L + MFWM_L
    visual_vec[7] = vec[BPMtype1_dict['MFG_R']] + vec[BPMtype1_dict['MFWM_R']] # MFG_R + MFWM_R
    visual_vec[8] = vec[BPMtype1_dict['MFG_DPFC_L']] + vec[BPMtype1_dict['MFWM_DPFC_L']] # MFG_DPFC_L + MFWM_DPFC_L
    visual_vec[9] = vec[BPMtype1_dict['MFG_DPFC_R']] + vec[BPMtype1_dict['MFWM_DPFC_R']] # MFG_DPFC_R + MFWM_DPFC_R
    visual_vec[10] = vec[BPMtype1_dict['IFG_opercularis_L']] + vec[BPMtype1_dict['IFG_orbitalis_L']] + vec[BPMtype1_dict['IFG_triangularis_L']] + vec[BPMtype1_dict['IFWM_opercularis_L']] + vec[BPMtype1_dict['IFWM_orbitalis_L']] + vec[BPMtype1_dict['IFWM_triangularis_L']] # IFG_opercularis_L + IFG_orbitalis_L + IFG_triangularis_L + IFWM_opercularis_L + IFWM_orbitalis_L + IFWM_triangularis_L
    visual_vec[11] = vec[BPMtype1_dict['IFG_opercularis_R']] + vec[BPMtype1_dict['IFG_orbitalis_R']] + vec[BPMtype1_dict['IFG_triangularis_R']] + vec[BPMtype1_dict['IFWM_opercularis_R']] + vec[BPMtype1_dict['IFWM_orbitalis_R']] + vec[BPMtype1_dict['IFWM_triangularis_R']] # IFG_opercularis_R + IFG_orbitalis_R + IFG_triangularis_R + IFWM_opercularis_R + IFWM_orbitalis_R + IFWM_triangularis_R
    visual_vec[12] = vec[BPMtype1_dict['LFOG_L']] + vec[BPMtype1_dict['MFOG_L']] + vec[BPMtype1_dict['RG_L']] + vec[BPMtype1_dict['LFOWM_L']] + vec[BPMtype1_dict['MFOWM_L']] + vec[BPMtype1_dict['RGWM_L']] # LFOG_L + MFOG_L + RG_L + LFOWM_L + MFOWM_L + RGWM_L
    visual_vec[13] = vec[BPMtype1_dict['LFOG_R']] + vec[BPMtype1_dict['MFOG_R']] + vec[BPMtype1_dict['RG_R']] + vec[BPMtype1_dict['LFOWM_R']] + vec[BPMtype1_dict['MFOWM_R']] + vec[BPMtype1_dict['RGWM_R']] # LFOG_R + MFOG_R + RG_R + LFOWM_R + MFOWM_R + RGWM_R
    visual_vec[14] = vec[BPMtype1_dict['PoCG_L']] + vec[BPMtype1_dict['PoCWM_L']] # PoCG_L + PoCWM_L
    visual_vec[15] = vec[BPMtype1_dict['PoCG_R']] + vec[BPMtype1_dict['PoCWM_R']] # PoCG_R + PoCWM_R
    visual_vec[16] = vec[BPMtype1_dict['PrCG_L']] + vec[BPMtype1_dict['PrCWM_L']] # PrCG_L + PrCWM_L
    visual_vec[17] = vec[BPMtype1_dict['PrCG_R']] + vec[BPMtype1_dict['PrCWM_R']] # PrCG_R + PrCWM_R
    visual_vec[18] = vec[BPMtype1_dict['SPG_L']] + vec[BPMtype1_dict['SPWM_L']] # SPG_L + SPWM_L
    visual_vec[19] = vec[BPMtype1_dict['SPG_R']] + vec[BPMtype1_dict['SPWM_R']] # SPG_R + SPWM_R
    visual_vec[20] = vec[BPMtype1_dict['SMG_L']] + vec[BPMtype1_dict['SMWM_L']] # SMG_L + SMWM_L
    visual_vec[21] = vec[BPMtype1_dict['SMG_R']] + vec[BPMtype1_dict['SMWM_R']] # SMG_R + SMWM_R
    visual_vec[22] = vec[BPMtype1_dict['AG_L']] + vec[BPMtype1_dict['AWM_L']] # AG_L + AWM_L
    visual_vec[23] = vec[BPMtype1_dict['AG_R']] + vec[BPMtype1_dict['AWM_R']] # AG_R + AWM_R
    visual_vec[24] = vec[BPMtype1_dict['PrCu_L']] + vec[BPMtype1_dict['PrCuWM_L']] # PrCu_L + PrCuWM_L
    visual_vec[25] = vec[BPMtype1_dict['PrCu_R']] + vec[BPMtype1_dict['PrCuWM_R']] # PrCu_R + PrCuWM_R
    visual_vec[26] = vec[BPMtype1_dict['STG_L']] + vec[BPMtype1_dict['STWM_L']] # STG_L + STWM_L
    visual_vec[27] = vec[BPMtype1_dict['STG_R']] + vec[BPMtype1_dict['STWM_R']] # STG_R + STWM_R
    visual_vec[28] = vec[BPMtype1_dict['STG_L_pole']] + vec[BPMtype1_dict['STWM_L_pole']] # STG_L_pole + STWM_L_pole
    visual_vec[29] = vec[BPMtype1_dict['STG_R_pole']] + vec[BPMtype1_dict['STWM_R_pole']] # STG_R_pole + STWM_R_pole
    visual_vec[30] = vec[BPMtype1_dict['MTG_L']] + vec[BPMtype1_dict['MTWM_L']] # MTG_L + MTWM_L
    visual_vec[31] = vec[BPMtype1_dict['MTG_R']] + vec[BPMtype1_dict['MTWM_R']] # MTG_R + MTWM_R
    visual_vec[32] = vec[BPMtype1_dict['MTG_L_pole']] + vec[BPMtype1_dict['MTWM_L_pole']] # MTG_L_pole + MTWM_L_pole
    visual_vec[33] = vec[BPMtype1_dict['MTG_R_pole']] + vec[BPMtype1_dict['MTWM_R_pole']] # MTG_R_pole + MTWM_R_pole
    visual_vec[34] = vec[BPMtype1_dict['ITG_L']] + vec[BPMtype1_dict['ITWM_L']] # ITG_L + ITWM_L
    visual_vec[35] = vec[BPMtype1_dict['ITG_R']] + vec[BPMtype1_dict['ITWM_R']] # ITG_R + ITWM_R
    visual_vec[36] = vec[BPMtype1_dict['PHG_L']] + vec[BPMtype1_dict['ENT_L']] + vec[BPMtype1_dict['Amyg_L']] + vec[BPMtype1_dict['Hippo_L']] + vec[BPMtype1_dict['CGH_L']] # PHG_L + ENT_L + Amyg_L + Hippo_L + CGH_L
    visual_vec[37] = vec[BPMtype1_dict['PHG_R']] + vec[BPMtype1_dict['ENT_R']] + vec[BPMtype1_dict['Amyg_R']] + vec[BPMtype1_dict['Hippo_R']] + vec[BPMtype1_dict['CGH_R']] # PHG_R + ENT_R + Amyg_R + Hippo_R + CGH_R
    visual_vec[38] = vec[BPMtype1_dict['FuG_L']] + vec[BPMtype1_dict['FuWM_L']] # FuG_L + FuWM_L
    visual_vec[39] = vec[BPMtype1_dict['FuG_R']] + vec[BPMtype1_dict['FuWM_R']] # FuG_R + FuWM_R
    visual_vec[40] = vec[BPMtype1_dict['SOG_L']] + vec[BPMtype1_dict['SOWM_L']] # SOG_L + SOWM_L
    visual_vec[41] = vec[BPMtype1_dict['SOG_R']] + vec[BPMtype1_dict['SOWM_R']] # SOG_R + SOWM_R
    visual_vec[42] = vec[BPMtype1_dict['MOG_L']] + vec[BPMtype1_dict['MOWM_L']] # MOG_L + MOWM_L
    visual_vec[43] = vec[BPMtype1_dict['MOG_R']] + vec[BPMtype1_dict['MOWM_R']] # MOG_R + MOWM_R
    visual_vec[44] = vec[BPMtype1_dict['IOG_L']] + vec[BPMtype1_dict['IOWM_L']] # IOG_L + IOWM_L
    visual_vec[45] = vec[BPMtype1_dict['IOG_R']] + vec[BPMtype1_dict['IOWM_R']] # IOG_R + IOWM_R
    visual_vec[46] = vec[BPMtype1_dict['Cu_L']] + vec[BPMtype1_dict['CuWM_L']] # Cu_L + CuWM_L
    visual_vec[47] = vec[BPMtype1_dict['Cu_R']] + vec[BPMtype1_dict['CuWM_R']] # Cu_R + CuWM_R
    visual_vec[48] = vec[BPMtype1_dict['LG_L']] + vec[BPMtype1_dict['LWM_L']] # LG_L + LWM_L
    visual_vec[49] = vec[BPMtype1_dict['LG_R']] + vec[BPMtype1_dict['LWM_R']] # LG_R + LWM_R
    visual_vec[50] = vec[BPMtype1_dict['rostral_ACC_L']] + vec[BPMtype1_dict['subcallosal_ACC_L']] + vec[BPMtype1_dict['subgenual_ACC_L']] + vec[BPMtype1_dict['dorsal_ACC_L']] + vec[BPMtype1_dict['PCC_L']] + vec[BPMtype1_dict['CGC_L']] + vec[BPMtype1_dict['rostralWM_ACC_L']] + vec[BPMtype1_dict['subcallosalWM_ACC_L']] + vec[BPMtype1_dict['subgenualWM_ACC_L']] + vec[BPMtype1_dict['dorsalWM_ACC_L']] + vec[BPMtype1_dict['PCCWM_L']] # rostral_ACC_L + subcallosal_ACC_L + subgenual_ACC_L + dorsal_ACC_L + PCC_L + CGC_L + rostralWM_ACC_L + subcallosalWM_ACC_L + subgenualWM_ACC_L + dorsalWM_ACC_L + PCCWM_L
    visual_vec[51] = vec[BPMtype1_dict['rostral_ACC_R']] + vec[BPMtype1_dict['subcallosal_ACC_R']] + vec[BPMtype1_dict['subgenual_ACC_R']] + vec[BPMtype1_dict['dorsal_ACC_R']] + vec[BPMtype1_dict['PCC_R']] + vec[BPMtype1_dict['CGC_R']] + vec[BPMtype1_dict['rostralWM_ACC_R']] + vec[BPMtype1_dict['subcallosalWM_ACC_R']] + vec[BPMtype1_dict['subgenualWM_ACC_R']] + vec[BPMtype1_dict['dorsalWM_ACC_R']] + vec[BPMtype1_dict['PCCWM_R']] # rostral_ACC_R + subcallosal_ACC_R + subgenual_ACC_R + dorsal_ACC_R + PCC_R + CGC_R + rostralWM_ACC_R + subcallosalWM_ACC_R + subgenualWM_ACC_R + dorsalWM_ACC_R + PCCWM_R
    visual_vec[52] = vec[BPMtype1_dict['Ins_L']] + vec[BPMtype1_dict['EC_L']] + vec[BPMtype1_dict['Pins_L']] # Ins_L + EC_L + Pins_L
    visual_vec[53] = vec[BPMtype1_dict['Ins_R']] + vec[BPMtype1_dict['EC_R']] + vec[BPMtype1_dict['Pins_R']] # Ins_R + EC_R + Pins_R
    visual_vec[54] = vec[BPMtype1_dict['Caud_L']] # Caud_L
    visual_vec[55] = vec[BPMtype1_dict['Caud_R']] # Caud_R
    visual_vec[56] = vec[BPMtype1_dict['Put_L']] # Put_L
    visual_vec[57] = vec[BPMtype1_dict['Put_R']] # Put_R
    visual_vec[58] = vec[BPMtype1_dict['GP_L']] # GP_L
    visual_vec[59] = vec[BPMtype1_dict['GP_R']] # GP_R
    visual_vec[60] = vec[BPMtype1_dict['Thal_L']] # Thal_L
    visual_vec[61] = vec[BPMtype1_dict['Thal_R']] # Thal_R
    visual_vec[62] = vec[BPMtype1_dict['HypoThalamus_L']] + vec[BPMtype1_dict['HypoThalamus_R']] + vec[BPMtype1_dict['Mynert_L']] + vec[BPMtype1_dict['Mynert_R']] + vec[BPMtype1_dict['NucAccumbens_L']] + vec[BPMtype1_dict['NucAccumbens_R']] + vec[BPMtype1_dict['RedNc_L']] + vec[BPMtype1_dict['RedNc_R']] + vec[BPMtype1_dict['Snigra_L']] + vec[BPMtype1_dict['Snigra_R']] + vec[BPMtype1_dict['CP_L']] + vec[BPMtype1_dict['CP_R']] + vec[BPMtype1_dict['Midbrain_L']] + vec[BPMtype1_dict['Midbrain_R']] + vec[BPMtype1_dict['CST_L']] + vec[BPMtype1_dict['CST_R']] # HypoThalamus_L + HypoThalamus_R + Mynert_L + Mynert_R + NucAccumbens_L + NucAccumbens_R + RedNc_L + RedNc_R + Snigra_L + Snigra_R + CP_L + CP_R + Midbrain_L + Midbrain_R + CST_L + CST_R
    visual_vec[63] = vec[BPMtype1_dict['Cerebellum_L']] + vec[BPMtype1_dict['SCP_L']] + vec[BPMtype1_dict['MCP_L']] + vec[BPMtype1_dict['PCT_L']] + vec[BPMtype1_dict['ICP_L']] + vec[BPMtype1_dict['CerebellumWM_L']] # Cerebellum_L + SCP_L + MCP_L + PCT_L + ICP_L + CerebellumWM_L
    visual_vec[64] = vec[BPMtype1_dict['Cerebellum_R']] + vec[BPMtype1_dict['SCP_R']] + vec[BPMtype1_dict['MCP_R']] + vec[BPMtype1_dict['PCT_R']] + vec[BPMtype1_dict['ICP_R']] + vec[BPMtype1_dict['CerebellumWM_R']] # Cerebellum_R + SCP_R + MCP_R + PCT_R + ICP_R + CerebellumWM_R
    visual_vec[65] = vec[BPMtype1_dict['ML_L']] + vec[BPMtype1_dict['ML_R']] + vec[BPMtype1_dict['Pons_L']] + vec[BPMtype1_dict['Pons_R']] + vec[BPMtype1_dict['Medulla_L']] + vec[BPMtype1_dict['Medulla_R']] # ML_L + ML_R + Pons_L + Pons_R + Medulla_L + Medulla_R
    visual_vec[66] = vec[BPMtype1_dict['ACR_L']] # ACR_L
    visual_vec[67] = vec[BPMtype1_dict['ACR_R']] # ACR_R
    visual_vec[68] = vec[BPMtype1_dict['SCR_L']] # SCR_L
    visual_vec[69] = vec[BPMtype1_dict['SCR_R']] # SCR_R
    visual_vec[70] = vec[BPMtype1_dict['PCR_L']] # PCR_L
    visual_vec[71] = vec[BPMtype1_dict['PCR_R']] # PCR_R
    visual_vec[72] = vec[BPMtype1_dict['GCC_L']] + vec[BPMtype1_dict['GCC_R']] + vec[BPMtype1_dict['BCC_L']] + vec[BPMtype1_dict['BCC_R']] + vec[BPMtype1_dict['SCC_L']] + vec[BPMtype1_dict['SCC_R']] # GCC_L + GCC_R + BCC_L + BCC_R + SCC_L + SCC_R
    visual_vec[73] = vec[BPMtype1_dict['ALIC_L']] + vec[BPMtype1_dict['PLIC_L']] + vec[BPMtype1_dict['RLIC_L']] # ALIC_L + PLIC_L + RLIC_L
    visual_vec[74] = vec[BPMtype1_dict['ALIC_R']] + vec[BPMtype1_dict['PLIC_R']] + vec[BPMtype1_dict['RLIC_R']] # ALIC_R + PLIC_R + RLIC_R
    visual_vec[75] = vec[BPMtype1_dict['Fx/ST_L']] + vec[BPMtype1_dict['PTR_L']] + vec[BPMtype1_dict['SS_L']] # Fx/ST_L + PTR_L + SS_L
    visual_vec[76] = vec[BPMtype1_dict['Fx/ST_R']] + vec[BPMtype1_dict['PTR_R']] + vec[BPMtype1_dict['SS_R']] # Fx/ST_R + PTR_R + SS_R
    visual_vec[77] = vec[BPMtype1_dict['IFO_L']] # IFO_L
    visual_vec[78] = vec[BPMtype1_dict['IFO_R']] # IFO_R
    visual_vec[79] = vec[BPMtype1_dict['SFO_L']] + vec[BPMtype1_dict['SLF_L']] # SFO_L + SLF_L
    visual_vec[80] = vec[BPMtype1_dict['SFO_R']] + vec[BPMtype1_dict['SLF_R']] # SFO_R + SLF_R
    visual_vec[81] = vec[BPMtype1_dict['UNC_L']] # UNC_L
    visual_vec[82] = vec[BPMtype1_dict['UNC_R']] # UNC_R
    visual_vec[83] = vec[BPMtype1_dict['PSTG_L']] + vec[BPMtype1_dict['PSTGWM_L']] # PSTG_L + PSTGWM_L
    visual_vec[84] = vec[BPMtype1_dict['PSTG_R']] + vec[BPMtype1_dict['PSTGWM_R']] # PSTG_R + PSTGWM_R
    visual_vec[85] = vec[BPMtype1_dict['PMTG_L']] + vec[BPMtype1_dict['PMTGWM_L']] # PMTG_L + PMTGWM_L
    visual_vec[86] = vec[BPMtype1_dict['PMTG_R']] + vec[BPMtype1_dict['PMTGWM_R']] # PMTG_R + PMTGWM_R
    visual_vec[87] = vec[BPMtype1_dict['PITG_L']] + vec[BPMtype1_dict['PITGWM_L']] # PITG_L + PITGWM_L
    visual_vec[88] = vec[BPMtype1_dict['PITG_R']] + vec[BPMtype1_dict['PITGWM_R']] # PITG_R + PITGWM_R
    return visual_vec

def get_BPM_visual_prob_comb(bpm_prob: np.ndarray) -> np.ndarray:
    out = np.zeros(46, dtype=bpm_prob.dtype)
    out[0] = bpm_prob[0] + bpm_prob[1] # SF: SF_L + SF_R
    out[1] = bpm_prob[2] + bpm_prob[3] # SF_PF: SF_PF_L + SF_PF_R
    out[2] = bpm_prob[4] + bpm_prob[5] # SF_pole: SF_pole_L + SF_pole_R
    out[3] = bpm_prob[6] + bpm_prob[7] # MF: MF_L + MF_R
    out[4] = bpm_prob[8] + bpm_prob[9] # MF_DPF: MF_DPF_L + MF_DPF_R
    out[5] = bpm_prob[10] + bpm_prob[11] # IF: IF_L + IF_R
    out[6] = bpm_prob[12] + bpm_prob[13] # FO: FO_L + FO_R
    out[7] = bpm_prob[14] + bpm_prob[15] # PoC: PoC_L + PoC_R
    out[8] = bpm_prob[16] + bpm_prob[17] # PrC: PrC_L + PrC_R
    out[9] = bpm_prob[18] + bpm_prob[19] # SP: SP_L + SP_R
    out[10] = bpm_prob[20] + bpm_prob[21] # SM: SM_L + SM_R
    out[11] = bpm_prob[22] + bpm_prob[23] # AG: AG_L + AG_R
    out[12] = bpm_prob[24] + bpm_prob[25] # PrCu: PrCu_L + PrCu_R
    out[13] = bpm_prob[26] + bpm_prob[27] # ST: ST_L + ST_R
    out[14] = bpm_prob[28] + bpm_prob[29] # ST_pole: ST_pole_L + ST_pole_R
    out[15] = bpm_prob[30] + bpm_prob[31] # MT: MT_L + MT_R
    out[16] = bpm_prob[32] + bpm_prob[33] # MT_pole: MT_pole_L + MT_pole_R
    out[17] = bpm_prob[34] + bpm_prob[35] # IT: IT_L + IT_R
    out[18] = bpm_prob[36] + bpm_prob[37] # Limbic: Limbic_L + Limbic_R
    out[19] = bpm_prob[38] + bpm_prob[39] # Fu: Fu_L + Fu_R
    out[20] = bpm_prob[40] + bpm_prob[41] # SO: SO_L + SO_R
    out[21] = bpm_prob[42] + bpm_prob[43] # MO: MO_L + MO_R
    out[22] = bpm_prob[44] + bpm_prob[45] # IO: IO_L + IO_R
    out[23] = bpm_prob[46] + bpm_prob[47] # Cu: Cu_L + Cu_R
    out[24] = bpm_prob[48] + bpm_prob[49] # Ling: Ling_L + Ling_R
    out[25] = bpm_prob[50] + bpm_prob[51] # Cing: Cing_L + Cing_R
    out[26] = bpm_prob[52] + bpm_prob[53] # Ins: Ins_L + Ins_R
    out[27] = bpm_prob[54] + bpm_prob[55] # Caud: Caud_L + Caud_R
    out[28] = bpm_prob[56] + bpm_prob[57] # Put: Put_L + Put_R
    out[29] = bpm_prob[58] + bpm_prob[59] # GP: GP_L + GP_R
    out[30] = bpm_prob[60] + bpm_prob[61] # Thal: Thal_L + Thal_R
    out[31] = bpm_prob[62] # midb: midb
    out[32] = bpm_prob[63] + bpm_prob[64] # cereb: cereb_L + cereb_R
    out[33] = bpm_prob[65] # pons: pons
    out[34] = bpm_prob[66] + bpm_prob[67] # ACR: ACR_L + ACR_R
    out[35] = bpm_prob[68] + bpm_prob[69] # SCR: SCR_L + SCR_R
    out[36] = bpm_prob[70] + bpm_prob[71] # PCR: PCR_L + PCR_R
    out[37] = bpm_prob[72] # CC: CC
    out[38] = bpm_prob[73] + bpm_prob[74] # IC: IC_L + IC_R
    out[39] = bpm_prob[75] + bpm_prob[76] # PTR: PTR_L + PTR_R
    out[40] = bpm_prob[77] + bpm_prob[78] # IFOF: IFOF_L + IFOF_R
    out[41] = bpm_prob[79] + bpm_prob[80] # SLF: SLF_L + SLF_R
    out[42] = bpm_prob[81] + bpm_prob[82] # Unc: Unc_L + Unc_R
    out[43] = bpm_prob[83] + bpm_prob[84] # PST: PST_L + PST_R
    out[44] = bpm_prob[85] + bpm_prob[86] # PMT: PMT_L + PMT_R
    out[45] = bpm_prob[87] + bpm_prob[88] # PIT: PIT_L + PIT_R
    return out

def get_Aspectpc_visual_prob_comb(pca_prob: np.ndarray) -> np.ndarray:
    """
    ASPECTPC QFV:[PCAL, PCAR, ThalamusL, ThalamusR, cerebellumL, cerebellumR, pons, midbrain]
    Output [PCA, Thalamus, cerebellum, pons, midbrain]
    """
    if pca_prob is None:
        return None
    pca_prob = np.asarray(pca_prob)
    if pca_prob.shape[0] < 8:
        raise ValueError(f"ASPECTPC prob length < 8, got {pca_prob.shape[0]}")
    lr_comb = pca_prob[:-2:2] + pca_prob[1:-2:2]
    return np.concatenate([lr_comb, pca_prob[-2:]])
