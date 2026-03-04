#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADS – Automatic Radiological Report (single-file version)

- Fixes _rn() arity bug by keeping ONE global _rn(roi)
- Normalizes ROI names via ReNaming_dict with robust fallbacks
- Preserves legacy report text/table layout
- Avoids pandas FutureWarning by explicit dtype handling
- Works with AA_models/{vascular,lobe,aspect,hydro}/*.pkl
- Expects QFV CSVs under subject reporting dir

Author: you + a very pedantic AI
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import nibabel as nib


# ===================== Human-facing names & normalization =====================

# Your explicit human-facing overrides
ReNaming_dict = {
    # Vascular
    'ACA': 'anterior cerebral artery',
    'MCA': 'middle cerebral artery',
    'PCA': 'posterior cerebral artery',
    'cerebellar': 'cerebellar artery',
    'basilar': 'basilar artery',
    'Lenticulostriate': 'Lateral Lenticulostriate',
    'LatLenticulostriate': 'Lateral Lenticulostriate',  # legacy key
    'Choroidal&Thalamoperforating': 'Choroidal & Thalamoperforating',
    'Choroidal and Thalamoperfurating': 'Choroidal & Thalamoperforating',
    'Choroidal_and_Thalamoperfurating': 'Choroidal & Thalamoperforating',
    'Choroidal & Thalamoperfurating': 'Choroidal & Thalamoperforating',

    # Lobe/anatomy
    'basal ganglia': 'basal ganglia',
    'deep white matter': 'deep white matter',
    'cerebellum': 'cerebellum',
    'frontal': 'frontal lobe',
    'insula': 'insula',
    'internal capsule': 'internal capsule',
    'brainstem': 'brainstem',
    'occipital': 'occipital lobe',
    'parietal': 'parietal lobe',
    'temporal': 'temporal lobe',
    'thalamus': 'thalamus',

    # ASPECTS
    'Caudate': 'Caudate',
    'lentiform': 'lentiform',
    'IC': 'internal capsule',
    'M1': 'M1', 'M2': 'M2', 'M3': 'M3', 'M4': 'M4', 'M5': 'M5', 'M6': 'M6',
    
    # PCA ASPECTS
    'AspectsPC': ['PCA', 'Thalamus', 'cerebellum', 'pons', 'midbrain'],

    # Hydro
    'hydrocephalus': 'hydrocephalus',
    'Aspect_total': 'Total Aspects',
}

# Reasonable defaults if an ROI isn’t present in your overrides
RENAMING_DEFAULTS = {
    # vascular
    'ACA': 'anterior cerebral artery',
    'MCA': 'middle cerebral artery',
    'PCA': 'posterior cerebral artery',
    'cerebellar': 'cerebellar artery',
    'basilar': 'basilar artery',
    'Lenticulostriate': 'Lateral Lenticulostriate',
    'Choroidal & Thalamoperfurating': 'Choroidal & Thalamoperforating',
    'watershed': 'watershed',
    # lobe / structures
    'basal ganglia': 'basal ganglia',
    'deep white matter': 'deep white matter',
    'cerebellum': 'cerebellum',
    'frontal': 'frontal lobe',
    'insula': 'insula',
    'internal capsule': 'internal capsule',
    'brainstem': 'brainstem',
    'occipital': 'occipital lobe',
    'parietal': 'parietal lobe',
    'temporal': 'temporal lobe',
    'thalamus': 'thalamus',
    # hydro
    'hydrocephalus': 'hydrocephalus',
    # ASPECTS
    'Caudate': 'Caudate',
    'lentiform': 'lentiform',
    'IC': 'internal capsule',
    'M1': 'M1', 'M2': 'M2', 'M3': 'M3', 'M4': 'M4', 'M5': 'M5', 'M6': 'M6',
}

def _rn(roi: str) -> str:
    """
    Safe rename: prefer user ReNaming_dict, then RENAMING_DEFAULTS, else raw roi.
    Keep this as the ONLY _rn in the module (do NOT shadow inside main()).
    """
    try:
        label = ReNaming_dict.get(roi)
        if label is not None:
            return label
    except NameError:
        pass
    return RENAMING_DEFAULTS.get(roi, roi)


# ============================== General utilities =============================

def load_nifti(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing NIfTI: {path}")
    return nib.as_closest_canonical(nib.load(str(path)))

def voxel_volume_ml(img: nib.Nifti1Image) -> float:
    z = img.header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2]) / 1000.0  # mm^3 -> ml

def compute_volumes_ml(lesion_arr: np.ndarray, img: nib.Nifti1Image) -> Tuple[float, float, float, np.ndarray]:
    vv_ml = voxel_volume_ml(img)
    lesion_bin = (lesion_arr > 0.5).astype(np.uint8)
    total_ml = lesion_bin.sum() * vv_ml
    x_mid = lesion_bin.shape[0] // 2  # RAS
    left_ml  = lesion_bin[:x_mid, :, :].sum() * vv_ml
    right_ml = lesion_bin[x_mid:, :, :].sum() * vv_ml
    return total_ml, left_ml, right_ml, lesion_bin

def estimate_icv_ml(mask_path: Path, img: nib.Nifti1Image) -> float:
    if mask_path.exists():
        m = nib.as_closest_canonical(nib.load(str(mask_path)))
        vv_ml = voxel_volume_ml(m)
        mask = (np.asanyarray(m.dataobj) > 0.5).astype(np.uint8)
        return float(mask.sum() * vv_ml)
    return 1200.0  # fallback


# ============================== Reporting helpers =============================

def _to_numeric_label(x):
    # Accept 1/0, True/False, "AFFECTED"/"NORMAL", or strings "1"/"0"
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x >= 0.5)
    xs = str(x).strip().lower()
    if xs in {"1", "true", "affected", "positive", "pos"}:
        return 1
    if xs in {"0", "false", "normal", "negative", "neg"}:
        return 0
    # Unknown → treat as 0 (safe)
    return 0


def _normalize_key(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("&", "and")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _row_value_by_name(row_like: pd.Series | dict, feature: str) -> float:
    target = _normalize_key(feature)
    for col in row_like:
        if _normalize_key(col) == target:
            val = row_like[col]
            return float(val) if pd.notna(val) else 0.0
    return 0.0


def _get_volume_value(row_like: pd.Series | dict, prefer: str) -> float:
    key_map = {
        "logvol": ["logvol", "log_ml", "logml", "stroke_volume_logml"],
        "volml": ["volml", "vol_ml", "volume_ml", "vol"],
    }
    for key in key_map.get(prefer, []):
        for col in row_like:
            if _normalize_key(col) == _normalize_key(key):
                return float(row_like[col])

    if prefer == "logvol":
        vol = _get_volume_value(row_like, "volml")
        return float(np.log10(max(vol, 1e-6)))

    raise KeyError(f"Missing volume feature for {prefer}")


def _build_x_from_feature_names(row_like: pd.Series | dict, feature_names: list[str]) -> np.ndarray:
    vals: list[float] = []
    for name in feature_names:
        key = _normalize_key(name)
        if key in {"logvol", "logml", "strokevolumelogml"}:
            vals.append(_get_volume_value(row_like, "logvol"))
        elif key in {"volml", "vol", "volume", "volumeml"}:
            vals.append(_get_volume_value(row_like, "volml"))
        else:
            vals.append(_row_value_by_name(row_like, name))
    return np.array(vals, dtype=float).reshape(1, -1)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns: ROI, RF_predict (0/1), RF_predict_prob (float or 'N.A.'), QFV (float or 'N.A.').
    Avoid pandas FutureWarning by explicit object casting before fillna with strings.
    """
    df = df.copy()

    # Standardize probability column name
    if "RF_predict_prob" not in df.columns and "RF_prob" in df.columns:
        df["RF_predict_prob"] = df["RF_prob"]

    # Standardize prediction column to numeric
    if "RF_predict" in df.columns:
        df["RF_predict"] = df["RF_predict"].apply(_to_numeric_label)
    else:
        df["RF_predict"] = 0

    # Ensure ROI exists
    if "ROI" not in df.columns:
        raise ValueError("Expected column 'ROI' in AA table.")

    # Ensure QFV exists
    if "QFV" not in df.columns:
        df["QFV"] = "N.A."

    # Ensure RF_predict_prob exists
    if "RF_predict_prob" not in df.columns:
        df["RF_predict_prob"] = "N.A."

    # Make texty columns object dtype, fill, then infer
    for c in ["RF_predict_prob", "QFV"]:
        df[c] = df[c].astype("object")
    df = df.fillna("N.A.").infer_objects(copy=False)
    return df

def _drop_watershed_if_present(df: pd.DataFrame) -> pd.DataFrame:
    if "ROI" in df.columns:
        return df[df["ROI"].str.lower() != "watershed"].reset_index(drop=True)
    return df

def _pick_pos_and_maybe(df: pd.DataFrame):
    """
    Return (positive_ROIs, maybe_ROIs) with thresholds 0.5 and 0.4–0.5 band.
    If probability is 'N.A.', fall back to RF_predict.
    """
    pos, maybe = [], []
    for _, r in df.iterrows():
        roi = r["ROI"]
        p = r.get("RF_predict_prob", "N.A.")
        y = int(r.get("RF_predict", 0))

        if isinstance(p, str) and p != "N.A.":
            try:
                p = float(p)
            except Exception:
                p = "N.A."

        if p == "N.A.":
            if y == 1:
                pos.append(roi)
        else:
            if p > 0.5:
                pos.append(roi)
            elif 0.4 < p < 0.5:
                maybe.append(roi)
    return pos, maybe


# =========================== Model loading & inference ===========================

def load_model_with_preprocessing(model_path: Path):
    """
    Load a model or pipeline from a .pkl.
    Supports dict bundle {'model'|'pipeline', 'scaler', 'feature_selector', ...}
    or a bare estimator.
    """
    with open(model_path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and ('model' in obj or 'pipeline' in obj):
        model = obj.get('model', obj.get('pipeline'))
        scaler = obj.get('scaler')
        selector = obj.get('feature_selector') or obj.get('selector')
        return model, scaler, selector
    return obj, None, None

def _get_classes(model):
    if hasattr(model, "classes_"):
        return model.classes_
    try:
        last = model.steps[-1][1]
        return getattr(last, "classes_", None)
    except Exception:
        return None

def _prob_of_class(model, X, target_label=1):
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)[0]
    classes = _get_classes(model)
    if classes is None:
        return float(proba[1]) if len(proba) >= 2 else None
    for i, c in enumerate(classes):
        if c == target_label:
            return float(proba[i])
    return float(proba[1]) if len(proba) >= 2 else None

def _safe_predict(model, X):
    y = model.predict(X)[0]
    p = _prob_of_class(model, X, target_label=1)
    return int(y) if hasattr(y, "__int__") else y, p

# def _safe_predict(model, X):
#     """
#     Returns (label, prob_pos_or_None). Works with plain estimators or sklearn Pipelines.
#     """
#     y = model.predict(X)[0]
#     prob = None
#     if hasattr(model, "predict_proba"):
#         p = model.predict_proba(X)[0]
#         if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
#             prob = float(p[1])
#     return int(y) if hasattr(y, "__int__") else y, prob

def _build_predict_df_from_models(model_dir: str | Path, aa_type: str, roi_list: list[str],
                                  qfv_vector: np.ndarray, model_tag: str = "RF") -> pd.DataFrame:
    """
    Build a wide DF with rows [RF_predict, RF_prob] and ROI columns.
    """
    mdl_path = Path(model_dir) / aa_type
    pred_row = {"index": f"{model_tag}_predict"}
    prob_row = {"index": f"{model_tag}_prob"}

    if not mdl_path.exists():
        for r in roi_list:
            pred_row[r] = "N.A."
            prob_row[r] = "N.A."
        return pd.DataFrame([pred_row, prob_row])

    qfv_values = np.asarray(qfv_vector).reshape(-1)
    qfv_row = {"volml": float(qfv_values[0]) if qfv_values.size > 0 else 0.0}
    for i, roi_name in enumerate(roi_list):
        if (i + 1) < qfv_values.size:
            qfv_row[roi_name] = float(qfv_values[i + 1])
        else:
            qfv_row[roi_name] = 0.0

    files = list(mdl_path.glob("*.pkl"))

    for roi in roi_list:
        token = roi.replace(" ", "_")
        cand = [f for f in files if f.stem.lower().endswith(f"_{token.lower()}")]
        if not cand:
            cand = [f for f in files if token.lower() in f.stem.lower()]
        if not cand:
            pred_row[roi] = "N.A."
            prob_row[roi] = "N.A."
            continue

        try:
            with open(cand[0], "rb") as f:
                model_obj = pickle.load(f)
            feature_names = model_obj.get("feature_cols") if isinstance(model_obj, dict) else None
            model, scaler, selector = load_model_with_preprocessing(cand[0])

            X = _build_x_from_feature_names(qfv_row, feature_names) if feature_names else np.asarray(qfv_vector).reshape(1, -1)
            if scaler is not None:
                if hasattr(scaler, 'feature_names_in_'):
                    X = pd.DataFrame(X, columns=scaler.feature_names_in_)
                X = scaler.transform(X)
            if selector is not None:
                X = selector.transform(X)
            y, p = _safe_predict(model, X)
            pred_row[roi] = "AFFECTED" if (str(y) == "1") else "NORMAL"
            prob_row[roi] = ("N.A." if p is None else round(float(p), 3))
        except Exception as e:
            pred_row[roi] = f"ERROR: {e}"
            prob_row[roi] = "N.A."

    return pd.DataFrame([pred_row, prob_row])


def _finalize_old_style(df_wide: pd.DataFrame, qfv_values_by_roi: dict) -> pd.DataFrame:
    """
    Append a 'QFV' row, then return ROI-major table:
    columns -> ['ROI', 'RF_predict', 'RF_prob', 'QFV'] (then normalized later)
    """
    add = {"index": "QFV"}
    add.update(qfv_values_by_roi)
    df2 = pd.concat([df_wide, pd.DataFrame([add])], ignore_index=True)
    result = (
        df2.set_index("index").T
           .rename_axis("ROI")
           .reset_index()
           .fillna("N.A.")
           .infer_objects(copy=False)
    )
    # rename RF_prob -> RF_predict_prob to match downstream usage
    if "RF_prob" in result.columns and "RF_predict_prob" not in result.columns:
        result = result.rename(columns={"RF_prob": "RF_predict_prob"})
    return result

def _build_predict_df_from_models_exact(
    model_dir: str | Path,
    aa_type: str,
    roi_to_filename: dict[str, str],
    X_vec: np.ndarray,
    model_tag: str = "RF",
) -> pd.DataFrame:
    mdl_path = Path(model_dir) / aa_type
    pred_row = {"index": f"{model_tag}_predict"}
    prob_row = {"index": f"{model_tag}_prob"}

    if not mdl_path.exists():
        for roi in roi_to_filename:
            pred_row[roi] = "N.A."
            prob_row[roi] = "N.A."
        return pd.DataFrame([pred_row, prob_row])

    X = np.asarray(X_vec).reshape(1, -1)

    for roi, fname in roi_to_filename.items():
        pkl_path = mdl_path / fname
        if not pkl_path.exists():
            pred_row[roi] = "N.A."
            prob_row[roi] = "N.A."
            continue

        try:
            model, scaler, selector = load_model_with_preprocessing(pkl_path)

            X_use = X
            if scaler is not None:
                if hasattr(scaler, "feature_names_in_"):
                    X_use = pd.DataFrame(X_use, columns=scaler.feature_names_in_)
                X_use = scaler.transform(X_use)
            if selector is not None:
                X_use = selector.transform(X_use)

            y, p = _safe_predict(model, X_use)

            pred_row[roi] = "AFFECTED" if (str(y) == "1") else "NORMAL"
            prob_row[roi] = ("N.A." if p is None else round(float(p), 3))

        except Exception as e:
            pred_row[roi] = f"ERROR: {e}"
            prob_row[roi] = "N.A."

    return pd.DataFrame([pred_row, prob_row])

# =============================== AA report tables ===============================

def get_AA_report_list(
    SubjDir: str,
    AAModelsDir: str,
    QFV_list: List[np.ndarray],
    *,
    report_dir: str | None = None,
    subject_id: str | None = None,
    hydro_feature_df: pd.DataFrame | None = None
) -> List[pd.DataFrame]:
    """
    Build [Vascular_df, Lobe_df, Hydro_df, Aspect_df] in legacy shape.
    QFV_list order expected:
      [0]=Vascular (logvol + 9 terms),
      [1]=Lobe (logvol + 11 lobes),
      [2]=ASPECTS (volml + 10 parts),
      [3]=Ventricles (volml + 20),
      optional others afterward.
    """

    # --- Vascular ---
    vas_roi = ['ACA', 'MCA', 'PCA', 'cerebellar', 'basilar',
               'Lenticulostriate', 'Choroidal & Thalamoperfurating', 'watershed']
    vas_qfv_vals = dict(zip(vas_roi, np.asarray(QFV_list[0][1:]).tolist()))  # skip logvol
    vas_df = _build_predict_df_from_models(AAModelsDir, "vascular", vas_roi, QFV_list[0])
    Vascular_predict_df = _finalize_old_style(vas_df, vas_qfv_vals)

    # --- Lobe ---
    lobe_roi = ['basal ganglia', 'deep white matter', 'cerebellum', 'frontal', 'insula',
                'internal capsule', 'brainstem', 'occipital', 'parietal', 'temporal', 'thalamus']
    lobe_qfv_vals = dict(zip(lobe_roi, np.asarray(QFV_list[1][1:]).tolist()))  # skip logvol
    lobe_df = _build_predict_df_from_models(AAModelsDir, "lobe", lobe_roi, QFV_list[1])
    Lobe_predict_df = _finalize_old_style(lobe_df, lobe_qfv_vals)

    # --- ASPECTS ---
    # Keep your original mapping quirk (M6/M4 swapped in CSV)
    aspect_roi_for_models = ['Caudate', 'lentiform', 'IC', 'insula', 'M1', 'M2', 'M3', 'M6', 'M5', 'M4']
    aspect_roi_for_qfv    = ['Caudate', 'lentiform', 'IC', 'insula', 'M1', 'M2', 'M3', 'M6', 'M5', 'M4']
    aspect_qfv_vals = dict(zip(aspect_roi_for_qfv, np.asarray(QFV_list[2][1:]).tolist()))  # skip volml
    aspect_df = _build_predict_df_from_models(AAModelsDir, "aspect", aspect_roi_for_models, QFV_list[2])
    Aspect_predict_df = _finalize_old_style(aspect_df, aspect_qfv_vals)

        # --- Hydro ---
    hydro_models_dir = Path(AAModelsDir) / "hydro"
    hydro_model_path = hydro_models_dir / "RF_hydro_hydrocephalus.pkl"
    hydro_roi = ["hydrocephalus"]

    if not hydro_model_path.exists():
        base = pd.DataFrame(
            {"ROI": hydro_roi, "RF_predict": ["N.A."], "RF_predict_prob": ["N.A."], "QFV": ["N.A."]}
        )
        Hydro_predict_df = base
    else:
        try:
            with open(hydro_model_path, "rb") as f:
                model_data = pickle.load(f)

            # New bundle format: {"pipeline": ..., "feature_cols": [...]}
            # Backward compatible with older keys.
            pipeline = model_data.get("pipeline", model_data.get("model", model_data))
            feature_names = model_data.get("feature_cols", model_data.get("raw_feature_names", None))
            if feature_names is None:
                raise ValueError("Hydro model pkl missing 'feature_cols' (or legacy 'raw_feature_names').")

            # Build features from reporting CSVs (preferred and reproducible)
            if hydro_feature_df is not None:
                # Caller can pass a ready DF with all columns.
                X = hydro_feature_df.reindex(columns=feature_names, fill_value=0.0).copy()
            elif (report_dir is not None) and (subject_id is not None):
                # Load three QFV tables and reconstruct the exact feature vector.
                df_vas = pd.read_csv(f"{report_dir}/{subject_id}_VASCULAR_QFV.csv")
                df_vas.columns = ['subject_id', 'logvol', 'ACA', 'MCA', 'PCA', 'cerebellar', 'basilar',
                                  'Lenticulostriate', 'Choroidal & Thalamoperfurating', 'watershed']

                df_lobe = pd.read_csv(f"{report_dir}/{subject_id}_LOBE_QFV.csv")
                df_lobe.columns = ['subject_id', 'logvol', 'basal ganglia', 'deep white matter', 'cerebellum',
                                   'frontal', 'insula', 'internal capsule', 'brainstem', 'occipital',
                                   'parietal', 'temporal', 'thalamus']

                df_vent = pd.read_csv(f"{report_dir}/{subject_id}_VENTRICLES_QFV.csv")
                df_vent.columns = ['subject_id', 'volml',
                                   'LVOR1','LVOR2','LVOR3','LVOR4','LVOR5','LVOR6','LVOR7','LVOR8','LVOR9','LVOR10',
                                   'LVIR1','LVIR2','LVIR3','LVIR4','LVIR5','LVIR6','LVIR7','LVIR8','LVIR9','LVIR10']

                # Ventricles: use LV features only (drop subject_id, volml)
                vent_feats = df_vent.drop(columns=['subject_id', 'volml']).iloc[0]

                # logvol: in your system logvol and volml can be aliases.
                # Use vascular 'logvol' column, which may actually be volml in some files (renamed above).
                logvol = float(df_vas.loc[0, 'logvol'])

                # Lobe: only the four deep-structure features used by the hydro model
                lobe_feats = df_lobe.loc[0, ['basal ganglia', 'deep white matter', 'internal capsule', 'thalamus']]

                # Compose one feature vector
                feats = pd.concat([vent_feats, pd.Series({'logvol': logvol}), lobe_feats])

                # If model expects 'volml' as well, mirror it from logvol for compatibility
                if "volml" in feature_names and "volml" not in feats.index:
                    feats = pd.concat([feats, pd.Series({'volml': logvol})])

                X = pd.DataFrame([feats]).reindex(columns=feature_names, fill_value=0.0)
            else:
                # Final fallback (not recommended): attempt from QFV_list
                vec = np.asarray(QFV_list[3]).reshape(-1)
                if vec.shape[0] != len(feature_names):
                    raise ValueError(
                        f"Hydro fallback features length {vec.shape[0]} != expected {len(feature_names)}"
                    )
                X = pd.DataFrame([vec], columns=feature_names)

            y = pipeline.predict(X)[0]
            p = None
            if hasattr(pipeline, "predict_proba"):
                # proba = pipeline.predict_proba(X)[0]
                # if len(proba) >= 2:
                #     p = float(proba[1])
                p = _prob_of_class(pipeline, X, target_label=1)

            pred_label = "AFFECTED" if int(y) == 1 else "NORMAL"
            Hydro_predict_df = pd.DataFrame(
                {"ROI": ["hydrocephalus"],
                 "RF_predict": [pred_label],
                 "RF_predict_prob": ["N.A." if p is None else round(p, 3)],
                 "QFV": ["N.A."]}  # composite, no single QFV scalar
            ).infer_objects(copy=False)
        except Exception as e:
            Hydro_predict_df = pd.DataFrame(
                {"ROI": ["hydrocephalus"], "RF_predict": [f"ERROR: {e}"], "RF_predict_prob": ["N.A."], "QFV": ["N.A."]}
            )

    return [Vascular_predict_df, Lobe_predict_df, Hydro_predict_df, Aspect_predict_df]


# ============================== Report text & tables ==============================
def _run_aspectpc_models(AAModelsDir: str, pca_qfv: np.ndarray, logvol: float):
    import joblib
    from pathlib import Path
    import numpy as np

    model_dir = Path(AAModelsDir) / "aspectpc"

    roi_to_file = {
        "PCA": "RF_aspectpc_PCA.pkl",
        "Thalamus": "RF_aspectpc_Thalamus.pkl",
        "cerebellum": "RF_aspectpc_cerebellum.pkl",
        "pons": "RF_aspectpc_pons.pkl",
        "midbrain": "RF_aspectpc_midbrain.pkl",
    }

    # Display only
    roi_display = {
        "PCA": "PCA",
        "Thalamus": "Thalamus",
        "cerebellum": "Cerebellum",
        "pons": "Pons",
        "midbrain": "Midbrain",
    }

    pca_qfv = np.asarray(pca_qfv).reshape(-1)

    qfv_row = {
        "volml": float(pca_qfv[0]) if pca_qfv.size > 0 else 0.0,
        "PCA": float(pca_qfv[1]) if pca_qfv.size > 1 else 0.0,
        "Thalamus": float(pca_qfv[2]) if pca_qfv.size > 2 else 0.0,
        "cerebellum": float(pca_qfv[3]) if pca_qfv.size > 3 else 0.0,
        "Cerebellum": float(pca_qfv[3]) if pca_qfv.size > 3 else 0.0,
        "Pons": float(pca_qfv[4]) if pca_qfv.size > 4 else 0.0,
        "Midbrain": float(pca_qfv[5]) if pca_qfv.size > 5 else 0.0,
        "pons y": float(pca_qfv[4]) if pca_qfv.size > 4 else 0.0,
        "midbrain y": float(pca_qfv[5]) if pca_qfv.size > 5 else 0.0,
        "logvol": float(logvol),
    }

    idx_map = {"PCA": 1, "Thalamus": 2, "cerebellum": 3, "pons": 4, "midbrain": 5}

    rows = []

    def _calibrated_probability(qfv_feature: str, prob: float | None) -> float | None:
        if prob is None:
            return None
        q = _row_value_by_name(qfv_row, qfv_feature)
        if q <= 0.01:
            return 0.0
        if q >= 0.1:
            return float(max(prob, 0.95))
        return float(prob * (q / 0.1))

    def _calibrated_prediction(qfv_feature: str, raw_pred: int, prob_cal: float | None) -> int:
        q = _row_value_by_name(qfv_row, qfv_feature)
        if q <= 0.01:
            return 0
        if q >= 0.1:
            return 1
        if prob_cal is None:
            return int(raw_pred)
        return int(prob_cal >= 0.5)

    # 1) ROI models (classification likely)
    for roi, fname in roi_to_file.items():
        qfv_val = float(pca_qfv[idx_map[roi]])
        fpath = model_dir / fname
        if not fpath.exists():
            rows.append((roi_display.get(roi, roi), "N.A.", "N.A.", qfv_val))
            continue

        try:
            model = joblib.load(fpath)
            feature_names = model.get("feature_cols") if isinstance(model, dict) else None

            # If saved as dict bundle
            if isinstance(model, dict) and ("model" in model or "pipeline" in model):
                clf = model.get("model", model.get("pipeline"))
                scaler = model.get("scaler")
                selector = model.get("feature_selector") or model.get("selector")
                X_use = _build_x_from_feature_names(qfv_row, feature_names) if feature_names else np.array(
                    [[qfv_row["PCA"], qfv_row["Thalamus"], qfv_row["cerebellum"], qfv_row["Pons"], qfv_row["Midbrain"], qfv_row["logvol"]]],
                    dtype=float,
                )
                if scaler is not None:
                    X_use = scaler.transform(X_use)
                if selector is not None:
                    X_use = selector.transform(X_use)
            else:
                clf = model
                X_use = np.array(
                    [[qfv_row["PCA"], qfv_row["Thalamus"], qfv_row["cerebellum"], qfv_row["Pons"], qfv_row["Midbrain"], qfv_row["logvol"]]],
                    dtype=float,
                )

            y, p = _safe_predict(clf, X_use)
            qfv_feature_name = roi_display.get(roi, roi)
            p_cal = _calibrated_probability(qfv_feature_name, (None if p is None else float(p)))
            y_cal = _calibrated_prediction(qfv_feature_name, int(y), p_cal)

            # classification output
            try:
                pred = int(y_cal)
            except Exception:
                pred = y_cal

            prob = "N.A." if p_cal is None else round(float(p_cal), 3)
            rows.append((roi_display.get(roi, roi), pred, prob, qfv_val))

        except Exception as e:
            rows.append((roi_display.get(roi, roi), f"ERROR: {e}", "N.A.", qfv_val))

    # 2) Total score model (GB, likely regression)
    #total_path = model_dir / "aspectpc_gb_RF.pkl"
    # if total_path.exists():
    #     try:
    #         total_obj = joblib.load(total_path)

    #         if isinstance(total_obj, dict) and ("model" in total_obj or "pipeline" in total_obj):
    #             reg = total_obj.get("model", total_obj.get("pipeline"))
    #             scaler = total_obj.get("scaler")
    #             selector = total_obj.get("feature_selector") or total_obj.get("selector")
    #             X_use = X
    #             if scaler is not None:
    #                 X_use = scaler.transform(X_use)
    #             if selector is not None:
    #                 X_use = selector.transform(X_use)
    #         else:
    #             reg = total_obj
    #             X_use = X

    #         y_total = reg.predict(X_use)[0]
    #         total_pred = round(float(y_total), 3)

    #         rows.append(("Total PCA-ASPECTS", total_pred, "N.A.", "N.A."))
    #     except Exception as e:
    #         rows.append(("Total PCA-ASPECTS", f"ERROR: {e}", "N.A.", "N.A."))
    # else:
    #     rows.append(("Total PCA-ASPECTS", "N.A.", "N.A.", "N.A."))

    # directly sum all other aspects prediction, and then use total score = 10 - sum(PCA + Thalamus + Cerebellum + 2* (Pons + Midbrain))
    pred_map = {r[0]: (int(r[1]) if str(r[1]).isdigit() else 0) for r in rows}
    rows.append(("Total PCA-ASPECTS", 10 - (pred_map.get("PCA") + pred_map.get("Thalamus") + pred_map.get("Cerebellum") + 2*(pred_map.get("Pons") + pred_map.get("Midbrain"))), "N.A.", "N.A."))
    
    return rows

def _should_write_aspectpc(vas_df: pd.DataFrame) -> bool:
    trigger_rois = ["PCA", "cerebellar", "basilar"]
    for r in trigger_rois:
        hit = vas_df[vas_df["ROI"] == r]
        if hit.empty:
            continue
        v = hit["RF_predict"].iloc[0]
        s = str(v).strip()
        if s in ["1", "AFFECTED", "Affected", "affected", "True", "true"]:
            return True
        try:
            if int(v) != 0:
                return True
        except Exception:
            pass
    return False

def gen_radiological_report(SubjDir: str, SubjID: str, side_vol: Tuple[float, float], predict_vol_ml: float,
                            AAModelsDir: str, QFV_list: List[np.ndarray], hydro_flag: bool = False, pca_qfv: np.ndarray | None = None) -> bool:
    """
    Generate the legacy text report and return final hydro flag used.
    """
    AA_df_list = get_AA_report_list(SubjDir, AAModelsDir, QFV_list,
                                    report_dir=SubjDir, subject_id=SubjID)

    vas_df_raw = _ensure_columns(AA_df_list[0])
    lobe_df    = _ensure_columns(AA_df_list[1])
    hydro_df   = _ensure_columns(AA_df_list[2])
    aspect_df  = _ensure_columns(AA_df_list[3])

    vas_df = _drop_watershed_if_present(vas_df_raw)

    pca_rows = None
    if pca_qfv is not None:
        try:
            logvol = float(QFV_list[0][0])  
        except Exception:
            logvol = 0.0
        pca_rows = _run_aspectpc_models(AAModelsDir, pca_qfv, logvol)

    write_pca = _should_write_aspectpc(vas_df) if vas_df is not None else False

    # Stacked table in legacy order (arterial, lobe, hydro, aspects)
    blocks = [vas_df, lobe_df, hydro_df]
    start_aspects_idx = sum(len(b) for b in blocks)
    final_df = pd.concat(blocks + [aspect_df], ignore_index=True).fillna("N.A.")

    # Collect positive/maybe sets
    Arterial_infarct_ROI, Arterial_infarct_ROI_s = _pick_pos_and_maybe(vas_df)
    Structure_infarct_ROI, Structure_infarct_ROI_s = _pick_pos_and_maybe(lobe_df)
    Hydro_infarct_ROI, Hydro_infarct_ROI_s = _pick_pos_and_maybe(hydro_df)

    # QC indices (defensive)
    try:
        QCi = float(np.median(QFV_list[3][1:]).round(3))
    except Exception:
        QCi = "N.A."
    try:
        QCo = float(np.median(QFV_list[4][1:]).round(3))
    except Exception:
        QCo = "N.A."

    # Lesion presence & laterality
    if predict_vol_ml < 0.02:
        lesion_detect = False
    else:
        lesion_detect = True
        r = (min(side_vol) / (max(side_vol) + 1e-5)) > 0.1
        if r:
            side_label = 'left and right'
        elif side_vol[0] >= side_vol[1]:
            side_label = 'left'
        else:
            side_label = 'right'

    ReportTxt_pth = os.path.join(SubjDir, SubjID + '_automatic_radiological_report.txt')
    os.makedirs(SubjDir, exist_ok=True)

    with open(ReportTxt_pth, 'w') as f:
        f.write('-'*35 + 'Automatic Radiological Report' + '-'*35 + '\n')

        if lesion_detect:
            if side_label == 'left and right':
                f.write('Area of restricted diffusion is bilateral, within both right and left'
                    + f' brain hemisphere, with {predict_vol_ml:.2f} ml, \n')
            else:
                f.write('Area of restricted diffusion is unilateral, within the ' + side_label
                        + f' brain hemisphere, with {predict_vol_ml:.2f} ml, \n')

            # Arterial text
            if len(Arterial_infarct_ROI) != 0:
                f.write('in the territory of ' + ', '.join([_rn(_) for _ in Arterial_infarct_ROI]))
                if len(Arterial_infarct_ROI_s) != 0:
                    f.write(', and possibly ' + ', '.join([_rn(_) for _ in Arterial_infarct_ROI_s]))
            elif len(Arterial_infarct_ROI_s) != 0:
                f.write('in the territory of ' + ', '.join([_rn(_) for _ in Arterial_infarct_ROI_s]))
            f.write('.\n')

            if (len(Arterial_infarct_ROI)==0 and len(Arterial_infarct_ROI_s)==0 and
                len(Structure_infarct_ROI)==0 and len(Structure_infarct_ROI_s)==0):
                f.write('No major arterial infarcted!')

            # Structure text
            f.write('The area involves the following brain regions: ')
            if len(Structure_infarct_ROI) != 0:
                f.write(', '.join([_rn(_) for _ in Structure_infarct_ROI]))
                if len(Structure_infarct_ROI_s) != 0:
                    f.write(', and possibly ')
                    f.write(', '.join([_rn(_) for _ in Structure_infarct_ROI_s]))
            elif len(Structure_infarct_ROI_s) != 0:
                f.write(', '.join([_rn(_) for _ in Structure_infarct_ROI_s]))
            f.write('.\n')

            if (len(Arterial_infarct_ROI)==0 and len(Arterial_infarct_ROI_s)==0 and
                len(Structure_infarct_ROI)==0 and len(Structure_infarct_ROI_s)==0):
                f.write('No major anatomical structure infarcted!')

            # Hydro text
            final_hydro_flag = (len(Hydro_infarct_ROI)!=0) or bool(hydro_flag)
            f.write('There is hydrocephalus.\n' if final_hydro_flag else 'There is no hydrocephalus.\n')

            # MCA / ASPECTS total if MCA flagged
            if ('MCA' in Arterial_infarct_ROI) or ('MCA' in Arterial_infarct_ROI_s):
                f.write('The predicted MCA – ASPECTS is ')
                try:
                    aspects_block = final_df.iloc[start_aspects_idx:].copy()
                    injured = int(aspects_block["RF_predict"].sum())
                    ASPECTS_total = 10 - injured
                except Exception:
                    ASPECTS_total = 10
                f.write(str(int(ASPECTS_total)) + '.\n')

            # ---------- Tables ----------
            # Arterial
            f.write('\n' + '-'*100 + '\n')
            f.write(f"{'Arterial territories affected':^100}\n")
            f.write('-'*100 + '\n')
            f.write(f"{'ROI':^30}{'predict':^20}{'predict probability':^20}{'QFV(portion of region injured)':^20}\n")
            for i in range(0, len(vas_df)):
                ROI  = final_df.loc[i, 'ROI']
                RFp  = int(final_df.loc[i, 'RF_predict'])
                RFpb = final_df.loc[i, 'RF_predict_prob']
                QFV  = final_df.loc[i, 'QFV']
                f.write(f"{_rn(ROI):^30}{str(RFp):^20}{str(RFpb):^20}{str(QFV):^20}\n")

            # Lobe
            start = len(vas_df)
            end   = start + len(lobe_df)
            f.write('\n' + '-'*100 + '\n')
            f.write(f"{'Anatomical structures affected':^100}\n")
            f.write('-'*100 + '\n')
            f.write(f"{'ROI':^30}{'predict':^20}{'predict probability':^20}{'QFV(portion of region injured)':^20}\n")
            for i in range(start, end):
                ROI  = final_df.loc[i, 'ROI']
                RFp  = int(final_df.loc[i, 'RF_predict'])
                RFpb = final_df.loc[i, 'RF_predict_prob']
                QFV  = final_df.loc[i, 'QFV']
                f.write(f"{_rn(ROI):^30}{str(RFp):^20}{str(RFpb):^20}{str(QFV):^20}\n")

            # Hydro
            hstart = end
            hend   = hstart + len(hydro_df)
            f.write('\n' + '-'*100 + '\n')
            f.write(f"{'Hydrocephalus':^100}\n")
            f.write('-'*100 + '\n')
            f.write(f"{'ROI':^30}{'predict':^20}{'predict probability':^20}{'QFV(portion of region injured)':^20}\n")
            for i in range(hstart, hend):
                ROI  = final_df.loc[i, 'ROI']
                RFp  = int(final_df.loc[i, 'RF_predict'])
                RFpb = final_df.loc[i, 'RF_predict_prob']
                QFV  = final_df.loc[i, 'QFV']
                f.write(f"{_rn(ROI):^30}{str(RFp):^20}{str(RFpb):^20}{str(QFV):^20}\n")

            # ASPECTS (only if MCA flagged)
            if ('MCA' in Arterial_infarct_ROI) or ('MCA' in Arterial_infarct_ROI_s):
                astart = hend
                f.write('\n' + '-'*100 + '\n')
                f.write(f"{'ASPECTS':^100}\n")
                f.write('-'*100 + '\n')
                f.write(f"{'ROI':^30}{'predict':^20}{'predict probability':^20}{'QFV(portion of region injured)':^20}\n")
                for i in range(astart, len(final_df)):
                    ROI  = final_df.loc[i, 'ROI']
                    RFp  = int(final_df.loc[i, 'RF_predict'])
                    RFpb = final_df.loc[i, 'RF_predict_prob']
                    QFV  = final_df.loc[i, 'QFV']
                    f.write(f"{_rn(ROI):^30}{str(RFp):^20}{str(RFpb):^20}{str(QFV):^20}\n")
            
            if write_pca and (pca_rows is not None):
                f.write("\n" + "-" * 100 + "\n")
                f.write(f"{'PC-ASPECTS':^100}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'ROI':^30}{'predict':^20}{'predict probability':^20}{'QFV(portion of region injured)':^30}\n")

                for roi, pred, prob, qfv_val in pca_rows:
                    #f.write(f"{str(roi):^30}{str(pred):^20}{str(prob):^20}{str(round(float(qfv_val),3)):^30}\n")
                    try:
                        qfv_str = str(round(float(qfv_val), 3))
                    except Exception:
                        qfv_str = str(qfv_val)

                    f.write(f"{str(roi):^30}{str(pred):^20}{str(prob):^20}{qfv_str:^30}\n")

        else:
            # No lesion detected — still report hydro
            final_hydro_flag = (len(Hydro_infarct_ROI)!=0) or bool(hydro_flag)
            f.write('There is hydrocephalus.\n' if final_hydro_flag else 'There is no hydrocephalus.\n')

    return (len(Hydro_infarct_ROI)!=0) or bool(hydro_flag)
