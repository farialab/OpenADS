from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# Safe headless plotting for servers/CI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

import shap
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------- #
# Utilities: model + data shape #
# ----------------------------- #

def _normalize_aa_type(aa_type: str) -> str:
    """Normalize AA type aliases for model folder compatibility."""
    t = str(aa_type).lower()
    aliases = {
        "aspects": "aspect",
    }
    return aliases.get(t, t)

def _resolve_model_path(
    models_root: Union[str, Path],
    aa_type: str,
    model_name: str,
    roi: str,
    filename_token: Optional[str] = None,
) -> Path:
    """
    Resolve model path based on naming conventions.
    """
    models_root = Path(models_root)
    token = filename_token or roi.replace(" ", "_")
    aa_dir = models_root / aa_type

    candidates: List[Path] = []

    # Legacy/default naming
    candidates.append(aa_dir / f"{model_name}_{aa_type}_{token}.pkl")

    # Atlas-specific variants
    if aa_type in {"vascular", "lobe", "aspect"}:
        # Current files for these atlases use *_label.pkl
        candidates.append(aa_dir / f"{model_name}_{aa_type}_{token}_label.pkl")

    if aa_type == "aspectpc":
        # Support renamed PCA-ASPECTS model files.
        candidates.append(aa_dir / f"{model_name}_pcaspect_{token}.pkl")

    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Model not found for ROI='{roi}'. Tried: {', '.join(str(p) for p in candidates)}")

def _check_model_exists(
    models_root: Union[str, Path],
    aa_type: str, 
    model_name: str,
    roi: str,
    filename_token: Optional[str] = None
) -> bool:
    """
    Check if model exists for given ROI without raising an exception.
    """
    models_root = Path(models_root)
    token = filename_token or roi.replace(" ", "_")
    aa_dir = models_root / aa_type

    candidates: List[Path] = []
    candidates.append(aa_dir / f"{model_name}_{aa_type}_{token}.pkl")
    if aa_type in {"vascular", "lobe", "aspect"}:
        candidates.append(aa_dir / f"{model_name}_{aa_type}_{token}_label.pkl")
    if aa_type == "aspectpc":
        candidates.append(aa_dir / f"{model_name}_pcaspect_{token}.pkl")

    return any(path.exists() for path in candidates)

def _load_estimator(model_path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a saved estimator or pipeline. Returns (estimator, meta_dict).
    """
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        est = obj.get("estimator") or obj.get("clf") or obj.get("model") or obj.get("pipeline")
        if not est:
            raise ValueError(f"Invalid model file at {model_path}: missing estimator.")
        meta = dict(obj.get("meta", {}))
        if "feature_cols" in obj and "feature_cols" not in meta:
            meta["feature_cols"] = obj.get("feature_cols")
        return est, meta
    return obj, {}


# def _feature_order_from_estimator(estimator: Any) -> Optional[List[str]]:
#     """
#     Extract feature order from a trained estimator or pipeline.
#     """
#     try:
#         return list(getattr(estimator, "feature_names_in_", None) or 
#                     getattr(estimator.steps[-1][1], "feature_names_in_", None))
#     except AttributeError:
#         return None

def _feature_order_from_estimator(estimator: Any) -> Optional[List[str]]:
    """
    Extract feature order from a trained estimator or pipeline.
    """
    direct = getattr(estimator, "feature_names_in_", None)
    if direct is not None:
        try:
            return list(direct)
        except TypeError:
            return None

    steps = getattr(estimator, "steps", None)
    if steps:
        try:
            last_est = steps[-1][1]
            nested = getattr(last_est, "feature_names_in_", None)
            if nested is not None:
                return list(nested)
        except Exception:
            return None

    return None

def _conform_features(QFV, estimator, feature_names, meta=None) -> pd.DataFrame:
    meta = meta or {}
    feature_names = list(feature_names) if feature_names is not None else None

    if isinstance(QFV, pd.DataFrame):
        if len(QFV) != 1:
            raise ValueError(f"Expected single row; got shape {QFV.shape}")
        x_df = QFV.copy()
    elif isinstance(QFV, pd.Series):
        x_df = QFV.to_frame().T
    else:
        arr = np.asarray(QFV).reshape(1, -1)
        cols = feature_names or [f"feat_{i}" for i in range(arr.shape[1])]
        if feature_names and len(feature_names) != arr.shape[1]:
            raise ValueError(f"feature_names length {len(feature_names)} != QFV length {arr.shape[1]}")
        x_df = pd.DataFrame(arr, columns=cols)

    order = (
        meta.get("raw_features")
        or meta.get("feature_names")
        or _feature_order_from_estimator(estimator)
        or (feature_names if feature_names else list(x_df.columns))
    )

    missing = [col for col in order if col not in x_df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return x_df.reindex(columns=order)

def _pick_output_index(clf, proba: Optional[np.ndarray]) -> int:
    """
    Determine the output index to visualize:
    - Binary: prefer class label 1 if present, else argmax.
    - Multiclass: use argmax probability.
    - No proba: default to 1 for binary, else 0.
    """
    if proba is not None and proba.ndim == 2 and proba.shape[0] == 1:
        return int(np.argmax(proba[0])) if proba.shape[1] > 1 else 1

    classes = getattr(clf, "classes_", None)
    if classes is not None and len(classes) == 2:
        return int(np.where(classes == 1)[0][0]) if 1 in classes else 1

    return 0

def _select_output_explanation(
    sv: shap.Explanation, X: pd.DataFrame, out_idx: int
) -> shap.Explanation:
    """
    Normalize a SHAP Explanation to a single-output Explanation by slicing the
    proper axis. Handles shapes like:
      - sv.values: (n_samples, n_features)
      - sv.values: (n_samples, n_features, n_outputs)
      - sv.values: (n_samples, n_outputs, n_features)
    Also supports the already-indexed case: sv[0].values ~ (n_features, n_outputs).
    """
    vals = sv.values
    nfeat = X.shape[1]

    # 3D cases (most common for multi-output)
    if vals.ndim == 3:
        n_samp = vals.shape[0]
        if vals.shape[1] == nfeat:
            # (n_samples, n_features, n_outputs)
            return sv[:, :, out_idx]
        if vals.shape[2] == nfeat:
            # (n_samples, n_outputs, n_features)
            return sv[:, out_idx, :]

    # 2D cases: could be (n_samples, n_features) [already single-output]
    # or after sv = sv[0], could be (n_features, n_outputs)
    if vals.ndim == 2:
        if vals.shape[0] == nfeat:
            # (n_features, n_outputs) — need per-output slice
            return sv[:, out_idx]
        # else: (n_samples, n_features) — already single-output
        return sv

    # Fallback: return as-is
    return sv


def _split_pipeline(model: Any) -> Tuple[Any, Any]:
    """
    Return (preprocess, final_estimator) where preprocess can be None.
    """
    if hasattr(model, "steps") and isinstance(getattr(model, "steps", None), list):
        if len(model.steps) == 1:
            return None, model.steps[0][1]
        return model[:-1], model.steps[-1][1]
    return None, model


def _conform_aspectpc_features(
    QFV: Union[np.ndarray, Sequence[float], pd.Series, pd.DataFrame],
    feature_names: Optional[Sequence[str]],
) -> pd.DataFrame:
    names = list(feature_names) if feature_names is not None else None
    if names is None or len(names) == 0:
        raise ValueError("ASPECTPC feature_names are required")

    if isinstance(QFV, pd.DataFrame):
        if len(QFV) != 1:
            raise ValueError(f"Expected single row; got shape {QFV.shape}")
        x_df = QFV.copy()
    elif isinstance(QFV, pd.Series):
        x_df = QFV.to_frame().T
    else:
        arr = np.asarray(QFV).reshape(1, -1)
        if len(names) != arr.shape[1]:
            raise ValueError(f"feature_names length {len(names)} != QFV length {arr.shape[1]}")
        x_df = pd.DataFrame(arr, columns=names)

    missing = [col for col in names if col not in x_df.columns]
    if missing:
        raise KeyError(f"Missing required ASPECTPC columns: {missing}")

    return x_df.reindex(columns=names).apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def _add_aspectpc_engineered_features(x_df: pd.DataFrame) -> pd.DataFrame:
    out = x_df.copy()
    def _col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0).astype(float)
        return pd.Series(0.0, index=out.index, dtype=float)

    vol = _col("volml")
    logvol = np.log10(np.clip(vol.to_numpy(dtype=float), 1e-6, None))
    out["logvol"] = logvol
    out["sqrtvol"] = np.sqrt(np.clip(vol.to_numpy(dtype=float), 0.0, None))
    out["PCA_x_vol"] = _col("PCA") * out["logvol"]
    out["Thalamus_x_vol"] = _col("Thalamus") * out["logvol"]
    out["Cerebellum_x_vol"] = _col("Cerebellum") * out["logvol"]
    out["Pons_x_vol"] = _col("Pons") * out["logvol"]
    out["Midbrain_x_vol"] = _col("Midbrain") * out["logvol"]
    return out


def shap_plot_aspectpc(
    QFV: Union[np.ndarray, Sequence[float], pd.Series, pd.DataFrame],
    ROI: str,
    models_root: Union[str, Path],
    Model_name: str,
    feature_names: Optional[Sequence[str]],
    filename_token: Optional[str] = None,
) -> Optional[matplotlib.figure.Figure]:
    """
    ASPECTPC-specific SHAP plotting path that supports sklearn Pipeline models.
    """
    model_path = _resolve_model_path(models_root, "aspectpc", Model_name, ROI, filename_token)
    clf, meta = _load_estimator(model_path)
    X = _conform_aspectpc_features(QFV, feature_names)
    X = _add_aspectpc_engineered_features(X)
    model_feature_cols = meta.get("feature_cols")
    if model_feature_cols:
        X = X.reindex(columns=list(model_feature_cols), fill_value=0.0)

    proba = None
    try:
        if hasattr(clf, "predict_proba"):
            proba = np.asarray(clf.predict_proba(X))
    except Exception:
        proba = None

    out_idx = _pick_output_index(clf, proba)
    pre, est = _split_pipeline(clf)
    X_disp = X.round(4)
    X_model = pre.transform(X_disp) if pre is not None else X_disp

    try:
        est_name = est.__class__.__name__
        if est_name in {"RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"}:
            explainer = shap.TreeExplainer(est, feature_names=list(X.columns))
            sv = explainer(X_model if pre is not None else X_disp.values)
        elif est_name in {"LogisticRegression", "SGDClassifier"}:
            # For linear models, explain the fitted estimator on transformed input space.
            x_lin = X_model if isinstance(X_model, pd.DataFrame) else pd.DataFrame(X_model, columns=list(X.columns))
            explainer = shap.LinearExplainer(est, x_lin, feature_names=list(x_lin.columns))
            sv = explainer(x_lin)
        else:
            # Generic fallback explainer
            x_ref = X_model if isinstance(X_model, pd.DataFrame) else pd.DataFrame(X_model, columns=list(X.columns))
            explainer = shap.Explainer(est, x_ref, feature_names=list(x_ref.columns), algorithm="auto", model_output="probability")
            sv = explainer(x_ref)

        sv_single = _select_output_explanation(sv, X, out_idx)
        # Display original QFV values while using transformed model-space SHAP values.
        sv_display = shap.Explanation(
            values=sv_single[0].values,
            base_values=sv_single[0].base_values,
            data=X_disp.iloc[0].values,
            feature_names=list(X.columns),
        )

        shap.plots.force(
            sv_display,
            matplotlib=True,
            show=False,
            contribution_threshold=0.02,
            text_rotation=10,
            out_names=ROI,
        )
        return plt.gcf()
    except Exception:
        return None


# -------------------- #
# SHAP plotting kernel #
# -------------------- #

def _make_explainer(estimator: Any, X_ref: pd.DataFrame):
    """
    Construct a SHAP explainer that renders to matplotlib.
    Tree models -> TreeExplainer; linear -> LinearExplainer; else -> Explainer.
    """
    # Try tree
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        tree_like = (RandomForestClassifier, GradientBoostingClassifier)
        if isinstance(estimator, tree_like) or estimator.__class__.__name__ in {"XGBClassifier", "LGBMClassifier"}:
            return shap.TreeExplainer(estimator, feature_names=list(X_ref.columns))
    except Exception:
        pass

    # Try linear
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        if isinstance(estimator, (LogisticRegression, SGDClassifier)):
            return shap.LinearExplainer(estimator, X_ref, feature_names=list(X_ref.columns))
    except Exception:
        pass

    # Fallback (model-agnostic). Using model_output='probability' often looks best for classifiers.
    return shap.Explainer(estimator, X_ref, feature_names=list(X_ref.columns), algorithm="auto", model_output="probability")

def shap_plot_robust(
    QFV: Union[np.ndarray, Sequence[float], pd.Series, pd.DataFrame],
    ROI: str,
    models_root: Union[str, Path],
    AA_type: str,                # "vascular" / "lobe" / "aspect" / "hydro"
    Model_name: str,             # "RF"
    feature_names: Optional[Sequence[str]],
    filename_token: str | None = None,
    top_n: int = 12,
    figsize: Tuple[int, int] = (10, 6),
) -> Optional[matplotlib.figure.Figure]:
    """
    Compute SHAP for a single subject vector and return a Matplotlib Figure with force plot.
    Returns None if visualization fails.
    """

    # -----------------------------
    # Normalize AA_type (legacy aliases)
    # -----------------------------
    if feature_names is not None:
        feature_names = list(feature_names)
    aa_type_norm = _normalize_aa_type(AA_type)
    model_path = _resolve_model_path(models_root, aa_type_norm, Model_name, ROI, filename_token)
    clf, meta = _load_estimator(model_path)
    X = _conform_features(QFV, clf, feature_names, meta)

    X_ref = X.copy()
    prob_text = ""
    proba = None
    
    try:
        if hasattr(clf, "predict_proba"):
            proba = np.asarray(clf.predict_proba(X))
            if proba.ndim == 2 and proba.shape[0] == 1:
                pos_idx = _pick_output_index(clf, proba)
                pos_prob = float(proba[0, pos_idx])
                prob_text = f" | P(class={pos_idx})={pos_prob:.4f}"
            else:
                pos_idx = _pick_output_index(clf, None)
        else:
            pos_idx = _pick_output_index(clf, None)
    except Exception:
        pos_idx = _pick_output_index(clf, None)

    # Handle sklearn Pipeline(tree) explicitly (same strategy as ASPECTPC path).
    pre, est = _split_pipeline(clf)
    est_name = est.__class__.__name__
    tree_like_names = {"RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"}
    if est_name in tree_like_names:
        X_disp = X.round(4)
        X_model = pre.transform(X_disp) if pre is not None else X_disp.values
        explainer = shap.TreeExplainer(est, feature_names=list(X.columns))
        shap_values = explainer(X_model)
    else:
        # Generic fallback for non-tree models.
        X_disp = X.round(4)
        explainer = _make_explainer(clf, X_ref)
        shap_values = explainer(X_disp)

    sv_single = _select_output_explanation(shap_values, X, pos_idx)

    # --- Plotting Section ---
    title = f"{AA_type.upper()} · {ROI}{prob_text}"

    try:
        sv_display = shap.Explanation(
            values=sv_single[0].values,
            base_values=sv_single[0].base_values,
            data=X_disp.iloc[0].values,
            feature_names=list(X.columns),
        )
        shap.plots.force(
            sv_display,
            matplotlib=True,
            show=False,
            contribution_threshold=0.02,  # Text is less cluttered
            text_rotation=10,             # Text is rotated to avoid overlap
            out_names=ROI
        )
        fig = plt.gcf()
        return fig
    except Exception:
        # Just return None if visualization fails
        return None

# --------------------------- #
# PDF assembly: one AA_type   #
# --------------------------- #

def gen_report_interpretation(
    SubjDir: Union[str, Path],
    SubjID: str,
    QFV: Union[np.ndarray, Sequence[float], pd.Series, pd.DataFrame],
    AAModelsDir: Union[str, Path],
    AA_type: str,                          # "vascular" / "lobe" / "aspect" / "hydro"
    ROI_list: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    Model_name: str = "RF",
    top_n: int = 12,
) -> Optional[Path]:
    """
    Create a single multi-page PDF for one analysis type (AA_type).
    Returns the path to the created PDF. Skips failed visualizations.
    Example usage: gen_report_interpretation(subj_dir, subj_id, qfv_vascular, models_dir, "vascular")
    """
    aa_type_norm = _normalize_aa_type(AA_type)

    if aa_type_norm == "aspectpc":
        return gen_report_interpretation_aspectpc(
            SubjDir=SubjDir,
            SubjID=SubjID,
            QFV=QFV,
            AAModelsDir=AAModelsDir,
            ROI_list=ROI_list,
            feature_names=feature_names,
            Model_name=Model_name,
            top_n=top_n,
        )

    SubjDir = Path(SubjDir)
    SubjDir.mkdir(parents=True, exist_ok=True)

    # Auto-discover ROIs if not provided
    if ROI_list is None or len(ROI_list) == 0:
        raise RuntimeError("No ROI models discovered")

    # Filter ROI list to only include those with available models
    AAModelsDir = Path(AAModelsDir)
    available_rois = []
    for roi in ROI_list:
        if _check_model_exists(AAModelsDir, aa_type_norm, Model_name, roi):
            available_rois.append(roi)
        else:
            print(f"[{AA_type.upper()}][{roi}] Skipping - model not found")

    if not available_rois:
        print(f"[{AA_type.upper()}] No models found for any ROI - skipping PDF generation")
        return None

    print(f"[{AA_type.upper()}] Generating interpretation for {len(available_rois)}/{len(ROI_list)} ROIs")
    ROI_list = available_rois

    #out_pdf = SubjDir / f"{SubjID}_{AA_type}_report_interpretation.pdf"
    out_pdf = SubjDir / f"{SubjID}_{AA_type.upper()}_report_interpretation.pdf"
    # Create a PDF document
    # Track if any plots were successful
    any_plots_saved = False
    successful_rois = []
    failed_rois = []

    with PdfPages(out_pdf) as pdf:
        # Each ROI page
        for roi in ROI_list:
            try:
                fig = shap_plot_robust(
                    QFV=QFV,
                    ROI=roi,
                    models_root=AAModelsDir,
                    AA_type=AA_type,
                    Model_name=Model_name,
                    feature_names=feature_names,
                    top_n=top_n,
                )

                # Only save if figure was successfully created
                if fig is not None:
                    pdf.savefig(fig, facecolor="white", bbox_inches="tight")
                    plt.close(fig)
                    any_plots_saved = True
                    successful_rois.append(roi)
                else:
                    failed_rois.append(roi)
            except Exception as e:
                # Silently skip failures
                print(f"[{AA_type.upper()}][{roi}] SHAP failed:", e)
                failed_rois.append(roi)
                continue

    # Report results
    if successful_rois:
        print(f"[{AA_type.upper()}] Successfully generated {len(successful_rois)} SHAP plots: {', '.join(successful_rois[:3])}{'...' if len(successful_rois) > 3 else ''}")
    if failed_rois:
        print(f"[{AA_type.upper()}] Failed to generate {len(failed_rois)} SHAP plots: {', '.join(failed_rois)}")

    # If no plots were successful, remove the empty PDF
    if not any_plots_saved and out_pdf.exists():
        out_pdf.unlink()
        print(f"[{AA_type.upper()}] No plots generated - PDF deleted")
        return None

    if any_plots_saved:
        print(f"[{AA_type.upper()}] PDF saved: {out_pdf.name} ({len(successful_rois)} pages)")

    return out_pdf


def gen_report_interpretation_aspectpc(
    SubjDir: Union[str, Path],
    SubjID: str,
    QFV: Union[np.ndarray, Sequence[float], pd.Series, pd.DataFrame],
    AAModelsDir: Union[str, Path],
    ROI_list: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    Model_name: str = "RF",
    top_n: int = 12,
) -> Optional[Path]:
    """
    Create ASPECTPC interpretation PDF using the pipeline-safe SHAP path.
    """
    SubjDir = Path(SubjDir)
    SubjDir.mkdir(parents=True, exist_ok=True)

    if ROI_list is None or len(ROI_list) == 0:
        raise RuntimeError("No ROI models discovered")

    AAModelsDir = Path(AAModelsDir)
    available_rois = []
    for roi in ROI_list:
        if _check_model_exists(AAModelsDir, "aspectpc", Model_name, roi):
            available_rois.append(roi)
        else:
            print(f"[ASPECTPC][{roi}] Skipping - model not found")

    if not available_rois:
        print("[ASPECTPC] No models found for any ROI - skipping PDF generation")
        return None

    print(f"[ASPECTPC] Generating interpretation for {len(available_rois)}/{len(ROI_list)} ROIs")
    out_pdf = SubjDir / f"{SubjID}_ASPECTSPC_report_interpretation.pdf"

    any_plots_saved = False
    successful_rois = []
    failed_rois = []

    with PdfPages(out_pdf) as pdf:
        for roi in available_rois:
            try:
                fig = shap_plot_aspectpc(
                    QFV=QFV,
                    ROI=roi,
                    models_root=AAModelsDir,
                    Model_name=Model_name,
                    feature_names=feature_names,
                )

                if fig is not None:
                    pdf.savefig(fig, facecolor="white", bbox_inches="tight")
                    plt.close(fig)
                    any_plots_saved = True
                    successful_rois.append(roi)
                else:
                    failed_rois.append(roi)
            except Exception as e:
                print(f"[ASPECTPC][{roi}] SHAP failed:", e)
                failed_rois.append(roi)
                continue

    if successful_rois:
        print(
            "[ASPECTPC] Successfully generated "
            f"{len(successful_rois)} SHAP plots: {', '.join(successful_rois[:3])}"
            f"{'...' if len(successful_rois) > 3 else ''}"
        )
    if failed_rois:
        print(f"[ASPECTPC] Failed to generate {len(failed_rois)} SHAP plots: {', '.join(failed_rois)}")

    if not any_plots_saved and out_pdf.exists():
        out_pdf.unlink()
        print("[ASPECTPC] No plots generated - PDF deleted")
        return None

    if any_plots_saved:
        print(f"[ASPECTPC] PDF saved: {out_pdf.name} ({len(successful_rois)} pages)")

    return out_pdf
