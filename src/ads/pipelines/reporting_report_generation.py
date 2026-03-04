"""High level reporting pipeline that orchestrates lesion, radiology, and interpretation outputs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from ads.reporting.radiology.dwi_radiology_report import (
    compute_volumes_ml,
    estimate_icv_ml,
    gen_radiological_report,
    load_nifti,
)
from ads.reporting.interpretation.shap_interpretation import gen_report_interpretation
from ads.reporting.radiology.lesion_volume_report import gen_lesion_report

@dataclass
class ReportArtifacts:
    lesion_report: Path
    radiology_report: Path
    interpretation_reports: Dict[str, Optional[Path]]
    lesion_volumes_ml: Tuple[float, float, float]
    icv_volume_ml: float


class ReportPipeline:
    """Generate lesion, radiology, and SHAP interpretation reports for a subject."""

    # Define columns for CSV loading
    COLUMN_DEFINITIONS: Dict[str, Sequence[str]] = {
        "Vascular": [
            "subject_id", "logvol", "ACA", "MCA", "PCA", "cerebellar", "basilar",
            "Lenticulostriate", "Choroidal & Thalamoperfurating", "watershed"
        ],
        "Lobe": [
            "subject_id", "logvol", "basal ganglia", "deep white matter", "cerebellum",
            "frontal", "insula", "internal capsule", "brainstem", "occipital",
            "parietal", "temporal", "thalamus"
        ],
        "Aspects": [
            "subject_id", "volml", "Caudate", "lentiform", "IC", "insula",
            "M1", "M2", "M3", "M6", "M5", "M4"
        ],
        "AspectsPC": [
            'subject_id', 'volml', 'PCA', 'Thalamus', 'cerebellum', 'pons', 'midbrain'
            ],
        "Ventricles": [
            "subject_id", "volml", "LVOR1", "LVOR2", "LVOR3", "LVOR4", "LVOR5",
            "LVOR6", "LVOR7", "LVOR8", "LVOR9", "LVOR10", "LVIR1", "LVIR2",
            "LVIR3", "LVIR4", "LVIR5", "LVIR6", "LVIR7", "LVIR8", "LVIR9", "LVIR10"
        ],
    }

    INTERPRETATION_ROIS: Dict[str, Sequence[str]] = {
        "vascular": [
            "ACA", "MCA", "PCA", "cerebellar", "basilar",
            "Lenticulostriate", "Choroidal & Thalamoperfurating", "watershed"
        ],
        "lobe": [
            "basal ganglia", "deep white matter", "cerebellum", "frontal", "insula",
            "internal capsule", "brainstem", "occipital", "parietal", "temporal", "thalamus"
        ],
        "aspects": [
            "Caudate", "lentiform", "IC", "insula", 
            "M1", "M2", "M3", "M6", "M5", "M4"
        ],
        "aspectpc": ["PCA", "Thalamus", "cerebellum", "pons", "midbrain"],
    }

    INTERPRETATION_FEATURES: Dict[str, Sequence[str]] = {
        "vascular": ["logvol", *INTERPRETATION_ROIS["vascular"]],
        "lobe": ["logvol", *INTERPRETATION_ROIS["lobe"]],
        "aspects": ["volml", *INTERPRETATION_ROIS["aspects"]],
        "aspectpc": ["volml", *INTERPRETATION_ROIS["aspectpc"]],
    }

    def __init__(
        self,
        subject_dir: Path | str,
        project_root: Path | str | None = None,
        atlas_dir: Path | str | None = None,
        aa_models_dir: Path | str | None = None,
        model_name: str = "RF",
        subject_id: str | None = None, # Added explicit subject_id
    ) -> None:
        self.subject_dir = Path(subject_dir).resolve()
        if not self.subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")

        # FIX: Use explicit ID if provided, otherwise fallback to folder name
        self.subject_id = subject_id if subject_id else self.subject_dir.name
        
        self.reporting_dir = self.subject_dir / "reporting"
        self.registration_dir = self.subject_dir / "registration"

        if not self.reporting_dir.exists():
            raise FileNotFoundError(f"Reporting directory missing: {self.reporting_dir}")
        if not self.registration_dir.exists():
            raise FileNotFoundError(f"Registration directory missing: {self.registration_dir}")

        self.project_root = Path(project_root) if project_root else None
        self.atlas_dir = Path(atlas_dir)
        self.aa_models_dir = Path(aa_models_dir)
        self.model_name = model_name

    def run(self, *, hydro_flag: bool = False) -> ReportArtifacts:
        lesion_ctx = self._load_lesion_context()
        qfv_tables = self._load_qfv_tables()
        qfv_list, qfv_series_map = self._prepare_qfv_structures(qfv_tables)

        lesion_report_path = gen_lesion_report(
            self.reporting_dir,
            self.subject_id,
            lesion_ctx["lesion_mask"],
            lesion_ctx["icv_ml"],
            lesion_ctx["lesion_total_ml"],
            self.atlas_dir,
        )

        pca_qfv = None
        if "AspectsPC" in qfv_tables:
            pca_qfv = self._row_vector(qfv_tables["AspectsPC"], numeric=True).to_numpy(dtype=float)

        gen_radiological_report(
            str(self.reporting_dir),
            self.subject_id,
            lesion_ctx["side_volumes_ml"],
            lesion_ctx["lesion_total_ml"],
            str(self.aa_models_dir),
            qfv_list,
            hydro_flag=hydro_flag,
            pca_qfv=pca_qfv,
        )
        radiology_report_path = self.reporting_dir / f"{self.subject_id}_automatic_radiological_report.txt"

        interpretation_reports = self._generate_interpretation_reports(qfv_series_map)

        return ReportArtifacts(
            lesion_report=lesion_report_path,
            radiology_report=radiology_report_path,
            interpretation_reports=interpretation_reports,
            lesion_volumes_ml=(
                lesion_ctx["lesion_total_ml"],
                lesion_ctx["side_volumes_ml"][0],
                lesion_ctx["side_volumes_ml"][1],
            ),
            icv_volume_ml=lesion_ctx["icv_ml"],
        )

    def _load_lesion_context(self) -> Dict[str, object]:
        lesion_path = self.subject_dir / "segment" / f"{self.subject_id}_stroke-mask_space-MNI152.nii.gz"
        mask_path = self.registration_dir / f"{self.subject_id}_DWIbrain-mask_space-MNI152_aff.nii.gz"

        if not lesion_path.exists():
            raise FileNotFoundError(
                f"Affine-space lesion mask not found (required for DWI volume calculation): {lesion_path}"
            )
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Affine-space brain mask not found (required for DWI volume calculation): {mask_path}"
            )
        
        lesion_img = load_nifti(lesion_path)
        lesion_arr = np.asanyarray(lesion_img.dataobj)
        lesion_total, left_ml, right_ml, lesion_mask = compute_volumes_ml(lesion_arr, lesion_img)
        icv_ml = estimate_icv_ml(mask_path, lesion_img)

        return {
            "lesion_mask": lesion_mask,
            "lesion_total_ml": float(lesion_total),
            "side_volumes_ml": (float(left_ml), float(right_ml)),
            "icv_ml": float(icv_ml),
        }
    

    def _load_qfv_tables(self) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        for name, columns in self.COLUMN_DEFINITIONS.items():
            csv_path = self.reporting_dir / f"{self.subject_id}_{name.upper()}_QFV.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing QFV CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError(f"QFV CSV has no rows: {csv_path}")
            df.columns = list(columns)
            tables[name] = df
        return tables

    def _prepare_qfv_structures(self, tables: Dict[str, pd.DataFrame]) -> Tuple[list[np.ndarray], Dict[str, pd.Series]]:
        qfv_list: list[np.ndarray] = []
        qfv_series_map: Dict[str, pd.Series] = {}

        vascular_row = self._row_vector(tables["Vascular"], numeric=True)
        qfv_list.append(vascular_row.to_numpy(dtype=float))
        qfv_series_map["vascular"] = vascular_row

        lobe_row = self._row_vector(tables["Lobe"], numeric=True)
        qfv_list.append(lobe_row.to_numpy(dtype=float))
        qfv_series_map["lobe"] = lobe_row

        aspect_row = self._row_vector(tables["Aspects"], numeric=True)
        qfv_list.append(aspect_row.to_numpy(dtype=float))
        qfv_series_map["aspects"] = aspect_row

        aspects_pca_row = self._row_vector(tables["AspectsPC"], numeric=True)
        qfv_series_map["aspectpc"] = aspects_pca_row

        ventricles_row = self._row_vector(tables["Ventricles"], numeric=True)
        qfv_list.append(ventricles_row.to_numpy(dtype=float))
        qfv_series_map["ventricles"] = ventricles_row

        return qfv_list, qfv_series_map

    @staticmethod
    def _row_vector(df: pd.DataFrame, *, numeric: bool = False) -> pd.Series:
        row = df.iloc[0, 1:].copy()
        if numeric:
            row = pd.to_numeric(row, errors="coerce").fillna(0.0)
            return row.astype(float)
        return row.fillna(0.0)

    def _generate_interpretation_reports(self, qfv_series_map: Dict[str, pd.Series]) -> Dict[str, Optional[Path]]:
        outputs: Dict[str, Optional[Path]] = {}

        for aa_type, roi_list in self.INTERPRETATION_ROIS.items():
            if aa_type not in qfv_series_map:
                continue

            print(f"[REPORT] Generating {aa_type} interpretation report...")

            feat_names = list(self.INTERPRETATION_FEATURES[aa_type])
            series = (
                qfv_series_map[aa_type]
                .reindex(feat_names)
                .fillna(0.0)
                .astype(float)
                .round(4)
            )

            csv_path = self.reporting_dir / f"{self.subject_id}_{aa_type.upper()}_QFV.csv"
            pdf_path = gen_report_interpretation(
                SubjDir=self.reporting_dir,
                SubjID=self.subject_id,
                QFV=series,
                AAModelsDir=self.aa_models_dir,
                AA_type=aa_type,
                ROI_list=list(roi_list),
                feature_names=list(feat_names),
                Model_name=self.model_name,
            )
            outputs[aa_type] = pdf_path
            print(f"[REPORT] {aa_type} PDF path = {pdf_path}")

        return outputs



__all__ = ["ReportPipeline", "ReportArtifacts"]
