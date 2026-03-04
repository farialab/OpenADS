"""CSV writing adapter for QFV results.

Handles file I/O for writing QFV CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ads.domain.reporting import QFVResult


class QFVCSVWriter:
    """Write QFV results to CSV files.

    Pure I/O adapter - writes pandas DataFrames to CSV.
    """

    # Standard QFV category names and their CSV suffixes
    CATEGORY_SUFFIXES = {
        "vascular": "VASCULAR_QFV",
        "lobe": "LOBE_QFV",
        "aspects": "ASPECTS_QFV",
        "aspectpc": "ASPECTPC_QFV",
        "ventricles": "VENTRICLES_QFV",
        "bpm": "BPM_TYPE1_QFV",
    }

    def write_qfv_csv(
        self,
        qfv_array: pd.Series,
        output_path: Path,
    ) -> Path:
        """Write single QFV array to CSV.

        Args:
            qfv_array: Pandas Series with QFV values (index = ROI names)
            output_path: Path for output CSV

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame if Series
        if isinstance(qfv_array, pd.Series):
            df = qfv_array.to_frame().T
        else:
            df = qfv_array

        # Write CSV
        df.to_csv(output_path, index=False)

        return output_path

    def write_qfv_csvs(
        self,
        qfv_result: QFVResult,
        output_dir: Path,
        roi_labels: Dict[str, List[str]],
    ) -> Dict[str, Path]:
        """Write all QFV CSVs for one subject.

        Args:
            qfv_result: QFVResult domain object
            output_dir: Directory for output files
            roi_labels: Dict mapping category -> list of ROI names

        Returns:
            Dict mapping category -> written CSV path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written_paths = {}

        # Map QFV arrays to categories
        qfv_arrays = {
            "vascular": qfv_result.vascular_qfv,
            "lobe": qfv_result.lobe_qfv,
            "aspects": qfv_result.aspects_qfv,
            "ventricles": qfv_result.ventricles_qfv,
        }

        # Add optional arrays
        if qfv_result.aspectpc_qfv is not None:
            qfv_arrays["aspectpc"] = qfv_result.aspectpc_qfv
        if qfv_result.bpm_qfv is not None:
            qfv_arrays["bpm"] = qfv_result.bpm_qfv

        # Write each category
        for category, qfv_array in qfv_arrays.items():
            if qfv_array is None:
                continue

            # Get ROI labels
            labels = roi_labels.get(category, [])

            # Create Series with labels
            if len(labels) == len(qfv_array):
                series = pd.Series(qfv_array, index=labels, name=qfv_result.subject_id)
            else:
                # Fallback: use numeric indices
                series = pd.Series(qfv_array, name=qfv_result.subject_id)

            # Construct filename
            suffix = self.CATEGORY_SUFFIXES.get(category, f"{category.upper()}_QFV")
            filename = f"{qfv_result.subject_id}_{suffix}.csv"
            output_path = output_dir / filename

            # Write CSV
            self.write_qfv_csv(series, output_path)
            written_paths[category] = output_path

        return written_paths

    def write_summary_csv(
        self,
        qfv_result: QFVResult,
        output_path: Path,
    ) -> Path:
        """Write summary CSV with lesion volume and ICV.

        Args:
            qfv_result: QFVResult domain object
            output_path: Path for output CSV

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create summary DataFrame
        summary = pd.DataFrame({
            "subject_id": [qfv_result.subject_id],
            "lesion_volume_ml": [qfv_result.lesion_volume_ml],
            "icv_volume_ml": [qfv_result.icv_volume_ml],
            "lesion_to_icv_ratio": [
                (qfv_result.lesion_volume_ml / qfv_result.icv_volume_ml * 100)
                if qfv_result.icv_volume_ml > 0 else 0.0
            ],
        })

        # Write CSV
        summary.to_csv(output_path, index=False)

        return output_path
