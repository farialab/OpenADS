"""Model loading adapter.

Handles file I/O for loading ML models (Random Forest, etc.).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib


class AAModelLoader:
    """Load AA (Anterior/Posterior) models from disk.

    Pure I/O adapter - loads pickle files and returns sklearn models.

    """

    def __init__(self, models_dir: Path):
        """Initialize model loader.

        Args:
            models_dir: Directory containing model subdirectories
        """
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")

    def load_model_with_preprocessing(
        self,
        model_path: Path,
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load model with optional scaler and feature selector.

        Args:
            model_path: Path to model pickle file

        Returns:
            Tuple of (model, scaler, selector)
            - model: sklearn estimator
            - scaler: sklearn scaler (or None)
            - selector: sklearn feature selector (or None)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Try joblib first (recommended for sklearn)
            data = joblib.load(model_path)
        except Exception:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

        # Handle different formats
        if isinstance(data, dict):
            # Dict format: {"model": ..., "scaler": ..., "selector": ...}
            model = data.get("model")
            scaler = data.get("scaler", None)
            selector = data.get("selector", None)
        elif isinstance(data, (list, tuple)) and len(data) == 3:
            # Tuple format: (model, scaler, selector)
            model, scaler, selector = data
        else:
            # Just the model
            model = data
            scaler = None
            selector = None

        return model, scaler, selector

    def load_roi_model(
        self,
        aa_type: str,
        roi: str,
        model_name: str = "RF",
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load model for one ROI.

        Searches for model file matching the ROI name pattern.

        Args:
            aa_type: Model category ("vascular", "lobe", "aspects", "aspectpc", "hydro")
            roi: ROI name (e.g., "ACA", "M1", "frontal")
            model_name: Model type prefix (default "RF" for Random Forest)

        Returns:
            Tuple of (model, scaler, selector)

        Raises:
            FileNotFoundError: If model not found
        """
        model_dir = self.models_dir / aa_type

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        # Search for model file
        # Try different naming patterns
        roi_token = roi.replace(" ", "_").lower()
        patterns = [
            f"{model_name}_{roi_token}.pkl",
            f"*_{roi_token}.pkl",
            f"*{roi_token}*.pkl",
        ]

        model_path = None
        for pattern in patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                model_path = matches[0]
                break

        if model_path is None:
            raise FileNotFoundError(f"No model found for {aa_type}/{roi} in {model_dir}")

        return self.load_model_with_preprocessing(model_path)

    def load_all_models(
        self,
        aa_type: str,
        roi_list: List[str],
        model_name: str = "RF",
    ) -> Dict[str, Tuple[Any, Optional[Any], Optional[Any]]]:
        """Load all models for one AA type.

        Args:
            aa_type: Model category
            roi_list: List of ROI names
            model_name: Model type prefix

        Returns:
            Dict mapping ROI name -> (model, scaler, selector)
            ROIs without models are omitted from dict
        """
        models = {}

        for roi in roi_list:
            try:
                model, scaler, selector = self.load_roi_model(aa_type, roi, model_name)
                models[roi] = (model, scaler, selector)
            except FileNotFoundError:
                print(f"Warning: Model not found for {aa_type}/{roi}")
                continue

        return models

    def load_multi_category_models(
        self,
        category_roi_lists: Dict[str, List[str]],
        model_name: str = "RF",
    ) -> Dict[str, Dict[str, Tuple[Any, Optional[Any], Optional[Any]]]]:
        """Load models for multiple categories.

        Args:
            category_roi_lists: Dict mapping category -> ROI list
            model_name: Model type prefix

        Returns:
            Dict mapping category -> (Dict mapping ROI -> (model, scaler, selector))
        """
        all_models = {}

        for category, roi_list in category_roi_lists.items():
            all_models[category] = self.load_all_models(category, roi_list, model_name)

        return all_models

    def get_available_categories(self) -> List[str]:
        """Get list of available model categories.

        Returns:
            List of category names that have directories in models_dir
        """
        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]
