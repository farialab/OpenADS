"""AA Model prediction service.

Pure business logic for AA (Anterior/Posterior) model predictions.
No model loading - assumes models are already loaded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ads.domain.reporting import AAModelPrediction


class AAModelPredictor:
    """
    AA model prediction logic (no model loading).
    Takes loaded models as input, returns predictions.
    """

    def predict_roi(
        self,
        qfv_vector: np.ndarray,
        model: Any,
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
        roi_name: str = "unknown",
    ) -> AAModelPrediction:
        """
        Predict one ROI (no model loading).

        Args:
            qfv_vector: QFV feature vector for this ROI
            model: Loaded sklearn model (RF, etc.)
            scaler: Optional scaler for feature normalization
            selector: Optional feature selector
            roi_name: ROI name for the prediction

        Returns:
            AAModelPrediction domain object
        """
        # Reshape to 2D if needed (sklearn expects 2D)
        if qfv_vector.ndim == 1:
            X = qfv_vector.reshape(1, -1)
        else:
            X = qfv_vector

        # Apply scaler if provided
        if scaler is not None:
            X = scaler.transform(X)

        # Apply feature selector if provided
        if selector is not None:
            X = selector.transform(X)

        # Make prediction
        prediction = int(model.predict(X)[0])

        # Get probability if model supports it
        probability = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)[0]
                # Probability of positive class (index 1)
                if len(proba) > 1:
                    probability = float(proba[1])
                else:
                    probability = float(proba[0])
            except Exception:
                # Some models may not support predict_proba
                pass

        # Get QFV value (original, not scaled)
        qfv_value = float(qfv_vector[0]) if qfv_vector.size > 0 else 0.0

        return AAModelPrediction(
            roi=roi_name,
            prediction=prediction,
            qfv=qfv_value,
            probability=probability,
        )

    def predict_all_rois(
        self,
        qfv_vector: np.ndarray,
        roi_models: Dict[str, Tuple[Any, Optional[Any], Optional[Any]]],
        roi_list: List[str],
    ) -> List[AAModelPrediction]:
        """Predict all ROIs for one AA type.

        Args:
            qfv_vector: Full QFV vector (all ROIs)
            roi_models: Dict mapping ROI name -> (model, scaler, selector)
            roi_list: List of ROI names (in order matching qfv_vector)

        Returns:
            List of AAModelPrediction objects, one per ROI
        """
        predictions = []

        for i, roi_name in enumerate(roi_list):
            # Get QFV value for this ROI
            if i < len(qfv_vector):
                roi_qfv = np.array([qfv_vector[i]])
            else:
                # ROI not in vector, skip
                continue

            # Get model for this ROI
            if roi_name not in roi_models:
                # No model for this ROI, skip or use default
                continue

            model, scaler, selector = roi_models[roi_name]

            # Predict
            pred = self.predict_roi(
                qfv_vector=roi_qfv,
                model=model,
                scaler=scaler,
                selector=selector,
                roi_name=roi_name,
            )

            predictions.append(pred)

        return predictions

    def predict_multi_category(
        self,
        qfv_vectors: Dict[str, np.ndarray],
        category_models: Dict[str, Dict[str, Tuple[Any, Optional[Any], Optional[Any]]]],
        category_roi_lists: Dict[str, List[str]],
    ) -> Dict[str, List[AAModelPrediction]]:
        """Predict multiple categories (vascular, lobe, aspects, etc.).

        Args:
            qfv_vectors: Dict mapping category -> QFV vector
            category_models: Dict mapping category -> roi_models dict
            category_roi_lists: Dict mapping category -> ROI list

        Returns:
            Dict mapping category -> list of predictions
        """
        all_predictions = {}

        for category in qfv_vectors.keys():
            if category not in category_models or category not in category_roi_lists:
                continue

            predictions = self.predict_all_rois(
                qfv_vector=qfv_vectors[category],
                roi_models=category_models[category],
                roi_list=category_roi_lists[category],
            )

            all_predictions[category] = predictions

        return all_predictions

    def threshold_prediction(
        self,
        prediction: AAModelPrediction,
        probability_threshold: float = 0.5,
    ) -> AAModelPrediction:
        """Apply probability threshold to override prediction.

        If probability is below threshold, override prediction to 0.

        Args:
            prediction: Original prediction
            probability_threshold: Minimum probability for positive prediction

        Returns:
            Potentially modified prediction
        """
        if prediction.probability is not None:
            if prediction.probability < probability_threshold:
                # Override to negative
                return AAModelPrediction(
                    roi=prediction.roi,
                    prediction=0,
                    qfv=prediction.qfv,
                    probability=prediction.probability,
                )

        return prediction
