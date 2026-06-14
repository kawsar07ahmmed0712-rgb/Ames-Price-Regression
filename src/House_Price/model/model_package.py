from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class AmesEnsembleModel:
    """
    Deployable ensemble model wrapper for Ames House Price Regression.

    This class does not train models.
    It stores already-trained base models and applies the final notebook blend logic.

    Final notebook logic:
    - inverse weighted blend
    - public-safe manual blend
    - final conservative blend = 0.45 * inverse + 0.55 * public-safe
    - optional high-price tail lift in log space
    - optional clipping after converting back to original price scale
    """

    def __init__(
        self,
        models: Dict[str, Any],
        inverse_weights: Dict[str, float],
        public_safe_weights: Dict[str, float],
        final_blend_weights: Optional[Dict[str, float]] = None,
        tail_lift_params: Optional[Dict[str, Any]] = None,
        price_clip_params: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.models = models
        self.inverse_weights = self._normalize_weights(inverse_weights)
        self.public_safe_weights = self._normalize_weights(public_safe_weights)

        self.final_blend_weights = final_blend_weights or {
            "inverse": 0.45,
            "public_safe": 0.55,
        }

        self.tail_lift_params = tail_lift_params or {
            "enabled": True,
            "strength": 0.010,
            "start_quantile": 0.90,
            "q_start": None,
            "q_max": None,
        }

        self.price_clip_params = price_clip_params or {
            "enabled": False,
            "lower": None,
            "upper": None,
        }

        self.metadata = metadata or {}

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights so their total becomes 1.0.
        """
        if not weights:
            return {}

        total_weight = sum(weights.values())

        if total_weight <= 0:
            raise ValueError("Weight sum must be greater than zero.")

        return {name: weight / total_weight for name, weight in weights.items()}

    def _predict_base_models(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate log-space predictions from each stored base model.
        """
        try:
            predictions = {}

            for model_name, model in self.models.items():
                predictions[model_name] = model.predict(X)

            prediction_df = pd.DataFrame(predictions, index=X.index)

            logger.info(
                f"Generated base model predictions with shape: {prediction_df.shape}"
            )

            return prediction_df

        except Exception as e:
            raise CustomException(e, sys)

    def _weighted_blend(
        self,
        prediction_df: pd.DataFrame,
        weights: Dict[str, float],
        blend_name: str,
    ) -> np.ndarray:
        """
        Apply weighted average over selected base model predictions.
        """
        if not weights:
            raise ValueError(f"{blend_name} weights are empty.")

        missing_models = [
            model_name for model_name in weights
            if model_name not in prediction_df.columns
        ]

        if missing_models:
            raise ValueError(
                f"{blend_name} uses missing models: {missing_models}"
            )

        blended = np.zeros(prediction_df.shape[0])

        for model_name, weight in weights.items():
            blended += weight * prediction_df[model_name].values

        return blended

    def _apply_high_price_lift(self, log_predictions: np.ndarray) -> np.ndarray:
        """
        Apply notebook-style small high-price calibration in log space.

        Notebook logic:
        - start from top prediction range
        - gradually increase prediction by a tiny strength
        - default strength = 0.010
        - default start_quantile = 0.90
        """
        params = self.tail_lift_params

        if not params.get("enabled", True):
            return log_predictions

        strength = float(params.get("strength", 0.010))
        start_quantile = float(params.get("start_quantile", 0.90))

        adjusted = np.array(log_predictions, dtype=float).copy()

        q_start = params.get("q_start")
        q_max = params.get("q_max")

        if q_start is None:
            q_start = np.quantile(adjusted, start_quantile)

        if q_max is None:
            q_max = np.max(adjusted)

        if q_max <= q_start:
            return adjusted

        lift_ratio = (adjusted - q_start) / (q_max - q_start)
        lift_ratio = np.clip(lift_ratio, 0, 1)

        adjusted = adjusted + strength * lift_ratio

        return adjusted

    def _apply_price_clipping(self, prices: np.ndarray) -> np.ndarray:
        """
        Clip final price prediction if clipping parameters are enabled.
        """
        params = self.price_clip_params

        if not params.get("enabled", False):
            return prices

        lower = params.get("lower")
        upper = params.get("upper")

        if lower is None or upper is None:
            return prices

        return np.clip(prices, lower, upper)

    def predict_log(
        self,
        X: pd.DataFrame,
        apply_tail_lift: bool = True,
    ) -> np.ndarray:
        """
        Predict log1p(SalePrice) using the final ensemble logic.
        """
        try:
            prediction_df = self._predict_base_models(X)

            inverse_blend = self._weighted_blend(
                prediction_df=prediction_df,
                weights=self.inverse_weights,
                blend_name="inverse_blend",
            )

            public_safe_blend = self._weighted_blend(
                prediction_df=prediction_df,
                weights=self.public_safe_weights,
                blend_name="public_safe_blend",
            )

            inverse_weight = float(self.final_blend_weights.get("inverse", 0.45))
            public_safe_weight = float(
                self.final_blend_weights.get("public_safe", 0.55)
            )

            final_blend = (
                inverse_weight * inverse_blend
                + public_safe_weight * public_safe_blend
            )

            if apply_tail_lift:
                final_blend = self._apply_high_price_lift(final_blend)

            return final_blend

        except Exception as e:
            logger.error("Log prediction failed.")
            raise CustomException(e, sys)

    def predict_price(
        self,
        X: pd.DataFrame,
        apply_tail_lift: bool = True,
        apply_clipping: bool = True,
    ) -> np.ndarray:
        """
        Predict final SalePrice in original price scale.
        """
        try:
            log_predictions = self.predict_log(
                X=X,
                apply_tail_lift=apply_tail_lift,
            )

            prices = np.expm1(log_predictions)

            if apply_clipping:
                prices = self._apply_price_clipping(prices)

            return prices

        except Exception as e:
            logger.error("Price prediction failed.")
            raise CustomException(e, sys)

    def get_manifest(self) -> Dict[str, Any]:
        """
        Return serializable ensemble manifest.
        """
        return {
            "model_type": "AmesEnsembleModel",
            "base_models": list(self.models.keys()),
            "inverse_weights": self.inverse_weights,
            "public_safe_weights": self.public_safe_weights,
            "final_blend_weights": self.final_blend_weights,
            "tail_lift_params": self.tail_lift_params,
            "price_clip_params": self.price_clip_params,
            "metadata": self.metadata,
        }