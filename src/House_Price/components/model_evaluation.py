import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from House_Price.entity.config_entity import ModelEvaluationConfig
from House_Price.utils.common import load_object, save_json
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class ModelEvaluation:
    """
    Model evaluation component.

    Responsibility:
    - Load final ensemble model
    - Load transformed training data
    - Collect OOF metrics from saved model metadata
    - Compute training-fit diagnostic metrics
    - Save reports/metrics.json
    - Save reports/model_report.md

    Important:
    - This component does NOT train models.
    - This component does NOT create Kaggle submission files.
    - The primary reported score is OOF RMSE from ModelTrainer metadata.
    """

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.target_column = "SalePrice_log"

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        self._check_file_exists(
            self.config.transformed_train_path,
            "Transformed train data",
        )

        train_df = pd.read_csv(self.config.transformed_train_path)

        if self.target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in transformed train data."
            )

        X = train_df.drop(columns=[self.target_column])
        y_log = train_df[self.target_column]
        y_price = np.expm1(y_log)

        logger.info(f"Evaluation data loaded with shape: {train_df.shape}")

        return X, y_log, y_price

    def _calculate_diagnostic_metrics(
        self,
        model: Any,
        X: pd.DataFrame,
        y_log: pd.Series,
        y_price: pd.Series,
    ) -> Dict[str, float]:
        """
        Calculate full-training-fit diagnostic metrics.

        These metrics are not the main validation score because the final model
        was already fitted on the full training data.
        """
        pred_log = model.predict_log(X)
        pred_price = model.predict_price(X)

        rmse_log = float(np.sqrt(mean_squared_error(y_log, pred_log)))
        rmse_price = float(np.sqrt(mean_squared_error(y_price, pred_price)))
        mae_price = float(mean_absolute_error(y_price, pred_price))

        return {
            "train_fit_rmse_log1p_diagnostic": rmse_log,
            "train_fit_rmse_price_diagnostic": rmse_price,
            "train_fit_mae_price_diagnostic": mae_price,
        }

    def _build_metrics_report(
        self,
        model: Any,
        diagnostic_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        metadata = getattr(model, "metadata", {})

        metrics = {
            "selected_strategy": metadata.get("selected_strategy"),
            "primary_metric": "OOF_RMSE_log1p",
            "feature_count": metadata.get("feature_count"),
            "training_rows": metadata.get("training_rows"),
            "best_single_score": metadata.get("best_single_score"),
            "inverse_blend_rmse": metadata.get("inverse_blend_rmse"),
            "public_safe_rmse": metadata.get("public_safe_rmse"),
            "final_conservative_blend_rmse": metadata.get(
                "final_conservative_blend_rmse"
            ),
            "tail_lift_rmse": metadata.get("tail_lift_rmse"),
            "selected_inverse_models": metadata.get("selected_inverse_models"),
            "all_base_models": metadata.get("all_base_models"),
            "diagnostic_metrics": diagnostic_metrics,
            "note": (
                "Primary score is out-of-fold RMSE from the model trainer. "
                "Training-fit metrics are diagnostic only and should not be "
                "reported as validation performance."
            ),
        }

        return metrics

    def _write_markdown_report(self, metrics: Dict[str, Any]) -> None:
        report_path = Path(self.config.model_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        selected_models = metrics.get("selected_inverse_models") or []
        all_models = metrics.get("all_base_models") or []
        diagnostics = metrics.get("diagnostic_metrics") or {}

        content = f"""# Ames House Price Regression — Model Evaluation Report

## Final Selected Strategy

`{metrics.get("selected_strategy")}`

## Primary Validation Metric

The primary score is based on out-of-fold validation from the model trainer.

| Metric | Value |
|---|---:|
| Best Single Model OOF RMSE log1p | {metrics.get("best_single_score"):.6f} |
| Inverse Weighted Blend OOF RMSE log1p | {metrics.get("inverse_blend_rmse"):.6f} |
| Public-Safe Blend OOF RMSE log1p | {metrics.get("public_safe_rmse"):.6f} |
| Final Conservative Blend OOF RMSE log1p | {metrics.get("final_conservative_blend_rmse"):.6f} |
| Tail-Lift Final Blend OOF RMSE log1p | {metrics.get("tail_lift_rmse"):.6f} |

## Final Model Summary

| Item | Value |
|---|---:|
| Training rows | {metrics.get("training_rows")} |
| Feature count | {metrics.get("feature_count")} |
| Primary metric | {metrics.get("primary_metric")} |

## Selected Inverse-Blend Models

{chr(10).join([f"- {model}" for model in selected_models])}

## All Base Models Used

{chr(10).join([f"- {model}" for model in all_models])}

## Diagnostic Training-Fit Metrics

These are diagnostic only because the final model is fitted on the full training data.

| Metric | Value |
|---|---:|
| Train-fit RMSE log1p | {diagnostics.get("train_fit_rmse_log1p_diagnostic"):.6f} |
| Train-fit RMSE price | {diagnostics.get("train_fit_rmse_price_diagnostic"):.2f} |
| Train-fit MAE price | {diagnostics.get("train_fit_mae_price_diagnostic"):.2f} |

## Notes

- No hyperparameter tuning was performed in the production trainer.
- Kaggle submission generation is intentionally kept outside model evaluation.
- The final production strategy is the notebook Step 10 tail-lift blend.
"""

        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(content)

        logger.info(f"Model markdown report saved at: {report_path}")

    def initiate_model_evaluation(self) -> Dict[str, Any]:
        """
        Run model evaluation process.
        """
        try:
            logger.info("Model evaluation started.")

            self._check_file_exists(self.config.final_model_path, "Final model")

            model = load_object(self.config.final_model_path)
            X, y_log, y_price = self._load_data()

            diagnostic_metrics = self._calculate_diagnostic_metrics(
                model=model,
                X=X,
                y_log=y_log,
                y_price=y_price,
            )

            metrics = self._build_metrics_report(
                model=model,
                diagnostic_metrics=diagnostic_metrics,
            )

            save_json(self.config.metrics_file_path, metrics)
            self._write_markdown_report(metrics)

            output = {
                "model_evaluation_completed": True,
                "metrics_file_path": self.config.metrics_file_path,
                "model_report_path": self.config.model_report_path,
                "selected_strategy": metrics.get("selected_strategy"),
                "tail_lift_rmse": metrics.get("tail_lift_rmse"),
                "feature_count": metrics.get("feature_count"),
            }

            logger.info("Model evaluation completed successfully.")
            return output

        except Exception as e:
            logger.error("Model evaluation failed.")
            raise CustomException(e, sys)