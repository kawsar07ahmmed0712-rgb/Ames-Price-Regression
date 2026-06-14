import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from House_Price.entity.config_entity import PredictionConfig
from House_Price.utils.common import load_object, save_json
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class BatchPredictionPipeline:
    """
    Batch prediction pipeline.

    Responsibility:
    - Load transformed test data
    - Load saved final ensemble model
    - Align features using feature_names.json
    - Predict SalePrice
    - Save batch prediction file
    - Save Kaggle-style submission file

    Important:
    - This pipeline does NOT train models.
    - This pipeline uses already transformed test features.
    """

    def __init__(self, config: PredictionConfig):
        self.config = config

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _load_feature_names(self) -> list[str]:
        self._check_file_exists(self.config.feature_names_path, "Feature names JSON")

        with open(self.config.feature_names_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        feature_names = data.get("feature_names")

        if not feature_names:
            raise ValueError("feature_names not found inside feature_names.json")

        return feature_names

    def _load_test_ids(self) -> pd.Series:
        """
        Load test IDs for submission.

        Priority:
        1. sample_submission.csv
        2. raw test.csv
        """
        id_column = self.config.schema["columns"]["id_column"]

        if Path(self.config.raw_sample_submission_path).exists():
            sample_submission = pd.read_csv(self.config.raw_sample_submission_path)

            if id_column not in sample_submission.columns:
                raise ValueError(
                    f"ID column '{id_column}' not found in sample submission."
                )

            return sample_submission[id_column]

        self._check_file_exists(self.config.raw_test_path, "Raw test data")

        raw_test = pd.read_csv(self.config.raw_test_path)

        if id_column not in raw_test.columns:
            raise ValueError(f"ID column '{id_column}' not found in raw test data.")

        return raw_test[id_column]

    def _load_transformed_test(self, feature_names: list[str]) -> pd.DataFrame:
        self._check_file_exists(
            self.config.transformed_test_path,
            "Transformed test data",
        )

        X_test = pd.read_csv(self.config.transformed_test_path)

        missing_features = [
            col for col in feature_names
            if col not in X_test.columns
        ]

        extra_features = [
            col for col in X_test.columns
            if col not in feature_names
        ]

        if missing_features:
            raise ValueError(
                f"Transformed test data is missing required features: {missing_features[:20]}"
            )

        if extra_features:
            logger.warning(
                f"Extra features found in transformed test data and will be dropped: {extra_features[:20]}"
            )

        X_test = X_test.reindex(columns=feature_names)

        return X_test

    def initiate_batch_prediction(self) -> Dict[str, Any]:
        try:
            logger.info("Batch prediction started.")

            Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.submission_output_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._check_file_exists(self.config.final_model_path, "Final model")

            feature_names = self._load_feature_names()
            X_test = self._load_transformed_test(feature_names)
            test_ids = self._load_test_ids()

            if len(test_ids) != len(X_test):
                raise ValueError(
                    f"ID count and prediction row count mismatch. "
                    f"IDs: {len(test_ids)}, X_test: {len(X_test)}"
                )

            model = load_object(self.config.final_model_path)

            predictions = model.predict_price(
                X=X_test,
                apply_tail_lift=True,
                apply_clipping=True,
            )

            predictions = np.asarray(predictions, dtype=float)

            id_column = self.config.schema["columns"]["id_column"]

            batch_prediction_df = pd.DataFrame({
                id_column: test_ids.values,
                "PredictedSalePrice": predictions,
            })

            submission_df = pd.DataFrame({
                id_column: test_ids.values,
                "SalePrice": predictions,
            })

            batch_prediction_df.to_csv(
                self.config.batch_prediction_output_path,
                index=False,
            )

            submission_df.to_csv(
                self.config.submission_output_path,
                index=False,
            )

            prediction_summary = {
                "batch_prediction_completed": True,
                "rows_predicted": int(len(predictions)),
                "feature_count": int(X_test.shape[1]),
                "batch_prediction_output_path": self.config.batch_prediction_output_path,
                "submission_output_path": self.config.submission_output_path,
                "prediction_min": float(np.min(predictions)),
                "prediction_mean": float(np.mean(predictions)),
                "prediction_median": float(np.median(predictions)),
                "prediction_max": float(np.max(predictions)),
                "note": (
                    "Batch prediction uses transformed test features and saved final ensemble model. "
                    "No model training is performed here."
                ),
            }

            save_json(
                file_path=self.config.prediction_output_path,
                data=prediction_summary,
            )

            logger.info("Batch prediction completed successfully.")
            return prediction_summary

        except Exception as e:
            logger.error("Batch prediction failed.")
            raise CustomException(e, sys)
        
class SinglePredictionPipeline:
    """
    Single raw-input prediction pipeline.

    Responsibility:
    - Load saved preprocessor.pkl
    - Load saved final_model.pkl
    - Convert raw Ames row into 199 engineered features
    - Predict SalePrice
    """

    def __init__(self, config: PredictionConfig):
        self.config = config

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def predict_from_dataframe(
        self,
        raw_df: pd.DataFrame,
        strict_unknown_categories: bool = True,
        save_latest: bool = True,
    ) -> Dict[str, Any]:
        try:
            logger.info("Single prediction from dataframe started.")

            self._check_file_exists(self.config.preprocessor_path, "Preprocessor")
            self._check_file_exists(self.config.final_model_path, "Final model")

            preprocessor = load_object(self.config.preprocessor_path)
            model = load_object(self.config.final_model_path)

            X = preprocessor.transform(
                raw_df,
                strict_unknown_categories=strict_unknown_categories,
            )

            predictions = model.predict_price(
                X=X,
                apply_tail_lift=True,
                apply_clipping=True,
            )

            predictions = np.asarray(predictions, dtype=float)

            id_column = self.config.schema["columns"]["id_column"]

            if id_column in raw_df.columns:
                ids = raw_df[id_column].tolist()
            else:
                ids = list(range(1, len(raw_df) + 1))

            prediction_records = []

            for row_id, price in zip(ids, predictions):
                prediction_records.append({
                    id_column: int(row_id),
                    "PredictedSalePrice": float(price),
                })

            result = {
                "single_prediction_completed": True,
                "rows_predicted": int(len(predictions)),
                "feature_count": int(X.shape[1]),
                "predictions": prediction_records,
                "prediction_min": float(np.min(predictions)),
                "prediction_mean": float(np.mean(predictions)),
                "prediction_max": float(np.max(predictions)),
            }

            if save_latest:
                save_json(
                    file_path=self.config.prediction_output_path,
                    data=result,
                )

            logger.info("Single prediction from dataframe completed successfully.")
            return result

        except Exception as e:
            logger.error("Single prediction from dataframe failed.")
            raise CustomException(e, sys)

    def predict_single_dict(
        self,
        raw_input: Dict[str, Any],
        strict_unknown_categories: bool = True,
    ) -> Dict[str, Any]:
        raw_df = pd.DataFrame([raw_input])

        return self.predict_from_dataframe(
            raw_df=raw_df,
            strict_unknown_categories=strict_unknown_categories,
            save_latest=True,
        )