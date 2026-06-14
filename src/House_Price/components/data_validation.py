import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from House_Price.entity.config_entity import DataValidationConfig
from House_Price.utils.common import save_json
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class DataValidation:
    """
    Data validation component.

    Responsibility:
    - Check ingested train/test files exist
    - Check train/test data are not empty
    - Check target column exists in train data
    - Check ID column exists in train/test data
    - Check train/test feature columns are aligned
    - Save validation status and schema report
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._check_file_exists(self.config.train_file_path, "Train")
        self._check_file_exists(self.config.test_file_path, "Test")

        train_df = pd.read_csv(self.config.train_file_path)
        test_df = pd.read_csv(self.config.test_file_path)

        return train_df, test_df

    def _get_required_column_names(self) -> tuple[str, str]:
        columns_config = self.config.schema.get("columns", {})

        id_column = columns_config.get("id_column")
        target_column = columns_config.get("target_column")

        if not id_column:
            raise ValueError("id_column is missing in config/schema.yaml")

        if not target_column:
            raise ValueError("target_column is missing in config/schema.yaml")

        return id_column, target_column

    def _validate_not_empty(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
        errors = []

        if train_df.empty:
            errors.append("Train data is empty.")

        if test_df.empty:
            errors.append("Test data is empty.")

        return errors

    def _validate_required_columns(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        id_column: str,
        target_column: str,
    ) -> List[str]:
        errors = []

        if id_column not in train_df.columns:
            errors.append(f"ID column '{id_column}' is missing from train data.")

        if id_column not in test_df.columns:
            errors.append(f"ID column '{id_column}' is missing from test data.")

        require_train_target = self.config.schema.get("data_validation", {}).get(
            "require_train_target", True
        )

        require_test_target = self.config.schema.get("data_validation", {}).get(
            "require_test_target", False
        )

        if require_train_target and target_column not in train_df.columns:
            errors.append(f"Target column '{target_column}' is missing from train data.")

        if require_test_target and target_column not in test_df.columns:
            errors.append(f"Target column '{target_column}' is missing from test data.")

        return errors

    def _validate_train_test_feature_alignment(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
    ) -> Dict[str, Any]:
        train_features = set(train_df.columns) - {target_column}
        test_features = set(test_df.columns) - {target_column}

        missing_in_test = sorted(list(train_features - test_features))
        extra_in_test = sorted(list(test_features - train_features))

        return {
            "train_feature_count_without_target": len(train_features),
            "test_feature_count_without_target": len(test_features),
            "missing_features_in_test": missing_in_test,
            "extra_features_in_test": extra_in_test,
            "is_aligned": len(missing_in_test) == 0 and len(extra_in_test) == 0,
        }

    def initiate_data_validation(self) -> Dict[str, Any]:
        try:
            logger.info("Data validation started.")

            train_df, test_df = self._load_data()
            id_column, target_column = self._get_required_column_names()

            errors = []

            errors.extend(self._validate_not_empty(train_df, test_df))

            errors.extend(
                self._validate_required_columns(
                    train_df=train_df,
                    test_df=test_df,
                    id_column=id_column,
                    target_column=target_column,
                )
            )

            feature_alignment_report = self._validate_train_test_feature_alignment(
                train_df=train_df,
                test_df=test_df,
                target_column=target_column,
            )

            if not feature_alignment_report["is_aligned"]:
                errors.append("Train/test feature columns are not aligned.")

            validation_passed = len(errors) == 0

            validation_status = {
                "validation_passed": validation_passed,
                "errors": errors,
            }

            schema_report = {
                "train_shape": list(train_df.shape),
                "test_shape": list(test_df.shape),
                "id_column": id_column,
                "target_column": target_column,
                "train_has_target": target_column in train_df.columns,
                "test_has_target": target_column in test_df.columns,
                "feature_alignment": feature_alignment_report,
                "train_missing_values_total": int(train_df.isna().sum().sum()),
                "test_missing_values_total": int(test_df.isna().sum().sum()),
            }

            save_json(self.config.validation_status_file, validation_status)
            save_json(self.config.schema_report_file, schema_report)

            if validation_passed:
                logger.info("Data validation completed successfully.")
            else:
                logger.error(f"Data validation failed with errors: {errors}")

            return {
                "validation_passed": validation_passed,
                "validation_status_file": self.config.validation_status_file,
                "schema_report_file": self.config.schema_report_file,
                "errors": errors,
            }

        except Exception as e:
            logger.error("Data validation failed due to unexpected error.")
            raise CustomException(e, sys)