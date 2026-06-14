import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from House_Price.entity.config_entity import DataTransformationConfig
from House_Price.model.feature_pipeline import AmesFeaturePipeline
from House_Price.utils.common import save_json, save_object
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class DataTransformation:
    """
    Data transformation component.

    Responsibility:
    - Load ingested train/test data
    - Apply Ames feature engineering pipeline
    - Save transformed train/test data
    - Save fitted preprocessor object
    - Save final feature names
    - Save transformation metadata
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._check_file_exists(self.config.train_file_path, "Ingested train")
        self._check_file_exists(self.config.test_file_path, "Ingested test")

        train_df = pd.read_csv(self.config.train_file_path)
        test_df = pd.read_csv(self.config.test_file_path)

        logger.info(f"Ingested train data loaded with shape: {train_df.shape}")
        logger.info(f"Ingested test data loaded with shape: {test_df.shape}")

        return train_df, test_df

    def _save_transformed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_log: pd.Series,
    ) -> None:
        train_output = X_train.copy()
        train_output["SalePrice_log"] = y_log.values

        Path(self.config.transformed_train_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_output.to_csv(self.config.transformed_train_path, index=False)
        X_test.to_csv(self.config.transformed_test_path, index=False)

        logger.info(
            f"Transformed train data saved at: {self.config.transformed_train_path}"
        )
        logger.info(
            f"Transformed test data saved at: {self.config.transformed_test_path}"
        )

    def _save_feature_names(self, feature_names: list[str]) -> None:
        feature_names_data = {
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        }

        save_json(self.config.feature_names_path, feature_names_data)

        logger.info(f"Feature names saved at: {self.config.feature_names_path}")

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        save_json(self.config.transformation_metadata_path, metadata)

        logger.info(
            f"Transformation metadata saved at: {self.config.transformation_metadata_path}"
        )

    def initiate_data_transformation(self) -> Dict[str, Any]:
        """
        Run complete data transformation process.

        Returns:
            Dictionary containing transformation artifact paths and summary.
        """
        try:
            logger.info("Data transformation started.")

            Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)

            train_df, test_df = self._load_data()

            feature_pipeline = AmesFeaturePipeline()

            X_train, X_test, y_log, metadata = feature_pipeline.fit_transform(
                train_df=train_df,
                test_df=test_df,
            )

            self._save_transformed_data(
                X_train=X_train,
                X_test=X_test,
                y_log=y_log,
            )

            save_object(
                file_path=self.config.preprocessor_path,
                obj=feature_pipeline,
            )

            logger.info(f"Preprocessor saved at: {self.config.preprocessor_path}")

            self._save_feature_names(feature_pipeline.feature_names_)
            self._save_metadata(metadata)

            transformation_output = {
                "transformation_completed": True,
                "preprocessor_path": self.config.preprocessor_path,
                "feature_names_path": self.config.feature_names_path,
                "transformation_metadata_path": self.config.transformation_metadata_path,
                "transformed_train_path": self.config.transformed_train_path,
                "transformed_test_path": self.config.transformed_test_path,
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
                "feature_count": len(feature_pipeline.feature_names_),
                "target_column": "SalePrice_log",
                "missing_values_train": int(X_train.isna().sum().sum()),
                "missing_values_test": int(X_test.isna().sum().sum()),
            }

            logger.info("Data transformation completed successfully.")
            return transformation_output

        except Exception as e:
            logger.error("Data transformation failed.")
            raise CustomException(e, sys)