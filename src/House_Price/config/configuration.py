import sys

from House_Price.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from House_Price.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    PredictionConfig,
)
from House_Price.utils.common import create_directories, read_yaml
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class ConfigurationManager:
    """
    Reads YAML files and returns clean config objects for each project component.
    """

    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH,
        schema_file_path=SCHEMA_FILE_PATH,
    ):
        try:
            self.config = read_yaml(config_file_path)
            self.params = read_yaml(params_file_path)
            self.schema = read_yaml(schema_file_path)

            artifacts_root = self.config["artifacts"]["root_dir"]
            create_directories([artifacts_root])

            logger.info("Configuration files loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config["data_ingestion"]
            data = self.config["data"]

            create_directories([config["root_dir"]])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config["root_dir"],
                raw_train_path=data["raw_train_path"],
                raw_test_path=data["raw_test_path"],
                raw_sample_submission_path=data["raw_sample_submission_path"],
                data_description_path=data["data_description_path"],
                train_file_path=config["train_file_path"],
                test_file_path=config["test_file_path"],
            )

            logger.info("DataIngestionConfig created successfully.")
            return data_ingestion_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config["data_validation"]
            ingestion_config = self.config["data_ingestion"]

            create_directories([config["root_dir"]])

            data_validation_config = DataValidationConfig(
                root_dir=config["root_dir"],
                validation_status_file=config["validation_status_file"],
                schema_report_file=config["schema_report_file"],
                train_file_path=ingestion_config["train_file_path"],
                test_file_path=ingestion_config["test_file_path"],
                schema=self.schema,
            )

            logger.info("DataValidationConfig created successfully.")
            return data_validation_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config["data_transformation"]
            ingestion_config = self.config["data_ingestion"]

            create_directories([config["root_dir"]])

            data_transformation_config = DataTransformationConfig(
                root_dir=config["root_dir"],
                train_file_path=ingestion_config["train_file_path"],
                test_file_path=ingestion_config["test_file_path"],
                preprocessor_path=config["preprocessor_path"],
                feature_names_path=config["feature_names_path"],
                transformation_metadata_path=config["transformation_metadata_path"],
                transformed_train_path=config["transformed_train_path"],
                transformed_test_path=config["transformed_test_path"],
                schema=self.schema,
            )

            logger.info("DataTransformationConfig created successfully.")
            return data_transformation_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config["model_trainer"]
            transformation_config = self.config["data_transformation"]

            create_directories([config["root_dir"]])

            model_trainer_config = ModelTrainerConfig(
                root_dir=config["root_dir"],
                transformed_train_path=transformation_config["transformed_train_path"],
                final_model_path=config["final_model_path"],
                blend_manifest_path=config["blend_manifest_path"],
                model_metadata_path=config["model_metadata_path"],
                params=self.params,
            )

            logger.info("ModelTrainerConfig created successfully.")
            return model_trainer_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config["model_evaluation"]
            trainer_config = self.config["model_trainer"]
            transformation_config = self.config["data_transformation"]

            create_directories([config["root_dir"]])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=config["root_dir"],
                transformed_train_path=transformation_config["transformed_train_path"],
                final_model_path=trainer_config["final_model_path"],
                metrics_file_path=config["metrics_file_path"],
                model_report_path=config["model_report_path"],
                params=self.params,
            )

            logger.info("ModelEvaluationConfig created successfully.")
            return model_evaluation_config

        except Exception as e:
            raise CustomException(e, sys)

    def get_prediction_config(self) -> PredictionConfig:
        try:
            config = self.config["prediction"]
            data = self.config["data"]
            transformation_config = self.config["data_transformation"]
            trainer_config = self.config["model_trainer"]

            create_directories([config["root_dir"]])

            prediction_config = PredictionConfig(
                root_dir=config["root_dir"],
                prediction_output_path=config["prediction_output_path"],
                batch_prediction_output_path=config["batch_prediction_output_path"],
                submission_output_path=config["submission_output_path"],
                raw_test_path=data["raw_test_path"],
                raw_sample_submission_path=data["raw_sample_submission_path"],
                transformed_test_path=transformation_config["transformed_test_path"],
                preprocessor_path=transformation_config["preprocessor_path"],
                feature_names_path=transformation_config["feature_names_path"],
                final_model_path=trainer_config["final_model_path"],
                blend_manifest_path=trainer_config["blend_manifest_path"],
                schema=self.schema,
                params=self.params,
            )

            logger.info("PredictionConfig created successfully.")
            return prediction_config

        except Exception as e:
            raise CustomException(e, sys)