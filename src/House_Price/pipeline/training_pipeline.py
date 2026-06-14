import json
import sys
from typing import Any, Dict

from House_Price.components.data_ingestion import DataIngestion
from House_Price.components.data_transformation import DataTransformation
from House_Price.components.data_validation import DataValidation
from House_Price.components.model_evaluation import ModelEvaluation
from House_Price.components.model_trainer import ModelTrainer
from House_Price.config.configuration import ConfigurationManager
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class TrainingPipeline:
    """
    Full training pipeline.

    Pipeline flow:
    1. Data ingestion
    2. Data validation
    3. Data transformation
    4. Model training
    5. Model evaluation
    """

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_data_ingestion(self) -> Dict[str, Any]:
        logger.info("Pipeline stage started: Data Ingestion")

        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        output = data_ingestion.initiate_data_ingestion()

        logger.info("Pipeline stage completed: Data Ingestion")
        return output

    def run_data_validation(self) -> Dict[str, Any]:
        logger.info("Pipeline stage started: Data Validation")

        data_validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        output = data_validation.initiate_data_validation()

        if not output.get("validation_passed", False):
            raise ValueError(
                f"Data validation failed with errors: {output.get('errors')}"
            )

        logger.info("Pipeline stage completed: Data Validation")
        return output

    def run_data_transformation(self) -> Dict[str, Any]:
        logger.info("Pipeline stage started: Data Transformation")

        data_transformation_config = self.config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        output = data_transformation.initiate_data_transformation()

        logger.info("Pipeline stage completed: Data Transformation")
        return output

    def run_model_training(self) -> Dict[str, Any]:
        logger.info("Pipeline stage started: Model Training")

        model_trainer_config = self.config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        output = model_trainer.initiate_model_training()

        logger.info("Pipeline stage completed: Model Training")
        return output

    def run_model_evaluation(self) -> Dict[str, Any]:
        logger.info("Pipeline stage started: Model Evaluation")

        model_evaluation_config = self.config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        output = model_evaluation.initiate_model_evaluation()

        logger.info("Pipeline stage completed: Model Evaluation")
        return output

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        """
        try:
            logger.info("Full training pipeline started.")

            pipeline_output = {
                "data_ingestion": self.run_data_ingestion(),
                "data_validation": self.run_data_validation(),
                "data_transformation": self.run_data_transformation(),
                "model_training": self.run_model_training(),
                "model_evaluation": self.run_model_evaluation(),
            }

            logger.info("Full training pipeline completed successfully.")
            return pipeline_output

        except Exception as e:
            logger.error("Full training pipeline failed.")
            raise CustomException(e, sys)


def run_training_pipeline() -> Dict[str, Any]:
    """
    Helper function for running the full training pipeline.
    """
    pipeline = TrainingPipeline()
    return pipeline.run()


if __name__ == "__main__":
    output = run_training_pipeline()
    print(json.dumps(output, indent=4))