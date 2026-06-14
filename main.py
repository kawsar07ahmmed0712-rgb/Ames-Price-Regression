import json
import sys

from House_Price.pipeline.training_pipeline import TrainingPipeline
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


def main():
    try:
        logger.info("Main execution started.")

        pipeline = TrainingPipeline()
        output = pipeline.run()

        print("\nTraining pipeline completed successfully.\n")
        print(json.dumps(output, indent=4))

        logger.info("Main execution completed successfully.")

    except Exception as e:
        logger.error("Main execution failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()