import shutil
import sys
from pathlib import Path
from typing import Dict

from House_Price.entity.config_entity import DataIngestionConfig
from House_Price.utils.exception import CustomException
from House_Price.utils.logger import logger


class DataIngestion:
    """
    Data ingestion component.

    Responsibility:
    - Check raw train/test files exist
    - Copy raw train/test files into artifacts/data_ingestion/
    - Keep raw source data unchanged
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _check_file_exists(self, file_path: str, file_label: str) -> None:
        """
        Check if a required file exists.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_label} file not found at path: {path}")

        if not path.is_file():
            raise ValueError(f"{file_label} path exists but is not a file: {path}")

    def _copy_file(self, source_path: str, destination_path: str) -> None:
        """
        Copy file from source path to destination path.
        Existing artifact file can be overwritten because artifacts are generated outputs.
        """
        source = Path(source_path)
        destination = Path(destination_path)

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        logger.info(f"Copied file from {source} to {destination}")

    def initiate_data_ingestion(self) -> Dict[str, str]:
        """
        Run data ingestion process.

        Returns:
            Dictionary containing artifact train/test file paths.
        """
        try:
            logger.info("Data ingestion started.")

            root_dir = Path(self.config.root_dir)
            root_dir.mkdir(parents=True, exist_ok=True)

            # Required raw files
            self._check_file_exists(self.config.raw_train_path, "Raw train")
            self._check_file_exists(self.config.raw_test_path, "Raw test")

            # Copy required files to artifact location
            self._copy_file(
                source_path=self.config.raw_train_path,
                destination_path=self.config.train_file_path,
            )

            self._copy_file(
                source_path=self.config.raw_test_path,
                destination_path=self.config.test_file_path,
            )

            # Optional supporting files
            optional_files = [
                self.config.raw_sample_submission_path,
                self.config.data_description_path,
            ]

            for optional_file in optional_files:
                optional_path = Path(optional_file)

                if optional_path.exists() and optional_path.is_file():
                    destination_path = root_dir / optional_path.name
                    self._copy_file(
                        source_path=str(optional_path),
                        destination_path=str(destination_path),
                    )
                else:
                    logger.warning(f"Optional file not found, skipped: {optional_path}")

            ingestion_output = {
                "train_file_path": self.config.train_file_path,
                "test_file_path": self.config.test_file_path,
            }

            logger.info("Data ingestion completed successfully.")
            return ingestion_output

        except Exception as e:
            logger.error("Data ingestion failed.")
            raise CustomException(e, sys)