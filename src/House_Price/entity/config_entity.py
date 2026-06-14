from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: str
    raw_train_path: str
    raw_test_path: str
    raw_sample_submission_path: str
    data_description_path: str
    train_file_path: str
    test_file_path: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: str
    validation_status_file: str
    schema_report_file: str
    train_file_path: str
    test_file_path: str
    schema: Dict[str, Any]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: str
    train_file_path: str
    test_file_path: str
    preprocessor_path: str
    feature_names_path: str
    transformation_metadata_path: str
    transformed_train_path: str
    transformed_test_path: str
    schema: Dict[str, Any]


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: str
    transformed_train_path: str
    final_model_path: str
    blend_manifest_path: str
    model_metadata_path: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: str
    transformed_train_path: str
    final_model_path: str
    metrics_file_path: str
    model_report_path: str
    params: Dict[str, Any]


@dataclass(frozen=True)
class PredictionConfig:
    root_dir: str
    prediction_output_path: str
    batch_prediction_output_path: str
    preprocessor_path: str
    feature_names_path: str
    final_model_path: str
    blend_manifest_path: str
    schema: Dict[str, Any]
    params: Dict[str, Any]