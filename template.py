from pathlib import Path

PROJECT_NAME = "House_Price"

# Folder list
directories = [
    "config",

    "data/raw",
    "data/processed",
    "data/sample",

    "artifacts/data_ingestion",
    "artifacts/data_validation",
    "artifacts/data_transformation",
    "artifacts/model_trainer",
    "artifacts/model_evaluation",
    "artifacts/prediction",

    "notebooks",

    "research",

    f"src/{PROJECT_NAME}",
    f"src/{PROJECT_NAME}/constants",
    f"src/{PROJECT_NAME}/entity",
    f"src/{PROJECT_NAME}/config",
    f"src/{PROJECT_NAME}/components",
    f"src/{PROJECT_NAME}/pipeline",
    f"src/{PROJECT_NAME}/model",
    f"src/{PROJECT_NAME}/utils",

    "app/templates",
    "app/static/css",
    "app/static/js",
    "app/static/images",

    "reports/figures",

    "tests",

    "logs",
]

# File list
files = [
    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/sample/.gitkeep",

    "artifacts/data_ingestion/.gitkeep",
    "artifacts/data_validation/.gitkeep",
    "artifacts/data_transformation/.gitkeep",
    "artifacts/model_trainer/.gitkeep",
    "artifacts/model_evaluation/.gitkeep",
    "artifacts/prediction/.gitkeep",

    "notebooks/.gitkeep",

    "research/01_eda_summary.md",
    "research/02_feature_engineering.md",
    "research/03_model_selection.md",

    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/constants/__init__.py",
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/entity/config_entity.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/components/data_ingestion.py",
    f"src/{PROJECT_NAME}/components/data_validation.py",
    f"src/{PROJECT_NAME}/components/data_transformation.py",
    f"src/{PROJECT_NAME}/components/model_trainer.py",
    f"src/{PROJECT_NAME}/components/model_evaluation.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/prediction_pipeline.py",
    f"src/{PROJECT_NAME}/model/__init__.py",
    f"src/{PROJECT_NAME}/model/feature_pipeline.py",
    f"src/{PROJECT_NAME}/model/model_package.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",
    f"src/{PROJECT_NAME}/utils/logger.py",
    f"src/{PROJECT_NAME}/utils/exception.py",

    "app/__init__.py",
    "app/main.py",
    "app/templates/index.html",
    "app/templates/predict.html",
    "app/templates/result.html",
    "app/static/css/style.css",
    "app/static/js/main.js",
    "app/static/images/.gitkeep",

    "reports/metrics.json",
    "reports/model_report.md",
    "reports/figures/.gitkeep",

    "tests/__init__.py",
    "tests/test_data_validation.py",
    "tests/test_prediction_pipeline.py",

    "main.py",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    "README.md",
]


def create_directories() -> None:
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created/Exists folder: {path}")


def create_files() -> None:
    for file_path in files:
        path = Path(file_path)

        # Make parent folder if it does not exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Do not overwrite existing files
        if path.exists():
            print(f"Skipped existing file: {path}")
            continue

        # Create new empty file
        path.touch()
        print(f"Created file: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Creating Ames House Price Regression project structure...")
    print("Safe mode: existing files will NOT be overwritten.")
    print("=" * 70)

    create_directories()
    create_files()

    print("=" * 70)
    print("Project structure creation completed successfully.")
    print("=" * 70)