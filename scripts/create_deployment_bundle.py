import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]

RAW_TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
RAW_TEST_PATH = ROOT / "data" / "raw" / "test.csv"

PREPROCESSOR_PATH = ROOT / "artifacts" / "data_transformation" / "preprocessor.pkl"
FEATURE_NAMES_PATH = ROOT / "artifacts" / "data_transformation" / "feature_names.json"
TRANSFORMED_TRAIN_PATH = ROOT / "artifacts" / "data_transformation" / "train_transformed.csv"

DEPLOY_DIR = ROOT / "artifacts" / "deployment"
DEPLOY_PREPROCESSOR_PATH = DEPLOY_DIR / "preprocessor.pkl"
DEPLOY_MODEL_PATH = DEPLOY_DIR / "deploy_model.pkl"
DEPLOY_METADATA_PATH = DEPLOY_DIR / "deploy_metadata.json"


def to_json_safe(value: Any):
    if pd.isna(value):
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    return value


def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj: Any, path: Path):
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_feature_names(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["feature_names", "features", "columns"]:
            if key in data:
                return data[key]

    raise ValueError("Could not read feature names from feature_names.json")


def create_target(train_raw: pd.DataFrame, expected_rows: int):
    clean_train = train_raw.copy()

    if len(clean_train) != expected_rows:
        if "GrLivArea" in clean_train.columns and "SalePrice" in clean_train.columns:
            outlier_mask = (
                (clean_train["GrLivArea"] > 4000)
                & (clean_train["SalePrice"] < 300000)
            )
            clean_train = clean_train.loc[~outlier_mask].copy()

    if len(clean_train) != expected_rows:
        raise ValueError(
            f"Target row count mismatch. X has {expected_rows} rows, "
            f"but target has {len(clean_train)} rows."
        )

    return np.log1p(clean_train["SalePrice"].values)


def main():
    required_paths = [
        RAW_TRAIN_PATH,
        RAW_TEST_PATH,
        PREPROCESSOR_PATH,
        FEATURE_NAMES_PATH,
        TRANSFORMED_TRAIN_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(RAW_TRAIN_PATH)
    test_raw = pd.read_csv(RAW_TEST_PATH)

    preprocessor = load_pickle(PREPROCESSOR_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)

    transformed_train = pd.read_csv(TRANSFORMED_TRAIN_PATH)

    if all(col in transformed_train.columns for col in feature_names):
        X_train = transformed_train[feature_names].copy()
    elif transformed_train.shape[1] == len(feature_names):
        X_train = transformed_train.copy()
        X_train.columns = feature_names
    else:
        raise ValueError(
            "Could not align transformed_train.csv with feature_names.json. "
            f"transformed_train shape: {transformed_train.shape}, "
            f"feature count: {len(feature_names)}"
        )

    y_train = create_target(train_raw=train_raw, expected_rows=len(X_train))

    model = Ridge(alpha=13.0)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))

    default_row = test_raw.iloc[0].copy()
    default_row["Id"] = 999999

    default_row_dict = {
        str(key): to_json_safe(value)
        for key, value in default_row.to_dict().items()
    }

    neighborhood_values = sorted(
        pd.concat(
            [
                train_raw["Neighborhood"].dropna(),
                test_raw["Neighborhood"].dropna(),
            ]
        )
        .astype(str)
        .unique()
        .tolist()
    )

    metadata = {
        "model_type": "Ridge deployment model",
        "target_transform": "log1p",
        "inverse_target_transform": "expm1",
        "feature_count": len(feature_names),
        "train_rmse_log": train_rmse,
        "feature_names": feature_names,
        "default_row": default_row_dict,
        "form_options": {
            "Neighborhood": neighborhood_values,
            "OverallQual": list(range(1, 11)),
            "GarageCars": [0, 1, 2, 3, 4],
            "FullBath": [0, 1, 2, 3, 4],
            "TotRmsAbvGrd": list(range(2, 15)),
            "quality_values": ["Po", "Fa", "TA", "Gd", "Ex"],
            "bsmt_quality_values": ["NoBasement", "Po", "Fa", "TA", "Gd", "Ex"],
            "garage_finish_values": ["NoGarage", "Unf", "RFn", "Fin"],
        },
        "default_form_values": {
            "Neighborhood": "NridgHt",
            "OverallQual": 7,
            "GrLivArea": 1650,
            "GarageCars": 2,
            "GarageArea": 500,
            "YearBuilt": 2005,
            "YearRemodAdd": 2005,
            "TotalBsmtSF": 950,
            "FirstFlrSF": 950,
            "FullBath": 2,
            "TotRmsAbvGrd": 6,
            "ExterQual": "Gd",
            "KitchenQual": "Gd",
            "BsmtQual": "Gd",
            "GarageFinish": "RFn",
        },
    }

    save_pickle(preprocessor, DEPLOY_PREPROCESSOR_PATH)
    save_pickle(model, DEPLOY_MODEL_PATH)

    with open(DEPLOY_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("Deployment bundle created successfully.")
    print(f"Preprocessor: {DEPLOY_PREPROCESSOR_PATH}")
    print(f"Model: {DEPLOY_MODEL_PATH}")
    print(f"Metadata: {DEPLOY_METADATA_PATH}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Train RMSE log: {train_rmse:.6f}")


if __name__ == "__main__":
    main()
